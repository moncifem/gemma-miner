"""The main agent loop.

A single ReAct-style loop. Each turn:
  1. Compute the current PHASE from observable state.
  2. Render the state brief.
  3. Ask the LLM for ONE tool call.
  4. Parse it. On parse failure, push an error observation and retry.
  5. Dispatch the tool. Append a TurnRecord.
  6. If `finish` succeeded, run a self-verification pass; if it surfaces fixable
     issues, the agent gets another chance (`finished` is reset to False).
  7. Loop until max_turns.

The system is intentionally generic: any user query, any source, any schema.
The agent infers the plan from the goal + observable state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gemma_miner.contracts import Contract, ContractBook, MinRowsContract
from gemma_miner.dataset import Dataset
from gemma_miner.llm import LLMClient
from gemma_miner.memory import Memory
from gemma_miner.parsing import ParseError, parse_tool_call
from gemma_miner.phases import current_phase
from gemma_miner.prompts import SYSTEM_PROMPT, render_state_brief
from gemma_miner.state import AgentState
from gemma_miner.tools.registry import ToolRegistry, default_registry
from gemma_miner.tracing import Tracer

log = logging.getLogger("gemma_miner")


@dataclass
class AgentConfig:
    max_turns: int = 80
    max_consecutive_parse_failures: int = 3
    max_consecutive_tool_errors: int = 6
    # Number of times the agent is allowed to fail self-verification and be
    # re-launched into the loop with corrective feedback. 0 disables it.
    max_verify_retries: int = 1
    verbose: bool = True
    # Live event subscribers — every Tracer event also flows to these
    # callables. The CLI plugs its ActivityFeed in here.
    event_subscribers: list = field(default_factory=list)


@dataclass
class RunResult:
    finished: bool
    finish_reason: str
    turns: int
    dataset_path: str
    n_rows: int
    contracts: list[dict] = field(default_factory=list)
    trace_log: str = ""
    trace_jsonl: str = ""
    verification: dict = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        extraction_llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ):
        self.llm = llm
        self.extraction_llm = extraction_llm or llm
        self.registry = registry or default_registry(
            llm=llm,
            extraction_llm=self.extraction_llm,
        )
        self.config = config or AgentConfig()

    # ──────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────

    def run(self, state: AgentState) -> RunResult:
        tracer = Tracer(state.workdir)
        for sub in self.config.event_subscribers:
            tracer.subscribe(sub)

        def _progress(fields: dict) -> None:
            tracer.event(event=fields.pop("event", "progress"), **fields)
        state.progress_hook = _progress

        tracer.note(
            "run_start", state.goal,
            contracts=[c.description for c in state.contracts.list()],
        )

        verification: dict = {}
        verify_attempts = 0
        turn = 0
        try:
            while True:
                turn = self._loop(state, tracer, start_turn=turn)
                if not state.finished:
                    break
                # Self-verification pass before we accept `finish`.
                verification = self._self_verify(state, tracer)
                if verification.get("ok") or verify_attempts >= self.config.max_verify_retries:
                    break
                # Re-open the loop with corrective feedback in memory.
                verify_attempts += 1
                state.finished = False
                state.finish_reason = ""
                fix_hint = (
                    "Self-verification FAILED. Fix the issues below before "
                    "calling finish again:\n"
                    + "\n".join(f"  - {p}" for p in verification.get("problems", []))
                )
                state.memory.set("_verify_hint", fix_hint)
                tracer.note("verify_retry", fix_hint)
                if turn >= self.config.max_turns:
                    break

            tracer.note(
                "run_end",
                f"finished={state.finished} reason={state.finish_reason or 'stopped'}",
                turns=turn,
                rows=len(state.dataset),
                verification=verification,
            )
        finally:
            tracer.close()

        return RunResult(
            finished=state.finished,
            finish_reason=state.finish_reason
                          or ("max turns reached" if turn >= self.config.max_turns else "stopped"),
            turns=turn,
            dataset_path=str(state.dataset.path),
            n_rows=len(state.dataset),
            contracts=state.contracts_snapshot(),
            trace_log=str(Path(state.workdir) / "trace.log"),
            trace_jsonl=str(Path(state.workdir) / "trace.jsonl"),
            verification=verification,
        )

    def _loop(self, state: AgentState, tracer: Tracer, *, start_turn: int) -> int:
        parse_failures = 0
        tool_errors = 0
        turn = start_turn
        while turn < self.config.max_turns and not state.finished:
            turn += 1
            min_rows = None
            for c in state.contracts.list():
                if isinstance(c, MinRowsContract):
                    min_rows = c.min_rows
                    break
            phase = current_phase(state, contract_min_rows=min_rows)
            tracer.event(event="phase", turn=turn, phase=phase.name)

            brief = render_state_brief(state, self.registry)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": brief},
            ]
            if self.config.verbose:
                log.info("── turn %d ── phase=%s", turn, phase.name)
            t0 = time.time()
            tracer.event(event="llm_start", turn=turn, model=self.llm.config.model)
            try:
                raw = self.llm.chat(messages, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                msg = f"ERROR: LLM call failed: {e}"
                state.record(turn, "", "_llm", {}, msg, error=True)
                tracer.turn(
                    turn=turn, thought="", tool="_llm", args={},
                    observation=msg, error=True,
                    contracts=state.contracts_snapshot(),
                    n_rows=len(state.dataset),
                    raw_model_output=None,
                    elapsed_ms=(time.time() - t0) * 1000,
                )
                err_text = str(e)
                fatal_markers = (
                    "402", "Payment Required",
                    "401", "Unauthorized",
                    "out of credits",
                    "model id may not exist",
                    "check your API key",
                )
                if any(m in err_text for m in fatal_markers):
                    state.finish_reason = (
                        f"aborted: provider error not worth retrying — {err_text[:200]}"
                    )
                    tracer.note("aborted", state.finish_reason)
                    break
                tool_errors += 1
                if tool_errors >= self.config.max_consecutive_tool_errors:
                    break
                continue
            llm_ms = (time.time() - t0) * 1000

            try:
                call = parse_tool_call(raw)
                parse_failures = 0
            except ParseError as e:
                parse_failures += 1
                obs = (
                    f"ERROR: could not parse your reply as a tool-call JSON object.\n"
                    f"Reason: {e}\n"
                    f'Reply must be exactly: {{"thought":"...","tool":"...","args":{{...}}}}\n'
                    f"Your raw output (first 500 chars): {raw[:500]}"
                )
                state.record(turn, "", "_parse", {"raw": raw[:500]}, obs, error=True)
                tracer.turn(
                    turn=turn, thought="", tool="_parse",
                    args={"raw": raw[:500]},
                    observation=obs, error=True,
                    contracts=state.contracts_snapshot(),
                    n_rows=len(state.dataset),
                    raw_model_output=raw,
                    elapsed_ms=llm_ms,
                )
                if parse_failures >= self.config.max_consecutive_parse_failures:
                    tracer.note("aborted", f"{parse_failures} parse failures")
                    break
                continue

            if self.config.verbose:
                log.info("→ %s  args=%s", call.tool, _short(call.args))

            t1 = time.time()
            result = self.registry.dispatch(call.tool, call.args, state)
            tool_ms = (time.time() - t1) * 1000

            state.record(
                turn, call.thought, call.tool, call.args, result.output,
                error=result.error,
            )
            tracer.turn(
                turn=turn, thought=call.thought, tool=call.tool,
                args=call.args, observation=result.output, error=result.error,
                contracts=state.contracts_snapshot(),
                n_rows=len(state.dataset),
                raw_model_output=raw,
                elapsed_ms=llm_ms + tool_ms,
            )

            if result.error:
                tool_errors += 1
                if tool_errors >= self.config.max_consecutive_tool_errors:
                    tracer.note("aborted", f"{tool_errors} consecutive tool errors")
                    break
            else:
                tool_errors = 0
        return turn

    # ──────────────────────────────────────────────────────────────────
    # Self-verification
    # ──────────────────────────────────────────────────────────────────

    def _self_verify(self, state: AgentState, tracer: Tracer) -> dict:
        """Final-pass sanity check the agent runs on itself before declaring done.

        Returns ``{"ok": bool, "problems": [str], "stats": {...}}``.
        Deterministic checks first, then an optional LLM critique on a small
        sample. Never raises.
        """
        problems: list[str] = []
        stats: dict[str, Any] = {}
        try:
            rows = state.dataset.rows()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "problems": [f"cannot read dataset: {e}"], "stats": {}}

        stats["n_rows"] = len(rows)

        # Contract-level checks.
        contract_snap = state.contracts_snapshot()
        stats["contracts"] = contract_snap
        for c in contract_snap:
            if not c.get("ok"):
                problems.append(f"contract not satisfied: {c.get('name')} — {c.get('detail')}")

        if rows:
            # Empty / null-heavy rows.
            for i, r in enumerate(rows[:200]):
                if not isinstance(r, dict) or len(r) == 0:
                    problems.append(f"row {i} is empty / not a dict")
                    break

            # Schema homogeneity: drop > 30% missing for any "well-known" key.
            keys: dict[str, int] = {}
            for r in rows:
                if isinstance(r, dict):
                    for k in r.keys():
                        keys[k] = keys.get(k, 0) + 1
            n = len(rows)
            sparse = [
                k for k, count in keys.items()
                if count < n * 0.7 and not k.startswith("_") and k != "id"
            ]
            if sparse:
                stats["sparse_fields"] = sparse[:10]

        # LLM critique on a small sample — best-effort, never blocking.
        try:
            critique = self._llm_critique(state, rows[:5])
            stats["llm_critique"] = critique
            if critique and critique.get("severity") in ("high", "blocker"):
                problems.extend(critique.get("issues", []))
        except Exception as e:  # noqa: BLE001
            stats["llm_critique_error"] = str(e)[:300]

        verdict = {"ok": len(problems) == 0, "problems": problems, "stats": stats}
        tracer.event(event="verify", verdict=verdict)
        return verdict

    def _llm_critique(self, state: AgentState, sample: list[dict]) -> dict:
        if not sample:
            return {}
        system = (
            "You audit a small sample of dataset rows produced by a scraping "
            "agent. Compare them against the user's GOAL. Return ONE JSON "
            "object: {\"severity\": \"none|low|high|blocker\", \"issues\": "
            "[\"...\"]}. Flag issues like: rows are placeholder/empty, fields "
            "the user explicitly asked for are missing in the sample, types "
            "are wrong (strings where numbers expected), values look wrong. "
            "If everything looks fine, severity is 'none' and issues is []."
        )
        user = (
            f"GOAL:\n{state.goal}\n\n"
            f"SAMPLE ROWS ({len(sample)} of {len(state.dataset)}):\n"
            + json.dumps(sample, ensure_ascii=False, indent=2)[:8000]
        )
        raw = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            # Strip fences and retry.
            s = raw.strip().lstrip("`").rstrip("`")
            try:
                return json.loads(s)
            except Exception:  # noqa: BLE001
                return {"severity": "none", "issues": [], "raw": raw[:400]}


def _short(args: dict) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) < 160 else s[:160] + "…"


def run_agent(
    goal: str,
    *,
    contracts: list[Contract],
    workdir: str | Path,
    llm: LLMClient | None = None,
    extraction_llm: LLMClient | None = None,
    dataset_schema: dict | None = None,
    unique_key: str | None = None,
    config: AgentConfig | None = None,
    extra_memory: dict[str, Any] | None = None,
) -> RunResult:
    """One-shot helper. Sets up state, runs the agent, returns the result."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(workdir / "dataset.jsonl", schema=dataset_schema, unique_key=unique_key)
    memory = Memory(workdir / "memory.json")
    if extra_memory:
        for k, v in extra_memory.items():
            memory.set(k, v)

    state = AgentState(
        goal=goal,
        dataset=dataset,
        contracts=ContractBook(contracts),
        memory=memory,
        workdir=str(workdir),
    )

    llm = llm or LLMClient()
    agent = Agent(llm=llm, extraction_llm=extraction_llm, config=config)
    return agent.run(state)
