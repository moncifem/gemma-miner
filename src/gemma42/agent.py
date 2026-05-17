"""The main agent loop.

A single ReAct-style loop. Each turn:
  1. Render the state brief.
  2. Ask the LLM for ONE tool call.
  3. Parse it. On parse failure, push an error observation and retry.
  4. Dispatch the tool. Append a TurnRecord.
  5. If `finish` succeeded, exit. Otherwise loop until max_turns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gemma42.contracts import Contract, ContractBook
from gemma42.dataset import Dataset
from gemma42.llm import LLMClient
from gemma42.memory import Memory
from gemma42.parsing import ParseError, parse_tool_call
from gemma42.prompts import SYSTEM_PROMPT, render_state_brief
from gemma42.state import AgentState
from gemma42.tools.registry import ToolRegistry, default_registry
from gemma42.tracing import Tracer

log = logging.getLogger("gemma42")


@dataclass
class AgentConfig:
    max_turns: int = 60
    max_consecutive_parse_failures: int = 3
    max_consecutive_tool_errors: int = 6
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

    def run(self, state: AgentState) -> RunResult:
        tracer = Tracer(state.workdir)
        for sub in self.config.event_subscribers:
            tracer.subscribe(sub)

        # Wire the state's progress hook into the tracer so long-running
        # tools (extract_items, codebook_design, …) can stream per-item
        # progress events through the same channel the CLI feed listens on.
        def _progress(fields: dict) -> None:
            tracer.event(event=fields.pop("event", "progress"), **fields)
        state.progress_hook = _progress

        # Open the autobiography (L3 project + L4 global) for this run.
        from gemma42.autobiography.store import global_db, project_db

        proj_db = project_db(state.workdir)
        try:
            episode = proj_db.start_episode(
                workdir=str(state.workdir),
                goal=state.goal,
                trace_path=str(tracer.log_path),
            )
            state.memory.set("_episode_id", episode.id)
        except Exception:  # noqa: BLE001
            episode = None
        finally:
            proj_db.close()

        tracer.note("run_start", state.goal, contracts=[c.description for c in state.contracts.list()])
        parse_failures = 0
        tool_errors = 0
        turn = 0
        try:
            while turn < self.config.max_turns and not state.finished:
                turn += 1
                from gemma42.phases import current_phase

                # Compute the phase BEFORE this turn so we can broadcast it.
                min_rows = None
                from gemma42.contracts import MinRowsContract as _MR
                for c in state.contracts.list():
                    if isinstance(c, _MR):
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
                    log.info("── turn %d ──", turn)
                t0 = time.time()
                tracer.event(event="llm_start", turn=turn,
                             model=self.llm.config.model)
                try:
                    raw = self.llm.chat(messages, temperature=0.2)
                except Exception as e:  # noqa: BLE001
                    msg = f"ERROR: LLM call failed: {e}"
                    state.record(turn, "", "_llm", {}, msg, error=True)
                    try:
                        from gemma42.failure_log import log_failure as _fl

                        _fl(state.workdir, kind="llm_call_failed",
                            tool="_llm", turn=turn,
                            payload={"error": str(e)[:600]})
                    except Exception:  # noqa: BLE001
                        pass
                    tracer.turn(
                        turn=turn, thought="", tool="_llm", args={},
                        observation=msg, error=True,
                        contracts=state.contracts_snapshot(),
                        n_rows=len(state.dataset),
                        raw_model_output=None,
                        elapsed_ms=(time.time() - t0) * 1000,
                    )
                    # Fail-fast on non-retryable provider errors. Retrying a
                    # 402 Payment Required / 401 Unauthorized just burns
                    # more turns producing the same error.
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
                        tracer.note(
                            "aborted",
                            f"non-retryable provider error: {err_text[:200]}",
                        )
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
                    # Record the failure with the FULL raw output for debugging.
                    try:
                        from gemma42.failure_log import log_failure as _fl

                        _fl(state.workdir, kind="parse_error",
                            tool="_parse", turn=turn,
                            raw_response=raw,
                            payload={"reason": str(e)[:300]})
                    except Exception:  # noqa: BLE001
                        pass
                    truncated_hint = ""
                    if "TRUNCATED" in str(e):
                        # Count recent extractor_define attempts — if the model
                        # keeps trying to write specs that don't fit, push it
                        # to use `llm_scrape` which doesn't require complex JSON.
                        recent_extractor_tries = sum(
                            1 for h in state.history[-5:]
                            if h.tool == "extractor_define"
                            or (h.tool == "_parse" and "extractor_define" in (h.args.get("raw") or ""))
                        )
                        if recent_extractor_tries >= 2:
                            truncated_hint = (
                                "\n\n🛑 STOP using `extractor_define`. Your replies "
                                "keep getting truncated. Use `llm_scrape` "
                                "instead — it takes a simple field list, no "
                                "long regex spec required:\n"
                                "  llm_scrape(\n"
                                "    source='<cache_path>',\n"
                                "    fields=[{'name':'title'}, {'name':'score'}],\n"
                                "    target=<min_rows>,\n"
                                "    push_to_dataset=true\n"
                                "  )"
                            )
                        else:
                            truncated_hint = (
                                "\nFIX: your spec is too long for one reply. "
                                "Use `llm_scrape` instead — it takes a simple "
                                "field list and the model reads the page directly."
                            )
                    obs = (
                        f"ERROR: I could not parse your reply as a tool-call JSON object.\n"
                        f"Reason: {e}{truncated_hint}\n"
                        f"Reply must be exactly: {{\"thought\":\"...\",\"tool\":\"...\",\"args\":{{...}}}}\n"
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
                        log.error("giving up after %d parse failures", parse_failures)
                        tracer.note("aborted", f"{parse_failures} parse failures")
                        break
                    continue

                if self.config.verbose:
                    log.info("→ %s  args=%s", call.tool, _short(call.args))

                t1 = time.time()
                result = self.registry.dispatch(call.tool, call.args, state)
                tool_ms = (time.time() - t1) * 1000

                state.record(turn, call.thought, call.tool, call.args, result.output, error=result.error)
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
                    # Log to the dedicated failure file.
                    try:
                        from gemma42.failure_log import log_failure as _fl

                        _fl(state.workdir, kind="tool_error",
                            tool=call.tool, turn=turn,
                            raw_response=raw,
                            payload={
                                "args": call.args,
                                "observation": result.output[:2000],
                            })
                    except Exception:  # noqa: BLE001
                        pass
                    if tool_errors >= self.config.max_consecutive_tool_errors:
                        log.error("giving up after %d consecutive tool errors", tool_errors)
                        tracer.note("aborted", f"{tool_errors} consecutive tool errors")
                        break
                else:
                    tool_errors = 0

                # Periodic reflection — distil "what just worked / what
                # didn't" from the recent turns so the next prompts include
                # in-run lessons. Best-effort; swallow any failure.
                try:
                    from gemma42.reflection import reflect, should_reflect

                    if should_reflect(state):
                        new = reflect(state, self.llm)
                        if new:
                            tracer.event(
                                event="reflection", turn=turn,
                                added=new, total=len(state.lessons),
                            )
                except Exception:  # noqa: BLE001
                    pass
            tracer.note(
                "run_end",
                f"finished={state.finished} reason={state.finish_reason or 'stopped'}",
                turns=turn,
                rows=len(state.dataset),
            )
            # Close the episode in the autobiography.
            try:
                if episode is not None:
                    pdb = project_db(state.workdir)
                    try:
                        pdb.finish_episode(
                            episode.id,
                            status="finished" if state.finished else "stopped",
                            n_rows=len(state.dataset),
                            summary=state.finish_reason or "",
                        )
                        # Persist in-run lessons so future runs benefit.
                        for lesson in state.lessons:
                            try:
                                pdb.add_lesson(
                                    kind="in_run_reflection",
                                    text=lesson,
                                    episode_id=episode.id,
                                )
                            except Exception:  # noqa: BLE001
                                pass
                    finally:
                        pdb.close()
            except Exception:  # noqa: BLE001
                pass
        finally:
            tracer.close()

        return RunResult(
            finished=state.finished,
            finish_reason=state.finish_reason or ("max turns reached" if turn >= self.config.max_turns else "stopped"),
            turns=turn,
            dataset_path=str(state.dataset.path),
            n_rows=len(state.dataset),
            contracts=state.contracts_snapshot(),
            trace_log=str(Path(state.workdir) / "trace.log"),
            trace_jsonl=str(Path(state.workdir) / "trace.jsonl"),
        )


def _short(args: dict) -> str:
    import json

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
