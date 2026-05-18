"""Per-run agent state. Holds everything that changes between turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from gemma_miner.contracts import ContractBook
from gemma_miner.dataset import Dataset
from gemma_miner.memory import Memory


class _JoinedDatasetView:
    """Duck-type Dataset facade over a precomputed list of merged rows.

    Contracts call `.rows()` and read `.path`; nothing else. Iteration and
    `__len__` are supported so any contract that uses them keeps working."""
    def __init__(self, rows: list[dict], path):
        self._rows = rows
        self.path = path
    def rows(self) -> list[dict]:
        return list(self._rows)
    def __iter__(self):
        return iter(self._rows)
    def __len__(self) -> int:
        return len(self._rows)


@dataclass
class TurnRecord:
    turn: int
    thought: str
    tool: str
    args: dict
    observation: str
    error: bool = False


@dataclass
class AgentState:
    goal: str
    dataset: Dataset
    contracts: ContractBook
    memory: Memory
    workdir: str
    history: list[TurnRecord] = field(default_factory=list)
    # Caps the stored observation per turn. Big enough for one html_inspect
    # sample block + a couple of dataset_sample rows, small enough that the
    # prompt brief sent over the wire stays under ~100KB. The full body is
    # always on disk in cache/<slug>.html; the agent can re-read it via
    # read_file or html_inspect when it needs more.
    max_observation_chars: int = 40_000
    finished: bool = False
    finish_reason: str = ""
    # In-run distilled "what worked / what didn't" notes, refreshed every K
    # turns by the reflection pass. Surfaced into every subsequent prompt so
    # the agent stops repeating the same mistake within a single run.
    lessons: list[str] = field(default_factory=list)
    last_reflection_turn: int = 0
    # Optional progress callback that long-running tools can call with
    # incremental updates (e.g. one extraction per item). The agent loop
    # wires this to the Tracer so the live CLI feed can render them.
    progress_hook: Callable[[dict], None] | None = None
    # Bronze (raw harvest = `self.dataset`) and silver (typed variables from
    # the codebook extraction) are stored as SEPARATE jsonl files keyed by
    # `id`. The silver dataset is created lazily the first time something
    # calls `extracted_dataset()`. They join cleanly at export time.
    _extracted_dataset: Dataset | None = None

    def extracted_dataset(self) -> Dataset:
        """Return the silver dataset (typed codebook variables, keyed by id).
        Created on demand at <workdir>/extracted.jsonl."""
        if self._extracted_dataset is None:
            from pathlib import Path
            self._extracted_dataset = Dataset(
                Path(self.workdir) / "extracted.jsonl",
                unique_key="id",
            )
        return self._extracted_dataset

    def emit_progress(self, **fields: Any) -> None:
        """Best-effort progress emission. Tools call this inside long loops
        (e.g. extract_items per item). Never raises."""
        hook = self.progress_hook
        if hook is None:
            return
        try:
            hook(dict(fields))
        except Exception:  # noqa: BLE001
            pass

    def record(
        self, turn: int, thought: str, tool: str, args: dict, observation: str, error: bool = False
    ) -> None:
        if len(observation) > self.max_observation_chars:
            observation = observation[: self.max_observation_chars] + f"\n... [truncated, total {len(observation)} chars]"
        self.history.append(
            TurnRecord(turn=turn, thought=thought, tool=tool, args=args, observation=observation, error=error)
        )

    def contracts_snapshot(self) -> list[dict]:
        """Contracts judge the user-visible dataset, which is the merge of
        bronze (raw harvest) + silver (typed extraction) joined by `id`.

        Without the merge, every contract checking a codebook variable like
        `has_injunction` would fail because those fields only exist in
        silver (extracted.jsonl) — the user would see "missing in 223/223
        rows" even after a successful extraction."""
        return self.contracts.status(self._joined_view())

    def _joined_view(self) -> "Dataset":
        """Read-only Dataset-shaped view that yields merged bronze+silver rows.
        If silver is empty, returns bronze unchanged (zero overhead)."""
        if self._extracted_dataset is None or len(self._extracted_dataset) == 0:
            return self.dataset
        silver_by_id: dict[str, dict] = {
            str(r.get("id")): r for r in self._extracted_dataset.rows()
            if r.get("id") is not None
        }
        merged_rows: list[dict] = []
        for r in self.dataset.rows():
            sid = str(r.get("id"))
            if sid in silver_by_id:
                m = dict(r)
                for k, v in silver_by_id[sid].items():
                    if k != "id":
                        m[k] = v
                merged_rows.append(m)
            else:
                merged_rows.append(r)
        return _JoinedDatasetView(merged_rows, self.dataset.path)

    def to_brief(self) -> dict[str, Any]:
        ex_n = 0
        ex_path = None
        if self._extracted_dataset is not None:
            ex_n = len(self._extracted_dataset)
            ex_path = str(self._extracted_dataset.path)
        return {
            "goal": self.goal,
            "dataset_rows": len(self.dataset),
            "dataset_path": str(self.dataset.path),
            "extracted_rows": ex_n,
            "extracted_path": ex_path,
            "contracts": self.contracts_snapshot(),
            "memory_keys": self.memory.keys(),
            "workdir": self.workdir,
        }
