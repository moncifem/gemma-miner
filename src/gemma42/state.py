"""Per-run agent state. Holds everything that changes between turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gemma42.contracts import ContractBook
from gemma42.dataset import Dataset
from gemma42.memory import Memory


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
    max_observation_chars: int = 8000
    finished: bool = False
    finish_reason: str = ""

    def record(
        self, turn: int, thought: str, tool: str, args: dict, observation: str, error: bool = False
    ) -> None:
        if len(observation) > self.max_observation_chars:
            observation = observation[: self.max_observation_chars] + f"\n... [truncated, total {len(observation)} chars]"
        self.history.append(
            TurnRecord(turn=turn, thought=thought, tool=tool, args=args, observation=observation, error=error)
        )

    def contracts_snapshot(self) -> list[dict]:
        return self.contracts.status(self.dataset)

    def to_brief(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "dataset_rows": len(self.dataset),
            "dataset_path": str(self.dataset.path),
            "contracts": self.contracts_snapshot(),
            "memory_keys": self.memory.keys(),
            "workdir": self.workdir,
        }
