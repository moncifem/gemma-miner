"""Tool base class & result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


@dataclass
class ToolResult:
    output: str
    error: bool = False
    artifact: Any = None  # optional structured payload (not shown to model)


class ToolError(Exception):
    pass


class Tool(ABC):
    name: str
    description: str
    args_schema: dict  # JSON-schema-ish for documentation

    @abstractmethod
    def run(self, args: dict, state: "AgentState") -> ToolResult: ...

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_schema,
        }
