"""Tool base class & result types."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_miner.state import AgentState

# Default output budget before the registry truncates. 8 KB is enough to
# convey any tool result to a small model without bloating the context.
DEFAULT_MAX_OUTPUT_CHARS: int = 8_000


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

    # How many chars the registry is allowed to send back to the model.
    # Outputs beyond this are truncated with a note. Set to math.inf for
    # tools whose output must NEVER be cut (self-bounding or path-returning).
    max_output_chars: int | float = DEFAULT_MAX_OUTPUT_CHARS

    # True -> tool never mutates shared state (http_get, html_inspect, ...).
    # False -> tool writes files, datasets, memory, queue (fail-closed default).
    is_readonly: bool = False

    # Key prefixes the CLI should extract for the one-line activity feed summary.
    # Each entry is the START of a "key: value" line in the tool's output.
    # The CLI joins matched lines into a compact summary instead of showing the
    # raw first line. Tools own this knowledge -- not the CLI.
    summary_fields: tuple[str, ...] = ()

    @abstractmethod
    def run(self, args: dict, state: "AgentState") -> ToolResult: ...

    def description_dynamic(self, args: dict, state: "AgentState") -> str | None:
        """Return a context-sensitive description override at call time.

        Return None to use the static self.description string. Tools override
        this when the useful description depends on runtime state (e.g. which
        cache files exist, how many rows are in the dataset).
        """
        return None

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_schema,
        }
