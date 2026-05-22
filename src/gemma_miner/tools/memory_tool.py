"""Memory tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


class MemorySetTool(Tool):
    name = "memory_set"
    description = (
        "Store a fact in long-term memory under a key. Use for selectors that "
        "worked, schemas that fit a site, pagination patterns, anything you "
        "want to recall later in this run or in future runs against the same "
        "site. Values may be any JSON value."
    )
    args_schema = {
        "key": {"type": "string"},
        "value": {"description": "Any JSON-serialisable value."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        key = args.get("key")
        if not key:
            return ToolResult(output="ERROR: 'key' required", error=True)
        state.memory.set(key, args.get("value"))
        return ToolResult(output=f"saved memory[{key!r}]")


class MemoryGetTool(Tool):
    name = "memory_get"
    is_readonly = True
    description = "Retrieve a previously stored memory value by key."
    args_schema = {"key": {"type": "string"}}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        key = args.get("key")
        if not key:
            return ToolResult(output="ERROR: 'key' required", error=True)
        val = state.memory.get(key)
        if val is None:
            return ToolResult(output=f"(no memory under {key!r})")
        return ToolResult(output=json.dumps(val, indent=2, ensure_ascii=False))


class MemoryListTool(Tool):
    name = "memory_list"
    is_readonly = True
    description = "List all memory keys (not values)."
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        keys = state.memory.keys()
        return ToolResult(output="\n".join(keys) if keys else "(memory empty)")
