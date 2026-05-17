"""Goal-Decomposition-Tree tool. The agent calls this whenever it wants
to know what's left to do against the user's actual goal (not just the
contract minimums)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma42.contracts import MinRowsContract
from gemma42.gdt import build_tree_for_goal, render_tree
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class GoalTreeTool(Tool):
    name = "goal_tree"
    description = (
        "Show the Goal Decomposition Tree — a checklist parsed from the "
        "user's original goal, with each leaf auto-evaluated against the "
        "current state. Use this to decide what to do next when the contract "
        "view is too coarse."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        min_rows = None
        for c in state.contracts.list():
            if isinstance(c, MinRowsContract):
                min_rows = c.min_rows
                break
        tree = build_tree_for_goal(state.goal, contract_min_rows=min_rows)
        evaluated = tree.evaluate(state)
        return ToolResult(output=render_tree(evaluated), artifact=evaluated)
