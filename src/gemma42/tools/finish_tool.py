"""The finish tool. Gated by contract satisfaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class FinishTool(Tool):
    name = "finish"
    description = (
        "Declare the task complete. ONLY allowed when every contract reports "
        "ok=True. If any contract still fails, this call is REJECTED with a "
        "list of what is still missing, and you must keep working."
    )
    args_schema = {
        "summary": {
            "type": "string",
            "description": "One-paragraph summary of what was produced.",
        }
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        snap = state.contracts_snapshot()
        failing = [c for c in snap if not c["ok"]]
        if failing:
            lines = ["REFUSED: cannot finish; contracts still failing:"]
            for c in failing:
                lines.append(f"  - {c['name']}: {c['detail']}")
            lines.append(
                "Keep working — add more rows, fix missing fields, or call "
                "add_contract if the spec has genuinely changed."
            )
            return ToolResult(output="\n".join(lines), error=True)
        state.finished = True
        state.finish_reason = args.get("summary", "done")
        return ToolResult(output=f"FINISHED: {state.finish_reason}")
