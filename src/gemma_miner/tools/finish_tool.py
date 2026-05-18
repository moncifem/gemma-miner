"""The finish tool. Gated by contract satisfaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


class FinishTool(Tool):
    name = "finish"
    description = (
        "Declare the task complete. Normally allowed when every contract "
        "reports ok=True. If contracts are STILL failing but you've made "
        "real progress (e.g. min_rows is met but a few rows are missing an "
        "optional field), pass `force=true` to finish anyway with a warning. "
        "The summary should describe what was produced AND what limitations "
        "remain."
    )
    args_schema = {
        "summary": {
            "type": "string",
            "description": "One-paragraph summary of what was produced.",
        },
        "force": {
            "type": "boolean",
            "default": False,
            "description": (
                "Finish even with failing contracts. Only set true when "
                "min_rows is met and the remaining failures are minor field "
                "gaps you can't fix without re-scraping."
            ),
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        snap = state.contracts_snapshot()
        failing = [c for c in snap if not c["ok"]]
        force = bool(args.get("force", False))
        if failing and not force:
            # Decide if we should AUTO-soft-finish: min_rows is met AND the
            # remaining failures are non-row-count contracts (field gaps).
            min_rows_satisfied = all(
                c["ok"] for c in snap if c["name"] == "min_rows"
            )
            non_count_failures = [c for c in failing if c["name"] != "min_rows"]
            if min_rows_satisfied and non_count_failures and len(failing) == len(non_count_failures):
                # Tell the model it can finish with force=true.
                lines = [
                    "PARTIAL: min_rows is met but other contracts still fail:"
                ]
                for c in failing:
                    lines.append(f"  - {c['name']}: {c['detail']}")
                lines.append(
                    "\nIf these failures are acceptable (e.g. a few rows "
                    "missing an optional field), call `finish` again with "
                    "`force=true` and a summary that mentions the limitation."
                )
                return ToolResult(output="\n".join(lines), error=True)
            # No row data yet — fully refuse.
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
        suffix = " (FORCED with failing contracts)" if (failing and force) else ""
        return ToolResult(output=f"FINISHED{suffix}: {state.finish_reason}")
