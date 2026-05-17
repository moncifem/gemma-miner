"""Tools that let the agent mutate its own contracts mid-run.

This is the heart of contract management: when the user (or the agent) decides
the spec has changed — "actually I need 200 rows" or "also require a 'points'
field" — call `add_contract` and the main loop will keep going until the new
contract is satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma42.contracts import FieldsContract, MinRowsContract, UniqueFieldContract
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class AddContractTool(Tool):
    name = "add_contract"
    description = (
        "Add (or replace) a contract that must be satisfied before the run can "
        "finish. Supported kinds:\n"
        "  - min_rows         args: {min_rows: int}\n"
        "  - required_fields  args: {fields: [str, ...]}\n"
        "  - unique_field     args: {field: str}\n"
        "If a contract with the same kind already exists, it is replaced."
    )
    args_schema = {
        "kind": {
            "type": "string",
            "enum": ["min_rows", "required_fields", "unique_field"],
        },
        "min_rows": {"type": "integer", "description": "for kind=min_rows"},
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "for kind=required_fields",
        },
        "field": {"type": "string", "description": "for kind=unique_field"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        kind = args.get("kind")
        if kind == "min_rows":
            n = int(args.get("min_rows") or 0)
            if n <= 0:
                return ToolResult(output="ERROR: 'min_rows' must be > 0", error=True)
            state.contracts.add(MinRowsContract(min_rows=n))
            return ToolResult(output=f"added contract: min_rows >= {n}")
        if kind == "required_fields":
            fields = args.get("fields") or []
            if not fields:
                return ToolResult(output="ERROR: 'fields' list required", error=True)
            state.contracts.add(FieldsContract(required_fields=list(fields)))
            return ToolResult(output=f"added contract: required_fields={fields}")
        if kind == "unique_field":
            f = args.get("field")
            if not f:
                return ToolResult(output="ERROR: 'field' required", error=True)
            state.contracts.add(UniqueFieldContract(field=f))
            return ToolResult(output=f"added contract: unique_field='{f}'")
        return ToolResult(output=f"ERROR: unknown kind '{kind}'", error=True)


class ContractStatusTool(Tool):
    name = "contract_status"
    description = "List every contract and whether it is currently satisfied."
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        snap = state.contracts_snapshot()
        if not snap:
            return ToolResult(output="(no contracts — finish is allowed)")
        lines = []
        for c in snap:
            mark = "OK" if c["ok"] else "FAIL"
            lines.append(f"[{mark}] {c['name']}: {c['detail']}  ({c['description']})")
        return ToolResult(output="\n".join(lines))
