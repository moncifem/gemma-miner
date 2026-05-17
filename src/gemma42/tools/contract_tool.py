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
        "Add (or replace) a hard contract that gates the `finish` tool. "
        "REQUIRED FIRST ARG `kind` — one of EXACTLY these three strings:\n"
        "  • \"min_rows\"         → also pass {min_rows: <int>}\n"
        "  • \"required_fields\"  → also pass {fields: [<str>, ...]}\n"
        "  • \"unique_field\"     → also pass {field: <str>}\n\n"
        "Examples:\n"
        "  add_contract(kind=\"min_rows\", min_rows=30)\n"
        "  add_contract(kind=\"required_fields\", fields=[\"title\",\"score\"])\n"
        "  add_contract(kind=\"unique_field\", field=\"id\")\n\n"
        "Most runs do NOT need to call this — contracts are set at run start "
        "from the user's request. Only use add_contract when the user changes "
        "their spec mid-run (e.g. 'actually I also need a date field')."
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
        valid_kinds = ("min_rows", "required_fields", "unique_field")
        if kind not in valid_kinds:
            return ToolResult(
                output=(
                    f"ERROR: 'kind' must be one of {valid_kinds}, got {kind!r}. "
                    "Example: add_contract(kind=\"required_fields\", "
                    "fields=[\"title\", \"score\"])."
                ),
                error=True,
            )
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
        # unique_field
        f = args.get("field")
        if not f:
            return ToolResult(output="ERROR: 'field' required", error=True)
        state.contracts.add(UniqueFieldContract(field=f))
        return ToolResult(output=f"added contract: unique_field='{f}'")


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
