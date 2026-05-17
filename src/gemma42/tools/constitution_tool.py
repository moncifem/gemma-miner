"""Constitution tools — declared rules + inferred-rule proposals + verification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.constitution import Constitution, Rule, infer_rules
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _load_constitution(state: "AgentState") -> Constitution:
    items = state.memory.get("rules") or []
    return Constitution.from_list(items)


def _save_constitution(state: "AgentState", c: Constitution) -> None:
    state.memory.set("rules", c.to_list())


class RuleAddTool(Tool):
    name = "rule_add"
    description = (
        "Add a constitutional rule that every dataset row must satisfy. "
        "Rules are typed JSON predicates. Supported kinds + ops:\n\n"
        "  per_row / check     : {field, eq/ne/gt/ge/lt/le/in/not_in/match/exists}\n"
        "  per_row / implies   : {when: {field, eq:...}, then: {field, gt:0}}\n"
        "  per_row / in_range  : {field, min, max} (numeric or YYYY-MM-DD)\n"
        "  per_row / enum_in   : {field, values: [...]}\n"
        "  cross_row / unique  : {fields: [...]}  (composite uniqueness)\n"
        "  cross_row / monotonic : {field, order: asc|desc}\n\n"
        "Args: name, kind, op, spec (dict), description (optional), "
        "severity ('error' default | 'warning')."
    )
    args_schema = {
        "name":        {"type": "string"},
        "kind":        {"type": "string", "enum": ["per_row", "cross_row"]},
        "op":          {"type": "string"},
        "spec":        {"type": "object"},
        "description": {"type": "string"},
        "severity":    {"type": "string", "enum": ["error", "warning"], "default": "error"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        name = args.get("name")
        if not name:
            return ToolResult(output="ERROR: 'name' required", error=True)
        try:
            rule = Rule.from_dict(args)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR: invalid rule: {e}", error=True)
        c = _load_constitution(state)
        c.add(rule)
        _save_constitution(state, c)
        return ToolResult(output=f"rule '{name}' added (now {len(c.rules)} rules)")


class RuleListTool(Tool):
    name = "rule_list"
    description = "Show every constitutional rule currently attached to this run."
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        c = _load_constitution(state)
        if not c.rules:
            return ToolResult(output="(no rules)")
        out = [f"{len(c.rules)} rule(s):"]
        for r in c.rules:
            inferred = " [inferred]" if r.inferred else ""
            out.append(f"  - {r.name}  ({r.kind}/{r.op}, {r.severity}){inferred}")
            if r.description:
                out.append(f"      {r.description}")
        return ToolResult(output="\n".join(out))


class RulesInferTool(Tool):
    name = "rules_infer"
    description = (
        "Look at the current dataset and PROPOSE rules the corpus appears "
        "to follow (date ranges, small-set enums, uniqueness, numeric "
        "ranges). The proposed rules are NOT added automatically; the tool "
        "returns them as a list. Set `auto_add=true` to add them immediately."
    )
    args_schema = {
        "auto_add": {"type": "boolean", "default": False},
        "min_rows": {"type": "integer", "default": 8},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        rows = state.dataset.rows()
        proposed = infer_rules(rows, min_rows=int(args.get("min_rows") or 8))
        out: list = []
        for r in proposed:
            out.append({
                "name": r.name, "kind": r.kind, "op": r.op,
                "spec": r.spec, "description": r.description,
                "inferred": True,
            })
        if args.get("auto_add") and proposed:
            c = _load_constitution(state)
            for r in proposed:
                c.add(r)
            _save_constitution(state, c)
            return ToolResult(output=(
                f"inferred + added {len(proposed)} rules:\n"
                + json.dumps(out, indent=2, ensure_ascii=False)
            ))
        return ToolResult(output=(
            f"proposed {len(proposed)} rules (not added yet):\n"
            + json.dumps(out, indent=2, ensure_ascii=False)
        ))


class DatasetVerifyTool(Tool):
    name = "dataset_verify"
    description = (
        "Run every constitutional rule against the dataset. Returns the "
        "list of failing rows + failing cross-row constraints. Use this "
        "to find quality issues before exporting."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        c = _load_constitution(state)
        rows = state.dataset.rows()
        report = c.evaluate(rows)
        lines = [
            f"rules:    {report['n_rules']}",
            f"errors:   {report['n_errors']}",
            f"OK:       {report['ok']}",
        ]
        if report["per_row_failures"]:
            lines.append(f"per-row failures (first 5):")
            for f in report["per_row_failures"][:5]:
                lines.append(f"  {f['rule']} on id={f['id']}: {f['message']}")
        if report["cross_row_failures"]:
            lines.append("cross-row failures (first 5):")
            for f in report["cross_row_failures"][:5]:
                lines.append(f"  {f['rule']} id={f['id']}: {f['message']}")
        return ToolResult(output="\n".join(lines), artifact=report)
