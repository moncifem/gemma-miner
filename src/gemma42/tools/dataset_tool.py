"""Tools that talk to the current run's Dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class DatasetAppendTool(Tool):
    name = "dataset_append"
    description = (
        "Append one or more rows to the current dataset (JSONL on disk). Each "
        "row is a JSON object. Rows that violate the dataset's schema, miss "
        "required fields, or duplicate the unique key are rejected — the "
        "output tells you which rows failed and why, so you can fix and retry. "
        "Always check `dataset_stats` afterwards to confirm progress."
    )
    args_schema = {
        "rows": {
            "type": "array",
            "description": "List of row objects to append.",
            "items": {"type": "object"},
        }
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        rows = args.get("rows")
        if rows is None and "row" in args:
            rows = [args["row"]]
        if not isinstance(rows, list):
            return ToolResult(output="ERROR: 'rows' must be a list of objects", error=True)
        added = 0
        failures: list[str] = []
        for i, row in enumerate(rows):
            ok, reason = state.dataset.append(row)
            if ok:
                added += 1
            else:
                preview = json.dumps(row, ensure_ascii=False)[:120]
                failures.append(f"  [{i}] {reason}  | row={preview}")
        out = [f"added: {added}/{len(rows)}", f"total_rows_now: {len(state.dataset)}"]
        if failures:
            out.append("failures:")
            out.extend(failures[:10])
            if len(failures) > 10:
                out.append(f"  ... and {len(failures) - 10} more")
        return ToolResult(output="\n".join(out))


class DatasetStatsTool(Tool):
    name = "dataset_stats"
    description = (
        "Summary of the current dataset: row count, per-field coverage, file path, "
        "and the live status of every active contract. Call this whenever you need "
        "to decide whether the run is done."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        stats = state.dataset.stats()
        contracts = state.contracts_snapshot()
        lines = [
            f"rows: {stats['n_rows']}",
            f"path: {stats['path']}",
            "field_coverage:",
        ]
        for k, v in sorted(stats["field_coverage"].items(), key=lambda x: -x[1]):
            pct = (v / stats["n_rows"] * 100) if stats["n_rows"] else 0
            lines.append(f"  {v:6d} ({pct:5.1f}%)  {k}")
        lines.append("contracts:")
        for c in contracts:
            mark = "OK" if c["ok"] else "FAIL"
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        return ToolResult(output="\n".join(lines))


class DatasetSampleTool(Tool):
    name = "dataset_sample"
    description = "Return the first N rows of the dataset as JSON, for inspection."
    args_schema = {"n": {"type": "integer", "default": 3}}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        n = int(args.get("n") or 3)
        rows = state.dataset.rows()[:n]
        return ToolResult(output=json.dumps(rows, indent=2, ensure_ascii=False))
