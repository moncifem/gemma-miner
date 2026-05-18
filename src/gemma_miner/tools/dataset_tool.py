"""Tools that talk to the current run's Dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


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
        # Common model mistake: pass a string that's the contents of a JSON
        # file (because the $file resolver expanded {"$file": "..."} into
        # the raw text). Try to parse it.
        if isinstance(rows, str):
            text = rows.strip()
            # Path-ish? Try to read the file.
            from pathlib import Path
            try:
                p = Path(text)
                if not p.is_absolute():
                    p = Path(state.workdir) / p
                if p.exists() and p.is_file() and p.stat().st_size < 50_000_000:
                    text = p.read_text(encoding="utf-8")
            except (OSError, ValueError):
                pass
            try:
                rows = json.loads(text)
            except Exception:  # noqa: BLE001
                # Try line-delimited JSON (.jsonl) as a fallback.
                lines = [l for l in text.splitlines() if l.strip()]
                parsed: list = []
                for ln in lines:
                    try:
                        parsed.append(json.loads(ln))
                    except Exception:  # noqa: BLE001
                        parsed = None  # type: ignore
                        break
                if parsed is not None:
                    rows = parsed
                else:
                    return ToolResult(
                        output=(
                            "ERROR: 'rows' is a string we can't parse as JSON or JSONL. "
                            "Either pass a list of row objects directly, or write the "
                            "JSON list to a file in the workdir and call "
                            "dataset_append(rows={\"$file\": \"<path>\"}) — the loader "
                            "will read and parse it."
                        ),
                        error=True,
                    )
        # Single dict shorthand → wrap in list.
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            return ToolResult(
                output=(
                    "ERROR: 'rows' must be a list of objects (or a string/path that "
                    "parses to one). Got: " + type(rows).__name__
                ),
                error=True,
            )
        # Auto-id any row that's missing one. Uses the central, idempotent
        # `synthesize_id` so the SAME row content always produces the SAME id
        # across re-runs and across tools (dataset_append / scrape_paginated /
        # llm_scrape). This is what makes bronze↔silver join work.
        from gemma_miner.dataset import ensure_row_id

        for row in rows:
            if isinstance(row, dict):
                ensure_row_id(row)
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


class DatasetFromQueueTool(Tool):
    name = "dataset_from_queue"
    description = (
        "Push every queued item directly into the dataset. Use this when the "
        "LISTING page already contains all the fields the user asked for — no "
        "detail pages or attachments needed. This bypasses process_queue "
        "(which is for runs that need a per-item detail page or PDF).\n\n"
        "Args:\n"
        "  fields    : optional list of queue-item field names to keep. "
        "Default = every non-id field.\n"
        "  field_map : optional {observed_name → canonical_name} rename map. "
        "If omitted, the tool auto-maps variants of your required_fields "
        "contract (e.g. queue has 'comments' but contract wants 'n_comments').\n"
        "  drop_empty_rows : default true. Skip queue items where every "
        "required field is null/empty.\n"
        "  max       : optional cap on how many to push (default: all).\n\n"
        "Marks each queue item as processed; running again is a no-op."
    )
    args_schema = {
        "fields":           {"type": "array"},
        "field_map":        {"type": "object"},
        "drop_empty_rows":  {"type": "boolean", "default": True},
        "max":              {"type": "integer"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        queue = state.memory.get("queue", []) or []
        processed_list = state.memory.get("processed", []) or []
        processed = {str(x) for x in processed_list}

        keep_fields = args.get("fields")
        field_map: dict = args.get("field_map") or {}
        drop_empty = bool(args.get("drop_empty_rows", True))
        cap = args.get("max")
        cap = int(cap) if cap else None

        # Auto-build field_map from the FieldsContract if not provided.
        # E.g. queue has "comments" but contract wants "n_comments".
        if not field_map:
            try:
                from gemma_miner.contracts import FieldsContract, _field_variants
                canonical: list[str] = []
                for c in state.contracts.list():
                    if isinstance(c, FieldsContract):
                        canonical.extend(c.required_fields)
                observed: set[str] = set()
                for q in queue:
                    if isinstance(q, dict):
                        observed.update(q.keys())
                for canon in canonical:
                    if canon in observed:
                        continue
                    for v in _field_variants(canon):
                        if v in observed and v != canon:
                            field_map[v] = canon
                            break
            except Exception:  # noqa: BLE001
                pass

        required_for_drop: list[str] = []
        try:
            from gemma_miner.contracts import FieldsContract
            for c in state.contracts.list():
                if isinstance(c, FieldsContract):
                    required_for_drop.extend(c.required_fields)
        except Exception:  # noqa: BLE001
            pass

        appended = 0
        skipped_processed = 0
        skipped_empty = 0
        failures = 0

        for q in queue:
            if cap is not None and appended >= cap:
                break
            if not isinstance(q, dict):
                continue
            qid = q.get("id")
            if qid is not None and str(qid) in processed:
                skipped_processed += 1
                continue
            # Build the output row.
            if keep_fields:
                base = {k: q.get(k) for k in keep_fields if k in q}
            else:
                base = {k: v for k, v in q.items() if not str(k).startswith("_")}
            # Apply renames.
            row: dict = {}
            for k, v in base.items():
                row[field_map.get(k, k)] = v
            # Optionally skip rows that are empty across required fields.
            if drop_empty and required_for_drop:
                if not any(row.get(f) not in (None, "") for f in required_for_drop):
                    skipped_empty += 1
                    if qid is not None:
                        processed_list.append(str(qid))
                        processed.add(str(qid))
                    continue
            # Ensure id is set (dataset's unique_key may rely on it).
            if "id" not in row and qid is not None:
                row["id"] = str(qid)
            ok, reason = state.dataset.append(row)
            if ok:
                appended += 1
                if qid is not None:
                    processed_list.append(str(qid))
                    processed.add(str(qid))
            else:
                failures += 1
                if failures <= 3:
                    pass  # we'll include sample failures in output

        state.memory.set("processed", processed_list)

        out = [
            f"dataset_from_queue:",
            f"  appended:           {appended}",
            f"  skipped (processed): {skipped_processed}",
            f"  skipped (empty):    {skipped_empty}",
            f"  failures:           {failures}",
            f"  total rows now:     {len(state.dataset)}",
            f"  remaining in queue: {len(queue) - len(processed)}",
        ]
        if field_map:
            out.append(f"  renamed columns:    {field_map}")
        return ToolResult(output="\n".join(out),
                           artifact={"appended": appended, "rows": len(state.dataset)})
