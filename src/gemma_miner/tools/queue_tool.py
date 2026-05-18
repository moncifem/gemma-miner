"""Queue tools: persistent, deduplicated, surfaced in every state brief.

The agent's biggest failure mode was re-extracting the same listing page over
and over because the queue lived in stdout. These tools store the queue in
the same `Memory` instance, so it survives across turns and is rendered in
the state brief automatically.

Schema:
  memory["queue"]     : list[dict]  — each item must have an `id` field; other
                                       fields are free-form (detail_url, title, ...)
  memory["processed"] : list[str]   — ids that have been successfully appended
                                       to the dataset
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


def _get_queue(state: "AgentState") -> list[dict]:
    q = state.memory.get("queue", [])
    return q if isinstance(q, list) else []


def _get_processed(state: "AgentState") -> list[str]:
    p = state.memory.get("processed", [])
    return [str(x) for x in p] if isinstance(p, list) else []


class QueueAddTool(Tool):
    name = "queue_add"
    description = (
        "Append items to the persistent work queue stored in memory under "
        "the key 'queue'. Each item MUST be a dict with at least an 'id' "
        "field; any other fields are free-form (detail_url, title, date, ...). "
        "Items whose `id` is already in the queue or already processed are "
        "silently skipped. Call this immediately after enumerating decisions "
        "from a listing page — never keep the queue in stdout."
    )
    args_schema = {
        "items": {
            "type": "array",
            "description": "List of dicts; each MUST have an 'id' field.",
            "items": {"type": "object"},
        }
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        items = args.get("items")
        if not isinstance(items, list):
            return ToolResult(output="ERROR: 'items' must be a list of objects", error=True)
        queue = _get_queue(state)
        processed = set(_get_processed(state))
        existing = {str(i.get("id")) for i in queue if isinstance(i, dict) and i.get("id")}
        added = 0
        skipped = 0
        for it in items:
            if not isinstance(it, dict):
                skipped += 1
                continue
            iid = it.get("id")
            if iid is None or str(iid) in existing or str(iid) in processed:
                skipped += 1
                continue
            queue.append(it)
            existing.add(str(iid))
            added += 1
        state.memory.set("queue", queue)
        if not state.memory.get("processed"):
            state.memory.set("processed", [])
        return ToolResult(
            output=(
                f"queue_add: added={added} skipped={skipped} "
                f"queue_len_now={len(queue)} processed_len={len(processed)}"
            )
        )


class QueueNextTool(Tool):
    name = "queue_next"
    description = (
        "Return the next queued item that has not yet been marked processed. "
        "Output is JSON. If the queue is empty (or every item is processed), "
        "returns 'null' — that means either the work is done OR you need to "
        "fetch more listing pages and queue_add them. Call this at the start "
        "of every 'process one item' cycle."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        queue = _get_queue(state)
        processed = set(_get_processed(state))
        for item in queue:
            if isinstance(item, dict) and str(item.get("id")) not in processed:
                return ToolResult(output=json.dumps(item, ensure_ascii=False, indent=2))
        return ToolResult(output="null")


class QueueMarkDoneTool(Tool):
    name = "queue_mark_done"
    description = (
        "Mark a queued id as processed (so queue_next will skip it). Call "
        "this RIGHT AFTER dataset_append succeeds for an item — never before. "
        "Idempotent: calling twice is fine."
    )
    args_schema = {"id": {"type": "string", "description": "The id to mark done."}}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        iid = args.get("id")
        if iid is None:
            return ToolResult(output="ERROR: 'id' required", error=True)
        iid = str(iid)
        processed = _get_processed(state)
        if iid in processed:
            return ToolResult(output=f"already_done: {iid}")
        processed.append(iid)
        state.memory.set("processed", processed)
        return ToolResult(
            output=f"marked_done: {iid}  (processed_now={len(processed)})"
        )


class QueueStatusTool(Tool):
    name = "queue_status"
    description = (
        "Show queue length, processed count, remaining count, and a preview "
        "of the next 3 unprocessed items."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        queue = _get_queue(state)
        processed = set(_get_processed(state))
        remaining = [i for i in queue if isinstance(i, dict) and str(i.get("id")) not in processed]
        lines = [
            f"queue_len: {len(queue)}",
            f"processed: {len(processed)}",
            f"remaining: {len(remaining)}",
        ]
        if remaining:
            lines.append("next 3:")
            for r in remaining[:3]:
                s = json.dumps(r, ensure_ascii=False)
                lines.append("  " + (s if len(s) < 200 else s[:200] + "…"))
        return ToolResult(output="\n".join(lines))
