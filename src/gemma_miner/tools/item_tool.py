"""save_attachment: one-call extract + store under items/item_NNNN/.

Before this tool existed the agent needed three turns to: extract_text →
bash cp the binary → write_file the .txt. That's 3 × ~60s of LLM latency
per attachment. This tool does all three in one Python call, returning
relative paths the agent can drop straight into a dataset_append row.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult
from gemma_miner.tools.extract_text_tool import _extract_bytes, _sniff

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


def _slugify_id(s: str) -> str:
    # the folder name uses item_NNNN regardless of id; the id itself is just
    # for the agent's bookkeeping. But we still sanitise in case it's used elsewhere.
    return re.sub(r"[^\w\-]", "_", s).strip("_")


class SaveAttachmentTool(Tool):
    name = "save_attachment"
    description = (
        "End-to-end persistence for one attachment of one item. Given a "
        "binary at `source_path` (typically a cache path returned by "
        "http_get) and an item identifier, this tool: "
        "(1) creates `items/item_NNNN/` (auto-numbered by counting existing "
        "    folders unless `item_slot` is provided), "
        "(2) copies the binary to `attachment_NN.<ext>` (auto-numbered), "
        "(3) runs the universal text extractor, "
        "(4) writes the extracted text to `attachment_NN.txt`. "
        "Returns a JSON-ish summary with the relative paths AND the first "
        "1500 chars of the extracted text (useful sanity check) — use these "
        "values directly in `dataset_append`."
    )
    args_schema = {
        "id": {"type": "string", "description": "Item id (for logging only)."},
        "source_path": {
            "type": "string",
            "description": "Absolute path to the cached binary (e.g. a PDF in workdir/cache/).",
        },
        "item_slot": {
            "type": "integer",
            "description": "Optional fixed item folder number (1-based). If omitted, the next free slot is used.",
        },
        "attachment_name_hint": {
            "type": "string",
            "description": "Optional filename hint to preserve (extension is also derived from source).",
        },
        "text_preview_chars": {
            "type": "integer",
            "default": 1500,
            "description": "How much extracted text to show in the tool output.",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src = args.get("source_path")
        if not src:
            return ToolResult(output="ERROR: 'source_path' required", error=True)
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = Path(state.workdir) / src_path
        if not src_path.exists() or not src_path.is_file():
            return ToolResult(output=f"ERROR: source not found: {src_path}", error=True)

        workdir = Path(state.workdir)
        items_root = workdir / "items"
        items_root.mkdir(parents=True, exist_ok=True)

        # Decide on item slot.
        slot = args.get("item_slot")
        if slot is None:
            existing = sorted([p.name for p in items_root.glob("item_*") if p.is_dir()])
            n = 1
            for name in existing:
                m = re.match(r"item_(\d+)", name)
                if m:
                    n = max(n, int(m.group(1)))
            slot = n + 1 if existing else 1
        item_dir = items_root / f"item_{int(slot):04d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        # Decide on attachment slot.
        existing_att = sorted(item_dir.glob("attachment_*"))
        used = set()
        for p in existing_att:
            m = re.match(r"attachment_(\d+)\.", p.name)
            if m:
                used.add(int(m.group(1)))
        att_n = 1
        while att_n in used:
            att_n += 1

        # Pick extension: keep source's suffix, fall back to sniff.
        ext = src_path.suffix.lower()
        if not ext:
            data_head = src_path.read_bytes()[:1024]
            ext = _sniff(data_head) or ".bin"

        dest_bin = item_dir / f"attachment_{att_n:02d}{ext}"
        shutil.copy2(src_path, dest_bin)

        # Extract text.
        data = src_path.read_bytes()
        text, meta = _extract_bytes(data, src_path.name)
        dest_txt = item_dir / f"attachment_{att_n:02d}.txt"
        dest_txt.write_text(text, encoding="utf-8")

        rel_bin = dest_bin.relative_to(workdir).as_posix()
        rel_txt = dest_txt.relative_to(workdir).as_posix()

        preview_chars = int(args.get("text_preview_chars") or 1500)
        preview = text[:preview_chars]
        if len(text) > preview_chars:
            preview += f"\n... [truncated, total {len(text)} chars]"

        out = (
            f"item_id: {args.get('id', '(unset)')}\n"
            f"item_slot: {slot}\n"
            f"item_dir: {item_dir.relative_to(workdir).as_posix()}\n"
            f"attachment_path: {rel_bin}\n"
            f"text_path: {rel_txt}\n"
            f"text_chars: {len(text)}\n"
            f"extractor_meta: {meta}\n"
            "--- text preview ---\n"
            f"{preview}\n"
            "--- use these in dataset_append ---\n"
            f"pdf_path={rel_bin}\n"
            f"text_path={rel_txt}\n"
        )
        return ToolResult(
            output=out,
            artifact={
                "item_slot": slot,
                "attachment_path": rel_bin,
                "text_path": rel_txt,
                "text_chars": len(text),
            },
        )
