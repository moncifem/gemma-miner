"""Provenance + reproducibility + diff tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.provenance import (
    LockManifest, build_manifest_from_state, diff_datasets, format_diff,
)
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


class ManifestWriteTool(Tool):
    name = "manifest_write"
    description = (
        "Write a `gemma42.lock` reproducibility manifest summarising the "
        "current run: goal, contracts, extractors, codebook, rules, "
        "model, fingerprints, sources. Anyone with this file (and the "
        "same source URLs accessible) can rebuild the dataset."
    )
    args_schema = {
        "llm_provider": {"type": "string", "default": "unknown"},
        "llm_model":    {"type": "string", "default": "unknown"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        # try to read the codebook
        cb_path = Path(state.workdir) / "codebook.json"
        cb_dict = None
        if cb_path.exists():
            try:
                cb_dict = json.loads(cb_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        manifest = build_manifest_from_state(
            state,
            llm_provider=args.get("llm_provider") or "unknown",
            llm_model=args.get("llm_model") or "unknown",
            codebook_dict=cb_dict,
        )
        out_path = Path(state.workdir) / "gemma42.lock"
        manifest.save(out_path)
        return ToolResult(
            output=(
                f"manifest written → {out_path}\n"
                f"lock_id: {manifest.hash()}\n"
                f"n_rows: {manifest.summary.get('n_rows')}\n"
                f"extractors: {list(manifest.extractors.keys())}\n"
                f"sources: {len(manifest.sources)}"
            ),
            artifact={"path": str(out_path), "lock_id": manifest.hash()},
        )


class DatasetDiffTool(Tool):
    name = "dataset_diff"
    description = (
        "Compare two JSONL datasets and report added/removed/changed rows. "
        "Pass `a` and `b` as paths (absolute or relative to workdir). "
        "Uses `id` as the row key by default; override with `key`."
    )
    args_schema = {
        "a":   {"type": "string"},
        "b":   {"type": "string"},
        "key": {"type": "string", "default": "id"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        a_arg = args.get("a") or ""
        b_arg = args.get("b") or ""
        if not a_arg or not b_arg:
            return ToolResult(output="ERROR: 'a' and 'b' required", error=True)

        def _load(p: str) -> list[dict]:
            path = Path(p)
            if not path.is_absolute():
                path = Path(state.workdir) / path
            if not path.exists():
                raise FileNotFoundError(path)
            out: list[dict] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        continue
            return out

        try:
            rows_a = _load(a_arg)
            rows_b = _load(b_arg)
        except FileNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        d = diff_datasets(rows_a, rows_b, key=args.get("key", "id"))
        return ToolResult(output=format_diff(d), artifact=d)
