"""Dataset validation, Parquet export, and Hugging Face Hub push.

These are the terminal tools of the pipeline. After extract_items has filled
every row's codebook columns, the agent calls:

  1. `dataset_validate`  — per-variable stats; spot dead/redundant columns.
  2. `dataset_export`    — write parquet + jsonl + auto-generated codebook.md.
  3. `hf_push`           — push the parquet to a public HuggingFace dataset repo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.export import write_codebook_md, write_parquet
from gemma42.stats import codebook_stats, format_stats
from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.codebook_tool import _load_codebook_from_state

if TYPE_CHECKING:
    from gemma42.state import AgentState


class DatasetValidateTool(Tool):
    name = "dataset_validate"
    description = (
        "Compute per-variable statistics over the whole dataset using the "
        "current codebook. Output includes coverage, summary stats per type "
        "(min/max/mean for numbers, distribution for enums, range for dates), "
        "and a list of data-quality issues (low coverage, near-constant "
        "booleans, single-value enums). Call this AFTER extract_items and "
        "BEFORE dataset_export."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook saved", error=True)
        rows = state.dataset.rows()
        s = codebook_stats(rows, cb)
        return ToolResult(output=format_stats(s), artifact=s)


class DatasetExportTool(Tool):
    name = "dataset_export"
    description = (
        "Write the dataset to Parquet (typed via the codebook), plus the "
        "JSONL (already exists at <workdir>/dataset.jsonl), plus an auto-"
        "generated codebook.md (dataset card) with all variables documented "
        "and their coverage statistics. The Parquet file is ready for pandas, "
        "polars, DuckDB, R/Arrow, and `datasets.Dataset.from_parquet(...)`."
    )
    args_schema = {
        "out_dir": {
            "type": "string",
            "description": "Where to write the exported files. Defaults to <workdir>/export.",
        },
        "source_url": {
            "type": "string",
            "description": "Optional original source URL to put in the dataset card.",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook saved", error=True)
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: dataset is empty", error=True)

        out_dir = Path(args.get("out_dir") or (Path(state.workdir) / "export"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Carry over common metadata fields that aren't in the codebook.
        var_names = {v.name for v in cb.variables}
        extra_meta = sorted({
            k for r in rows[:50] for k in r.keys()
            if k not in var_names and not k.startswith("_")
        })

        # Parquet
        try:
            parquet_path = write_parquet(
                rows,
                cb,
                out_dir / f"{cb.name}.parquet",
                extra_metadata_fields=tuple(extra_meta),
            )
        except RuntimeError as e:  # pyarrow missing
            parquet_path = None
            parquet_err = str(e)
        else:
            parquet_err = None

        # Codebook MD
        stats = codebook_stats(rows, cb)
        md_path = write_codebook_md(
            cb,
            stats,
            out_dir / "codebook.md",
            title=cb.name,
            source_url=args.get("source_url"),
        )

        # Codebook JSON (canonical)
        json_path = out_dir / "codebook.json"
        cb.save(json_path)

        # JSONL copy
        import shutil

        jsonl_src = Path(state.dataset.path)
        jsonl_dst = out_dir / f"{cb.name}.jsonl"
        shutil.copy2(jsonl_src, jsonl_dst)

        out = [
            f"export → {out_dir}",
            f"  jsonl:    {jsonl_dst}  ({len(rows)} rows)",
            f"  codebook: {json_path}",
            f"  card:     {md_path}",
        ]
        if parquet_path:
            out.append(f"  parquet:  {parquet_path}")
        else:
            out.append(f"  parquet:  SKIPPED ({parquet_err})")
        return ToolResult(
            output="\n".join(out),
            artifact={"out_dir": str(out_dir), "n_rows": len(rows)},
        )


class HFPushTool(Tool):
    name = "hf_push"
    description = (
        "Push the exported dataset to a Hugging Face Hub dataset repository. "
        "Requires HF_TOKEN in the environment (or HUGGINGFACE_HUB_TOKEN) AND "
        "the `datasets` library installed (`pip install gemma42[hf]`). The "
        "tool pushes the Parquet file + the codebook.md (as README.md) so "
        "anyone can `load_dataset(<repo_id>)` and start doing statistics."
    )
    args_schema = {
        "repo_id": {
            "type": "string",
            "description": "Target repo, e.g. 'yourname/cnil-sanctions'.",
        },
        "private": {"type": "boolean", "default": True},
        "export_dir": {
            "type": "string",
            "description": "Directory containing the exported parquet + codebook.md. Defaults to <workdir>/export.",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        repo_id = args.get("repo_id")
        if not repo_id:
            return ToolResult(output="ERROR: 'repo_id' required", error=True)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            return ToolResult(
                output="ERROR: HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is not set in the environment",
                error=True,
            )
        try:
            from datasets import Dataset  # type: ignore
            from huggingface_hub import HfApi  # type: ignore
        except ImportError:
            return ToolResult(
                output="ERROR: install extras: pip install 'gemma42[hf]'",
                error=True,
            )
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook saved", error=True)
        export_dir = Path(args.get("export_dir") or (Path(state.workdir) / "export"))
        parquet = export_dir / f"{cb.name}.parquet"
        if not parquet.exists():
            return ToolResult(
                output=f"ERROR: parquet not found at {parquet}. Run dataset_export first.",
                error=True,
            )
        ds = Dataset.from_parquet(str(parquet))
        ds.push_to_hub(repo_id, private=bool(args.get("private", True)), token=token)
        # Also upload the dataset card.
        api = HfApi(token=token)
        readme = export_dir / "codebook.md"
        if readme.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(readme),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            except Exception as e:  # noqa: BLE001
                return ToolResult(
                    output=f"pushed {len(ds)} rows to {repo_id}, but README upload failed: {e}",
                )
        return ToolResult(output=f"pushed {len(ds)} rows + README → https://huggingface.co/datasets/{repo_id}")
