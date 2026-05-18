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
import re
from pathlib import Path
from typing import TYPE_CHECKING

from gemma_miner.export import write_codebook_md, write_parquet
from gemma_miner.stats import codebook_stats, format_stats
from gemma_miner.tools.base import Tool, ToolResult
from gemma_miner.tools.codebook_tool import _load_codebook_from_state

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


def _infer_dataset_stats(rows: list[dict]) -> dict:
    """Per-column inferred stats when no codebook is defined."""
    import statistics

    n = len(rows)
    cols: dict[str, list] = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith("_"):
                continue
            cols.setdefault(k, []).append(v)
    out: list[dict] = []
    for name, values in cols.items():
        nn = [v for v in values if v not in (None, "")]
        info = {
            "name": name,
            "coverage": len(nn) / n if n else 0.0,
            "n_non_null": len(nn),
        }
        # Infer type from observations
        if all(isinstance(v, bool) for v in nn) and nn:
            info["type"] = "boolean"
            info["pct_true"] = sum(1 for v in nn if v) / len(nn)
        elif nn and all(isinstance(v, int) and not isinstance(v, bool) for v in nn):
            info["type"] = "integer"
            info["min"] = min(nn); info["max"] = max(nn)
            info["mean"] = sum(nn) / len(nn)
            if len(nn) > 1:
                info["stdev"] = statistics.stdev(nn)
        elif nn and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nn):
            info["type"] = "float"
            info["min"] = min(nn); info["max"] = max(nn)
            info["mean"] = sum(nn) / len(nn)
        elif nn and all(isinstance(v, str) for v in nn):
            info["type"] = "string"
            info["n_unique"] = len({str(v) for v in nn})
            lens = [len(v) for v in nn]
            info["mean_len"] = sum(lens) / len(lens) if lens else 0
        else:
            info["type"] = "mixed"
        out.append(info)
    return {"n_rows": n, "n_columns": len(cols), "variables": out,
             "issues": []}


def _synthesise_codebook_from_rows(rows: list[dict], *,
                                    workdir: str | Path) -> "Codebook":
    """Build a Codebook with VariableSpec for each observed column.

    Used when the user wanted a simple table dump without designing a
    research-grade codebook. Types are inferred from the data.
    """
    from gemma_miner.codebook import Codebook, VariableSpec

    cols: dict[str, list] = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith("_"):
                continue
            cols.setdefault(k, []).append(v)

    variables: list[VariableSpec] = []
    for name, values in cols.items():
        nn = [v for v in values if v not in (None, "")]
        if not nn:
            t = "string"
        elif all(isinstance(v, bool) for v in nn):
            t = "boolean"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in nn):
            t = "integer"
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nn):
            t = "float"
        else:
            t = "string"
        variables.append(VariableSpec(
            name=name, type=t,
            description=f"Auto-inferred column from observed data.",
        ))

    cb_name = Path(workdir).name or "dataset"
    cb_name = re.sub(r"[^\w]+", "_", cb_name).strip("_") or "dataset"
    return Codebook(
        name=cb_name,
        description="Auto-synthesised codebook (no LLM-designed schema requested).",
        variables=variables,
    )


def _format_inferred_stats(s: dict) -> str:
    lines = [
        f"dataset: {s['n_rows']} rows × {s['n_columns']} columns",
        "",
    ]
    for v in s["variables"]:
        head = f"  {v['name']:<25}  type={v['type']:<8}  coverage={v['coverage']:.0%}"
        extras: list[str] = []
        if v["type"] in ("integer", "float") and "mean" in v:
            extras.append(f"min={v['min']} max={v['max']} mean={v['mean']:.2f}")
        elif v["type"] == "boolean" and "pct_true" in v:
            extras.append(f"true={v['pct_true']:.0%}")
        elif v["type"] == "string" and "mean_len" in v:
            extras.append(f"unique={v.get('n_unique', '?')}  avg_len={v['mean_len']:.0f}")
        if extras:
            head += "  " + " ".join(extras)
        lines.append(head)
    return "\n".join(lines)


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
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: dataset is empty", error=True)
        if cb is not None:
            s = codebook_stats(rows, cb)
            return ToolResult(output=format_stats(s), artifact=s)
        # No codebook → infer per-column stats from the observed rows.
        s = _infer_dataset_stats(rows)
        return ToolResult(
            output=("(no codebook — using observed column types)\n\n"
                     + _format_inferred_stats(s)),
            artifact=s,
        )


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
        raw_rows = state.dataset.rows()
        if not raw_rows:
            return ToolResult(output="ERROR: dataset is empty", error=True)

        # Pull silver (typed variables from Gemma extraction) if it exists.
        # Bronze + silver are kept in separate files at runtime so the counts
        # are clean; we JOIN them here at export time by `id`.
        extracted_path = Path(state.workdir) / "extracted.jsonl"
        extracted_by_id: dict[str, dict] = {}
        if extracted_path.exists() and extracted_path.stat().st_size > 0:
            for line in extracted_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    import json as _json
                    r = _json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(r, dict) and r.get("id") is not None:
                    extracted_by_id[str(r["id"])] = r
        # Build a merged view for the main parquet: raw fields ⊕ typed fields.
        rows: list[dict] = []
        for r in raw_rows:
            merged = dict(r)
            silver = extracted_by_id.get(str(r.get("id")), {})
            for k, v in silver.items():
                if k != "id":
                    merged[k] = v
            rows.append(merged)

        out_dir = Path(args.get("out_dir") or (Path(state.workdir) / "export"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # No-codebook fallback: synthesise one from the observed columns so
        # the parquet writer + card still work.
        synthesised_codebook = False
        if cb is None:
            cb = _synthesise_codebook_from_rows(rows, workdir=state.workdir)
            synthesised_codebook = True

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

        # JSONL copies — both bronze (raw) and the merged (raw ⊕ typed).
        import shutil
        jsonl_raw_src = Path(state.dataset.path)
        jsonl_raw_dst = out_dir / f"{cb.name}_raw.jsonl"
        shutil.copy2(jsonl_raw_src, jsonl_raw_dst)
        # Merged JSONL (what most consumers want).
        jsonl_dst = out_dir / f"{cb.name}.jsonl"
        with jsonl_dst.open("w", encoding="utf-8") as f:
            for r in rows:
                import json as _json
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        # Silver-only JSONL (id + typed cols), if extraction happened.
        typed_jsonl_dst = None
        if extracted_by_id:
            typed_jsonl_dst = out_dir / f"{cb.name}_typed.jsonl"
            shutil.copy2(extracted_path, typed_jsonl_dst)

        # Optional typed-only parquet — id + codebook columns, no text bloat.
        typed_parquet_path = None
        if extracted_by_id:
            typed_rows: list[dict] = []
            keep = {"id"} | {v.name for v in cb.variables}
            for r in rows:
                typed_rows.append({k: v for k, v in r.items() if k in keep})
            try:
                typed_parquet_path = write_parquet(
                    typed_rows,
                    cb,
                    out_dir / f"{cb.name}_typed.parquet",
                    extra_metadata_fields=("id",),
                )
            except Exception:  # noqa: BLE001
                typed_parquet_path = None

        out = [
            f"export → {out_dir}",
            f"  jsonl:        {jsonl_dst}  ({len(rows)} rows, raw ⊕ typed)",
            f"  jsonl_raw:    {jsonl_raw_dst}  (bronze, raw harvest only)",
        ]
        if typed_jsonl_dst:
            out.append(f"  jsonl_typed:  {typed_jsonl_dst}  ({len(extracted_by_id)} extracted)")
        out.append(
            f"  codebook:     {json_path}"
            + ('  (auto-synthesised from columns)' if synthesised_codebook else '')
        )
        out.append(f"  card:         {md_path}")
        if parquet_path:
            out.append(f"  parquet:      {parquet_path}")
        else:
            out.append(f"  parquet:      SKIPPED ({parquet_err})")
        if typed_parquet_path:
            out.append(f"  parquet_typed: {typed_parquet_path}  (id + codebook columns only)")
        return ToolResult(
            output="\n".join(out),
            artifact={
                "out_dir": str(out_dir),
                "n_rows": len(rows),
                "n_extracted": len(extracted_by_id),
            },
        )


class HFPushTool(Tool):
    name = "hf_push"
    description = (
        "Push the exported dataset to a Hugging Face Hub dataset repository. "
        "Requires HF_TOKEN in the environment (or HUGGINGFACE_HUB_TOKEN) AND "
        "the `datasets` library installed (`pip install gemma-miner[hf]`). The "
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
                output="ERROR: install extras: pip install 'gemma-miner[hf]'",
                error=True,
            )
        private = bool(args.get("private", True))
        export_dir = Path(args.get("export_dir") or (Path(state.workdir) / "export"))

        # Strategy: prefer the silver parquet (codebook-typed dataset). Fall
        # back to the bronze parquet, then to dataset.jsonl directly. This
        # way `hf_push` works whether the run produced a codebook or not.
        cb = _load_codebook_from_state(state)
        parquet: Path | None = None
        source_kind = ""
        if cb is not None:
            cand = export_dir / f"{cb.name}.parquet"
            if cand.exists():
                parquet = cand
                source_kind = "silver"
        if parquet is None and export_dir.exists():
            pqs = sorted(export_dir.glob("*.parquet"))
            if pqs:
                parquet = pqs[0]
                source_kind = "parquet"
        ds = None
        if parquet is not None:
            ds = Dataset.from_parquet(str(parquet))
        else:
            jsonl = Path(state.dataset.path)
            if not jsonl.exists():
                return ToolResult(
                    output=f"ERROR: nothing to push — no parquet under {export_dir} and no {jsonl}",
                    error=True,
                )
            ds = Dataset.from_json(str(jsonl))
            source_kind = "bronze-jsonl"

        ds.push_to_hub(repo_id, private=private, token=token)

        # Upload the dataset card if present.
        api = HfApi(token=token)
        readme = export_dir / "codebook.md"
        readme_status = "no README uploaded"
        if readme.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(readme),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                readme_status = "README uploaded"
            except Exception as e:  # noqa: BLE001
                readme_status = f"README upload failed: {e}"
        return ToolResult(
            output=(
                f"pushed {len(ds)} rows (source={source_kind}) "
                f"→ https://huggingface.co/datasets/{repo_id}  ·  {readme_status}"
            )
        )
