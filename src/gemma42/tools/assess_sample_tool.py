"""assess_sample — quality gate between PILOT and SCALE steps.

The agent calls this after a small harvest or a small extraction batch to
decide: SCALE UP or FIX FIRST? The tool inspects the current rows (bronze,
silver, or joined) and reports:

  • per-field coverage (% non-null)
  • per-field cardinality (n unique values)
  • a sample of 3 rows so the model can eyeball
  • warnings about suspect patterns:
      - field is null in >50% of sampled rows
      - field has the SAME value in every row (likely broken regex)
      - field has values that don't match its declared type
      - two fields hold identical values (column collision)
      - row count is far below the planned target
  • a verdict: SCALE_OK | FIX_FIRST | INCONCLUSIVE
  • specific next-step recommendations

This is the "pilot-then-scale" gate. The agent SHOULD call it:
  - after the first scrape_paginated call (and before scaling beyond 1 page)
  - after extract_items on a small batch (and before extracting everything)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _classify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _summarize_field(name: str, values: list) -> dict:
    n = len(values)
    nonnull = [v for v in values if v not in (None, "")]
    coverage = len(nonnull) / n if n else 0
    types = Counter(_classify(v) for v in values)
    types_summary = ", ".join(f"{k}={v}" for k, v in types.most_common())
    unique_count = len({str(v) for v in nonnull})
    most_common: list = []
    if nonnull:
        c = Counter(str(v) for v in nonnull).most_common(3)
        most_common = [(v, cnt) for v, cnt in c]
    return {
        "name":      name,
        "coverage":  coverage,
        "types":     types_summary,
        "unique":    unique_count,
        "n_nonnull": len(nonnull),
        "most_common": most_common,
    }


def _detect_collisions(rows: list[dict], field_names: list[str]) -> list[tuple[str, str]]:
    """Pairs of fields that hold identical values on >=80% of rows where both are populated."""
    pairs: list[tuple[str, str]] = []
    if len(rows) < 3:
        return pairs
    for i in range(len(field_names)):
        for j in range(i + 1, len(field_names)):
            a, b = field_names[i], field_names[j]
            both = 0
            same = 0
            for r in rows:
                va, vb = r.get(a), r.get(b)
                if va in (None, "") or vb in (None, ""):
                    continue
                both += 1
                if va == vb:
                    same += 1
            if both >= 3 and same / both >= 0.8:
                pairs.append((a, b))
    return pairs


class AssessSampleTool(Tool):
    name = "assess_sample"
    description = (
        "Quality gate. Inspect what you've harvested or extracted so far and "
        "decide: SCALE UP or FIX FIRST? Returns coverage / cardinality / "
        "collisions / type mismatches per field, three sample rows, and an "
        "explicit verdict (SCALE_OK | FIX_FIRST | INCONCLUSIVE) with next-step "
        "advice.\n\n"
        "Args:\n"
        "  layer       : 'bronze' (raw harvest, default) | 'silver' (typed "
        "extracted vars) | 'joined' (bronze ⊕ silver merged by id)\n"
        "  limit       : how many rows to sample (default: all)\n"
        "  expected    : optional list of field names you expect populated. "
        "When given, the tool flags any expected field with coverage < 50% "
        "as a problem.\n\n"
        "When to call:\n"
        "  • After your first scrape_paginated, before scaling beyond page 1.\n"
        "  • After extract_items on a 3-5 item pilot, before extracting all rows.\n"
        "  • Any time the brief says rows look weird and you want a second opinion."
    )
    args_schema = {
        "layer":    {"type": "string"},
        "limit":    {"type": "integer"},
        "expected": {"type": "array"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        layer = (args.get("layer") or "bronze").strip().lower()
        if layer == "silver":
            rows = state.extracted_dataset().rows()
        elif layer in ("joined", "merge", "merged"):
            view = state._joined_view() if state._extracted_dataset else state.dataset
            rows = view.rows()
        else:
            rows = state.dataset.rows()
            layer = "bronze"

        if not rows:
            return ToolResult(
                output=f"assess_sample (layer={layer}): empty dataset — nothing to assess.",
                error=False,
            )

        limit = args.get("limit")
        if isinstance(limit, int) and limit > 0 and len(rows) > limit:
            rows = rows[:limit]

        # Collect field stats.
        all_fields: dict[str, list] = {}
        for r in rows:
            for k, v in r.items():
                all_fields.setdefault(k, []).append(v)
        # Stable ordering — by first appearance.
        field_order = list(all_fields.keys())
        # Drop trivially boring fields from collision detection.
        meaningful = [
            f for f in field_order
            if f != "id" and not str(f).startswith("_")
            and "url" not in f.lower() and "path" not in f.lower()
        ]
        summaries = [_summarize_field(f, all_fields[f]) for f in field_order]

        # Warnings
        warnings: list[str] = []
        expected: list = args.get("expected") or []

        low_cov = [s for s in summaries if s["coverage"] < 0.5 and s["name"] != "id"]
        if low_cov:
            for s in low_cov[:6]:
                warnings.append(
                    f"low coverage: `{s['name']}` is populated in only "
                    f"{int(s['coverage']*100)}% of rows ({s['n_nonnull']}/{len(rows)})"
                )
        for ef in expected:
            s = next((x for x in summaries if x["name"] == ef), None)
            if s is None:
                warnings.append(f"expected field MISSING entirely: `{ef}`")
            elif s["coverage"] < 0.5:
                warnings.append(
                    f"expected field `{ef}` has low coverage "
                    f"{int(s['coverage']*100)}%"
                )

        # All-rows-same-value (likely broken regex / hardcoded value)
        for s in summaries:
            if s["unique"] == 1 and s["n_nonnull"] >= 3 and s["name"] != "id":
                warnings.append(
                    f"`{s['name']}` has the SAME value in every row "
                    f"({s['most_common'][0][0]!r}) — likely a broken regex."
                )

        # Type drift
        for s in summaries:
            t = s["types"]
            # e.g. "int=5, string=2" → mixed
            distinct_types = [tok.split("=")[0] for tok in t.split(", ")
                              if not tok.startswith("null=")]
            if len(set(distinct_types)) > 1:
                warnings.append(
                    f"`{s['name']}` has MIXED types ({t}) across rows — extraction "
                    "may need a coercion or stricter prompt."
                )

        # Cross-field collisions
        collisions = _detect_collisions(rows, meaningful)
        for a, b in collisions[:4]:
            warnings.append(
                f"`{a}` and `{b}` hold identical values on most rows — "
                "the field regexes may be capturing the same column."
            )

        # Plan-vs-actual sanity
        plan = state.memory.get("plan") or {}
        target = plan.get("target_rows")
        if isinstance(target, int) and len(rows) < target * 0.1 and len(rows) < 20:
            warnings.append(
                f"only {len(rows)} rows so far vs plan.target_rows={target}. "
                "Either you're piloting (good — proceed to scale) OR the harvest "
                "is stalled. Check queue_status."
            )

        # Verdict
        verdict = "SCALE_OK"
        if any("MISSING" in w or "low coverage" in w or "SAME value" in w or
               "identical values" in w for w in warnings):
            verdict = "FIX_FIRST"
        if len(rows) < 3:
            verdict = "INCONCLUSIVE"

        # Build output
        lines = [
            f"assess_sample (layer={layer}, n={len(rows)})",
            f"verdict: {verdict}",
            "",
            f"fields ({len(summaries)}):",
        ]
        for s in summaries[:30]:
            cov_pct = int(s["coverage"] * 100)
            mc_str = ""
            if s["most_common"]:
                mc_str = "  · " + ", ".join(
                    f"{v!r}×{cnt}" for v, cnt in s["most_common"][:2]
                )
            lines.append(
                f"  {s['name']:<28}  cov={cov_pct:>3d}%  "
                f"unique={s['unique']:<4}  types=[{s['types']}]{mc_str}"
            )
        if len(summaries) > 30:
            lines.append(f"  … {len(summaries) - 30} more fields")
        if warnings:
            lines.append("")
            lines.append("⚠ warnings:")
            for w in warnings[:10]:
                lines.append(f"  - {w}")
            if len(warnings) > 10:
                lines.append(f"  … {len(warnings) - 10} more")
        lines.append("")
        lines.append("sample rows (first 3):")
        for r in rows[:3]:
            lines.append("  " + json.dumps(r, ensure_ascii=False)[:400])

        # Action advice keyed off the verdict.
        lines.append("")
        if verdict == "SCALE_OK":
            lines.append(
                "→ SCALE_OK. Coverage and shape look healthy. Proceed to the "
                "next step (scale the harvest, run extract_items on the full "
                "dataset, or move to the next phase)."
            )
        elif verdict == "FIX_FIRST":
            lines.append(
                "→ FIX_FIRST. Resolve the warnings above BEFORE scaling. "
                "Common moves:\n"
                "  • Broken regex / collision → re-run extractor_define with "
                "    column-specific anchors, OR use llm_scrape / python.\n"
                "  • Low coverage on an expected field → that field probably "
                "    lives on the detail page or in an attachment — switch to "
                "    process_queue(mode='text' or 'multi_asset').\n"
                "  • Mixed types → tighten the codebook variable's type or its "
                "    extraction_hint."
            )
        else:
            lines.append(
                "→ INCONCLUSIVE. Sample too small to judge. Harvest a few more "
                "items, then call assess_sample again."
            )

        return ToolResult(
            output="\n".join(lines),
            artifact={
                "verdict": verdict, "n_rows": len(rows),
                "warnings": warnings, "layer": layer,
            },
        )
