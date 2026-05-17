"""Per-variable statistics for codebook quality assessment.

Given a list of rows and a codebook, produce a summary that helps the agent
decide which variables to keep, drop, or refine. The output is human-readable
AND machine-readable (drives the dataset_validate tool + the auto-generated
codebook.md).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from gemma42.codebook import Codebook, VariableSpec


def _non_null(values: list[Any]) -> list[Any]:
    return [v for v in values if v is not None and v != ""]


def _variable_stats(values: list[Any], var: VariableSpec) -> dict:
    nn = _non_null(values)
    total = len(values)
    n = len(nn)
    cov = n / total if total else 0.0
    out: dict = {
        "name": var.name,
        "type": var.type,
        "n_total": total,
        "n_non_null": n,
        "coverage": cov,
    }
    if var.type == "boolean":
        n_true = sum(1 for v in nn if bool(v))
        out["pct_true"] = n_true / n if n else None
    elif var.type in ("integer", "float"):
        nums = [v for v in nn if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums:
            out["min"] = min(nums)
            out["max"] = max(nums)
            out["mean"] = sum(nums) / len(nums)
            if len(nums) > 1:
                out["stdev"] = statistics.stdev(nums)
            out["median"] = statistics.median(nums)
    elif var.type == "enum":
        counts: dict[str, int] = {}
        for v in nn:
            counts[str(v)] = counts.get(str(v), 0) + 1
        out["distribution"] = dict(
            sorted(counts.items(), key=lambda x: -x[1])
        )
    elif var.type == "string":
        lens = [len(v) for v in nn if isinstance(v, str)]
        if lens:
            out["min_len"] = min(lens)
            out["max_len"] = max(lens)
            out["mean_len"] = sum(lens) / len(lens)
        unique = len({str(v) for v in nn})
        out["n_unique"] = unique
    elif var.type == "date":
        dates = sorted(v for v in nn if isinstance(v, str) and v[:4].isdigit())
        if dates:
            out["min_date"] = dates[0]
            out["max_date"] = dates[-1]
    elif var.type == "array":
        sizes = [len(v) for v in nn if isinstance(v, list)]
        if sizes:
            out["min_size"] = min(sizes)
            out["max_size"] = max(sizes)
            out["mean_size"] = sum(sizes) / len(sizes)
    return out


def codebook_stats(rows: list[dict], codebook: Codebook) -> dict:
    """Compute stats for every variable + global issues."""
    by_var = []
    issues: list[str] = []
    for var in codebook.variables:
        values = [r.get(var.name) for r in rows]
        s = _variable_stats(values, var)
        by_var.append(s)
        if s["coverage"] < 0.1 and len(rows) >= 5:
            issues.append(
                f"  {var.name}: only {s['coverage']:.1%} coverage — consider dropping or rewriting"
            )
        elif var.type == "boolean" and s["coverage"] >= 0.5:
            pt = s.get("pct_true")
            if pt is not None and (pt < 0.02 or pt > 0.98):
                issues.append(
                    f"  {var.name}: boolean is {pt:.1%} true — near-constant, low info value"
                )
        elif var.type == "enum":
            dist = s.get("distribution", {})
            if dist and len(dist) == 1:
                issues.append(
                    f"  {var.name}: only one observed value {next(iter(dist))!r}"
                )
        elif var.type == "string":
            if s.get("n_unique", 0) == 1 and s["coverage"] >= 0.5:
                issues.append(f"  {var.name}: string is constant — drop it")
    return {
        "n_rows": len(rows),
        "n_variables": len(codebook.variables),
        "type_breakdown": codebook.type_breakdown(),
        "numeric_or_boolean_ratio": codebook.numeric_or_boolean_ratio(),
        "variables": by_var,
        "issues": issues,
    }


def format_stats(s: dict) -> str:
    lines = [
        f"dataset: {s['n_rows']} rows × {s['n_variables']} variables",
        f"type breakdown: {s['type_breakdown']}",
        f"numeric_or_boolean_ratio: {s['numeric_or_boolean_ratio']:.0%}",
        "",
    ]
    for v in s["variables"]:
        head = f"  {v['name']:<25}  type={v['type']:<8}  coverage={v['coverage']:.1%}"
        extras: list[str] = []
        if v["type"] in ("integer", "float") and "mean" in v:
            extras.append(f"min={v['min']} max={v['max']} mean={v['mean']:.2f}")
        elif v["type"] == "boolean" and "pct_true" in v:
            extras.append(f"true={v['pct_true']:.0%}" if v["pct_true"] is not None else "")
        elif v["type"] == "enum" and "distribution" in v:
            d = v["distribution"]
            top3 = list(d.items())[:3]
            extras.append("top: " + ", ".join(f"{k}={n}" for k, n in top3))
        elif v["type"] == "string" and "mean_len" in v:
            extras.append(f"avg_len={v['mean_len']:.0f} n_unique={v.get('n_unique','?')}")
        elif v["type"] == "date" and "min_date" in v:
            extras.append(f"{v['min_date']} → {v['max_date']}")
        if extras:
            head += "  " + " ".join(extras)
        lines.append(head)
    if s["issues"]:
        lines.append("")
        lines.append("issues:")
        lines.extend(s["issues"])
    return "\n".join(lines)
