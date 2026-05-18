"""Per-variable coverage / type statistics over the silver (extracted) dataset.

Used by `dataset_export` and `dataset_validate` to produce a human-readable
quality report alongside the parquet output.
"""

from __future__ import annotations

from typing import Any

from gemma_miner.codebook import Codebook


def codebook_stats(rows: list[dict], cb: Codebook) -> dict[str, Any]:
    """Compute per-variable coverage and basic stats over ``rows`` for the
    variables in ``cb``. Returns a dict suitable for JSON serialisation."""
    n = len(rows)
    out: dict[str, Any] = {
        "n_rows": n,
        "codebook": cb.name,
        "n_variables": len(cb.variables),
        "variables": {},
    }
    for var in cb.variables:
        present = 0
        non_null = 0
        numeric_values: list[float] = []
        bool_values: list[bool] = []
        distinct: set = set()
        for r in rows:
            if var.name in r:
                present += 1
                v = r.get(var.name)
                if v is None or v == "":
                    continue
                non_null += 1
                if isinstance(v, bool):
                    bool_values.append(v)
                elif isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                try:
                    if isinstance(v, (str, int, float, bool)):
                        distinct.add(v)
                except TypeError:
                    pass
        info: dict[str, Any] = {
            "type": var.type,
            "present_pct": round(100 * present / n, 1) if n else 0.0,
            "non_null_pct": round(100 * non_null / n, 1) if n else 0.0,
            "n_distinct": len(distinct),
        }
        if numeric_values:
            info["min"] = min(numeric_values)
            info["max"] = max(numeric_values)
            info["mean"] = round(sum(numeric_values) / len(numeric_values), 4)
        if bool_values:
            info["pct_true"] = round(100 * sum(bool_values) / len(bool_values), 1)
        out["variables"][var.name] = info
    # Aggregate: fraction of vars that are numeric or boolean.
    types = [v.type for v in cb.variables]
    numeric_or_bool = sum(1 for t in types if t in ("integer", "number", "boolean"))
    out["numeric_or_boolean_ratio"] = (
        round(numeric_or_bool / max(1, len(types)), 3)
    )
    return out


def format_stats(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"dataset: {stats.get('codebook', '?')}")
    lines.append(f"rows: {stats.get('n_rows', 0)}")
    lines.append(f"variables: {stats.get('n_variables', 0)}")
    lines.append(
        f"numeric_or_boolean_ratio: {stats.get('numeric_or_boolean_ratio', 0):.0%}"
    )
    lines.append("")
    lines.append("per-variable coverage:")
    variables = stats.get("variables") or {}
    if not variables:
        lines.append("  (none)")
    else:
        name_w = max(len(n) for n in variables) + 2
        for name, info in variables.items():
            row = (
                f"  {name.ljust(name_w)}  type={info.get('type', '?'):<8}  "
                f"non_null={info.get('non_null_pct', 0):>5.1f}%  "
                f"distinct={info.get('n_distinct', 0)}"
            )
            if "pct_true" in info:
                row += f"  true={info['pct_true']:.0f}%"
            if "mean" in info:
                row += (
                    f"  min={info['min']:g}  mean={info['mean']:g}  "
                    f"max={info['max']:g}"
                )
            lines.append(row)
    return "\n".join(lines)
