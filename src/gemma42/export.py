"""Parquet + dataset-card (codebook.md) export.

Uses `pyarrow` if available (ships via the `[hf]`/`[parquet]` extras). The
schema is built directly from the codebook so every column has its proper
Arrow type and the file can be loaded by pandas, polars, R/Arrow, DuckDB,
or pushed to the Hugging Face Hub with full type fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gemma42.codebook import Codebook, VariableSpec


# ── Arrow schema builder ───────────────────────────────────────────────────


def _pa() -> Any:
    """Lazy import of pyarrow with a helpful error."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is not installed. Install with: pip install 'gemma42[hf]'  "
            "or pip install pyarrow"
        ) from e
    return __import__("pyarrow")


def _arrow_type(var: VariableSpec):
    pa = _pa()
    if var.type == "boolean":
        return pa.bool_()
    if var.type == "integer":
        return pa.int64()
    if var.type == "float":
        return pa.float64()
    if var.type == "date":
        # Store as Arrow date32 if values are ISO strings; null otherwise.
        return pa.date32()
    if var.type == "enum":
        # Dictionary-encoded string (a.k.a. Categorical).
        return pa.dictionary(pa.int32(), pa.string())
    if var.type == "array":
        # Best effort: list of strings unless item_schema indicates struct.
        if isinstance(var.item_schema, dict) and var.item_schema.get("type") == "object":
            props = var.item_schema.get("properties") or {}
            fields = []
            for k, prop in props.items():
                t = (prop or {}).get("type") if isinstance(prop, dict) else None
                if t == "boolean":
                    at = pa.bool_()
                elif t == "integer":
                    at = pa.int64()
                elif t == "number":
                    at = pa.float64()
                else:
                    at = pa.string()
                fields.append(pa.field(k, at))
            return pa.list_(pa.struct(fields))
        return pa.list_(pa.string())
    return pa.string()


def codebook_arrow_schema(codebook: Codebook, extra_metadata_fields: list[str] = ()):
    pa = _pa()
    fields = []
    for v in codebook.variables:
        fields.append(pa.field(v.name, _arrow_type(v)))
    # Carry-over fields from the harvest (id, title, urls, paths) live next to
    # the structured ones so the parquet is self-describing.
    for k in extra_metadata_fields:
        if any(f.name == k for f in fields):
            continue
        fields.append(pa.field(k, pa.string()))
    return pa.schema(fields)


# ── Parquet writer ─────────────────────────────────────────────────────────


def write_parquet(
    rows: list[dict],
    codebook: Codebook,
    out_path: Path,
    extra_metadata_fields: tuple[str, ...] = (),
) -> Path:
    """Write rows to a Parquet file using the codebook's Arrow schema.

    Rows that don't carry a codebook field get null for that column. Rows
    can include extra metadata fields (id, title, etc.) — they're added to
    the schema as strings via `extra_metadata_fields`.
    """
    pa = _pa()
    import pyarrow.parquet as pq

    schema = codebook_arrow_schema(codebook, extra_metadata_fields)
    columns: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        for f in schema:
            columns[f.name].append(_coerce_for_arrow(r.get(f.name), f.type))
    table = pa.table(columns, schema=schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    return out_path


def _coerce_for_arrow(value: Any, arrow_type: Any) -> Any:
    pa = _pa()
    if value is None:
        return None
    if arrow_type == pa.date32():
        if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
            from datetime import date

            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None
    if arrow_type == pa.bool_():
        return bool(value) if not isinstance(value, bool) else value
    if arrow_type == pa.int64():
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if arrow_type == pa.float64():
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if pa.types.is_list(arrow_type):
        if isinstance(value, list):
            return value
        return None
    if pa.types.is_dictionary(arrow_type):
        return str(value) if value is not None else None
    if pa.types.is_string(arrow_type):
        return str(value) if value is not None else None
    return value


# ── Markdown dataset card ──────────────────────────────────────────────────


def write_codebook_md(
    codebook: Codebook,
    stats: dict,
    out_path: Path,
    *,
    title: str | None = None,
    source_url: str | None = None,
) -> Path:
    title = title or codebook.name
    lines: list[str] = [
        "---",
        f"dataset: {codebook.name}",
        f"version: {codebook.version}",
        "---",
        "",
        f"# {title}",
        "",
        codebook.description.strip(),
        "",
    ]
    if codebook.domain:
        lines.append(f"**Domain:** {codebook.domain}")
        lines.append("")
    if source_url:
        lines.append(f"**Source URL:** {source_url}")
        lines.append("")
    lines.append(f"**Rows:** {stats.get('n_rows', '?')}")
    lines.append(f"**Variables:** {stats.get('n_variables', len(codebook.variables))}")
    tb = stats.get("type_breakdown", {})
    if tb:
        lines.append(f"**Type breakdown:** " + ", ".join(f"{k}={v}" for k, v in tb.items()))
    lines.append("")
    lines.append("## Variables")
    lines.append("")
    lines.append("| Name | Type | Coverage | Description |")
    lines.append("|---|---|---:|---|")
    by_name = {s["name"]: s for s in stats.get("variables", [])}
    for v in codebook.variables:
        s = by_name.get(v.name, {})
        cov = s.get("coverage")
        cov_s = f"{cov:.0%}" if isinstance(cov, (int, float)) else "—"
        descr = v.description
        if v.type == "enum" and v.enum_values:
            descr += f" Enum: {', '.join(v.enum_values)}."
        if v.unit:
            descr += f" Unit: {v.unit}."
        descr = descr.replace("|", "\\|")
        lines.append(f"| `{v.name}` | {v.type} | {cov_s} | {descr} |")
    lines.append("")
    if stats.get("issues"):
        lines.append("## Known data-quality notes")
        lines.append("")
        for issue in stats["issues"]:
            lines.append(f"- {issue.strip()}")
        lines.append("")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
