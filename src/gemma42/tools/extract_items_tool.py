"""Extract the codebook's structured fields from every item in the dataset.

For each item we:
  1. Load the primary text (from `text_path` if present, else `pdf_text`).
  2. Send the codebook's JSON Schema + the text to the LLM.
  3. Coerce every returned value to its declared type.
  4. UPSERT the merged row into the dataset, keyed by `id` (unique).
  5. Save extraction stats (per-variable coverage) to memory for later phases.

We deliberately do ONE LLM call per item — small models handle one structured
object reliably; batching gets brittle fast. The whole job is a macro tool so
the model sees just one big tool call, not 100 turns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma42.codebook import Codebook
from gemma42.coercion import coerce_row
from gemma42.parsing import _candidates, _repair_invalid_escapes, _strip_trailing_commas
from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.codebook_tool import _load_codebook_from_state, _row_text

if TYPE_CHECKING:
    from gemma42.llm import LLMClient
    from gemma42.state import AgentState


_EXTRACT_SYSTEM = """You are a strict structured-data extractor.

Given a CODEBOOK (a JSON Schema describing variables to extract) and a TEXT,
produce ONE JSON OBJECT whose keys are exactly the codebook variable names.

RULES:
- Output ONLY the JSON object. No prose. No ```json fences.
- boolean → true or false (never strings).
- integer / number → plain numeric value (no units, no thousand separators).
- date → "YYYY-MM-DD" string.
- enum → EXACTLY one of the listed values, or null.
- array → empty list [] if none mentioned.
- UNKNOWN / NOT STATED → null. Do not guess.
- Use the description and hints of each variable to choose the right value.

The JSON object must include every variable name in the codebook.
"""


def _parse_extraction_json(raw: str) -> dict | None:
    for cand in _candidates(raw):
        for variant in (
            cand,
            _strip_trailing_commas(cand),
            _repair_invalid_escapes(cand),
            _strip_trailing_commas(_repair_invalid_escapes(cand)),
        ):
            try:
                obj = json.loads(variant)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return obj
    return None


def extract_one_item(
    llm: "LLMClient",
    row: dict,
    codebook: Codebook,
    workdir: str | Path,
    *,
    max_chars: int = 24000,
    temperature: float = 0.0,
) -> tuple[dict, dict[str, str]]:
    """Run one extraction on `row`. Returns (merged_row, coercion_warnings)."""
    text = _row_text(row, str(workdir))
    if not text:
        raise ValueError("no text content available for this item")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[truncated, full {len(text)} chars]"

    schema = codebook.to_json_schema()
    user = (
        "CODEBOOK (JSON Schema):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "TEXT:\n<<<\n"
        f"{text}\n>>>\n\n"
        "Return one JSON object."
    )
    raw = llm.chat(
        [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    obj = _parse_extraction_json(raw)
    if obj is None:
        raise ValueError(f"could not parse JSON object from model output (first 400 chars):\n{raw[:400]}")
    coerced, warnings = coerce_row(obj, codebook.variables)
    # Merge: keep all existing row metadata, then overlay the structured fields.
    merged = dict(row)
    for k, v in coerced.items():
        merged[k] = v
    return merged, warnings


class ExtractItemsTool(Tool):
    name = "extract_items"
    description = (
        "Apply the current codebook to EVERY item in the dataset (or a "
        "subset). For each item we read its source text, send the codebook's "
        "JSON Schema + text to the LLM, coerce every returned value to the "
        "declared type, and UPSERT the merged row back into the dataset "
        "(keyed by `id`). One LLM call per item; one tool call handles the "
        "whole corpus.\n\n"
        "Recommended workflow:\n"
        "  1. After harvesting, run `codebook_propose` → `codebook_test` → "
        "iterate until you're happy.\n"
        "  2. Run `extract_items(limit=...)` to fill the codebook columns "
        "for every item. Watch the per-variable coverage in the output.\n"
        "  3. Run `dataset_export` to produce parquet + codebook.md."
    )
    args_schema = {
        "limit": {"type": "integer", "description": "Only process the first N items (default all)."},
        "skip_existing": {
            "type": "boolean",
            "default": True,
            "description": "Skip rows that already have at least 50% of codebook variables populated.",
        },
        "max_chars_per_item": {"type": "integer", "default": 24000},
        "delay_ms": {"type": "integer", "default": 0},
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook saved", error=True)
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: dataset is empty", error=True)
        limit = args.get("limit")
        skip_existing = bool(args.get("skip_existing", True))
        max_chars = int(args.get("max_chars_per_item") or 24000)
        delay = max(0, int(args.get("delay_ms") or 0)) / 1000.0

        var_names = [v.name for v in cb.variables]
        targets: list[dict] = []
        for r in rows:
            if skip_existing:
                present = sum(1 for n in var_names if r.get(n) is not None)
                if present >= max(1, len(var_names) // 2):
                    continue
            targets.append(r)
            if limit is not None and len(targets) >= int(limit):
                break

        if not targets:
            return ToolResult(
                output="(nothing to do — every row already has codebook fields. Pass skip_existing=false to re-extract.)"
            )

        extracted_count = 0
        warnings_total = 0
        per_var_coverage: dict[str, int] = {n: 0 for n in var_names}
        errors: list[str] = []
        for i, r in enumerate(targets):
            try:
                merged, warn = extract_one_item(
                    self.llm, r, cb, state.workdir, max_chars=max_chars
                )
                # Upsert by id.
                key = r.get("id")
                state.dataset.upsert(merged)
                extracted_count += 1
                warnings_total += len(warn)
                for n in var_names:
                    if merged.get(n) is not None:
                        per_var_coverage[n] += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"  - id={r.get('id', '?')}: {type(e).__name__}: {e}")
            if delay:
                time.sleep(delay)

        # Coverage summary
        n = extracted_count
        cov_lines: list[str] = []
        if n > 0:
            for v in cb.variables:
                c = per_var_coverage.get(v.name, 0)
                pct = c / n
                cov_lines.append(f"  {v.name:<28} {pct:.0%}  ({c}/{n})")

        out = [
            f"extract_items: processed={extracted_count}  errors={len(errors)}  warnings={warnings_total}",
            f"dataset_rows_now: {len(state.dataset)}",
            "",
            "per-variable coverage:",
            *cov_lines,
        ]
        if errors:
            out.append("")
            out.append("errors:")
            out.extend(errors[:10])
            if len(errors) > 10:
                out.append(f"  ... and {len(errors) - 10} more")
        return ToolResult(output="\n".join(out))
