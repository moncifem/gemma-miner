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

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_miner.codebook import Codebook
from gemma_miner.coercion import coerce_row
from gemma_miner.parsing import _candidates, _repair_invalid_escapes, _strip_trailing_commas
from gemma_miner.tools.base import Tool, ToolResult
from gemma_miner.tools.codebook_tool import _load_codebook_from_state, _row_text

if TYPE_CHECKING:
    from gemma_miner.llm import LLMClient
    from gemma_miner.state import AgentState


_EXTRACT_SYSTEM = """You are a strict structured-data extractor.

Given a CODEBOOK (a JSON Schema describing variables to extract) and a TEXT,
produce ONE JSON OBJECT whose keys are exactly the codebook variable names.

THE NULL-NOT-FALSE DISCIPLINE (read carefully — this is the most important rule):

  • A boolean variable is `null` UNLESS the TEXT explicitly establishes the
    fact one way or the other. Silence is NOT evidence of absence.
      - `has_llm: true`   ← only if the text mentions LLMs / language models /
                              specific LLM products (GPT, Claude, Gemini, …).
      - `has_llm: false`  ← only if the text EXPLICITLY denies LLM use
                              ("no LLMs", "not an AI product", etc.).
      - `has_llm: null`   ← the text doesn't mention the topic at all.
    Most boolean variables will be `null` for most rows. THAT IS CORRECT.
    Manufacturing `false` from silence makes the dataset useless for stats.

  • Same principle for `is_*` flags, enums, numbers, dates: null when the
    text does NOT state the fact. Do NOT infer "0" from absence of a count;
    do NOT pick a "default" enum value.

OUTPUT RULES:
  • Output ONLY the JSON object. No prose. No ```json fences.
  • Use EXACTLY the variable names in the codebook as JSON keys (do not
    rename, do not add raw scraped fields like `decision_date` or `title`).
  • boolean → true / false / null (NEVER strings).
  • integer / number → plain numeric value (no units, no thousand separators).
  • date → "YYYY-MM-DD" string.
  • enum → EXACTLY one of the listed values, or null.
  • array → empty list [] only when the text mentions the category but lists
    no entries; null when the topic isn't covered at all.

The JSON object must include every variable name in the codebook (with null
when unknown — that's what null is for).
"""


# A row is "extractable" if its accessible source text crosses this threshold.
# Below it, we don't call the LLM — we'd just be inventing data.
_MIN_TEXT_CHARS = 200


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
    """Run one extraction on `row`. Returns (typed_row, coercion_warnings).

    The returned `typed_row` contains ONLY the `id` (join key) and the
    codebook columns — NOT the raw harvest fields. Bronze (raw harvest)
    and silver (typed variables) are stored as separate datasets joined
    by `id`, so the row count is meaningful in both layers.
    """
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
    # Typed-only row: just the id (join key) + codebook columns. No raw fields.
    # If bronze somehow lacks an `id`, mint a deterministic one from its
    # content so silver still has a stable join key — never propagate null.
    from gemma_miner.dataset import synthesize_id

    iid = row.get("id")
    if iid is None or iid == "":
        iid = synthesize_id(row)
    merged: dict = {"id": iid}
    for k, v in coerced.items():
        merged[k] = v
    return merged, warnings


class ExtractItemsTool(Tool):
    name = "extract_items"
    description = (
        "Apply the current codebook to items in the dataset. For each item we "
        "read its source text, send the codebook's JSON Schema + text to the "
        "LLM, coerce every returned value to the declared type, and write the "
        "typed row to extracted.jsonl (keyed by `id`). One LLM call per item.\n\n"
        "PILOT-THEN-SCALE PROTOCOL (built in):\n"
        "  • Default `limit=3` if you don't set one — a tiny pilot so you can "
        "    inspect coverage and value sanity BEFORE running on hundreds of "
        "    rows. Read the per-variable coverage in the output, then call "
        "    `assess_sample(layer='silver')` for a verdict.\n"
        "  • Once the pilot looks good, call again with `limit=null` (or a "
        "    large explicit number) to extract the rest. The skip_existing "
        "    default ensures we don't re-extract rows already done.\n\n"
        "Pass `limit=0` or `limit=null` and `pilot=false` to bypass the pilot "
        "default and process EVERY row in one call (use this only when you "
        "have evidence the codebook is solid)."
    )
    args_schema = {
        "limit": {"type": "integer", "description": "Only process the first N items. Default 3 (pilot)."},
        "pilot": {
            "type": "boolean", "default": True,
            "description": "If true (default) and no explicit limit, run a small pilot batch first.",
        },
        "skip_existing": {
            "type": "boolean",
            "default": True,
            "description": "Skip rows already present in extracted.jsonl (by id).",
        },
        "force": {
            "type": "boolean",
            "default": False,
            "description": (
                "Override the safety guard that refuses to re-extract more "
                "than 50 already-extracted rows. Only set this when you've "
                "genuinely changed the codebook AND you've thought about the cost."
            ),
        },
        "fill_new_only": {
            "type": "boolean",
            "default": True,
            "description": (
                "When the codebook has new variables since the last extract, "
                "only fill the new variables on existing rows (cheap & "
                "non-destructive). When false, every row is re-extracted "
                "from scratch (expensive)."
            ),
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
        force = bool(args.get("force", False))
        fill_new_only = bool(args.get("fill_new_only", True))
        max_chars = int(args.get("max_chars_per_item") or 24000)
        delay = max(0, int(args.get("delay_ms") or 0)) / 1000.0

        # PILOT-THEN-SCALE: if the caller didn't specify a limit AND silver
        # is currently empty, run a 3-item pilot first. This stops the agent
        # from burning hundreds of LLM calls on a broken codebook.
        extracted_ds = state.extracted_dataset()
        existing_extracted_rows = {
            str(r.get("id")): r for r in extracted_ds.rows()
            if r.get("id") is not None
        }
        existing_extracted_ids = set(existing_extracted_rows.keys())
        pilot_default = bool(args.get("pilot", True))
        is_pilot_run = False
        if limit is None and pilot_default and len(existing_extracted_ids) == 0:
            limit = 3
            is_pilot_run = True

        var_names = [v.name for v in cb.variables]

        # FORCE GUARD: refuse `skip_existing=false` when it would re-extract
        # a large block of rows the agent has already paid for.
        if not skip_existing and not force:
            already_in_silver = sum(
                1 for r in rows if str(r.get("id")) in existing_extracted_ids
            )
            if already_in_silver > 50:
                return ToolResult(
                    output=(
                        f"ERROR: skip_existing=false would re-extract "
                        f"{already_in_silver} rows already in extracted.jsonl. "
                        "That's an expensive operation and almost always a mistake.\n\n"
                        "If the codebook changed and you only need to fill the "
                        "NEW variables on existing rows, call:\n"
                        "  extract_items(skip_existing=false, fill_new_only=true)\n"
                        "which is cheap and non-destructive.\n\n"
                        "If you truly want to redo every row (e.g. you suspect "
                        "the previous extraction model was broken), pass "
                        "`force=true` AND state why in your `thought`."
                    ),
                    error=True,
                )

        # PARTIAL FILL: when the codebook has new variables vs. the last
        # extracted snapshot AND the caller asked for fill_new_only, restrict
        # the per-item LLM call to ONLY the new variables. Existing fields are
        # merged from the silver row.
        last_vars = state.memory.get("last_extracted_codebook_variables") or []
        new_vars = [n for n in var_names if n not in set(last_vars)]
        old_vars = [n for n in var_names if n in set(last_vars)]
        is_partial_fill = (
            fill_new_only
            and not skip_existing
            and len(last_vars) > 0
            and len(new_vars) > 0
            and len(new_vars) < len(var_names)
        )

        targets: list[dict] = []
        for r in rows:
            rid = str(r.get("id"))
            if skip_existing and rid in existing_extracted_ids:
                continue
            if is_partial_fill and rid not in existing_extracted_ids:
                # Partial fill is only meaningful for rows we've already done.
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
        model_name = getattr(getattr(self.llm, "config", None), "model", "extractor")
        # For partial-fill, build a reduced codebook used per item.
        effective_cb = cb
        if is_partial_fill:
            reduced_vars = [v for v in cb.variables if v.name in set(new_vars)]
            effective_cb = Codebook(
                name=cb.name,
                description=cb.description,
                domain=cb.domain,
                version=cb.version,
                variables=reduced_vars,
            )
        state.emit_progress(
            event="extract_start", total=len(targets),
            n_variables=len(effective_cb.variables) if is_partial_fill else len(var_names),
            model=model_name,
        )
        skipped_no_text = 0
        for i, r in enumerate(targets):
            iid = r.get("id", f"row_{i}")

            # INSUFFICIENT-TEXT GATE: if the row's accessible text is too thin
            # for meaningful extraction, mark and skip. Writing all-null rows
            # would inflate "coverage" with garbage and produce false negatives
            # (booleans defaulting to false). Better to surface honestly.
            from gemma_miner.tools.codebook_tool import _row_text as _check_text

            text_for_check = _check_text(r, str(state.workdir))
            if not text_for_check or len(text_for_check) < _MIN_TEXT_CHARS:
                if r.get("id") is not None:
                    skip_row = {
                        "id": r["id"],
                        "_extract_status": "no_text",
                        "_extract_text_chars": len(text_for_check or ""),
                    }
                    extracted_ds.upsert(skip_row)
                skipped_no_text += 1
                state.emit_progress(
                    event="extract_item_failed",
                    index=i + 1, total=len(targets), id=str(iid),
                    error="insufficient_text",
                )
                continue

            state.emit_progress(
                event="extract_item_start",
                index=i + 1, total=len(targets), id=iid, model=model_name,
            )
            try:
                merged, warn = extract_one_item(
                    self.llm, r, effective_cb, state.workdir, max_chars=max_chars
                )
                # In partial-fill mode, blend the new variables into the
                # existing silver row instead of replacing it.
                if is_partial_fill:
                    rid = str(r.get("id"))
                    prior = existing_extracted_rows.get(rid, {})
                    blended: dict = {"id": rid}
                    for k, v in prior.items():
                        if k != "id":
                            blended[k] = v
                    for k, v in merged.items():
                        if k != "id":
                            blended[k] = v
                    merged = blended
                # Write the typed-only row to the SILVER dataset
                # (extracted.jsonl). The raw harvest row in `state.dataset`
                # is left untouched.
                # If the bronze row had no `id`, the silver upsert will
                # silently fail (silver is keyed by id). Detect and surface
                # this rather than counting it as a success.
                if merged.get("id") is None:
                    errors.append(
                        f"  - row {i}: BRONZE row has no `id` field, so the typed "
                        "row can't be saved (silver is keyed by id). Re-harvest "
                        "via scrape_paginated (auto-ids rows) or call dataset_append "
                        "again (which now auto-ids missing ids)."
                    )
                    state.emit_progress(
                        event="extract_item_failed",
                        index=i + 1, total=len(targets), id="<no-id>",
                        error="bronze row missing id",
                    )
                    continue
                ok, reason = extracted_ds.upsert(merged)
                if not ok:
                    errors.append(f"  - id={iid}: silver upsert refused: {reason}")
                    state.emit_progress(
                        event="extract_item_failed",
                        index=i + 1, total=len(targets), id=str(iid),
                        error=reason,
                    )
                    continue
                extracted_count += 1
                warnings_total += len(warn)
                filled = 0
                for n in var_names:
                    if merged.get(n) is not None:
                        per_var_coverage[n] += 1
                        filled += 1
                state.emit_progress(
                    event="extract_item_done",
                    index=i + 1, total=len(targets), id=iid,
                    filled=filled, n_variables=len(var_names),
                    warnings=len(warn),
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"  - id={r.get('id', '?')}: {type(e).__name__}: {e}")
                state.emit_progress(
                    event="extract_item_failed",
                    index=i + 1, total=len(targets), id=iid,
                    error=str(e)[:200],
                )
            if delay:
                time.sleep(delay)
        state.emit_progress(
            event="extract_done",
            total=len(targets), extracted=extracted_count,
            errors=len(errors), warnings=warnings_total,
        )

        # Coverage summary — denominator is rows we ACTUALLY extracted from
        # (not "no_text" skips), so coverage is meaningful.
        n = extracted_count
        cov_lines: list[str] = []
        if n > 0:
            for v in cb.variables:
                c = per_var_coverage.get(v.name, 0)
                pct = c / n
                cov_lines.append(f"  {v.name:<28} {pct:.0%}  ({c}/{n})")
            cb_path = Path(state.memory.get("codebook_path") or (Path(state.workdir) / "codebook.json"))
            if cb_path.exists():
                # Store the EXTRACTION signature (cosmetic changes won't
                # invalidate it). See Codebook.extraction_signature.
                state.memory.set(
                    "last_extracted_codebook_hash",
                    cb.extraction_signature(),
                )
                state.memory.set("last_extracted_codebook_variables", var_names)

            # Sticky post-extract flag for the phase machine's hysteresis rule:
            # set it as soon as silver covers ≥90% of bronze and the pilot is
            # NOT the only thing that ran. The flag is observed in phases.py
            # to refuse falling back into ENUMERATE/PROCESS for marginal gains.
            if not is_pilot_run:
                bronze_n = len(state.dataset)
                silver_n = len(extracted_ds)
                if bronze_n > 0 and silver_n >= max(1, int(bronze_n * 0.9)):
                    state.memory.set("_post_extract_done", True)

        out = [
            f"extract_items: processed={extracted_count}  "
            f"skipped_no_text={skipped_no_text}  errors={len(errors)}  "
            f"warnings={warnings_total}"
            + ("  (PARTIAL FILL — new vars only)" if is_partial_fill else ""),
            f"raw_rows: {len(state.dataset)}   extracted_rows: {len(extracted_ds)}",
            f"silver_path: {extracted_ds.path}",
            "",
            "per-variable coverage:",
            *cov_lines,
        ]
        if skipped_no_text:
            out.append("")
            out.append(
                f"⚠ {skipped_no_text} row(s) skipped because they had <{_MIN_TEXT_CHARS} "
                "chars of source text. Those rows were marked with "
                "`_extract_status='no_text'`. If many rows are skipped, the "
                "bronze harvest never captured the actual body — re-harvest "
                "with a detail extractor or process_queue(mode='text')."
            )
        if is_partial_fill:
            out.insert(
                1,
                f"partial_fill: filled {len(new_vars)} new variable(s) {new_vars[:8]}"
                + (" …" if len(new_vars) > 8 else "")
                + f" on {extracted_count} existing rows; "
                + f"{len(old_vars)} pre-existing variable(s) were preserved.",
            )
        if errors:
            out.append("")
            out.append("errors:")
            out.extend(errors[:10])
            if len(errors) > 10:
                out.append(f"  ... and {len(errors) - 10} more")

        # PILOT verdict — synthesised here so the agent doesn't need a
        # follow-up assess_sample call for the simple case.
        if is_pilot_run:
            avg_cov = (
                sum(per_var_coverage.values()) / (extracted_count * len(var_names))
                if extracted_count and var_names else 0
            )
            zero_cov_vars = [n for n in var_names if per_var_coverage.get(n, 0) == 0]
            verdict = "SCALE_OK"
            advice: list[str] = []
            # Inspect the bronze sample text — a common failure is that the
            # raw rows have no actual text content (e.g. the listing
            # extractor captured ids but not titles/abstracts).
            empty_text_rows = 0
            for r in targets[:extracted_count]:
                t = _row_text(r, str(state.workdir)) if 'targets' in dir() else ""
                if not t or len(t) < 50:
                    empty_text_rows += 1
            if extracted_count == 0:
                verdict = "FIX_FIRST"
                advice.append("0 items extracted — every pilot row failed. Check the codebook + the row text.")
            elif avg_cov < 0.10:
                # Catastrophic — almost certainly the BRONZE rows have no
                # real text content. Don't waste another LLM call until the
                # bronze is fixed.
                verdict = "FIX_BRONZE_FIRST"
                advice.append(
                    f"average fill is {avg_cov*100:.0f}% — pilot rows had nothing for the LLM to read."
                )
                if empty_text_rows >= extracted_count // 2:
                    advice.append(
                        f"{empty_text_rows}/{extracted_count} pilot rows have < 50 chars of source text. "
                        "Your BRONZE rows are mostly empty. Run `assess_sample(layer='bronze')` to confirm, "
                        "then either: (a) re-run extractor_define so the listing actually captures the "
                        "text fields, or (b) switch to listing+detail and use process_queue(mode='text') "
                        "to fetch each item's detail page."
                    )
                else:
                    advice.append(
                        "Possible causes: codebook variables don't match the content, the extraction "
                        "prompt is too generic, or the text is in a language the codebook didn't anticipate."
                    )
            elif avg_cov < 0.40:
                verdict = "FIX_FIRST"
                advice.append(
                    f"average coverage is only {avg_cov*100:.0f}% — most variables aren't being "
                    "filled. Likely causes: (a) the codebook variables don't match what the text "
                    "contains, (b) row text is too short / missing, (c) extraction prompt is unclear."
                )
            elif len(zero_cov_vars) >= max(3, len(var_names) // 4):
                verdict = "FIX_FIRST"
                advice.append(
                    f"{len(zero_cov_vars)} variables have 0% coverage: "
                    f"{zero_cov_vars[:6]}{'…' if len(zero_cov_vars) > 6 else ''}. "
                    "Drop or rewrite them via codebook_edit, then re-pilot."
                )
            out.append("")
            out.append(f"PILOT verdict: {verdict}  (pilot size={extracted_count}, avg coverage={avg_cov*100:.0f}%)")
            if verdict == "SCALE_OK":
                out.append(
                    "→ Looks healthy. Call extract_items again WITHOUT a limit "
                    "(or with limit=null) to process the remaining "
                    f"{len(state.dataset) - extracted_count} rows."
                )
            else:
                for a in advice:
                    out.append(f"  • {a}")
                out.append("→ Fix the codebook (codebook_edit / codebook_design with revised hints) BEFORE scaling.")
        # Surface as a real error when the batch produced nothing usable.
        # Otherwise the repeated-failure detector misses 10 consecutive
        # 0-row pilot runs (because `error=False` looks fine to it).
        had_total_failure = (
            extracted_count == 0
            and skipped_no_text == 0
            and len(errors) > 0
        )
        return ToolResult(
            output="\n".join(out),
            error=had_total_failure,
            artifact={
                "extracted": extracted_count,
                "pilot": is_pilot_run,
                "n_variables": len(var_names),
            },
        )
