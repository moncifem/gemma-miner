"""LLM-driven scraping: when regex specs don't work, ask the model.

This is the intelligent fallback. The agent gives the tool:
  - a cached HTML body (path or url)
  - a list of fields it wants (with optional types/descriptions)
  - an item count

The tool:
  1. Strips heavy chrome (head/script/style/nav/footer).
  2. Chunks the page into the K most plausible row regions (using the LLM,
     not hand-rolled rules).
  3. Asks the LLM to extract each row as JSON conforming to the fields.
  4. Coerces values, deduplicates, returns rows.

No regex spec needed. Works on any page the model can read. The regex
`extractor_define` path is still preferred when it works (it's free + fast +
deterministic), but this is the escape hatch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_miner.failure_log import log_failure as _fail_log
from gemma_miner.parsing import (
    _candidates,
    _repair_invalid_escapes,
    _strip_trailing_commas,
)
from gemma_miner.tools.base import Tool, ToolResult
from gemma_miner.tools.html_tool import _load_or_raise, _resolve_source_arg, _SourceNotFoundError


def _log_failure(state, *, kind: str, payload: dict | None = None) -> None:
    """Thin wrapper so the file can call _log_failure(state, …) without
    constructing the workdir each time."""
    raw = None
    if payload and "raw_response" in payload:
        raw = payload.pop("raw_response")
    _fail_log(getattr(state, "workdir", None),
              kind=kind, tool="llm_scrape",
              raw_response=raw, payload=payload)

if TYPE_CHECKING:
    from gemma_miner.llm import LLMClient
    from gemma_miner.state import AgentState


_SYS = """You are a structured information extractor. You receive a chunk of HTML and a list of fields. Return a JSON OBJECT with a single key "items" whose value is an array of one object per repeating item visible in the HTML. Output JSON only — no prose, no fences.

OUTPUT SHAPE (EXACTLY):
{"items": [
  {"<field_1>": ..., "<field_2>": ...},
  {"<field_1>": ..., "<field_2>": ...},
  ...
]}

Rules:
 - One inner object per repeating item (one story, one row, one card).
 - The "items" array MUST contain ALL items visible in the HTML up to the requested count — DO NOT return just one.
 - For each item, set every requested field. Use null ONLY if the field is genuinely absent / unstated for THAT item.
 - For COUNT fields (comments, votes, replies, etc.), a "discuss" / "reply" link or blank slot means ZERO — use 0, not null.
 - Booleans → true / false. Counts → integers (no commas, no units). Dates → "YYYY-MM-DD" when possible.
 - DO NOT invent items not in the HTML. DO NOT include chrome (nav, footer, ads). Only the main repeating content.
"""


def _strip_chrome(html: str) -> str:
    """Drop heavy non-content sections so the LLM sees signal."""
    out = html
    # Drop script / style / svg / noscript wholesale.
    for tag in ("script", "style", "svg", "noscript"):
        out = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", out, flags=re.DOTALL | re.IGNORECASE)
    # Drop <head> and HTML comments.
    out = re.sub(r"<head\b.*?</head>", "", out, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
    # Drop the boilerplate header/footer/nav blocks if present.
    for tag in ("header", "footer", "nav"):
        out = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", out, flags=re.DOTALL | re.IGNORECASE)
    # Trim long CSS class strings (no signal).
    out = re.sub(r'class="[^"]{60,}"', 'class=""', out)
    # Drop deeply nested style/inline attributes that bloat tokens.
    out = re.sub(r'\s+style="[^"]*"', "", out)
    out = re.sub(r'\s+(?:cellpadding|cellspacing|bgcolor|border|width|height|align|valign)="[^"]*"', "", out)
    # Collapse runs of whitespace to one space (preserves separators).
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n[ \t]*\n+", "\n", out)
    return out


def _chunk(html: str, max_chars: int = 30_000) -> list[str]:
    """Slice the (chrome-stripped) HTML into chunks ≤ max_chars."""
    if len(html) <= max_chars:
        return [html]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(html):
        end = min(len(html), cursor + max_chars)
        # try to break at the nearest closing tag for cleanliness
        if end < len(html):
            cut = html.rfind(">", cursor + max_chars - 2000, end)
            if cut > cursor:
                end = cut + 1
        chunks.append(html[cursor:end])
        cursor = end
    return chunks


def _parse_array(raw: str) -> list[dict]:
    """Best-effort parse of an array of objects from the LLM output.

    Accepts:
      • a JSON array `[{...}, ...]`
      • an envelope dict `{"items": [...]}` (also list/results/data/rows/stories…)
      • a SINGLE JSON object → treated as a 1-element list. Ollama with
        response_format={"type":"json_object"} forces this collapsed shape,
        so we accept it gracefully rather than dropping a real row to zero.
    """
    ENVELOPE_KEYS = ("items", "rows", "data", "results", "list",
                       "stories", "records", "entries", "objects")

    def _interpret(obj):
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            for k in ENVELOPE_KEYS:
                v = obj.get(k)
                if isinstance(v, list):
                    return [o for o in v if isinstance(o, dict)]
            if obj and not any(k in obj for k in ENVELOPE_KEYS):
                return [obj]
        return None

    # 1. Try the whole text first (so a top-level array isn't shadowed by
    # `_candidates` returning the first balanced `{...}` block).
    candidates: list[str] = [raw.strip()]
    # Also try every candidate substring _candidates() proposes.
    for c in _candidates(raw):
        if c not in candidates:
            candidates.append(c)
    # Plus: if raw begins with text + an array, try to find the array directly.
    arr_start = raw.find("[")
    if arr_start >= 0:
        # Find the matching closing bracket
        depth = 0
        for i in range(arr_start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[arr_start:i + 1])
                    break

    for cand in candidates:
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
            result = _interpret(obj)
            if result:
                return result
    return []


def _build_canonical_map(state: "AgentState", rows: list[dict]) -> dict[str, str]:
    """Build a map {observed_key → canonical_key} so we can rename the rows
    we're about to push so they use the names the user actually asked for.

    The canonical names come from the FieldsContract on the run. We only
    rename when an observed key is a known *variant* of a canonical name
    AND the canonical name isn't already present in the row. This makes
    `n_comments` win over `comments` when the user requested `n_comments`.
    """
    try:
        from gemma_miner.contracts import FieldsContract, _field_variants
    except Exception:  # noqa: BLE001
        return {}
    canonicals: list[str] = []
    for c in state.contracts.list():
        if isinstance(c, FieldsContract):
            canonicals.extend(c.required_fields)
    if not canonicals:
        return {}
    observed: set[str] = set()
    for r in rows:
        observed.update(r.keys())
    mapping: dict[str, str] = {}
    for canon in canonicals:
        if canon in observed:
            continue
        for variant in _field_variants(canon):
            if variant in observed and variant != canon:
                mapping[variant] = canon
                break
    return mapping


def _apply_canonical_map(row: dict, mapping: dict[str, str]) -> dict:
    if not mapping:
        return row
    out: dict = {}
    for k, v in row.items():
        out[mapping.get(k, k)] = v
    return out


def _required_field_names(state: "AgentState") -> list[str]:
    try:
        from gemma_miner.contracts import FieldsContract
    except Exception:  # noqa: BLE001
        return []
    required: list[str] = []
    for c in state.contracts.list():
        if isinstance(c, FieldsContract):
            required.extend(c.required_fields)
    return required


def _pick_signature_key(required: list[str]) -> str | None:
    """Pick a single field whose value uniquely identifies a row, when
    available. URL-like > id-like > first required field. Generic — applies
    to any website that exposes a stable URL or id per item."""
    if not required:
        return None
    low = [(f, f.lower()) for f in required]
    for f, l in low:
        if l.endswith("_url") or l == "url":
            return f
    for f, l in low:
        if l.endswith("_id") or l == "id" or l.endswith("_slug"):
            return f
    return required[0]


def _row_quality_score(row: dict, required: list[str]) -> float:
    """Fraction of the user's required fields that are non-empty on this row.
    Used as a quality gate so we can reject extractions where the model
    returned shapes that don't match what the user asked for."""
    if not required:
        return 1.0
    from gemma_miner.contracts import _field_variants, _is_meaningful
    hit = 0
    for f in required:
        variants = _field_variants(f)
        if any(_is_meaningful(row.get(v)) for v in variants):
            hit += 1
    return hit / len(required)


def _normalize_optional_required_fields(row: dict, required: list[str]) -> dict:
    """Fill boolean fact fields with explicit negative values.

    The planner maps optional facts ("any X", "whether X") to `has_x`/`is_x`.
    Those fields should be rectangular booleans, not nullable strings.
    """
    if not required:
        return row
    out = dict(row)
    for name in required:
        v = out.get(name)
        if v not in (None, ""):
            continue
        lname = name.lower()
        if lname.startswith("has_") or lname.startswith("is_"):
            out[name] = False
    return out


class LLMScrapeTool(Tool):
    name = "llm_scrape"
    description = (
        "INTELLIGENT FALLBACK extractor. When `extractor_define` keeps "
        "matching 0 rows, use this. Give it a cached HTML page (`source` or "
        "`url`) and the list of `fields` you want; the model reads the page "
        "and returns rows as JSON. No regex required.\n\n"
        "Args:\n"
        "  source     : path to cached HTML (or pass `url` if already fetched)\n"
        "  fields     : list of {name, description?, type?} dicts OR list of bare strings\n"
        "  target     : how many rows to aim for (the LLM will return up to this many)\n"
        "  context    : optional one-line hint (e.g. 'one row per Hacker News story')\n"
        "  push_to_dataset : DEFAULT TRUE. Every extracted row is "
        "                    appended directly to the dataset (id auto-fill).\n\n"
        "Returns the rows it found. Use this when you're tired of fighting "
        "regex — the model will figure out the page structure."
    )
    args_schema = {
        "source":          {"type": "string"},
        "url":             {"type": "string"},
        "fields":          {"type": "array"},
        "target":          {"type": "integer", "default": 30},
        "context":         {"type": "string"},
        "push_to_dataset": {"type": "boolean", "default": True},
        "force": {
            "type": "boolean", "default": False,
            "description": (
                "Override source-lock guards (push when silver is already "
                "populated, or when row shape differs significantly from "
                "existing rows). Only set this when you've understood the "
                "consequences for downstream extraction."
            ),
        },
        "max_chars_per_chunk": {
            "type": "integer",
            "description": "Page slice size in CHARS. Default 80000 (~20K tokens, fits 128K-context models with room for output).",
        },
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        fields_arg = args.get("fields") or []
        if not isinstance(fields_arg, list) or not fields_arg:
            return ToolResult(output="ERROR: 'fields' must be a non-empty array",
                              error=True)
        # Normalise fields to a uniform [{name, description, type}] shape
        fields: list[dict] = []
        for f in fields_arg:
            if isinstance(f, str):
                fields.append({"name": f})
            elif isinstance(f, dict) and "name" in f:
                fields.append({k: v for k, v in f.items() if k in ("name", "type", "description")})
            else:
                return ToolResult(output=f"ERROR: field entry invalid: {f}", error=True)

        target = int(args.get("target") or 30)
        # Accept common alias names — the model writes `goal=...` or
        # `hint=...` or `description=...` half the time. They all mean
        # "one-line context for the extraction."
        context = (
            args.get("context")
            or args.get("goal")
            or args.get("hint")
            or args.get("description")
            or args.get("instruction")
            or ""
        )
        # DEFAULT push=True. If the model calls llm_scrape without setting it,
        # the rows reach the dataset — preventing the failure mode where an
        # agent scrapes 5 pages and the rows are stuck in transient artifacts.
        push = bool(args.get("push_to_dataset", True))
        # Pick a chunk size that uses the model's actual context window.
        # Rule of thumb: ~4 chars/token, leave headroom for the system prompt
        # (~2K tokens), field schema (~500), and output (~max_tokens).
        ctx_tokens = getattr(self.llm.config, "context_window", 128_000)
        out_tokens = getattr(self.llm.config, "max_tokens", 16_384)
        budget_chars = max(8_000, (ctx_tokens - out_tokens - 3_000) * 4)
        default_chunk = min(budget_chars, 80_000)
        max_chars = int(args.get("max_chars_per_chunk") or default_chunk)

        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        cleaned = _strip_chrome(html)
        chunks = _chunk(cleaned, max_chars=max_chars)

        # User-facing field schema
        field_lines = []
        for f in fields:
            tag = f.get("type", "string")
            descr = f.get("description", "")
            field_lines.append(f"  - {f['name']} ({tag}): {descr}")
        field_block = "\n".join(field_lines)

        rows: list[dict] = []
        seen: set[str] = set()
        empty_chunks: list[dict] = []   # (chunk_idx, raw_response) — for debug logging

        for ci, chunk in enumerate(chunks):
            if len(rows) >= target:
                break
            remaining = target - len(rows)
            user = (
                (f"CONTEXT: {context}\n\n" if context else "")
                + f"FIELDS to extract (use these exact JSON keys):\n{field_block}\n\n"
                + f"Return UP TO {remaining} items wrapped in "
                + '{"items": [...]}. The "items" array MUST contain MANY '
                + f"items (not just one); include every repeating row you see in the HTML, "
                + f"up to {remaining}.\n\n"
                + f"HTML CHUNK ({ci+1}/{len(chunks)}):\n<<<\n{chunk}\n>>>"
            )
            # NOTE: we do NOT pass response_format={"type":"json_object"}
            # here because Ollama interprets it as "single object" and
            # collapses our array into one row. The prompt asks for
            # {"items": [...]} explicitly; the parser is also tolerant
            # enough to handle bare arrays and single objects.
            raw = self.llm.chat(
                [{"role": "system", "content": _SYS},
                 {"role": "user", "content": user}],
                temperature=0.0,
            )
            chunk_rows = _parse_array(raw)
            if not chunk_rows:
                empty_chunks.append({
                    "chunk_index": ci,
                    "chunk_size":  len(chunk),
                    "raw_response": raw,
                })
                _log_failure(state, kind="llm_scrape_empty_chunk", payload={
                    "chunk_index": ci, "of": len(chunks),
                    "chunk_chars": len(chunk),
                    "fields": [f.get("name") for f in fields],
                    "raw_response": raw,
                })
                continue
            for r in chunk_rows:
                # Light dedup by JSON signature of present keys
                key = json.dumps({k: r.get(k) for k in sorted(r.keys())},
                                  default=str, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
                if len(rows) >= target:
                    break

        # Compute net-new BEFORE the push runs (used to amend the headline).
        existing_rows_count = len(state.dataset)
        out_lines = [
            f"llm_scrape: extracted {len(rows)} row(s) (target {target}) from "
            f"{len(chunks)} chunk(s) of {len(cleaned)} chars.",
        ]
        for i, r in enumerate(rows[:3]):
            out_lines.append(f"--- row {i} ---")
            out_lines.append(json.dumps(r, ensure_ascii=False, indent=2))
        if len(rows) > 3:
            out_lines.append(f"... +{len(rows) - 3} more rows in the artifact")

        # Surface empty-chunk debug info so the agent (and the user reading
        # the trace) can see what the model actually returned.
        if empty_chunks:
            out_lines.append("")
            out_lines.append(
                f"⚠ {len(empty_chunks)} chunk(s) returned ZERO rows. "
                "The model output was unparseable as a JSON array. "
                "See <workdir>/failures.log for the raw responses."
            )
            for ec in empty_chunks[:2]:
                preview = (ec["raw_response"] or "")[:500].replace("\n", " ")
                out_lines.append(
                    f"  chunk #{ec['chunk_index']} raw[:500]: {preview!r}"
                )
            if not rows:
                out_lines.append("")
                out_lines.append(
                    "Diagnose: the response was likely either (a) empty, "
                    "(b) prose instead of JSON, (c) a single object instead of "
                    "an array. Inspect <workdir>/failures.log for details, "
                    "then try `extractor_define` with regex (faster + free) "
                    "or pass smaller `max_chars_per_chunk` for a retry."
                )

        # ── QUALITY GATE ──────────────────────────────────────────────────
        # Generic check applicable to ANY website: if the user asked for
        # specific fields (via FieldsContract) and the model extracted rows
        # where most of those fields are null, the source most likely does
        # not contain the requested data. Refuse the push and escalate so
        # the agent fetches a different URL or finishes with an explicit
        # "source lacks the data" summary, instead of accumulating garbage.
        required_fields_for_gate = _required_field_names(state)
        garbage_warning: str | None = None  # noqa: F841 — surfaced later
        if required_fields_for_gate and rows:
            scores = [_row_quality_score(r, required_fields_for_gate) for r in rows]
            mean_quality = sum(scores) / len(scores)
            empty_rows = sum(1 for s in scores if s == 0.0)
            empty_frac = empty_rows / len(rows)
            # >=80% of rows have NONE of the required fields → almost certainly
            # the wrong shape (e.g. participant list when projects were asked).
            if empty_frac >= 0.8:
                # Track consecutive failures to escalate after the second one.
                prev_fail = int(state.memory.get("_llm_scrape_consec_garbage", 0) or 0)
                state.memory.set("_llm_scrape_consec_garbage", prev_fail + 1)
                msg_lines = [
                    f"llm_scrape: REJECTED — {empty_rows}/{len(rows)} extracted rows "
                    f"have NONE of the user-required fields ({required_fields_for_gate}). "
                    f"Mean required-field coverage: {mean_quality:.0%}.",
                    "",
                    "The source most likely does not contain the requested data, OR "
                    "the model is reading the wrong section of the page. Do ONE of:",
                    "  (a) http_get a DIFFERENT URL that actually lists the items "
                    "      you want (project submissions, articles, etc.), OR",
                    "  (b) extract a SAMPLE row with `html_inspect` / `html_find` to "
                    "      verify the fields are actually on the page, OR",
                    "  (c) if you have evidence the data does not exist here, call "
                    "      `finish(force=true, summary='source URL does not contain "
                    "      <fields>; data is unavailable')` to end the run honestly.",
                ]
                if prev_fail >= 1:
                    msg_lines.append("")
                    msg_lines.append(
                        f"⚠ This is the {prev_fail + 1}{'nd' if prev_fail == 1 else 'rd+'} "
                        "consecutive empty-quality extraction. Stop scraping this URL — "
                        "fetch a different source or finish with force=true."
                    )
                _log_failure(state, kind="llm_scrape_quality_gate", payload={
                    "n_rows": len(rows), "empty_rows": empty_rows,
                    "mean_quality": mean_quality,
                    "required_fields": required_fields_for_gate,
                    "consec_failures": prev_fail + 1,
                })
                return ToolResult(output="\n".join(msg_lines), error=True,
                                  artifact={"rows": rows, "rejected": True})
            else:
                # At least some rows are valid — clear the consecutive-failure
                # counter so future legitimate empties don't compound.
                state.memory.set("_llm_scrape_consec_garbage", 0)
                if mean_quality < 0.5:
                    garbage_warning = (
                        f"⚠ Low extraction quality: mean required-field coverage "
                        f"only {mean_quality:.0%} across {len(rows)} rows. "
                        "Many rows will fail the FieldsContract — consider "
                        "fetching a more targeted URL or re-running with a "
                        "smaller `max_chars_per_chunk` so the model focuses."
                    )

        # If no rows were extracted at all from a multi-chunk page, that is
        # also a hard failure (not a "retry quietly" condition).
        if rows == [] and len(chunks) > 1:
            prev_fail = int(state.memory.get("_llm_scrape_consec_empty", 0) or 0)
            state.memory.set("_llm_scrape_consec_empty", prev_fail + 1)
            if prev_fail >= 1:
                return ToolResult(
                    output=(
                        "llm_scrape: REJECTED — zero rows extracted from "
                        f"{len(chunks)} chunk(s), and this is the "
                        f"{prev_fail + 1}{'nd' if prev_fail == 1 else 'rd+'} "
                        "consecutive empty extraction on this source. "
                        "Stop retrying llm_scrape on this URL. Options:\n"
                        "  (a) fetch a different URL (paginate, follow links), OR\n"
                        "  (b) finish(force=true, summary='source does not yield "
                        "the requested items')."
                    ),
                    error=True,
                    artifact={"rows": [], "rejected": True},
                )
        else:
            state.memory.set("_llm_scrape_consec_empty", 0)

        # ── SOURCE-LOCK GUARDS ────────────────────────────────────────────
        # Once silver (extracted.jsonl) has rows, adding new bronze rows
        # invalidates downstream extraction. Refuse the push by default.
        # Same story for shape drift against existing bronze rows.
        force_push = bool(args.get("force", False))
        if push and rows and not force_push:
            silver_len = (
                len(state._extracted_dataset)
                if state._extracted_dataset is not None else 0
            )
            if silver_len > 0:
                push = False
                out_lines.append("")
                out_lines.append(
                    "🛑 SOURCE LOCK: refused to push — extracted.jsonl already "
                    f"has {silver_len} typed rows. Adding new bronze rows now "
                    "would force a full re-extract and mix sources. Either:\n"
                    "  • stop scraping and call dataset_export, OR\n"
                    "  • pass force=true if you genuinely want to harvest more "
                    "    (and accept that another extraction sweep is required)."
                )
            else:
                existing_for_shape = state.dataset.rows()
                if existing_for_shape and rows:
                    existing_keys = set()
                    for r in existing_for_shape[:20]:
                        existing_keys.update(
                            k for k in r.keys()
                            if not str(k).startswith("_") and k != "id"
                        )
                    incoming_keys: set = set()
                    for r in rows[:20]:
                        incoming_keys.update(
                            k for k in r.keys()
                            if not str(k).startswith("_") and k != "id"
                        )
                    if existing_keys and incoming_keys:
                        overlap = (
                            len(existing_keys & incoming_keys)
                            / max(1, len(existing_keys | incoming_keys))
                        )
                        if overlap < 0.5:
                            push = False
                            out_lines.append("")
                            out_lines.append(
                                "🛑 SHAPE LOCK: refused to push — incoming rows "
                                f"share only {overlap:.0%} of keys with existing "
                                "rows. Mixing shapes produces a heterogeneous "
                                "dataset. Either match the existing shape or "
                                "pass force=true."
                            )
                            out_lines.append(
                                f"  existing keys: {sorted(existing_keys)[:10]}"
                            )
                            out_lines.append(
                                f"  incoming keys: {sorted(incoming_keys)[:10]}"
                            )

        if push and rows:
            # Canonical rename: the user asked for specific column names
            # (carried by the FieldsContract). The LLM may produce variants
            # (e.g. "comments" when the user asked for "n_comments"). Rewrite
            # row keys to the canonical names so the exported dataset matches
            # what the user requested.
            canonical_map = _build_canonical_map(state, rows)
            if canonical_map:
                rows = [_apply_canonical_map(r, canonical_map) for r in rows]
            required_fields = _required_field_names(state)
            if required_fields:
                rows = [_normalize_optional_required_fields(r, required_fields) for r in rows]
            # Build a set of value-signatures from rows ALREADY in the dataset.
            # This prevents the model from inflating the dataset by calling
            # llm_scrape repeatedly with push_to_dataset=true.
            #
            # Signature strategy: if the user gave us a "natural primary key"
            # via FieldsContract (something URL-like or id-like), use it as
            # the signature — that catches "same item re-scraped with slightly
            # different cell values". Otherwise fall back to the full-row
            # JSON signature. This is generic and works for any website that
            # exposes a stable identifier per item.
            primary_key = _pick_signature_key(required_fields)
            def _sig(r: dict) -> str:
                if primary_key:
                    pk_val = r.get(primary_key)
                    if pk_val not in (None, ""):
                        return f"__pk__:{primary_key}={pk_val}"
                return json.dumps(
                    {k: r.get(k) for k in sorted(r.keys())
                     if not k.startswith("_") and k != "id"},
                    default=str, ensure_ascii=False,
                )
            existing = state.dataset.rows()
            needed_new = max(0, target - len(existing))
            existing_sigs: set[str] = {_sig(r) for r in existing}
            appended = 0
            duplicates = 0
            failures = 0
            for i, r in enumerate(rows):
                sig = _sig(r)
                if sig in existing_sigs:
                    duplicates += 1
                    continue
                row = dict(r)
                # Use the shared deterministic id synthesiser so the same row
                # gets the same id across tools and across re-runs. The old
                # `row_<position>` scheme was NOT idempotent and broke the
                # bronze↔silver join when re-scraping the same page.
                from gemma_miner.dataset import ensure_row_id

                ensure_row_id(row)
                ok, _ = state.dataset.append(row)
                if ok:
                    appended += 1
                    existing_sigs.add(sig)
                else:
                    failures += 1
            out_lines.append("")
            out_lines.append(
                f"appended to dataset: {appended}  "
                f"duplicates skipped: {duplicates}  "
                f"failed: {failures}  "
                f"total rows now: {len(state.dataset)}"
            )
            # Promote the net-new count into the HEADLINE — small models
            # focus on the first line and were missing the "appended" detail.
            out_lines[0] = (
                f"llm_scrape: NET-NEW {appended} row(s) added to dataset "
                f"({duplicates} duplicates skipped). "
                f"Extracted {len(rows)} from {len(chunks)} chunk(s); "
                f"total now {len(state.dataset)}."
            )
            if duplicates and appended == 0:
                out_lines.append(
                    "🛑 ALL rows were duplicates. The same source has already "
                    "been scraped — calling llm_scrape on it again will not "
                    "help. Do ONE of:\n"
                    "  (a) http_get a DIFFERENT URL (the next page if the "
                    "site paginates), OR\n"
                    "  (b) If you have enough rows, move on: codebook_propose, "
                    "extract_items, then dataset_export, OR\n"
                    "  (c) If the goal is reached, call finish(summary='...')."
                )
            elif duplicates > appended and appended > 0:
                out_lines.append(
                    f"⚠ More duplicates ({duplicates}) than new rows ({appended}). "
                    "The page is mostly the same as last time — you're hitting "
                    "diminishing returns. Move to codebook_propose / extract_items "
                    "with what you have, OR fetch a different URL."
                )
            elif needed_new and appended < needed_new and appended > 0:
                out_lines.append(
                    f"⚠ Got fewer rows than target ({appended} new vs "
                    f"{needed_new} needed). The LLM may have truncated; either "
                    "retry with smaller `max_chars_per_chunk` (e.g. 8000), "
                    "fetch a DIFFERENT page if the site paginates, OR if "
                    "you have ≥80% of target, call finish(force=true)."
                )

        if garbage_warning:
            out_lines.append("")
            out_lines.append(garbage_warning)
        return ToolResult(output="\n".join(out_lines), artifact={"rows": rows})
