"""Declarative extractors + paginated scrape + queue processing macros.

This is the architecture that makes the agent robust for small models. The
flow is:

  1. The model writes a SPEC (JSON, ~20 lines) describing how to slice the
     HTML of a listing page into rows + per-row fields. The spec is stored
     in memory under a name.

  2. `extractor_test` runs the spec on one cached page so the model can see
     if the regexes are right BEFORE scaling up.

  3. `scrape_paginated` fetches N listing pages and applies the spec to each,
     auto-deduplicating and queueing items. One LLM call replaces ~30 turns.

  4. The model writes a second spec for the DETAIL page (find the attachment
     URL, optionally other scalar fields).

  5. `process_queue` pops K queued items, fetches each detail page, applies
     the detail spec, downloads the attachment, runs extract_text, builds the
     row (with $file references so the LLM never round-trips the text), and
     calls dataset_append. One LLM call processes K items.

Everything is stdlib regex + httpx; no JS, no XPath. For pages that need
something more flexible, the existing `python` tool is the escape hatch.

Spec format (listing):
    {
      "row_pattern":      "regex that matches one repeating block (DOTALL)",
      "row_group":        1,              # optional, default whole match
      "include_substring": "must contain",  # optional
      "exclude_substring": "must NOT contain",  # optional
      "base_url":         "https://example.com",  # optional
      "fields": {
        "id":          {"regex": "...", "group": 1, "transform": "strip"},
        "detail_url":  {"regex": "href=\"([^\"]+)\"", "prefix_base": true},
        "title":       {"regex": "...", "fallback_regex": "..."},
        "date":        {"regex": "datetime=\"([^\"T]+)"}
      }
    }

Spec format (detail) — same shape, but `row_pattern` is omitted; the
extractor treats the whole page as one row.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from gemma_miner.tools.base import Tool, ToolResult
from gemma_miner.tools.extract_text_tool import _extract_bytes, _sniff
from gemma_miner.tools.http_tool import (
    _ext_from_bytes,
    _ext_from_ctype,
    _ext_from_url,
    _slug,
)

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


# ── spec application ────────────────────────────────────────────────────────


def _html_unescape(s: str) -> str:
    import html

    return html.unescape(s)


def _strip_html(s: str) -> str:
    """Remove tags + collapse whitespace + unescape entities. Useful for fields
    captured from HTML-wrapped containers."""
    if not isinstance(s, str):
        return s
    import html as _h

    s = re.sub(r"<[^>]+>", " ", s)
    s = _h.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


_TRANSFORMS = {
    "strip": lambda s: s.strip() if isinstance(s, str) else s,
    # JS / common-name aliases for strip.
    "trim":      lambda s: s.strip() if isinstance(s, str) else s,
    "clean":     lambda s: s.strip() if isinstance(s, str) else s,
    "whitespace": lambda s: re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s,
    "lower": lambda s: s.lower() if isinstance(s, str) else s,
    "upper": lambda s: s.upper() if isinstance(s, str) else s,
    "id_normalize": lambda s: re.sub(r"[^\w]+", "", s).upper() if isinstance(s, str) else s,
    "first_line": lambda s: (s.splitlines()[0] if isinstance(s, str) and s.splitlines() else s),
    "html_unescape": lambda s: _html_unescape(s) if isinstance(s, str) else s,
    # Aliases the model invents in the wild — all map to the same operation.
    "strip_html": _strip_html,
    "html_strip": _strip_html,
    "strip_tags": _strip_html,
    "text":       _strip_html,
    "plain":      _strip_html,
    # No-op (common pattern: model sets transform="none" to mean "leave as-is").
    "none":       lambda s: s,
    "identity":   lambda s: s,
    "raw":        lambda s: s,
    # Type-coercion aliases the model invents. The actual coercion happens
    # downstream (codebook or dataset writer); here they're no-ops.
    "integer":    lambda s: s,
    "int":        lambda s: s,
    "number":     lambda s: s,
    "float":      lambda s: s,
    "decimal":    lambda s: s,
    "boolean":    lambda s: s,
    "bool":       lambda s: s,
    "date":       lambda s: s,
    "string":     lambda s: s,
    "str":        lambda s: s,
}


def _apply_field(row: str, conf: dict, base_url: str) -> Any:
    main_re = conf.get("regex")
    if not main_re:
        return None
    # `multi` (alias: multiple / find_all / list / all) → return EVERY match
    # as a list, instead of just the first. Essential for list-valued fields
    # like tags, categories, authors.
    multi = bool(
        conf.get("multi") or conf.get("multiple")
        or conf.get("find_all") or conf.get("list") or conf.get("all")
    )

    def _post(val: Any) -> Any:
        if val is None:
            return None
        if conf.get("prefix_base") and base_url and isinstance(val, str) and val.startswith("/"):
            val = base_url.rstrip("/") + val
        tform = conf.get("transform")
        if tform:
            if tform in _TRANSFORMS:
                val = _TRANSFORMS[tform](val)
            else:
                if isinstance(val, str):
                    val = val.strip()
                conf["_unknown_transform"] = tform
        return val

    if multi:
        try:
            matches = re.findall(main_re, row, re.DOTALL)
        except re.error as e:
            return f"<regex error: {e}>"
        if not matches and conf.get("fallback_regex"):
            try:
                matches = re.findall(conf["fallback_regex"], row, re.DOTALL)
            except re.error:
                pass
        if not matches:
            return []
        # If the regex has one capture group, findall returns strings; if
        # multiple, tuples. Take the configured `group` from each tuple.
        grp = conf.get("group", 1)
        out: list = []
        for m_val in matches:
            if isinstance(m_val, tuple):
                try:
                    val = m_val[grp - 1] if 1 <= grp <= len(m_val) else m_val[0]
                except IndexError:
                    val = m_val[0]
            else:
                val = m_val
            val = _post(val)
            if val is not None and val != "":
                out.append(val)
        return out

    try:
        m = re.search(main_re, row, re.DOTALL)
    except re.error as e:
        return f"<regex error: {e}>"
    if not m and conf.get("fallback_regex"):
        try:
            m = re.search(conf["fallback_regex"], row, re.DOTALL)
        except re.error:
            pass
    if not m:
        return None
    grp = conf.get("group", 1)
    try:
        val = m.group(grp) if grp <= (m.lastindex or 0) else m.group(0)
    except IndexError:
        val = m.group(0)
    return _post(val)


def apply_listing_spec(html: str, spec: dict) -> list[dict]:
    """Apply a listing extractor spec to one page; return list of row dicts."""
    rp = spec.get("row_pattern")
    if not rp:
        raise ValueError("listing spec requires 'row_pattern'")
    try:
        rx = re.compile(rp, re.DOTALL)
    except re.error as e:
        raise ValueError(f"invalid row_pattern regex: {e}") from e

    incl = spec.get("include_substring")
    excl = spec.get("exclude_substring")
    base = spec.get("base_url", "")
    row_group = spec.get("row_group")
    fields_conf = spec.get("fields") or {}

    out: list[dict] = []
    for m in rx.finditer(html):
        if row_group is not None:
            try:
                block = m.group(row_group)
            except IndexError:
                block = m.group(0)
        else:
            # If the regex has EXACTLY one capture group → use it (common case:
            # `<tr>(.*?)</tr>`). Otherwise use the whole match (m.group(0)) —
            # this handles multi-group patterns like `<dt>(.*?)</dt><dd>(.*?)</dd>`
            # without the model needing to set row_group.
            n_groups = len(m.groups())
            if n_groups == 1:
                block = m.group(1)
            else:
                block = m.group(0)
        if incl and incl not in block:
            continue
        if excl and excl in block:
            continue
        item: dict[str, Any] = {}
        for fname, fconf in fields_conf.items():
            item[fname] = _apply_field(block, fconf, base)
        out.append(item)
    return out


def apply_detail_spec(html: str, spec: dict) -> dict:
    """Apply a detail-page spec; same as listing but treats the whole page as one row."""
    base = spec.get("base_url", "")
    fields_conf = spec.get("fields") or {}
    out: dict[str, Any] = {}
    for fname, fconf in fields_conf.items():
        out[fname] = _apply_field(html, fconf, base)
    return out


# ── tool: extractor_define ─────────────────────────────────────────────────


def _first_row_block(html: str, spec: dict) -> str | None:
    """Return the raw HTML of the first row matched by spec.row_pattern."""
    rp = spec.get("row_pattern")
    if not rp:
        return None
    try:
        rx = re.compile(rp, re.DOTALL)
    except re.error:
        return None
    incl = spec.get("include_substring")
    excl = spec.get("exclude_substring")
    row_group = spec.get("row_group")
    for m in rx.finditer(html):
        if row_group is not None:
            try:
                block = m.group(row_group)
            except IndexError:
                block = m.group(0)
        else:
            block = m.group(1) if m.lastindex else m.group(0)
        if incl and incl not in block:
            continue
        if excl and excl in block:
            continue
        return block
    return None


def _diagnose_fields(row: dict) -> tuple[list[str], dict[str, str]]:
    """Detect null fields and field values that look suspicious."""
    null_fields: list[str] = []
    suspicious: dict[str, str] = {}
    for k, v in row.items():
        if v is None or v == "":
            null_fields.append(k)
            continue
        if isinstance(v, str):
            lk = k.lower()
            if "url" in lk and ("?" in v and ("field_sector" in v or "filter" in v or "search=" in v)):
                suspicious[k] = (
                    "looks like a sector/filter URL, not a detail URL. "
                    "Try targeting the `about=\"...\"` attribute or an "
                    "<h2><a> link instead."
                )
            elif "url" in lk and v.startswith("/"):
                suspicious[k] = (
                    "URL starts with '/' — likely missing the base. "
                    "Set prefix_base=true on this field and ensure spec.base_url is set."
                )
            elif "date" in lk and "T" in v and ":" in v:
                # 2026-04-16T12:00:00Z → keep YYYY-MM-DD only
                suspicious[k] = (
                    "looks like a full ISO timestamp. If the goal is YYYY-MM-DD, "
                    "tighten the regex (e.g. `datetime=\"([^\"T]+)`)."
                )
            elif "id" in lk and len(v) <= 5 and ":" not in v:
                # short numeric ID is fine; flag only if it looks like a substring fragment
                if re.fullmatch(r"\d{2}T\d{2}", v):
                    suspicious[k] = (
                        f"value '{v}' looks like a fragment of a timestamp, not an "
                        "ID. Your regex matched the wrong part of the row."
                    )
    return null_fields, suspicious


def _newest_html_in_cache(state: "AgentState") -> Path | None:
    cache_dir = Path(state.workdir) / "cache"
    if not cache_dir.exists():
        return None
    htmls = [p for p in cache_dir.glob("*.html") if p.is_file()]
    if not htmls:
        return None
    htmls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return htmls[0]


def _is_naive_td_capture(rx: str) -> bool:
    """True when `rx` looks like '<td...>...(capture)...</td>' — i.e. a
    single-<td>-block regex with one capture group, which always matches
    the FIRST <td> of any row block."""
    s = (rx or "").strip()
    if not re.match(r"^<\s*td\b", s, re.IGNORECASE):
        return False
    if not re.search(r"<\s*/\s*td\s*>\s*$", s, re.IGNORECASE):
        return False
    # Count parens. Must have at least one capture group; multiple non-cap
    # groups (`(?:...)`) are fine.
    return "(" in s and ")" in s


def _try_table_positional_autofix(
    spec: dict, html: str, field_names: list[str]
) -> tuple[list[dict], dict] | None:
    """When ≥2 fields look like `<td...>(...)</td>` (each grabbing the FIRST
    <td>), rewrite each field's regex to anchor on its declared position
    in the table — field K skips K leading <td> blocks.

    Returns (new_rows, new_spec) if the rewrite produces rows where the
    previously-duplicated fields no longer match. Returns None if the
    rewrite isn't applicable or doesn't actually help.
    """
    fields_conf = spec.get("fields") or {}
    if not isinstance(fields_conf, dict):
        return None
    candidates: list[str] = []
    for fname in field_names:
        fc = fields_conf.get(fname)
        if not isinstance(fc, dict):
            continue
        if _is_naive_td_capture(fc.get("regex", "")):
            candidates.append(fname)
    if len(candidates) < 2:
        return None
    new_fields = {k: dict(v) for k, v in fields_conf.items()
                  if isinstance(v, dict)}
    # Reasonable default capture: anything up to the closing </td>.
    default_capture = r"([^<]+)"
    for idx, fname in enumerate(candidates):
        # Try to preserve the user's inner capture if it's simple, else
        # fall back to `[^<]+`.
        original = new_fields[fname]["regex"]
        # Find the FIRST balanced `(...)` group in the original regex.
        cap_match = re.search(r"\([^()]*\)", original)
        capture = cap_match.group(0) if cap_match else default_capture
        if idx == 0:
            anchored = rf"<td[^>]*>{capture}</td>"
        else:
            anchored = (
                rf"(?:<td[^>]*>[^<]*</td>\s*){{{idx}}}<td[^>]*>{capture}</td>"
            )
        new_fields[fname]["regex"] = anchored
        new_fields[fname]["group"] = 1
        # Default to strip_html so HTML tags inside cells don't pollute
        # the captured value.
        new_fields[fname].setdefault("transform", "strip_html")
    new_spec = dict(spec)
    new_spec["fields"] = new_fields
    try:
        new_rows = apply_listing_spec(html, new_spec)
    except Exception:  # noqa: BLE001
        return None
    if not new_rows:
        return None
    r0 = new_rows[0]
    distinct_values = {r0.get(f) for f in candidates if r0.get(f) not in (None, "")}
    if len(distinct_values) < min(2, len(candidates)):
        return None
    return new_rows, new_spec


class ExtractorDefineTool(Tool):
    name = "extractor_define"
    summary_fields = ("saved", "matched_rows")
    description = (
        "Save an extraction spec AND IMMEDIATELY test it against the most "
        "recently cached HTML page (or the path you pass in `test_path`). "
        "The output shows the spec was saved AND the first 3 rows it "
        "produced — so one tool call is all you need to validate a spec.\n\n"
        "Spec kinds:\n"
        "  • LISTING — has `row_pattern`; slices a page into N row blocks. "
        "Each block is queried by per-field regexes.\n"
        "  • DETAIL — no `row_pattern`; treats the whole page as one row.\n\n"
        "Fields support: regex, group, fallback_regex, prefix_base (prepend "
        "spec.base_url to relative paths), transform (strip, lower, upper, "
        "id_normalize, first_line, html_unescape, **strip_html** — drops "
        "inner tags and collapses whitespace; alias: html_strip, strip_tags, text).\n\n"
        "Just write a spec from what you saw in the body preview and "
        "submit — don't over-investigate. The auto-test gives you instant "
        "feedback on whether it works."
    )
    args_schema = {
        "name": {"type": "string", "description": "Name to remember this spec under."},
        "spec": {"type": "object", "description": "The spec dict."},
        "test_path": {
            "type": "string",
            "description": "Optional path to a cached HTML file to test against. If omitted, the most recently cached HTML is used.",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        name = args.get("name")
        spec = args.get("spec")
        if not name or not isinstance(spec, dict):
            return ToolResult(output="ERROR: 'name' (str) and 'spec' (object) required", error=True)

        # Validate spec keys: hard-fail on typos that change MEANING (e.g.
        # `row_regex` → not used, agent thinks row_pattern is set), but
        # silently drop benign extras (e.g. `options`, `notes`) with a soft
        # warning in the output so the model doesn't get stuck.
        valid_keys = {
            "row_pattern", "row_group", "include_substring", "exclude_substring",
            "base_url", "fields",
        }
        # Critical typos: keys that LOOK like a valid key but aren't. These
        # change spec semantics — must fail loudly.
        critical_typos = {
            "row_regex":     "row_pattern",
            "rowpattern":    "row_pattern",
            "rows":          "row_pattern",
            "regex":         "row_pattern",
            "pattern":       "row_pattern",
            "include":       "include_substring",
            "exclude":       "exclude_substring",
            "url":           "base_url",
            "base":          "base_url",
        }
        unknown = set(spec.keys()) - valid_keys
        critical = unknown & set(critical_typos)
        if critical:
            suggestions = ", ".join(f"{k} → {critical_typos[k]}" for k in sorted(critical))
            return ToolResult(
                output=(
                    f"ERROR: unknown spec key(s) that look like typos: {sorted(critical)}. "
                    f"Did you mean: {suggestions}? Valid keys: {sorted(valid_keys)}."
                ),
                error=True,
            )
        # Strip the benign extras silently — emit a soft note in the output.
        dropped: list[str] = []
        for k in list(spec.keys()):
            if k not in valid_keys:
                dropped.append(k)
                spec = {kk: vv for kk, vv in spec.items() if kk != k}

        # Validate compilable regexes up front.
        if spec.get("row_pattern"):
            try:
                re.compile(spec["row_pattern"], re.DOTALL)
            except re.error as e:
                return ToolResult(output=f"ERROR: invalid row_pattern: {e}", error=True)
        fields_conf = spec.get("fields") or {}
        # Accept LIST form too: [{"name": "x", "regex": "..."}, ...]
        # Small models reach for this shape constantly. Normalize to dict.
        if isinstance(fields_conf, list):
            normalized: dict[str, dict] = {}
            for entry in fields_conf:
                if not isinstance(entry, dict):
                    return ToolResult(
                        output=f"ERROR: each entry in fields list must be an object, got {type(entry).__name__}",
                        error=True,
                    )
                fname = entry.get("name") or entry.get("field") or entry.get("key")
                if not fname:
                    return ToolResult(
                        output="ERROR: each entry in the fields list needs a 'name'.",
                        error=True,
                    )
                normalized[fname] = {k: v for k, v in entry.items()
                                       if k not in ("name", "field", "key")}
            fields_conf = normalized
            spec["fields"] = fields_conf
        if not isinstance(fields_conf, dict):
            return ToolResult(
                output=(
                    f"ERROR: 'fields' must be an object (dict of name → config) "
                    f"or a list of {{name, regex, ...}} objects. Got {type(fields_conf).__name__}."
                ),
                error=True,
            )
        if not fields_conf:
            return ToolResult(
                output=(
                    "ERROR: spec has no 'fields'. A spec must include at least "
                    "one field. Example: {\"fields\": {\"id\": {\"regex\": \"...\"}}}"
                ),
                error=True,
            )
        valid_field_keys = {"regex", "group", "fallback_regex", "prefix_base",
                              "transform", "multi"}
        # Friendly aliases — silently rename to the canonical key.
        field_key_aliases = {
            "group_index":   "group",
            "group_num":     "group",
            "capture_group": "group",
            "groupId":       "group",
            "pattern":       "regex",
            "re":            "regex",
            "regexp":        "regex",
            "fallback":      "fallback_regex",
            "alt_regex":     "fallback_regex",
            "fallback_pattern": "fallback_regex",
            "base_prefix":   "prefix_base",
            "with_base":     "prefix_base",
            "prepend_base":  "prefix_base",
            "post_process":  "transform",
            "postprocess":   "transform",
            "multiple":      "multi",
            "find_all":      "multi",
            "findall":       "multi",
            "list":          "multi",
            "all":           "multi",
            "many":          "multi",
        }
        # Reject CSS-selector-style field configs with a helpful pointer.
        # The model often writes `{"selector": "td:nth-child(1)"}`, which is
        # not what this regex-based extractor takes. Surface that clearly
        # instead of an opaque "unknown key" error or a bad regex.
        css_selector_keys = {"selector", "css", "css_selector", "xpath"}
        # Benign extras the model frequently invents — silently strip with a note.
        benign_field_keys = {"type", "description", "unit", "enum_values",
                              "min_value", "max_value", "required"}
        valid_transforms = set(_TRANSFORMS.keys())
        stripped_field_keys: dict[str, list[str]] = {}
        for fname, fconf in fields_conf.items():
            if not isinstance(fconf, dict):
                return ToolResult(
                    output=f"ERROR: field '{fname}' must be an object, got {type(fconf).__name__}",
                    error=True,
                )
            # Rewrite friendly aliases to canonical keys.
            renamed_keys: list[str] = []
            for old in list(fconf.keys()):
                if old in field_key_aliases:
                    new = field_key_aliases[old]
                    if new not in fconf:
                        fconf[new] = fconf[old]
                        renamed_keys.append(f"{old}→{new}")
                    fconf.pop(old, None)
            # Strip benign extras silently.
            extras = {k: v for k, v in fconf.items() if k in benign_field_keys}
            if extras:
                for k in extras:
                    fconf.pop(k, None)
                stripped_field_keys[fname] = list(extras.keys())
            if renamed_keys:
                stripped_field_keys.setdefault(fname, []).extend(renamed_keys)
            fk_unknown = set(fconf.keys()) - valid_field_keys
            if fk_unknown:
                if fk_unknown & css_selector_keys:
                    return ToolResult(
                        output=(
                            f"ERROR: field '{fname}' uses CSS-selector keys "
                            f"({sorted(fk_unknown & css_selector_keys)}), but "
                            "this extractor is REGEX-based, not CSS-based.\n\n"
                            "TWO good ways to recover:\n"
                            "  1. (FAST) Switch to `llm_scrape(source=<cache_path>, "
                            "fields=[{'name': '...'}], target=<N>, push_to_dataset=true)`. "
                            "The LLM reads the page and produces rows — no regex needed. "
                            "Best for sites with lots of nested HTML, list-valued fields, "
                            "or markup the regex spec can't express cleanly.\n"
                            "  2. (DETERMINISTIC) Re-write each field as a regex on the "
                            "row's raw HTML. For the Nth <td>, use "
                            '`<td[^>]*>(.*?)</td>` (column 1) or '
                            '`<td[^>]*>[^<]*</td>\\s*<td[^>]*>(.*?)</td>` (column 2). '
                            "For list-valued fields (tags/categories/authors), set "
                            "`multi: true` to return every match as a list."
                        ),
                        error=True,
                    )
                return ToolResult(
                    output=(
                        f"ERROR: field '{fname}' has unknown key(s): {sorted(fk_unknown)}. "
                        f"Valid: {sorted(valid_field_keys)}."
                    ),
                    error=True,
                )
            if not fconf.get("regex"):
                return ToolResult(
                    output=f"ERROR: field '{fname}' missing 'regex'",
                    error=True,
                )
            try:
                re.compile(fconf["regex"], re.DOTALL)
            except re.error as e:
                return ToolResult(
                    output=f"ERROR: invalid regex for field '{fname}': {e}",
                    error=True,
                )
            t = fconf.get("transform")
            if t and t not in valid_transforms:
                return ToolResult(
                    output=(
                        f"ERROR: field '{fname}' has unknown transform '{t}'. "
                        f"Valid: {sorted(valid_transforms)}. "
                        "TIP: for HTML-wrapped text, use 'strip_html' to drop "
                        "inner tags and collapse whitespace."
                    ),
                    error=True,
                )

        # Save.
        extractors = state.memory.get("extractors", {}) or {}
        extractors[name] = spec
        state.memory.set("extractors", extractors)
        kind = "listing" if spec.get("row_pattern") else "detail"

        # Auto-test.
        path_arg = args.get("test_path")
        test_path: Path | None
        if path_arg:
            p = Path(path_arg)
            test_path = p if p.is_absolute() else Path(state.workdir) / p
        else:
            test_path = _newest_html_in_cache(state)

        lines = [
            f"saved: name='{name}' kind={kind} fields={list(fields_conf.keys())}",
        ]
        if dropped:
            lines.append(f"NOTE: ignored unknown spec keys (not errors): {dropped}")
        if stripped_field_keys:
            lines.append(
                "NOTE: stripped codebook-style keys from field configs "
                f"(extractor doesn't need them): {stripped_field_keys}"
            )
        if test_path is None or not test_path.exists():
            lines.append("(no cached HTML found to auto-test against)")
            return ToolResult(output="\n".join(lines))

        try:
            html = test_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            lines.append(f"could not read test file: {e}")
            return ToolResult(output="\n".join(lines))

        lines.append(f"auto-tested on: {test_path.name}")
        try:
            if spec.get("row_pattern"):
                # Apply spec.
                rows = apply_listing_spec(html, spec)
                # Always re-extract the raw FIRST row block too — the model
                # uses this to diagnose bad field regexes.
                raw_first = _first_row_block(html, spec)
                lines.append(f"matched_rows: {len(rows)}")
                if len(rows) == 0:
                    diag_extra = ""
                    # Auto-rewrite: try a LOOSE version of the row_pattern in
                    # which every `class="EXACT"` is converted to
                    # `class="[^"]*\bEXACT\b[^"]*"`. If THAT matches, save the
                    # loose spec automatically — the agent stops thrashing.
                    original_rp = spec.get("row_pattern", "") or ""

                    def _make_loose(rp: str) -> str:
                        def _swap(m):
                            tokens = m.group(1).split()
                            if not tokens:
                                return m.group(0)
                            # Build a class regex that requires EACH token loosely
                            parts = [
                                f"(?=[^\\\"]*\\b{re.escape(t)}\\b)" for t in tokens
                            ]
                            return f'class="{"".join(parts)}[^"]*"'
                        return re.sub(r'class="([^"]+)"', _swap, rp)

                    loose_rp = _make_loose(original_rp)
                    auto_fixed = False
                    if loose_rp != original_rp:
                        try:
                            loose_spec = dict(spec)
                            loose_spec["row_pattern"] = loose_rp
                            loose_rows = apply_listing_spec(html, loose_spec)
                        except Exception:  # noqa: BLE001
                            loose_rows = []
                        if loose_rows:
                            # Save the FIXED spec, replacing the broken one.
                            extractors[name] = loose_spec
                            state.memory.set("extractors", extractors)
                            auto_fixed = True
                            lines.append(
                                f"AUTO-FIXED: your row_pattern used exact "
                                f"class strings that didn't match (HTML elements "
                                f"often have multiple class tokens). I rewrote "
                                f"every `class=\"X\"` in row_pattern to a loose "
                                f"`class=\"[^\\\"]*\\bX\\b[^\\\"]*\"` and "
                                f"matched {len(loose_rows)} rows. Spec saved with "
                                f"the loose row_pattern."
                            )
                            # Now show the rows like a normal success.
                            lines.append(f"matched_rows: {len(loose_rows)}")
                            lines.append("--- row 0 (extracted) ---")
                            lines.append(json.dumps(loose_rows[0], ensure_ascii=False, indent=2))
                            if len(loose_rows) > 1:
                                lines.append("--- row 1 (extracted) ---")
                                lines.append(json.dumps(loose_rows[1], ensure_ascii=False, indent=2))
                            null_fields, suspicious = _diagnose_fields(loose_rows[0])
                            if null_fields:
                                lines.append(f"NULL fields in row 0: {null_fields}")
                            return ToolResult(output="\n".join(lines))
                    # Couldn't auto-fix — produce the educated diagnostic.
                    m = re.search(r'class="([^"]+?)"', original_rp)
                    if m:
                        cls_tokens = m.group(1).split()
                        actual_classes = []
                        for ct in cls_tokens:
                            n = len(re.findall(rf'class="[^"]*\b{re.escape(ct)}\b[^"]*"', html))
                            actual_classes.append((ct, n))
                        diag_extra = (
                            "\n  Class tokens in row_pattern vs page hits: "
                            + ", ".join(f"{c}={n}" for c, n in actual_classes)
                        )
                    lines.append(
                        "DIAGNOSIS: 0 rows. Causes (most common first):\n"
                        "  1. row_pattern doesn't match — try a simpler pattern.\n"
                        "  2. include_substring is set and doesn't appear in any block.\n"
                        "  3. exclude_substring is set and matches every block."
                        + diag_extra
                    )
                else:
                    lines.append("--- row 0 (extracted) ---")
                    lines.append(json.dumps(rows[0], ensure_ascii=False, indent=2))
                    if len(rows) > 1:
                        lines.append("--- row 1 (extracted) ---")
                        lines.append(json.dumps(rows[1], ensure_ascii=False, indent=2))
                    # PER-FIELD COVERAGE across ALL matched rows. Row 0/1
                    # are not representative when a page has multiple row
                    # shapes (e.g. major vs simplified sanctions on the CNIL
                    # page). Show the facts; the agent decides.
                    coverage_lines: list[str] = []
                    if rows and isinstance(rows[0], dict):
                        n_total = len(rows)
                        for k in rows[0].keys():
                            if str(k).startswith("_"):
                                continue
                            n_set = sum(
                                1 for r in rows
                                if isinstance(r, dict) and r.get(k) not in (None, "")
                            )
                            pct = n_set / n_total
                            coverage_lines.append(
                                f"  {k:<24} {pct:>5.0%}  ({n_set}/{n_total})"
                            )
                    if coverage_lines:
                        lines.append("")
                        lines.append("per-field coverage across all matched rows:")
                        lines.extend(coverage_lines)
                    # Find pairs of fields that capture IDENTICAL values across
                    # multiple sampled rows. That pattern is almost always a
                    # sign each field's regex matches the same column (e.g.
                    # the first <td>) instead of its own column.
                    sample = [r for r in rows[:5] if isinstance(r, dict)]
                    field_names = [
                        k for k in sample[0].keys()
                        if not k.startswith("_") and k != "id"
                        and "url" not in k.lower()
                    ] if sample else []
                    duplicate_pairs: list[tuple[str, str, str]] = []
                    for i in range(len(field_names)):
                        for j in range(i + 1, len(field_names)):
                            a, b = field_names[i], field_names[j]
                            matched = 0
                            for r in sample:
                                va, vb = r.get(a), r.get(b)
                                if va in (None, "") or vb in (None, ""):
                                    matched = -1
                                    break
                                if va == vb:
                                    matched += 1
                            if matched >= 2 and matched == len(sample):
                                duplicate_pairs.append((a, b, sample[0].get(a)))
                    if duplicate_pairs:
                        # AUTO-FIX: when the duplication is the classic
                        # "every field used the same <td...>(...)</td> regex"
                        # pattern, rewrite each field's regex with a positional
                        # anchor and re-test. If the rewrite produces distinct
                        # values, save THAT spec instead of rejecting.
                        auto_fixed_rows = _try_table_positional_autofix(
                            spec, html, field_names,
                        )
                        if auto_fixed_rows is not None:
                            fixed_rows, fixed_spec = auto_fixed_rows
                            extractors[name] = fixed_spec
                            state.memory.set("extractors", extractors)
                            lines.append("")
                            lines.append(
                                "AUTO-FIXED: every field used the same "
                                "`<td>(.*?)</td>` regex, so each was capturing "
                                "the first column. I rewrote each field's "
                                "regex with a positional <td> anchor (field "
                                "K gets the K-th column) and re-tested."
                            )
                            lines.append(f"matched_rows: {len(fixed_rows)}")
                            lines.append("--- row 0 (after auto-fix) ---")
                            lines.append(json.dumps(fixed_rows[0], ensure_ascii=False, indent=2))
                            if len(fixed_rows) > 1:
                                lines.append("--- row 1 (after auto-fix) ---")
                                lines.append(json.dumps(fixed_rows[1], ensure_ascii=False, indent=2))
                            return ToolResult(output="\n".join(lines))
                        # Couldn't auto-fix — roll back and ask the model to
                        # write a real positional spec.
                        extractors.pop(name, None)
                        state.memory.set("extractors", extractors)
                        pair_str = ", ".join(
                            f"`{a}` == `{b}` (={v!r})" for a, b, v in duplicate_pairs[:4]
                        )
                        return ToolResult(
                            output=(
                                "\n".join(lines)
                                + f"\n\nREJECTED: these fields captured the "
                                f"SAME value across every sampled row: "
                                f"{pair_str}.\n"
                                "Each field's regex is matching the same "
                                "column instead of its own. For column N, "
                                "use a positional anchor:\n"
                                "  column 1: <td[^>]*>(.*?)</td>\n"
                                "  column 2: <td[^>]*>[^<]*</td>\\s*<td[^>]*>(.*?)</td>\n"
                                "  column 3: (?:<td[^>]*>[^<]*</td>\\s*){2}<td[^>]*>(.*?)</td>\n"
                                "Spec NOT saved — call extractor_define again."
                            ),
                            error=True,
                        )
                    # Field-level diagnostics.
                    null_fields, suspicious = _diagnose_fields(rows[0])
                    if null_fields or suspicious:
                        lines.append("")
                        if null_fields:
                            lines.append(f"NULL fields in row 0: {null_fields}")
                        if suspicious:
                            for f, why in suspicious.items():
                                lines.append(f"SUSPICIOUS '{f}': {why}")
                        lines.append(
                            "→ Examine the raw row HTML below and rewrite the "
                            "field regexes to target the SPECIFIC elements you "
                            "want (e.g. for 'detail_url' on a Drupal Views site, "
                            "use the `about=\"...\"` attribute instead of "
                            "`href=\"...\"`, because the first href is usually "
                            "a sector/filter link)."
                        )
                    if raw_first:
                        lines.append("")
                        lines.append("--- RAW HTML of row 0 (use this to fix field regexes) ---")
                        lines.append(raw_first[:2500])
            else:
                result = apply_detail_spec(html, spec)
                lines.append("detail extracted:")
                lines.append(json.dumps(result, ensure_ascii=False, indent=2))
                null_fields, suspicious = _diagnose_fields(result)
                if null_fields:
                    lines.append(f"NULL fields: {null_fields}")
                for f, why in suspicious.items():
                    lines.append(f"SUSPICIOUS '{f}': {why}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"TEST ERROR: {e}")

        return ToolResult(output="\n".join(lines))


# ── tool: extractor_test ───────────────────────────────────────────────────


class ExtractorTestTool(Tool):
    name = "extractor_test"
    description = (
        "Run a stored extractor against an HTML file in the cache and show "
        "the first 3 rows so you can verify the spec is correct BEFORE "
        "scraping many pages."
    )
    args_schema = {
        "name": {"type": "string"},
        "source_path": {"type": "string", "description": "Path to a cached HTML file."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        name = args.get("name")
        path = args.get("source_path")
        if not name or not path:
            return ToolResult(output="ERROR: 'name' and 'source_path' required", error=True)
        extractors = state.memory.get("extractors", {}) or {}
        spec = extractors.get(name)
        if spec is None:
            return ToolResult(output=f"ERROR: no extractor named '{name}'", error=True)
        p = Path(path)
        if not p.is_absolute():
            p = Path(state.workdir) / p
        if not p.exists():
            return ToolResult(output=f"ERROR: not found: {p}", error=True)
        html = p.read_text(encoding="utf-8", errors="replace")
        if spec.get("row_pattern"):
            rows = apply_listing_spec(html, spec)
            out = [f"matched_rows: {len(rows)}"]
            for i, r in enumerate(rows[:3]):
                out.append(f"--- row {i} ---")
                out.append(json.dumps(r, ensure_ascii=False, indent=2))
            if not rows:
                out.append("(0 rows — check row_pattern and include/exclude substrings)")
            return ToolResult(output="\n".join(out), artifact={"n_rows": len(rows)})
        result = apply_detail_spec(html, spec)
        return ToolResult(
            output=f"detail extracted:\n{json.dumps(result, ensure_ascii=False, indent=2)}",
            artifact=result,
        )


# ── tool: scrape_paginated ─────────────────────────────────────────────────


def _http_get(
    url: str,
    cache_dir: Path,
    *,
    headers: dict | None = None,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> tuple[Path, bytes, str]:
    """Cached HTTP GET with retry. Returns (cache_path, content, content_type).

    Retries: 5xx + 429 use exponential backoff (1s → 4s → 16s → 60s) and respect
    `Retry-After` when present. 4xx other than 429 are NOT retried (the server
    has refused — retrying won't help, fail fast so the caller can pivot).
    """
    slug = _slug(url)
    for existing in cache_dir.glob(f"{slug}.*"):
        return existing.resolve(), existing.read_bytes(), ""
    h = {"User-Agent": "Mozilla/5.0 (gemma-miner research agent)"}
    if headers:
        h.update(headers)

    last_err: Exception | None = None
    content: bytes | None = None
    ctype: str = ""
    for attempt in range(max_retries):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                r = client.get(url, headers=h)
                if r.status_code == 429 or r.status_code >= 500:
                    # Respect Retry-After if the server sends one.
                    retry_after = r.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = min(60.0, float(retry_after))
                            time.sleep(wait)
                            last_err = httpx.HTTPStatusError(
                                f"{r.status_code} (retry-after honored)",
                                request=r.request, response=r,
                            )
                            continue
                        except ValueError:
                            pass
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}", request=r.request, response=r
                    )
                if 400 <= r.status_code < 500:
                    # Non-retryable client error — surface immediately.
                    raise RuntimeError(
                        f"http_get got {r.status_code} from {url} "
                        f"(not retryable). Pick a different URL or stop "
                        "fetching this domain."
                    )
                content = r.content
                ctype = r.headers.get("content-type", "")
                break
        except RuntimeError:
            raise
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:
            last_err = e
            wait = min(60.0, (4 ** attempt))  # 1, 4, 16, 60
            time.sleep(wait)
            continue
    else:
        raise RuntimeError(f"http_get failed after {max_retries} attempts: {last_err}")

    if content is None:
        raise RuntimeError(f"http_get failed after {max_retries} attempts: {last_err}")

    ext = _ext_from_ctype(ctype) or _ext_from_url(url) or _ext_from_bytes(content) or ".bin"
    cache_path = (cache_dir / f"{slug}{ext}").resolve()
    cache_path.write_bytes(content)
    return cache_path, content, ctype


class ScrapePaginatedTool(Tool):
    name = "scrape_paginated"
    max_output_chars = 5_000  # progress log; detail stays in queue/dataset
    summary_fields = ("queue_len", "total_added")
    description = (
        "Apply a stored LISTING extractor to many paginated URLs and add "
        "every extracted row to the queue. The URL template uses '{page}' "
        "as a placeholder (e.g. 'https://x.com/list?page={page}'). "
        "Iteration stops when EITHER target_count items are in the queue OR "
        "a fetched page yields 0 new items OR max_pages is reached. "
        "Items are deduplicated against the existing queue and processed "
        "list by their 'id' field — running the tool twice is safe."
    )
    args_schema = {
        "url_template": {
            "type": "string",
            "description": (
                "URL with literal '{page}' placeholder. For sites with no "
                "pagination (the whole table is on one URL), simply pass "
                "the URL with no placeholder — the tool will fetch once and stop."
            ),
        },
        "extractor_name": {"type": "string"},
        "start_page": {"type": "integer", "default": 0},
        "max_pages": {"type": "integer", "default": 20},
        "target_count": {"type": "integer", "default": 100, "description": "Stop when queue has at least this many remaining."},
        "delay_ms": {"type": "integer", "default": 250, "description": "Politeness delay between pages."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        template = args.get("url_template")
        name = args.get("extractor_name")
        if not template or not name:
            return ToolResult(output="ERROR: 'url_template' and 'extractor_name' required", error=True)
        # Whether the caller wants pagination at all.
        has_page_placeholder = "{page}" in template
        if not has_page_placeholder:
            # Single-page mode: fetch once and stop. Don't auto-inject
            # ?page= because most sites that don't paginate would return
            # the same content or 404 on page=N.
            pass
        extractors = state.memory.get("extractors", {}) or {}
        spec = extractors.get(name)
        if not spec:
            return ToolResult(output=f"ERROR: no extractor named '{name}'", error=True)

        # Sanity-test the spec on the most recent cached HTML BEFORE scraping
        # 20 pages with a broken spec.
        newest = _newest_html_in_cache(state)
        if newest is not None:
            try:
                html_test = newest.read_text(encoding="utf-8", errors="replace")
                test_rows = apply_listing_spec(html_test, spec)
                if not test_rows:
                    return ToolResult(
                        output=(
                            f"REFUSED: extractor '{name}' produced 0 rows on the "
                            f"most recent cached page ({newest.name}). Fix the spec "
                            "first (call extractor_define again) — running this "
                            "across many pages with a broken spec would waste time."
                        ),
                        error=True,
                    )
                # Detect "every row's URL is a filter URL" — wrong field regex.
                bad_urls = sum(
                    1 for r in test_rows[:5]
                    if isinstance(r.get("detail_url"), str)
                    and (
                        "field_sector" in r["detail_url"]
                        or "?filter=" in r["detail_url"]
                    )
                )
                if bad_urls >= 2:
                    return ToolResult(
                        output=(
                            f"REFUSED: extractor '{name}' is producing filter URLs "
                            "for `detail_url`, not real detail-page URLs. Fix the "
                            "regex (most Drupal sites have an `about=\"/...\"` "
                            "attribute on each row that is the canonical URL) "
                            "and re-run extractor_define."
                        ),
                        error=True,
                    )
                # Refuse if MOST fields are null/empty in the first 3 rows.
                # "Empty" means None, "", [], or {} — extractor regex
                # mismatches commonly produce empty strings or empty multi=true
                # lists, both of which look "populated" to a naive check but
                # contain no data.
                def _is_empty(v) -> bool:
                    return v is None or v == "" or v == [] or v == {}
                check_rows = test_rows[:3]
                if check_rows:
                    field_names = [k for k in check_rows[0].keys() if not k.startswith("_")]
                    if field_names:
                        all_null_fields = [
                            f for f in field_names
                            if all(_is_empty(r.get(f)) for r in check_rows)
                        ]
                        ratio_null = len(all_null_fields) / len(field_names)
                        # Trigger if >50% of fields are null across the sample.
                        if ratio_null > 0.5:
                            return ToolResult(
                                output=(
                                    f"REFUSED: extractor '{name}' returns null for "
                                    f"{len(all_null_fields)}/{len(field_names)} fields "
                                    f"({int(ratio_null*100)}%) in the first 3 rows: "
                                    f"{all_null_fields}.\n"
                                    "The row_pattern matched but those field regexes "
                                    "miss the actual content. Re-run extractor_define "
                                    "with corrected regexes — look at the RAW HTML of "
                                    "row 0 in its auto-test output and rewrite the "
                                    "failing field regexes against that markup.\n"
                                    "If the fields aren't on the listing at all (e.g. "
                                    "an abstract that lives only on the detail page), "
                                    "switch to a listing+detail strategy: keep the "
                                    "listing extractor for what IS visible, then use "
                                    "process_queue(mode='text') with a detail extractor "
                                    "for the rest."
                                ),
                                error=True,
                            )
            except Exception:  # noqa: BLE001
                pass

        start = int(args.get("start_page") or 0)
        max_pages = int(args.get("max_pages") or 20)
        target = int(args.get("target_count") or 100)
        delay = max(0, int(args.get("delay_ms") or 250)) / 1000.0

        cache_dir = Path(state.workdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        log: list[str] = []

        # Auto-detect 1-indexed sites: if start_page=0 is the default AND
        # the template has {page}, probe page=0 vs page=1 to see which has more rows.
        if has_page_placeholder and start == 0 and not args.get("start_page"):
            try:
                _, c0, _ = _http_get(template.format(page=0), cache_dir)
                rows0 = apply_listing_spec(c0.decode("utf-8", errors="replace"), spec)
                if len(rows0) == 0:
                    # page=0 empty — try page=1
                    _, c1, _ = _http_get(template.format(page=1), cache_dir)
                    rows1 = apply_listing_spec(c1.decode("utf-8", errors="replace"), spec)
                    if len(rows1) > 0:
                        start = 1
                        log.append(
                            f"auto-detected 1-indexed pagination: page=0→{len(rows0)} rows, "
                            f"page=1→{len(rows1)} rows. Overriding start_page to 1."
                        )
            except Exception:
                pass  # keep start=0 if probe fails

        # Auto-size max_pages from the plan when the caller used the default
        if not args.get("max_pages") and target > 100:
            plan = state.memory.get("plan") or {}
            plan_pages = plan.get("pages_needed")
            plan_ipp = plan.get("items_per_page")
            if isinstance(plan_pages, int) and plan_pages > max_pages:
                max_pages = plan_pages + 5  # small buffer
                log.append(f"auto-sized max_pages={max_pages} from plan.pages_needed={plan_pages}")
            elif isinstance(plan_ipp, int) and plan_ipp > 0 and target > 0:
                computed = (target // plan_ipp) + 5
                if computed > max_pages:
                    max_pages = computed
                    log.append(f"auto-sized max_pages={max_pages} from target={target}/ipp={plan_ipp}")

        queue = state.memory.get("queue", []) or []
        processed = set(str(x) for x in (state.memory.get("processed", []) or []))
        existing_ids = {str(i.get("id")) for i in queue if isinstance(i, dict) and i.get("id")}

        total_added = 0
        consecutive_zero_new = 0
        page_range = range(start, start + max_pages) if has_page_placeholder else [start]
        consecutive_fetch_errors = 0
        for page in page_range:
            url = template.format(page=page) if has_page_placeholder else template
            try:
                _path, content, _ct = _http_get(url, cache_dir)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                log.append(f"page={page} FETCH ERROR: {err}")
                consecutive_fetch_errors += 1
                # Rate-limited or transient: pause and try the next page
                # before giving up on the whole scrape.
                if ("429" in err or "Retry-After" in err) and consecutive_fetch_errors < 3:
                    log.append(f"page={page} rate-limited — pausing 10s before next page")
                    time.sleep(10.0)
                    continue
                if consecutive_fetch_errors >= 3:
                    log.append("STOP: 3 consecutive fetch errors — server is unhappy with us.")
                break
            consecutive_fetch_errors = 0
            html = content.decode("utf-8", errors="replace")
            try:
                items = apply_listing_spec(html, spec)
            except Exception as e:  # noqa: BLE001
                log.append(f"page={page} EXTRACT ERROR: {e}")
                break
            items_have_ids = any(isinstance(it, dict) and it.get("id") for it in items)
            # AUTO-ID: if the extractor doesn't produce an `id` field, dedupe
            # by a content signature (so the same row across page=0 and
            # page=1 collapses). This lets scrape_paginated work on listings
            # where the model didn't include an `id` field.
            if items and not items_have_ids:
                from gemma_miner.dataset import ensure_row_id

                for it in items:
                    if isinstance(it, dict):
                        ensure_row_id(it)
                items_have_ids = True
            new_items = []
            for it in items:
                iid = it.get("id")
                if iid is None:
                    continue
                sid = str(iid)
                if sid in existing_ids or sid in processed:
                    continue
                existing_ids.add(sid)
                new_items.append(it)
            queue.extend(new_items)
            total_added += len(new_items)
            log.append(f"page={page}  found={len(items)}  new={len(new_items)}  queue_now={len(queue)}")
            if new_items:
                consecutive_zero_new = 0
            else:
                consecutive_zero_new += 1
            if not new_items and len(items) == 0:
                log.append("0 items on this page — stopping pagination.")
                break
            if consecutive_zero_new >= 2:
                log.append(
                    f"STOP: {consecutive_zero_new} consecutive pages with 0 new items. "
                    "Either the site doesn't paginate via ?page= OR every page "
                    "returns the same content. Try a different URL pattern, or "
                    "switch to `llm_scrape(push_to_dataset=true)` on the listing "
                    "page directly."
                )
                break
            remaining = sum(1 for q in queue if str(q.get("id")) not in processed)
            if remaining >= target:
                log.append(f"target reached: remaining={remaining} >= target={target}")
                break
            if delay:
                time.sleep(delay)

        state.memory.set("queue", queue)
        if not state.memory.get("processed"):
            state.memory.set("processed", [])

        # PER-FIELD COVERAGE across the queue this sweep produced. Row 0 of
        # the autotest is not representative when the page has multiple row
        # shapes (e.g. major vs simplified sanctions: only major ones have a
        # <p><a href> structure). Show the facts; the agent decides whether
        # the regex needs another pass.
        cov_lines: list[str] = []
        if queue:
            n_total = len(queue)
            keys_seen: dict[str, int] = {}
            for q in queue:
                if isinstance(q, dict):
                    for k in q.keys():
                        if str(k).startswith("_"):
                            continue
                        if q.get(k) not in (None, ""):
                            keys_seen[k] = keys_seen.get(k, 0) + 1
                        else:
                            keys_seen.setdefault(k, 0)
            for k in sorted(keys_seen):
                n_set = keys_seen[k]
                pct = n_set / n_total
                cov_lines.append(f"  {k:<24} {pct:>5.0%}  ({n_set}/{n_total})")

        out = (
            "scrape_paginated:\n  " + "\n  ".join(log)
            + f"\n\ntotal_added: {total_added}\nqueue_len: {len(queue)}"
        )
        if cov_lines:
            out += "\n\nper-field coverage across the queue:\n" + "\n".join(cov_lines)
        return ToolResult(
            output=out,
            artifact={"added": total_added, "queue_len": len(queue)},
        )


def _harvest_assets_for_item(
    *,
    dhtml: str,
    base_url: str,
    item_dir: Path,
    cache_dir: Path,
    workdir: Path,
    max_assets: int,
    min_score: int,
    max_bytes: int,
) -> tuple[list[dict], str]:
    """For one item: scan its detail HTML, pick top-K data-asset URLs, fetch
    each, extract text. Returns (assets, combined_text) where each asset is
    a dict with: url, kind, score, anchor, bin_path, text_path, n_chars."""
    # Minimal inlined helpers (the old discover_assets_tool module was removed).
    import re as _re
    _HREF_RE = _re.compile(
        r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', _re.IGNORECASE | _re.DOTALL
    )
    def _strip_tags(s: str) -> str:
        return _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", " ", s)).strip()
    _DATA_EXT = {
        ".pdf": ("pdf", 90), ".xml": ("xml", 80), ".json": ("json", 80),
        ".csv": ("csv", 80), ".tsv": ("tsv", 75), ".xlsx": ("xlsx", 75),
        ".xls": ("xls", 70), ".docx": ("docx", 70), ".doc": ("doc", 65),
        ".zip": ("zip", 60), ".tar": ("tar", 60), ".gz": ("gz", 55),
        ".txt": ("txt", 50),
    }
    def _classify(url: str, anchor: str) -> tuple[str | None, int, str]:
        u = url.lower()
        for ext, (kind, score) in _DATA_EXT.items():
            if u.endswith(ext) or f"{ext}?" in u or f"{ext}#" in u:
                return kind, score, f"ext={ext}"
        return None, 0, ""

    cand: list[dict] = []
    seen_urls: set[str] = set()
    for m in _HREF_RE.finditer(dhtml):
        href = (m.group(1) or "").strip()
        anchor = _strip_tags(m.group(2) or "")
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        from urllib.parse import urljoin
        abs_url = urljoin(base_url, href) if base_url else href
        if abs_url in seen_urls:
            continue
        seen_urls.add(abs_url)
        kind, score, reason = _classify(abs_url, anchor)
        if not kind or score < min_score:
            continue
        cand.append({"url": abs_url, "kind": kind, "score": score, "anchor": anchor[:120]})

    cand.sort(key=lambda c: -c["score"])
    cand = cand[:max_assets]

    assets: list[dict] = []
    text_chunks: list[str] = []
    for idx, c in enumerate(cand, start=1):
        url = c["url"]
        try:
            apath, content, _ctype = _http_get(url, cache_dir)
        except Exception as e:  # noqa: BLE001
            assets.append({**c, "error": f"fetch: {type(e).__name__}: {e}"})
            continue
        if len(content) > max_bytes:
            assets.append({**c, "error": f"too large ({len(content)} > {max_bytes})"})
            continue
        # copy into the item dir
        ext = Path(apath).suffix.lower() or ".bin"
        dest_bin = item_dir / f"asset_{idx:02d}{ext}"
        try:
            shutil.copy2(apath, dest_bin)
        except Exception:  # noqa: BLE001
            dest_bin.write_bytes(content)
        try:
            text, _meta = _extract_bytes(content, dest_bin.name)
        except Exception as e:  # noqa: BLE001
            text = f"<extract error: {type(e).__name__}: {e}>"
        dest_txt = item_dir / f"asset_{idx:02d}.txt"
        dest_txt.write_text(text, encoding="utf-8")
        assets.append({
            **c,
            "bin_path":  dest_bin.relative_to(workdir).as_posix(),
            "text_path": dest_txt.relative_to(workdir).as_posix(),
            "n_chars":   len(text),
        })
        if text.strip():
            text_chunks.append(f"--- asset_{idx:02d} ({c['kind']}): {c['anchor']} ---\n{text}")

    combined = "\n\n".join(text_chunks)
    return assets, combined


# ── tool: process_queue ────────────────────────────────────────────────────


class ProcessQueueTool(Tool):
    name = "process_queue"
    max_output_chars = 5_000  # per-item progress; full results in dataset
    summary_fields = ("appended", "remaining_in_queue")
    description = (
        "Process queued items end-to-end. THREE modes (auto-detected):\n\n"
        "  ATTACHMENT — detail spec has `attachment_url`. Fetch detail → "
        "    find attachment URL → download → extract text → row with $file ref.\n"
        "  TEXT — detail spec has no `attachment_url`. Fetch detail → apply "
        "    spec → row from queue + detail fields. No download.\n"
        "  MULTI_ASSET — no `detail_extractor` given OR mode='multi_asset'. "
        "    For each item: fetch the detail page, scan it for ALL data-asset "
        "    links (PDF, XML, CSV, archive, …), download the top-K, extract "
        "    text from each. The row carries an `assets` list with paths and "
        "    a concatenated `text` field for codebook design.\n\n"
        "MULTI_ASSET is the right pick when the listing has detail pages with "
        "varied attachments (one decision PDF, an XML annex, a CSV of figures, "
        "…) and you want every variable the page exposes, not just one PDF."
    )
    args_schema = {
        "detail_extractor": {
            "type": "string",
            "description": "Stored DETAIL extractor (optional in multi_asset mode).",
        },
        "attachment_url_field": {
            "type": "string",
            "default": "attachment_url",
        },
        "mode": {
            "type": "string",
            "description": (
                "'attachment' | 'text' | 'multi_asset'. Default: auto-detect."
            ),
        },
        "row_template": {
            "type": "object",
            "description": (
                "How to build the row. Placeholders: {queue.<k>}, {detail.<k>}, "
                "{paths.attachment}, {paths.text}, {paths.item_dir}. In "
                "multi_asset mode the row gets `assets` + `text` automatically "
                "if row_template is omitted."
            ),
        },
        "max_assets_per_item": {"type": "integer", "default": 5},
        "min_asset_score":     {"type": "integer", "default": 15},
        "max_asset_bytes":     {"type": "integer", "default": 20_000_000},
        "batch_size": {"type": "integer", "default": 5},
        "delay_ms": {"type": "integer", "default": 250},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        forced_mode = (args.get("mode") or "").strip().lower() or None
        det_name = args.get("detail_extractor")
        extractors = state.memory.get("extractors", {}) or {}
        det_spec = extractors.get(det_name) if det_name else None
        att_field = args.get("attachment_url_field") or "attachment_url"

        # Auto-detect mode:
        #   - explicit `mode` wins
        #   - no detail_extractor → multi_asset (harvest every data link)
        #   - detail spec has attachment_url field → attachment
        #   - otherwise → text
        if forced_mode in ("multi_asset", "multiasset", "multi-asset", "assets"):
            mode = "multi_asset"
        elif forced_mode in ("text", "attachment"):
            mode = forced_mode
        elif det_spec is None:
            mode = "multi_asset"
        else:
            spec_field_names = set((det_spec.get("fields") or {}).keys())
            mode = "attachment" if att_field in spec_field_names else "text"

        if mode != "multi_asset":
            if not det_name:
                return ToolResult(output="ERROR: 'detail_extractor' required for attachment/text mode", error=True)
            if not det_spec:
                return ToolResult(output=f"ERROR: no extractor named '{det_name}'", error=True)

        row_tmpl = args.get("row_template")
        if not isinstance(row_tmpl, dict) or not row_tmpl:
            if mode == "multi_asset":
                # Sensible default for multi_asset: carry queue fields +
                # paths.text (concatenated) as a $file reference so codebook
                # design can read across all assets per item.
                row_tmpl = None
            else:
                return ToolResult(output="ERROR: 'row_template' (object) required", error=True)
        max_assets = int(args.get("max_assets_per_item") or 5)
        min_asset_score = int(args.get("min_asset_score") or 15)
        max_asset_bytes = int(args.get("max_asset_bytes") or 20_000_000)
        batch_size = int(args.get("batch_size") or 5)
        # Cap batch_size so a single call can't burn 100 detail fetches when
        # the first few are failing. Early-bail below will stop sooner.
        if batch_size > 20:
            batch_size = 20
        delay = max(0, int(args.get("delay_ms") or 250)) / 1000.0

        cache_dir = Path(state.workdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        items_root = Path(state.workdir) / "items"
        items_root.mkdir(parents=True, exist_ok=True)

        queue = state.memory.get("queue", []) or []
        processed_list = state.memory.get("processed", []) or []
        processed = set(str(x) for x in processed_list)

        # Decide slot numbering (continuous across calls).
        existing = sorted([p.name for p in items_root.glob("item_*") if p.is_dir()])
        next_slot = 1
        for name in existing:
            m = re.match(r"item_(\d+)", name)
            if m:
                next_slot = max(next_slot, int(m.group(1)) + 1)

        report: list[str] = []
        appended = 0
        errors = 0

        for item in queue:
            if appended + errors >= batch_size:
                break
            # Early bail: if the first 5 attempts all errored, the approach
            # isn't working — stop wasting fetches.
            if appended == 0 and errors >= 5:
                report.append(
                    f"EARLY BAIL: 5 consecutive failures, 0 successes. "
                    "Stopping this batch."
                )
                break
            if not isinstance(item, dict):
                continue
            iid = item.get("id")
            if iid is None or str(iid) in processed:
                continue
            detail_url = item.get("detail_url") or item.get("url")
            if not detail_url:
                report.append(f"{iid}: SKIP (no detail_url)")
                continue
            try:
                # 1. fetch detail page (cached)
                _dpath, dcontent, _ = _http_get(detail_url, cache_dir)
                dhtml = dcontent.decode("utf-8", errors="replace")
                # 2. apply detail spec (only if we have one)
                detail = apply_detail_spec(dhtml, det_spec) if det_spec else {}

                if mode == "multi_asset":
                    item_dir = items_root / f"item_{next_slot:04d}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    assets, combined_text = _harvest_assets_for_item(
                        dhtml=dhtml, base_url=detail_url,
                        item_dir=item_dir, cache_dir=cache_dir,
                        workdir=Path(state.workdir),
                        max_assets=max_assets,
                        min_score=min_asset_score,
                        max_bytes=max_asset_bytes,
                    )
                    if not assets:
                        report.append(
                            f"{iid}: NO assets found on detail page "
                            "(no PDF/XML/CSV/archive links scored above "
                            f"min_score={min_asset_score}). Falling back to "
                            "the detail HTML text only."
                        )
                        # Still produce a row carrying the detail page text.
                        dest_txt = item_dir / "detail.txt"
                        dest_txt.write_text(
                            re.sub(r"<[^>]+>", " ", dhtml), encoding="utf-8",
                        )
                        text_rel = dest_txt.relative_to(state.workdir).as_posix()
                    else:
                        # Concatenate every asset's text into one item-level file.
                        dest_txt = item_dir / "combined.txt"
                        dest_txt.write_text(combined_text, encoding="utf-8")
                        text_rel = dest_txt.relative_to(state.workdir).as_posix()
                    # Build the row. Default template carries queue fields +
                    # an `assets` array + a `$file:` ref to combined text.
                    if row_tmpl is None:
                        row = dict(item)
                        row["assets"] = assets
                        row["text_path"] = text_rel
                        row["text"] = {"$file": text_rel}
                        row["item_dir"] = item_dir.relative_to(state.workdir).as_posix()
                    else:
                        ctx = {
                            "queue": item,
                            "detail": detail,
                            "paths": {
                                "item_dir": item_dir.relative_to(state.workdir).as_posix(),
                                "text": text_rel,
                            },
                            "assets": assets,
                        }
                        row = _render_row(row_tmpl, ctx)
                        row.setdefault("assets", assets)
                        row.setdefault("text_path", text_rel)
                    ok, reason = state.dataset.append(row)
                    if not ok:
                        report.append(f"{iid}: APPEND REJECTED: {reason}")
                        errors += 1
                        continue
                    processed_list.append(str(iid))
                    processed.add(str(iid))
                    appended += 1
                    next_slot += 1
                    n_total_chars = sum(int(a.get("n_chars") or 0) for a in assets)
                    report.append(
                        f"{iid}: OK  assets={len(assets)}  "
                        f"total_chars={n_total_chars}"
                    )
                    if delay:
                        time.sleep(delay)
                    continue

                if mode == "attachment":
                    att_url = detail.get(att_field)
                    if not att_url:
                        report.append(f"{iid}: NO attachment_url ({att_field}) in detail")
                        errors += 1
                        continue
                    # 3. fetch attachment (cached)
                    apath, _acontent, _ = _http_get(att_url, cache_dir)
                    # 4-5. copy & extract
                    item_dir = items_root / f"item_{next_slot:04d}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    ext = Path(apath).suffix.lower() or ".bin"
                    dest_bin = item_dir / f"attachment_01{ext}"
                    shutil.copy2(apath, dest_bin)
                    data = dest_bin.read_bytes()
                    text, _meta = _extract_bytes(data, dest_bin.name)
                    dest_txt = item_dir / "attachment_01.txt"
                    dest_txt.write_text(text, encoding="utf-8")
                    paths_ctx = {
                        "attachment": dest_bin.relative_to(state.workdir).as_posix(),
                        "text": dest_txt.relative_to(state.workdir).as_posix(),
                        "item_dir": item_dir.relative_to(state.workdir).as_posix(),
                    }
                    extra_log = f"  bin={dest_bin.name}  text_chars={len(text)}"
                else:
                    # TEXT mode: refuse to harvest if the detail spec produced
                    # no fields at all — there'd be nothing useful to add
                    # beyond the queue item itself.
                    if not detail or all(v in (None, "") for v in detail.values()):
                        report.append(
                            f"{iid}: TEXT-mode detail spec returned no fields "
                            "from the page. Either the spec's regexes are "
                            "wrong, or there's nothing extra on the detail "
                            "page — in the latter case, use "
                            "`dataset_from_queue` instead."
                        )
                        errors += 1
                        continue
                    paths_ctx = {}
                    extra_log = f"  detail_fields={[k for k, v in detail.items() if v]}"
                # 6. build row from template
                ctx = {
                    "queue": item,
                    "detail": detail,
                    "paths": paths_ctx,
                }
                row = _render_row(row_tmpl, ctx)
                ok, reason = state.dataset.append(row)
                if not ok:
                    report.append(f"{iid}: APPEND REJECTED: {reason}")
                    errors += 1
                    continue
                processed_list.append(str(iid))
                processed.add(str(iid))
                appended += 1
                if mode == "attachment":
                    next_slot += 1
                report.append(f"{iid}: OK{extra_log}")
            except Exception as e:  # noqa: BLE001
                report.append(f"{iid}: ERROR {type(e).__name__}: {e}")
                errors += 1
            if delay:
                time.sleep(delay)

        state.memory.set("processed", processed_list)

        remaining = sum(1 for q in queue if str(q.get("id")) not in processed)

        # When EVERY attempt errored AND nothing was appended, the detail
        # pages are not yielding what we need. Surface a strong pointer
        # toward `dataset_from_queue` so the agent stops thrashing on the
        # detail-extractor regex.
        trailer = ""
        if appended == 0 and errors > 0:
            # Detect the common root causes so the message is concrete.
            no_att = sum(1 for r in report if "NO attachment_url" in r)
            no_text = sum(1 for r in report if "TEXT-mode detail spec returned no fields" in r)
            http_err = sum(1 for r in report if "ERROR" in r and ("403" in r or "404" in r))
            trailer = (
                f"\n\n⚠ 0 rows appended after {errors} errors (mode={mode}). "
                "The detail-page approach is failing"
            )
            if no_att:
                trailer += f" — {no_att} items have no attachment_url in the cached detail HTML"
            if no_text:
                trailer += f" — {no_text} detail pages have nothing extractable beyond the queue item"
            if http_err:
                trailer += f" — {http_err} detail fetches returned HTTP errors"
            trailer += (
                ".\n\n→ If the LISTING page already has every required "
                "field, abandon process_queue and call `dataset_from_queue` "
                "to push the queue items straight into the dataset. "
                "process_queue is only needed when the user asked for "
                "something that lives on the detail page (PDF text, etc.)."
            )

        return ToolResult(
            output=(
                f"process_queue: appended={appended}  errors={errors}  "
                f"remaining_in_queue={remaining}  rows_in_dataset={len(state.dataset)}\n\n"
                + "\n".join(report)
                + trailer
            ),
            artifact={"appended": appended, "errors": errors, "remaining": remaining},
        )


# ── row template rendering ─────────────────────────────────────────────────


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _lookup(ctx: dict, dotted: str) -> Any:
    cur: Any = ctx
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _render_value(v: Any, ctx: dict) -> Any:
    if isinstance(v, str):
        if v.startswith("$file:"):
            inner = v[len("$file:"):]
            resolved = _PLACEHOLDER.sub(lambda m: str(_lookup(ctx, m.group(1)) or ""), inner)
            return {"$file": resolved}
        # Pure {x.y} substitution — if the entire string is one placeholder,
        # return the underlying value (preserving type). Otherwise format.
        m_full = re.fullmatch(r"\{([^{}]+)\}", v)
        if m_full:
            return _lookup(ctx, m_full.group(1))
        return _PLACEHOLDER.sub(lambda m: str(_lookup(ctx, m.group(1)) or ""), v)
    if isinstance(v, list):
        return [_render_value(x, ctx) for x in v]
    if isinstance(v, dict):
        return {k: _render_value(x, ctx) for k, x in v.items()}
    return v


def _render_row(template: dict, ctx: dict) -> dict:
    row: dict[str, Any] = {}
    for k, v in template.items():
        row[k] = _render_value(v, ctx)
    return row
