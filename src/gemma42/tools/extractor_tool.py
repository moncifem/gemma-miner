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

from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.extract_text_tool import _extract_bytes, _sniff
from gemma42.tools.http_tool import (
    _ext_from_bytes,
    _ext_from_ctype,
    _ext_from_url,
    _slug,
)

if TYPE_CHECKING:
    from gemma42.state import AgentState


# ── spec application ────────────────────────────────────────────────────────


_TRANSFORMS = {
    "strip": lambda s: s.strip() if isinstance(s, str) else s,
    "lower": lambda s: s.lower() if isinstance(s, str) else s,
    "upper": lambda s: s.upper() if isinstance(s, str) else s,
    "id_normalize": lambda s: re.sub(r"[^\w]+", "", s).upper() if isinstance(s, str) else s,
    "first_line": lambda s: (s.splitlines()[0] if isinstance(s, str) and s.splitlines() else s),
    "html_unescape": lambda s: _html_unescape(s) if isinstance(s, str) else s,
}


def _html_unescape(s: str) -> str:
    import html

    return html.unescape(s)


def _apply_field(row: str, conf: dict, base_url: str) -> Any:
    main_re = conf.get("regex")
    if not main_re:
        return None
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
    if val is None:
        return None
    if conf.get("prefix_base") and base_url and isinstance(val, str) and val.startswith("/"):
        val = base_url.rstrip("/") + val
    tform = conf.get("transform")
    if tform and tform in _TRANSFORMS:
        val = _TRANSFORMS[tform](val)
    elif tform and isinstance(val, str):
        val = val.strip()  # default safety: strip
    return val


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
            # If there's a capture group, prefer it; otherwise the whole match.
            block = m.group(1) if m.lastindex else m.group(0)
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


class ExtractorDefineTool(Tool):
    name = "extractor_define"
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
        "id_normalize, first_line, html_unescape).\n\n"
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

        # Reject typos in spec keys.
        valid_keys = {
            "row_pattern", "row_group", "include_substring", "exclude_substring",
            "base_url", "fields",
        }
        unknown = set(spec.keys()) - valid_keys
        if unknown:
            return ToolResult(
                output=(
                    f"ERROR: unknown spec key(s): {sorted(unknown)}. "
                    f"Valid keys are: {sorted(valid_keys)}. "
                    f"Common typo: did you mean 'row_pattern' (not 'row_regex')?"
                ),
                error=True,
            )

        # Validate compilable regexes up front.
        if spec.get("row_pattern"):
            try:
                re.compile(spec["row_pattern"], re.DOTALL)
            except re.error as e:
                return ToolResult(output=f"ERROR: invalid row_pattern: {e}", error=True)
        fields_conf = spec.get("fields") or {}
        if not fields_conf:
            return ToolResult(
                output=(
                    "ERROR: spec has no 'fields'. A spec must include at least "
                    "one field. Example: {\"fields\": {\"id\": {\"regex\": \"...\"}}}"
                ),
                error=True,
            )
        valid_field_keys = {"regex", "group", "fallback_regex", "prefix_base", "transform"}
        for fname, fconf in fields_conf.items():
            if not isinstance(fconf, dict):
                return ToolResult(
                    output=f"ERROR: field '{fname}' must be an object, got {type(fconf).__name__}",
                    error=True,
                )
            fk_unknown = set(fconf.keys()) - valid_field_keys
            if fk_unknown:
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
                    lines.append(
                        "DIAGNOSIS: 0 rows. Causes (most common first):\n"
                        "  1. row_pattern doesn't match — try a simpler pattern.\n"
                        "  2. include_substring is set and doesn't appear in any block.\n"
                        "  3. exclude_substring is set and matches every block.\n"
                        "Try removing include/exclude first to see if row_pattern works."
                    )
                else:
                    lines.append("--- row 0 (extracted) ---")
                    lines.append(json.dumps(rows[0], ensure_ascii=False, indent=2))
                    if len(rows) > 1:
                        lines.append("--- row 1 (extracted) ---")
                        lines.append(json.dumps(rows[1], ensure_ascii=False, indent=2))
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
    max_retries: int = 3,
) -> tuple[Path, bytes, str]:
    """Cached HTTP GET with retry. Returns (cache_path, content, content_type)."""
    slug = _slug(url)
    for existing in cache_dir.glob(f"{slug}.*"):
        return existing.resolve(), existing.read_bytes(), ""
    h = {"User-Agent": "Mozilla/5.0 (gemma42 research agent)"}
    if headers:
        h.update(headers)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                r = client.get(url, headers=h)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}", request=r.request, response=r
                    )
                content = r.content
                ctype = r.headers.get("content-type", "")
                break
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
            continue
    else:
        raise RuntimeError(f"http_get failed after {max_retries} attempts: {last_err}")

    ext = _ext_from_ctype(ctype) or _ext_from_url(url) or _ext_from_bytes(content) or ".bin"
    cache_path = (cache_dir / f"{slug}{ext}").resolve()
    cache_path.write_bytes(content)
    return cache_path, content, ctype


class ScrapePaginatedTool(Tool):
    name = "scrape_paginated"
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
            except Exception:  # noqa: BLE001
                pass

        start = int(args.get("start_page") or 0)
        max_pages = int(args.get("max_pages") or 20)
        target = int(args.get("target_count") or 100)
        delay = max(0, int(args.get("delay_ms") or 250)) / 1000.0

        cache_dir = Path(state.workdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        queue = state.memory.get("queue", []) or []
        processed = set(str(x) for x in (state.memory.get("processed", []) or []))
        existing_ids = {str(i.get("id")) for i in queue if isinstance(i, dict) and i.get("id")}

        log: list[str] = []
        total_added = 0
        page_range = range(start, start + max_pages) if has_page_placeholder else [start]
        for page in page_range:
            url = template.format(page=page) if has_page_placeholder else template
            try:
                _path, content, _ct = _http_get(url, cache_dir)
            except Exception as e:  # noqa: BLE001
                log.append(f"page={page} FETCH ERROR: {e}")
                break
            html = content.decode("utf-8", errors="replace")
            try:
                items = apply_listing_spec(html, spec)
            except Exception as e:  # noqa: BLE001
                log.append(f"page={page} EXTRACT ERROR: {e}")
                break
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
            if not new_items and len(items) == 0:
                log.append("0 items on this page — stopping pagination.")
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

        return ToolResult(
            output="scrape_paginated:\n  " + "\n  ".join(log) + f"\n\ntotal_added: {total_added}\nqueue_len: {len(queue)}",
            artifact={"added": total_added, "queue_len": len(queue)},
        )


# ── tool: process_queue ────────────────────────────────────────────────────


class ProcessQueueTool(Tool):
    name = "process_queue"
    description = (
        "Process up to `batch_size` queued items end-to-end in a single tool "
        "call (no per-item LLM round-trips). For each item the tool: "
        "(1) fetches `item.detail_url` (cached), "
        "(2) applies the named DETAIL extractor to find the attachment URL "
        "    and any extra scalar fields, "
        "(3) fetches the attachment, "
        "(4) copies it to items/item_NNNN/attachment_NN.<ext>, "
        "(5) runs the universal text extractor and writes attachment_NN.txt, "
        "(6) calls dataset_append with a row that includes $file references "
        "    to the .txt (so pdf_text is resolved from disk at append time), "
        "(7) marks the item done.\n"
        "Use `row_template` to specify how to build the row dict from the "
        "queue item, the detail-spec output, and the attachment paths. The "
        "template uses placeholders like {queue.id}, {detail.title}, "
        "{paths.attachment}, {paths.text}. Any field whose value starts "
        "with '$file:' is converted to a real $file reference."
    )
    args_schema = {
        "detail_extractor": {
            "type": "string",
            "description": "Name of a stored DETAIL extractor that has an 'attachment_url' field (or set attachment_url_field).",
        },
        "attachment_url_field": {
            "type": "string",
            "default": "attachment_url",
            "description": "Which field of the detail extractor's output holds the binary URL.",
        },
        "row_template": {
            "type": "object",
            "description": (
                "Dict mapping output-row field names to template strings or "
                "literal values. Placeholders: {queue.<k>}, {detail.<k>}, "
                "{paths.attachment}, {paths.text}, {paths.item_dir}. "
                "Prefix a value with '$file:' (e.g. '$file:{paths.text}') "
                "to emit a $file reference instead of the literal."
            ),
        },
        "batch_size": {"type": "integer", "default": 5},
        "delay_ms": {"type": "integer", "default": 250},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        det_name = args.get("detail_extractor")
        if not det_name:
            return ToolResult(output="ERROR: 'detail_extractor' required", error=True)
        extractors = state.memory.get("extractors", {}) or {}
        det_spec = extractors.get(det_name)
        if not det_spec:
            return ToolResult(output=f"ERROR: no extractor named '{det_name}'", error=True)
        att_field = args.get("attachment_url_field") or "attachment_url"
        row_tmpl = args.get("row_template")
        if not isinstance(row_tmpl, dict) or not row_tmpl:
            return ToolResult(output="ERROR: 'row_template' (object) required", error=True)
        batch_size = int(args.get("batch_size") or 5)
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
                # 2. apply detail spec
                detail = apply_detail_spec(dhtml, det_spec)
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
                # 6. build row from template
                ctx = {
                    "queue": item,
                    "detail": detail,
                    "paths": {
                        "attachment": dest_bin.relative_to(state.workdir).as_posix(),
                        "text": dest_txt.relative_to(state.workdir).as_posix(),
                        "item_dir": item_dir.relative_to(state.workdir).as_posix(),
                    },
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
                next_slot += 1
                report.append(
                    f"{iid}: OK  slot={next_slot - 1}  bin={dest_bin.name}  text_chars={len(text)}"
                )
            except Exception as e:  # noqa: BLE001
                report.append(f"{iid}: ERROR {type(e).__name__}: {e}")
                errors += 1
            if delay:
                time.sleep(delay)

        state.memory.set("processed", processed_list)

        remaining = sum(1 for q in queue if str(q.get("id")) not in processed)
        return ToolResult(
            output=(
                f"process_queue: appended={appended}  errors={errors}  "
                f"remaining_in_queue={remaining}  rows_in_dataset={len(state.dataset)}\n\n"
                + "\n".join(report)
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
