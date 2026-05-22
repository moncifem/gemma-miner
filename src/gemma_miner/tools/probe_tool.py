"""field_probe and pagination_probe tools."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


def _printable(s: str) -> str:
    """Strip non-printable characters, keep ASCII + common Unicode."""
    return "".join(c for c in s if c.isprintable() or c in "\n\r\t")


def _suggest_regex(html: str, value: str, match_start: int) -> str:
    """Walk backward to find an opening tag, forward to find closing tag,
    build a simple regex pattern around the value."""
    # Walk backward from match_start to find an opening tag
    before = html[:match_start]
    tag_match = re.search(r"<(\w+)[^>]*>\s*$", before)
    if tag_match:
        tag = tag_match.group(1)
        escaped = re.escape(value)
        return f"<{tag}[^>]*>([^<]*?{escaped}[^<]*?)</{tag}>"

    # Fallback: look for any tag ending just before our value
    tag_match2 = re.search(r"<(\w+)[^>]*>[^<]*$", before)
    if tag_match2:
        tag = tag_match2.group(1)
        escaped = re.escape(value)
        return f"<{tag}[^>]*>([^<]*?{escaped}[^<]*?)</{tag}>"

    # Generic fallback
    escaped = re.escape(value)
    return f">([^<]*?{escaped}[^<]*?)<"


class FieldProbeTool(Tool):
    name = "field_probe"
    is_readonly = True
    max_output_chars = 5_000
    description = (
        "Search cached HTML for specific field values and return surrounding markup "
        "context. Call this BEFORE writing extractor_define to discover WHERE each "
        "field's value lives in the HTML."
    )
    args_schema = {
        "values": {
            "type": "array",
            "description": "Sample values to search for (e.g. ['862B', 'deepseek-ai', '128K context']).",
            "items": {"type": "string"},
        },
        "context_chars": {
            "type": "integer",
            "default": 400,
            "description": "Chars of HTML around each hit.",
        },
        "max_hits_per_value": {
            "type": "integer",
            "default": 2,
            "description": "Stop after N hits per value (avoid flooding).",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        values = args.get("values") or []
        if isinstance(values, str):
            values = [values]
        if not values:
            return ToolResult(output="ERROR: 'values' must be a non-empty list of strings.", error=True)
        context_chars = int(args.get("context_chars") or 400)
        max_hits = int(args.get("max_hits_per_value") or 2)

        cache_dir = Path(state.workdir) / "cache"
        html_files: list[Path] = []
        if cache_dir.exists():
            htmls = [p for p in cache_dir.glob("*.html") if p.is_file()]
            htmls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            html_files = htmls[:2]

        if not html_files:
            return ToolResult(
                output=(
                    "No cached HTML files found in cache/. "
                    "Call http_get on the listing page first to populate the cache."
                ),
                error=True,
            )

        output_blocks: list[str] = []
        total_chars = 0
        cap = 8000

        for value in values:
            if not isinstance(value, str) or not value:
                continue
            block_lines: list[str] = [f"=== value: {value!r} ==="]
            hits_found = 0

            for html_path in html_files:
                if hits_found >= max_hits:
                    break
                try:
                    html = html_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:  # noqa: BLE001
                    block_lines.append(f"  [could not read {html_path.name}: {e}]")
                    continue

                start = 0
                while hits_found < max_hits:
                    idx = html.find(value, start)
                    if idx == -1:
                        break
                    # Extract context_chars centred on the match
                    ctx_start = max(0, idx - context_chars // 2)
                    ctx_end = min(len(html), idx + len(value) + context_chars // 2)
                    snippet = html[ctx_start:ctx_end]
                    snippet = _printable(snippet)

                    block_lines.append(f"\n  [hit {hits_found + 1} in {html_path.name}, offset {idx}]")
                    block_lines.append("  --- context ---")
                    block_lines.append(snippet)
                    block_lines.append("  --- end context ---")

                    # Suggest a regex
                    suggested = _suggest_regex(html, value, idx)
                    block_lines.append(f"  suggested regex: {suggested}")

                    hits_found += 1
                    start = idx + len(value)

            if hits_found == 0:
                block_lines.append(
                    f"  NOT FOUND in any cached HTML. "
                    "Re-examine your assumption — this value may not be on the "
                    "listing page, or you may need to fetch a different URL."
                )

            block_text = "\n".join(block_lines)
            if total_chars + len(block_text) > cap:
                output_blocks.append(f"\n[output cap {cap} chars reached — truncated]")
                break
            output_blocks.append(block_text)
            total_chars += len(block_text)

        return ToolResult(output="\n\n".join(output_blocks))


class PaginationProbeTool(Tool):
    name = "pagination_probe"
    is_readonly = True
    max_output_chars = 4_000
    description = (
        "Test multiple pagination URL patterns on a listing URL to discover which "
        "scheme the site uses. Compares row counts across page variants "
        "(page=0/1/2, offset=0/N, p=1/2). Call this in DISCOVER_LISTING when "
        "you're not sure how pagination works."
    )
    args_schema = {
        "base_url": {
            "type": "string",
            "description": "The listing URL WITHOUT pagination params (e.g. https://site.com/models).",
        },
        "extractor_name": {
            "type": "string",
            "description": (
                "If a 'listing' extractor is saved, use it to count rows; "
                "otherwise compare response sizes."
            ),
        },
        "items_per_page": {
            "type": "integer",
            "default": 0,
            "description": "Known items per page (used to compute offset variants).",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        base_url = args.get("base_url", "").strip()
        if not base_url:
            return ToolResult(output="ERROR: 'base_url' is required.", error=True)
        extractor_name = args.get("extractor_name", "").strip() or None
        ipp = int(args.get("items_per_page") or 0)

        # Build test URLs
        has_query = "?" in base_url
        sep = "&" if has_query else "?"

        test_urls: list[str] = [
            f"{base_url}{sep}page=0",
            f"{base_url}{sep}page=1",
            f"{base_url}{sep}page=2",
            f"{base_url}{sep}p=1",
            f"{base_url}{sep}p=2",
        ]
        if ipp > 0:
            test_urls += [
                f"{base_url}{sep}offset=0",
                f"{base_url}{sep}offset={ipp}",
                f"{base_url}{sep}offset={ipp * 2}",
            ]
        # If already has ?, add &page= variants
        if has_query:
            test_urls += [
                f"{base_url}&page=1",
                f"{base_url}&page=2",
            ]

        # Load extractor spec if given
        spec = None
        if extractor_name:
            extractors = state.memory.get("extractors", {}) or {}
            spec = extractors.get(extractor_name)

        from gemma_miner.tools.extractor_tool import apply_listing_spec

        rows: dict[str, str] = {}  # url -> rows_or_bytes str
        hashes: dict[str, str] = {}  # url -> sha1

        with httpx.Client(follow_redirects=True, timeout=15) as client:
            for url in test_urls:
                try:
                    r = client.get(url, headers={"User-Agent": "Mozilla/5.0 (gemma-miner probe)"})
                    content = r.content
                    sha = hashlib.sha1(content).hexdigest()[:10]
                    hashes[url] = sha
                    if spec and spec.get("row_pattern"):
                        html = content.decode("utf-8", errors="replace")
                        try:
                            extracted = apply_listing_spec(html, spec)
                            rows[url] = f"{len(extracted)} rows (sha={sha})"
                        except Exception as e:  # noqa: BLE001
                            rows[url] = f"extract error: {e} (sha={sha})"
                    else:
                        rows[url] = f"{len(content)} bytes (sha={sha})"
                except Exception as e:  # noqa: BLE001
                    rows[url] = f"ERROR: {e}"
                    hashes[url] = "error"

        # Build table
        first_sha = hashes.get(test_urls[0], "") if test_urls else ""
        lines = ["url | status | rows_or_bytes | same_as_first?"]
        lines.append("-" * 80)
        for url in test_urls:
            val = rows.get(url, "?")
            sha = hashes.get(url, "")
            same = "YES" if sha == first_sha and sha not in ("", "error") else "NO"
            lines.append(f"{url} | {val} | same_as_first={same}")

        # Derive recommendation
        recommendation = _derive_recommendation(test_urls, rows, hashes)
        lines.append("")
        lines.append(recommendation)

        return ToolResult(output="\n".join(lines))


def _derive_recommendation(
    test_urls: list[str],
    rows: dict[str, str],
    hashes: dict[str, str],
) -> str:
    """Heuristic: figure out which pagination param actually changes the content."""

    def _row_count(url: str) -> int | None:
        val = rows.get(url, "")
        m = re.match(r"^(\d+) rows", val)
        if m:
            return int(m.group(1))
        return None

    def _byte_count(url: str) -> int | None:
        val = rows.get(url, "")
        m = re.match(r"^(\d+) bytes", val)
        if m:
            return int(m.group(1))
        return None

    def _sha(url: str) -> str:
        return hashes.get(url, "")

    # Check page= variants
    p0 = next((u for u in test_urls if u.endswith("page=0")), None)
    p1 = next((u for u in test_urls if u.endswith("page=1") and "offset" not in u), None)
    p2 = next((u for u in test_urls if u.endswith("page=2") and "offset" not in u), None)

    if p0 and p1 and p2:
        rc0 = _row_count(p0)
        rc1 = _row_count(p1)
        rc2 = _row_count(p2)
        if rc0 is not None and rc1 is not None and rc2 is not None:
            if rc0 == 0 and rc1 > 0:
                return (
                    f"RECOMMENDATION: use `?page={{page}}` with start_page=1 "
                    f"(page=0→{rc0} rows, page=1→{rc1} rows)"
                )
            if rc0 > 0 and rc1 > 0 and rc2 > 0 and _sha(p0) != _sha(p1):
                return (
                    f"RECOMMENDATION: use `?page={{page}}` with start_page=0 "
                    f"(page=0→{rc0} rows, page=1→{rc1} rows, distinct pages)"
                )
            if _sha(p0) == _sha(p1) == _sha(p2):
                return (
                    "RECOMMENDATION: all ?page= variants return the same content — "
                    "site may not paginate via URL; try the JSON API or llm_scrape"
                )

        # Fall back to byte comparison
        b0 = _byte_count(p0)
        b1 = _byte_count(p1)
        b2 = _byte_count(p2)
        if b0 is not None and b1 is not None and b2 is not None:
            if _sha(p0) != _sha(p1) and _sha(p1) != _sha(p2):
                return (
                    "RECOMMENDATION: use `?page={page}` with start_page=0 "
                    "(pages 0/1/2 return different content by byte hash)"
                )
            if _sha(p0) == _sha(p1) == _sha(p2):
                return (
                    "RECOMMENDATION: all ?page= variants return the same content — "
                    "site may not paginate via URL; try the JSON API or llm_scrape"
                )

    # Check offset= variants
    off0 = next((u for u in test_urls if "offset=0" in u), None)
    off1 = next((u for u in test_urls if re.search(r"offset=\d+$", u) and "offset=0" not in u), None)
    if off0 and off1:
        if _sha(off0) != _sha(off1) and _sha(off1) not in ("", "error"):
            return (
                "RECOMMENDATION: use `?offset={N}` pagination — "
                "offset=0 and offset=N return different content"
            )

    # Check p= variants
    pp1 = next((u for u in test_urls if u.endswith("p=1")), None)
    pp2 = next((u for u in test_urls if u.endswith("p=2")), None)
    if pp1 and pp2:
        rc1 = _row_count(pp1)
        rc2 = _row_count(pp2)
        if rc1 is not None and rc2 is not None and rc1 > 0 and _sha(pp1) != _sha(pp2):
            return (
                f"RECOMMENDATION: use `?p={{page}}` with start_page=1 "
                f"(p=1→{rc1} rows, p=2→{rc2} rows)"
            )

    return (
        "RECOMMENDATION: unable to auto-detect pagination scheme from the probe results above. "
        "Inspect the page source manually (html_inspect / html_find) or use python() to "
        "trace the XHR requests."
    )
