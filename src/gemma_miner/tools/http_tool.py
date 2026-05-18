"""HTTP fetcher. Stores the full body to disk in the workdir and returns a preview."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


def _slug(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


# Map content-type prefixes / suffixes to a sensible file extension.
_CTYPE_EXT = [
    ("application/json", ".json"),
    ("application/ld+json", ".json"),
    ("application/x-ndjson", ".jsonl"),
    ("application/jsonl", ".jsonl"),
    ("text/html", ".html"),
    ("application/xhtml", ".html"),
    ("text/xml", ".xml"),
    ("application/xml", ".xml"),
    ("application/atom+xml", ".xml"),
    ("application/rss+xml", ".xml"),
    ("text/csv", ".csv"),
    ("text/tab-separated-values", ".tsv"),
    ("application/pdf", ".pdf"),
    ("application/zip", ".zip"),
    ("application/gzip", ".gz"),
    ("application/x-gzip", ".gz"),
    ("application/x-tar", ".tar"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
    ("application/vnd.oasis.opendocument.text", ".odt"),
    ("application/epub", ".epub"),
    ("application/rtf", ".rtf"),
    ("text/yaml", ".yaml"),
    ("application/yaml", ".yaml"),
    ("text/plain", ".txt"),
    ("text/markdown", ".md"),
    ("image/png", ".png"),
    ("image/jpeg", ".jpg"),
    ("image/gif", ".gif"),
    ("image/webp", ".webp"),
    ("image/svg", ".svg"),
]


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix and re.fullmatch(r"\.[a-z0-9]{1,6}", suffix):
        return suffix
    return ""


def _ext_from_ctype(ctype: str) -> str:
    c = ctype.split(";", 1)[0].strip().lower()
    for prefix, ext in _CTYPE_EXT:
        if c.startswith(prefix):
            return ext
    return ""


def _body_preview(text: str, n: int) -> tuple[str, int]:
    """For HTML, return a preview anchored at the first interesting body
    content (skipping <head>, <script>, <style>, <nav>, navbar markup).

    Returns (preview_text, start_offset). For non-HTML, returns (text[:n], 0).
    """
    if not text:
        return "", 0
    lower = text[:8000].lower()
    if "<html" not in lower:
        return text[:n], 0

    # Strip head + script + style + svg sections destructively for the preview.
    cleaned = text
    for tag in ("head", "script", "style", "svg", "noscript"):
        cleaned = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
        )

    # Skip navbar / header chrome to find the main content.
    anchors = [
        '<main',
        'id="main-content"',
        'class="main-content"',
        'class="view-content"',
        'class="views-row"',
        'role="main"',
        '<article',
        'id="content"',
    ]
    start = 0
    for a in anchors:
        i = cleaned.find(a)
        if i >= 0:
            start = i
            break
    if start == 0:
        # Last resort: skip past <body opening.
        m = re.search(r"<body\b", cleaned, re.IGNORECASE)
        if m:
            start = m.end()

    preview = cleaned[start : start + n]
    return preview, start


def _ext_from_bytes(data: bytes) -> str:
    if not data:
        return ""
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):
        return ".zip"
    if data.startswith(b"\x1f\x8b"):
        return ".gz"
    head = data[:512].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<svg"):
        return ".xml"
    if head[:16].lower().startswith((b"<!doctype html", b"<html")):
        return ".html"
    if head[:1] in (b"{", b"["):
        return ".json"
    try:
        data[:256].decode("utf-8")
        return ".txt"
    except UnicodeDecodeError:
        return ".bin"


class HttpGetTool(Tool):
    name = "http_get"
    description = (
        "HTTP GET a URL. Saves the full response body to a file in the workdir's "
        "`cache/` folder, named `<sha1-prefix><real-extension>` (.json, .html, "
        ".pdf, .xml, .csv, .zip, etc. — picked from Content-Type, then URL "
        "suffix, then magic bytes). Returns: status code, content-type, byte "
        "length, the cache file path, and a body preview.\n\n"
        "Preview defaults to 8 000 characters (anchored at <main>/<article>/<body> "
        "for HTML so you don't waste tokens on chrome). For deeper inspection, "
        "either pass `preview_chars=<bigger N>` on this call, or use "
        "`html_inspect`/`read_file` on the cache_path afterwards (cheaper than "
        "putting the whole body in every subsequent prompt). Set `preview_chars=0` "
        "to skip the preview entirely.\n\nUse this for any web fetch — never "
        "use bash curl for the same purpose."
    )
    args_schema = {
        "url": {"type": "string", "description": "Absolute URL to fetch."},
        "user_agent": {
            "type": "string",
            "description": "Optional User-Agent header.",
            "default": "Mozilla/5.0 (gemma-miner research agent)",
        },
        "preview_chars": {
            "type": "integer",
            "description": "How many chars of body to include in the output. Default 8000.",
            "default": 8_000,
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        url = args.get("url")
        if not url:
            return ToolResult(output="ERROR: 'url' is required", error=True)
        ua = args.get("user_agent") or "Mozilla/5.0 (gemma-miner research agent)"
        preview = int(args.get("preview_chars") or 8_000)

        cache_dir = Path(state.workdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache HIT — reuse the body so we don't re-pay network latency.
        slug = _slug(url)
        for existing in cache_dir.glob(f"{slug}.*"):
            data = existing.read_bytes()
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = "<binary>"
            body_preview = text[:preview]
            return ToolResult(
                output=(
                    f"status: 200 (cache-hit)\n"
                    f"content_type: (from cache)\n"
                    f"bytes: {len(data)}\n"
                    f"cache_path: {existing.resolve()}\n"
                    f"--- body preview ({min(preview, len(text))} of {len(text)} chars) ---\n"
                    f"{body_preview}"
                ),
                artifact={"path": str(existing.resolve()), "status": 200, "cache_hit": True},
            )

        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                r = client.get(url, headers={"User-Agent": ua})
                content = r.content
                ctype = r.headers.get("content-type", "")
                status = r.status_code
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR fetching {url}: {e}", error=True)

        # Pick a real extension: content-type → URL suffix → magic bytes → .bin
        ext = (
            _ext_from_ctype(ctype)
            or _ext_from_url(url)
            or _ext_from_bytes(content)
            or ".bin"
        )
        # Heuristic: NDJSON-ish payloads with one JSON object per line should be .jsonl
        if ext == ".json" and content[:1] not in (b"[", b"") and b"\n{" in content[:4096]:
            ext = ".jsonl"

        cache_path = (cache_dir / f"{_slug(url)}{ext}").resolve()
        cache_path.write_bytes(content)
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = "<binary>"
        body_preview, body_offset = _body_preview(text, preview)

        # Flag non-success responses so the agent doesn't try to extract data
        # from a 403/404/etc. page. We still cache the body (it can help
        # diagnose) but mark the tool result as an error.
        is_error_status = status >= 400
        warning = ""
        if is_error_status:
            warning = (
                f"\n⚠ HTTP {status} — this is NOT the page content. "
                "The site refused (403), the URL is wrong (404), or the "
                "server failed (5xx). Trying to scrape this cached body will "
                "yield nothing useful. Options:\n"
                "  • Pick a different URL (the listing page may already have "
                "    what you need — try `dataset_from_queue` on existing items).\n"
                "  • Set a real User-Agent header via http_get(user_agent='...').\n"
                "  • Stop trying to fetch this domain.\n"
            )
        out = (
            f"status: {status}\n"
            f"content_type: {ctype}\n"
            f"bytes: {len(content)}\n"
            f"cache_path: {cache_path}\n"
            + warning
            + f"--- body preview ({len(body_preview)} chars, offset {body_offset} of {len(text)}) ---\n"
            f"{body_preview}"
        )
        return ToolResult(
            output=out, error=is_error_status,
            artifact={"path": str(cache_path), "status": status},
        )
