"""discover_assets: find ranked data-asset links on a fetched HTML page.

Given a cached HTML page (or a URL we've already cached), this tool returns
a list of OUTBOUND LINKS that look like they point to extractable data —
PDFs, XML, CSV, JSON, archives, plus same-domain HTML pages whose anchor
text suggests structured content ("annexe", "décision", "full text", etc.).

Each candidate is scored by:
  • file extension (.pdf .xml .csv .json .tar .zip → high)
  • anchor text relevance (data-y words → bonus)
  • URL keywords (download, attachment, full, raw → bonus)

The agent uses this to decide WHICH outbound paths are worth fetching from
a listing or detail page, instead of guessing or burning fetches on
sector/filter URLs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.html_tool import _load_or_raise, _resolve_source_arg, _SourceNotFoundError

if TYPE_CHECKING:
    from gemma42.state import AgentState


_DATA_EXTS = {
    ".pdf":   ("pdf",      40),
    ".xml":   ("xml",      35),
    ".csv":   ("csv",      35),
    ".tsv":   ("csv",      30),
    ".json":  ("json",     35),
    ".jsonl": ("json",     35),
    ".ndjson": ("json",    30),
    ".tar":   ("archive",  30),
    ".tgz":   ("archive",  30),
    ".gz":    ("archive",  25),
    ".bz2":   ("archive",  25),
    ".zip":   ("archive",  30),
    ".docx":  ("docx",     25),
    ".doc":   ("docx",     20),
    ".odt":   ("docx",     20),
    ".rtf":   ("docx",     20),
    ".xlsx":  ("xlsx",     30),
    ".xls":   ("xlsx",     25),
    ".ods":   ("xlsx",     25),
    ".html":  ("html",     12),
    ".htm":   ("html",     12),
    ".txt":   ("text",     15),
}

_DATA_KEYWORDS_HIGH = (
    "decision", "décision", "deliberation", "délibération",
    "ruling", "judgement", "judgment", "annexe", "annex",
    "attachment", "full text", "full-text", "fulltext",
    "data set", "dataset", "raw data", "download all", "export",
)
_DATA_KEYWORDS_LOW = (
    "download", "attachment", "raw", "fichier", "document",
    "full", "complete", "détail", "detail",
)


_HREF_RE = re.compile(r'<a\s+[^>]*?href="([^"#?]+(?:\?[^"#]*)?)"[^>]*>(.*?)</a>',
                        re.IGNORECASE | re.DOTALL)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _classify(url: str, anchor_text: str) -> tuple[str | None, int, str]:
    """Return (kind, score, reason) for a candidate link."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    ext = ""
    if "." in path.rsplit("/", 1)[-1]:
        ext = "." + path.rsplit(".", 1)[-1]
    # extension hit
    kind = None
    score = 0
    reason_parts: list[str] = []
    if ext in _DATA_EXTS:
        kind, score = _DATA_EXTS[ext]
        reason_parts.append(f"ext={ext}")
    # anchor / URL keyword bonuses
    text_l = anchor_text.lower()
    url_l = url.lower()
    for kw in _DATA_KEYWORDS_HIGH:
        if kw in text_l or kw in url_l:
            score += 15
            reason_parts.append(f"kw+={kw}")
            break
    for kw in _DATA_KEYWORDS_LOW:
        if kw in text_l or kw in url_l:
            score += 5
            reason_parts.append(f"kw={kw}")
            break
    # downgrade plain HTML if nothing else suggests structured content
    if kind == "html" and score <= 12:
        return None, 0, ""
    if kind is None and score == 0:
        return None, 0, ""
    return kind, score, ",".join(reason_parts)


class DiscoverAssetsTool(Tool):
    name = "discover_assets"
    description = (
        "Scan a cached HTML page (or a URL we've already fetched) and return "
        "every outbound link that LOOKS like a data asset (PDF, XML, CSV, "
        "JSON, archive, structured HTML detail page, …), ranked by a "
        "deterministic score from extension + anchor text + URL keywords.\n\n"
        "Use this to decide what to follow on a detail page BEFORE burning "
        "fetches. Output includes the absolute URL, classified `kind`, "
        "score, and the snippet of anchor text that triggered the score.\n\n"
        "Args:\n"
        "  source / url  : cached HTML path or URL (any of the usual aliases)\n"
        "  base_url      : optional base for resolving relative hrefs. If\n"
        "                  omitted, derived from the page's <base> tag or\n"
        "                  the URL we resolved the source from.\n"
        "  max_results   : default 30\n"
        "  min_score     : default 10 (anything below is dropped)"
    )
    args_schema = {
        "source":      {"type": "string"},
        "url":         {"type": "string"},
        "base_url":    {"type": "string"},
        "max_results": {"type": "integer", "default": 30},
        "min_score":   {"type": "integer", "default": 10},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)

        base_url = (args.get("base_url") or "").strip()
        if not base_url:
            # Try <base href="..."> in head
            m = re.search(r'<base\s+[^>]*?href="([^"]+)"', html, re.IGNORECASE)
            if m:
                base_url = m.group(1)
        if not base_url and isinstance(args.get("url"), str):
            base_url = args["url"]

        max_results = int(args.get("max_results") or 30)
        min_score = int(args.get("min_score") or 10)

        seen: set[str] = set()
        candidates: list[dict] = []
        for m in _HREF_RE.finditer(html):
            href = (m.group(1) or "").strip()
            anchor = _strip_tags(m.group(2) or "")
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            # Resolve to absolute when we have a base.
            abs_url = href
            if base_url and not re.match(r"^https?://", href):
                try:
                    abs_url = urljoin(base_url, href)
                except Exception:  # noqa: BLE001
                    abs_url = href
            if abs_url in seen:
                continue
            seen.add(abs_url)
            kind, score, reason = _classify(abs_url, anchor)
            if not kind or score < min_score:
                continue
            candidates.append({
                "url":     abs_url,
                "kind":    kind,
                "score":   score,
                "anchor":  anchor[:120],
                "reason":  reason,
            })

        # Sort and trim.
        candidates.sort(key=lambda c: (-c["score"], c["url"]))
        candidates = candidates[:max_results]

        lines = [f"discover_assets: {len(candidates)} candidate(s)"]
        kind_counts: dict[str, int] = {}
        for c in candidates:
            kind_counts[c["kind"]] = kind_counts.get(c["kind"], 0) + 1
        if kind_counts:
            lines.append("by kind: " + ", ".join(
                f"{k}={v}" for k, v in sorted(kind_counts.items(), key=lambda kv: -kv[1])
            ))
        lines.append("")
        for c in candidates[:15]:
            lines.append(
                f"  [{c['score']:3d}] {c['kind']:8s} {c['url']}  "
                f"({c['reason']}) :: {c['anchor']!r}"
            )
        if len(candidates) > 15:
            lines.append(f"  ... and {len(candidates) - 15} more (see artifact)")

        return ToolResult(output="\n".join(lines),
                           artifact={"candidates": candidates})
