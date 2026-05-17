"""HTML inspection & extraction helpers built on stdlib only.

`html_inspect` gives the agent a structural overview (frequent tags & classes)
so it can decide on a repeating-unit selector. `html_extract` then runs
regex- or simple-selector-based extraction against a cached HTML file.
"""

from __future__ import annotations

import html as htmllib
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _looks_like_path_not_content(s: str) -> bool:
    """Heuristic: this string is a path the user expected to be loaded, not raw HTML.

    Catches obvious filename-shaped inputs so we can raise instead of silently
    parsing the filename as if it were HTML.
    """
    if not s or len(s) > 512:
        return False
    if "<" in s or "\n" in s:
        return False
    # ends in a common HTML/data extension OR looks like a cache slug
    return bool(
        re.search(r"\.(html?|xml|json|jsonl|txt|csv|tsv|pdf)$", s, re.IGNORECASE)
        or re.search(r"^[\w./-]+/[\w./-]+$", s)  # path-like with slashes
        or re.fullmatch(r"[0-9a-f]{12}\.\w+", s)  # cache_path pattern
    )


def _load(path_or_text: str, state: "AgentState") -> str:
    # Cheap path/content discrimination — anything with a newline, a `<`,
    # or longer than 1 KB cannot be a usable filesystem path.
    if not path_or_text:
        return ""
    if (
        len(path_or_text) > 1024
        or "\n" in path_or_text
        or "<" in path_or_text[:8]
    ):
        return path_or_text

    # Try as filesystem path. Resolve relative paths against the WORKDIR
    # (not the process CWD), and also try common locations under workdir
    # so the model can pass either 'foo.html' or 'cache/foo.html' or an
    # absolute path interchangeably.
    workdir = Path(state.workdir)
    candidates = []
    try:
        p = Path(path_or_text)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(workdir / p)              # workdir/foo
            candidates.append(workdir / "cache" / p.name)  # workdir/cache/foo
            candidates.append(workdir / "items" / p)    # workdir/items/foo
    except (OSError, ValueError):
        pass

    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                try:
                    return cand.read_text(encoding="utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    return cand.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            continue
    return path_or_text


class _SourceNotFoundError(Exception):
    """Raised when the agent passed a filename-shaped string we couldn't load."""


def _load_or_raise(path_or_text: str, state: "AgentState") -> str:
    """Load the source, raising _SourceNotFoundError if it looks like a path
    that we couldn't find anywhere."""
    if not path_or_text:
        return ""
    result = _load(path_or_text, state)
    # If `_load` returned the input unchanged AND the input looks path-shaped,
    # the file wasn't found anywhere we looked.
    if result is path_or_text or result == path_or_text:
        if _looks_like_path_not_content(path_or_text):
            workdir = Path(state.workdir)
            tried = [
                f"  - {Path(path_or_text)}",
                f"  - {workdir / path_or_text}",
                f"  - {workdir / 'cache' / Path(path_or_text).name}",
            ]
            raise _SourceNotFoundError(
                f"could not find file '{path_or_text}'. Looked in:\n"
                + "\n".join(tried)
                + "\nTip: pass the ABSOLUTE cache_path returned by http_get, "
                "verbatim. If you want to inspect a URL's cached body, you can "
                "also pass `url=...` and the tool will resolve the cache for you."
            )
    return result


def _resolve_source_arg(state: "AgentState", args: dict) -> tuple[str | None, str | None]:
    """Resolve the `source` arg, optionally fetching a URL on the fly.

    Returns (source_string, error_message). If `url` is given and there is no
    cached file for it yet, the caller should fetch it via http_get first.
    """
    src = args.get("source") or args.get("path") or args.get("file") or args.get("html")
    if src:
        return src, None
    url = args.get("url")
    if url:
        # Look for a cache hit so we can use it transparently.
        import hashlib

        slug = hashlib.sha1(url.encode()).hexdigest()[:12]
        cache_dir = Path(state.workdir) / "cache"
        if cache_dir.exists():
            for existing in cache_dir.glob(f"{slug}.*"):
                return str(existing.resolve()), None
        return None, (
            f"ERROR: a 'url' was given but no cached body exists. "
            f"Call http_get(url='{url}') first; then this tool can use the cache_path."
        )
    return None, (
        "ERROR: 'source' is required (file path, cache path, or raw HTML). "
        "Or pass 'url' but only AFTER calling http_get on it."
    )


class _StructParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: Counter = Counter()
        self.classes: Counter = Counter()
        self.ids: Counter = Counter()

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        d = dict(attrs)
        cls = d.get("class")
        if cls:
            for c in cls.split():
                self.classes[c] += 1
        if d.get("id"):
            self.ids[d["id"]] += 1


class HtmlInspectTool(Tool):
    name = "html_inspect"
    description = (
        "Summarise the structure of an HTML page so you can decide on a "
        "repeating-unit selector. REQUIRED ARG: `source` (the file path OR raw "
        "HTML — aliases `path`/`file`/`html` are also accepted). Returns the "
        "most frequent tags, CSS classes, and ids. Always run this BEFORE "
        "writing an extractor."
    )
    args_schema = {
        "source": {
            "type": "string",
            "description": "File path to an HTML file, OR raw HTML string.",
        },
        "top_n": {"type": "integer", "default": 25, "description": "How many of each to show."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        top_n = int(args.get("top_n") or args.get("n") or 25)
        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        p = _StructParser()
        try:
            p.feed(html)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR parsing html: {e}", error=True)
        lines = [f"html_size: {len(html)} chars", ""]
        lines.append(f"--- top {top_n} tags ---")
        for t, n in p.tags.most_common(top_n):
            lines.append(f"  {n:6d}  <{t}>")
        lines.append("")
        lines.append(f"--- top {top_n} classes ---")
        for c, n in p.classes.most_common(top_n):
            lines.append(f"  {n:6d}  .{c}")
        if p.ids:
            lines.append("")
            lines.append(f"--- top {top_n} ids ---")
            for i, n in p.ids.most_common(top_n):
                lines.append(f"  {n:6d}  #{i}")

        # Sample block: show the first ~2.5 KB containing the most-frequent
        # *content* class so the model sees an actual row to base regexes on.
        sample = _find_sample_block(html, p.classes)
        if sample:
            lines.append("")
            lines.append(f"--- sample block (class='{sample[0]}', {len(sample[1])} chars) ---")
            lines.append(sample[1])
        return ToolResult(output="\n".join(lines))


def _find_sample_block(html: str, classes: Counter) -> tuple[str, str] | None:
    """Find one occurrence of a content-bearing repeating class and return its
    first ~2.5 KB. Skips classes that are clearly chrome (nav, button, header,
    breadcrumb, menu, lang) so the model gets a *row* example."""
    chrome_substrings = (
        "nav",
        "btn",
        "button",
        "header",
        "footer",
        "breadcrumb",
        "menu",
        "language",
        "search-bar",
        "form-",
        "carousel",
        "logo",
    )
    # Walk frequencies looking for a sensible repeating class.
    for cls, n in classes.most_common(20):
        if n < 3 or n > 200:
            continue
        cls_lower = cls.lower()
        if any(c in cls_lower for c in chrome_substrings):
            continue
        # Find the first opening tag with this exact class token.
        pattern = re.compile(
            rf'<(\w+)\b[^>]*\bclass="[^"]*\b{re.escape(cls)}\b[^"]*"', re.IGNORECASE
        )
        m = pattern.search(html)
        if not m:
            continue
        # Capture a chunk starting at the opening tag.
        start = m.start()
        end = min(len(html), start + 2500)
        # Try to end at a clean tag boundary.
        last_close = html.rfind(">", start, end)
        if last_close > start:
            end = last_close + 1
        return cls, html[start:end]
    return None


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


class HtmlExtractTool(Tool):
    name = "html_extract"
    description = (
        "Extract repeating blocks from an HTML file using a regex over the raw "
        "HTML. REQUIRED ARGS: `source` (file path or HTML; aliases path/file/"
        "html) and `pattern` (Python regex; alias `regex`). Optional `limit` "
        "(int, default 10). The regex implicitly uses re.DOTALL so '.' "
        "matches newlines. Returns first N matches and total count. Use this "
        "to verify your selector before you scale up. If total=0, do NOT "
        "retry similar regexes — call html_inspect to find the real classes."
    )
    args_schema = {
        "source": {"type": "string", "description": "Path to HTML file or raw HTML."},
        "pattern": {
            "type": "string",
            "description": "Python regex with at least one capture group.",
        },
        "limit": {"type": "integer", "default": 10, "description": "Max matches to show."},
        "strip_tags": {
            "type": "boolean",
            "default": False,
            "description": "If true, strip HTML tags from each match. Default False — you usually want the raw HTML when exploring structure.",
        },
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        pattern = args.get("pattern") or args.get("regex") or args.get("re")
        if not pattern:
            return ToolResult(output="ERROR: 'pattern' required", error=True)
        limit = int(args.get("limit") or args.get("n") or 10)
        strip = bool(args.get("strip_tags", False))
        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        try:
            rx = re.compile(pattern, re.DOTALL)
        except re.error as e:
            return ToolResult(output=f"ERROR: invalid regex: {e}", error=True)
        matches = rx.findall(html)
        total = len(matches)
        sample = matches[:limit]
        if total == 0:
            hint = (
                "total_matches: 0\n"
                "HINT: your regex did not match anything. Do NOT retry with a "
                "near-identical pattern — instead, call html_inspect on this "
                "source to see what tag/class names are actually present, then "
                "build a pattern around one of THOSE class names. Many sites "
                "use specific classes like 'views-row', 'search-index', "
                "'field--item', etc."
            )
            return ToolResult(output=hint, artifact={"total": 0})
        out = [f"total_matches: {total}"]
        for i, m in enumerate(sample):
            if isinstance(m, tuple):
                text = " | ".join(_strip_tags(x) if strip else x for x in m)
            else:
                text = _strip_tags(m) if strip else m
            if len(text) > 1200:
                text = text[:1200] + "…"
            out.append(f"[{i}] {text}")
        return ToolResult(output="\n".join(out), artifact={"total": total})
