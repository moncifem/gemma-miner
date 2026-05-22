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
from typing import Any, TYPE_CHECKING

from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


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


def _unwrap_file_ref(v: Any) -> Any:
    """If `v` is a `{"$file": "path"}` reference, return the path (not its content).

    The model often wraps path-typed args in $file references it sees mentioned
    in the prompt. For tools that take a PATH (not file content), we just need
    the path string itself.
    """
    if isinstance(v, dict) and "$file" in v and isinstance(v["$file"], str):
        return v["$file"]
    return v


def _cache_lookup_by_url(state: "AgentState", url: str) -> str | None:
    """Return the absolute cache path for `url` if it's been fetched."""
    import hashlib
    slug = hashlib.sha1(url.encode()).hexdigest()[:12]
    cache_dir = Path(state.workdir) / "cache"
    if cache_dir.exists():
        for existing in cache_dir.glob(f"{slug}.*"):
            return str(existing.resolve())
    return None


def _try_resolve_path_like(state: "AgentState", val: str) -> str | None:
    """Liberal path resolution.

    Accept anything that vaguely looks like a path or cache reference and
    try every plausible location:
      - the value itself, if it's an existing file
      - <workdir>/<value>
      - <workdir>/cache/<basename>
      - <workdir>/cache/<value>
      - just the basename inside cache (e.g. "cache/abc.html" → "cache/abc.html")
    Returns the absolute path string if any candidate exists.
    """
    if not isinstance(val, str) or not val:
        return None
    workdir = Path(state.workdir)
    candidates: list[Path] = []
    try:
        p = Path(val)
    except (OSError, ValueError):
        return None
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            workdir / p,
            workdir / "cache" / p.name,
            workdir / "cache" / p,
        ])
        # If the value starts with "cache/", also try without the prefix.
        if val.startswith("cache/"):
            candidates.append(workdir / val[len("cache/"):])
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return None


def _resolve_source_arg(state: "AgentState", args: dict) -> tuple[str | None, str | None]:
    """Resolve the `source` arg.

    LIBERAL POLICY: collect candidate strings from EVERY plausible arg name
    (source, path, file, html, url) and for each, try in order:
      1. raw HTML (starts with `<`)
      2. real URL (http(s)://…) → look up cache
      3. path-like → resolve under workdir, workdir/cache, etc.
    The first one that resolves wins. Only error out if nothing resolves.
    This protects the agent against the constant mistake of passing a path
    in `url=` (or a URL in `source=`, or `cache/foo.html`, etc.).
    """
    # Collect candidates in priority order.
    arg_names = ("source", "path", "file", "html", "url")
    candidates: list[tuple[str, str]] = []  # (arg_name, value)
    for k in arg_names:
        v = _unwrap_file_ref(args.get(k))
        if v is None:
            continue
        if not isinstance(v, str):
            return None, (
                f"ERROR: '{k}' must be a string path, URL, or raw HTML; got "
                f"{type(v).__name__}. If you meant a path, pass just the string."
            )
        if v:
            candidates.append((k, v))

    if not candidates:
        return None, (
            "ERROR: pass `source` (file path or raw HTML) or `url` (a real URL "
            "already fetched via http_get)."
        )

    unresolved_urls: list[str] = []
    for name, v in candidates:
        # 1. raw HTML
        if "<" in v[:8] or "\n" in v:
            return v, None
        # 2. real URL
        if re.match(r"^https?://", v):
            cached = _cache_lookup_by_url(state, v)
            if cached:
                return cached, None
            unresolved_urls.append(v)
            continue
        # 3. path-like — try the filesystem (workdir, cache, …)
        resolved = _try_resolve_path_like(state, v)
        if resolved:
            return resolved, None
        # 4. fallback: hash-as-URL (in case the agent passed a real URL
        # that happens not to start with http://)
        cached = _cache_lookup_by_url(state, v)
        if cached:
            return cached, None

    if unresolved_urls:
        return None, (
            f"ERROR: '{unresolved_urls[0]}' has not been fetched yet. "
            f"Call http_get(url='{unresolved_urls[0]}') first; this tool will "
            "then load it from the cache automatically."
        )
    sample = candidates[0][1]
    return None, (
        f"ERROR: could not resolve {sample!r} as a file path under {state.workdir!r}, "
        "as a cached URL, or as raw HTML. "
        "If it is a URL, call http_get first. If it is a cached file, pass the "
        "absolute cache_path that http_get returned."
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
    is_readonly = True
    max_output_chars = 6_000  # tag/class summary; rarely needs more
    summary_fields = ("html_size",)
    description = (
        "Summarise the structure of an HTML page so you can decide on a "
        "repeating-unit selector. REQUIRED ARG: `source` (the file path OR raw "
        "HTML -- aliases `path`/`file`/`html` are also accepted). Returns the "
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

    def description_dynamic(self, args: dict, state: "AgentState") -> str | None:
        from pathlib import Path as _P
        cache = _P(state.workdir) / "cache"
        if not cache.exists():
            return None
        html_files = sorted(
            [f for f in cache.iterdir() if f.suffix in (".html", ".htm") and f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not html_files:
            return None
        names = ", ".join(f.name for f in html_files[:5])
        suffix = f" (+{len(html_files) - 5} more)" if len(html_files) > 5 else ""
        return (
            self.description
            + f" Cache has {len(html_files)} HTML file(s): {names}{suffix}."
        )

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        top_n = int(args.get("top_n") or args.get("n") or 25)
        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        # Cap the analysed window. Files in the wild can be 5 MB+ (10 minute
        # parses), and the top-N tag/class summary saturates fast — the
        # marginal info from byte 1.5 MB+ is near zero.
        MAX_SCAN = 1_500_000
        full_len = len(html)
        truncated = full_len > MAX_SCAN
        scan = html[:MAX_SCAN] if truncated else html
        p = _StructParser()
        try:
            p.feed(scan)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR parsing html: {e}", error=True)
        lines = [f"html_size: {full_len} chars", ""]
        if truncated:
            lines.append(
                f"⚠ analysed the first {MAX_SCAN:,} chars (full file is "
                f"{full_len:,}). Tag / class counts are from the scanned window only."
            )
            lines.append("")
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
        # Always run against the SCANNED window (not the full file) — for
        # huge files this is the difference between 1s and 10 min.
        sample = _find_sample_block(scan, p.classes)
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
        end = min(len(html), start + 30_000)
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


class HtmlFindTool(Tool):
    name = "html_find"
    is_readonly = True
    max_output_chars = 5_000
    description = (
        "Find HTML elements by a CLASS TOKEN (no regex required). Pass "
        "`source` (file path or HTML) and `class_token` (e.g. 'athing' or "
        "'views-row'); the tool returns the first N elements whose `class=` "
        "attribute contains that token (correctly handling multi-class "
        "attributes like `class=\"athing submission\"`). Optional: "
        "`tag` (default any) limits to a specific tag. Use this when the "
        "exact-class regex isn't matching — `html_find(class_token='athing')` "
        "Just Works."
    )
    args_schema = {
        "source":      {"type": "string"},
        "url":         {"type": "string"},
        "class_token": {"type": "string"},
        "tag":         {"type": "string", "description": "Optional: limit to a tag like 'tr', 'div', 'article'."},
        "limit":       {"type": "integer", "default": 5},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        src, err = _resolve_source_arg(state, args)
        if err:
            return ToolResult(output=err, error=True)
        cls = (
            args.get("class_token")
            or args.get("class")
            or args.get("class_name")
            or args.get("cls")
            or args.get("token")
            or ""
        )
        # Also accept CSS-style selectors like "tr.athing" — extract the class.
        sel = args.get("selector") or args.get("css")
        if not cls and isinstance(sel, str):
            # Take the LAST class in a comma list (model usually intends the
            # most-specific). e.g. "tr.athing, tr.source" → "athing".
            sel_first = sel.split(",")[0].strip()
            m = re.search(r"\.([\w-]+)", sel_first)
            if m:
                cls = m.group(1)
                # Also infer tag if present
                tag_m = re.match(r"^([a-zA-Z][\w-]*)", sel_first)
                if tag_m and not args.get("tag"):
                    args["tag"] = tag_m.group(1)
        if not cls:
            return ToolResult(
                output=(
                    "ERROR: pass `class_token` (e.g. 'athing') OR `selector` "
                    "(e.g. 'tr.athing'). The tool finds repeating elements by "
                    "a CSS class TOKEN — no full CSS selector engine."
                ),
                error=True,
            )
        limit = int(args.get("limit") or 5)
        tag = (args.get("tag") or r"\w+").strip()
        try:
            html = _load_or_raise(src, state)
        except _SourceNotFoundError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)

        # Match elements: <tag ... class="...token..."...> ... </tag>
        # We don't try to balance nested same-tag elements perfectly —
        # for repeating elements (tr/div/li) we want the closing of the
        # OUTER instance, but a non-greedy match returns the closest closing
        # tag which is the right answer for >95% of real-world cases.
        pattern = (
            rf'<({tag})\b[^>]*\bclass="[^"]*\b{re.escape(cls)}\b[^"]*"[^>]*>'
            r'(.*?)'
            rf'</\1>'
        )
        try:
            rx = re.compile(pattern, re.DOTALL | re.IGNORECASE)
        except re.error as e:
            return ToolResult(output=f"ERROR: regex compile: {e}", error=True)
        matches = list(rx.finditer(html))
        if not matches:
            # Fall back to just counting class hits — useful diagnosis
            count = len(re.findall(rf'class="[^"]*\b{re.escape(cls)}\b[^"]*"', html))
            return ToolResult(output=(
                f"0 elements with class token '{cls}' and tag '{tag}'.\n"
                f"  Class-token total hits in the page: {count}\n"
                "  TIP: maybe try a different tag, or call html_inspect to "
                "see what's available."
            ), artifact={"matches": 0})
        sample = matches[:limit]
        out = [f"total_matches: {len(matches)}", f"tag: {tag}  class_token: {cls}"]
        for i, m in enumerate(sample):
            block = m.group(0)
            if len(block) > 1200:
                block = block[:1200] + "…"
            out.append(f"--- match {i} ({len(m.group(0))} chars) ---")
            out.append(block)
        # Suggest a row_pattern the agent can drop into extractor_define.
        out.append("")
        out.append(
            f"You can use this regex as your row_pattern: "
            f'`<{tag}\\b[^>]*\\bclass="[^"]*\\b{re.escape(cls)}\\b[^"]*"[^>]*>(.*?)</{tag}>`'
        )
        return ToolResult(output="\n".join(out), artifact={"matches": len(matches)})


class HtmlExtractTool(Tool):
    name = "html_extract"
    is_readonly = True
    max_output_chars = 5_000
    summary_fields = ("total_matches",)
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
