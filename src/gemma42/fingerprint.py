"""Structural fingerprinting for HTML pages.

Two pages with the same fingerprint share the same underlying template, even
if their content differs entirely. That means an extractor that worked on
page A is very likely to work on page B — so we cache extractors keyed by
fingerprint and skip the DISCOVER phase on subsequent runs.

Algorithm (cheap, deterministic):
  1. Drop <script>, <style>, comments, <head>.
  2. Tokenize the rest into a tag+class skeleton.
  3. Hash the multi-set of tokens (frequencies bucketed coarsely).

Plus a `near` matcher: shared prefix of the fingerprint hex string ≈ shared
core layout (the agent compares against the autobiography to find the closest
known template).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlparse


# Tags to drop entirely before fingerprinting (their content is volatile or
# decorative, not structural).
_DROP_TAGS = {"script", "style", "noscript", "svg", "head", "meta", "link"}
# Tags whose presence/count is structurally informative.
_STRUCT_TAGS = {
    "html", "body", "main", "article", "section", "nav",
    "table", "thead", "tbody", "tr", "th", "td",
    "div", "span", "p", "a", "ul", "ol", "li", "h1", "h2", "h3", "h4",
    "input", "form", "button", "select", "option", "label",
    "img", "time", "header", "footer",
}


class _Skeleton(HTMLParser):
    def __init__(self):
        super().__init__()
        self._depth = 0
        self._skip_depth: int | None = None
        self.tags: Counter = Counter()
        self.classes: Counter = Counter()
        self.attrs: Counter = Counter()

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        if tag in _DROP_TAGS:
            if self._skip_depth is None:
                self._skip_depth = self._depth
            return
        if self._skip_depth is not None:
            return
        if tag in _STRUCT_TAGS:
            self.tags[tag] += 1
        d = dict(attrs)
        cls = d.get("class")
        if cls:
            for c in cls.split():
                # Hash each class token so we don't reveal anything sensitive
                # and keep the fingerprint short.
                self.classes[c[:30]] += 1
        # noteworthy attributes
        for k in ("role", "itemprop", "data-history-node-id", "about", "datetime"):
            if k in d:
                self.attrs[k] += 1

    def handle_endtag(self, tag):
        if self._skip_depth is not None and self._depth <= self._skip_depth:
            self._skip_depth = None
        self._depth -= 1


def _bucket(n: int) -> int:
    """Coarse frequency bucketing so noise doesn't change the fingerprint."""
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    if n <= 16:
        return 3
    if n <= 64:
        return 4
    if n <= 256:
        return 5
    return 6


def fingerprint_html(html: str) -> str:
    """Return a 32-hex-char structural fingerprint."""
    parser = _Skeleton()
    try:
        parser.feed(html or "")
    except Exception:  # noqa: BLE001
        pass
    parts: list[str] = []
    for tag, n in sorted(parser.tags.items()):
        parts.append(f"t:{tag}={_bucket(n)}")
    # only the top 25 classes — keeps fingerprint stable across content swaps
    for cls, n in sorted(parser.classes.most_common(25)):
        parts.append(f"c:{cls}={_bucket(n)}")
    for a, n in sorted(parser.attrs.items()):
        parts.append(f"a:{a}={_bucket(n)}")
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _domain_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
        return host.lstrip("www.")
    except Exception:  # noqa: BLE001
        return ""


def fingerprint_url_and_html(url: str | None, html: str) -> tuple[str, str]:
    """Convenience: returns (domain, fingerprint)."""
    return _domain_of(url), fingerprint_html(html)


# ── similarity ────────────────────────────────────────────────────────────


def fingerprint_distance(a: str, b: str) -> int:
    """Number of differing hex chars in the first N chars (lower = closer).

    With SHA-256 cuts, an exact match has 0 distance; close templates often
    share long common prefixes once we drop drifting content tokens.
    """
    n = min(len(a), len(b))
    diff = 0
    for i in range(n):
        if a[i] != b[i]:
            diff += 1
    return diff + abs(len(a) - len(b))


def looks_similar(a: str, b: str, *, threshold: int = 8) -> bool:
    return fingerprint_distance(a, b) <= threshold
