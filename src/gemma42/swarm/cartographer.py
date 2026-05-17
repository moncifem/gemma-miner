"""Cartographer: maps a site's URL + DOM structure to a short narrative.

Output:
  {
    "kind": "drupal_views" | "table" | "wp_article" | "li_card" | "rss_atom" | "json_api" | "generic_div",
    "summary":     "<one paragraph>",
    "pagination":  "drupal_zero_indexed" | "wp_slash_page" | "offset" | "none" | "unknown",
    "url_template": "...{page}..." (if applicable, else null),
    "candidate_row_classes": ["views-row", ...],
    "candidate_detail_url_attr": "about" | "href" | "data-href",
    "needs_js_render": false,
    "language": "fr" | "en" | "..."
  }

Used in the DISCOVER_LISTING phase BEFORE writing an extractor spec.
"""

from __future__ import annotations

from typing import Any

from gemma42.swarm.base import llm_json


_SYS = """You are a SITE CARTOGRAPHER. Given a small sample of HTML, classify the page's structure and produce a structured map.

Output ONE JSON object (no prose) with:
  - kind: one of {"drupal_views","table","wp_article","li_card","rss_atom","json_api","generic_div"}
  - summary: one sentence
  - pagination: one of {"drupal_zero_indexed","wp_slash_page","offset","none","unknown"}
  - url_template: example URL with {page} placeholder, or null
  - candidate_row_classes: array of likely repeating-row CSS class tokens
  - candidate_detail_url_attr: which HTML attribute holds the canonical detail URL ("about", "href", "data-href", etc.)
  - needs_js_render: boolean
  - language: ISO short code

Be DECISIVE. Pick ONE kind even if uncertain; the agent will verify.
"""


def map_site(llm: Any, html_sample: str, *, url: str = "") -> dict:
    user = (
        (f"URL: {url}\n\n" if url else "")
        + "HTML SAMPLE (first ~25KB):\n<<<\n"
        + html_sample[:25_000]
        + "\n>>>"
    )
    return llm_json(llm, _SYS, user, temperature=0.1)
