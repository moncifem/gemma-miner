"""Phase machine for the scrape-with-details playbook.

A phase narrows the agent's choices to a handful of relevant tools and tells
it the *one thing* it needs to accomplish before advancing. Small models
behave dramatically better when the prompt says "do X" instead of "here are
24 tools, figure it out."

The phase is determined automatically each turn from observable state:

  DISCOVER_LISTING:  no listing extractor saved yet
  ENUMERATE:         listing extractor exists, but queue (remaining) too small
  DISCOVER_DETAIL:   queue has items, but no detail extractor
  PROCESS:           detail extractor exists, queue has items
  FINISH:            contracts all satisfied

The DISCOVER_LISTING hint is computed dynamically: we sniff the most recent
cached HTML and pick a pattern template appropriate to its structure
(`<table>` rows, Drupal Views `<div class="views-row">`, WordPress
`<article>`, etc.). This makes the agent succeed on heterogeneous sites
without manual hint engineering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma42.state import AgentState


@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    tools: tuple[str, ...]
    hint: str


# ── Pattern detection + per-pattern templates ─────────────────────────────


def _newest_html_path(state: "AgentState") -> Path | None:
    cache = Path(state.workdir) / "cache"
    if not cache.exists():
        return None
    htmls = [p for p in cache.glob("*.html") if p.is_file()]
    if not htmls:
        return None
    htmls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return htmls[0]


def _detect_listing_pattern(html: str) -> str:
    """Return a short identifier for the listing structure: 'table', 'drupal_views',
    'wp_article', or 'generic_div'."""
    sample = html[:200_000]
    # Count occurrences of the markers; pick the dominant one.
    n_tr = len(re.findall(r"<tr\b", sample, re.IGNORECASE))
    n_views_row = len(re.findall(r'class="[^"]*\bviews-row\b', sample))
    n_search_index = len(re.findall(r'class="[^"]*\bsearch-index\b', sample))
    n_article = len(re.findall(r"<article\b", sample, re.IGNORECASE))
    n_li_post = len(re.findall(r'<li[^>]*class="[^"]*\b(post|item|card|result)\b', sample))
    # Highest-confidence patterns first.
    if n_views_row >= 5 and n_search_index >= 3:
        return "drupal_views"
    if n_tr >= 20 and n_tr > n_views_row * 2 and n_tr > n_article * 2:
        return "table"
    if n_article >= 5:
        return "wp_article"
    if n_li_post >= 5:
        return "li_card"
    return "generic_div"


_LISTING_TEMPLATES: dict[str, str] = {
    "table": (
        "Site uses an HTML <table> with one <tr> per row (CNIL, gov sites, "
        "static reports). Each row's data lives in <td> columns. Example:\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<tr[^>]*>(.*?)</tr>",\n'
        '      "exclude_substring": "<th",                # skip the header row\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "date":         {"regex": "<td[^>]*>([0-9/.-]+)</td>"},\n'
        '        "title":        {"regex": "<td[^>]*>([^<]+?)</td>", "group": 1, "transform": "strip"},\n'
        '        "detail_url":   {"regex": "href=\\"(https?://[^\\"]+|/[^\\"]+)\\"", "prefix_base": true},\n'
        "      }}\n"
        "  )\n"
        "TIP: <td> columns appear in order. If the columns are date/organisation/"
        "violations/decision, capture each one with its own field regex, or use "
        "an indexed approach with python first if the regex gets too hairy."
    ),
    "drupal_views": (
        "Site is built on Drupal/Views — common on .gouv.fr, ADLC, CNIL pages "
        "that have search filters. Each row is `<div class=\"views-row\">` "
        "with internal classes like `field--name-field-id`, `field--item`, "
        "and an `about=\"/...\"` attribute on the inner div = the canonical URL.\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<div class=\\"views-row\\">(.*?)(?=<div class=\\"views-row\\">|$)",\n'
        '      "include_substring": "search-index",   # narrow if needed\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "id":            {"regex": "field--name-field-id[^>]*>([^<]+)<", "transform": "strip"},\n'
        '        "detail_url":    {"regex": "about=\\"([^\\"]+)\\"", "prefix_base": true},\n'
        '        "title":         {"regex": "<br/>\\\\s*([^<]+?)\\\\s*</a>", "transform": "strip"},\n'
        '        "date_decision": {"regex": "datetime=\\"([^\\"T]+)"}\n'
        "      }}\n"
        "  )\n"
        "TRAP: never use the FIRST `href=\"...\"` for detail_url — that's a "
        "sector/filter link. Use the `about` attribute or an <h2><a> link."
    ),
    "wp_article": (
        "Site uses semantic <article> blocks (WordPress and similar).\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<article\\\\b[^>]*>(.*?)</article>",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "detail_url":    {"regex": "<a[^>]+href=\\"([^\\"]+)\\"", "prefix_base": true},\n'
        '        "title":         {"regex": "<h\\\\d[^>]*>\\\\s*(?:<a[^>]*>)?([^<]+)", "transform": "strip"},\n'
        '        "date":          {"regex": "datetime=\\"([^\\"T]+)"}\n'
        "      }}\n"
        "  )"
    ),
    "li_card": (
        "Site uses repeating <li class=\"post|item|card|result\">.\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<li[^>]*class=\\"[^\\"]*(?:post|item|card|result)[^\\"]*\\"[^>]*>(.*?)</li>",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "detail_url":    {"regex": "<a[^>]+href=\\"([^\\"]+)\\"", "prefix_base": true},\n'
        '        "title":         {"regex": "<a[^>]*>([^<]+)</a>", "transform": "strip"}\n'
        "      }}\n"
        "  )"
    ),
    "generic_div": (
        "Couldn't auto-detect a common pattern. Look at the sample block in "
        "the html_inspect output and write a spec around its outer-most "
        "repeating class. The general shape:\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<TAG class=\\"REPEATING_CLASS\\">(.*?)</TAG>",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "title":      {"regex": "..."},\n'
        '        "detail_url": {"regex": "href=\\"([^\\"]+)\\"", "prefix_base": true}\n'
        "      }}\n"
        "  )"
    ),
}


def _dynamic_listing_hint(state: "AgentState") -> str:
    """Compose the DISCOVER_LISTING hint using whatever cached HTML exists."""
    base = (
        "STOP INVESTIGATING. WRITE THE SPEC NOW.\n\n"
        "After ONE http_get you have enough. The `extractor_define` tool "
        "auto-tests against the cached HTML, prints sample rows, AND shows "
        "the raw HTML of row 0 so you can iterate. Aim for one or two "
        "calls to land a clean spec.\n\n"
        "Common field-regex traps:\n"
        "  - detail_url: don't grab the FIRST `<a href=...>` — that's usually "
        "a category/filter link. Look for an `about` attribute or an "
        "<h2>/<h3> heading link.\n"
        "  - id: anchor on a specific class, not a loose pattern like "
        "`[0-9]{2}[A-Z][0-9]{2}` (matches timestamp fragments).\n"
        "  - date YYYY-MM-DD: use `datetime=\"([^\"T]+)` to stop before the "
        "'T' in ISO timestamps.\n\n"
    )
    p = _newest_html_path(state)
    if p is None:
        return base + "Once you've fetched a listing page, the system will detect its pattern and show you a tailored template."
    try:
        html = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return base + _LISTING_TEMPLATES["generic_div"]
    pattern = _detect_listing_pattern(html)
    pretty = {
        "table": "HTML <table>",
        "drupal_views": "Drupal/Views <div class='views-row'>",
        "wp_article": "WordPress <article>",
        "li_card": "<li class='post|item|card|result'>",
        "generic_div": "generic",
    }[pattern]
    return (
        base
        + f"Detected pattern in {p.name}: **{pretty}**\n\n"
        + _LISTING_TEMPLATES[pattern]
    )


PHASES: dict[str, Phase] = {
    "DISCOVER_LISTING": Phase(
        name="DISCOVER_LISTING",
        goal=(
            "Define a LISTING extractor — a JSON spec describing how to slice "
            "a listing page into row blocks and pull per-row fields."
        ),
        tools=(
            "http_get",
            "extractor_define",
            "html_inspect",
            "python",
        ),
        hint=(
            "STOP INVESTIGATING. WRITE THE SPEC NOW.\n\n"
            "After ONE http_get you have enough. The `extractor_define` tool "
            "auto-tests your spec against the cached HTML, shows the first "
            "rows it extracted, AND prints the raw HTML of row 0 so you can "
            "fix bad field regexes by looking at the actual data.\n\n"
            "Common field-regex traps (read before writing):\n"
            "  - `detail_url`: the FIRST <a href> in a row is usually a "
            "sector/filter link, not the detail link. Look for an `about` "
            "attribute on the row container, or an <h2><a> link. NEVER use "
            "the first generic `href=\"...\"` if there are multiple links.\n"
            "  - `id`: anchor your regex on a SPECIFIC class. A generic "
            "  `[0-9]{2}[A-Z][0-9]{2}` will match timestamp fragments like "
            "`16T12`. Use `class=\"...field-id...\">([^<]+)<` instead.\n"
            "  - `date`: if you want YYYY-MM-DD, use `datetime=\"([^\"T]+)` "
            "to stop at the 'T'. Otherwise you get the full timestamp.\n\n"
            "Concrete template (Drupal-Views — most common pattern on .fr sites):\n"
            "  extractor_define(\n"
            "    name='listing',\n"
            "    spec={\n"
            '      "row_pattern": "<div class=\\"views-row\\">(.*?)(?=<div class=\\"views-row\\">|$)",\n'
            '      "include_substring": "decision search-index",\n'
            '      "base_url": "https://example.com",\n'
            '      "fields": {\n'
            '        "id":            {"regex": "field--name-field-id[^>]*>([^<]+)<", "transform": "strip"},\n'
            '        "detail_url":    {"regex": "about=\\"([^\\"]+)\\"", "prefix_base": true},\n'
            '        "title":         {"regex": "<br/>\\\\s*([^<]+?)\\\\s*</a>", "transform": "strip"},\n'
            '        "date_decision": {"regex": "datetime=\\"([^\\"T]+)"}\n'
            "      }}\n"
            "  )\n\n"
            "Phase complete when: matched_rows > 0 AND no SUSPICIOUS warnings "
            "AND no NULL fields in row 0. The auto-test will tell you which."
        ),
    ),
    "ENUMERATE": Phase(
        name="ENUMERATE",
        goal="Populate the queue by running the listing extractor across paginated URLs.",
        tools=("scrape_paginated", "queue_status", "http_get"),
        hint=(
            "Steps for this phase:\n"
            "  1. scrape_paginated(url_template='<listing-url>?page={page}', "
            "extractor_name='listing', target_count=<min_rows>). The tool will "
            "fetch pages, dedupe, and queue_add automatically.\n"
            "  2. queue_status to confirm `remaining` >= contract minimum."
        ),
    ),
    "DISCOVER_DETAIL": Phase(
        name="DISCOVER_DETAIL",
        goal=(
            "Define a DETAIL extractor — same shape as the listing extractor, "
            "but no row_pattern. It must produce an `attachment_url` field at "
            "minimum (the binary to download for each item)."
        ),
        tools=(
            "queue_next",
            "http_get",
            "extractor_define",
            "python",
        ),
        hint=(
            "Two steps only:\n"
            "  1. queue_next() to get one item; http_get its detail_url.\n"
            "  2. extractor_define a 'detail' spec. The auto-test will tell "
            "you immediately if it works.\n\n"
            "Template (start here, adjust the regex to match the actual "
            "attachment URL pattern you see in the cached detail HTML):\n"
            "  extractor_define(\n"
            "    name='detail',\n"
            "    spec={\n"
            '      "fields": {\n'
            '        "attachment_url": {"regex": "href=\\"(https?://[^\\"]+\\\\.pdf)\\""}\n'
            "      }}\n"
            "  )\n\n"
            "If 'attachment_url' is null in the auto-test, the regex didn't "
            "match — try a more permissive one (e.g. drop the https? prefix "
            "or use a different file extension)."
        ),
    ),
    "PROCESS": Phase(
        name="PROCESS",
        goal=(
            "Process queued items in batches. Each call of process_queue "
            "handles up to `batch_size` items end-to-end (fetch detail, find "
            "attachment, download, extract text, append row, mark done)."
        ),
        tools=(
            "process_queue",
            "dataset_stats",
            "queue_status",
            "dataset_sample",
        ),
        hint=(
            "Steps for this phase:\n"
            "  1. Call process_queue once with:\n"
            "       detail_extractor='detail',\n"
            "       batch_size=10  (raise gradually if no errors),\n"
            "       row_template={\n"
            '         "id":            "{queue.id}",\n'
            '         "title":         "{queue.title}",\n'
            '         "date_decision": "{queue.date_decision}",\n'
            '         "detail_url":    "{queue.detail_url}",\n'
            '         "pdf_path":      "{paths.attachment}",\n'
            '         "pdf_text":      "$file:{paths.text}"\n'
            "       }\n"
            "     The `$file:` prefix means the .txt content is resolved from "
            "disk at append time — your reply stays small.\n"
            "  2. dataset_stats to see contract progress.\n"
            "  3. Repeat process_queue until all contracts are OK. If errors "
            "  pile up, fall back to per-item python."
        ),
    ),
    "CODEBOOK": Phase(
        name="CODEBOOK",
        goal=(
            "Design a CODEBOOK of 20–60 typed variables that capture all "
            "structured information present in each item's text. Variables "
            "should be mostly numeric/boolean so the final dataset is ready "
            "for statistics and ML."
        ),
        tools=(
            "codebook_propose",
            "codebook_show",
            "codebook_test",
            "codebook_edit",
            "dataset_sample",
        ),
        hint=(
            "Workflow for this phase:\n"
            "  1. codebook_propose(sample_size=4, domain_hint='<one line>'). "
            "This reads the actual texts and proposes 20–60 typed variables.\n"
            "  2. codebook_test(sample_size=3). Per-variable coverage + issues.\n"
            "  3. If coverage is bad, codebook_propose again with feedback, OR "
            "codebook_edit(operation='drop', names=[...]).\n"
            "  4. When ≥60% variables are numeric/boolean and coverage looks "
            "good on the sample, the phase is complete (EXTRACT will fill "
            "every row)."
        ),
    ),
    "EXTRACT": Phase(
        name="EXTRACT",
        goal=(
            "Apply the locked codebook to EVERY item in the dataset, filling "
            "the structured columns via LLM extraction + deterministic type "
            "coercion."
        ),
        tools=("extract_items", "dataset_validate", "dataset_stats"),
        hint=(
            "Just call:\n"
            "  extract_items()         # processes all un-extracted items\n"
            "It runs one LLM call per item (≈ 5–30 s each). Watch the "
            "per-variable coverage in the output. After it finishes, call "
            "dataset_validate to inspect distributions; if any variable has "
            "<30% coverage, you can go back and codebook_edit to drop it."
        ),
    ),
    "EXPORT": Phase(
        name="EXPORT",
        goal=(
            "Write the dataset to Parquet + JSONL + codebook.md. Optionally "
            "push to Hugging Face."
        ),
        tools=("dataset_validate", "dataset_export", "hf_push", "finish"),
        hint=(
            "  1. dataset_validate                — final per-variable stats.\n"
            "  2. dataset_export                  — writes export/<name>.parquet, "
            "export/<name>.jsonl, export/codebook.md, export/codebook.json.\n"
            "  3. (optional) hf_push(repo_id='you/your-dataset')\n"
            "  4. finish(summary='produced dataset X with Y rows × Z variables')"
        ),
    ),
    "FINISH": Phase(
        name="FINISH",
        goal="All contracts satisfied — call finish.",
        tools=("dataset_stats", "contract_status", "finish"),
        hint="finish(summary='built a dataset of N items') — no other work needed.",
    ),
}


def current_phase(state: "AgentState", contract_min_rows: int | None = None) -> Phase:
    """Pick the phase based on observable state.

    Order: DISCOVER_LISTING → ENUMERATE → DISCOVER_DETAIL → PROCESS (harvest)
           → CODEBOOK → EXTRACT → EXPORT → FINISH.
    """
    extractors = state.memory.get("extractors", {}) or {}
    has_listing = any(spec.get("row_pattern") for spec in extractors.values())
    has_detail = any(not spec.get("row_pattern") for spec in extractors.values())
    queue = state.memory.get("queue", []) or []
    processed = set(str(x) for x in (state.memory.get("processed", []) or []))
    remaining = sum(1 for q in queue if isinstance(q, dict) and str(q.get("id")) not in processed)

    # Contracts satisfied → FINISH.
    if state.contracts.list() and state.contracts.all_satisfied(state.dataset):
        return PHASES["FINISH"]

    # The four scraping phases.
    if not has_listing:
        base = PHASES["DISCOVER_LISTING"]
        return Phase(
            name=base.name, goal=base.goal, tools=base.tools,
            hint=_dynamic_listing_hint(state),
        )

    target = contract_min_rows or 50
    if remaining < max(1, target - len(state.dataset)):
        return PHASES["ENUMERATE"]

    if not has_detail:
        return PHASES["DISCOVER_DETAIL"]

    # If the queue still has unprocessed items, keep harvesting.
    has_dataset_rows = len(state.dataset) > 0
    if remaining > 0:
        return PHASES["PROCESS"]

    # Now: harvest is done. Move to the codebook-driven pipeline.
    from pathlib import Path

    codebook_path = Path(state.workdir) / "codebook.json"
    if has_dataset_rows and not codebook_path.exists():
        return PHASES["CODEBOOK"]

    if has_dataset_rows and codebook_path.exists():
        # Check whether extraction has happened: do any rows have codebook columns?
        try:
            from gemma42.codebook import Codebook

            cb = Codebook.load(codebook_path)
            rows = state.dataset.rows()
            sample = rows[: min(5, len(rows))]
            var_names = [v.name for v in cb.variables]
            populated = 0
            for r in sample:
                if any(r.get(n) is not None for n in var_names):
                    populated += 1
            if populated == 0:
                return PHASES["EXTRACT"]
            # Some rows have codebook fields. If the export directory exists,
            # we're done; otherwise the EXPORT phase is next.
            export_dir = Path(state.workdir) / "export"
            if (export_dir / f"{cb.name}.parquet").exists():
                return PHASES["FINISH"]
            return PHASES["EXPORT"]
        except Exception:  # noqa: BLE001
            return PHASES["EXTRACT"]

    return PHASES["PROCESS"]
