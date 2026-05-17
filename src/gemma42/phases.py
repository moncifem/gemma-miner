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
import hashlib
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
        "Site uses an HTML <table> with one <tr> per row (common on public, regulatory, "
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
        "Site is built on Drupal/Views — common on public-sector and regulator pages "
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
    # Check the autobiography for priors *before* the generic discover spiel,
    # so the model sees the most actionable advice first.
    prior_hint = ""
    try:
        from gemma42.autobiography.store import global_db, project_db

        pdb = project_db(state.workdir); gdb = global_db()
        try:
            ps, gs = pdb.stats(), gdb.stats()
            n_sites = ps.get("sites", 0) + gs.get("sites", 0)
            n_recipes = ps.get("recipes", 0) + gs.get("recipes", 0)
        finally:
            pdb.close(); gdb.close()
        if n_sites + n_recipes > 0:
            prior_hint = (
                f"📚 AUTOBIOGRAPHY: {n_sites} site(s) and {n_recipes} recipe(s) "
                "in long-term memory.\n"
                "  STEP 0: call `autobiography_recall` with a keyword from the "
                "goal. If a related codebook or lesson exists, read it before "
                "anything else.\n"
                "  STEP 0b (after http_get): call `fingerprint_check` — if it "
                "returns an EXACT or NEAR match with cached_recipes, ADOPT "
                "that spec via `extractor_define` and skip the rest of this "
                "phase.\n\n"
            )
    except Exception:  # noqa: BLE001
        pass

    base = (
        prior_hint
        + "RECOMMENDED FIRST MOVE: identify the ITEM type and repeating unit. "
        "Use `llm_scrape` when the listing structure is messy; it reads the "
        "page and returns rows as JSON while preserving detail_url/evidence "
        "links when requested.\n\n"
        "  llm_scrape(\n"
        '    source="<cache_path returned by http_get>",\n'
        '    fields=[{"name":"title"}, {"name":"score","type":"integer"},\n'
        '            {"name":"comments","type":"integer"},\n'
        '            {"name":"detail_url","description":"canonical item detail page if present"}],\n'
        "    target=30,\n"
        '    context="one row per primary item requested by the user",\n'
        "    push_to_dataset=true\n"
        "  )\n\n"
        "ALTERNATE PATH (for large models or simple/cached templates): "
        "`extractor_define` writes a regex spec. Fast and free when it works, "
        "but if it returns matched_rows: 0 even once, switch to `llm_scrape` "
        "above — do NOT iterate on the regex.\n\n"
        "Common field-regex traps:\n"
        "  - For HTML-wrapped text (e.g. `<div class=\"list-title\">...<span>real title</span></div>`), "
        "use transform='strip_html' to drop inner tags and collapse whitespace.\n"
        "  - detail_url: don't grab the FIRST `<a href=...>` — that's usually "
        "a category/filter link. Look for an `about` attribute or an "
        "<h2>/<h3> heading link.\n"
        "  - id: anchor on a specific class, not a loose pattern like "
        "`[0-9]{2}[A-Z][0-9]{2}` (matches timestamp fragments).\n"
        "  - date YYYY-MM-DD: use `datetime=\"([^\"T]+)` to stop before the "
        "'T' in ISO timestamps.\n"
        "  - END the phase by calling `recipe_save` with the working spec so "
        "future runs against this site/template start instantly.\n\n"
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


def _codebook_contract_ok(state: "AgentState") -> bool:
    try:
        from gemma42.contracts import CodebookContract

        contracts = [c for c in state.contracts.list() if isinstance(c, CodebookContract)]
        if not contracts:
            return True
        return all(c.check(state.dataset)[0] for c in contracts)
    except Exception:  # noqa: BLE001
        return True


def _codebook_needs_extraction(state: "AgentState") -> bool:
    """True when a saved codebook has not been applied to current rows.

    This prevents a run from exporting after replacing/refining a codebook but
    before re-running schema extraction for that new schema.
    """
    codebook_path = Path(state.workdir) / "codebook.json"
    if not codebook_path.exists() or len(state.dataset) == 0:
        return False
    try:
        from gemma42.codebook import Codebook

        cb = Codebook.load(codebook_path)
        var_names = [v.name for v in cb.variables]
        if not var_names:
            return False
        current_hash = hashlib.sha256(codebook_path.read_bytes()).hexdigest()
        if state.memory.get("last_extracted_codebook_hash") != current_hash:
            return True
        rows = state.dataset.rows()
        sample = rows[: min(10, len(rows))]
        # If none of the sampled rows even carries a codebook key, extraction
        # definitely has not run for this schema.
        if not any(any(n in r for n in var_names) for r in sample):
            return True
        return False
    except Exception:  # noqa: BLE001
        return True


PHASES: dict[str, Phase] = {
    "DISCOVER_LISTING": Phase(
        name="DISCOVER_LISTING",
        goal=(
            "Identify the site's ITEM type and repeating unit, then get item "
            "rows into the queue or dataset. An item is the primary entity "
            "the user wants one dataset row for; preserve detail_url and "
            "external evidence links when present."
        ),
        tools=(
            "autobiography_recall",
            "fingerprint_check",
            "http_get",
            "set_plan",
            "show_plan",
            "extractor_define",
            "llm_scrape",
            "html_find",
            "html_inspect",
            "python",
            "recipe_save",
        ),
        hint=(
            "STEP 1 — LOOK at the page. Use the http_get preview + html_inspect's\n"
            "top tags & classes to find the REPEATING UNIT. Items typically live\n"
            "in one of:\n"
            "  - <tr>  rows inside <table> / <tbody>\n"
            "  - <div> blocks sharing a class (`views-row`, `card`, `post`, ...)\n"
            "  - <article>, <li>, or <section> with a stable class\n"
            "  - JSON embedded in a <script> tag or hidden API call\n"
            "Whichever it is, the inspect output will show a high-count tag or class.\n\n"
            "STEP 2 — CLASSIFY THE ITEM. Write down (in your `thought`):\n"
            "  • what one row represents (one sanction, one paper, one filing, …)\n"
            "  • which fields are visible on the listing\n"
            "  • whether you'll need a detail page per item, or attachments, or neither\n\n"
            "STEP 3 — WRITE A SPEC. Anchor `row_pattern` on the REAL repeating\n"
            "container you observed. Each field regex should reference markup\n"
            "UNIQUE to that field — class names, attribute names, surrounding tags\n"
            "— so it doesn't accidentally match the wrong column.\n\n"
            "Common regex pitfalls (universal, not site-specific):\n"
            "  • the FIRST `<a href>` in a row is often a sector/filter link, not\n"
            "    the detail link. If multiple links exist, pick the one anchored to\n"
            "    an explicit attribute (`about=`, `data-href=`, `itemprop=\"url\"`)\n"
            "    or a heading link (`<h2><a>`, `<h3><a>`).\n"
            "  • Don't reuse the same column regex for multiple fields — they'll all\n"
            "    capture the same value. Either anchor each on its class, or use a\n"
            "    positional template (the auto-fixer will sometimes do this for you,\n"
            "    but writing it right the first time is faster).\n"
            "  • For ISO dates that include a time, use `datetime=\"([^\"T]+)\"` to\n"
            "    stop at the 'T' so you get YYYY-MM-DD.\n\n"
            "The auto-test prints `matched_rows`, the first 2 rows extracted, and the\n"
            "RAW HTML of row 0 — read that block before re-writing the spec.\n\n"
            "Phase done when: matched_rows ≥ 1 row per visible item AND no NULL\n"
            "fields in row 0 AND no SUSPICIOUS warnings."
        ),
    ),
    "ENUMERATE": Phase(
        name="ENUMERATE",
        goal="Populate the queue by running the listing extractor across paginated URLs OR by hitting an API directly.",
        tools=(
            "scrape_paginated",
            "queue_status",
            "queue_add",
            "http_get",
            "python",
            "read_file",
            "show_plan",
            "set_plan",
        ),
        hint=(
            "DO THE OBVIOUS THING FIRST. If a 'listing' extractor is already\n"
            "saved (see `Extractors saved` in the brief), call:\n\n"
            "  scrape_paginated(\n"
            "    url_template='<base>?page={page}'  (or whatever the plan says),\n"
            "    extractor_name='listing',\n"
            "    target_count=<plan.target_rows>\n"
            "  )\n\n"
            "That ONE call paginates, dedupes, and queues 1000 rows in seconds.\n"
            "It is the right tool for almost every paginated site.\n\n"
            "DO NOT re-implement pagination yourself by hand-rolling a Python\n"
            "loop + llm_scrape on each page. That's how the previous run thrashed\n"
            "for 39 turns and produced 30 rows: each llm_scrape stored 30 items\n"
            "in a transient artifact instead of the dataset, and after dedup the\n"
            "dataset never grew.\n\n"
            "FALLBACK paths, in order of preference:\n"
            "  1. scrape_paginated  — almost always the answer when the listing\n"
            "     extractor already matches >0 rows.\n"
            "  2. Python loop calling http_get for each page + dataset_append in\n"
            "     a single batch — only when pagination doesn't fit a simple\n"
            "     `{page}` template (cursor-based, offset+limit API, etc.).\n"
            "  3. llm_scrape(source=..., push_to_dataset=true) — only when there\n"
            "     is no clean repeating HTML structure to write a spec against.\n\n"
            "If scrape_paginated is rejected (e.g. 'extractor produced 0 rows on\n"
            "cached page'), fix the extractor with extractor_define first; don't\n"
            "switch to a different harvest mechanism.\n\n"
            "When `dataset_rows >= plan.target_rows`, the phase is done."
        ),
    ),
    "DISCOVER_DETAIL": Phase(
        name="DISCOVER_DETAIL",
        goal=(
            "Inspect one queued item's detail page. Decide whether details "
            "are scalar HTML fields, one primary attachment, or multiple "
            "dependencies (PDF/XML/JSON/CSV/TAR/ZIP/external pages) that "
            "must be harvested for the item."
        ),
        tools=(
            "queue_next",
            "fingerprint_check",
            "http_get",
            "discover_assets",
            "extractor_define",
            "python",
            "recipe_save",
        ),
        hint=(
            "Do this on one representative item:\n"
            "  1. queue_next() to get one item; http_get its detail_url.\n"
            "  2. discover_assets(source='<detail cache path>', base_url='<detail_url>') "
            "to classify PDFs, XML, JSON, CSV, archives, and external evidence pages.\n"
            "  3. If there is one primary attachment, define a 'detail' spec "
            "with attachment_url. If there are varied assets, skip the detail "
            "spec and use process_queue(mode='multi_asset') in PROCESS.\n\n"
            "Primary-attachment template (adjust the regex to match the "
            "actual attachment URL pattern you see in the cached detail HTML):\n"
            "  extractor_define(\n"
            "    name='detail',\n"
            "    spec={\n"
            '      "fields": {\n'
            '        "attachment_url": {"regex": "href=\\"(https?://[^\\"]+\\\\.pdf)\\""}\n'
            "      }}\n"
            "  )\n\n"
            "If discover_assets shows multiple meaningful assets, do not "
            "force them into one attachment_url. Move to PROCESS and call "
            "process_queue(mode='multi_asset', batch_size=5)."
        ),
    ),
    "PROCESS": Phase(
        name="PROCESS",
        goal=(
            "Turn the queued items into dataset rows. Decide the smallest "
            "harvest path that delivers every variable the codebook will "
            "need: listing-only → detail-text → detail+attachments → "
            "multi-asset crawl. Don't fetch what you don't need."
        ),
        tools=(
            "dataset_from_queue",
            "process_queue",
            "discover_assets",
            "assess_sample",
            "dataset_stats",
            "queue_status",
            "dataset_sample",
            "python",
            "dataset_append",
            "read_file",
            "extract_text",
        ),
        hint=(
            "Pick the harvest mode by what the user needs:\n\n"
            "  A. Listing already has every required field →\n"
            "       dataset_from_queue()   (no detail fetches)\n\n"
            "  B. Detail page has extra HTML fields (title, body) but no "
            "files →\n"
            "       process_queue(detail_extractor='detail', mode='text', \n"
            "                     row_template={…})\n\n"
            "  C. Detail page has ONE clear PDF/attachment →\n"
            "       process_queue(detail_extractor='detail', mode='attachment',\n"
            "                     row_template={'pdf_text': '$file:{paths.text}'})\n\n"
            "  D. Detail pages have VARIED assets (PDF + XML + CSV + …) "
            "and you want every variable the page exposes →\n"
            "       process_queue(mode='multi_asset', batch_size=5)\n"
            "     (no detail_extractor needed; the tool discovers and "
            "harvests every data link on each detail page automatically)\n\n"
            "To preview what mode D would find on one item, call:\n"
            "  discover_assets(url='<one detail page URL>')\n\n"
            "If a call returns 0 appended with N errors twice in a row, "
            "downgrade (D → C → B → A) — don't keep retrying the same mode."
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
            "codebook_design",
            "codebook_show",
            "codebook_edit",
            "codebook_test",
            "dataset_sample",
        ),
        hint=(
            "Single call: `codebook_design(sample_size=4, domain_hint='<one line>')`\n"
            "This runs the full adversarial pipeline in one tool call: a Curator "
            "drafts 30-60 variables, an Adversary critiques them (drops dead "
            "ones, retypes strings to enums, tightens descriptions with negative "
            "examples), a Consolidator merges, and the result is saved to "
            "<workdir>/codebook.json.\n\n"
            "After it returns, inspect with `codebook_show`. If you want to "
            "tweak, use `codebook_edit`. Then move to EXTRACT — no need to "
            "test separately, codebook_design has already grounded the spec."
        ),
    ),
    "EXTRACT": Phase(
        name="EXTRACT",
        goal=(
            "Apply the locked codebook to EVERY item in the dataset, filling "
            "the structured columns via LLM extraction + deterministic type "
            "coercion."
        ),
        tools=("extract_items", "assess_sample", "codebook_edit",
               "dataset_validate", "audit_dataset",
               "rules_infer", "dataset_stats"),
        hint=(
            "EXTRACT is a pilot-then-scale phase. Concretely:\n"
            "  1. `extract_items()` — runs a 3-row pilot AUTOMATICALLY on the\n"
            "     first call. Read the per-variable coverage at the bottom of\n"
            "     the output and the PILOT verdict (SCALE_OK | FIX_FIRST).\n"
            "  2. If PILOT says SCALE_OK → call `extract_items(limit=null)` to\n"
            "     extract the remaining rows (skip_existing keeps the pilot\n"
            "     rows). Run `assess_sample(layer='silver')` mid-way through\n"
            "     a big run if you want a quality re-check.\n"
            "  3. If PILOT says FIX_FIRST → call `codebook_edit` to drop or\n"
            "     rewrite the under-filled variables (look for 0% coverage),\n"
            "     then re-pilot with `extract_items(limit=3, skip_existing=false)`.\n"
            "  4. (optional) `rules_infer(auto_add=true)` — propose constitutional "
            "rules from the observed distributions; auto-adds the safe ones.\n"
            "  5. (optional) `audit_dataset(sample_size=6)` — sample 6 rows, "
            "send (extracted, source_text) to the Auditor specialist; get "
            "per-row confidence + common failure themes.\n"
            "4. `dataset_validate` — final per-variable stats. If a variable "
            "has <30% coverage, `codebook_edit` to drop it."
        ),
    ),
    "EXPORT": Phase(
        name="EXPORT",
        goal=(
            "Write the dataset to Parquet + JSONL + codebook.md (auto-"
            "synthesised if there isn't one), then call `finish`."
        ),
        tools=("dataset_export", "dataset_validate", "manifest_write",
               "recipe_save", "autobiography_remember", "finish",
               "hf_push", "dataset_report", "dataset_verify"),
        hint=(
            "Minimal happy path (use these 2 calls, then you're done):\n"
            "  1. `dataset_export()`     — writes parquet + jsonl + codebook "
            "(auto-synthesised if needed) under <workdir>/export/.\n"
            "  2. `finish(summary='produced dataset X with Y rows')`\n\n"
            "Optional extras (call only if the user asked for them):\n"
            "  • `dataset_validate`     — per-variable stats.\n"
            "  • `dataset_report`       — Statistician writes findings.md.\n"
            "  • `manifest_write`       — drop a gemma42.lock for repro.\n"
            "  • `recipe_save`          — persist your extractor for next time.\n"
            "  • `hf_push(repo_id=...)` — publish to Hugging Face.\n\n"
            "DO NOT call codebook_design here unless the user explicitly "
            "asked for statistical variables. If you're stuck on a contract "
            "failure (e.g. a few rows missing a field), just call "
            "`dataset_export` + `finish` — the export tool tolerates it."
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

    # Contracts satisfied → FINISH, except when a newly saved/refined codebook
    # still needs to be applied to the rows.
    if (
        state.contracts.list()
        and state.contracts.all_satisfied(state.dataset)
        and not _codebook_needs_extraction(state)
    ):
        return PHASES["FINISH"]

    target = contract_min_rows or 50
    n_rows = len(state.dataset)

    # SOFT-MET threshold: if we have ≥80% of the target AND the agent has
    # been spinning (≥3 consecutive errors or duplicate-only scrapes), don't
    # keep them stuck in DISCOVER_LISTING for the last few rows. Ship what
    # we have.
    soft_target = max(1, int(target * 0.8))
    if n_rows >= soft_target and n_rows < target:
        recent = state.history[-5:]
        if recent:
            recent_errors = sum(1 for h in recent if h.error)
            recent_dup_scrapes = sum(
                1 for h in recent
                if h.tool == "llm_scrape" and "duplicates skipped" in (h.observation or "")
                and "appended to dataset: 0" in (h.observation or "")
            )
            if recent_errors >= 3 or recent_dup_scrapes >= 1:
                return PHASES["EXPORT"]

    # If we have enough rows already (e.g. via `llm_scrape` directly into the
    # dataset), we are PAST the discover/harvest stage even without a saved
    # extractor spec. Don't loop back into DISCOVER_LISTING when the data
    # is already there.
    if n_rows >= target:
        # Decide between CODEBOOK / EXPORT / FINISH.
        from pathlib import Path

        codebook_path = Path(state.workdir) / "codebook.json"
        # If the user didn't ask for a codebook, contract failures are
        # field-name mismatches → just go to FINISH/EXPORT.
        if not codebook_path.exists():
            wants_codebook = bool(state.memory.get("wants_codebook"))
            sample = state.dataset.rows()[:5]
            # 1. Always go to CODEBOOK if there's heavy text content (PDF body, etc).
            has_long_text = any(
                isinstance(r.get(f), str) and len(r.get(f, "")) > 200
                for r in sample
                for f in ("pdf_text", "text", "body", "content")
            ) or any(
                isinstance(r.get(f), str) and r.get(f) for r in sample
                for f in ("text_path", "txt_path", "txt_file")
            )
            if has_long_text:
                return PHASES["CODEBOOK"]
            # 2. If the user wants a codebook, trigger as long as ANY non-id
            # string field has free-form-ish content (≥30 chars or contains
            # spaces/punctuation that suggest it's decomposable).
            if wants_codebook:
                def _looks_decomposable(v: object) -> bool:
                    if not isinstance(v, str) or not v:
                        return False
                    if len(v) >= 30:
                        return True
                    # Short but contains words AND non-trivial punctuation /
                    # multiple tokens — e.g. "Amende 27 M€ et injonction"
                    return " " in v.strip() and any(c in v for c in "€$%,.()")
                has_decomposable = any(
                    _looks_decomposable(r.get(f))
                    for r in sample
                    for f in r.keys()
                    if f != "id" and not str(f).startswith("_")
                )
                if has_decomposable:
                    return PHASES["CODEBOOK"]
            return PHASES["EXPORT"]
        if not _codebook_contract_ok(state):
            return PHASES["CODEBOOK"]
        if _codebook_needs_extraction(state):
            return PHASES["EXTRACT"]
        # codebook exists and has been extracted; move toward export/finish.
        return PHASES["EXPORT"]

    # The four scraping phases.
    if not has_listing:
        base = PHASES["DISCOVER_LISTING"]
        return Phase(
            name=base.name, goal=base.goal, tools=base.tools,
            hint=_dynamic_listing_hint(state),
        )

    if remaining < max(1, target - n_rows):
        return PHASES["ENUMERATE"]

    # NEW: if the queue items already contain every field the user asked for
    # (or a known variant — e.g. "comments" satisfies "n_comments"), skip
    # DISCOVER_DETAIL entirely and head straight to PROCESS, where
    # `dataset_from_queue` will harvest them in one call. No detail pages
    # needed.
    queue_satisfies_required = False
    if remaining > 0:
        try:
            from gemma42.contracts import FieldsContract, _field_variants

            required: list[str] = []
            for c in state.contracts.list():
                if isinstance(c, FieldsContract):
                    required.extend(c.required_fields)
            if required:
                # Sample the first unprocessed queue item.
                sample = next(
                    (q for q in queue
                     if isinstance(q, dict) and str(q.get("id")) not in processed),
                    None,
                )
                if sample is not None:
                    queue_keys = set(sample.keys())
                    queue_satisfies_required = all(
                        any(v in queue_keys for v in _field_variants(f))
                        for f in required
                    )
        except Exception:  # noqa: BLE001
            queue_satisfies_required = False

    if queue_satisfies_required:
        return PHASES["PROCESS"]

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
        if not _codebook_contract_ok(state):
            return PHASES["CODEBOOK"]
        if _codebook_needs_extraction(state):
            return PHASES["EXTRACT"]
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
