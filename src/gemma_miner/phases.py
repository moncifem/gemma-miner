"""Phase machine.

A phase narrows the agent's choices to a handful of relevant tools and tells
it the one thing it needs to accomplish before advancing. The phase is
recomputed each turn from observable state — there's no persistent state
machine, just a function of (extractors, queue, dataset, codebook).

Order:
  DISCOVER_LISTING → ENUMERATE → DISCOVER_DETAIL → PROCESS →
  CODEBOOK → EXTRACT → EXPORT → FINISH
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    tools: tuple[str, ...]
    hint: str


# ── Pattern detection + per-pattern templates (purely structural — no
# domain hardcoding) ────────────────────────────────────────────────────────


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
    sample = html[:200_000]
    n_tr = len(re.findall(r"<tr\b", sample, re.IGNORECASE))
    n_views_row = len(re.findall(r'class="[^"]*\bviews-row\b', sample))
    n_article = len(re.findall(r"<article\b", sample, re.IGNORECASE))
    n_li_post = len(re.findall(r'<li[^>]*class="[^"]*\b(post|item|card|result)\b', sample))
    if n_views_row >= 5:
        return "drupal_views"
    if n_tr >= 20 and n_tr > n_article * 2:
        return "table"
    if n_article >= 5:
        return "wp_article"
    if n_li_post >= 5:
        return "li_card"
    return "generic_div"


_LISTING_TEMPLATES: dict[str, str] = {
    "table": (
        "Site uses an HTML <table> with one <tr> per row. Example:\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<tr[^>]*>(.*?)</tr>",\n'
        '      "exclude_substring": "<th",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "title":      {"regex": "<td[^>]*>([^<]+?)</td>", "transform": "strip"},\n'
        '        "detail_url": {"regex": "href=\\"([^\\"]+)\\"", "prefix_base": true}\n'
        "      }}\n"
        "  )"
    ),
    "drupal_views": (
        "Site uses Drupal/Views — `<div class=\"views-row\">` repeats per item.\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<div class=\\"views-row\\">(.*?)(?=<div class=\\"views-row\\">|$)",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "title":      {"regex": "<h\\\\d[^>]*>\\\\s*(?:<a[^>]*>)?([^<]+)", "transform": "strip"},\n'
        '        "detail_url": {"regex": "about=\\"([^\\"]+)\\"", "prefix_base": true}\n'
        "      }}\n"
        "  )"
    ),
    "wp_article": (
        "Site uses semantic <article> blocks.\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<article\\\\b[^>]*>(.*?)</article>",\n'
        '      "base_url": "https://example.com",\n'
        '      "fields": {\n'
        '        "detail_url": {"regex": "<a[^>]+href=\\"([^\\"]+)\\"", "prefix_base": true},\n'
        '        "title":      {"regex": "<h\\\\d[^>]*>\\\\s*(?:<a[^>]*>)?([^<]+)", "transform": "strip"}\n'
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
        '        "detail_url": {"regex": "<a[^>]+href=\\"([^\\"]+)\\"", "prefix_base": true},\n'
        '        "title":      {"regex": "<a[^>]*>([^<]+)</a>", "transform": "strip"}\n'
        "      }}\n"
        "  )"
    ),
    "generic_div": (
        "No common pattern auto-detected. Inspect the html_inspect output for "
        "the dominant repeating class and write a spec around it. Shape:\n"
        "  extractor_define(\n"
        "    name='listing',\n"
        "    spec={\n"
        '      "row_pattern": "<TAG class=\\"REPEATING_CLASS\\">(.*?)</TAG>",\n'
        '      "fields": {"title": {"regex": "..."},\n'
        '                 "detail_url": {"regex": "href=\\"([^\\"]+)\\"", "prefix_base": true}}\n'
        "    }\n"
        "  )"
    ),
}


def _dynamic_listing_hint(state: "AgentState") -> str:
    base = (
        "RECOMMENDED FIRST MOVE: identify the ITEM type and repeating unit.\n\n"
        "For JSON APIs or non-HTML sources, use `python` directly:\n"
        "  python(code='import httpx, json\\n"
        "rows = httpx.get(\"https://api.example.com/items?page=1\", "
        "timeout=30).json()\\nprint(json.dumps(rows[:2], indent=2))')\n\n"
        "For messy HTML or when you want the model to do the structural work:\n"
        "  llm_scrape(\n"
        '    source="<cache_path returned by http_get>",\n'
        '    fields=[{"name":"title"}, {"name":"detail_url"}],\n'
        "    target=20,\n"
        '    context="one row per primary item the user wants",\n'
        "    push_to_dataset=true\n"
        "  )\n\n"
        "For clean repeating HTML, write an `extractor_define` spec (cheap, "
        "deterministic, paginates with one call).\n\n"
    )
    p = _newest_html_path(state)
    if p is None:
        return base + "Fetch the entry page first; the system will then propose a tailored template."
    try:
        html = p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return base + _LISTING_TEMPLATES["generic_div"]
    pattern = _detect_listing_pattern(html)
    pretty = {
        "table": "HTML <table>",
        "drupal_views": "Drupal/Views <div class='views-row'>",
        "wp_article": "<article>",
        "li_card": "<li class='post|item|card|result'>",
        "generic_div": "generic",
    }[pattern]
    return base + f"Detected pattern in {p.name}: **{pretty}**\n\n" + _LISTING_TEMPLATES[pattern]


def _codebook_contract_ok(state: "AgentState") -> bool:
    try:
        from gemma_miner.contracts import CodebookContract

        contracts = [c for c in state.contracts.list() if isinstance(c, CodebookContract)]
        if not contracts:
            return True
        return all(c.check(state.dataset)[0] for c in contracts)
    except Exception:  # noqa: BLE001
        return True


def _bronze_text_coverage(state: "AgentState") -> tuple[float, int, int]:
    """Return (fraction_with_text, n_with_text, n_total) over the bronze rows.

    A row "has text" when ANY of:
      • `text_path`-style field points at a file with ≥500 chars, OR
      • an inline text field (pdf_text/text/body/content/abstract/description/
        full_description_text/short_summary) is a string ≥500 chars.

    We use 500 chars (vs. extract_items' 200) to gate the CODEBOOK phase
    conservatively: designing a 30-variable schema off near-empty rows is
    the upstream cause of the all-null/false-default output we saw.
    """
    rows = state.dataset.rows()
    if not rows:
        return 0.0, 0, 0
    text_path_fields = ("text_path", "txt_path", "txt_file", "text_file")
    inline_fields = (
        "pdf_text", "text", "body", "content", "abstract",
        "description", "full_description_text", "short_summary", "summary",
    )
    n_with_text = 0
    workdir = Path(state.workdir)
    for r in rows:
        has_text = False
        for f in inline_fields:
            v = r.get(f)
            if isinstance(v, str) and len(v) >= 500:
                has_text = True
                break
        if not has_text:
            for f in text_path_fields:
                p = r.get(f)
                if isinstance(p, str):
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = workdir / pp
                    try:
                        if pp.exists() and pp.stat().st_size >= 500:
                            has_text = True
                            break
                    except OSError:
                        pass
        if has_text:
            n_with_text += 1
    return n_with_text / len(rows), n_with_text, len(rows)


def _codebook_needs_extraction(state: "AgentState") -> bool:
    codebook_path = Path(state.workdir) / "codebook.json"
    if not codebook_path.exists() or len(state.dataset) == 0:
        return False
    try:
        from gemma_miner.codebook import Codebook

        cb = Codebook.load(codebook_path)
        var_names = [v.name for v in cb.variables]
        if not var_names:
            return False
        # Hash only EXTRACTION-relevant fields. Cosmetic changes (required
        # flags, adversary notes, etc.) should NOT trigger re-extraction —
        # they don't affect what the LLM sees per item.
        current_hash = cb.extraction_signature()
        if state.memory.get("last_extracted_codebook_hash") != current_hash:
            return True
        rows = state.dataset.rows()
        sample = rows[: min(10, len(rows))]
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
            "rows into the queue or dataset."
        ),
        tools=(
            "http_get",
            "set_plan",
            "show_plan",
            "extractor_define",
            "llm_scrape",
            "html_find",
            "html_inspect",
            "field_probe",
            "pagination_probe",
            "python",
        ),
        hint=(
            "STEP 1 — LOOK at the page. http_get returns up to 100 KB of preview;\n"
            "html_inspect shows top tags & classes. Find the REPEATING UNIT.\n\n"
            "STEP 2 — CLASSIFY THE ITEM. In your `thought`, state:\n"
            "  • what one row represents,\n"
            "  • which fields are visible on the listing,\n"
            "  • whether you'll need a detail page per item, attachments, or neither.\n\n"
            "STEP 2.5 — if fields are unclear, call field_probe(values=['sample value 1', ...])\n"
            "to find WHERE each field appears in the cached HTML before writing the extractor spec.\n\n"
            "STEP 3 — if pagination is unclear, call pagination_probe(base_url=...) to discover\n"
            "the correct URL pattern before committing.\n\n"
            "STEP 4 — WRITE A SPEC (or call llm_scrape for messy HTML, or python\n"
            "for JSON APIs). Phase done when ≥1 row is matched and visible fields\n"
            "are non-null."
        ),
    ),
    "ENUMERATE": Phase(
        name="ENUMERATE",
        goal=(
            "Populate the queue by running the listing extractor across paginated "
            "URLs OR by hitting an API directly."
        ),
        tools=(
            "scrape_paginated",
            "queue_status",
            "queue_add",
            "http_get",
            "pagination_probe",
            "python",
            "read_file",
            "show_plan",
            "set_plan",
        ),
        hint=(
            "If a 'listing' extractor is saved, call:\n"
            "  scrape_paginated(\n"
            "    url_template='<base>?page={page}',\n"
            "    extractor_name='listing',\n"
            "    start_page=<1 unless you've verified ?page=0 returns rows>,\n"
            "    max_pages=<ceil(plan.target_rows / plan.items_per_page) + a few>,\n"
            "    target_count=<plan.target_rows>\n"
            "  )\n\n"
            "SIZING — defaults will sandbag you. start_page defaults to 0 and\n"
            "max_pages defaults to 20 (≈ 840 rows at 42/page). If your plan needs\n"
            "more than that, you MUST pass max_pages explicitly. Compute it from\n"
            "the plan: pages_needed = ceil(target_rows / items_per_page).\n\n"
            "INDEXING — most paginated sites are 1-indexed (page=1 is the first\n"
            "real page; page=0 returns empty or a redirect). The base URL with\n"
            "NO ?page param is usually equivalent to page=1, so compare it to\n"
            "?page=1 and ?page=2 in cache to confirm. If your first call returns\n"
            "`total_added: 0` AND the base URL had real rows, your start_page is\n"
            "wrong — RE-CALL scrape_paginated with start_page=1, do NOT fall back\n"
            "to llm_scrape on the same cached page (it just re-reads page 1).\n\n"
            "For non-`{page}` pagination (cursor, offset+limit, next-token), write\n"
            "a Python loop calling http_get per page and dataset_append the rows in\n"
            "one batch at the end. DO NOT re-implement pagination when\n"
            "scrape_paginated already fits — that's how runs thrash and produce\n"
            "30 rows instead of 1000.\n\n"
            "Phase done when `dataset_rows >= plan.target_rows`."
        ),
    ),
    "DISCOVER_DETAIL": Phase(
        name="DISCOVER_DETAIL",
        goal=(
            "Inspect one queued item's detail page. Decide whether details are "
            "scalar HTML fields, one primary attachment, or multiple linked "
            "dependencies."
        ),
        tools=(
            "queue_next",
            "http_get",
            "extractor_define",
            "python",
        ),
        hint=(
            "On ONE representative item:\n"
            "  1. queue_next() → http_get its detail_url.\n"
            "  2. Inspect the cached HTML — find the fields you need.\n"
            "  3. If there's one primary attachment, define a 'detail' spec\n"
            "     with attachment_url. If varied assets, skip a detail spec\n"
            "     and call process_queue(mode='multi_asset') in PROCESS."
        ),
    ),
    "PROCESS": Phase(
        name="PROCESS",
        goal=(
            "Turn the queued items into dataset rows. Pick the smallest harvest "
            "path that delivers every variable the codebook will need."
        ),
        tools=(
            "dataset_from_queue",
            "process_queue",
            "dataset_stats",
            "queue_status",
            "dataset_sample",
            "python",
            "dataset_append",
            "read_file",
            "extract_text",
        ),
        hint=(
            "Pick the mode by what the user needs:\n\n"
            "  A. Listing has every required field →\n"
            "       dataset_from_queue()\n\n"
            "  B. Detail page adds HTML fields →\n"
            "       process_queue(detail_extractor='detail', mode='text',\n"
            "                     row_template={…})\n\n"
            "  C. Detail page has ONE attachment →\n"
            "       process_queue(detail_extractor='detail', mode='attachment',\n"
            "                     row_template={'text': '$file:{paths.text}'})\n\n"
            "  D. Detail pages have varied assets →\n"
            "       process_queue(mode='multi_asset', batch_size=5)\n\n"
            "If a call returns 0 appended with N errors twice in a row, downgrade\n"
            "(D → C → B → A) — don't keep retrying the same mode."
        ),
    ),
    "CODEBOOK": Phase(
        name="CODEBOOK",
        goal=(
            "Design a CODEBOOK of 20–60 typed variables (booleans, ints, floats, "
            "enums, dates) capturing the structured information present in each "
            "item."
        ),
        tools=(
            "codebook_propose",
            "codebook_show",
            "codebook_edit",
            "codebook_test",
            "dataset_sample",
        ),
        hint=(
            "Call `codebook_propose(sample_size=4)` to draft 20–60 variables.\n"
            "Inspect with `codebook_show`. Tweak with `codebook_edit` if needed.\n"
            "Then advance to EXTRACT — `codebook_test` is optional."
        ),
    ),
    "EXTRACT": Phase(
        name="EXTRACT",
        goal=(
            "Apply the locked codebook to EVERY item in the dataset, filling the "
            "structured columns via LLM extraction + deterministic type coercion."
        ),
        tools=(
            "extract_items",
            "codebook_edit",
            "dataset_validate",
            "dataset_stats",
        ),
        hint=(
            "  1. `extract_items()` — auto-runs a 3-row pilot on the first call.\n"
            "     Read PILOT verdict + per-variable coverage.\n"
            "  2. SCALE_OK → `extract_items(limit=null)` to extract remaining rows.\n"
            "  3. FIX_FIRST → `codebook_edit` the under-filled vars, re-pilot.\n"
            "  4. `dataset_validate` — final per-variable stats; drop <30 % coverage vars."
        ),
    ),
    "EXPORT": Phase(
        name="EXPORT",
        goal=(
            "Write the dataset to Parquet + JSONL + codebook.md, then optionally "
            "push to Hugging Face, then call `finish`."
        ),
        tools=(
            "dataset_export",
            "dataset_validate",
            "hf_push",
            "finish",
        ),
        hint=(
            "Minimal happy path:\n"
            "  1. `dataset_export()` — writes parquet + jsonl + codebook to\n"
            "     <workdir>/export/.\n"
            "  2. If the goal mentions Hugging Face, `hf_push(repo_id='...')`.\n"
            "  3. `finish(summary='produced dataset X with Y rows')`."
        ),
    ),
    "FINISH": Phase(
        name="FINISH",
        goal="All contracts satisfied — call finish.",
        tools=("dataset_stats", "finish"),
        hint="finish(summary='built a dataset of N items') — no other work needed.",
    ),
}


def current_phase(state: "AgentState", contract_min_rows: int | None = None) -> Phase:
    """Pick the phase based on observable state.

    Hysteresis rule (added to stop EXPORT↔ENUMERATE bouncing): once the
    extracted dataset (silver) covers ≥ soft_target rows, the phase machine
    will not fall back into harvest phases. The only paths out from there are
    EXPORT and FINISH. Mid-run "I'd like 16 more rows" is not worth re-mixing
    sources and re-extracting everything.
    """
    extractors = state.memory.get("extractors", {}) or {}
    has_listing = any(spec.get("row_pattern") for spec in extractors.values())
    has_detail = any(not spec.get("row_pattern") for spec in extractors.values())
    queue = state.memory.get("queue", []) or []
    processed = set(str(x) for x in (state.memory.get("processed", []) or []))
    remaining = sum(
        1 for q in queue
        if isinstance(q, dict) and str(q.get("id")) not in processed
    )

    if (
        state.contracts.list()
        and state.contracts.all_satisfied(state.dataset)
        and not _codebook_needs_extraction(state)
    ):
        return PHASES["FINISH"]

    target = contract_min_rows or 50
    n_rows = len(state.dataset)
    soft_target = max(1, int(target * 0.9))

    # HYSTERESIS: once silver has been populated to ≥ soft_target, lock to
    # EXPORT/FINISH. The flag below is also stickied by extract_items at the
    # end of a full sweep.
    silver_n = len(state.extracted_dataset()) if state._extracted_dataset else 0
    post_extract_done = bool(state.memory.get("_post_extract_done"))
    if post_extract_done or silver_n >= soft_target:
        # Codebook still needs (re-)extraction? Stay in EXTRACT until done.
        if _codebook_needs_extraction(state):
            return PHASES["EXTRACT"]
        codebook_path = Path(state.workdir) / "codebook.json"
        if codebook_path.exists():
            try:
                from gemma_miner.codebook import Codebook

                cb = Codebook.load(codebook_path)
                export_dir = Path(state.workdir) / "export"
                if (export_dir / f"{cb.name}.parquet").exists():
                    return PHASES["FINISH"]
            except Exception:  # noqa: BLE001
                pass
        return PHASES["EXPORT"]

    # Soft-met threshold (pre-extract): ≥80 % AND the agent is spinning.
    pre_soft = max(1, int(target * 0.8))
    if n_rows >= pre_soft and n_rows < target:
        recent = state.history[-5:]
        if recent:
            recent_errors = sum(1 for h in recent if h.error)
            if recent_errors >= 3:
                return PHASES["EXPORT"]

    # POST-EXTRACT EXIT: silver has been "completed" (either ≥90% of bronze
    # or zero rows-left-to-try) and the agent is still calling extract_items
    # over and over. Send it to EXPORT — the data won't get any better by
    # re-running the same extraction.
    if post_extract_done:
        recent = state.history[-8:]
        extract_replays = sum(
            1 for h in recent
            if h.tool == "extract_items"
        )
        if extract_replays >= 3:
            return PHASES["EXPORT"]

    if n_rows >= target:
        codebook_path = Path(state.workdir) / "codebook.json"
        if not codebook_path.exists():
            wants_codebook = bool(state.memory.get("wants_codebook"))
            sample = state.dataset.rows()[:5]
            has_long_text = any(
                isinstance(r.get(f), str) and len(r.get(f, "")) > 200
                for r in sample
                for f in ("pdf_text", "text", "body", "content", "abstract")
            )
            if has_long_text or wants_codebook:
                # BRONZE-TEXT GATE: refuse to enter CODEBOOK when most bronze
                # rows don't have substantial text. Codebook + extract on
                # near-empty rows is the cause of the all-null output we saw.
                cov, n_ok, n_tot = _bronze_text_coverage(state)
                if cov < 0.8 and not state.memory.get("_bronze_text_gate_bypassed"):
                    # Route the agent back to PROCESS to enrich the rows with
                    # detail/body text via process_queue(mode='text') or python.
                    return Phase(
                        name="PROCESS",
                        goal=(
                            "Most rows lack substantive text content — fetch the "
                            "detail/body text BEFORE designing a codebook."
                        ),
                        tools=PHASES["PROCESS"].tools,
                        hint=(
                            f"🛑 BRONZE-TEXT GATE: only {n_ok}/{n_tot} ({cov:.0%}) "
                            "rows have ≥500 chars of source text. Designing a "
                            "30-variable codebook off near-empty rows produces "
                            "all-null/false-default output (the failure we keep "
                            "seeing in production).\n\n"
                            "Do ONE of:\n"
                            "  1. process_queue(detail_extractor='detail', "
                            "mode='text') — adds the detail page body to each "
                            "row.\n"
                            "  2. python — fetch the detail/body for each item "
                            "via the source's API, set the result on the row's "
                            "`text`/`body`/`content`/`text_path` field, then "
                            "dataset_append (upsert by id).\n"
                            "  3. If the user genuinely wants a metadata-only "
                            "dataset (no body text), set "
                            "memory_set(key='_bronze_text_gate_bypassed', "
                            "value=true) and re-enter — the codebook will be "
                            "limited to what's already in the rows."
                        ),
                    )
                return PHASES["CODEBOOK"]
            return PHASES["EXPORT"]
        if not _codebook_contract_ok(state):
            return PHASES["CODEBOOK"]
        if _codebook_needs_extraction(state):
            return PHASES["EXTRACT"]
        return PHASES["EXPORT"]

    if not has_listing:
        base = PHASES["DISCOVER_LISTING"]
        return Phase(
            name=base.name, goal=base.goal, tools=base.tools,
            hint=_dynamic_listing_hint(state),
        )

    if remaining < max(1, target - n_rows):
        return PHASES["ENUMERATE"]

    # If the queue items already carry every required field, skip DETAIL.
    queue_satisfies_required = False
    if remaining > 0:
        try:
            from gemma_miner.contracts import FieldsContract, _field_variants

            required: list[str] = []
            for c in state.contracts.list():
                if isinstance(c, FieldsContract):
                    required.extend(c.required_fields)
            if required:
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

    has_dataset_rows = len(state.dataset) > 0
    if remaining > 0:
        return PHASES["PROCESS"]

    codebook_path = Path(state.workdir) / "codebook.json"
    if has_dataset_rows and not codebook_path.exists():
        return PHASES["CODEBOOK"]

    if has_dataset_rows and codebook_path.exists():
        if not _codebook_contract_ok(state):
            return PHASES["CODEBOOK"]
        if _codebook_needs_extraction(state):
            return PHASES["EXTRACT"]
        try:
            from gemma_miner.codebook import Codebook

            cb = Codebook.load(codebook_path)
            rows = state.dataset.rows()
            sample = rows[: min(5, len(rows))]
            var_names = [v.name for v in cb.variables]
            populated = sum(
                1 for r in sample
                if any(r.get(n) is not None for n in var_names)
            )
            if populated == 0:
                return PHASES["EXTRACT"]
            export_dir = Path(state.workdir) / "export"
            if (export_dir / f"{cb.name}.parquet").exists():
                return PHASES["FINISH"]
            return PHASES["EXPORT"]
        except Exception:  # noqa: BLE001
            return PHASES["EXTRACT"]

    return PHASES["PROCESS"]
