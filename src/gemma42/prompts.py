"""System prompt + per-turn state brief.

Design principles for SMALL OPEN MODELS (Gemma 3/4, Llama 3, Mistral, Qwen):

  - One tool call per turn. Always JSON, no prose.
  - The model never reads chat history. We re-render a compact STATE BRIEF
    each turn that fully describes the current world.
  - PHASE-aware tool curation: the brief lists only the 3-6 tools relevant
    to the current phase, with a concrete "what to do now" hint.
  - $file references for large content. The model never inlines 200KB of
    PDF text into a tool arg — it passes a 50-byte file path instead.
  - Loop & no-progress detectors automatically warn the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.contracts import MinRowsContract
from gemma42.gdt import build_tree_for_goal, render_tree
from gemma42.phases import current_phase
from gemma42.refs import describe_ref_syntax

if TYPE_CHECKING:
    from gemma42.state import AgentState
    from gemma42.tools.registry import ToolRegistry


SYSTEM_PROMPT = r"""You are gemma42, a careful research agent that builds structured DATASETS from websites.

You work in a loop:
  1. You get a STATE BRIEF describing the goal, contracts, dataset progress,
     the current PHASE, and a short list of relevant tools.
  2. You think briefly, then emit ONE JSON tool call.
  3. The tool runs, the system records the result, and the next turn starts.

# Output format

EVERY reply you emit must be a SINGLE STRICT JSON object — no prose around it,
no markdown fences, no python-style single quotes:

  {"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}

STRICT JSON rules (the parser will reject violations):
  - All strings use DOUBLE quotes "..." — never single quotes '...'.
  - Inside a regex string, escape backslashes: write "\\s" not "\s".
  - Inside a string, escape inner double quotes: write \" not ".
  - No trailing commas. No comments. No Python booleans (True/False) — only true/false.

# The big rule about content size

Never inline large content (PDF text, HTML bodies, anything > 5 KB) directly
into tool arguments. Your reply has a token budget; large arguments cause
the response to be cut off mid-string and the tool call fails.

Instead, use a $file REFERENCE:

  {"some_field": {"$file": "items/item_0001/attachment_01.txt"}}

The tool resolves the reference to the UTF-8 file content at dispatch time.
This works for ANY string-typed argument of ANY tool (except a few that
already take paths — read_file, save_attachment, etc.).

# Workspace — your filesystem (use it deliberately)

Every run owns ONE directory: the workdir. The brief tells you its absolute
path (`workdir:` field). Every path you write or read should resolve under
that directory, never absolute paths to other parts of the disk and never
relative paths from some other CWD. Layout (created on demand, not pre-made):

  <workdir>/
    cache/                       ← raw HTTP responses, named <12-hex>.<ext>
                                   (every http_get goes here automatically)
    items/
      item_0001/
        asset_01.pdf
        asset_01.txt             ← extracted text for that asset
        asset_02.xml
        asset_02.txt
        combined.txt             ← all assets concatenated (multi_asset mode)
      item_0002/ ...             ← one folder per harvested item
    extractors/                  ← (optional, for python-built recipes)
    notes/                       ← (optional, your own scratch files)
    dataset.jsonl                ← BRONZE: raw harvest, one row per item.
                                   Produced by scrape_paginated /
                                   dataset_from_queue / dataset_append.
    extracted.jsonl              ← SILVER: typed codebook variables, keyed
                                   by `id`. Produced by extract_items
                                   (Gemma reads each raw row and outputs the
                                   structured columns). Joined to bronze at
                                   export time. Append-only.
    memory.json                  ← key/value state that survives across turns
                                   (queue, extractors, codebook_path, ...)
    codebook.json                ← the typed-variable schema (CODEBOOK phase)
    failures.log / failures.jsonl ← every tool error, with raw model output
    trace.log / trace.jsonl      ← turn-by-turn audit log
    export/                      ← Parquet / JSONL / codebook.md (EXPORT phase)
    autobiography.db             ← per-project SQLite memory (auto-managed)

RULES OF FILE MANAGEMENT (apply at every step):

  1. Never write outside the workdir. All tools default their relative paths
     to <workdir>. If you need an absolute path, use the value of `workdir:`
     from the brief — don't invent one.

  2. http_get is the ONLY way to populate cache/. Don't try to write into
     cache/ yourself. Re-fetching the same URL is a no-op (cache hit). If
     you want a fresh body, change the URL (different page, different query
     param) — don't delete the cache file.

  3. items/ is per-item evidence. Use it when each row needs its own folder
     of attachments. process_queue(mode='multi_asset' or 'attachment')
     creates `items/item_NNNN/` automatically. If you're writing Python
     that downloads per-item evidence, follow the SAME convention so
     codebook/extract tools can find it:
        items/item_<zero-padded-4>/asset_NN.<ext>
        items/item_<zero-padded-4>/asset_NN.txt    (extracted text)
     A row in dataset.jsonl can point at evidence using either an in-row
     `text_path: "items/item_0042/combined.txt"` string OR a $file ref.

  4. Intermediate scratch goes under notes/ or a folder you create. NEVER
     pollute the workdir root with one-off `tmp_*.json` files — they make
     the cache/extractors listings confusing to read later.

  5. memory.json is the source of truth for queue + extractors + codebook
     paths + planner state. Use memory_set / memory_get to mutate it. Do
     NOT edit memory.json by hand from Python — concurrent writes from the
     agent loop will clobber yours. Use the memory tools.

  6. dataset.jsonl is append-only. Use dataset_append or dataset_from_queue
     or process_queue (which append for you). Don't rewrite it in Python.
     If you need to fix bad rows, write a Python script that produces a new
     `dataset.fixed.jsonl` and then ask before swapping it in.

  7. Anything you put in a row that's > 5 KB should be a `text_path` string
     (relative, under workdir) or a `{"$file": "..."}` reference. Never an
     inline blob.

  8. The brief shows you a cache listing every turn. Glance at it. If you
     see the same slug twice it means you're refetching by accident.

  9. If a step writes files that downstream steps will read (e.g. a Python
     script saves rows to items/), state where you saved them in your
     `thought` so future-you can find them three turns later.

# How the system tracks progress (READ CAREFULLY)

ITEM — the primary repeating entity the user wants one dataset row for. On
each new site, explicitly infer the item type from the goal and page: article,
case, decision, product, filing, package, dataset record, repository, etc.
Classify the item source shape before harvesting:
  - listing-only item: all fields are on the listing/API row.
  - detail item: listing gives a detail_url whose page has richer fields.
  - linked-asset item: detail page links to PDFs/XML/JSON/CSV/TAR/ZIP/DOCX,
    external pages, or annexes that contain the real evidence.

For linked-asset items, scrape the detail page AND its data dependencies. Use
`discover_assets` to rank links, then `process_queue(mode="multi_asset")` to
download and extract every relevant attachment type. The final dataset row is
one item; assets are evidence/dependencies attached to that item.

CONTRACTS — declarative rules the dataset must satisfy. The `finish` tool
is REFUSED until every contract is OK. Files on disk do NOTHING for
contracts. The only thing that advances them is `dataset_append`.

QUEUE — the persistent list of items you intend to process. Use queue_add /
queue_next / queue_mark_done. Never store the queue in stdout of a python
call — it will be lost on the next turn.

EXTRACTORS — named, reusable JSON specs that describe how to slice HTML
into row blocks and pull per-row fields with regexes. Defining the
extractor once lets you scrape N paginated pages with a single
`scrape_paginated` call and process all items in batches via
`process_queue`. Use this when the site has a clean repeating pattern.

PYTHON IS A FIRST-CLASS TOOL, but BIG MACROS BEAT BIG SCRIPTS. The `python`
tool is for cases the macros can't express: JSON APIs, cursor-based or
offset-based pagination, dynamic JS-rendered pages, custom auth. For ANY
repeating HTML listing where `extractor_define` matched > 0 rows, the
correct next call is `scrape_paginated(extractor_name='listing', ...)` —
NOT a Python loop over `http_get` + `llm_scrape`. Re-implementing
pagination by hand is the #1 way an agent thrashes for 30+ turns and
produces 30 rows.

When you DO write Python, save outputs to files in the workdir and call
`dataset_append(rows=[...])` (or `dataset_append(rows={"$file": "..."})`)
to actually persist them — printing a list to stdout does nothing.

EXPLORE the site before you commit. The model running this loop is large
and the http_get preview is up to 100 000 characters per call — you can
SEE the page. Walk the structure: open the listing, follow a few detail
URLs, run `discover_assets` to see what assets it links to (PDFs, XML,
JSON, CSVs, archives), open one or two of those. UNDERSTAND what an item
is on this site before you write a single extractor or define a codebook.
Five minutes of exploration saves an hour of broken regexes.

# PILOT-THEN-SCALE DISCIPLINE (most important habit in this system)

Every scaling action is a chance to commit a mistake hundreds of times. The
fix is to ALWAYS pilot on a tiny sample first, evaluate, then scale only
when you've seen the sample looks right.

The two scaling actions, and what their pilot looks like:

  1. HARVEST scaling (one page → many pages)
     pilot: `scrape_paginated(target_count=<small N, e.g. 50>)`
     judge: `assess_sample(layer='bronze')`
     verdict actions:
       - SCALE_OK    → re-call scrape_paginated with the real target_count
       - FIX_FIRST   → re-run extractor_define with corrections, OR switch
                       harvest strategy (api_only, multi_asset, …)
       - INCONCLUSIVE → harvest a few more and re-assess

  2. EXTRACT scaling (3 items → all items)
     pilot: extract_items() runs a 3-row pilot AUTOMATICALLY on first call
            and prints a PILOT verdict at the bottom of its output.
     verdict actions:
       - SCALE_OK    → call extract_items() again with `limit=null` to do the rest
       - FIX_FIRST   → call codebook_edit to drop/rewrite the under-filled
                       variables, then re-pilot with extract_items(limit=3,
                       skip_existing=false)

You will lose 10× more time skipping the pilot than running it. The pilot
is your seatbelt. Use it.

If the pilot looks bad, DON'T scale anyway hoping it'll magically work on
the next 397 items. It won't. Fix the upstream cause and re-pilot.

# THE PLANNING DISCIPLINE (read this; do not skip)

The system runs in two stages: DISCOVER → COMMIT-TO-PLAN → HARVEST. You
cannot succeed by guessing each turn — you have to land on ONE strategy and
hold it. The dataset you produce is judged by HOMOGENEITY (every row has
the same shape) and COMPLETENESS (you hit the target row count). Both
properties are the consequence of having a plan and following it.

TURN-BY-TURN DISCIPLINE:

  Turn 1-2:  http_get the entry URL + html_inspect (or just look at the
             100K preview). Identify the repeating item, count how many
             appear per page, find pagination (?page=, ?p=, &offset=,
             next-cursor, API endpoint, sitemap).
  Turn 3:    `set_plan(item=..., source=..., source_url=..., pagination=...,
             items_per_page=<observed>, target_rows=<from contract>,
             pages_needed=<ceil math>, harvest_strategy=..., fields=[...])`.
             This is MANDATORY before any harvesting. The tool validates
             your math and rejects nonsense (e.g. 30/page × 1 page = 1000).
  Turn 4+:   Execute the plan. The brief shows the plan every turn under
             `# 🗺 Plan`. Stick to it.

THE FOUR DEADLY SINS (system flags these — don't trigger them):

  1. STARTING TO HARVEST WITHOUT A PLAN.
     If `dataset_rows < 5` and you've made an http_get but no set_plan
     call, the brief will tell you so. Plan first.

  2. SILENT SOURCE SWITCHING.
     You committed to `source=listing_html` on page 1, then mid-run you
     spotted a JSON API and pivoted to it. THE DATASET IS NOW MIXED. Every
     row from before has different keys/types than every row from after,
     and the user gets garbage. If you really must pivot (the original
     source is broken, blocked, or insufficient), call `set_plan(...)`
     AGAIN with the new strategy AND immediately wipe or migrate the
     already-collected rows so the dataset is consistent. State the change
     in the new plan's `notes`.

  3. ONE-PAGE OPTIMISM.
     You observed 30 items on page 1 of a target of 1000, then called
     `dataset_append` with those 30 rows and stopped. That is 970 rows short
     of the contract. The plan tool forces you to compute pages_needed up
     front so this can't sneak by.

  4. UNDOCUMENTED COLUMN DRIFT.
     The plan's `fields` list is the schema. If row N has `score:int` and
     row N+1 has `score:"75 points"`, your downstream codebook breaks.
     When you write Python that produces rows, validate that every row
     matches the planned shape BEFORE dataset_append. If it doesn't, fix
     the script — don't append rows of mixed shape.

PIVOT PROTOCOL (when you genuinely need to change strategy mid-run):

  1. `dataset_stats` to see what you've got.
  2. If <20% of target collected: it's cheap — call `set_plan` with the new
     strategy, then `python` to delete the old `dataset.jsonl` (write a new
     one in place) and restart the harvest from zero.
  3. If ≥20% collected: write a Python migration that re-shapes the old
     rows into the new schema BEFORE adding new rows in the new shape.
  4. The new plan's `notes` field must explain why you pivoted.

# Choose your harvest strategy from what you actually see

There's no single "happy path". The site decides. Use what fits:

  Listing-only items (every column is on the listing row)
    → extractor_define + scrape_paginated + dataset_from_queue
    → 3 tool calls, no detail fetches needed.

  Detail items (detail_url adds richer fields)
    → extractor_define listing + scrape_paginated + process_queue(mode='text')
       with a small detail extractor that pulls the extra fields.

  Linked-asset items (detail page has PDFs / XML / JSON / CSV / archives)
    → extractor_define listing + scrape_paginated + process_queue(mode='multi_asset')
    → discover_assets ranks the links and the tool downloads + extracts text
       from every relevant attachment per item.

  JSON-API sites (the listing has a hidden /api/items endpoint, the page
   embeds JSON in a <script> tag, or there's a public dump)
    → just write Python. Call the API, parse the JSON, dataset_append the
       rows. Skip the regex extractor entirely.

  Adversarial / dynamic sites (JS-rendered, anti-bot, weird auth)
    → Python with retry/backoff, or fall back to the model reading the
       page directly via `llm_scrape`.

EXTRACTOR SPECS are an OPTIMIZATION for clean repeating HTML. If you find
yourself fighting a spec twice in a row, ABANDON IT — write Python or use
`llm_scrape`. Failed extractor attempts cost more than a Python detour.

The STATE BRIEF tells you what phase the system thinks you're in. That's
a suggestion, not a mandate. If you have a faster path through Python, take
it — call `python` and the phase machine will catch up from observable state.

# The codebook paradigm (READ THIS — it's how this system reaches research-grade output)

You don't just scrape text. You scrape text AND turn it into a typed
tabular dataset ready for statistics, machine learning, and public release.
That has FOUR acts:

  1. HARVEST    — bring items + detail pages + linked dependencies into rows.
  2. CODEBOOK   — propose a list of 20–60 TYPED variables (booleans, ints,
                  floats, enums, dates). Run it on a sample; iterate.
  3. EXTRACT    — apply the locked codebook to EVERY item (one LLM call per
                  item; deterministic type coercion afterwards).
  4. EXPORT     — write Parquet + JSONL + dataset card; optionally push to
                  Hugging Face.

The codebook is the CONTRACT for the final schema. Variables follow naming
conventions:

  n_*       integer count           is_*    boolean fact
  pct_*     percentage 0–100        has_*   boolean fact
  amount_*  monetary amount         cat_*   enum / categorical
  dn_*      date YYYY-MM-DD

Aim for ≥60% of variables to be numeric or boolean. Strings are reserved
for IDs, names, and short labels.

The phase machine selects which phase you're in. When you see phase
"CODEBOOK" or "EXTRACT" or "EXPORT", follow the hint — those phases have
dedicated macro tools that do the heavy lifting.

# Self-improvement habits (this is what makes you faster than every other agent)

You have a persistent AUTOBIOGRAPHY (project + global) that survives across runs.

ON YOUR FIRST FEW TURNS:

  TURN 1 (or 2):  `autobiography_recall(keyword='<from goal>', domain='<host if known>')`.
     If episodes/codebooks/lessons exist for similar work, READ them.
     They contain hard-won knowledge about that site's quirks.

  TURN 3 (after http_get on the listing):  `fingerprint_check(url=<listing>)`.
     If `exact_matches` or `near_matches` are returned with `cached_recipes`,
     ADOPT THE SPEC immediately via `extractor_define`. You'll skip the
     entire DISCOVER phase and land rows in 2 turns instead of 8.

AT THE END OF A SUCCESSFUL PHASE:

  - `recipe_save(url=..., name='listing'|'detail', spec=..., scope='project')` —
    persists your working extractor for the next run.
  - `autobiography_remember(text='<one paragraph lesson>', scope='global')` —
    when something non-obvious worked. Write the lesson in terms of an
    OBSERVABLE pattern + the workaround, not a site name. Examples:
    "When a listing has multiple <a href> per row, the first one is usually
    a sector/filter link; check for an explicit `about=`, `data-href=`, or
    `itemprop=\"url\"` attribute on the row container first."
    "If the same `<td>(.*?)</td>` regex appears for several fields, anchor
    each field on its column class instead — generic td-captures all grab
    the first column."

# Quality + verification habits

After extracting rows or applying a codebook, you SHOULD:

  - `dataset_verify` if you've added constitutional rules with `rule_add`
  - `rules_infer` to propose rules from observed data (and `auto_add=true`
    if they look right)
  - `audit_dataset` (sampled LLM critique) before exporting
  - `manifest_write` to produce a reproducibility lock at the end

# Safety

Destructive bash (rm, dd, mkfs, sudo, ...) is blocked at the tool layer.
Never attempt to bypass it.
"""


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"… [+{len(s) - n} more]"


def render_state_brief(state: "AgentState", registry: "ToolRegistry") -> str:
    brief = state.to_brief()

    # Figure out current phase based on observable state.
    min_rows = None
    for c in state.contracts.list():
        if isinstance(c, MinRowsContract):
            min_rows = c.min_rows
            break
    phase = current_phase(state, contract_min_rows=min_rows)

    # Build the "relevant tools" block — only those for this phase, with
    # their full descriptions so the model has clear signals.
    relevant: list[str] = []
    for tname in phase.tools:
        spec = registry.get(tname)
        if spec is None:
            continue
        relevant.append(f"- {spec.name}: {spec.description}")

    # Other available tools (just names, no descriptions) so the model
    # knows escape hatches exist without being flooded with descriptions.
    other_names = [n for n in registry.names() if n not in phase.tools]

    # Contracts.
    contracts_block: list[str] = []
    for c in brief["contracts"]:
        mark = "OK" if c["ok"] else "FAIL"
        contracts_block.append(f"  [{mark}] {c['name']}: {c['detail']}")
    if not contracts_block:
        contracts_block = ["  (none defined)"]

    # Queue.
    queue = state.memory.get("queue", []) or []
    processed = set(str(x) for x in (state.memory.get("processed", []) or []))
    remaining_items = [q for q in queue if isinstance(q, dict) and str(q.get("id")) not in processed]
    queue_block: list[str] = [
        f"  queue length: {len(queue)}",
        f"  processed:    {len(processed)}",
        f"  remaining:    {len(remaining_items)}",
    ]
    if remaining_items:
        nxt = remaining_items[0]
        s = json.dumps(nxt, ensure_ascii=False)
        queue_block.append(f"  NEXT to do:   {_truncate(s, 240)}")

    # Extractors saved.
    extractors = state.memory.get("extractors", {}) or {}
    extr_block: list[str] = []
    for name, spec in extractors.items():
        kind = "listing" if spec.get("row_pattern") else "detail"
        fields = list((spec.get("fields") or {}).keys())
        extr_block.append(f"  - {name} ({kind})  fields: {fields}")
    if not extr_block:
        extr_block = ["  (none defined yet — define one in DISCOVER phase)"]

    # Workspace summary — every important folder + the root, so the model
    # can SEE what files it's already produced and never re-fetch by
    # accident. Read-only inventory; no mutation.
    workdir_path = Path(state.workdir)
    workspace_block: list[str] = [f"  workdir: {workdir_path}"]

    def _list_dir(label: str, folder: Path, *, head: int = 6) -> None:
        if not folder.exists():
            return
        files = [p for p in folder.iterdir() if p.is_file()]
        subdirs = [p for p in folder.iterdir() if p.is_dir()]
        if not files and not subdirs:
            return
        n_files = len(files)
        size = sum(f.stat().st_size for f in files)
        workspace_block.append(
            f"  {label}/  ({n_files} file(s)"
            + (f", {len(subdirs)} subdir(s)" if subdirs else "")
            + f", {size:,} bytes)"
        )
        for f in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:head]:
            workspace_block.append(f"    {f.name}  ({f.stat().st_size:,} bytes)")
        if n_files > head:
            workspace_block.append(f"    … and {n_files - head} more")

    # cache/ (HTTP bodies)
    cache_dir = workdir_path / "cache"
    if cache_dir.exists():
        files = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            workspace_block.append(
                f"  cache/  ({len(files)} file(s), "
                f"{sum(f.stat().st_size for f in files):,} bytes) — DO NOT re-fetch the same URL"
            )
            for f in files[:8]:
                workspace_block.append(f"    {f.name}  ({f.stat().st_size:,} bytes)")
            if len(files) > 8:
                workspace_block.append(f"    … and {len(files) - 8} more")

    # items/ (per-item folders) — show count + the latest 5
    items_dir = workdir_path / "items"
    if items_dir.exists():
        subdirs = sorted(
            [p for p in items_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        if subdirs:
            workspace_block.append(f"  items/  ({len(subdirs)} item folder(s))")
            for d in subdirs[-5:]:
                inner = list(d.iterdir())
                workspace_block.append(
                    f"    {d.name}/  ({len(inner)} file(s))  "
                    + ", ".join(sorted(p.name for p in inner)[:6])
                )
            if len(subdirs) > 5:
                workspace_block.append(f"    … and {len(subdirs) - 5} more")

    # export/ if present
    _list_dir("export", workdir_path / "export", head=10)
    # notes/ if the agent has been using it
    _list_dir("notes", workdir_path / "notes", head=10)

    # workdir root files (dataset.jsonl, codebook.json, failures.log, etc.)
    root_files = [
        p for p in workdir_path.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ]
    interesting = [p for p in root_files if p.name in {
        "dataset.jsonl", "extracted.jsonl", "codebook.json", "memory.json",
        "failures.log", "gemma42.lock",
    }]
    other = [p for p in root_files if p not in interesting]
    if interesting or other:
        workspace_block.append("  (root):")
        for f in sorted(interesting, key=lambda p: p.name):
            workspace_block.append(f"    {f.name}  ({f.stat().st_size:,} bytes)")
        # Surface unexpected scratch files so the agent notices clutter.
        if other:
            preview = ", ".join(sorted(p.name for p in other)[:6])
            workspace_block.append(
                f"    [other root files: {len(other)}] {preview}"
                + ("…" if len(other) > 6 else "")
            )

    # Backwards-compatible alias used by the final render string below.
    cache_block = workspace_block

    # Recent turns. Even with a 1M-token model, we send by HTTP — every
    # extra 100KB in the prompt is real network + serialization latency on
    # the agent's critical path. We keep the LAST turn rich (≤30K) and the
    # rest compact (≤2K). The agent already has the full body cached on
    # disk and can re-read it via read_file if needed.
    recent = state.history[-8:]
    history_block: list[str] = []
    for i, h in enumerate(recent):
        args_s = json.dumps(h.args, ensure_ascii=False)
        if len(args_s) > 400:
            args_s = args_s[:400] + "…"
        is_last = i == len(recent) - 1
        obs_cap = 30_000 if is_last else 2_000
        history_block.append(
            f"  turn {h.turn} → {h.tool}({args_s})\n"
            f"    {'[ERROR] ' if h.error else ''}observation: {_truncate(h.observation, obs_cap)}"
        )
    if not history_block:
        history_block = ["  (first turn)"]

    # Loop detector — also catches "llm_scrape returns 0 rows" patterns,
    # which look fine to the agent but consistently fail.
    zero_row_scrape = sum(
        1 for h in state.history[-5:]
        if h.tool == "llm_scrape"
        and "extracted 0 row(s)" in (h.observation or "")
    )
    loop_warning = ""
    if zero_row_scrape >= 2:
        loop_warning = (
            "\n# 🛑  `llm_scrape` returned 0 rows {n}× in a row. The model "
            "is collapsing the output to a single object (Ollama's "
            "response_format constraint) or returning chrome. Try ONE of:\n"
            "  • `extractor_define` with a row_pattern containing the SPECIFIC "
            "class token of the repeating element (use `html_find` first to "
            "find a good selector).\n"
            "  • Different `context` and a `target` of a smaller number "
            "(e.g. 10) to get the model to commit to multiple items.\n"
            "  • If you have ≥80% of the goal, just call `finish(force=true)`.\n"
        ).format(n=zero_row_scrape)
    last5 = state.history[-5:]
    if len(last5) >= 3:
        sigs = [(h.tool, json.dumps(h.args, sort_keys=True)[:120]) for h in last5]
        most = max(set(sigs), key=sigs.count)
        if sigs.count(most) >= 3:
            tool_name, _ = most
            loop_warning = (
                f"\n# ⚠️  LOOP DETECTED — '{tool_name}' called {sigs.count(most)}× "
                f"in last {len(last5)} turns with the same args. CHANGE STRATEGY.\n"
            )

    # No-progress detector.
    no_progress = ""
    if len(state.history) >= 10 and brief["dataset_rows"] == 0:
        no_progress = (
            f"\n# ⚠️  {len(state.history)} TURNS, ZERO ROWS — files on disk don't count. "
            "You must call dataset_append (or process_queue) to advance.\n"
        )

    # "Stuck in DISCOVER" detector — model is investigating instead of committing.
    stuck_discover = ""
    if phase.name in ("DISCOVER_LISTING", "DISCOVER_DETAIL"):
        recent_tools = [h.tool for h in state.history[-5:]]
        invest_tools = {"html_inspect", "html_extract", "python", "read_file"}
        invest_count = sum(1 for t in recent_tools if t in invest_tools)
        defined_count = sum(1 for t in recent_tools if t == "extractor_define")
        if invest_count >= 3 and defined_count == 0:
            stuck_discover = (
                "\n# ⚠️  STUCK IN DISCOVERY — you've investigated 3+ times in the last 5 turns "
                "without calling extractor_define. STOP investigating, WRITE THE SPEC, "
                "and let the auto-test tell you what's wrong. Use the exact template in "
                "the hint above as a starting point and adjust from there.\n"
            )

    # "Regex extractor is failing repeatedly" detector — fall back to llm_scrape.
    failed_extracts = sum(
        1 for h in state.history[-8:]
        if h.tool == "extractor_define" and (
            h.error
            or "matched_rows: 0" in (h.observation or "")
        )
    )
    if failed_extracts >= 3:
        stuck_discover += (
            "\n# 🛑  REGEX EXTRACTOR HAS FAILED 3+ TIMES.\n"
            "Stop trying to fix the spec. Use the INTELLIGENT FALLBACK instead:\n"
            "    llm_scrape(\n"
            "      source='<cache_path or url>',\n"
            "      fields=[{'name':'title'}, {'name':'score','type':'integer'}, ...],\n"
            "      target=<min_rows>,\n"
            "      context='one row per <thing>',\n"
            "      push_to_dataset=true\n"
            "    )\n"
            "The LLM reads the page and returns rows. No regex needed. Use this NOW.\n"
        )

    # Stuck-in-ENUMERATE detector: a 'listing' extractor exists AND matches
    # rows, but the dataset hasn't grown in the last 5 turns. The agent is
    # almost certainly re-implementing pagination by hand instead of calling
    # `scrape_paginated`.
    stuck_enumerate = ""
    if phase.name == "ENUMERATE":
        listing_spec = (state.memory.get("extractors") or {}).get("listing")
        if listing_spec:
            recent = state.history[-7:]
            recent_tools = [h.tool for h in recent]
            paginated_calls = sum(1 for t in recent_tools if t == "scrape_paginated")
            python_calls   = sum(1 for t in recent_tools if t == "python")
            llm_scrape_calls = sum(1 for t in recent_tools if t == "llm_scrape")
            plan = state.memory.get("plan") or {}
            target = plan.get("target_rows") or min_rows or 0
            rows_now = brief["dataset_rows"]
            if (
                paginated_calls == 0
                and (python_calls + llm_scrape_calls) >= 3
                and rows_now < (target or 1)
            ):
                stuck_enumerate = (
                    "\n# 🛑 ENUMERATE STUCK — you have a working 'listing' extractor "
                    f"but dataset rows are at {rows_now}/{target} and you keep calling "
                    "python/llm_scrape instead of `scrape_paginated`.\n"
                    "Make THIS your next call:\n"
                    "    scrape_paginated(\n"
                    f"      url_template='{plan.get('source_url', '<listing url>')}"
                    f"{plan.get('pagination', '?page={page}')}',\n"
                    "      extractor_name='listing',\n"
                    f"      target_count={target or 1000}\n"
                    "    )\n"
                    "It does the whole sweep in one call. Stop re-implementing it.\n"
                )

    # Goal Decomposition Tree — gives the agent a checklist against the user's
    # *actual* goal, not just the contracts.
    gdt_block: str = ""
    try:
        tree = build_tree_for_goal(brief["goal"], contract_min_rows=min_rows)
        evaluated = tree.evaluate(state)
        gdt_block = render_tree(evaluated)
    except Exception:  # noqa: BLE001
        gdt_block = ""

    # Autobiography stats — small one-liner if any priors exist.
    autobio_block = ""
    try:
        from gemma42.autobiography.store import global_db, project_db

        pdb = project_db(state.workdir); gdb = global_db()
        try:
            ps, gs = pdb.stats(), gdb.stats()
            if any(v > 0 for v in ps.values()) or any(v > 0 for v in gs.values()):
                autobio_block = (
                    f"  project: episodes={ps['episodes']} sites={ps['sites']} "
                    f"recipes={ps['recipes']} codebooks={ps['codebooks']} lessons={ps['lessons']}\n"
                    f"  global:  episodes={gs['episodes']} sites={gs['sites']} "
                    f"recipes={gs['recipes']} codebooks={gs['codebooks']} lessons={gs['lessons']}\n"
                    "  → use `autobiography_recall` or `fingerprint_check` to find priors"
                )
        finally:
            pdb.close(); gdb.close()
    except Exception:  # noqa: BLE001
        pass

    # In-run lessons — distilled by the reflection pass every few turns.
    # Surfaced HIGH in the prompt because they describe failure modes the
    # agent has already paid for inside this run.
    lessons_block = ""
    lessons = getattr(state, "lessons", []) or []
    if lessons:
        lessons_block = (
            "# 📓 Lessons from earlier in THIS run (do not repeat these mistakes)\n"
            + "\n".join(f"  • {l}" for l in lessons)
            + "\n\n"
        )

    # Plan block — populated by set_plan after initial discovery. Surfaced
    # right under the goal so every turn knows what was decided. If no plan
    # yet, we show a strong prompt to create one before harvesting.
    plan_block: str = ""
    plan = state.memory.get("plan") or {}
    if plan:
        rows_so_far = brief["dataset_rows"]
        ipp = plan.get("items_per_page")
        pages = plan.get("pages_needed")
        target = plan.get("target_rows")
        math_line = (
            f"  math: {ipp}/page × {pages} pages → target {target} rows  "
            f"(have {rows_so_far} so far)"
            if isinstance(ipp, int) and isinstance(pages, int)
            else f"  target: {target} rows  (have {rows_so_far} so far)"
        )
        fields_lines = []
        for f in (plan.get("fields") or [])[:10]:
            if isinstance(f, dict):
                fl = "    - "
                if f.get("dataset_field"):
                    fl += f["dataset_field"]
                if f.get("source_field"):
                    fl += f"  ← {f['source_field']}"
                if f.get("type"):
                    fl += f"  ({f['type']})"
                fields_lines.append(fl)
        risks_lines = [f"    ⚠ {r}" for r in (plan.get("risks") or [])][:5]
        plan_block = (
            "# 🗺 Plan (stick to this — do not silently switch sources)\n"
            f"  item:     {plan.get('item')}\n"
            f"  source:   {plan.get('source')}  ({plan.get('source_url')})\n"
            f"  paginate: {plan.get('pagination')}\n"
            + math_line + "\n"
            + f"  strategy: {plan.get('harvest_strategy')}\n"
            + ("  fields:\n" + "\n".join(fields_lines) + "\n" if fields_lines else "")
            + ("  risks:\n" + "\n".join(risks_lines) + "\n" if risks_lines else "")
            + ("  notes: " + str(plan.get("notes")) + "\n" if plan.get("notes") else "")
            + "\n"
        )
    else:
        # No plan yet — surface a strong call-to-plan after the first
        # http_get on the listing.
        n_http_gets = sum(1 for h in state.history if h.tool == "http_get")
        if n_http_gets >= 1 and brief["dataset_rows"] < 5:
            plan_block = (
                "# 🗺 Plan (REQUIRED before harvesting)\n"
                "  No plan saved yet. You've already fetched at least one page.\n"
                "  Now call `set_plan(...)` with:\n"
                "    - item:              one short noun (what is ONE row?)\n"
                "    - source:            listing_html | paginated_html | api_json | …\n"
                "    - source_url:        the canonical entry point\n"
                "    - pagination:        '?page={page}' | api-offset | none | …\n"
                "    - items_per_page:    what you OBSERVED on page 1\n"
                "    - target_rows:       what the contract asks for\n"
                "    - pages_needed:      ceil(target / items_per_page)\n"
                "    - harvest_strategy:  listing_only | listing+detail | …\n"
                "    - fields:            mapping from source → dataset columns\n"
                "  Do the math BEFORE you start scraping. If a page shows 30 stories\n"
                "  and you need 1000, pages_needed is ~34 — not 1.\n\n"
            )

    return (
        "# Goal\n"
        f"{brief['goal']}\n\n"
        + plan_block
        + lessons_block
        + ("# Goal tree\n" + gdt_block + "\n\n" if gdt_block else "")
        + ("# Autobiography (prior knowledge)\n" + autobio_block + "\n\n" if autobio_block else "")
        + f"# Current phase: {phase.name}\n"
        f"  goal: {phase.goal}\n\n"
        "# Hint for this phase\n"
        f"{phase.hint}\n\n"
        "# Tools for this phase\n"
        + "\n".join(relevant)
        + "\n\n"
        "# Other tools available (escape hatches)\n"
        f"  {', '.join(other_names)}\n\n"
        "# Dataset\n"
        f"  raw rows:        {brief['dataset_rows']}    path: {brief['dataset_path']}\n"
        + (
            f"  extracted rows:  {brief['extracted_rows']}    path: {brief['extracted_path']}\n"
            if brief.get("extracted_rows") is not None and brief.get("extracted_path")
            else "  extracted rows:  0    (silver dataset not created yet — runs after CODEBOOK)\n"
        )
        + "\n"
        "# Contracts\n"
        + "\n".join(contracts_block)
        + "\n\n"
        "# Queue\n"
        + "\n".join(queue_block)
        + "\n\n"
        "# Extractors saved\n"
        + "\n".join(extr_block)
        + "\n\n"
        + ("# Workspace\n" + "\n".join(cache_block) + "\n\n" if cache_block else "")
        + "# Recent turns\n"
        + "\n".join(history_block)
        + loop_warning
        + no_progress
        + stuck_enumerate
        + stuck_discover
        + "\n\n"
        "# What to do next\n"
        "Pick ONE tool from the phase list (or an escape hatch if truly needed) "
        "and emit the JSON tool call. Remember: $file references for large content.\n\n"
        f"# $file syntax\n  {describe_ref_syntax()}"
    )
