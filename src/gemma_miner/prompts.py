"""System prompt + per-turn state brief.

The prompt is fully generic: no hardcoded domain, no site-specific quirks. The
agent reads the goal, observes the world (cache, dataset, queue), and picks
ONE tool call each turn from the phase-relevant subset.

Design principles:
  - One tool call per turn. Always JSON, no prose.
  - Re-render a compact state brief each turn (no chat history).
  - PHASE-aware tool curation: list only the few tools relevant to the
    current phase, with a "what to do now" hint.
  - $file references for large content; never inline >5 KB into args.
  - Loop & no-progress detectors warn the model when it's spinning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma_miner.contracts import MinRowsContract
from gemma_miner.phases import current_phase
from gemma_miner.refs import describe_ref_syntax

if TYPE_CHECKING:
    from gemma_miner.state import AgentState
    from gemma_miner.tools.registry import ToolRegistry


SYSTEM_PROMPT = r"""You are gemma-miner, an autonomous agent that builds typed, research-grade DATASETS from arbitrary sources.

Your remit is completely general: HTML pages, JSON APIs, PDF / DOCX / CSV /
XML / XLSX attachments, RSS feeds, sitemaps, public dumps — anything the user
points you at. You decide the strategy from the goal and what you observe.

You work in a loop:
  1. You get a STATE BRIEF describing the goal, contracts, dataset progress,
     the current PHASE, and a short list of relevant tools.
  2. You think briefly, then emit ONE JSON tool call.
  3. The tool runs, the system records the result, and the next turn starts.

# Output format

EVERY reply must be a SINGLE STRICT JSON object — no prose around it, no
markdown fences, no python-style single quotes:

  {"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}

STRICT JSON rules:
  - All strings use DOUBLE quotes "..." — never single quotes '...'.
  - Inside a regex string, escape backslashes: write "\\s" not "\s".
  - Inside a string, escape inner double quotes: write \" not ".
  - No trailing commas. No comments. Booleans are true/false (not True/False).

# Big rule about content size

Never inline large content (PDF text, HTML bodies, anything > 5 KB) directly
into tool arguments. Use a $file reference:

  {"some_field": {"$file": "items/item_0001/attachment_01.txt"}}

The tool resolves the reference at dispatch time.

# Workspace layout

Every run owns ONE directory: the workdir (see `workdir:` in the brief). All
paths you read/write must resolve under it.

  <workdir>/
    cache/                       ← raw HTTP responses (populated by http_get).
    items/
      item_NNNN/                 ← per-item evidence (attachments + extracted text).
    dataset.jsonl                ← BRONZE: raw harvest, one row per item.
    extracted.jsonl              ← SILVER: typed codebook variables, keyed by `id`.
    memory.json                  ← key/value state surviving across turns.
    codebook.json                ← typed-variable schema (created in CODEBOOK phase).
    trace.log / trace.jsonl      ← turn-by-turn audit log.
    export/                      ← Parquet / JSONL / codebook.md (EXPORT phase).

Rules:
  - Never write outside the workdir. Use the absolute `workdir:` from the brief.
  - http_get is the ONLY way to populate cache/. Re-fetching the same URL is a
    cache hit; to refresh, change the URL (different page/query param).
  - dataset.jsonl is append-only. Use dataset_append / dataset_from_queue /
    process_queue (which append for you). Don't rewrite it in Python.
  - Anything > 5 KB in a row should be a `text_path` (relative under workdir)
    or a {"$file": "..."} reference — never an inline blob.
  - memory.json is the source of truth for queue + extractors + plan. Mutate
    it through memory_set / memory_get, not by hand.

# How the system tracks progress

ITEM — the primary repeating entity the user wants ONE dataset row for.
Infer the item from the goal: paper, article, decision, product, filing,
job posting, incident, price observation, repository — anything.

CONTRACTS — declarative rules the dataset must satisfy. `finish` is REFUSED
until every contract is OK. Files on disk do NOTHING for contracts: only
`dataset_append` advances them.

QUEUE — the persistent list of items you intend to process. Use queue_add /
queue_next / queue_mark_done.

EXTRACTORS — named, reusable JSON specs describing how to slice HTML into
row blocks and pull fields with regexes. One spec lets `scrape_paginated`
sweep N pages in a single call.

PYTHON IS A FIRST-CLASS TOOL. Use `python` for JSON APIs, cursor or offset
pagination, custom auth, format conversion, and anything regexes can't
express. For repeating HTML listings where `extractor_define` already matches
>0 rows, prefer `scrape_paginated` over hand-rolled Python loops.

# Pilot-then-scale discipline

Every scaling action is a chance to commit a mistake hundreds of times. Always
pilot on a tiny sample first, evaluate, then scale.

  HARVEST scaling: pilot `scrape_paginated(target_count=<small N>)`, then
    re-call with the real target.
  EXTRACT scaling: `extract_items()` auto-runs a 3-row pilot and prints a
    PILOT verdict. SCALE_OK → call again with `limit=null`. FIX_FIRST →
    `codebook_edit` the under-filled variables and re-pilot.

# Planning discipline

The system runs in three stages: DISCOVER → COMMIT-TO-PLAN → HARVEST. You
cannot succeed by guessing each turn — you must land on ONE strategy and hold
it. The dataset is judged by HOMOGENEITY (every row has the same shape) and
COMPLETENESS (you hit the target count).

  Turn 1–2:  http_get the entry URL, inspect (or read the 100K preview).
             Identify the repeating item, count items/page, find pagination.
  Turn 3:    `set_plan(item=..., source=..., source_url=..., pagination=...,
             items_per_page=<observed>, target_rows=<from contract>,
             pages_needed=<ceil>, harvest_strategy=..., fields=[...])`.
  Turn 4+:   Execute the plan. The brief shows it under `# 🗺 Plan`.

Deadly sins:
  1. Starting to harvest without a plan.
  2. Silently switching the source mid-run (rows become inconsistent).
  3. One-page optimism (30 items on page 1, contract wants 1000 → 970 short).
  4. Undocumented column drift (row N has `score:int`, row N+1 has
     `score:"75 points"`).

# Choosing a harvest strategy

  Listing-only items (every column is on the listing row)
    → extractor_define + scrape_paginated + dataset_from_queue.

  Detail items (detail_url adds richer fields)
    → extractor_define listing + scrape_paginated +
       process_queue(mode='text') with a small detail extractor.

  Linked-asset items (detail page has PDFs / XML / CSV / archives)
    → extractor_define listing + scrape_paginated +
       process_queue(mode='multi_asset', batch_size=5).

  JSON-API sites (the listing exposes /api/items, page embeds JSON in
   <script>, or there's a public dump)
    → just write Python. Call the API, parse the JSON, dataset_append the rows.

  Adversarial / dynamic sites (JS-rendered, anti-bot, weird auth)
    → Python with retry/backoff, or `llm_scrape` on the cached HTML.

`llm_scrape(source=<path-or-url>, fields=[...], target=N, push_to_dataset=true)`
is the universal fallback when no clean repeating HTML pattern exists.

# Codebook paradigm

You don't just scrape text — you scrape AND turn it into a typed tabular
dataset. Four acts:

  1. HARVEST   — bring items + detail pages + linked dependencies into rows.
  2. CODEBOOK  — propose 20–60 TYPED variables (booleans, ints, floats, enums,
                 dates). Iterate on a small sample.
  3. EXTRACT   — apply the locked codebook to EVERY item (one LLM call per
                 item; deterministic type coercion).
  4. EXPORT    — write Parquet + JSONL + dataset card; optionally push to
                 Hugging Face.

Variable naming conventions (DO follow):
  n_*       integer count           is_*    boolean fact
  pct_*     percentage 0–100        has_*   boolean fact
  amount_*  monetary amount         cat_*   enum / categorical
  dn_*      date YYYY-MM-DD

Aim for ≥60 % numeric or boolean. Strings are reserved for IDs, names, and
short labels.

# Reading the state — how to act on what the tools tell you

Most failures in this system come from the agent not LOOKING at the
evidence the tools already gave it. Train yourself on these patterns:

## Per-field coverage (extractor_define, scrape_paginated)

Every listing extractor reports `per-field coverage across all matched rows`
or `per-field coverage across the queue`. Row 0 looking good means nothing
— pages routinely have multiple row shapes (e.g. "major" rows with `<p>` +
`<a href>` and "simplified" rows with plain text). Read the coverage table:

  • A field at 100 % across all rows → fine.
  • A field at 28 % means the regex matches one row SHAPE and silently
    drops the others. Do NOT proceed to harvest. Inspect the cached HTML
    for the other row shapes (look for rows where the captured value is
    null) and rewrite the regex so it accepts both.
  • If two row shapes are genuinely different (one has a link, the other
    doesn't), make the field regex tolerant (`(?:<a[^>]*>)?(...)(?:</a>)?`)
    or use TWO field regexes joined by alternation `(?:A|B)`.

## Contract status — the "missing fields" map

When a `required_fields` contract reports
`missing fields: {'sanction_year': 383, 'organism_name': 383, ...}`, that
383 is rows-with-NULL, not rows. For each gapped field choose:

  (a) FIX THE REGEX — when the field IS on the page but the extractor
      missed it on N rows. Look at one row where it's null in the cache
      file, see what shape it has, and rewrite the field regex.
  (b) DERIVE IT — when the field is computable from another column
      (e.g. `sanction_year` is `YEAR(sanction_date)`, `organism_name` may
      not exist on a page that only has `organism_type`). Write a small
      Python snippet that loads the bronze, computes the value, and calls
      `dataset_append` to upsert each row.
  (c) RELAX THE CONTRACT — when the field doesn't exist in the source at
      all. Call `add_contract` with a new FieldsContract that drops the
      unreachable field, AND state in your `thought` why.

Never call `finish` while `missing fields:` is non-empty. Either fix,
derive, or drop.

## NO PLACEHOLDER STUFFING (this is a forbidden shortcut)

When a `required_fields` contract is failing, do NOT make it pass by
writing the same value ("N/A", "Unknown", "-", "0", "TBD", or any
arbitrary string) into the missing column on every row. That value
satisfies nothing — it just hides the missing data behind a constant.

The system surfaces this kind of stuffing automatically via a
**low-cardinality signal**: a required field whose mode value covers
most rows shows up in the contract detail as
`low-cardinality: <field>=<value> on N/M rows`. Whether that value is a
real constant (like `"FR"` for a country column on a France-only source)
or a placeholder, the agent has to look at it and decide.

The three legitimate options when a field is genuinely missing:
  (a) FIX the regex (the field IS on the page, the extractor missed it).
  (b) DERIVE the value (e.g. `sanction_year = YEAR(sanction_date)`).
  (c) RELAX the contract via `add_contract` — pass a new FieldsContract
      that lists ONLY the fields the source actually carries, with an
      honest `notes` explanation of what's dropped and why.

If none of (a)/(b)/(c) applies, the field genuinely doesn't exist in the
source. Choose (c) — don't invent data.

## NEVER edit dataset.jsonl from `python` directly

The bronze dataset is mutated through the dataset tools, not by writing to
the JSONL file by hand. Hand-edits desync the in-memory state in older
versions of this system and create write loops where `extract_items` keeps
seeing the pre-edit rows. The dataset now detects external mtime changes
and reloads, but you should still use `dataset_append` (which auto-ids
missing rows and validates), not `f.write(...)` on the JSONL.

## Always write JSON with `ensure_ascii=False`

When you DO write JSON files via the `python` tool (intermediate scratch,
debug dumps, anything), **always pass `ensure_ascii=False`**:

  json.dumps(row, ensure_ascii=False)              # good — keeps "É"
  json.dump(rows, f, ensure_ascii=False)           # good
  json.dumps(row)                                  # bad  — writes "É"

Default `json.dumps` escapes every non-ASCII character to `\uXXXX`. That's
valid JSON but unreadable to humans, breaks `grep`, and bloats files. Any
source that touches non-English text (French, Spanish, German, Arabic,
Chinese, accented Latin) will end up unreadable on disk.

Read the same way:

  open(path, encoding="utf-8")                     # always pass encoding
  Path(p).read_text(encoding="utf-8")              # always pass encoding

## Tool errors that report state, not just "no"

When a tool fails, read the FULL output. Modern tools (codebook_propose,
extract_items, etc.) print the observed state on error: bronze count,
queue length, codebook hash, etc. Use those numbers to decide your next
move. Example: "dataset is empty" with `queue items: 383` means call
`dataset_from_queue`, not `scrape_paginated` again.

## "Added 0 net new" — codebook_edit add reports replaced vs added

`codebook_edit operation=add` will tell you when a variable already
existed and got overwritten ("replaced N EXISTING variables"). If every
one of your "adds" was a replace, your codebook didn't actually grow.
That's almost always a sign you're trying to tweak existing variables —
use `operation=rename` or pass `variables=[{name: 'new_name', ...}]` with
fresh names instead.

# Self-verification (the bar you must clear before `finish`)

After your last harvest+extract+export, you'll be asked to self-verify:
  - All contracts OK.
  - The dataset is non-empty and rows are homogeneous (same keys, same types).
  - Sampled rows actually answer the user's question — no placeholder rows,
    no missing fields the user explicitly asked for.

If verification fails, you'll be re-launched with the issues listed; fix
them, then call `finish` again.

# Hugging Face push

If the goal mentions Hugging Face (or the user passed --push), call
`hf_push(repo_id="<owner>/<name>", private=true|false)` AFTER `dataset_export`
and BEFORE `finish`. The tool reads from <workdir>/export and pushes parquet
+ codebook.md + README.

# Safety

Destructive bash (rm, dd, mkfs, sudo, ...) is blocked at the tool layer.
Don't try to bypass it.
"""


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"… [+{len(s) - n} more]"


def render_state_brief(state: "AgentState", registry: "ToolRegistry") -> str:
    brief = state.to_brief()

    min_rows = None
    for c in state.contracts.list():
        if isinstance(c, MinRowsContract):
            min_rows = c.min_rows
            break
    phase = current_phase(state, contract_min_rows=min_rows)

    relevant: list[str] = []
    for tname in phase.tools:
        spec = registry.get(tname)
        if spec is None:
            continue
        relevant.append(f"- {spec.name}: {spec.description}")

    other_names = [n for n in registry.names() if n not in phase.tools]

    contracts_block: list[str] = []
    for c in brief["contracts"]:
        mark = "OK" if c["ok"] else "FAIL"
        contracts_block.append(f"  [{mark}] {c['name']}: {c['detail']}")
    if not contracts_block:
        contracts_block = ["  (none defined)"]

    # Surface per-field gaps if a required_fields contract is reporting
    # missing-counts. The contract `detail` already contains them but it's
    # easy to skim past — re-render them in human form right under the
    # contract block so the agent sees what to fix.
    field_gap_lines: list[str] = []
    rows_total = brief.get("dataset_rows") or 0
    for c in brief["contracts"]:
        if c.get("ok") or "missing fields" not in (c.get("detail") or ""):
            continue
        import re as _re
        # detail looks like: "missing fields: {'a': 4, 'b': 275}"
        m = _re.search(r"\{([^}]*)\}", c["detail"])
        if not m:
            continue
        pairs = _re.findall(r"'([^']+)'\s*:\s*(\d+)", m.group(1))
        if not pairs and rows_total == 0:
            continue
        field_gap_lines.append(
            "# 🔍 Per-field gaps (rows where the field is missing or null)"
        )
        for name, n_missing_s in pairs:
            n_missing = int(n_missing_s)
            n_present = max(0, rows_total - n_missing)
            pct_present = (n_present / rows_total) if rows_total else 0.0
            field_gap_lines.append(
                f"  {name:<24} present in {n_present}/{rows_total} ({pct_present:.0%})  ·  missing in {n_missing}"
            )
        field_gap_lines.append(
            "  → For each gap, decide one of:\n"
            "     (a) the extractor is dropping rows of a second shape — fix the regex (extractor_define).\n"
            "     (b) the field is DERIVABLE from another column (e.g. year = YEAR(date)) — compute it via python + dataset_append (upsert).\n"
            "     (c) the field doesn't exist in this source at all — drop the contract via add_contract (with a relaxed required_fields list)."
        )
        break

    queue = state.memory.get("queue", []) or []
    processed = set(str(x) for x in (state.memory.get("processed", []) or []))
    remaining_items = [
        q for q in queue
        if isinstance(q, dict) and str(q.get("id")) not in processed
    ]
    queue_block: list[str] = [
        f"  queue length: {len(queue)}",
        f"  processed:    {len(processed)}",
        f"  remaining:    {len(remaining_items)}",
    ]
    if remaining_items:
        nxt = remaining_items[0]
        s = json.dumps(nxt, ensure_ascii=False)
        queue_block.append(f"  NEXT to do:   {_truncate(s, 240)}")

    extractors = state.memory.get("extractors", {}) or {}
    extr_block: list[str] = []
    for name, spec in extractors.items():
        kind = "listing" if spec.get("row_pattern") else "detail"
        fields = list((spec.get("fields") or {}).keys())
        extr_block.append(f"  - {name} ({kind})  fields: {fields}")
    if not extr_block:
        extr_block = ["  (none defined yet — define one in DISCOVER phase)"]

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

    _list_dir("export", workdir_path / "export", head=10)
    _list_dir("notes", workdir_path / "notes", head=10)

    root_files = [
        p for p in workdir_path.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ]
    interesting = [p for p in root_files if p.name in {
        "dataset.jsonl", "extracted.jsonl", "codebook.json", "memory.json",
    }]
    other = [p for p in root_files if p not in interesting]
    if interesting or other:
        workspace_block.append("  (root):")
        for f in sorted(interesting, key=lambda p: p.name):
            workspace_block.append(f"    {f.name}  ({f.stat().st_size:,} bytes)")
        if other:
            preview = ", ".join(sorted(p.name for p in other)[:6])
            workspace_block.append(
                f"    [other root files: {len(other)}] {preview}"
                + ("…" if len(other) > 6 else "")
            )

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

    # Loop detector.
    loop_warning = ""
    last5 = state.history[-5:]
    if len(last5) >= 3:
        sigs = [(h.tool, json.dumps(h.args, sort_keys=True)[:120]) for h in last5]
        most = max(set(sigs), key=sigs.count)
        if sigs.count(most) >= 3:
            tool_name, _ = most
            loop_warning = (
                f"\n# ⚠️  LOOP DETECTED — '{tool_name}' called {sigs.count(most)}× "
                f"in the last {len(last5)} turns with the same args. CHANGE STRATEGY.\n"
            )

    # Generic repeated-failure detector: any tool that has failed ≥3 times in
    # the last 8 turns is a sign the agent is fighting the same wall. Force a
    # strategy change instead of letting it retry the same call shape.
    last8 = state.history[-8:]
    fail_counts: dict[str, int] = {}
    sample_obs: dict[str, str] = {}
    # A "soft failure" is a call that didn't raise but produced nothing useful
    # — e.g. extractor_define matching 0 rows, scrape_paginated adding 0,
    # extract_items processing 0 rows. These are the silent thrash patterns.
    SOFT_FAIL_MARKERS = (
        "matched_rows: 0",
        "total_added: 0",
        "processed=0 ",
        "appended to dataset: 0",
        "extracted 0 row",
        "nothing to do",
    )
    for h in last8:
        if not h.tool or h.tool.startswith("_"):
            continue
        obs = h.observation or ""
        soft_fail = any(m in obs for m in SOFT_FAIL_MARKERS)
        if h.error or soft_fail:
            fail_counts[h.tool] = fail_counts.get(h.tool, 0) + 1
            sample_obs.setdefault(h.tool, obs[:200])
    repeated_failures = [(t, c) for t, c in fail_counts.items() if c >= 3]
    if repeated_failures:
        alt_hints = {
            "codebook_edit": (
                "STOP calling codebook_edit. If the tool keeps rejecting the "
                "args, you're probably using the wrong operation shape. Read "
                "its spec: operation must be one of drop/rename/add/set_required, "
                "and each operation takes a SPECIFIC key (drop→names, rename→"
                "renames, add→variables). If the schema is fine and you just "
                "want different vars, use codebook_propose(replace=true)."
            ),
            "codebook_propose": (
                "STOP calling codebook_propose. The tool is write-once by "
                "default. If a codebook already exists, use codebook_edit to "
                "tweak it; only call codebook_propose(replace=true) when you "
                "truly want to throw away the schema and re-extract everything."
            ),
            "extract_items": (
                "STOP calling extract_items in the same shape. If it keeps "
                "saying 'nothing to do', the silver is already complete — "
                "move to dataset_export. If it's refusing skip_existing=false, "
                "use fill_new_only=true (cheap) or set force=true."
            ),
            "extractor_define": (
                "STOP iterating on the same regex spec. Switch to llm_scrape, "
                "which reads the page directly with no regex needed."
            ),
            "llm_scrape": (
                "STOP calling llm_scrape with the same source. If it's "
                "returning 0 rows or refusing the push, switch to "
                "extractor_define (with html_inspect output to anchor the "
                "row_pattern) or use python to call the underlying JSON API."
            ),
            "dataset_append": (
                "STOP calling dataset_append in the same shape. The rows are "
                "probably failing validation (missing required field, wrong "
                "schema). Inspect one row with dataset_sample and fix the "
                "shape before appending more."
            ),
            "python": (
                "STOP retrying the same python snippet. If it's raising the "
                "same exception, read the traceback and fix it — don't retry "
                "blindly. If the API is rate-limiting, add backoff or switch "
                "to llm_scrape on the cached HTML."
            ),
        }
        lines = ["\n# 🛑 REPEATED FAILURES — change strategy, do NOT retry the same call shape."]
        for tool, count in repeated_failures:
            lines.append(f"  • '{tool}' failed {count}× in the last {len(last8)} turns.")
            if tool in alt_hints:
                lines.append(f"    → {alt_hints[tool]}")
            obs = sample_obs.get(tool, "")
            if obs:
                lines.append(f"    last error sample: {obs}")
        loop_warning += "\n".join(lines) + "\n"

    no_progress = ""
    if len(state.history) >= 10 and brief["dataset_rows"] == 0:
        no_progress = (
            f"\n# ⚠️  {len(state.history)} TURNS, ZERO ROWS — files on disk don't count. "
            "Call dataset_append (or process_queue / scrape_paginated) to advance.\n"
        )

    # BRONZE CHURN DETECTOR: dataset row count flips up/down repeatedly,
    # which means the agent is re-harvesting the same source with slightly
    # different settings, producing a different shape each time. Mixed
    # shapes = unusable downstream. Tell the agent to stop.
    churn_warning = ""
    if "_row_count_history" not in state.memory.keys():
        state.memory.set("_row_count_history", [])
    rc_hist = list(state.memory.get("_row_count_history") or [])
    current_n = int(brief["dataset_rows"] or 0)
    if not rc_hist or rc_hist[-1] != current_n:
        rc_hist.append(current_n)
        if len(rc_hist) > 12:
            rc_hist = rc_hist[-12:]
        try:
            state.memory.set("_row_count_history", rc_hist)
        except Exception:  # noqa: BLE001
            pass
    if len(rc_hist) >= 5:
        recent_counts = rc_hist[-6:]
        flips = sum(
            1 for i in range(2, len(recent_counts))
            if (recent_counts[i] > recent_counts[i - 1]) != (recent_counts[i - 1] > recent_counts[i - 2])
        )
        if flips >= 2:
            churn_warning = (
                "\n# 🛑  BRONZE CHURN DETECTED — row count went "
                + " → ".join(str(n) for n in recent_counts)
                + " over the last few turns. The dataset shape is changing each "
                + "harvest, which means downstream extraction will be inconsistent.\n"
                + "  STOP scraping. Either:\n"
                + "    • call dataset_export with the current rows, OR\n"
                + "    • dataset_sample to see what's already there, OR\n"
                + "    • set_plan to re-lock the strategy, then ONE final harvest with force=true.\n"
            )

    # Verification hint, if the agent is re-entering the loop after a failed verify.
    verify_hint = state.memory.get("_verify_hint")
    verify_block = ""
    if verify_hint:
        verify_block = (
            "# 🧪 Self-verification feedback from your previous `finish` attempt\n"
            f"{verify_hint}\n\n"
        )

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
        plan_block = (
            "# 🗺 Plan (stick to this — do not silently switch sources)\n"
            f"  item:     {plan.get('item')}\n"
            f"  source:   {plan.get('source')}  ({plan.get('source_url')})\n"
            f"  paginate: {plan.get('pagination')}\n"
            + math_line + "\n"
            + f"  strategy: {plan.get('harvest_strategy')}\n"
            + ("  fields:\n" + "\n".join(fields_lines) + "\n" if fields_lines else "")
            + ("  notes: " + str(plan.get("notes")) + "\n" if plan.get("notes") else "")
            + "\n"
        )
    else:
        n_http_gets = sum(1 for h in state.history if h.tool == "http_get")
        if n_http_gets >= 1 and brief["dataset_rows"] < 5:
            plan_block = (
                "# 🗺 Plan (REQUIRED before harvesting)\n"
                "  No plan saved yet. Call `set_plan(...)` with:\n"
                "    item, source, source_url, pagination, items_per_page,\n"
                "    target_rows, pages_needed, harvest_strategy, fields.\n\n"
            )

    return (
        "# Goal\n"
        f"{brief['goal']}\n\n"
        + verify_block
        + plan_block
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
        + (("\n".join(field_gap_lines) + "\n\n") if field_gap_lines else "")
        + "# Queue\n"
        + "\n".join(queue_block)
        + "\n\n"
        "# Extractors saved\n"
        + "\n".join(extr_block)
        + "\n\n"
        "# Workspace\n"
        + "\n".join(workspace_block)
        + "\n\n"
        "# Recent turns\n"
        + "\n".join(history_block)
        + loop_warning
        + no_progress
        + churn_warning
        + "\n\n"
        "# What to do next\n"
        "Pick ONE tool from the phase list (or an escape hatch if truly needed) "
        "and emit the JSON tool call. Remember: $file references for large content.\n\n"
        f"# $file syntax\n  {describe_ref_syntax()}"
    )
