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

EVERY reply you emit must be a single JSON object — no prose around it, no
markdown fences:

  {"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}

# The big rule about content size

Never inline large content (PDF text, HTML bodies, anything > 5 KB) directly
into tool arguments. Your reply has a token budget; large arguments cause
the response to be cut off mid-string and the tool call fails.

Instead, use a $file REFERENCE:

  {"some_field": {"$file": "items/item_0001/attachment_01.txt"}}

The tool resolves the reference to the UTF-8 file content at dispatch time.
This works for ANY string-typed argument of ANY tool (except a few that
already take paths — read_file, save_attachment, etc.).

# How the system tracks progress (READ CAREFULLY)

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
`process_queue`. This is the central abstraction; prefer it over writing
ad-hoc python regex code on each turn.

# The happy path for "scrape N items with details"

  Phase DISCOVER_LISTING — http_get + html_inspect + python to figure out
  the row regex, then extractor_define(name='listing', spec={...}) and
  extractor_test to verify.

  Phase ENUMERATE — scrape_paginated(url_template='.../?page={page}',
  extractor_name='listing', target_count=<min_rows>). ONE call paginates
  and queues everything.

  Phase DISCOVER_DETAIL — queue_next + http_get the detail page + python
  to find the attachment URL pattern. Then extractor_define a 'detail' spec
  whose `attachment_url` field captures the binary URL.

  Phase PROCESS — process_queue(detail_extractor='detail', batch_size=10,
  row_template={...}). ONE call processes K items end-to-end — fetches
  detail, downloads attachment, extracts text, appends row, marks done.
  Use $file: prefixed placeholders in the row_template for any field that
  comes from a written file (e.g. "pdf_text": "$file:{paths.text}").

  Phase FINISH — dataset_stats then finish(summary='...').

The STATE BRIEF will tell you which phase you're in and which tools to use.

# The codebook paradigm (READ THIS — it's how this system reaches research-grade output)

You don't just scrape text. You scrape text AND turn it into a typed
tabular dataset ready for statistics, machine learning, and public release.
That has FOUR acts:

  1. HARVEST    — bring items + their primary text into the dataset (existing).
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

    # Cache contents.
    cache_block: list[str] = []
    cache_dir = Path(state.workdir) / "cache"
    if cache_dir.exists():
        files = sorted(cache_dir.iterdir())
        if files:
            cache_block.append(f"  {len(files)} files cached (DO NOT re-fetch the same URL)")
            for f in files[-8:]:
                cache_block.append(f"    {f.name}  ({f.stat().st_size:,} bytes)")

    # Recent turns.
    recent = state.history[-6:]
    history_block: list[str] = []
    for h in recent:
        args_s = json.dumps(h.args, ensure_ascii=False)
        if len(args_s) > 200:
            args_s = args_s[:200] + "…"
        history_block.append(
            f"  turn {h.turn} → {h.tool}({args_s})\n"
            f"    {'[ERROR] ' if h.error else ''}observation: {_truncate(h.observation, 800)}"
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

    return (
        "# Goal\n"
        f"{brief['goal']}\n\n"
        f"# Current phase: {phase.name}\n"
        f"  goal: {phase.goal}\n\n"
        "# Hint for this phase\n"
        f"{phase.hint}\n\n"
        "# Tools for this phase\n"
        + "\n".join(relevant)
        + "\n\n"
        "# Other tools available (escape hatches)\n"
        f"  {', '.join(other_names)}\n\n"
        "# Dataset\n"
        f"  rows: {brief['dataset_rows']}    path: {brief['dataset_path']}\n\n"
        "# Contracts\n"
        + "\n".join(contracts_block)
        + "\n\n"
        "# Queue\n"
        + "\n".join(queue_block)
        + "\n\n"
        "# Extractors saved\n"
        + "\n".join(extr_block)
        + "\n\n"
        + ("# Cache\n" + "\n".join(cache_block) + "\n\n" if cache_block else "")
        + "# Recent turns\n"
        + "\n".join(history_block)
        + loop_warning
        + no_progress
        + stuck_discover
        + "\n\n"
        "# What to do next\n"
        "Pick ONE tool from the phase list (or an escape hatch if truly needed) "
        "and emit the JSON tool call. Remember: $file references for large content.\n\n"
        f"# $file syntax\n  {describe_ref_syntax()}"
    )
