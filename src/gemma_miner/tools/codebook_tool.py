"""Codebook tools: propose, show, edit, test.

These tools manage the named `Codebook` stored at `<workdir>/codebook.json`.
The codebook is the structured-extraction contract — once locked, every row
in the final dataset must conform to it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_miner.codebook import Codebook, VariableSpec
from gemma_miner.parsing import _candidates, _strip_trailing_commas, _repair_invalid_escapes
from gemma_miner.stats import codebook_stats, format_stats
from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.llm import LLMClient
    from gemma_miner.state import AgentState


# ── shared silver-migration helpers (used by edit + replace flows) ─────────


def _scrub_silver(
    state: "AgentState",
    *,
    drop_keys: set[str] | None = None,
    rename_map: dict[str, str] | None = None,
    keep_only: set[str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Mutate extracted.jsonl in place to honour codebook changes.

    Returns (n_rows_touched, changes_summary).
    """
    drop_keys = drop_keys or set()
    rename_map = rename_map or {}
    extracted_ds = state.extracted_dataset()
    rows = extracted_ds.rows()
    summary: dict[str, int] = {"dropped_keys": 0, "renamed_keys": 0, "out_of_schema": 0}
    if not rows:
        return 0, summary
    new_rows: list[dict] = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            if k in drop_keys:
                summary["dropped_keys"] += 1
                continue
            new_k = rename_map.get(k, k)
            if new_k != k:
                summary["renamed_keys"] += 1
            if keep_only is not None and new_k not in keep_only and not str(new_k).startswith("_") and new_k != "id":
                summary["out_of_schema"] += 1
                continue
            nr[new_k] = v
        new_rows.append(nr)
    # Rewrite the silver file from scratch.
    p = extracted_ds.path
    import os as _os

    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(tmp, p)
    # Force the in-memory Dataset to reload by clearing the cached attribute.
    state._extracted_dataset = None
    return len(new_rows), summary


def _wipe_silver(state: "AgentState") -> int:
    """Empty extracted.jsonl. Returns the number of rows discarded."""
    extracted_ds = state.extracted_dataset()
    n = len(extracted_ds)
    p = extracted_ds.path
    p.write_text("", encoding="utf-8")
    state._extracted_dataset = None
    # Clear the codebook-hash memory so the phase machine doesn't think the
    # current rows are already extracted.
    state.memory.set("last_extracted_codebook_hash", None)
    state.memory.set("last_extracted_codebook_variables", [])
    state.memory.set("_post_extract_done", False)
    return n


_PROPOSAL_SYSTEM = """You are a senior data scientist designing a CODEBOOK.

A codebook is a typed list of variables (columns) to extract from a corpus of
documents. The goal is RESEARCH-GRADE STRUCTURED DATA: data ready for
statistics, machine learning, and public release.

RULES:
1. Propose 20–60 variables.
2. Aim for ≥50% NUMERIC (integer/float) or BOOLEAN variables when the source
   plausibly supports counts, dates, amounts, or boolean facts. If the
   USER'S REQUIRED FIELDS are mostly identifiers / titles / descriptions
   (i.e. metadata extraction), prefer STRING/ENUM variables that match the
   user's brief over inventing speculative booleans. Never invent a variable
   that cannot be answered from the sampled text — placeholder booleans like
   `is_fintech` on a generic listing page produce all-null columns and bias
   downstream stats.
3. ALWAYS include the user's required fields verbatim (same exact names),
   when supplied. They are the contract; the rest of the codebook is
   additional structure.
4. Use ENUMS for any categorical dimension with a small fixed set of values.
5. Use DATES (YYYY-MM-DD) for every time fact.
6. Each variable needs a clear ONE-SENTENCE description.
7. Only include variables PLAUSIBLY extractable from the documents you saw.
8. Cover the full dimensionality: counts, dates, amounts, categories,
   boolean facts, identifiers, severities, party roles, outcomes.

NAMING CONVENTIONS:
  n_*       integer count
  pct_*     percentage 0–100 (float)
  amount_*  monetary amount (float, declare unit)
  is_*      boolean fact
  has_*     boolean fact
  cat_*     categorical / enum
  dn_*      date (YYYY-MM-DD)

OUTPUT FORMAT — A SINGLE JSON OBJECT ONLY (no prose, no fences):

{
  "name": "<short snake_case id, e.g. cnil_sanctions>",
  "description": "<one-paragraph summary of the dataset>",
  "domain": "<short domain label, e.g. 'GDPR enforcement'>",
  "variables": [
    {"name": "n_violations", "type": "integer", "description": "...", "min_value": 0},
    {"name": "is_security_breach", "type": "boolean", "description": "..."},
    {"name": "amount_fine_eur", "type": "float", "unit": "euros", "description": "..."},
    {"name": "cat_org_type", "type": "enum", "enum_values": ["..."], "description": "..."},
    {"name": "dn_decision", "type": "date", "description": "..."}
  ]
}
"""


def _parse_codebook_proposal(raw: str) -> dict:
    """Tolerant parse of the LLM proposal (handles fences, repaired escapes)."""
    for cand in _candidates(raw):
        for variant in (cand, _strip_trailing_commas(cand), _repair_invalid_escapes(cand)):
            try:
                obj = json.loads(variant)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and "variables" in obj:
                return obj
    raise ValueError("could not parse a codebook proposal from the model output")


class CodebookProposeTool(Tool):
    name = "codebook_propose"
    description = (
        "Sample N items' text content, ask the LLM to design a CODEBOOK of "
        "20–60 typed variables (booleans, integers, floats, enums, dates), "
        "then save it to `<workdir>/codebook.json`. Run this ONCE after the "
        "harvest phase to produce the dataset schema, then iterate by calling "
        "`codebook_test` to see coverage and `codebook_propose` again with a "
        "feedback hint to refine. The proposal sees the FULL text of the "
        "sampled items (truncated to a budget), so the variables are tailored "
        "to your corpus, not a generic template."
    )
    args_schema = {
        "sample_size": {"type": "integer", "default": 4, "description": "How many items to feed the LLM."},
        "max_chars_per_item": {"type": "integer", "default": 12000},
        "domain_hint": {"type": "string", "description": "Optional short label, e.g. 'GDPR enforcement', 'competition law'."},
        "feedback": {"type": "string", "description": "Optional natural-language note for the proposer (used when iterating)."},
        "replace": {
            "type": "boolean",
            "default": False,
            "description": (
                "Required (true) to overwrite an existing codebook.json. The "
                "codebook is write-once by default: replacing it invalidates "
                "every previously-extracted row. Use codebook_edit for "
                "incremental tweaks instead of regenerating the whole schema."
            ),
        },
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        sample_size = int(args.get("sample_size") or 4)
        max_chars = int(args.get("max_chars_per_item") or 12000)
        rows = state.dataset.rows()
        if not rows:
            # Report the state the agent needs to see; don't assume the cause.
            queue = state.memory.get("queue", []) or []
            extractors = state.memory.get("extractors", {}) or {}
            facts = [
                "ERROR: dataset (bronze) is empty — codebook design needs sampled item text to work from.",
                "",
                "Observed state:",
                f"  bronze rows:        {len(state.dataset)}",
                f"  queue items:        {len(queue)}",
                f"  extractors defined: {list(extractors.keys()) or '(none)'}",
            ]
            if queue:
                facts.append(
                    "  → the queue has items but bronze is empty. Tools that flush queue → dataset:"
                    " dataset_from_queue (listing-only), process_queue (detail/text/multi_asset)."
                )
            else:
                facts.append(
                    "  → no queue either. Tools that produce rows: scrape_paginated (with a saved"
                    " listing extractor), llm_scrape, python (for JSON APIs)."
                )
            return ToolResult(output="\n".join(facts), error=True)

        # WRITE-ONCE GUARD: refuse to overwrite an existing codebook silently.
        cb_path = Path(state.workdir) / "codebook.json"
        replace_flag = bool(args.get("replace"))
        if cb_path.exists() and not replace_flag:
            try:
                existing = Codebook.load(cb_path)
                n_vars = len(existing.variables)
            except Exception:  # noqa: BLE001
                n_vars = -1
            return ToolResult(
                output=(
                    "ERROR: a codebook already exists at "
                    f"{cb_path} ({n_vars} variables). codebook_propose is "
                    "WRITE-ONCE by default — overwriting it invalidates every "
                    "previously-extracted row and forces a full re-extract.\n\n"
                    "Use one of:\n"
                    "  • `codebook_edit(operation='drop'|'rename'|'add'|...)` "
                    "for incremental tweaks (cheap, keeps extracted rows valid).\n"
                    "  • `codebook_propose(replace=true, feedback='...')` if "
                    "you REALLY want to redesign from scratch (expensive)."
                ),
                error=True,
            )

        # Sample items deterministically (first/middle/last + random) for diversity.
        sampled = _stratified_sample(rows, sample_size)
        item_texts: list[tuple[str, str]] = []
        for r in sampled:
            text = _row_text(r, state.workdir)
            if text:
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n[truncated, full {len(text)} chars]"
                item_texts.append((str(r.get("id", "?")), text))
        if not item_texts:
            return ToolResult(
                output=(
                    "ERROR: none of the sampled rows yielded text content. Make sure rows have "
                    "either a `text_path` field pointing to a .txt file, or a `pdf_text` field "
                    "with inline content."
                ),
                error=True,
            )

        user_lines: list[str] = []
        if args.get("domain_hint"):
            user_lines.append(f"DOMAIN HINT: {args['domain_hint']}")
        if args.get("feedback"):
            user_lines.append("FEEDBACK ON PREVIOUS PROPOSAL: " + str(args["feedback"]))
        # Surface the user's required-field contract so the proposer cannot
        # ignore the user's brief. Generic — applies to any website / domain.
        try:
            required_from_contract = sorted(state.contracts.locked_required_fields())
        except Exception:  # noqa: BLE001
            required_from_contract = []
        if required_from_contract:
            user_lines.append(
                "USER REQUIRED FIELDS (must appear verbatim in the codebook): "
                + ", ".join(required_from_contract)
            )
        user_lines.append(f"\nGoal: {state.goal}\n")
        for iid, text in item_texts:
            user_lines.append(f"\n=== SAMPLE ITEM {iid} ===\n{text}\n")
        user = "\n".join(user_lines)

        raw = self.llm.chat(
            [
                {"role": "system", "content": _PROPOSAL_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        try:
            data = _parse_codebook_proposal(raw)
        except ValueError as e:
            return ToolResult(
                output=f"ERROR parsing proposal: {e}\nRaw output (first 1200 chars):\n{raw[:1200]}",
                error=True,
            )
        try:
            cb = Codebook.from_dict(data)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR building codebook: {e}", error=True)
        errs = cb.validate()
        if errs:
            return ToolResult(
                output="ERROR: invalid codebook:\n  " + "\n  ".join(errs),
                error=True,
            )

        # Duplicate-variable detector: warn (and auto-rename one of each pair
        # to avoid carrying stale columns alongside their replacement).
        dup_groups = cb.find_duplicate_groups()
        dup_lines: list[str] = []
        if dup_groups:
            for group in dup_groups:
                dup_lines.append(
                    f"  • duplicates of same semantic stem: {group} — "
                    "kept the first; downstream extraction will not see the rest."
                )
            # Keep the FIRST in each group; drop the others.
            survivors: set[str] = set()
            to_drop: set[str] = set()
            for group in dup_groups:
                survivors.add(group[0])
                to_drop.update(group[1:])
            cb.variables = [v for v in cb.variables if v.name not in to_drop]

        path = Path(state.workdir) / "codebook.json"
        cb.save(path)
        state.memory.set("codebook_path", str(path))

        # SCHEMA MIGRATION on replace: wipe the silver dataset so we don't end
        # up with rows half-extracted under the old schema and half under the
        # new one (the symptom that produced the HN dataset with 50 columns).
        wipe_note = ""
        if replace_flag and cb_path.exists():
            n_wiped = _wipe_silver(state)
            if n_wiped:
                wipe_note = f"  (replace=true → wiped {n_wiped} silver rows; re-extract from scratch)"

        tb = cb.type_breakdown()
        ratio = cb.numeric_or_boolean_ratio()
        out = [
            f"codebook saved → {path}",
            f"name: {cb.name}",
            f"variables: {len(cb.variables)}",
            f"type breakdown: {tb}",
            f"numeric_or_boolean_ratio: {ratio:.0%}",
            "",
            "First 8 variables:",
        ]
        for v in cb.variables[:8]:
            extra = f" [{','.join(v.enum_values)}]" if v.enum_values else ""
            out.append(f"  - {v.name}: {v.type}{extra} — {v.description[:80]}")
        if dup_lines:
            out.append("")
            out.append("⚠ duplicate-stem variables removed (codebook drift):")
            out.extend(dup_lines)
        if wipe_note:
            out.append("")
            out.append(wipe_note)
        if ratio < 0.5:
            out.append("")
            out.append(
                "⚠️  numeric+boolean ratio is low. Consider iterating with feedback "
                "like 'replace string variables with booleans or enums where possible'."
            )
        return ToolResult(output="\n".join(out))


# ── helpers ─────────────────────────────────────────────────────────────────


def _stratified_sample(rows: list[dict], k: int) -> list[dict]:
    if len(rows) <= k:
        return rows
    indices = set()
    # take first, last
    indices.add(0)
    indices.add(len(rows) - 1)
    # spread the rest
    rnd = random.Random(42)
    while len(indices) < k:
        indices.add(rnd.randrange(0, len(rows)))
    return [rows[i] for i in sorted(indices)]


def _row_text(row: dict, workdir: str) -> str:
    """Return textual content from a row.

    Priority:
      1. File-path fields (text_path, txt_path, …) — read from disk.
      2. Heavy in-row text fields (pdf_text, text, body, content) > 200 chars.
      3. Fallback: any non-id, non-url string field (or list of strings).
         Concatenated as `field: value` lines so the codebook designer has
         SOMETHING to work with, even when the rows are short structured
         records (e.g. {date, org_type, decision_adopted}).
    """
    # 1. file-path fields
    for field in ("text_path", "txt_path", "txt_file", "text_file"):
        path = row.get(field)
        if isinstance(path, str):
            p = Path(path)
            if not p.is_absolute():
                p = Path(workdir) / p
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass
    # 2. heavy in-row text
    for field in ("pdf_text", "text", "body", "content"):
        v = row.get(field)
        if isinstance(v, str) and len(v) > 200:
            return v
    # 3. fallback: concatenate every meaningful string field on the row.
    SKIP = {"id"} | {
        "text_path", "txt_path", "txt_file", "text_file",
        "pdf_text", "pdf_path", "attachment", "attachment_path",
    }
    pieces: list[str] = []
    for k, v in row.items():
        if k in SKIP or str(k).startswith("_"):
            continue
        if isinstance(v, str):
            if not v.strip():
                continue
            pieces.append(f"{k}: {v}")
        elif isinstance(v, (int, float, bool)):
            pieces.append(f"{k}: {v}")
        elif isinstance(v, list) and v:
            joined = ", ".join(str(x) for x in v[:20] if x is not None)
            if joined:
                pieces.append(f"{k}: {joined}")
    return "\n".join(pieces)


# ── codebook_show ──────────────────────────────────────────────────────────


def _load_codebook_from_state(state: "AgentState") -> Codebook | None:
    path = state.memory.get("codebook_path")
    if path is None:
        candidate = Path(state.workdir) / "codebook.json"
        if candidate.exists():
            path = str(candidate)
            state.memory.set("codebook_path", path)
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Codebook.load(p)
    except Exception:  # noqa: BLE001
        return None


class CodebookShowTool(Tool):
    name = "codebook_show"
    description = (
        "Pretty-print the current codebook (variables, types, descriptions). "
        "Use this when you want to inspect the schema before extracting."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="(no codebook saved yet — call codebook_propose first)")
        out = [
            f"name: {cb.name}",
            f"description: {cb.description}",
            f"version: {cb.version}  domain: {cb.domain}",
            f"variables: {len(cb.variables)}    type breakdown: {cb.type_breakdown()}",
            f"numeric_or_boolean_ratio: {cb.numeric_or_boolean_ratio():.0%}",
            "",
        ]
        for v in cb.variables:
            tag = v.type
            if v.enum_values:
                tag += f" [{', '.join(v.enum_values)}]"
            if v.unit:
                tag += f" ({v.unit})"
            out.append(f"  {v.name:<28} {tag}")
            out.append(f"    {v.description}")
        return ToolResult(output="\n".join(out))


# ── codebook_edit ──────────────────────────────────────────────────────────


class CodebookEditTool(Tool):
    name = "codebook_edit"
    description = (
        "Mutate the saved codebook. Supported operations:\n"
        "  - drop:    {operation: 'drop', names: ['x','y']}\n"
        "  - rename:  {operation: 'rename', renames: {'old':'new'}}\n"
        "  - add:     {operation: 'add', variables: [{name,type,description,...}]}\n"
        "  - set_required: {operation: 'set_required', names: [...]}\n"
        "Use this to tighten the codebook after `codebook_test` — drop dead "
        "variables, narrow enums, etc."
    )
    args_schema = {
        "operation": {"type": "string", "enum": ["drop", "rename", "add", "set_required"]},
        "names": {"type": "array", "items": {"type": "string"}},
        "renames": {"type": "object"},
        "variables": {"type": "array", "items": {"type": "object"}},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook saved", error=True)
        op = args.get("operation")
        silver_migration: dict[str, Any] = {}
        # User-locked required fields cannot be dropped/renamed via codebook_edit.
        # The user named these in their brief; weakening them silently is contract
        # gaming. Generic — works for any website / any user-supplied field set.
        locked_fields: set[str] = set()
        try:
            locked_fields = state.contracts.locked_required_fields()
        except Exception:  # noqa: BLE001
            locked_fields = set()
        if op == "drop":
            names = args.get("names") or []
            blocked = [n for n in names if n in locked_fields]
            if blocked:
                return ToolResult(
                    output=(
                        f"ERROR: refused — cannot drop user-locked required field(s): {blocked}. "
                        "These fields are part of the user's brief; the run must satisfy them or "
                        "call finish(force=true) with a summary explaining why."
                    ),
                    error=True,
                )
            removed = []
            for n in names:
                if cb.remove_variable(n):
                    removed.append(n)
            result = f"dropped {len(removed)}: {removed}"
            # Migrate silver: drop the same keys from extracted rows.
            if removed:
                touched, summary = _scrub_silver(state, drop_keys=set(removed))
                silver_migration = {"silver_rows_touched": touched, **summary}
        elif op == "rename":
            renames = args.get("renames") or {}
            blocked = [k for k in renames.keys() if k in locked_fields]
            if blocked:
                return ToolResult(
                    output=(
                        f"ERROR: refused — cannot rename user-locked required field(s): {blocked}. "
                        "These names are part of the user's brief and must appear unchanged in "
                        "the final dataset."
                    ),
                    error=True,
                )
            done = []
            for old, new in renames.items():
                v = cb.variable(old)
                if v:
                    v.name = new
                    done.append((old, new))
            result = f"renamed {len(done)}: {done}"
            if done:
                touched, summary = _scrub_silver(state, rename_map=dict(done))
                silver_migration = {"silver_rows_touched": touched, **summary}
        elif op == "add":
            new_vars = args.get("variables") or []
            existing_names = {v.name for v in cb.variables}
            added: list[str] = []
            replaced: list[str] = []
            for raw in new_vars:
                try:
                    v = VariableSpec(**raw)
                    errs = v.validate()
                    if errs:
                        return ToolResult(output=f"ERROR in {raw.get('name')!r}: {errs}", error=True)
                    if v.name in existing_names:
                        replaced.append(v.name)
                    else:
                        added.append(v.name)
                    cb.upsert_variable(v)
                except Exception as e:  # noqa: BLE001
                    return ToolResult(output=f"ERROR parsing variable: {e}", error=True)
            parts = [f"added {len(added)}: {added}"]
            if replaced:
                parts.append(
                    f"replaced {len(replaced)} EXISTING variable(s): {replaced}  "
                    "(net new: 0 for these — `add` overwrites a variable that "
                    "already exists; use a different name if you wanted both)"
                )
            result = "\n".join(parts)
        elif op == "set_required":
            names = set(args.get("names") or [])
            changed = []
            for v in cb.variables:
                if v.name in names and not v.required:
                    v.required = True
                    changed.append(v.name)
            result = f"set required on {len(changed)}: {changed}"
        else:
            return ToolResult(output=f"ERROR: unknown operation {op!r}", error=True)
        path = Path(state.memory.get("codebook_path") or (Path(state.workdir) / "codebook.json"))
        cb.save(path)
        out = result + f"\nsaved → {path}\nnow has {len(cb.variables)} variables"
        if silver_migration:
            out += (
                f"\nsilver migrated: rows_touched={silver_migration.get('silver_rows_touched', 0)}  "
                f"dropped_keys={silver_migration.get('dropped_keys', 0)}  "
                f"renamed_keys={silver_migration.get('renamed_keys', 0)}"
            )
        return ToolResult(output=out)


# ── codebook_test ──────────────────────────────────────────────────────────


class CodebookTestTool(Tool):
    name = "codebook_test"
    description = (
        "Apply the current codebook to a SAMPLE of dataset items via "
        "structured LLM extraction, then report per-variable coverage and "
        "data-quality issues. Use this to find dead/redundant variables "
        "before the expensive full-corpus extraction."
    )
    args_schema = {
        "sample_size": {"type": "integer", "default": 3},
        "max_chars_per_item": {"type": "integer", "default": 16000},
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook (run codebook_propose first)", error=True)
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: dataset is empty", error=True)
        sample = _stratified_sample(rows, int(args.get("sample_size") or 3))
        max_chars = int(args.get("max_chars_per_item") or 16000)

        from gemma_miner.tools.extract_items_tool import extract_one_item

        extracted: list[dict] = []
        errors: list[str] = []
        for r in sample:
            try:
                row_out, _warn = extract_one_item(
                    self.llm, r, cb, state.workdir, max_chars=max_chars
                )
                extracted.append(row_out)
            except Exception as e:  # noqa: BLE001
                errors.append(f"  - {r.get('id', '?')}: {e}")
        if not extracted:
            return ToolResult(
                output="ERROR: no items extracted successfully.\n" + "\n".join(errors),
                error=True,
            )
        s = codebook_stats(extracted, cb)
        out = [format_stats(s)]
        if errors:
            out.append("")
            out.append(f"extraction errors on {len(errors)} items:")
            out.extend(errors)
        return ToolResult(output="\n\n".join(out))
