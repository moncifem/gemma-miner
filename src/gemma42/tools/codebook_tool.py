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
from typing import TYPE_CHECKING

from gemma42.codebook import Codebook, VariableSpec
from gemma42.parsing import _candidates, _strip_trailing_commas, _repair_invalid_escapes
from gemma42.stats import codebook_stats, format_stats
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.llm import LLMClient
    from gemma42.state import AgentState


_PROPOSAL_SYSTEM = """You are a senior data scientist designing a CODEBOOK.

A codebook is a typed list of variables (columns) to extract from a corpus of
documents. The goal is RESEARCH-GRADE STRUCTURED DATA: data ready for
statistics, machine learning, and public release.

RULES:
1. Propose 20–60 variables.
2. At least 60% MUST be NUMERIC (integer/float) or BOOLEAN. Strings are
   reserved for IDs, names, and short categorical labels.
3. Use ENUMS for any categorical dimension with a small fixed set of values.
4. Use DATES (YYYY-MM-DD) for every time fact.
5. Each variable needs a clear ONE-SENTENCE description.
6. Only include variables PLAUSIBLY extractable from the documents you saw.
7. Cover the full dimensionality: counts, dates, amounts, categories,
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
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        sample_size = int(args.get("sample_size") or 4)
        max_chars = int(args.get("max_chars_per_item") or 12000)
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(
                output="ERROR: dataset is empty. Run the HARVEST phase first so there are items to sample.",
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

        path = Path(state.workdir) / "codebook.json"
        cb.save(path)
        state.memory.set("codebook_path", str(path))

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
    """Return the textual content of a row by reading from text_path, else pdf_text."""
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
    for field in ("pdf_text", "text", "body", "content"):
        v = row.get(field)
        if isinstance(v, str) and len(v) > 200:
            return v
    return ""


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
        if op == "drop":
            names = args.get("names") or []
            removed = []
            for n in names:
                if cb.remove_variable(n):
                    removed.append(n)
            result = f"dropped {len(removed)}: {removed}"
        elif op == "rename":
            renames = args.get("renames") or {}
            done = []
            for old, new in renames.items():
                v = cb.variable(old)
                if v:
                    v.name = new
                    done.append((old, new))
            result = f"renamed {len(done)}: {done}"
        elif op == "add":
            new_vars = args.get("variables") or []
            added = []
            for raw in new_vars:
                try:
                    v = VariableSpec(**raw)
                    errs = v.validate()
                    if errs:
                        return ToolResult(output=f"ERROR in {raw.get('name')!r}: {errs}", error=True)
                    cb.upsert_variable(v)
                    added.append(v.name)
                except Exception as e:  # noqa: BLE001
                    return ToolResult(output=f"ERROR parsing variable: {e}", error=True)
            result = f"added {len(added)}: {added}"
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
        return ToolResult(output=result + f"\nsaved → {path}\nnow has {len(cb.variables)} variables")


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

        from gemma42.tools.extract_items_tool import extract_one_item

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
