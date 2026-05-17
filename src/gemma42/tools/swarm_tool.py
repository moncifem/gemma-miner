"""Specialist-swarm tools: codebook_design (Curator + Adversary + Consolidator
in one call), audit_dataset, dataset_report.

Each tool wraps a specialist (or a small chain of specialists) so the
orchestrating agent only spends ONE LLM-tool round trip per macro action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.audit import audit_dataset as _audit, format_report
from gemma42.codebook import Codebook
from gemma42.stats import codebook_stats, format_stats
from gemma42.swarm import (
    consolidate_codebook,
    critique_codebook,
    propose_codebook,
    write_report,
)
from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.codebook_tool import _load_codebook_from_state, _stratified_sample, _row_text

if TYPE_CHECKING:
    from gemma42.llm import LLMClient
    from gemma42.state import AgentState


# ── codebook_design : Curator → Adversary → Consolidator (one tool) ────────


class CodebookDesignTool(Tool):
    name = "codebook_design"
    description = (
        "Run the full adversarial codebook design pipeline in ONE call:\n"
        "  1. CURATOR proposes 30-60 typed variables (numeric/boolean-heavy).\n"
        "  2. ADVERSARY critiques: identifies null-most, redundant, ambiguous,\n"
        "     mistyped, ungrounded, leaky variables. Proposes drop/merge/retype/tighten.\n"
        "  3. CONSOLIDATOR deterministically applies the critique.\n"
        "Saves the final codebook to <workdir>/codebook.json. Pre-empts the\n"
        "old codebook_propose+codebook_test loop with a single, sharper call.\n\n"
        "Args: sample_size (default 4), domain_hint (optional), min_variables (default 30)."
    )
    args_schema = {
        "sample_size":   {"type": "integer", "default": 4},
        "domain_hint":   {"type": "string"},
        "min_variables": {"type": "integer", "default": 30},
        "max_chars_per_item": {"type": "integer", "default": 12000},
        "variables":     {
            "type": "array",
            "description": (
                "Optional seed list of variables you want the codebook to "
                "include (each entry: {name, type, description}). Used as a "
                "STRONG hint to the curator; the adversary may still rename, "
                "merge, or retype. Pass when the user's prompt already enumerated "
                "the variables they want."
            ),
        },
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: dataset is empty; harvest first", error=True)
        sample = _stratified_sample(rows, int(args.get("sample_size") or 4))
        max_chars = int(args.get("max_chars_per_item") or 12000)
        samples: list[tuple[str, str]] = []
        for r in sample:
            t = _row_text(r, state.workdir)
            if t:
                samples.append((str(r.get("id", "?")), t[:max_chars]))
        if not samples:
            return ToolResult(output="ERROR: no items have text; populate text_path or pdf_text first",
                              error=True)
        domain_hint = args.get("domain_hint") or ""
        # Optional seed variables: fold into the domain_hint as a strong
        # nudge so the curator builds on top of them instead of starting blind.
        seed_vars = args.get("variables") or []
        if isinstance(seed_vars, list) and seed_vars:
            lines = []
            for v in seed_vars[:60]:
                if not isinstance(v, dict):
                    continue
                line = f"  - {v.get('name')} ({v.get('type','string')})"
                if v.get("description"):
                    line += f": {v['description']}"
                lines.append(line)
            if lines:
                domain_hint = (
                    (domain_hint + "\n" if domain_hint else "")
                    + "Seed variables the user explicitly requested (include "
                    "these, then add MORE to reach the target count):\n"
                    + "\n".join(lines)
                )
        min_vars = int(args.get("min_variables") or 30)

        # 1. Curator
        try:
            proposal = propose_codebook(
                self.llm, samples, domain_hint=domain_hint, min_variables=min_vars,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR (Curator): {e}", error=True)
        if "variables" not in proposal:
            return ToolResult(output=f"ERROR: Curator returned no variables. raw={json.dumps(proposal)[:400]}",
                              error=True)

        # 2. Adversary
        critique: dict = {}
        adversary_note = ""
        try:
            critique = critique_codebook(self.llm, proposal, samples)
        except Exception as e:  # noqa: BLE001
            # The curator proposal is already a valid codebook candidate. If
            # the critique step emits messy JSON, continue with the raw
            # proposal instead of returning early and leaving no codebook.
            critique = {}
            adversary_note = f"⚠ Adversary failed; saved raw curator proposal instead: {e}"

        # 3. Consolidator (deterministic, no LLM)
        consolidated = consolidate_codebook(proposal, critique if isinstance(critique, dict) else {})
        if len(consolidated.get("variables") or []) < min_vars <= len(proposal.get("variables") or []):
            adversary_note = (
                f"⚠ Adversary/consolidator reduced the codebook below the requested "
                f"minimum ({len(consolidated.get('variables') or [])}/{min_vars}); "
                "saved the curator proposal instead."
            )
            critique = {}
            consolidated = proposal

        # Save & summarize
        try:
            cb = Codebook.from_dict(consolidated)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR building codebook: {e}", error=True)
        errs = cb.validate()
        if errs:
            return ToolResult(output="ERROR: invalid codebook:\n  " + "\n  ".join(errs),
                              error=True)
        path = Path(state.workdir) / "codebook.json"
        cb.save(path)
        state.memory.set("codebook_path", str(path))

        tb = cb.type_breakdown()
        ratio = cb.numeric_or_boolean_ratio()
        def _safelen(key: str) -> int:
            if not isinstance(critique, dict):
                return 0
            v = critique.get(key) or []
            return len(v) if hasattr(v, "__len__") else 0
        n_drops    = _safelen("drop")
        n_merges   = _safelen("merge")
        n_retypes  = _safelen("retype")
        n_tightens = _safelen("tighten")
        out = [
            f"codebook saved → {path}",
            f"variables: {len(cb.variables)}    breakdown: {tb}",
            f"numeric_or_boolean_ratio: {ratio:.0%}",
            f"adversary edits: drop={n_drops} merge={n_merges} retype={n_retypes} tighten={n_tightens}",
        ]
        if adversary_note:
            out.append(adversary_note)
        out.extend(["", "first 8 variables:"])
        for v in cb.variables[:8]:
            extra = f" [{','.join(v.enum_values)}]" if v.enum_values else ""
            out.append(f"  - {v.name}: {v.type}{extra} — {v.description[:90]}")
        return ToolResult(output="\n".join(out))


# ── audit_dataset : Auditor ────────────────────────────────────────────────


class AuditDatasetTool(Tool):
    name = "audit_dataset"
    description = (
        "Sample N rows of the dataset, send (extracted_row, source_text) "
        "pairs to the AUDITOR specialist, and report per-row confidence "
        "plus common failure themes. Use this AFTER `extract_items` and "
        "BEFORE `dataset_export` so you ship calibrated quality."
    )
    args_schema = {
        "sample_size": {"type": "integer", "default": 8},
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: empty dataset", error=True)
        n = int(args.get("sample_size") or 8)
        rep = _audit(self.llm, rows, state.workdir, sample_size=n)
        # Persist for the dashboard / future analyses.
        state.memory.set("last_audit", rep.to_dict())
        return ToolResult(output=format_report(rep), artifact=rep.to_dict())


# ── dataset_report : Statistician ──────────────────────────────────────────


class DatasetReportTool(Tool):
    name = "dataset_report"
    description = (
        "Write a short plain-English findings report from the dataset + "
        "codebook + stats. Result is saved to <workdir>/findings.md and "
        "echoed back. Run at the end, after dataset_validate."
    )
    args_schema = {}

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cb = _load_codebook_from_state(state)
        if cb is None:
            return ToolResult(output="ERROR: no codebook", error=True)
        rows = state.dataset.rows()
        if not rows:
            return ToolResult(output="ERROR: empty dataset", error=True)
        stats = codebook_stats(rows, cb)
        try:
            report = write_report(self.llm, cb.to_dict(), stats)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR (Statistician): {e}", error=True)
        path = Path(state.workdir) / "findings.md"
        body = (
            f"# Findings — {cb.name}\n\n"
            f"**{report.get('headline','')}**\n\n"
            f"{report.get('narrative','')}\n\n"
            "## Key findings\n"
            + "".join(f"- {f}\n" for f in (report.get('findings') or []))
            + "\n## Caveats\n"
            + "".join(f"- {c}\n" for c in (report.get('caveats') or []))
        )
        path.write_text(body, encoding="utf-8")
        return ToolResult(output=f"wrote findings → {path}\n\n" + body[:1500],
                          artifact=report)
