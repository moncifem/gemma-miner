"""gemma42 CLI — friendly, prompt-first, with a live Rich activity feed.

Use cases:

    # one-shot, plain English — the system extracts URL + count + names a workdir
    gemma42 "scrape 1000 articles from https://arxiv.org/list/cs.LG/recent and make a stats dataset"

    # interactive REPL
    gemma42
    gemma42 chat

    # explicit run with flags (power users)
    gemma42 run --goal "..." --rows 100 --workdir ./runs/myrun

    # admin commands
    gemma42 providers
    gemma42 export-hf <dataset.jsonl> --repo-id you/your-dataset
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.box import ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from gemma42.agent import AgentConfig, run_agent
from gemma42.contracts import (
    CodebookContract,
    FieldsContract,
    MinRowsContract,
    UniqueFieldContract,
)
from gemma42.providers import list_providers, make_llm

app = typer.Typer(add_completion=False, no_args_is_help=False,
                  help="gemma42 — scrape any website into a research-grade dataset.")
console = Console()

DEFAULT_AGENT_PROVIDER = os.getenv("GEMMA42_AGENT_PROVIDER") or "openrouter"
DEFAULT_EXTRACTION_PROVIDER = os.getenv("GEMMA42_EXTRACTION_PROVIDER") or "openrouter"
DEFAULT_EXTRACTION_MODEL = (
    os.getenv("GEMMA42_EXTRACTION_MODEL") or "google/gemini-3.1-flash-lite"
)


# ── tool palette ────────────────────────────────────────────────────────────


TOOL_STYLES: dict[str, tuple[str, str]] = {
    # fetch + inspect
    "http_get":           ("🌐", "cyan"),
    "html_inspect":       ("🔍", "cyan"),
    "html_extract":       ("🧬", "cyan"),
    "extract_text":       ("📄", "cyan"),
    "read_file":          ("📖", "dim white"),
    "list_dir":           ("📁", "dim white"),
    "write_file":         ("✏ ", "dim white"),
    # declarative scrape
    "extractor_define":   ("📐", "magenta"),
    "scrape_paginated":   ("⚙ ", "bright_magenta"),
    "process_queue":      ("🚀", "bright_magenta"),
    # queue
    "queue_add":          ("➕", "blue"),
    "queue_next":         ("→ ", "blue"),
    "queue_mark_done":    ("✓", "green"),
    "queue_status":       ("📊", "blue"),
    # codebook + extract
    "codebook_propose":   ("✨", "bright_yellow"),
    "codebook_show":      ("👁 ", "yellow"),
    "codebook_edit":      ("✏ ", "yellow"),
    "codebook_test":      ("🧪", "yellow"),
    "extract_items":      ("🧬", "bright_yellow"),
    "dataset_from_queue": ("➕", "green"),
    "discover_assets":    ("🛰", "cyan"),
    "set_plan":           ("🗺", "bold cyan"),
    "show_plan":          ("🗺", "dim cyan"),
    "extract_structured": ("🧬", "yellow"),
    # dataset
    "dataset_append":     ("➕", "green"),
    "dataset_stats":      ("📊", "green"),
    "dataset_sample":     ("👁 ", "green"),
    "dataset_validate":   ("✅", "bright_green"),
    "dataset_export":     ("📦", "bright_green"),
    "hf_push":            ("☁ ", "bright_cyan"),
    # code
    "python":             ("🐍", "yellow"),
    "bash":               ("$ ", "dim white"),
    "save_attachment":    ("💾", "green"),
    # memory + contracts
    "memory_set":         ("🧠", "white"),
    "memory_get":         ("🧠", "white"),
    "memory_list":        ("🧠", "white"),
    "add_contract":       ("📜", "white"),
    "contract_status":    ("📜", "white"),
    "finish":             ("🏁", "bold bright_green"),
}

PHASE_COLORS = {
    "DISCOVER_LISTING": "cyan",
    "ENUMERATE":        "bright_blue",
    "DISCOVER_DETAIL":  "magenta",
    "PROCESS":          "bright_magenta",
    "CODEBOOK":         "bright_yellow",
    "EXTRACT":          "yellow",
    "EXPORT":           "bright_green",
    "FINISH":           "bold bright_green",
}


def _icon(name: str | None) -> tuple[str, str]:
    return TOOL_STYLES.get(name or "", ("•", "white"))


def _short(s: Any, n: int = 200) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_args_inline(args: Any) -> str:
    if isinstance(args, dict):
        parts: list[str] = []
        for k, v in args.items():
            sv = json.dumps(v, ensure_ascii=False, default=str)
            if len(sv) > 80:
                parts.append(f"{k}=<{type(v).__name__}:{len(sv)}b>")
            else:
                parts.append(f"{k}={sv}")
        return _short(", ".join(parts), 240)
    try:
        return _short(json.dumps(args, ensure_ascii=False, default=str), 240)
    except Exception:  # noqa: BLE001
        return _short(args, 240)


def _print_banner(
    provider: str,
    model: str,
    workdir: Path,
    *,
    extraction_provider: str | None = None,
    extraction_model: str | None = None,
) -> None:
    extraction_line = ""
    if extraction_provider and extraction_model:
        extraction_line = f"\n[dim]extract: {extraction_provider}/{extraction_model}[/dim]"
    text = (
        f"[bold cyan]gemma42[/bold cyan]  ·  text-to-dataset agent\n"
        f"[dim]agent:  {provider}/{model}[/dim]"
        f"{extraction_line}\n"
        f"[dim]workdir: {workdir}[/dim]"
    )
    console.print(Panel.fit(text, border_style="cyan", box=ROUNDED))


# ── prompt parsing ──────────────────────────────────────────────────────────


# The only fallback heuristic: the first 2–4 digit number in the prompt.
# We don't try to be clever about it — the real intent parsing is done by
# `_plan_with_llm` below.
_FALLBACK_COUNT_RE = re.compile(r"\b(\d{2,4})\b")


def _extract_url(prompt: str) -> str | None:
    m = re.search(r"https?://[^\s)>\]\"']+", prompt)
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?")
    return url


def _extract_count(prompt: str) -> int | None:
    """Cheap heuristic fallback. The real planning is `_plan_with_llm`."""
    m = _FALLBACK_COUNT_RE.search(prompt)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _plan_with_llm(prompt: str, llm: Any) -> dict:
    """One LLM call: turn a free-text user prompt into structured intent.

    Returns: {count: int | null, target_fields: [str], source_url: str | null,
              wants_codebook: bool, notes: str}

    This is *not* hardcoded keyword matching. The LLM reads the goal and
    decides — same way a junior data engineer would.
    """
    system = (
        "You translate a user's plain-English scraping request into a "
        "structured plan. The system has TWO layers:\n"
        "  BRONZE (raw harvest)  — fields scraped directly from the page,\n"
        "                          one observation per item (date, title, raw\n"
        "                          decision text, organisation name, URL, …).\n"
        "  SILVER (typed extract) — codebook variables a downstream model fills\n"
        "                          from the raw text (booleans, enums, fine\n"
        "                          amounts in EUR, derived categories, …).\n"
        "Keep the two layers distinct in your plan.\n\n"
        "Return ONE JSON object with these fields:\n"
        '  count          : integer (how many records the user asked for, or null)\n'
        '  target_fields  : SNAKE_CASE list of fields the BRONZE row must carry\n'
        '                    (i.e. directly visible/scrapable evidence). DO NOT\n'
        '                    include here variables that obviously need to be\n'
        '                    parsed out of free text (those belong in the codebook\n'
        '                    and live in SILVER). Naming: "n_comments" not\n'
        '                    "number of comments", "published_date" not "publication\n'
        '                    date". Only include fields that can be populated for\n'
        '                    EVERY row. For optional facts phrased as "any X" or\n'
        '                    "whether X", prefer a boolean `has_x`/`is_x` and put\n'
        '                    it in the codebook seed, not in target_fields.\n'
        '  source_url     : a URL if the user gave one, else null\n'
        '  source_hint    : short text describing the source if no URL\n'
        '  wants_codebook : boolean — DEFAULT TRUE. The goal of this system is to\n'
        '                    turn webpages into stats/ML-ready datasets of TYPED\n'
        '                    variables, not raw text. Set false ONLY when the\n'
        '                    user explicitly asks for raw text/links with no\n'
        '                    derived analysis.\n'
        '  unique_field   : pick a field ONLY IF you can name an explicit primary\n'
        '                    key in the source (an ID-shaped string: "doi",\n'
        '                    "ticker", "post_id", "isbn", "ark_id", "arxiv_id").\n'
        '                    DO NOT pick fields that can repeat across rows (dates,\n'
        '                    names, titles, categories). When in doubt: null. The\n'
        '                    system auto-synthesises a content-hash id when null.\n'
        '  notes          : one short sentence with anything else important.\n\n'
        "Quick discrimination test for each candidate field:\n"
        "  • If a human could COPY-PASTE the value off the listing page → BRONZE\n"
        "    (put in target_fields).\n"
        "  • If the value requires reading the decision/abstract/body and INFERRING\n"
        "    a class, count, flag, or amount → SILVER (do NOT put in target_fields;\n"
        "    let the codebook design phase pick it up).\n"
        "Output JSON only, no fences."
    )
    raw = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    from gemma42.swarm.base import parse_json_obj

    obj = parse_json_obj(raw) or {}
    # Normalize field names to snake_case (last-line defence if the model slips).
    raw_fields = obj.get("target_fields") or []
    norm_fields: list[str] = []
    prompt_l = prompt.lower()
    for f in raw_fields:
        if not isinstance(f, str):
            continue
        s = f.strip().lower()
        # "number of comments" → "n_comments"
        s = re.sub(r"\bnumber\s+of\s+", "n_", s)
        # spaces/hyphens/punctuation → underscores
        s = re.sub(r"[^\w]+", "_", s).strip("_")
        # Generic optional-fact rewrite: if the user said "any <field>" or
        # "whether <field>", make the required field an explicit boolean
        # instead of a nullable string. This is domain-agnostic.
        phrase = re.escape(s.replace("_", " "))
        optional_re = rf"\b(?:any|whether|if)\s+(?:an?\s+|the\s+)?{phrase}s?\b"
        if (
            s
            and not s.startswith(("has_", "is_", "n_", "amount_", "pct_", "cat_", "dn_"))
            and re.search(optional_re, prompt_l)
        ):
            s = f"has_{s}"
        if s and s not in norm_fields:
            norm_fields.append(s)
    return {
        "count":          obj.get("count"),
        "target_fields":  norm_fields,
        "source_url":     obj.get("source_url"),
        "source_hint":    obj.get("source_hint"),
        "wants_codebook": bool(obj.get("wants_codebook", True)),
        "unique_field":   obj.get("unique_field"),
        "notes":          obj.get("notes") or "",
    }


def _slugify_for_workdir(prompt: str, url: str | None) -> str:
    if url:
        host = urlparse(url).netloc.replace("www.", "").replace(".", "_")
        return host or "run"
    words = re.sub(r"[^\w\s-]", "", prompt.lower()).split()
    slug = "_".join(words[:5])[:40].strip("_")
    return slug or "run"


def _auto_workdir(prompt: str, url: str | None, base: Path) -> Path:
    slug = _slugify_for_workdir(prompt, url)
    target = base / slug
    if not target.exists():
        return target
    # find next available numeric suffix
    for i in range(2, 100):
        cand = base / f"{slug}_{i}"
        if not cand.exists():
            return cand
    return target


# ── live activity feed ──────────────────────────────────────────────────────


class ActivityFeed:
    """Renders the live state of an agent run as a Rich Group."""

    def __init__(self) -> None:
        self.steps: list[Any] = []
        self.step_count = 0
        self.start = time.time()
        self.spinner = Spinner("dots", text=Text("starting…", style="cyan"))
        self.spinner_label = "starting…"
        self.spinner_started = time.time()
        self.spinner_active = True
        # current phase + dataset progress (rendered in the status bar)
        self.phase = ""
        self.rows = 0
        self.extracted_rows = 0
        self.contracts: list[dict] = []
        # in-flight tool calls: index → start_ts
        self._inflight: list[float] = []
        self.live: Any = None
        self._graduated: set[int] = set()

    # ── feed input ─────────────────────────────────────────────────────────

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind == "phase":
            phase = ev.get("phase", "")
            if phase and phase != self.phase:
                self.phase = phase
                self._on_phase_change(phase)
            else:
                self.phase = phase
        elif kind == "llm_start":
            model = ev.get("model", "?")
            self.set_thinking(f"thinking with [bold]{model}[/bold]")
        elif kind == "turn":
            self._on_turn(ev)
        elif kind == "reflection":
            self._on_reflection(ev)
        elif kind == "extract_start":
            self._on_extract_start(ev)
        elif kind == "extract_item_start":
            self._on_extract_item_start(ev)
        elif kind == "extract_item_done":
            self._on_extract_item_done(ev)
        elif kind == "extract_item_failed":
            self._on_extract_item_failed(ev)
        elif kind == "extract_done":
            self._on_extract_done(ev)
        elif kind == "run_end":
            self.spinner_active = False
        elif kind == "aborted":
            self.spinner_active = False

    # ── phase transition headers ──────────────────────────────────────────
    _PHASE_LABELS = {
        "DISCOVER_LISTING": ("🧭", "exploring the site"),
        "ENUMERATE":        ("📚", "gathering the items"),
        "DISCOVER_DETAIL":  ("🔬", "studying one item closely"),
        "PROCESS":          ("⚗ ", "harvesting items + attachments"),
        "CODEBOOK":         ("📓", "designing the codebook"),
        "EXTRACT":          ("🧬", "extracting variables per item"),
        "EXPORT":           ("📦", "writing the final dataset"),
        "FINISH":           ("🏁", "wrapping up"),
    }

    def _on_phase_change(self, phase: str) -> None:
        icon, label = self._PHASE_LABELS.get(phase, ("·", phase.lower()))
        color = PHASE_COLORS.get(phase, "white")
        text = Text.from_markup(
            f"\n[{color}]{icon}  {phase}[/{color}]  [dim]· {label}[/dim]"
        )
        self.steps.append(text)
        self._graduate(len(self.steps) - 1)

    # ── extraction-progress handlers (Gemma-on-each-item loop) ────────────
    _ext_total: int = 0
    _ext_model: str = ""
    _ext_done: int = 0
    _ext_filled_avg: float = 0.0
    _ext_failed: int = 0

    def _on_extract_start(self, ev: dict[str, Any]) -> None:
        self._ext_total = int(ev.get("total") or 0)
        self._ext_model = str(ev.get("model") or "extractor")
        self._ext_done = 0
        self._ext_filled_avg = 0.0
        self._ext_failed = 0
        n_vars = int(ev.get("n_variables") or 0)
        text = Text.from_markup(
            f"  [bold bright_yellow]🧬 extracting variables[/bold bright_yellow]  "
            f"[dim]· {self._ext_total} item(s) × {n_vars} variable(s)  "
            f"· model: {self._ext_model}[/dim]"
        )
        self.steps.append(text)
        self._graduate(len(self.steps) - 1)

    def _on_extract_item_start(self, ev: dict[str, Any]) -> None:
        i, total = ev.get("index", 0), ev.get("total", self._ext_total)
        # Replace the spinner label with rolling per-item status.
        avg_pct = (self._ext_filled_avg * 100) if self._ext_done else 0
        self.set_thinking(
            f"extracting item [bold]{i}/{total}[/bold] with "
            f"[bold]{self._ext_model}[/bold]  · avg fill {avg_pct:.0f}%"
        )

    def _on_extract_item_done(self, ev: dict[str, Any]) -> None:
        self._ext_done += 1
        filled = int(ev.get("filled") or 0)
        n_vars = max(1, int(ev.get("n_variables") or 1))
        # Running average of fill ratio.
        ratio = filled / n_vars
        # Online mean update.
        self._ext_filled_avg = (
            (self._ext_filled_avg * (self._ext_done - 1)) + ratio
        ) / self._ext_done
        # Compact per-item line every 10 items (or at the end).
        if self._ext_done % 10 == 0 or self._ext_done == self._ext_total:
            pct = self._ext_done * 100 // max(1, self._ext_total)
            text = Text.from_markup(
                f"     [dim]→ {self._ext_done}/{self._ext_total} items "
                f"({pct}%)  · avg fill {self._ext_filled_avg*100:.0f}%[/dim]"
            )
            self.steps.append(text)
            self._graduate(len(self.steps) - 1)

    def _on_extract_item_failed(self, ev: dict[str, Any]) -> None:
        # Count silently — don't render errors in the live feed (the user
        # asked us not to display them; they're still in failures.log).
        self._ext_failed += 1

    def _on_extract_done(self, ev: dict[str, Any]) -> None:
        extracted = int(ev.get("extracted") or 0)
        self.extracted_rows = max(self.extracted_rows, extracted)
        text = Text.from_markup(
            f"  [bold green]✓ extraction complete[/bold green]  "
            f"[dim]· {extracted}/{self._ext_total} items processed  "
            f"· avg fill {self._ext_filled_avg*100:.0f}%[/dim]"
        )
        self.steps.append(text)
        self._graduate(len(self.steps) - 1)

    def _on_reflection(self, ev: dict[str, Any]) -> None:
        added = ev.get("added") or []
        if not added:
            return
        bullets = "\n".join(f"  • {l}" for l in added)
        text = Text.from_markup(
            f"[bold yellow]📓 lesson{'s' if len(added) != 1 else ''} learned[/bold yellow]\n{bullets}",
        )
        self.steps.append(text)
        self._graduate(len(self.steps) - 1)

    def _on_turn(self, ev: dict[str, Any]) -> None:
        turn = ev.get("turn", 0)
        tool = ev.get("tool", "?")
        args = ev.get("args") or {}
        observation = ev.get("observation", "")
        is_error = bool(ev.get("error"))
        elapsed_ms = ev.get("elapsed_ms") or 0
        contracts = ev.get("contracts") or []
        n_rows = ev.get("n_rows", 0)

        self.step_count = max(self.step_count, turn)
        self.rows = n_rows
        self.contracts = contracts

        call_line = self._tool_call_line(turn, tool, args)
        result_line = self._result_renderable(tool, observation, is_error, elapsed_ms)
        block = Group(call_line, result_line)
        # Add and immediately graduate completed blocks to keep live region small.
        idx = len(self.steps)
        self.steps.append(block)
        self._graduate(idx)
        self.set_thinking("thinking…")

    def set_thinking(self, label: str) -> None:
        self.spinner_label = label
        self.spinner_started = time.time()
        self.spinner_active = True

    def stop_spinner(self) -> None:
        self.spinner_active = False

    # ── rendering ──────────────────────────────────────────────────────────

    def _live_spinner(self) -> Spinner:
        elapsed = time.time() - self.spinner_started
        try:
            label_part = Text.from_markup(self.spinner_label, style="cyan")
        except Exception:  # noqa: BLE001
            label_part = Text(self.spinner_label, style="cyan")
        self.spinner.text = Text.assemble(label_part, (f"  ({elapsed:.1f}s)", "dim"))
        return self.spinner

    def _status_bar(self) -> Text:
        elapsed = int(time.time() - self.start)
        m, s = divmod(elapsed, 60)
        phase_color = PHASE_COLORS.get(self.phase, "white")
        phase_part = f"[{phase_color}]{self.phase or '—'}[/{phase_color}]"
        # Show contract progress as a count, not as a red failure label.
        c_part = ""
        if self.contracts:
            ok_count = sum(1 for c in self.contracts if c.get("ok"))
            total = len(self.contracts)
            color = "green" if ok_count == total else "yellow"
            c_part = (
                f"  [dim]·[/dim]  "
                f"contracts [bold {color}]{ok_count}/{total}[/bold {color}]"
            )
        rows_part = f"raw {self.rows}"
        if self.extracted_rows:
            rows_part += f"  [dim]·[/dim]  extracted {self.extracted_rows}"
        return Text.from_markup(
            f"  phase: {phase_part}  [dim]·[/dim]  "
            f"step {self.step_count}  [dim]·[/dim]  "
            f"{rows_part}  [dim]·[/dim]  "
            f"{m}m{s:02d}s" + c_part
        )

    def render(self) -> Group:
        elements: list[Any] = []
        # only non-graduated (visible in live region)
        for st in self.steps:
            if st is not None:
                elements.append(st)
        if self.spinner_active:
            elements.append(self._live_spinner())
        elements.append(self._status_bar())
        return Group(*elements)

    def _graduate(self, idx: int) -> None:
        if idx in self._graduated or self.live is None:
            return
        renderable = self.steps[idx]
        if renderable is None:
            return
        try:
            self.live.console.print(renderable)
        except Exception:  # noqa: BLE001
            return
        self._graduated.add(idx)
        self.steps[idx] = None

    # ── helpers ────────────────────────────────────────────────────────────

    def _tool_call_line(self, step: int, name: str, args: Any) -> Text:
        icon, color = _icon(name)
        return Text.assemble(
            (f"  {step:>3} ", "dim"),
            (f"{icon} ", color),
            (f"{name}", f"bold {color}"),
            ("  ", ""),
            (_fmt_args_inline(args), "dim"),
        )

    def _result_renderable(self, name: str, content: str, is_error: bool, elapsed_ms: float) -> Any:
        # Errors are tracked silently in failures.log; the live feed only
        # shows a muted "retrying…" hint so the user isn't drowning in red.
        ms = f"  · {elapsed_ms / 1000.0:.1f}s" if elapsed_ms else ""
        if is_error:
            return Text.assemble(
                ("    ↻ ", "dim yellow"),
                (f"{name} retrying with a different approach", "dim"),
                (ms, "dim"),
            )
        summary = self._inline_summary(name, content)
        if len(summary) > 180 or "\n" in summary:
            return Panel(
                Text(summary[:1200], style="white", overflow="fold"),
                border_style="dim",
                box=SIMPLE,
                padding=(0, 1),
            )
        return Text.assemble(
            ("    ↳ ", "dim"),
            (summary, "white"),
            (ms, "dim"),
        )

    def _inline_summary(self, name: str, content: str) -> str:
        """Extract a useful one-line summary from a tool's raw output."""
        if not content:
            return "ok"
        text = content.strip()

        # Many tools use a `key: value` header; capture the most useful one.
        priorities = {
            "http_get":          ["status", "bytes"],
            "html_inspect":      ["html_size"],
            "html_extract":      ["total_matches"],
            "extractor_define":  ["matched_rows", "saved"],
            "scrape_paginated":  ["queue_len", "total_added"],
            "process_queue":     ["appended", "remaining_in_queue"],
            "dataset_append":    ["added", "total_rows_now"],
            "dataset_stats":     ["rows"],
            "extract_items":     ["processed"],
            "extract_text":      ["text_chars", "n_pages"],
            "save_attachment":   ["attachment_path", "text_chars"],
            "codebook_propose":  ["variables", "type breakdown"],
            "codebook_test":     ["dataset", "numeric_or_boolean_ratio"],
            "dataset_export":    ["export →", "parquet"],
            "queue_status":      ["queue_len", "remaining"],
            "queue_add":         ["queue_add"],
            "queue_next":        [""],
        }
        wants = priorities.get(name) or []
        parts: list[str] = []
        for line in text.splitlines()[:12]:
            line = line.strip()
            if not line:
                continue
            for w in wants:
                if w and line.startswith(w):
                    parts.append(line)
                    break
        if parts:
            return "  ·  ".join(parts)[:240]
        # Generic fallback: first non-empty line.
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        return _short(first, 200)


# ── core run ────────────────────────────────────────────────────────────────


def _setup_logging(level: int = logging.WARNING) -> None:
    # Quiet the underlying httpx + asyncio noise; the live feed shows progress.
    logging.basicConfig(level=level, format="%(message)s", handlers=[logging.NullHandler()])
    for name in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _run_with_live_feed(
    *,
    goal: str,
    workdir: Path,
    min_rows: int,
    required_fields: list[str],
    unique_field: str | None,
    provider: str,
    model: str | None,
    extraction_provider: str,
    extraction_model: str | None,
    max_turns: int,
    want_codebook: bool,
) -> Any:
    """Run the agent with a live Rich feed. Returns the RunResult."""
    _setup_logging()
    workdir.mkdir(parents=True, exist_ok=True)

    llm = make_llm(provider, model=model)
    extraction_llm = make_llm(extraction_provider, model=extraction_model)
    _print_banner(
        provider,
        llm.config.model,
        workdir,
        extraction_provider=extraction_provider,
        extraction_model=extraction_llm.config.model,
    )

    contracts: list = []
    if min_rows > 0:
        contracts.append(MinRowsContract(min_rows=min_rows))
    if required_fields:
        contracts.append(FieldsContract(required_fields=required_fields))
    if unique_field:
        contracts.append(UniqueFieldContract(field=unique_field))
    if want_codebook:
        contracts.append(CodebookContract(min_variables=20, min_numeric_or_boolean_ratio=0.5))

    # show what we decided
    decided = Table.grid(padding=(0, 2))
    decided.add_column(style="cyan"); decided.add_column()
    decided.add_row("goal", _short(goal, 220))
    decided.add_row("min rows", str(min_rows) if min_rows else "—")
    decided.add_row("workdir", str(workdir))
    decided.add_row("agent model", f"{provider}/{llm.config.model}")
    decided.add_row("extract model", f"{extraction_provider}/{extraction_llm.config.model}")
    decided.add_row("codebook", "yes (auto-design 20–60 typed vars)" if want_codebook else "no")
    console.print(Panel(decided, title="run plan", border_style="cyan", box=SIMPLE))
    console.print(Rule(style="dim cyan"))

    feed = ActivityFeed()
    cfg = AgentConfig(
        max_turns=max_turns,
        verbose=False,
        event_subscribers=[feed.on_event],
    )

    feed.set_thinking("contacting model…")
    with Live(get_renderable=feed.render, console=console, refresh_per_second=4,
              vertical_overflow="visible", transient=False) as live:
        feed.live = live
        try:
            result = run_agent(
                goal=goal,
                contracts=contracts,
                workdir=workdir,
                llm=llm,
                extraction_llm=extraction_llm,
                unique_key=unique_field or None,
                config=cfg,
                extra_memory={"wants_codebook": bool(want_codebook)},
            )
        finally:
            feed.stop_spinner()
            live.update(feed.render())

    console.print(Rule(style="dim cyan"))
    _print_summary(result)
    return result


def _print_summary(result: Any) -> None:
    color = "green" if result.finished else "yellow"
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan"); t.add_column()
    t.add_row("finished", str(result.finished))
    t.add_row("reason", result.finish_reason)
    t.add_row("turns", str(result.turns))
    t.add_row("rows", str(result.n_rows))
    t.add_row("dataset", result.dataset_path)
    t.add_row("trace", result.trace_log)
    if hasattr(result, "trace_jsonl"):
        t.add_row("trace.jsonl", result.trace_jsonl)
    # Show failure-log paths if anything failed.
    workdir = Path(result.dataset_path).parent
    fail_log = workdir / "failures.log"
    if fail_log.exists() and fail_log.stat().st_size > 0:
        t.add_row("failures.log", str(fail_log))
    console.print(Panel(t, title="run result", border_style=color, box=ROUNDED))
    if result.contracts:
        ct = Table(title="contracts", header_style="bold cyan", box=SIMPLE)
        ct.add_column("name"); ct.add_column("ok"); ct.add_column("detail")
        for c in result.contracts:
            mark = "[green]OK[/green]" if c["ok"] else "[red]FAIL[/red]"
            ct.add_row(c["name"], mark, c["detail"])
        console.print(ct)
    # Surface the export folder if the EXPORT phase wrote one.
    workdir = Path(result.dataset_path).parent
    export_dir = workdir / "export"
    if export_dir.exists():
        files = sorted(export_dir.iterdir())
        if files:
            elines = "\n".join(
                f"  [bold]{f.name}[/bold]  [dim]({f.stat().st_size:,} bytes)[/dim]"
                for f in files if f.is_file()
            )
            console.print(Panel(elines,
                title=f"export → {export_dir}",
                title_align="left",
                border_style="bright_green",
                box=ROUNDED,
            ))
            parquet = next((f for f in files if f.suffix == ".parquet"), None)
            dataset_jsonl = Path(result.dataset_path)
            load_path = parquet or (export_dir / "<name>.parquet")
            console.print(Text.from_markup(
                f"  [dim]load it:[/dim] [white]pd.read_parquet('{load_path}')[/white]  "
                f"[dim]· push:[/dim] [white]gemma42 export-hf {dataset_jsonl} --repo-id <repo-id>[/white]"
            ))


# ── orchestrator: free-text prompt → run ────────────────────────────────────


def _orchestrate(
    prompt: str,
    *,
    rows: int | None,
    workdir: Path | None,
    provider: str,
    model: str | None,
    extraction_provider: str,
    extraction_model: str | None,
    max_turns: int,
    no_codebook: bool,
    push: str | None,
    public: bool,
) -> Any:
    """One-call entry point: parse the prompt with the LLM, run."""
    prompt = prompt.strip()
    url = _extract_url(prompt)
    fallback_count = _extract_count(prompt)

    # Ask the LLM to parse the prompt — no hardcoded keyword lists.
    llm_for_plan = make_llm(provider, model=model)
    plan: dict = {}
    try:
        plan = _plan_with_llm(prompt, llm_for_plan)
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]planner failed: {e}; falling back to heuristics[/dim]")
    if plan.get("source_url") and not url:
        url = plan["source_url"]

    target_rows = (
        rows
        if rows is not None
        else (plan.get("count") if isinstance(plan.get("count"), int)
              else (fallback_count or 30))
    )
    target_fields = plan.get("target_fields") or []
    unique_field = plan.get("unique_field")
    wants_codebook = (not no_codebook) and plan.get("wants_codebook", True)

    if workdir is None:
        workdir = _auto_workdir(prompt, url, Path("./runs"))

    # Tell the user what we understood — visibility, not magic.
    plan_text = (
        f"  count target:   {target_rows}\n"
        f"  fields:         {target_fields or '(any)'}\n"
        f"  source URL:     {url or plan.get('source_hint') or '(agent picks)'}\n"
        f"  codebook phase: {'yes' if wants_codebook else 'no'}\n"
        + (f"  notes:          {plan.get('notes')}\n" if plan.get('notes') else "")
    )
    console.print(Panel(plan_text, title="what I understood", border_style="cyan", box=SIMPLE))

    # Build the agent's goal: the user's prompt + the planner's structured intent.
    goal_lines = [prompt]
    if target_fields:
        goal_lines.append(
            f"\nTarget output fields: {', '.join(target_fields)}."
        )
    if wants_codebook:
        goal_lines.append(
            "\nAfter the harvest phase, automatically design a codebook of "
            "20–60 typed variables (booleans, integers, floats, enums, "
            "dates) suitable for statistical analysis, extract those "
            "variables for every item, validate, and export to "
            f"{workdir}/export/ as parquet + jsonl + codebook.md."
        )
    if push:
        goal_lines.append(
            f"\nFinally push the dataset to Hugging Face repo '{push}' "
            f"({'public' if public else 'private'})."
        )

    result = _run_with_live_feed(
        goal="\n".join(goal_lines),
        workdir=workdir,
        min_rows=target_rows,
        required_fields=target_fields,
        unique_field=unique_field,
        provider=provider,
        model=model,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        max_turns=max_turns,
        want_codebook=wants_codebook,
    )
    return result


# ── Typer commands ──────────────────────────────────────────────────────────


@app.command("ask")
def ask_cmd(
    prompt: str = typer.Argument(
        ...,
        help='Plain-English request. e.g. "scrape 1000 articles from arxiv and make me a stats dataset"',
    ),
    rows: int = typer.Option(
        None, "--rows", "-n",
        help="Target row count. Overrides any count parsed from the prompt. Default 50 if neither.",
    ),
    workdir: Path = typer.Option(
        None, "--workdir", "-w",
        help="Working directory (auto-named under ./runs/ if omitted).",
    ),
    provider: str = typer.Option(
        None, "--provider",
        help=(f"LLM provider: {', '.join(list_providers())}. "
              "Default: openrouter for agentic discovery/recon."),
    ),
    model: str = typer.Option(None, "--model", "-m"),
    extraction_provider: str = typer.Option(
        None, "--extract-provider",
        help="Provider used only for schema-constrained extraction. Default: ollama.",
    ),
    extraction_model: str = typer.Option(
        None, "--extract-model",
        help="Model used only for schema-constrained extraction. Default: gemma4:latest.",
    ),
    max_turns: int = typer.Option(120, "--max-turns"),
    no_codebook: bool = typer.Option(
        False, "--no-codebook",
        help="Skip the auto codebook + extract + export phases. Stop after harvest.",
    ),
    push: str = typer.Option(
        None, "--push",
        help="Hugging Face repo id to push the final dataset to (e.g. you/my-dataset).",
    ),
    public: bool = typer.Option(
        False, "--public",
        help="Make the pushed HF dataset public (default private).",
    ),
):
    """One-shot prompt-first run. The agent extracts URL+count from the prompt."""
    provider = provider or DEFAULT_AGENT_PROVIDER
    extraction_provider = extraction_provider or DEFAULT_EXTRACTION_PROVIDER
    extraction_model = extraction_model or DEFAULT_EXTRACTION_MODEL
    _orchestrate(
        prompt, rows=rows, workdir=workdir, provider=provider, model=model,
        extraction_provider=extraction_provider, extraction_model=extraction_model,
        max_turns=max_turns, no_codebook=no_codebook, push=push, public=public,
    )


# ── Claude-Code-style REPL ──────────────────────────────────────────────────


_CHAT_SYSTEM = """You are gemma42, an interactive agent that turns any website into a research-grade structured dataset.

You operate in TWO modes:

  1. CHAT — the user wants to talk: greetings, questions about how the system
     works, advice on what to scrape. Reply in plain Markdown, friendly,
     concise (2-6 sentences). NO tool calls.

  2. TASK — the user wants you to scrape something or build a dataset.
     Acknowledge briefly (one short sentence), then the orchestrator will
     spawn a full agent run; your conversational reply is just an
     acknowledgement, not the work itself.

The orchestrator decides per-turn which mode applies based on whether the
message contains a URL or scraping verbs. You ONLY produce the natural
language reply — never JSON tool calls in this mode.

Capabilities you can describe when asked:
  - Scrape any website (HTML tables, Drupal, WordPress, JSON APIs, RSS).
  - Auto-discover the listing structure and follow detail pages.
  - Download PDFs/XML/CSV attachments and extract their text.
  - Design a 20–60 typed-variable codebook from sample documents.
  - Apply that codebook to every item via structured LLM extraction.
  - Export to Parquet + JSONL + a dataset card; push to Hugging Face.

When the user gives a scrape request, briefly summarise what you'll do (1-2
sentences) so they know what's about to happen, then the live feed shows
the actual work.
"""


def _classify_intent(message: str) -> str:
    """Heuristic: TASK if message contains a URL or scraping verbs; else CHAT."""
    if _extract_url(message):
        return "TASK"
    verbs = re.compile(
        r"\b(scrape|extract|crawl|fetch|harvest|collect|gather|"
        r"get\s+me|give\s+me|build\s+(?:me\s+)?a?\s*dataset|"
        r"download\s+\d+)\b",
        re.IGNORECASE,
    )
    if verbs.search(message):
        return "TASK"
    return "CHAT"


def _chat_completion(llm: Any, history: list[dict], user_msg: str) -> str:
    """One free-form LLM reply, no tool calls. Returns the assistant's text."""
    messages = [{"role": "system", "content": _CHAT_SYSTEM}]
    # keep history small for small models
    for m in history[-8:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})
    return llm.chat(messages, temperature=0.4)


def _list_runs(base: Path) -> list[dict]:
    rows: list[dict] = []
    if not base.exists():
        return rows
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        ds = d / "dataset.jsonl"
        if not ds.exists():
            continue
        try:
            n = sum(1 for _ in ds.open("r", encoding="utf-8"))
        except Exception:  # noqa: BLE001
            n = 0
        cb = d / "codebook.json"
        export_dir = d / "export"
        has_pq = any(export_dir.glob("*.parquet")) if export_dir.exists() else False
        rows.append({
            "name": d.name,
            "rows": n,
            "codebook": "yes" if cb.exists() else "no",
            "parquet": "yes" if has_pq else "no",
            "path": str(d),
        })
    return rows


def _show_help() -> None:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("[cyan]/help[/cyan]", "show this help")
    t.add_row("[cyan]/datasets[/cyan]", "list datasets produced in this workspace")
    t.add_row("[cyan]/workdir[/cyan] [<path>]", "show or change the base workdir (./runs)")
    t.add_row("[cyan]/provider[/cyan] [<name>]", "show or switch LLM provider (together/ollama/...)")
    t.add_row("[cyan]/model[/cyan] [<id>]", "show or switch model")
    t.add_row("[cyan]/history[/cyan]", "show conversation history")
    t.add_row("[cyan]/clear[/cyan]", "clear the screen and history")
    t.add_row("[cyan]/trace[/cyan]", "open the last run's trace.log path")
    t.add_row("[cyan]/quit[/cyan], [cyan]/exit[/cyan]", "leave the REPL")
    console.print(Panel(t, title="commands", title_align="left", border_style="dim"))


class REPL:
    """Persistent conversational shell — chats for small-talk, runs the
    full agent pipeline for scrape tasks. Keeps conversation history so the
    model can refer back."""

    def __init__(self, provider: str, model: str | None) -> None:
        self.provider = provider
        self.llm = make_llm(provider, model=model)
        self.model = self.llm.config.model
        self.base_workdir = Path("./runs")
        self.history: list[dict] = []
        self.last_workdir: Path | None = None

    def show_banner(self) -> None:
        console.print(Panel.fit(
            f"[bold cyan]gemma42[/bold cyan]  ·  text-to-dataset agent\n"
            f"[dim]model:  {self.provider}/{self.model}[/dim]\n"
            f"[dim]workdir: {self.base_workdir}[/dim]\n"
            f"[dim]type [white]/help[/white] for commands, or just describe what you want.[/dim]",
            border_style="cyan",
            box=ROUNDED,
        ))

    def show_status_line(self) -> None:
        info = (
            f"[dim]{self.provider}/{self.model}  ·  "
            f"workdir {self.base_workdir}  ·  "
            f"history {len(self.history)} turns[/dim]"
        )
        console.print(info)

    # ── slash commands ─────────────────────────────────────────────────────

    def handle_slash(self, line: str) -> bool:
        """Return True if it was a slash command (and we handled it)."""
        if not line.startswith("/"):
            return False
        parts = line[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("help", "?"):
            _show_help()
        elif cmd == "datasets":
            rows = _list_runs(self.base_workdir)
            if not rows:
                console.print(f"[dim]no datasets in {self.base_workdir}[/dim]")
            else:
                t = Table(title="datasets", header_style="bold cyan", box=ROUNDED)
                t.add_column("name"); t.add_column("rows", justify="right")
                t.add_column("codebook"); t.add_column("parquet"); t.add_column("path")
                for r in rows:
                    t.add_row(r["name"], str(r["rows"]), r["codebook"], r["parquet"], r["path"])
                console.print(t)
        elif cmd == "workdir":
            if arg:
                self.base_workdir = Path(arg).expanduser().resolve()
                console.print(f"[green]workdir → {self.base_workdir}[/green]")
            else:
                console.print(f"[cyan]workdir:[/cyan] {self.base_workdir}")
        elif cmd == "provider":
            if arg:
                try:
                    self.llm = make_llm(arg)
                    self.provider = arg
                    self.model = self.llm.config.model
                    console.print(f"[green]switched to {self.provider}/{self.model}[/green]")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]could not switch: {e}[/red]")
            else:
                console.print(f"[cyan]provider:[/cyan] {self.provider}  [dim](try /provider ollama, /provider together)[/dim]")
        elif cmd == "model":
            if arg:
                try:
                    self.llm = make_llm(self.provider, model=arg)
                    self.model = arg
                    console.print(f"[green]model → {arg}[/green]")
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]could not switch: {e}[/red]")
            else:
                console.print(f"[cyan]model:[/cyan] {self.model}")
        elif cmd == "history":
            if not self.history:
                console.print("[dim]no history yet[/dim]")
            else:
                for m in self.history[-10:]:
                    role = m["role"]
                    style = "bold green" if role == "user" else "bold cyan"
                    glyph = "you" if role == "user" else "gemma42"
                    console.print(Text.assemble((f"  {glyph}: ", style),
                                                 (_short(m["content"], 200), "white")))
        elif cmd == "clear":
            console.clear()
            self.history.clear()
            self.show_banner()
        elif cmd == "trace":
            if self.last_workdir is None:
                console.print("[dim]no run yet[/dim]")
            else:
                t = self.last_workdir / "trace.log"
                console.print(f"  [cyan]tail -F {t}[/cyan]")
        elif cmd in ("quit", "exit", "q"):
            raise EOFError()
        else:
            console.print(f"[red]unknown command: /{cmd}[/red]  (try /help)")
        return True

    # ── one turn ───────────────────────────────────────────────────────────

    def handle_turn(self, message: str) -> None:
        intent = _classify_intent(message)
        self.history.append({"role": "user", "content": message})

        if intent == "CHAT":
            self._render_chat_reply(message)
            return

        # TASK — acknowledge briefly + run the agent live.
        try:
            ack = _chat_completion(self.llm, self.history, message)
        except Exception:  # noqa: BLE001
            ack = "Got it — starting the run."
        ack = (ack or "").strip()
        if ack:
            console.print(Panel(
                Markdown(ack),
                border_style="cyan",
                box=ROUNDED,
                title="gemma42",
                title_align="left",
            ))
            self.history.append({"role": "assistant", "content": ack})

        result = _orchestrate(
            message,
            rows=None,
            workdir=None if self.last_workdir is None else None,  # auto-name each run
            provider=self.provider,
            model=self.model,
            extraction_provider=DEFAULT_EXTRACTION_PROVIDER,
            extraction_model=DEFAULT_EXTRACTION_MODEL,
            max_turns=120,
            no_codebook=False,
            push=None,
            public=False,
        )
        try:
            self.last_workdir = Path(result.dataset_path).parent
        except Exception:  # noqa: BLE001
            pass
        summary = (
            f"Done — {result.n_rows} rows in {result.turns} turns. "
            f"Files in `{Path(result.dataset_path).parent}/export/`."
        )
        self.history.append({"role": "assistant", "content": summary})

    def _render_chat_reply(self, message: str) -> None:
        spinner = Spinner("dots", text=Text("thinking…", style="cyan"))
        with Live(spinner, console=console, refresh_per_second=8, transient=True):
            try:
                reply = _chat_completion(self.llm, self.history, message)
            except Exception as e:  # noqa: BLE001
                reply = f"(could not reach the model: {e})"
        reply = (reply or "").strip()
        if not reply:
            reply = "..."
        console.print(Panel(
            Markdown(reply),
            border_style="cyan",
            box=ROUNDED,
            title="gemma42",
            title_align="left",
        ))
        self.history.append({"role": "assistant", "content": reply})


@app.command("chat")
def chat_cmd(
    provider: str = typer.Option(None, "--provider"),
    model: str = typer.Option(None, "--model", "-m"),
) -> None:
    """Interactive Claude-Code-style shell. Type freely, /help for commands."""
    provider = provider or DEFAULT_AGENT_PROVIDER
    repl = REPL(provider=provider, model=model)
    repl.show_banner()
    while True:
        try:
            line = Prompt.ask("\n[bold green]›[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not line:
            continue
        try:
            if repl.handle_slash(line):
                continue
        except EOFError:
            console.print("[dim]bye[/dim]")
            return
        try:
            repl.handle_turn(line)
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸  interrupted[/yellow]")
        except Exception as e:  # noqa: BLE001
            console.print(Panel(Text(str(e), style="red"),
                                 title="error", border_style="red", box=ROUNDED))


@app.command("run")
def run_cmd(
    goal: str = typer.Option(..., help="Plain-English description of the dataset to build."),
    workdir: Path = typer.Option(Path("./gemma42_run"), help="Where dataset, cache, memory live."),
    min_rows: int = typer.Option(0, help="Minimum rows the dataset must contain."),
    required_fields: str = typer.Option(
        "", help="Comma-separated list of fields that must be present on every row."
    ),
    unique_field: str = typer.Option("", help="Field that must be unique across rows."),
    schema_file: Path | None = typer.Option(
        None, help="JSON file containing a JSON-Schema object that every row must satisfy."
    ),
    provider: str = typer.Option(
        None,
        help=f"LLM provider. One of: {', '.join(list_providers())}. Default: auto.",
    ),
    model: str = typer.Option("", help="Model id. Empty = provider default."),
    base_url: str = typer.Option("", help="Override the provider's base URL."),
    extraction_provider: str = typer.Option(
        None,
        "--extract-provider",
        help="Provider used only for schema-constrained extraction. Default: ollama.",
    ),
    extraction_model: str = typer.Option(
        None,
        "--extract-model",
        help="Model used only for schema-constrained extraction. Default: gemma4:latest.",
    ),
    max_turns: int = typer.Option(120, help="Hard limit on agent turns."),
    no_codebook: bool = typer.Option(False, help="Skip the codebook contract."),
    live: bool = typer.Option(True, "--live/--no-live", help="Use the live Rich feed."),
):
    """Power-user run with explicit flags. Use `ask` for the friendly prompt-first form."""
    provider = provider or DEFAULT_AGENT_PROVIDER
    extraction_provider = extraction_provider or DEFAULT_EXTRACTION_PROVIDER
    extraction_model = extraction_model or DEFAULT_EXTRACTION_MODEL
    if live:
        # Route through the live-feed orchestrator using the explicit goal.
        contracts_kwargs = {
            "rows": min_rows if min_rows > 0 else None,
            "workdir": workdir,
            "provider": provider,
            "model": model or None,
            "extraction_provider": extraction_provider,
            "extraction_model": extraction_model,
            "max_turns": max_turns,
            "no_codebook": no_codebook,
            "push": None,
            "public": False,
        }
        _orchestrate(goal, **contracts_kwargs)
        return

    # Plain (non-Live) path — useful when piping output to a file.
    _setup_logging(logging.INFO)
    contracts = []
    if min_rows > 0:
        contracts.append(MinRowsContract(min_rows=min_rows))
    if required_fields.strip():
        fields = [f.strip() for f in required_fields.split(",") if f.strip()]
        contracts.append(FieldsContract(required_fields=fields))
    if unique_field.strip():
        contracts.append(UniqueFieldContract(field=unique_field.strip()))
    if not no_codebook:
        contracts.append(CodebookContract(min_variables=20))
    schema = json.loads(Path(schema_file).read_text()) if schema_file else None
    llm = make_llm(provider, model=model or None, base_url=base_url or None)
    extraction_llm = make_llm(extraction_provider, model=extraction_model or None)
    result = run_agent(
        goal=goal,
        contracts=contracts,
        workdir=workdir,
        llm=llm,
        extraction_llm=extraction_llm,
        dataset_schema=schema,
        unique_key=unique_field.strip() or None,
        config=AgentConfig(max_turns=max_turns, verbose=True),
    )
    _print_summary(result)


@app.command("providers")
def providers_cmd() -> None:
    """List built-in LLM providers and their default models."""
    from gemma42.providers import PRESETS

    t = Table(title="providers", header_style="bold cyan", box=ROUNDED)
    t.add_column("name"); t.add_column("base_url"); t.add_column("default_model"); t.add_column("api_key_env")
    for p in PRESETS.values():
        t.add_row(p.name, p.base_url or "(pass --base-url)",
                  p.default_model or "(pass --model)", p.api_key_env or "(none)")
    console.print(t)


@app.command("export-hf")
def export_hf_cmd(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl produced by a run."),
    repo_id: str = typer.Option(..., help="Hugging Face dataset repo (e.g. yourname/my-dataset)."),
    private: bool = typer.Option(True),
):
    """Push a JSONL dataset to the Hugging Face Hub."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise typer.BadParameter("install extras: pip install gemma42[hf]") from e
    ds = load_dataset("json", data_files=str(dataset), split="train")
    ds.push_to_hub(repo_id, private=private)
    console.print(f"[green]pushed[/green] {len(ds)} rows → {repo_id}")


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    # Bare `gemma42` → REPL.
    chat_cmd()
    raise typer.Exit(0)


# ── default-route: `gemma42 "free text"` → `gemma42 ask "<text>"` ───────────


def _route_default_to_ask(argv: list[str]) -> list[str]:
    """If the first positional looks like free-text/URL (not a known subcommand),
    rewrite the argv so it becomes `ask <joined-tokens>`. Known subcommands
    pass through untouched."""
    known = {"ask", "chat", "run", "providers", "export-hf"}
    # Walk past leading global flags (none right now; future-proof).
    i = 0
    if i >= len(argv):
        return argv
    first = argv[i]
    if first in known or first.startswith("-"):
        return argv
    # Find the first index where an `ask`-style flag begins; join everything
    # before that as the prompt.
    ask_opts = {
        "--rows", "-n", "--workdir", "-w", "--provider", "--model", "-m",
        "--extract-provider", "--extract-model", "--max-turns", "--no-codebook",
        "--push", "--public",
    }
    msg_parts: list[str] = []
    tail: list[str] = []
    j = i
    while j < len(argv):
        tok = argv[j]
        if tok in ask_opts:
            tail = argv[j:]
            break
        msg_parts.append(tok)
        j += 1
    if not msg_parts:
        return argv
    return ["ask", " ".join(msg_parts), *tail]


def main() -> None:
    sys.argv = [sys.argv[0]] + _route_default_to_ask(sys.argv[1:])
    app()


if __name__ == "__main__":
    main()
