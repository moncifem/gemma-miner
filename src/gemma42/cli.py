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
    "extract_items":      ("🧮", "bright_yellow"),
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


def _print_banner(provider: str, model: str, workdir: Path) -> None:
    text = (
        f"[bold cyan]gemma42[/bold cyan]  ·  text-to-dataset agent\n"
        f"[dim]model:  {provider}/{model}[/dim]\n"
        f"[dim]workdir: {workdir}[/dim]"
    )
    console.print(Panel.fit(text, border_style="cyan", box=ROUNDED))


# ── prompt parsing ──────────────────────────────────────────────────────────


_COUNT_PATTERNS = [
    re.compile(r"\btop\s+(\d{2,7})\b", re.IGNORECASE),
    re.compile(
        r"\b(\d{2,7})\s*(?:articles?|rows?|items?|records?|papers?|entries?|"
        r"decisions?|sanctions?|cases?|posts?|pages?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:min|at least|atleast|>=?)\s*(\d{2,7})\b", re.IGNORECASE),
]


def _extract_url(prompt: str) -> str | None:
    m = re.search(r"https?://[^\s)>\]\"']+", prompt)
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?")
    return url


def _extract_count(prompt: str) -> int | None:
    for rx in _COUNT_PATTERNS:
        m = rx.search(prompt)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


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
        self.contracts: list[dict] = []
        # in-flight tool calls: index → start_ts
        self._inflight: list[float] = []
        self.live: Any = None
        self._graduated: set[int] = set()

    # ── feed input ─────────────────────────────────────────────────────────

    def on_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        if kind == "phase":
            self.phase = ev.get("phase", "")
        elif kind == "llm_start":
            model = ev.get("model", "?")
            self.set_thinking(f"calling [bold]{model}[/bold]…")
        elif kind == "turn":
            self._on_turn(ev)
        elif kind == "run_end":
            self.spinner_active = False
        elif kind == "aborted":
            self.spinner_active = False

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
        # show first failing contract for context
        failing = next((c for c in self.contracts if not c.get("ok")), None)
        c_part = ""
        if failing:
            c_part = (
                f"  [dim]·[/dim]  "
                f"[red]{failing['name']}[/red] [dim]{failing.get('detail','')}[/dim]"
            )
        return Text.from_markup(
            f"  phase: {phase_part}  [dim]·[/dim]  "
            f"step {self.step_count}  [dim]·[/dim]  "
            f"rows {self.rows}  [dim]·[/dim]  "
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
        # extract a useful one-line summary
        summary = self._inline_summary(name, content)
        ms = f"  · {elapsed_ms / 1000.0:.1f}s" if elapsed_ms else ""
        glyph = "✗" if is_error else "↳"
        style = "bold red" if is_error else "dim"
        # multi-line / longer summaries get a tiny panel
        if len(summary) > 180 or "\n" in summary:
            return Panel(
                Text(summary[:1200], style="white", overflow="fold"),
                border_style="red" if is_error else "dim",
                box=SIMPLE,
                padding=(0, 1),
            )
        return Text.assemble(
            ("    " + glyph + " ", style),
            (summary, "red" if is_error else "white"),
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
            "extract_items":     ["processed", "errors"],
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
    max_turns: int,
    want_codebook: bool,
) -> Any:
    """Run the agent with a live Rich feed. Returns the RunResult."""
    _setup_logging()
    workdir.mkdir(parents=True, exist_ok=True)

    llm = make_llm(provider, model=model)
    _print_banner(provider, llm.config.model, workdir)

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
                unique_key=unique_field or None,
                config=cfg,
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
            console.print(Text(
                f"  load it: [white]pd.read_parquet('{export_dir}/<name>.parquet')[/white]  "
                f"·  push: [white]gemma42 hf-push <repo-id> --workdir {workdir}[/white]",
                style="dim",
            ))


# ── orchestrator: free-text prompt → run ────────────────────────────────────


def _orchestrate(
    prompt: str,
    *,
    rows: int | None,
    workdir: Path | None,
    provider: str,
    model: str | None,
    max_turns: int,
    no_codebook: bool,
    push: str | None,
    public: bool,
) -> Any:
    """One-call entry point: parse the prompt, derive defaults, run."""
    prompt = prompt.strip()
    url = _extract_url(prompt)
    parsed_count = _extract_count(prompt)
    target_rows = rows if rows is not None else (parsed_count or 50)

    if workdir is None:
        workdir = _auto_workdir(prompt, url, Path("./runs"))

    # build the agent's goal: the user's prompt + a structured-data nudge
    # so the agent enters the codebook phase after harvesting.
    goal_lines = [prompt]
    if not no_codebook:
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
        required_fields=[],
        unique_field=None,
        provider=provider,
        model=model,
        max_turns=max_turns,
        want_codebook=not no_codebook,
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
        "together", "--provider",
        help=f"LLM provider: {', '.join(list_providers())}",
    ),
    model: str = typer.Option(None, "--model", "-m"),
    max_turns: int = typer.Option(120, "--max-turns"),
    no_codebook: bool = typer.Option(
        False, "--no-codebook",
        help="Skip the auto codebook + extract + export phases. Stop after harvest.",
    ),
    push: str = typer.Option(
        None, "--push",
        help="Hugging Face repo id to push the final dataset to (e.g. you/cnil-sanctions).",
    ),
    public: bool = typer.Option(
        False, "--public",
        help="Make the pushed HF dataset public (default private).",
    ),
):
    """One-shot prompt-first run. The agent extracts URL+count from the prompt."""
    _orchestrate(
        prompt, rows=rows, workdir=workdir, provider=provider, model=model,
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
    provider: str = typer.Option("together", "--provider"),
    model: str = typer.Option(None, "--model", "-m"),
) -> None:
    """Interactive Claude-Code-style shell. Type freely, /help for commands."""
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
        "together",
        help=f"LLM provider. One of: {', '.join(list_providers())}.",
    ),
    model: str = typer.Option("", help="Model id. Empty = provider default."),
    base_url: str = typer.Option("", help="Override the provider's base URL."),
    max_turns: int = typer.Option(120, help="Hard limit on agent turns."),
    no_codebook: bool = typer.Option(False, help="Skip the codebook contract."),
    live: bool = typer.Option(True, "--live/--no-live", help="Use the live Rich feed."),
):
    """Power-user run with explicit flags. Use `ask` for the friendly prompt-first form."""
    if live:
        # Route through the live-feed orchestrator using the explicit goal.
        contracts_kwargs = {
            "rows": min_rows if min_rows > 0 else None,
            "workdir": workdir,
            "provider": provider,
            "model": model or None,
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
    result = run_agent(
        goal=goal,
        contracts=contracts,
        workdir=workdir,
        llm=llm,
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
    repo_id: str = typer.Option(..., help="Hugging Face dataset repo (e.g. yourname/cnil-sanctions)."),
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
        "--max-turns", "--no-codebook", "--push", "--public",
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
