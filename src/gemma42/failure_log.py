"""Failure log — every parseable-or-not problem we hit, fully detailed.

Two files written to <workdir>/:
  - failures.jsonl  : one JSON object per failure (machine-readable)
  - failures.log    : human-readable, with full raw model outputs

Goes hand-in-hand with trace.log, but specifically curates the FAILURES so
the developer/user can debug without skimming the whole trace. Append-only,
flushed + fsync'd on every write so a crash never costs the tail.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def log_failure(
    workdir: str | Path | None,
    *,
    kind: str,
    payload: dict[str, Any] | None = None,
    tool: str | None = None,
    turn: int | None = None,
    raw_response: str | None = None,
) -> None:
    """Append a failure record. Safe in any context — silent on I/O errors.

    Fields:
      kind          short tag, e.g. 'parse_error', 'llm_scrape_empty_chunk',
                    'tool_error', 'llm_truncated', 'scrape_paginated_refused'
      tool          which tool, if applicable
      turn          agent turn number, if applicable
      raw_response  the full untruncated model output, when we have it
      payload       any extra structured context
    """
    if workdir is None:
        return
    wd = Path(workdir)
    try:
        wd.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    record: dict[str, Any] = {
        "ts":   time.time(),
        "kind": kind,
    }
    if tool:          record["tool"] = tool
    if turn is not None: record["turn"] = turn
    if raw_response is not None:
        record["raw_response"] = raw_response
        record["raw_length"]   = len(raw_response)
    if payload:
        # Keep payload separate so structure is clean.
        record["payload"] = payload

    # 1. JSONL (machine)
    jsonl = wd / "failures.jsonl"
    try:
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except OSError:
        pass

    # 2. Human log
    log_path = wd / "failures.log"
    try:
        with log_path.open("a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["ts"]))
            head = f"\n[{ts}] {kind}"
            if tool:
                head += f"  tool={tool}"
            if turn is not None:
                head += f"  turn={turn}"
            f.write(head + "\n")
            if payload:
                for k, v in payload.items():
                    if k == "raw_response":
                        continue
                    s = json.dumps(v, ensure_ascii=False, default=str)
                    if len(s) > 600:
                        s = s[:600] + "…"
                    f.write(f"  {k}: {s}\n")
            if raw_response is not None:
                f.write(f"  raw_response ({len(raw_response)} chars):\n")
                # Don't truncate the raw in the log — that's the point.
                indented = "\n".join("    " + ln for ln in raw_response.splitlines() or [""])
                f.write(indented + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except OSError:
        pass


def failure_paths(workdir: str | Path | None) -> tuple[Path | None, Path | None]:
    if workdir is None:
        return None, None
    wd = Path(workdir)
    return (wd / "failures.log", wd / "failures.jsonl")
