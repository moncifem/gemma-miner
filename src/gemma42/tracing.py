"""Per-run tracing.

Every agent run produces two artefacts in `<workdir>/`:

  - `trace.jsonl` : one JSON object per turn, suitable for replay / analysis.
                    Fields: ts, turn, event ("turn"|"llm_error"|"parse_error"),
                    thought, tool, args, observation, error, contracts,
                    n_rows, raw_model_output.
  - `trace.log`   : human-readable, line-wrapped, with full observations
                    (not truncated). This is the file to `tail -f`.

Both files are flushed and fsync'd after every write so a crash never loses
the trace tail.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.workdir / "trace.jsonl"
        self.log_path = self.workdir / "trace.log"
        self._jf = self.jsonl_path.open("a", encoding="utf-8")
        self._lf = self.log_path.open("a", encoding="utf-8")
        self._subscribers: list[Any] = []
        self._write_log_header()

    def subscribe(self, fn: Any) -> None:
        """Register a callable invoked with every event dict (for live UIs)."""
        self._subscribers.append(fn)

    def _write_log_header(self) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._lf.write(f"\n{'=' * 70}\n[{ts}] NEW RUN\n{'=' * 70}\n")
        self._lf.flush()

    def event(self, **fields: Any) -> None:
        fields.setdefault("ts", time.time())
        line = json.dumps(fields, ensure_ascii=False, default=str)
        self._jf.write(line + "\n")
        self._jf.flush()
        try:
            os.fsync(self._jf.fileno())
        except OSError:
            pass
        for fn in list(self._subscribers):
            try:
                fn(fields)
            except Exception:  # noqa: BLE001
                pass

    def turn(
        self,
        *,
        turn: int,
        thought: str,
        tool: str,
        args: dict,
        observation: str,
        error: bool,
        contracts: list[dict],
        n_rows: int,
        raw_model_output: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        self.event(
            event="turn",
            turn=turn,
            thought=thought,
            tool=tool,
            args=args,
            observation=observation,
            error=error,
            contracts=contracts,
            n_rows=n_rows,
            raw_model_output=raw_model_output,
            elapsed_ms=elapsed_ms,
        )
        ts = time.strftime("%H:%M:%S")
        mark = "ERR " if error else "    "
        args_s = json.dumps(args, ensure_ascii=False)
        if len(args_s) > 400:
            args_s = args_s[:400] + "…"
        header = (
            f"\n[{ts}] turn {turn:>3} {mark}→ {tool}\n"
            f"          thought: {thought}\n"
            f"          args:    {args_s}\n"
        )
        if raw_model_output is not None:
            preview = raw_model_output if len(raw_model_output) < 600 else raw_model_output[:600] + "…"
            header += f"          raw:     {preview!r}\n"
        if elapsed_ms is not None:
            header += f"          elapsed: {elapsed_ms:.0f} ms\n"
        header += "          observation:\n"
        body = "\n".join("            " + ln for ln in observation.splitlines())
        contracts_block = ""
        if contracts:
            contracts_block = "          contracts:\n"
            for c in contracts:
                m = "OK  " if c["ok"] else "FAIL"
                contracts_block += f"            [{m}] {c['name']}: {c['detail']}\n"
        rows_line = f"          rows: {n_rows}\n"
        self._lf.write(header + body + "\n" + contracts_block + rows_line)
        self._lf.flush()
        try:
            os.fsync(self._lf.fileno())
        except OSError:
            pass

    def note(self, kind: str, message: str, **extra: Any) -> None:
        self.event(event=kind, message=message, **extra)
        ts = time.strftime("%H:%M:%S")
        self._lf.write(f"\n[{ts}] {kind}: {message}\n")
        for k, v in extra.items():
            self._lf.write(f"           {k}: {v}\n")
        self._lf.flush()

    def close(self) -> None:
        try:
            self._jf.close()
        except OSError:
            pass
        try:
            self._lf.close()
        except OSError:
            pass
