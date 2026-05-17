"""Persistent stateful Python kernel for the agent.

Replaces the per-call subprocess. Variables, functions, parsed objects
persist across snippets. Each snippet executes in the same Python process
(an isolated subprocess for safety, kept alive between tool calls) and
returns its captured stdout/stderr.

Plus:
  - Static analysis BEFORE execution catches the model's regex/quote bugs.
  - Skill promotion: when a function is called repeatedly, prompt to lift
    it into a named tool stored in the autobiography.

Protocol (line-oriented JSON, both directions; robust to subprocess buffering):
  master → kernel : one line of JSON  {"code_b64": "<base64-encoded source>"}
  kernel → master : one line of JSON  {"status":"ok|error|exit",
                                        "stdout":"...","stderr":"...",
                                        "vars":[...]}
"""

from __future__ import annotations

import ast
import base64
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ── static analysis ────────────────────────────────────────────────────────


def lint_snippet(code: str) -> list[str]:
    warnings: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        msg = f"SyntaxError at line {e.lineno}: {e.msg}"
        if e.text:
            line = e.text.rstrip()
            if "[^" in line and "'" in line[: e.offset or len(line)]:
                msg += (
                    "  (common mistake: a character class like [^\"'] "
                    "inside a single-quoted Python string closes the string. "
                    "Use r\"...\" or escape the inner quote.)"
                )
        return [msg]
    has_print = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name == "requests":
                    warnings.append(
                        "imports `requests` — not installed. "
                        "Use `urllib.request` or `httpx` instead."
                    )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                has_print = True
            if isinstance(func, ast.Attribute) and func.attr == "print":
                has_print = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            if node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    v = a.value
                    if not v.startswith(("/", "~")) and "cache" not in v:
                        warnings.append(
                            f"open({v!r}) uses a non-absolute path — make sure "
                            "it's relative to the workdir or use the absolute "
                            "cache_path returned by http_get."
                        )
    if not has_print:
        warnings.append(
            "no `print(...)` calls — you won't see any output. "
            "Add prints for values you want to inspect."
        )
    return warnings


# ── persistent kernel ──────────────────────────────────────────────────────


_BOOTSTRAP = r'''
import base64, io, json, sys, traceback

_globals = {"__name__": "__main__"}

def _emit(obj):
    sys.__stdout__.write(json.dumps(obj, default=str) + "\n")
    sys.__stdout__.flush()

while True:
    line = sys.__stdin__.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        code = base64.b64decode(req["code_b64"]).decode("utf-8")
    except Exception as e:
        _emit({"status":"error","stdout":"","stderr":f"protocol: {e}","vars":[]})
        continue
    out_buf, err_buf = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out_buf, err_buf
    status = "ok"
    try:
        exec(compile(code, "<snippet>", "exec"), _globals)
    except SystemExit:
        status = "exit"
    except BaseException:
        status = "error"
        err_buf.write(traceback.format_exc())
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    payload = {
        "status": status,
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "vars":   [k for k in _globals.keys()
                   if not k.startswith("_") and k != "json"],
    }
    _emit(payload)
'''


class PersistentKernel:
    def __init__(self, workdir: str | Path, *, timeout: int = 120):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.call_counter: Counter = Counter()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _BOOTSTRAP],
            cwd=str(self.workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._proc = None

    def reset(self) -> None:
        self.close()
        self.call_counter.clear()
        self._start()

    # ── execution ──────────────────────────────────────────────────────

    def run(self, code: str, *, timeout: int | None = None) -> dict:
        if not code.strip():
            return {"status": "ok", "stdout": "", "stderr": "", "vars": []}
        with self._lock:
            self._start()
            assert self._proc and self._proc.stdin and self._proc.stdout

            payload = json.dumps({
                "code_b64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
            })
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.reset()
                return self.run(code, timeout=timeout)

            deadline = time.time() + (timeout or self.timeout)
            # Read the single-line JSON response with a soft timeout.
            response_line: str | None = None

            def _reader_target(result: list[str]) -> None:
                try:
                    line = self._proc.stdout.readline()
                except Exception:  # noqa: BLE001
                    line = ""
                result.append(line)

            box: list[str] = []
            thread = threading.Thread(target=_reader_target, args=(box,), daemon=True)
            thread.start()
            thread.join(max(0.1, deadline - time.time()))
            if thread.is_alive():
                self.reset()
                return {"status": "timeout", "stdout": "", "stderr": "kernel timeout",
                        "vars": []}
            if not box or not box[0]:
                self.reset()
                return {"status": "error", "stdout": "", "stderr": "kernel died",
                        "vars": []}
            response_line = box[0].strip()
            try:
                resp = json.loads(response_line)
            except json.JSONDecodeError as e:
                return {"status": "error", "stdout": "",
                        "stderr": f"kernel JSON parse: {e}\nraw: {response_line[:300]}",
                        "vars": []}
            self._record_call_frequencies(code)
            return resp

    # ── skill promotion ────────────────────────────────────────────────

    def _record_call_frequencies(self, code: str) -> None:
        for m in re.finditer(r"\b([a-z_][a-z_0-9]{3,})\s*\(", code):
            name = m.group(1)
            if name in {"print", "open", "len", "range", "int", "float", "str",
                        "list", "dict", "set", "tuple", "max", "min", "sum",
                        "sorted", "filter", "map", "any", "all", "enumerate",
                        "isinstance", "type", "format"}:
                continue
            self.call_counter[name] += 1

    def skill_candidates(self, *, min_calls: int = 3) -> list[tuple[str, int]]:
        return [(n, c) for n, c in self.call_counter.most_common() if c >= min_calls]
