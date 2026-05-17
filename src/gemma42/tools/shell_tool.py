"""Python execution & bash tools.

Bash is intentionally limited: a blocklist on destructive commands (rm, mkfs,
shutdown, dd, …). The python tool runs each snippet in a fresh subprocess so
state doesn't leak between turns; if persistent state is needed, the agent
should write to a file in the workdir.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


# any token matching these (as a whole word) is rejected
_BANNED = [
    r"\brm\b",
    r"\brmdir\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\bmv\s+/\s",
    r":\(\)\s*\{",  # fork bomb
    r"\bchown\s+-R\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bsudo\b",
    r"\bsu\b",
    r"/etc/passwd",
    r"/etc/shadow",
    r">\s*/dev/sd",
]
_BANNED_RE = re.compile("|".join(_BANNED))


def _check_safety(cmd: str) -> str | None:
    m = _BANNED_RE.search(cmd)
    if m:
        return f"refused: command contains forbidden token {m.group(0)!r}. " \
               "Destructive operations (rm, mkfs, dd, shutdown, sudo, ...) are blocked."
    return None


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command. The command runs from the agent's workdir. "
        "Destructive commands are blocked (rm, mv to /, mkfs, dd, shutdown, "
        "sudo, chown -R, chmod -R 777, fork bombs). To delete a file, write "
        "an empty file over it; never invoke rm. Output is truncated to 8 KB."
    )
    args_schema = {
        "command": {"type": "string", "description": "Single bash command (use `bash -c`)."},
        "timeout": {"type": "integer", "default": 60, "description": "Seconds."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        cmd = args.get("command", "")
        if not cmd:
            return ToolResult(output="ERROR: 'command' required", error=True)
        err = _check_safety(cmd)
        if err:
            return ToolResult(output=f"REFUSED: {err}", error=True)
        timeout = int(args.get("timeout") or 60)
        try:
            p = subprocess.run(
                ["bash", "-c", cmd],
                cwd=state.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(output=f"ERROR: command timed out after {timeout}s", error=True)
        out = p.stdout
        if p.stderr:
            out += "\n--- stderr ---\n" + p.stderr
        if len(out) > 8000:
            out = out[:8000] + f"\n... [truncated, total {len(out)} bytes]"
        out = f"exit_code: {p.returncode}\n{out}"
        return ToolResult(output=out, error=p.returncode != 0)


def _python_error_hints(stderr: str) -> str:
    """Tack on actionable hints for the most common small-model Python mistakes."""
    hints: list[str] = []
    if "SyntaxError" in stderr and "[^" in stderr and "']" in stderr:
        hints.append(
            'HINT: you wrote a character class like [^"\'] inside a Python string '
            "delimited by '...'. The inner ' closes the string. Either escape it "
            "(\\') or use double-quoted strings (\"...\") for the python source. "
            "Or BETTER: stop fighting Python regex; use the declarative "
            "`extractor_define` tool instead — its spec is JSON, no quote escaping."
        )
    if "SyntaxError" in stderr and "EOL while scanning" in stderr:
        hints.append(
            "HINT: unterminated string literal. Check that every '\"' is closed."
        )
    if "FileNotFoundError" in stderr:
        hints.append(
            "HINT: file not found. Tool paths are ABSOLUTE — pass the cache_path "
            "from http_get verbatim, do not strip or prepend anything."
        )
    if "ModuleNotFoundError" in stderr and "requests" in stderr:
        hints.append(
            "HINT: 'requests' is NOT installed. Use urllib.request or httpx instead."
        )
    return "\n\n".join(hints)


class PythonExecTool(Tool):
    name = "python"
    description = (
        "Execute a Python 3 snippet in a fresh subprocess from the workdir. "
        "Useful for transforms, parsing, or quick ad-hoc analysis. The "
        "interpreter is the same one running the agent. Each call is "
        "stateless — to persist data, write to a file in the workdir. "
        "Print whatever you want shown back; output is truncated to 8 KB."
    )
    args_schema = {
        "code": {"type": "string", "description": "Python source code."},
        "timeout": {"type": "integer", "default": 60, "description": "Seconds."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        code = args.get("code", "")
        if not code:
            return ToolResult(output="ERROR: 'code' required", error=True)
        err = _check_safety(code)
        if err:
            return ToolResult(output=f"REFUSED: {err}", error=True)
        timeout = int(args.get("timeout") or 60)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, dir=state.workdir
        ) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.run(
                [sys.executable, path],
                cwd=state.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(output=f"ERROR: python timed out after {timeout}s", error=True)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        out = p.stdout
        if p.stderr:
            out += "\n--- stderr ---\n" + p.stderr
        if len(out) > 8000:
            out = out[:8000] + f"\n... [truncated, total {len(out)} bytes]"
        hints = _python_error_hints(p.stderr) if p.returncode != 0 else ""
        out = f"exit_code: {p.returncode}\n{out}"
        if hints:
            out += "\n\n" + hints
        return ToolResult(output=out, error=p.returncode != 0)
