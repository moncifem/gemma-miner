"""A stateful Python tool backed by gemma42.kernel.PersistentKernel.

This is a drop-in better cousin of the existing `python` tool. The big
differences:
  - Variables persist across snippets (no more re-reading the same file
    five times).
  - Static analysis runs BEFORE execution and surfaces common bugs
    (regex-quote mistakes, missing print(), relative paths, `requests`).
  - Suggests `skill_promote` when a function is repeatedly called.

We keep the existing `python` tool too so behaviour stays backwards-
compatible; the new tool is `pykernel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma42.kernel import PersistentKernel, lint_snippet
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


_KERNELS: dict[str, PersistentKernel] = {}


def _get_kernel(state: "AgentState") -> PersistentKernel:
    key = str(state.workdir)
    if key not in _KERNELS:
        _KERNELS[key] = PersistentKernel(state.workdir)
    return _KERNELS[key]


class PyKernelTool(Tool):
    name = "pykernel"
    description = (
        "Execute Python in a PERSISTENT kernel — variables, functions, and "
        "parsed objects survive between calls within this run. Static "
        "analysis runs before execution and warns about common bugs "
        "(regex-quote escapes, missing prints, relative paths, importing "
        "`requests`). Each snippet's stdout/stderr is captured and "
        "returned. Use this whenever you'd otherwise re-parse the same "
        "file repeatedly."
    )
    args_schema = {
        "code":     {"type": "string"},
        "timeout":  {"type": "integer", "default": 60},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        code = args.get("code", "")
        if not code:
            return ToolResult(output="ERROR: 'code' required", error=True)
        warnings = lint_snippet(code)
        kernel = _get_kernel(state)
        result = kernel.run(code, timeout=int(args.get("timeout") or 60))
        status = result.get("status", "ok")
        out_parts: list[str] = []
        if warnings:
            out_parts.append("--- lint ---")
            for w in warnings:
                out_parts.append("  ⚠ " + w)
        stdout = (result.get("stdout") or "").rstrip()
        stderr = (result.get("stderr") or "").rstrip()
        if stdout:
            out_parts.append("--- stdout ---")
            out_parts.append(stdout[:8000])
        if stderr:
            out_parts.append("--- stderr ---")
            out_parts.append(stderr[:4000])
        out_parts.append(f"status: {status}    vars in kernel: {len(result.get('vars') or [])}")
        # Skill promotion hint
        candidates = kernel.skill_candidates(min_calls=3)
        if candidates:
            out_parts.append(
                "TIP: " + ", ".join(f"{n}({c}×)" for n, c in candidates[:3])
                + " — consider `skill_promote` to make these reusable across runs."
            )
        is_error = status == "error" or status == "timeout"
        return ToolResult(output="\n".join(out_parts), error=is_error)


class KernelResetTool(Tool):
    name = "pykernel_reset"
    description = "Reset the persistent Python kernel (clear all variables/imports)."
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        kernel = _get_kernel(state)
        kernel.reset()
        return ToolResult(output="kernel reset")


class SkillPromoteTool(Tool):
    name = "skill_promote"
    description = (
        "Persist a Python helper as a 'skill' so it's available in every "
        "future run on this machine. The body is stored in the GLOBAL "
        "autobiography; promoted skills appear in the kernel's namespace "
        "automatically next session (TODO: auto-import). Args: name, body, "
        "description."
    )
    args_schema = {
        "name":        {"type": "string"},
        "body":        {"type": "string"},
        "description": {"type": "string"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        from gemma42.autobiography.store import global_db

        name = (args.get("name") or "").strip()
        body = args.get("body") or ""
        descr = args.get("description") or ""
        if not name or not body:
            return ToolResult(output="ERROR: 'name' and 'body' required", error=True)
        gl = global_db()
        try:
            skill = gl.save_skill(name=name, body=body, description=descr)
        finally:
            gl.close()
        return ToolResult(output=f"skill promoted: {skill.name} (id={skill.id})")
