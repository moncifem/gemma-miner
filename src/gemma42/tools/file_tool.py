"""Read / write / list files inside the workdir. Paths are confined to the workdir."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _resolve(state: "AgentState", path: str) -> Path:
    base = Path(state.workdir).resolve()
    p = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    # confinement: only allow paths inside workdir
    try:
        p.relative_to(base)
    except ValueError as e:
        raise ValueError(f"path escapes workdir: {p}") from e
    return p


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workdir. Returns the first 8 KB by default."
    args_schema = {
        "path": {"type": "string", "description": "Path relative to workdir."},
        "max_chars": {"type": "integer", "default": 8000},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        path = args.get("path")
        if not path:
            return ToolResult(output="ERROR: 'path' required", error=True)
        max_chars = int(args.get("max_chars") or 8000)
        try:
            p = _resolve(state, path)
        except ValueError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        if not p.exists():
            return ToolResult(output=f"ERROR: not found: {p}", error=True)
        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR reading {p}: {e}", error=True)
        out = data[:max_chars]
        if len(data) > max_chars:
            out += f"\n... [truncated, total {len(data)} chars]"
        return ToolResult(output=out)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write a UTF-8 text file inside the workdir. Overwrites existing content. "
        "To 'delete' a file, write an empty string to it."
    )
    args_schema = {
        "path": {"type": "string", "description": "Path relative to workdir."},
        "content": {"type": "string", "description": "File contents."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        path = args.get("path")
        content = args.get("content", "")
        if not path:
            return ToolResult(output="ERROR: 'path' required", error=True)
        try:
            p = _resolve(state, path)
        except ValueError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(output=f"wrote {len(content)} chars to {p}")


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files in a directory under the workdir."
    args_schema = {
        "path": {"type": "string", "default": ".", "description": "Path relative to workdir."},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        path = args.get("path") or "."
        try:
            p = _resolve(state, path)
        except ValueError as e:
            return ToolResult(output=f"ERROR: {e}", error=True)
        if not p.exists():
            return ToolResult(output=f"ERROR: not found: {p}", error=True)
        if not p.is_dir():
            return ToolResult(output=f"ERROR: not a directory: {p}", error=True)
        lines = []
        for entry in sorted(p.iterdir()):
            kind = "d" if entry.is_dir() else "f"
            size = entry.stat().st_size if entry.is_file() else 0
            lines.append(f"{kind} {size:>10}  {entry.name}")
        return ToolResult(output="\n".join(lines) or "(empty)")
