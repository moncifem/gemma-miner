"""File/cache references for tool arguments.

The single most failure-prone pattern for small models is having to round-trip
large content (a PDF text, a long HTML body) through the LLM as a string
argument. The model hits its max_tokens budget and gets truncated mid-string,
which the parser then can't recover.

The fix: any tool argument may be a *reference object* instead of a literal:

    {"$file": "items/item_0001/attachment_01.txt"}  # relative to workdir
    {"$file": "/abs/path/to/file.txt"}              # absolute path also accepted
    {"$file": "items/.../x.txt", "encoding": "utf-8"}

References resolve to the file's content (decoded as utf-8 by default) at
tool-dispatch time. The model now passes a 50-byte reference instead of a
200,000-byte string. The dataset, the extractor, and the structured-extract
tool all use this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_refs(value: Any, workdir: str | Path) -> Any:
    """Recursively expand any ``{"$file": "..."}`` references in *value*.

    Leaves everything else untouched. Returns a new structure; the input
    is not mutated.
    """
    workdir = Path(workdir)
    return _resolve(value, workdir)


def _resolve(value: Any, workdir: Path) -> Any:
    if isinstance(value, dict):
        if "$file" in value:
            return _read_file(value, workdir)
        return {k: _resolve(v, workdir) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, workdir) for v in value]
    return value


def _read_file(ref: dict, workdir: Path) -> str:
    raw = ref["$file"]
    if not isinstance(raw, str):
        raise ValueError(f"$file must be a string path, got {type(raw).__name__}")
    p = Path(raw)
    if not p.is_absolute():
        p = workdir / p
    if not p.exists():
        raise FileNotFoundError(f"$file reference not found: {p}")
    encoding = ref.get("encoding", "utf-8")
    try:
        return p.read_text(encoding=encoding, errors="replace")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"failed to read $file {p}: {e}") from e


def describe_ref_syntax() -> str:
    """Single source of truth for the system prompt."""
    return (
        'Any string-valued argument may be replaced with a reference object '
        '`{"$file": "<path>"}` where the path is relative to the workdir '
        '(absolute paths are also accepted). At tool-dispatch time the '
        'reference is replaced with the UTF-8 content of the file. Use this '
        'to avoid round-tripping large content (PDF text, HTML bodies) '
        'through the LLM — pass the file path instead.'
    )
