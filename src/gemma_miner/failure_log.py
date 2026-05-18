"""Best-effort failure logger.

Tool-level errors are already captured in ``trace.log`` / ``trace.jsonl``;
this module writes an additional, lighter ``failures.log`` so the user can
``tail -F`` a single file during long runs. Never raises.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def log_failure(
    workdir: str | Path | None,
    *,
    kind: str,
    tool: str | None = None,
    turn: int | None = None,
    raw_response: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if not workdir:
        return
    try:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "kind": kind,
            "tool": tool,
            "turn": turn,
            "payload": payload or {},
        }
        if raw_response is not None:
            record["raw_response"] = raw_response[:2000]
        with (wd / "failures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        return
