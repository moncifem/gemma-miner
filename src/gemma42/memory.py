"""Persistent JSON-backed key/value memory the agent can read and write across turns."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Memory:
    """Append-mostly JSON KV store.

    Keys are strings, values are anything JSON-serialisable. The whole store
    is small enough to fit in RAM — we write the full file on each update for
    crash safety. Use this for facts the agent has discovered (selectors,
    site quirks, schemas) so it can recall them in future turns.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text() or "{}")
            except json.JSONDecodeError:
                # corrupt — back up and start fresh
                self.path.rename(self.path.with_suffix(".corrupt.json"))
                self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._flush()
                return True
            return False

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)
