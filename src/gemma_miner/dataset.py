"""JSONL-backed Dataset with optional JSON-Schema validation.

Resilient to mid-run crashes: every append is flushed and fsync'd. The full
dataset is recoverable by re-reading the JSONL file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterator


# ── Stable id synthesis ────────────────────────────────────────────────────


def synthesize_id(row: dict, *, prefix: str = "sig_") -> str:
    """Compute a deterministic, idempotent id from a row's user-meaningful
    content. Use the SAME function everywhere we need to mint an id so the
    bronze rows produced by `dataset_from_queue`, `process_queue`, and
    `llm_scrape` all converge on the same value for the same input — and
    silver rows can always join back to bronze.

    Excluded from the hash:
      • any field starting with `_`
      • the `id` field itself (avoids self-reference)
      • non-stable fields like `text_path` (path differs across runs)
    """
    SKIP = {"id", "text_path", "txt_path", "txt_file", "text_file", "cache_path"}
    payload = {
        k: v for k, v in sorted(row.items())
        if not str(k).startswith("_") and k not in SKIP
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}"


def ensure_row_id(row: dict) -> dict:
    """Return a row with an `id` field, minting one deterministically if missing.

    Does NOT overwrite existing ids. Returns the same dict (mutated in place
    for callers that don't expect a copy) so callers can use it inline."""
    if not isinstance(row, dict):
        return row
    if not row.get("id"):
        row["id"] = synthesize_id(row)
    return row


class Dataset:
    def __init__(
        self,
        path: str | Path,
        *,
        schema: dict | None = None,
        unique_key: str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.unique_key = unique_key
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._seen: set[str] = set()
        self._mtime: float = 0.0
        self._load_from_disk()

    # ── disk sync ──────────────────────────────────────────────────────

    def _current_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime if self.path.exists() else 0.0
        except OSError:
            return 0.0

    def _load_from_disk(self) -> None:
        """(Re)populate the in-memory rows + dedup set from the JSONL file."""
        self._rows = []
        self._seen = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._rows.append(row)
                if self.unique_key and self.unique_key in row:
                    self._seen.add(str(row[self.unique_key]))
        self._mtime = self._current_mtime()

    def _sync_if_externally_modified(self) -> None:
        """If the file was rewritten outside this process (e.g. by the
        agent's `python` tool), reload our in-memory view from disk.

        Stale in-memory state is a real failure mode: the agent rewrites
        `dataset.jsonl` adding `id` fields, then `extract_items` keeps
        seeing the OLD rows (no `id`) and fails the same way 10× in a row.
        """
        on_disk = self._current_mtime()
        if on_disk > self._mtime + 0.0001:
            self._load_from_disk()

    # ── public ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        self._sync_if_externally_modified()
        return len(self._rows)

    def __iter__(self) -> Iterator[dict]:
        self._sync_if_externally_modified()
        return iter(list(self._rows))

    def rows(self) -> list[dict]:
        with self._lock:
            self._sync_if_externally_modified()
            return list(self._rows)

    def append(self, row: dict) -> tuple[bool, str]:
        """Append a row. Returns (added, reason)."""
        if not isinstance(row, dict):
            return False, "row must be a dict"
        if self.schema:
            ok, err = _validate_against_schema(row, self.schema)
            if not ok:
                return False, f"schema violation: {err}"
        if self.unique_key:
            key = row.get(self.unique_key)
            if key is None:
                return False, f"missing unique key '{self.unique_key}'"
            if str(key) in self._seen:
                return False, f"duplicate {self.unique_key}={key}"
        with self._lock:
            self._sync_if_externally_modified()
            self._rows.append(row)
            if self.unique_key:
                self._seen.add(str(row[self.unique_key]))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._mtime = self._current_mtime()
        return True, "ok"

    def upsert(self, row: dict) -> tuple[bool, str]:
        """Insert or REPLACE a row by its `unique_key`. Returns (changed, reason).

        Used by the extraction phase: the harvest produced a row with metadata,
        and we want to merge in the structured columns without duplicating.
        """
        if not isinstance(row, dict):
            return False, "row must be a dict"
        key_field = self.unique_key
        if key_field is None and "id" in row:
            # Most harvesting tools synthesize a stable `id` even when the
            # user did not request a unique field. Treat it as the implicit
            # upsert key so schema extraction enriches rows instead of
            # duplicating the corpus.
            key_field = "id"
        if not key_field:
            ok, reason = self.append(row)
            return ok, reason
        key = row.get(key_field)
        if key is None:
            return False, f"missing unique key '{key_field}'"
        key_s = str(key)
        with self._lock:
            self._sync_if_externally_modified()
            replaced = False
            for i, existing in enumerate(self._rows):
                if str(existing.get(key_field)) == key_s:
                    # Merge keeping new values where non-null, else old.
                    merged = dict(existing)
                    for k, v in row.items():
                        if v is not None or k not in merged:
                            merged[k] = v
                    if self.schema:
                        ok, err = _validate_against_schema(merged, self.schema)
                        if not ok:
                            return False, f"schema violation: {err}"
                    self._rows[i] = merged
                    replaced = True
                    break
            if not replaced:
                # Brand new — append normally (but bypass the schema/dedupe
                # path because we already manage that).
                self._rows.append(row)
                self._seen.add(key_s)
            # Rewrite the file (small datasets — fine in practice).
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for r in self._rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._mtime = self._current_mtime()
        return True, "upserted" if replaced else "appended"

    def stats(self) -> dict[str, Any]:
        with self._lock:
            field_counts: dict[str, int] = {}
            for r in self._rows:
                for k in r:
                    field_counts[k] = field_counts.get(k, 0) + 1
            return {
                "n_rows": len(self._rows),
                "field_coverage": field_counts,
                "path": str(self.path),
            }

    def to_jsonl(self) -> str:
        return str(self.path)


# ── minimal JSON-Schema validator (subset) ─────────────────────────────


_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _validate_against_schema(value: Any, schema: dict, path: str = "$") -> tuple[bool, str]:
    t = schema.get("type")
    if t == "object" or (t is None and "properties" in schema):
        if not isinstance(value, dict):
            return False, f"{path}: expected object, got {type(value).__name__}"
        for key in schema.get("required", []):
            if key not in value:
                return False, f"{path}.{key}: required"
        for key, sub in schema.get("properties", {}).items():
            if key in value and value[key] is not None:
                ok, err = _validate_against_schema(value[key], sub, f"{path}.{key}")
                if not ok:
                    return ok, err
        return True, ""
    if t == "array":
        if not isinstance(value, list):
            return False, f"{path}: expected array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                ok, err = _validate_against_schema(item, item_schema, f"{path}[{i}]")
                if not ok:
                    return ok, err
        return True, ""
    if isinstance(t, str) and t in _TYPES:
        expected = _TYPES[t]
        if value is None and "null" not in schema.get("type", []):
            # allow null for nullable fields handled by caller
            pass
        if not isinstance(value, expected) and value is not None:
            return False, f"{path}: expected {t}, got {type(value).__name__}"
    if isinstance(t, list):
        if not any(isinstance(value, _TYPES[x]) for x in t if x in _TYPES):
            return False, f"{path}: expected one of {t}"
    if "enum" in schema and value not in schema["enum"]:
        return False, f"{path}: '{value}' not in enum {schema['enum']}"
    return True, ""
