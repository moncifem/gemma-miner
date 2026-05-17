"""Per-row provenance, reproducibility manifest, and dataset diffs.

The agent's output isn't trustworthy unless we can answer three questions:

  1. Where did THIS row's values come from?            → `_prov` per row
  2. How do I rebuild the entire dataset?               → gemma42.lock
  3. What changed since the last run on the same site?  → semantic diff
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ── per-row provenance ─────────────────────────────────────────────────────


PROV_KEY = "_prov"   # the row field that holds the provenance blob


def make_row_prov(
    *,
    source_url: str | None = None,
    extractor_name: str | None = None,
    codebook_version: str | None = None,
    llm_model: str | None = None,
    prompt_hash: str | None = None,
    fingerprint: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a provenance blob to attach to a single dataset row."""
    blob = {
        "source_url": source_url,
        "extractor": extractor_name,
        "codebook_version": codebook_version,
        "llm_model": llm_model,
        "prompt_hash": prompt_hash,
        "fingerprint": fingerprint,
        "at": time.time(),
    }
    if extra:
        blob.update(extra)
    return {k: v for k, v in blob.items() if v is not None}


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def attach_prov(row: dict, prov: dict) -> dict:
    """Return a new dict = row + {_prov: …, merged with any existing _prov}."""
    out = dict(row)
    existing = out.get(PROV_KEY) or {}
    if not isinstance(existing, dict):
        existing = {}
    merged = {**existing, **prov}
    out[PROV_KEY] = merged
    return out


# ── reproducibility manifest (gemma42.lock) ────────────────────────────────


@dataclass
class LockManifest:
    """A complete, self-contained description of a run.

    Two runs that share the same lock SHOULD produce the same dataset given
    deterministic LLMs (temperature=0, same model). Even with stochastic
    LLMs, the lock is enough to rebuild the run programmatically.
    """

    version: str = "gemma42.lock/1"
    created_at: float = field(default_factory=time.time)
    goal: str = ""
    workdir: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    seed: int | None = None
    contracts: list[dict] = field(default_factory=list)
    extractors: dict[str, dict] = field(default_factory=dict)
    codebook: dict | None = None
    rules: list[dict] = field(default_factory=list)        # constitutional rules
    fingerprints: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)       # listing URLs
    summary: dict = field(default_factory=dict)            # n_rows, etc.

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """A stable hash useful as the lock id."""
        # Hash a canonical sort so reordering doesn't change the id.
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        return short_hash(canonical)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["lock_id"] = self.hash()
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "LockManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("lock_id", None)
        return cls(**data)


def build_manifest_from_state(state: Any, *, llm_provider: str, llm_model: str,
                              codebook_dict: dict | None = None) -> LockManifest:
    """Snapshot a current AgentState into a LockManifest."""
    extractors = state.memory.get("extractors", {}) or {}
    queue = state.memory.get("queue", []) or []
    fingerprints = []
    for q in queue:
        fp = (q or {}).get("_fp") if isinstance(q, dict) else None
        if fp:
            fingerprints.append(fp)
    sources = []
    for q in queue:
        if isinstance(q, dict) and q.get("detail_url"):
            sources.append(q["detail_url"])
    contracts = state.contracts_snapshot()
    rules = state.memory.get("rules", []) or []
    return LockManifest(
        goal=state.goal,
        workdir=str(state.workdir),
        llm_provider=llm_provider,
        llm_model=llm_model,
        contracts=[{"name": c["name"], "description": c.get("description", "")} for c in contracts],
        extractors={k: v for k, v in extractors.items()},
        codebook=codebook_dict,
        rules=rules,
        fingerprints=sorted(set(fingerprints)),
        sources=sorted(set(sources))[:50],
        summary={"n_rows": len(state.dataset)},
    )


# ── semantic dataset diff ──────────────────────────────────────────────────


def diff_datasets(rows_a: list[dict], rows_b: list[dict], *,
                  key: str = "id", ignore: tuple[str, ...] = (PROV_KEY,)) -> dict:
    """Compare two row collections, keyed by `key`.

    Returns a structured diff:
      - added:   ids present in B but not in A
      - removed: ids present in A but not in B
      - changed: list of {id, field_diffs: {field: {before, after}}}
      - new_columns / dropped_columns
    """
    by_a = {str(r.get(key)): r for r in rows_a if r.get(key) is not None}
    by_b = {str(r.get(key)): r for r in rows_b if r.get(key) is not None}
    cols_a = {k for r in rows_a for k in r.keys()} - set(ignore)
    cols_b = {k for r in rows_b for k in r.keys()} - set(ignore)

    added = sorted(set(by_b) - set(by_a))
    removed = sorted(set(by_a) - set(by_b))
    changed: list[dict] = []
    for k in sorted(set(by_a) & set(by_b)):
        a, b = by_a[k], by_b[k]
        fd: dict[str, dict] = {}
        for col in (cols_a | cols_b):
            if col in ignore:
                continue
            va, vb = a.get(col), b.get(col)
            if va != vb:
                fd[col] = {"before": va, "after": vb}
        if fd:
            changed.append({"id": k, "field_diffs": fd})
    return {
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "added": added,
        "removed": removed,
        "changed": changed,
        "new_columns": sorted(cols_b - cols_a),
        "dropped_columns": sorted(cols_a - cols_b),
    }


def format_diff(d: dict, *, limit: int = 10) -> str:
    lines = [
        f"rows: {d['n_a']} → {d['n_b']}",
        f"added:   {len(d['added'])}  removed: {len(d['removed'])}  changed: {len(d['changed'])}",
    ]
    if d["new_columns"]:
        lines.append(f"new columns:    {d['new_columns']}")
    if d["dropped_columns"]:
        lines.append(f"dropped columns: {d['dropped_columns']}")
    if d["added"][:limit]:
        lines.append(f"added ids ({limit}): {d['added'][:limit]}")
    if d["removed"][:limit]:
        lines.append(f"removed ids ({limit}): {d['removed'][:limit]}")
    if d["changed"]:
        lines.append("changed rows (first 5):")
        for c in d["changed"][:5]:
            fields = list(c["field_diffs"].keys())[:6]
            lines.append(f"  {c['id']}: {fields}")
    return "\n".join(lines)
