"""Federated recipe cloud (skeleton).

This is the L5 layer of the autobiography hierarchy — an opt-in public
registry where users share extractor recipes and codebooks (NEVER raw
data). The API mirrors the local autobiography: search · push · pull.

The HTTP backend is intentionally pluggable. The default `CloudClient`
talks to a JSON-over-HTTPS endpoint specified by `GEMMA42_CLOUD_URL`. A
sensible default endpoint (gemma42.dev/registry) ships later; for now
this layer is the scaffolding so individual users can self-host a tiny
recipe server.

Schema of a published recipe (intentionally minimal, no PII):
  {
    "id": "<sha256:16 of payload>",
    "kind": "recipe" | "codebook",
    "domain": "cnil.fr",
    "fingerprint": "<32hex>",
    "name": "listing" | "detail" | <codebook_name>,
    "spec": {...},
    "stats": {"n_uses": int, "avg_coverage": float | null,
              "n_success": int, "n_failure": int},
    "license": "CC0-1.0" | "MIT" | "...",
    "publisher": "<opaque user id>",
    "created_at": <epoch seconds>,
  }
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from gemma42.fingerprint import looks_similar


DEFAULT_CLOUD_URL_ENV = "GEMMA42_CLOUD_URL"
DEFAULT_USER_ID_FILE = Path(os.path.expanduser("~")) / ".gemma42" / "user_id"
LOCAL_CACHE_DIR = Path(os.path.expanduser("~")) / ".gemma42" / "cloud_cache"


def _ensure_user_id() -> str:
    DEFAULT_USER_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_USER_ID_FILE.exists():
        return DEFAULT_USER_ID_FILE.read_text().strip()
    import secrets

    uid = secrets.token_hex(8)
    DEFAULT_USER_ID_FILE.write_text(uid)
    return uid


@dataclass
class CloudArtifact:
    id: str
    kind: str           # "recipe" | "codebook"
    domain: str
    fingerprint: str
    name: str
    spec: dict
    stats: dict
    license: str = "CC0-1.0"
    publisher: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CloudArtifact":
        return cls(**d)


def make_artifact_id(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CloudClient:
    """Talks to a remote recipe registry. Falls back to a local cache if
    the cloud is unreachable or no URL is configured."""

    def __init__(self, base_url: str | None = None, *, user_id: str | None = None,
                 timeout: float = 8.0):
        self.base_url = (base_url or os.getenv(DEFAULT_CLOUD_URL_ENV) or "").rstrip("/")
        self.user_id = user_id or _ensure_user_id()
        self.timeout = timeout
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── local cache (mirrors what we've pulled) ───────────────────────

    def _cache_path(self, artifact_id: str) -> Path:
        return LOCAL_CACHE_DIR / f"{artifact_id}.json"

    def cached(self, artifact_id: str) -> CloudArtifact | None:
        p = self._cache_path(artifact_id)
        if not p.exists():
            return None
        try:
            return CloudArtifact.from_dict(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            return None

    def list_local_cache(self) -> list[CloudArtifact]:
        out: list[CloudArtifact] = []
        for p in LOCAL_CACHE_DIR.glob("*.json"):
            try:
                out.append(CloudArtifact.from_dict(json.loads(p.read_text())))
            except Exception:  # noqa: BLE001
                continue
        return out

    # ── HTTP I/O ──────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def push(self, art: CloudArtifact) -> dict:
        payload = art.to_dict()
        payload["publisher"] = self.user_id
        if not art.id:
            payload["id"] = make_artifact_id(
                {k: v for k, v in payload.items() if k not in ("id", "created_at")}
            )
        # Cache locally first — this works even if there's no cloud URL.
        self._cache_path(payload["id"]).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False))
        if not self.is_configured():
            return {"status": "cached_local", "id": payload["id"]}
        try:
            r = httpx.post(
                f"{self.base_url}/v1/artifacts",
                json=payload, timeout=self.timeout,
                headers={"X-Gemma42-User": self.user_id},
            )
            return {"status": r.status_code, "id": payload["id"], "body": r.text[:400]}
        except Exception as e:  # noqa: BLE001
            return {"status": "offline", "id": payload["id"], "error": str(e)}

    def search(self, *, domain: str | None = None, fingerprint: str | None = None,
               keyword: str | None = None, kind: str | None = None,
               limit: int = 10) -> list[CloudArtifact]:
        # 1. Try remote.
        if self.is_configured():
            try:
                params = {k: v for k, v in {
                    "domain": domain, "fingerprint": fingerprint,
                    "q": keyword, "kind": kind, "limit": str(limit),
                }.items() if v is not None}
                r = httpx.get(f"{self.base_url}/v1/artifacts",
                              params=params, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    items = data.get("items") if isinstance(data, dict) else data
                    out = []
                    for d in (items or [])[:limit]:
                        try:
                            art = CloudArtifact.from_dict(d)
                            self._cache_path(art.id).write_text(
                                json.dumps(art.to_dict(), indent=2, ensure_ascii=False))
                            out.append(art)
                        except Exception:  # noqa: BLE001
                            continue
                    if out:
                        return out
            except Exception:  # noqa: BLE001
                pass
        # 2. Fall back to local cache (works offline).
        items = self.list_local_cache()
        if kind:
            items = [a for a in items if a.kind == kind]
        if domain:
            items = [a for a in items if a.domain == domain]
        if fingerprint:
            items = [a for a in items if looks_similar(a.fingerprint, fingerprint)]
        if keyword:
            kw = keyword.lower()
            items = [a for a in items if kw in (a.name + a.domain).lower()]
        return items[:limit]

    def pull(self, artifact_id: str) -> CloudArtifact | None:
        # local cache first
        c = self.cached(artifact_id)
        if c is not None:
            return c
        if not self.is_configured():
            return None
        try:
            r = httpx.get(f"{self.base_url}/v1/artifacts/{artifact_id}",
                          timeout=self.timeout)
            if r.status_code == 200:
                art = CloudArtifact.from_dict(r.json())
                self._cache_path(art.id).write_text(
                    json.dumps(art.to_dict(), indent=2, ensure_ascii=False))
                return art
        except Exception:  # noqa: BLE001
            pass
        return None
