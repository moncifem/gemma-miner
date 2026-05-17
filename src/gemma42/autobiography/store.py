"""SQLite-backed autobiography store.

Two physical stores are created on demand:
  - project: <workdir>/autobiography.db  (this run's project)
  - global:  ~/.gemma42/autobiography.db (across all your projects)

The `Autobiography` class is the typed API. It exposes typed dataclass rows
(Site / Recipe / Codebook / Episode / Lesson / Skill) and keyword retrieval.

Search is intentionally simple (LIKE + scoring) — no embeddings, no model
calls. It works offline and is fast enough for thousands of episodes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    domain        TEXT NOT NULL,
    url_pattern   TEXT,
    fingerprint   TEXT NOT NULL,
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    n_runs        INTEGER DEFAULT 0,
    n_success     INTEGER DEFAULT 0,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sites_fp     ON sites(fingerprint);
CREATE INDEX IF NOT EXISTS idx_sites_domain ON sites(domain);

CREATE TABLE IF NOT EXISTS recipes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id      INTEGER NOT NULL,
    name         TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    confidence   REAL DEFAULT 0.5,
    n_uses       INTEGER DEFAULT 0,
    n_success    INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    FOREIGN KEY (site_id) REFERENCES sites(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site_id);

CREATE TABLE IF NOT EXISTS codebooks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_hint  TEXT,
    name         TEXT NOT NULL,
    spec_json    TEXT NOT NULL,
    n_uses       INTEGER DEFAULT 0,
    avg_coverage REAL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workdir     TEXT NOT NULL,
    goal        TEXT NOT NULL,
    status      TEXT,
    n_rows      INTEGER DEFAULT 0,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    summary     TEXT,
    trace_path  TEXT,
    fingerprints_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);

CREATE TABLE IF NOT EXISTS lessons (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER,
    kind         TEXT,
    text         TEXT NOT NULL,
    confidence   REAL DEFAULT 1.0,
    n_referenced INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_kind ON lessons(kind);

CREATE TABLE IF NOT EXISTS skills (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT UNIQUE NOT NULL,
    body         TEXT NOT NULL,
    description  TEXT,
    n_uses       INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
"""


# ── typed rows ─────────────────────────────────────────────────────────────


@dataclass
class Site:
    id: int | None
    domain: str
    url_pattern: str | None
    fingerprint: str
    first_seen: float
    last_seen: float
    n_runs: int = 0
    n_success: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Recipe:
    id: int | None
    site_id: int
    name: str
    spec: dict
    confidence: float
    n_uses: int
    n_success: int
    created_at: float
    updated_at: float


@dataclass
class Codebook:
    id: int | None
    name: str
    domain_hint: str | None
    spec: dict
    n_uses: int
    avg_coverage: float | None
    created_at: float
    updated_at: float


@dataclass
class Episode:
    id: int | None
    workdir: str
    goal: str
    status: str | None
    n_rows: int
    started_at: float
    ended_at: float | None
    summary: str | None
    trace_path: str | None
    fingerprints: list[str] = field(default_factory=list)


@dataclass
class Lesson:
    id: int | None
    episode_id: int | None
    kind: str
    text: str
    confidence: float
    n_referenced: int
    created_at: float


@dataclass
class Skill:
    id: int | None
    name: str
    body: str
    description: str | None
    n_uses: int
    created_at: float
    updated_at: float


# ── store ──────────────────────────────────────────────────────────────────


class Autobiography:
    """Wraps a single SQLite store (project OR global). Use both for L3 + L4."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ── sites ──────────────────────────────────────────────────────────

    def upsert_site(self, *, domain: str, fingerprint: str,
                    url_pattern: str | None = None,
                    metadata: dict | None = None) -> Site:
        now = time.time()
        cur = self._conn.execute(
            "SELECT * FROM sites WHERE fingerprint = ? AND domain = ? LIMIT 1",
            (fingerprint, domain),
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                """INSERT INTO sites(domain, url_pattern, fingerprint,
                                      first_seen, last_seen, n_runs, metadata_json)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (domain, url_pattern, fingerprint, now, now,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
            cur = self._conn.execute(
                "SELECT * FROM sites WHERE fingerprint = ? AND domain = ? LIMIT 1",
                (fingerprint, domain),
            )
            row = cur.fetchone()
        else:
            self._conn.execute(
                "UPDATE sites SET last_seen = ?, url_pattern = COALESCE(?, url_pattern) WHERE id = ?",
                (now, url_pattern, row["id"]),
            )
        return _row_to_site(row)

    def bump_site(self, site_id: int, *, success: bool) -> None:
        if success:
            self._conn.execute(
                "UPDATE sites SET n_runs = n_runs + 1, n_success = n_success + 1 WHERE id = ?",
                (site_id,),
            )
        else:
            self._conn.execute(
                "UPDATE sites SET n_runs = n_runs + 1 WHERE id = ?", (site_id,),
            )

    def find_sites_by_fingerprint(self, fingerprint: str,
                                  *, near_prefix_chars: int = 8) -> list[Site]:
        """Exact or near (prefix) match."""
        rows = self._conn.execute(
            "SELECT * FROM sites WHERE fingerprint = ? ORDER BY last_seen DESC",
            (fingerprint,),
        ).fetchall()
        out = [_row_to_site(r) for r in rows]
        if out:
            return out
        prefix = fingerprint[:near_prefix_chars]
        rows = self._conn.execute(
            "SELECT * FROM sites WHERE fingerprint LIKE ? ORDER BY last_seen DESC LIMIT 5",
            (prefix + "%",),
        ).fetchall()
        return [_row_to_site(r) for r in rows]

    def find_sites_by_domain(self, domain: str) -> list[Site]:
        rows = self._conn.execute(
            "SELECT * FROM sites WHERE domain = ? ORDER BY last_seen DESC",
            (domain,),
        ).fetchall()
        return [_row_to_site(r) for r in rows]

    # ── recipes ────────────────────────────────────────────────────────

    def upsert_recipe(self, *, site_id: int, name: str, spec: dict,
                       confidence: float = 0.7) -> Recipe:
        now = time.time()
        row = self._conn.execute(
            "SELECT * FROM recipes WHERE site_id = ? AND name = ? LIMIT 1",
            (site_id, name),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """INSERT INTO recipes(site_id, name, spec_json, confidence,
                                       n_uses, n_success, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
                (site_id, name, json.dumps(spec, ensure_ascii=False),
                 confidence, now, now),
            )
        else:
            self._conn.execute(
                """UPDATE recipes SET spec_json = ?, confidence = ?, updated_at = ?
                   WHERE id = ?""",
                (json.dumps(spec, ensure_ascii=False), confidence, now, row["id"]),
            )
        row = self._conn.execute(
            "SELECT * FROM recipes WHERE site_id = ? AND name = ? LIMIT 1",
            (site_id, name),
        ).fetchone()
        return _row_to_recipe(row)

    def get_recipes(self, site_id: int) -> list[Recipe]:
        rows = self._conn.execute(
            "SELECT * FROM recipes WHERE site_id = ? ORDER BY confidence DESC, n_success DESC",
            (site_id,),
        ).fetchall()
        return [_row_to_recipe(r) for r in rows]

    def bump_recipe(self, recipe_id: int, *, success: bool) -> None:
        # Bayesian-ish update on confidence.
        row = self._conn.execute(
            "SELECT n_uses, n_success, confidence FROM recipes WHERE id = ?",
            (recipe_id,),
        ).fetchone()
        if row is None:
            return
        n_uses = row["n_uses"] + 1
        n_success = row["n_success"] + (1 if success else 0)
        # Beta(α=2, β=2) prior; α = n_success + 2, β = n_uses - n_success + 2
        a = n_success + 2
        b = n_uses - n_success + 2
        confidence = a / (a + b)
        self._conn.execute(
            "UPDATE recipes SET n_uses = ?, n_success = ?, confidence = ?, updated_at = ? WHERE id = ?",
            (n_uses, n_success, confidence, time.time(), recipe_id),
        )

    # ── codebooks ──────────────────────────────────────────────────────

    def save_codebook(self, *, name: str, spec: dict, domain_hint: str | None = None,
                       avg_coverage: float | None = None) -> Codebook:
        now = time.time()
        row = self._conn.execute(
            "SELECT * FROM codebooks WHERE name = ? LIMIT 1", (name,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """INSERT INTO codebooks(domain_hint, name, spec_json, n_uses,
                                         avg_coverage, created_at, updated_at)
                   VALUES (?, ?, ?, 0, ?, ?, ?)""",
                (domain_hint, name, json.dumps(spec, ensure_ascii=False),
                 avg_coverage, now, now),
            )
        else:
            self._conn.execute(
                """UPDATE codebooks SET spec_json = ?, domain_hint = COALESCE(?, domain_hint),
                                          avg_coverage = COALESCE(?, avg_coverage),
                                          updated_at = ?
                   WHERE id = ?""",
                (json.dumps(spec, ensure_ascii=False), domain_hint,
                 avg_coverage, now, row["id"]),
            )
        row = self._conn.execute(
            "SELECT * FROM codebooks WHERE name = ? LIMIT 1", (name,),
        ).fetchone()
        return _row_to_codebook(row)

    def search_codebooks(self, domain_hint: str | None = None,
                         keyword: str | None = None,
                         limit: int = 5) -> list[Codebook]:
        q = "SELECT * FROM codebooks WHERE 1=1"
        params: list = []
        if domain_hint:
            q += " AND (domain_hint LIKE ? OR name LIKE ?)"
            params += [f"%{domain_hint}%", f"%{domain_hint}%"]
        if keyword:
            q += " AND (spec_json LIKE ? OR name LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]
        q += " ORDER BY n_uses DESC, updated_at DESC LIMIT ?"
        params += [limit]
        rows = self._conn.execute(q, params).fetchall()
        return [_row_to_codebook(r) for r in rows]

    # ── episodes ───────────────────────────────────────────────────────

    def start_episode(self, workdir: str, goal: str,
                      trace_path: str | None = None) -> Episode:
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO episodes(workdir, goal, status, n_rows, started_at,
                                    trace_path, fingerprints_json)
               VALUES (?, ?, 'running', 0, ?, ?, '[]')""",
            (workdir, goal, now, trace_path),
        )
        row = self._conn.execute("SELECT * FROM episodes WHERE id = ?",
                                  (cur.lastrowid,)).fetchone()
        return _row_to_episode(row)

    def finish_episode(self, episode_id: int, *, status: str, n_rows: int,
                       summary: str | None = None,
                       fingerprints: list[str] | None = None) -> None:
        self._conn.execute(
            """UPDATE episodes SET status = ?, n_rows = ?, ended_at = ?,
                                    summary = COALESCE(?, summary),
                                    fingerprints_json = COALESCE(?, fingerprints_json)
               WHERE id = ?""",
            (status, n_rows, time.time(), summary,
             json.dumps(fingerprints or [], ensure_ascii=False), episode_id),
        )

    def recent_episodes(self, limit: int = 10) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def search_episodes(self, keyword: str, limit: int = 5) -> list[Episode]:
        rows = self._conn.execute(
            """SELECT * FROM episodes
               WHERE goal LIKE ? OR summary LIKE ?
               ORDER BY started_at DESC LIMIT ?""",
            (f"%{keyword}%", f"%{keyword}%", limit),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    # ── lessons ────────────────────────────────────────────────────────

    def add_lesson(self, *, kind: str, text: str, episode_id: int | None = None,
                   confidence: float = 1.0) -> Lesson:
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO lessons(episode_id, kind, text, confidence,
                                    n_referenced, created_at)
               VALUES (?, ?, ?, ?, 0, ?)""",
            (episode_id, kind, text, confidence, now),
        )
        row = self._conn.execute("SELECT * FROM lessons WHERE id = ?",
                                  (cur.lastrowid,)).fetchone()
        return _row_to_lesson(row)

    def search_lessons(self, keyword: str, *, kind: str | None = None,
                       limit: int = 6) -> list[Lesson]:
        q = "SELECT * FROM lessons WHERE text LIKE ?"
        params: list = [f"%{keyword}%"]
        if kind:
            q += " AND kind = ?"
            params.append(kind)
        q += " ORDER BY n_referenced DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [_row_to_lesson(r) for r in rows]

    def bump_lesson(self, lesson_id: int) -> None:
        self._conn.execute(
            "UPDATE lessons SET n_referenced = n_referenced + 1 WHERE id = ?",
            (lesson_id,),
        )

    # ── skills ─────────────────────────────────────────────────────────

    def save_skill(self, *, name: str, body: str, description: str | None) -> Skill:
        now = time.time()
        row = self._conn.execute(
            "SELECT * FROM skills WHERE name = ? LIMIT 1", (name,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """INSERT INTO skills(name, body, description, n_uses,
                                       created_at, updated_at)
                   VALUES (?, ?, ?, 0, ?, ?)""",
                (name, body, description, now, now),
            )
        else:
            self._conn.execute(
                """UPDATE skills SET body = ?, description = COALESCE(?, description),
                                       updated_at = ?
                   WHERE id = ?""",
                (body, description, now, row["id"]),
            )
        row = self._conn.execute(
            "SELECT * FROM skills WHERE name = ? LIMIT 1", (name,),
        ).fetchone()
        return _row_to_skill(row)

    def list_skills(self) -> list[Skill]:
        rows = self._conn.execute(
            "SELECT * FROM skills ORDER BY n_uses DESC, updated_at DESC",
        ).fetchall()
        return [_row_to_skill(r) for r in rows]

    def bump_skill(self, name: str) -> None:
        self._conn.execute(
            "UPDATE skills SET n_uses = n_uses + 1 WHERE name = ?", (name,),
        )

    # ── stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        c = self._conn
        return {
            "sites":     c.execute("SELECT COUNT(*) FROM sites").fetchone()[0],
            "recipes":   c.execute("SELECT COUNT(*) FROM recipes").fetchone()[0],
            "codebooks": c.execute("SELECT COUNT(*) FROM codebooks").fetchone()[0],
            "episodes":  c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "lessons":   c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
            "skills":    c.execute("SELECT COUNT(*) FROM skills").fetchone()[0],
        }


# ── row → dataclass helpers ────────────────────────────────────────────────


def _row_to_site(row) -> Site:
    return Site(
        id=row["id"], domain=row["domain"], url_pattern=row["url_pattern"],
        fingerprint=row["fingerprint"], first_seen=row["first_seen"],
        last_seen=row["last_seen"], n_runs=row["n_runs"],
        n_success=row["n_success"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def _row_to_recipe(row) -> Recipe:
    return Recipe(
        id=row["id"], site_id=row["site_id"], name=row["name"],
        spec=json.loads(row["spec_json"]), confidence=row["confidence"],
        n_uses=row["n_uses"], n_success=row["n_success"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _row_to_codebook(row) -> Codebook:
    return Codebook(
        id=row["id"], name=row["name"], domain_hint=row["domain_hint"],
        spec=json.loads(row["spec_json"]), n_uses=row["n_uses"],
        avg_coverage=row["avg_coverage"], created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_episode(row) -> Episode:
    return Episode(
        id=row["id"], workdir=row["workdir"], goal=row["goal"],
        status=row["status"], n_rows=row["n_rows"] or 0,
        started_at=row["started_at"], ended_at=row["ended_at"],
        summary=row["summary"], trace_path=row["trace_path"],
        fingerprints=json.loads(row["fingerprints_json"] or "[]"),
    )


def _row_to_lesson(row) -> Lesson:
    return Lesson(
        id=row["id"], episode_id=row["episode_id"], kind=row["kind"],
        text=row["text"], confidence=row["confidence"],
        n_referenced=row["n_referenced"], created_at=row["created_at"],
    )


def _row_to_skill(row) -> Skill:
    return Skill(
        id=row["id"], name=row["name"], body=row["body"],
        description=row["description"], n_uses=row["n_uses"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


# ── L3 / L4 convenience ────────────────────────────────────────────────────


def project_db(workdir: str | Path) -> Autobiography:
    return Autobiography(Path(workdir) / "autobiography.db")


def global_db() -> Autobiography:
    home = Path(os.path.expanduser("~")) / ".gemma42"
    return Autobiography(home / "autobiography.db")
