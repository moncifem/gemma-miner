"""Fingerprint + recipe-cache tools.

These two tools are how the agent SKIPS the DISCOVER phase when it has
seen a similar site before. They:

  - fingerprint_check(url)  → fingerprint + nearest matches + cached recipes
  - recipe_save(domain, fp, name, spec) → store a successful extractor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma42.autobiography.store import global_db, project_db
from gemma42.fingerprint import (
    fingerprint_html,
    fingerprint_url_and_html,
    looks_similar,
)
from gemma42.tools.base import Tool, ToolResult
from gemma42.tools.http_tool import _slug as _url_slug

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _read_cached_html(state: "AgentState", url: str) -> tuple[Path | None, str | None]:
    cache = Path(state.workdir) / "cache"
    if not cache.exists():
        return None, None
    slug = _url_slug(url)
    for ext in (".html", ".htm", ".xml", ".bin"):
        p = cache / f"{slug}{ext}"
        if p.exists():
            try:
                return p, p.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
    # any file starting with slug
    for f in cache.glob(f"{slug}.*"):
        try:
            return f, f.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
    return None, None


class FingerprintCheckTool(Tool):
    name = "fingerprint_check"
    description = (
        "Compute a structural fingerprint of a cached HTML page and look up "
        "the autobiography (project + global) for previously-successful "
        "extractors that match (exact OR near-match). If found, you can "
        "load the cached recipe with `recipe_load` and SKIP the entire "
        "DISCOVER phase. Pass either `url` (uses the cache for that URL) "
        "or `source_path` (an absolute path to a cached HTML file).\n\n"
        "Result includes: domain, fingerprint, exact matches, near matches, "
        "and any cached recipes you can adopt immediately."
    )
    args_schema = {
        "url":         {"type": "string", "description": "URL whose cached HTML to fingerprint"},
        "source_path": {"type": "string", "description": "Absolute path to a cached HTML file"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        url = args.get("url") or ""
        src = args.get("source_path") or ""
        html: str | None = None
        if src:
            p = Path(src)
            if p.exists():
                html = p.read_text(encoding="utf-8", errors="replace")
        if html is None and url:
            _, html = _read_cached_html(state, url)
        if html is None:
            return ToolResult(
                output=("ERROR: no source available. Call http_get(url=...) first, "
                        "then call fingerprint_check(url=...) — or pass source_path."),
                error=True,
            )
        domain, fp = fingerprint_url_and_html(url, html)

        proj, gl = project_db(state.workdir), global_db()
        out: dict = {"domain": domain, "fingerprint": fp,
                     "exact_matches": [], "near_matches": [], "cached_recipes": []}
        try:
            for db_name, db in [("project", proj), ("global", gl)]:
                exacts = db.find_sites_by_fingerprint(fp)
                for s in exacts:
                    out["exact_matches"].append({
                        "scope": db_name, "domain": s.domain,
                        "n_runs": s.n_runs, "n_success": s.n_success,
                        "last_seen": s.last_seen,
                    })
                    for r in db.get_recipes(s.id):
                        out["cached_recipes"].append({
                            "scope": db_name, "site_domain": s.domain,
                            "name": r.name, "confidence": r.confidence,
                            "n_uses": r.n_uses, "n_success": r.n_success,
                            "spec": r.spec,
                        })
                # near match (different fingerprint, same domain)
                if not exacts and domain:
                    for s in db.find_sites_by_domain(domain):
                        if looks_similar(s.fingerprint, fp):
                            out["near_matches"].append({
                                "scope": db_name, "domain": s.domain,
                                "fingerprint": s.fingerprint, "n_success": s.n_success,
                            })
                            for r in db.get_recipes(s.id):
                                out["cached_recipes"].append({
                                    "scope": db_name, "site_domain": s.domain,
                                    "name": r.name, "confidence": r.confidence,
                                    "match": "near", "spec": r.spec,
                                })
        finally:
            proj.close(); gl.close()
        return ToolResult(output=json.dumps(out, indent=2, ensure_ascii=False),
                          artifact={"fingerprint": fp, "domain": domain,
                                    "has_recipes": bool(out["cached_recipes"])})


class RecipeSaveTool(Tool):
    name = "recipe_save"
    description = (
        "Persist a successful extractor spec to the autobiography for future "
        "runs. Call this AFTER `extractor_define` produced a working spec "
        "and `scrape_paginated` (or similar) confirmed rows. Pass:\n"
        "  - url: the listing or detail URL the extractor targets\n"
        "  - name: which extractor ('listing' or 'detail')\n"
        "  - spec: the spec dict (same shape extractor_define accepts)\n"
        "  - scope: 'project' (default) or 'global' (share across all projects)\n"
        "The recipe is keyed by (domain, fingerprint, name)."
    )
    args_schema = {
        "url":   {"type": "string"},
        "name":  {"type": "string", "enum": ["listing", "detail"]},
        "spec":  {"type": "object"},
        "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        url = args.get("url", "")
        name = args.get("name") or ""
        spec = args.get("spec")
        if not url or not name or not isinstance(spec, dict):
            return ToolResult(output="ERROR: 'url', 'name' (listing|detail), and 'spec' (object) required",
                              error=True)
        _, html = _read_cached_html(state, url)
        if html is None:
            return ToolResult(
                output="ERROR: no cached body for this url. Run http_get first.",
                error=True,
            )
        domain, fp = fingerprint_url_and_html(url, html)
        scope = args.get("scope", "project")
        db = project_db(state.workdir) if scope == "project" else global_db()
        try:
            site = db.upsert_site(domain=domain, fingerprint=fp, url_pattern=url)
            recipe = db.upsert_recipe(site_id=site.id, name=name, spec=spec, confidence=0.7)
        finally:
            db.close()
        return ToolResult(
            output=(
                f"saved recipe '{name}' for {domain} ({scope})\n"
                f"  fingerprint={fp[:12]}…  id={recipe.id}  confidence={recipe.confidence:.2f}"
            )
        )
