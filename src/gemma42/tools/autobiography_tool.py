"""Autobiography tools — read/write the 4-level persistent memory.

The agent uses these to:
  - search past episodes for relevant priors before starting a new run
  - save lessons learned at the end of a run
  - look up site-specific recipes from past success
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.autobiography.store import global_db, project_db
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


def _open_dbs(state: "AgentState"):
    return project_db(state.workdir), global_db()


class AutobiographyStatsTool(Tool):
    name = "autobiography_stats"
    description = (
        "Counts of stored knowledge in the agent's autobiography "
        "(project-level + global). Tells you how much prior experience "
        "is available to inform the current task."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        proj, gl = _open_dbs(state)
        out = {"project": proj.stats(), "global": gl.stats()}
        proj.close(); gl.close()
        return ToolResult(output=json.dumps(out, indent=2))


class AutobiographyRecallTool(Tool):
    name = "autobiography_recall"
    description = (
        "Retrieve past lessons, episodes, recipes, and codebooks relevant to "
        "the current task. Pass `keyword` (matches goal/summary/text), "
        "and optionally `domain` to narrow by site. Returns a list of items "
        "sorted by relevance. Run this BEFORE the DISCOVER phase if you "
        "suspect we've scraped a similar site before."
    )
    args_schema = {
        "keyword": {"type": "string", "description": "free-text keyword"},
        "domain":  {"type": "string", "description": "optional domain filter (e.g. 'cnil.fr')"},
        "limit":   {"type": "integer", "default": 5},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        kw = (args.get("keyword") or "").strip()
        dom = (args.get("domain") or "").strip()
        limit = int(args.get("limit") or 5)
        proj, gl = _open_dbs(state)
        try:
            results: dict = {"episodes": [], "lessons": [], "codebooks": [], "sites": []}
            for db_name, db in [("project", proj), ("global", gl)]:
                if kw:
                    for e in db.search_episodes(kw, limit=limit):
                        results["episodes"].append({
                            "scope": db_name, "goal": e.goal[:160],
                            "n_rows": e.n_rows, "status": e.status,
                            "summary": (e.summary or "")[:200],
                        })
                    for l in db.search_lessons(kw, limit=limit):
                        results["lessons"].append({
                            "scope": db_name, "kind": l.kind, "text": l.text[:240],
                        })
                if dom:
                    for s in db.find_sites_by_domain(dom):
                        results["sites"].append({
                            "scope": db_name, "domain": s.domain,
                            "fingerprint": s.fingerprint[:8],
                            "n_runs": s.n_runs, "n_success": s.n_success,
                        })
                for c in db.search_codebooks(domain_hint=dom or None,
                                              keyword=kw or None, limit=limit):
                    results["codebooks"].append({
                        "scope": db_name, "name": c.name,
                        "domain_hint": c.domain_hint,
                        "n_variables": len(c.spec.get("variables", [])),
                        "n_uses": c.n_uses,
                    })
        finally:
            proj.close(); gl.close()
        return ToolResult(output=json.dumps(results, indent=2, ensure_ascii=False))


class AutobiographyLessonTool(Tool):
    name = "autobiography_remember"
    description = (
        "Save a lesson learned to long-term memory. `text` is the lesson "
        "(one short paragraph). `kind` categorises it: 'extraction', "
        "'codebook', 'fingerprint', 'general'. `scope` is 'project' "
        "(default — this workdir only) or 'global' (across all projects). "
        "Call this at the end of a phase when something non-obvious worked."
    )
    args_schema = {
        "text":  {"type": "string"},
        "kind":  {"type": "string", "default": "general"},
        "scope": {"type": "string", "enum": ["project", "global"], "default": "project"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        text = (args.get("text") or "").strip()
        if not text:
            return ToolResult(output="ERROR: 'text' required", error=True)
        scope = args.get("scope", "project")
        kind = args.get("kind", "general")
        db = project_db(state.workdir) if scope == "project" else global_db()
        try:
            lesson = db.add_lesson(kind=kind, text=text)
        finally:
            db.close()
        return ToolResult(output=f"remembered: id={lesson.id} kind={kind} scope={scope}")
