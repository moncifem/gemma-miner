"""set_plan — record the run's harvest plan after initial discovery.

The agent calls this once, right after the first 1-3 recon turns, to commit
to a single harvest strategy. The plan is stored under `state.memory["plan"]`
and surfaced in every subsequent state brief, so the agent (and the user)
can see what was decided and whether the run is sticking to it.

Why a dedicated tool, not just memory_set:
  - schema validation: required fields are checked at the boundary
  - the brief renders it specially as "Plan"
  - the planner does the math up front (items_per_page × pages_needed = target)
  - one source of truth — discourages silent mid-run source switches
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


_REQUIRED_KEYS = ("item", "source", "harvest_strategy")
_VALID_SOURCES = ("listing_html", "paginated_html", "api_json", "feed",
                   "sitemap", "archive", "custom")
_VALID_STRATEGIES = (
    "listing_only",      # listing has every required field
    "listing+detail",    # detail page adds more HTML fields
    "listing+attachment", # detail page has one primary attachment
    "listing+multi_asset", # detail page has varied attachments
    "api_only",          # JSON API call, no HTML
    "custom_python",     # custom Python script, no extractor spec
)


class SetPlanTool(Tool):
    name = "set_plan"
    description = (
        "Record the harvest plan for this run. Call this ONCE, right after "
        "initial discovery (typically turn 2-3, after you've fetched the "
        "listing page and looked at one detail page). The plan is then "
        "surfaced in every state brief so future-you sticks to it.\n\n"
        "Plan fields (all required unless noted):\n"
        "  item              : short noun describing one row (e.g. 'Hacker News story',\n"
        "                       'CNIL sanction decision', 'arXiv preprint').\n"
        "  source            : one of listing_html | paginated_html | api_json |\n"
        "                       feed | sitemap | archive | custom.\n"
        "  source_url        : canonical listing URL or API endpoint.\n"
        "  pagination        : how to advance — '?page={page}', '?p={page}',\n"
        "                       'cursor=<token>', 'api-offset=<n>', 'none', etc.\n"
        "                       Use null when the source returns everything in one call.\n"
        "  items_per_page    : integer, what you observed on page 1 (or `target` if API).\n"
        "  target_rows       : the contract's min_rows.\n"
        "  pages_needed      : ceil(target_rows / items_per_page). Don't lie to yourself —\n"
        "                       if you observed 30/page and need 1000, that's ~34 pages.\n"
        "  harvest_strategy  : one of listing_only | listing+detail | listing+attachment |\n"
        "                       listing+multi_asset | api_only | custom_python.\n"
        "  fields            : list of {source_field, dataset_field, type, notes} —\n"
        "                       the mapping you'll use for every row. THIS IS THE\n"
        "                       SCHEMA the dataset must converge on. Don't change\n"
        "                       it mid-run; if reality forces a change, call set_plan\n"
        "                       again and explicitly state what changed in `notes`.\n"
        "  risks             : optional list of strings — anticipated failure modes\n"
        "                       (rate limits, JS rendering, mixed shapes, …).\n"
        "  notes             : free text.\n\n"
        "Validation: the tool rejects plans missing required fields or with\n"
        "math that doesn't add up (e.g. items_per_page=30, target_rows=1000,\n"
        "pages_needed=2). Fix the numbers before you commit."
    )
    args_schema = {
        "item":             {"type": "string"},
        "source":           {"type": "string"},
        "source_url":       {"type": "string"},
        "pagination":       {"type": "string"},
        "items_per_page":   {"type": "integer"},
        "target_rows":      {"type": "integer"},
        "pages_needed":     {"type": "integer"},
        "harvest_strategy": {"type": "string"},
        "fields":           {"type": "array"},
        "risks":            {"type": "array"},
        "notes":            {"type": "string"},
    }

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        plan: dict = {}
        # Accept either a flat plan or a wrapped `{plan: {...}}` shape.
        if "plan" in args and isinstance(args["plan"], dict):
            plan = dict(args["plan"])
        else:
            plan = {k: v for k, v in args.items() if v is not None}

        missing = [k for k in _REQUIRED_KEYS if k not in plan]
        if missing:
            return ToolResult(
                output=f"ERROR: plan is missing required field(s): {missing}. "
                       f"Required: {list(_REQUIRED_KEYS)}.",
                error=True,
            )
        source = str(plan["source"]).lower().strip()
        if source not in _VALID_SOURCES:
            return ToolResult(
                output=(
                    f"ERROR: source='{plan['source']}' is not valid. "
                    f"Pick one of: {list(_VALID_SOURCES)}."
                ),
                error=True,
            )
        plan["source"] = source

        strategy = str(plan["harvest_strategy"]).lower().strip()
        if strategy not in _VALID_STRATEGIES:
            return ToolResult(
                output=(
                    f"ERROR: harvest_strategy='{plan['harvest_strategy']}' "
                    f"is not valid. Pick one of: {list(_VALID_STRATEGIES)}."
                ),
                error=True,
            )
        plan["harvest_strategy"] = strategy

        # Sanity-check the math, but only when the numbers are present.
        ipp = plan.get("items_per_page")
        target = plan.get("target_rows")
        pages = plan.get("pages_needed")
        if isinstance(ipp, int) and isinstance(target, int) and ipp > 0:
            expected_pages = -(-target // ipp)  # ceil division
            if isinstance(pages, int):
                if pages < expected_pages - 1 or pages > expected_pages * 2:
                    return ToolResult(
                        output=(
                            f"ERROR: math doesn't add up. items_per_page={ipp}, "
                            f"target_rows={target} → pages_needed should be ~{expected_pages}, "
                            f"you wrote {pages}. Re-do the arithmetic and resubmit."
                        ),
                        error=True,
                    )
            else:
                plan["pages_needed"] = expected_pages

        # Persist.
        state.memory.set("plan", plan)

        # Friendly summary.
        lines = [f"plan saved → memory['plan']"]
        lines.append(f"  item:             {plan.get('item')}")
        lines.append(f"  source:           {plan.get('source')}  ({plan.get('source_url')})")
        lines.append(f"  pagination:       {plan.get('pagination')}")
        lines.append(
            f"  math:             {plan.get('items_per_page')}/page × "
            f"{plan.get('pages_needed')} pages → {plan.get('target_rows')} target rows"
        )
        lines.append(f"  strategy:         {plan.get('harvest_strategy')}")
        fields = plan.get("fields") or []
        if fields:
            lines.append(f"  fields ({len(fields)}):")
            for f in fields[:8]:
                if isinstance(f, dict):
                    line = "    - "
                    if f.get("dataset_field"):
                        line += f"{f['dataset_field']}"
                    if f.get("source_field"):
                        line += f"  ← {f['source_field']}"
                    if f.get("type"):
                        line += f"  ({f['type']})"
                    lines.append(line)
                else:
                    lines.append(f"    - {f}")
            if len(fields) > 8:
                lines.append(f"    … and {len(fields) - 8} more")
        risks = plan.get("risks") or []
        if risks:
            lines.append("  risks:")
            for r in risks:
                lines.append(f"    ⚠ {r}")
        if plan.get("notes"):
            lines.append(f"  notes: {plan['notes']}")

        return ToolResult(output="\n".join(lines), artifact={"plan": plan})


class ShowPlanTool(Tool):
    name = "show_plan"
    description = (
        "Print the currently-saved plan (or 'no plan yet' if set_plan hasn't "
        "been called)."
    )
    args_schema = {}

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        plan = state.memory.get("plan") or {}
        if not plan:
            return ToolResult(output="no plan yet — call set_plan after initial discovery.")
        return ToolResult(
            output="current plan:\n" + json.dumps(plan, indent=2, ensure_ascii=False),
            artifact={"plan": plan},
        )
