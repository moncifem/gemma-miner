"""Goal Decomposition Tree — visible progress against the user's actual goal.

The user types: "scrape 1000 articles from arxiv and make a stats dataset".
We parse this into a tree of measurable success criteria. The tree is
visible in the state brief at all times; each leaf is auto-checked from
state so the agent can never lie about progress.

Node states: ☐ pending · ◐ in_progress · ✓ done · ✗ blocked
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ── primitives ──────────────────────────────────────────────────────────────


@dataclass
class GDTNode:
    name: str
    description: str
    check: Callable[[Any], tuple[bool, str]] | None = None   # (ok, detail)
    children: list["GDTNode"] = field(default_factory=list)
    # State is *computed* from `check`, never stored.

    def evaluate(self, state: Any) -> dict:
        if self.children:
            child_results = [c.evaluate(state) for c in self.children]
            all_done = all(c["status"] == "done" for c in child_results)
            any_progress = any(c["status"] in ("done", "in_progress") for c in child_results)
            status = "done" if all_done else ("in_progress" if any_progress else "pending")
            return {
                "name": self.name,
                "description": self.description,
                "status": status,
                "detail": f"{sum(1 for c in child_results if c['status']=='done')}/{len(child_results)}",
                "children": child_results,
            }
        if self.check is None:
            return {"name": self.name, "description": self.description,
                    "status": "pending", "detail": "no check", "children": []}
        ok, detail = self.check(state)
        return {
            "name": self.name, "description": self.description,
            "status": "done" if ok else "pending",
            "detail": detail, "children": [],
        }


# ── parse a user goal into a tree ───────────────────────────────────────────


_COUNT_RE = re.compile(r"\b(\d{2,7})\b", re.IGNORECASE)
_NUMERIC_VERBS = (
    "scrape", "extract", "harvest", "build", "collect", "gather",
    "fetch", "make", "get me", "give me",
)


def _looks_like_with_details(goal: str) -> bool:
    return bool(re.search(
        r"\b(detail|details?|each item|per item|for each|with their|"
        r"download|with documents|with pdfs?|with attachments?)\b",
        goal, re.IGNORECASE,
    ))


def _looks_like_with_stats(goal: str) -> bool:
    return bool(re.search(
        r"\b(stat|statistics?|analysis|dataset for stats?|"
        r"ml\b|machine learning|variables?|features?|parquet|huggingface|hf\b)\b",
        goal, re.IGNORECASE,
    ))


def _parsed_count(goal: str) -> int | None:
    for m in _COUNT_RE.finditer(goal):
        n = int(m.group(1))
        if 5 <= n <= 1_000_000:
            return n
    return None


def build_tree_for_goal(goal: str, *, contract_min_rows: int | None = None) -> GDTNode:
    """Parse a free-form goal into a checkable tree."""
    target = contract_min_rows or _parsed_count(goal) or 50
    with_details = _looks_like_with_details(goal)
    with_stats = _looks_like_with_stats(goal)

    # ── leaf checks ────────────────────────────────────────────────────────

    def has_listing_extractor(state) -> tuple[bool, str]:
        ex = state.memory.get("extractors", {}) or {}
        ok = any(v.get("row_pattern") for v in ex.values())
        return ok, "listing extractor defined" if ok else "no listing extractor yet"

    def queue_filled(state) -> tuple[bool, str]:
        q = state.memory.get("queue", []) or []
        processed = set(str(x) for x in (state.memory.get("processed", []) or []))
        remaining = sum(1 for i in q if isinstance(i, dict) and str(i.get("id")) not in processed)
        ok = (len(q) >= target) or (remaining + len(state.dataset) >= target)
        return ok, f"queue has {len(q)} ({remaining} remaining)"

    def harvested_target(state) -> tuple[bool, str]:
        n = len(state.dataset)
        return n >= target, f"{n} / {target} items harvested"

    def has_detail_extractor(state) -> tuple[bool, str]:
        ex = state.memory.get("extractors", {}) or {}
        ok = any(not v.get("row_pattern") for v in ex.values())
        return ok, "detail extractor defined" if ok else "no detail extractor yet"

    def codebook_designed(state) -> tuple[bool, str]:
        from pathlib import Path

        cb = Path(state.workdir) / "codebook.json"
        if not cb.exists():
            return False, "no codebook yet"
        try:
            import json

            data = json.loads(cb.read_text(encoding="utf-8"))
            n = len(data.get("variables") or [])
            ok = n >= 20
            return ok, f"{n} variables"
        except Exception as e:  # noqa: BLE001
            return False, f"codebook unreadable: {e}"

    def codebook_extracted(state) -> tuple[bool, str]:
        from pathlib import Path
        import json

        cb = Path(state.workdir) / "codebook.json"
        if not cb.exists():
            return False, "no codebook"
        try:
            data = json.loads(cb.read_text(encoding="utf-8"))
            var_names = [v["name"] for v in (data.get("variables") or [])]
        except Exception:  # noqa: BLE001
            return False, "codebook unreadable"
        rows = state.dataset.rows()
        if not rows:
            return False, "no rows"
        populated = sum(
            1 for r in rows
            if any(r.get(n) is not None for n in var_names)
        )
        pct = populated / len(rows)
        return pct >= 0.8, f"{populated}/{len(rows)} rows have codebook fields ({pct:.0%})"

    def parquet_exported(state) -> tuple[bool, str]:
        from pathlib import Path

        export_dir = Path(state.workdir) / "export"
        if not export_dir.exists():
            return False, "no export/"
        pqs = list(export_dir.glob("*.parquet"))
        return bool(pqs), f"{len(pqs)} parquet file(s)"

    # ── tree ──────────────────────────────────────────────────────────────

    nodes: list[GDTNode] = []

    corpus_children: list[GDTNode] = [
        GDTNode("listing", "listing extractor identifies items", check=has_listing_extractor),
        GDTNode("queue", f"≥ {target} items queued", check=queue_filled),
        GDTNode("harvest", f"{target} items harvested with text", check=harvested_target),
    ]
    if with_details:
        corpus_children.insert(2, GDTNode("detail", "detail extractor finds attachments",
                                           check=has_detail_extractor))
    nodes.append(GDTNode("corpus", "items collected", children=corpus_children))

    if with_stats:
        nodes.append(GDTNode("schema", "research-grade typed schema", children=[
            GDTNode("codebook", "≥20 typed variables designed", check=codebook_designed),
            GDTNode("populated", "≥80% rows have codebook fields", check=codebook_extracted),
        ]))
        nodes.append(GDTNode("artefacts", "publishable artefacts", children=[
            GDTNode("parquet", "parquet exported", check=parquet_exported),
        ]))

    return GDTNode("ROOT", goal, children=nodes)


def render_tree(node_result: dict, indent: int = 0) -> str:
    """Render an evaluated tree (dict result) as a multi-line string."""
    glyph = {
        "done":        "✓",
        "in_progress": "◐",
        "pending":     "☐",
        "blocked":     "✗",
    }.get(node_result["status"], "?")
    pad = "  " * indent
    head = f"{pad}{glyph} {node_result['name']}: {node_result['description']}"
    if node_result.get("detail"):
        head += f"  [{node_result['detail']}]"
    lines = [head]
    for child in node_result.get("children", []):
        lines.append(render_tree(child, indent + 1))
    return "\n".join(lines)
