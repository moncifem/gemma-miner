"""In-run reflection: distill 'what worked / what didn't' from recent turns.

Called every K turns by the agent loop. The output is a short list of
lessons (≤140 chars each) that get injected into every subsequent prompt
so the model stops repeating the same mistake inside a single run.

The list is capped at MAX_LESSONS — new lessons evict old ones when the
list grows beyond the cap. We dedupe near-identical lessons by their first
40 chars to keep the list compact.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma42.parsing import _candidates, _repair_invalid_escapes, _strip_trailing_commas

if TYPE_CHECKING:
    from gemma42.llm import LLMClient
    from gemma42.state import AgentState


REFLECT_EVERY = 4
MAX_LESSONS = 10
MAX_LESSON_CHARS = 200


_REFLECT_SYS = """You are a coaching observer for a research agent. The agent just took several turns trying to build a dataset. Your job: distill the last few turns into 1-4 SHORT lessons the agent should remember for the rest of this run.

Output JSON ONLY in this shape — no prose, no fences:
{"lessons": ["…", "…"]}

Rules:
 - Each lesson is ≤ 140 characters, imperative voice, ACTIONABLE.
 - Prefer lessons about WHAT TO STOP DOING (a tool/arg pattern that failed) and WHAT WORKS (a tool/arg pattern that produced rows).
 - Cite the offending tool + arg by name when relevant. e.g. "llm_scrape with source=<URL> returns 0 rows — pass the cache_path instead".
 - If nothing new is worth recording, return {"lessons": []}.
 - Do NOT restate the goal, the phase, or generic advice.
"""


def _build_user_prompt(state: "AgentState", since_turn: int) -> str:
    recent = [h for h in state.history if h.turn > since_turn][-8:]
    lines: list[str] = []
    for h in recent:
        args = json.dumps(h.args, ensure_ascii=False)
        if len(args) > 240:
            args = args[:240] + "…"
        obs = (h.observation or "").replace("\n", " ")
        if len(obs) > 360:
            obs = obs[:360] + "…"
        mark = "ERROR " if h.error else ""
        lines.append(f"turn {h.turn}  {mark}{h.tool}({args})\n  → {obs}")
    existing = "\n".join(f"  - {x}" for x in state.lessons) or "  (none yet)"
    return (
        f"GOAL: {state.goal}\n\n"
        f"EXISTING LESSONS:\n{existing}\n\n"
        f"RECENT TURNS (most recent last):\n" + "\n".join(lines)
        + "\n\nReturn JSON only."
    )


def _parse_lessons(raw: str) -> list[str]:
    for cand in (raw.strip(), *list(_candidates(raw))):
        for variant in (
            cand,
            _strip_trailing_commas(cand),
            _repair_invalid_escapes(cand),
        ):
            try:
                obj = json.loads(variant)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and isinstance(obj.get("lessons"), list):
                return [str(x).strip() for x in obj["lessons"] if str(x).strip()]
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
    return []


def _dedupe_key(s: str) -> str:
    """Normalize a lesson string for dedup: lowercase, drop non-word chars,
    collapse whitespace, take first 30 chars."""
    import re as _re
    norm = _re.sub(r"\W+", " ", s.lower()).strip()
    return norm[:30]


def _merge_lessons(existing: list[str], fresh: list[str]) -> list[str]:
    """Dedupe by a normalized prefix. Newest wins."""
    seen: dict[str, str] = {}
    for lesson in existing + fresh:
        if not lesson:
            continue
        trimmed = lesson[:MAX_LESSON_CHARS].strip()
        if not trimmed:
            continue
        seen[_dedupe_key(trimmed)] = trimmed
    merged = list(seen.values())
    if len(merged) > MAX_LESSONS:
        merged = merged[-MAX_LESSONS:]
    return merged


def should_reflect(state: "AgentState", *, every: int = REFLECT_EVERY) -> bool:
    """Reflect every `every` turns after enough history accumulates."""
    n = len(state.history)
    if n < every:
        return False
    return (n - state.last_reflection_turn) >= every


def reflect(state: "AgentState", llm: "LLMClient") -> list[str]:
    """Distill lessons from the recent turns. Mutates state.lessons and
    state.last_reflection_turn. Returns the list of NEWLY added lessons.

    Failures are swallowed — reflection must never break the agent loop.
    """
    prompt = _build_user_prompt(state, since_turn=state.last_reflection_turn)
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": _REFLECT_SYS},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception:  # noqa: BLE001
        return []
    fresh = _parse_lessons(raw)
    before = set(state.lessons)
    state.lessons = _merge_lessons(state.lessons, fresh)
    state.last_reflection_turn = len(state.history)
    return [l for l in state.lessons if l not in before]
