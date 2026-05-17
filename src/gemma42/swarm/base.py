"""Shared helpers for specialist agents."""

from __future__ import annotations

import json
from typing import Any

from gemma42.parsing import _candidates, _repair_invalid_escapes, _strip_trailing_commas


def parse_json_obj(raw: str) -> dict | None:
    """Tolerant JSON-object parse from messy model output."""
    for cand in _candidates(raw):
        for variant in (
            cand,
            _strip_trailing_commas(cand),
            _repair_invalid_escapes(cand),
            _strip_trailing_commas(_repair_invalid_escapes(cand)),
        ):
            try:
                obj = json.loads(variant)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_json_array(raw: str) -> list | None:
    for cand in _candidates(raw):
        for variant in (
            cand,
            _strip_trailing_commas(cand),
            _repair_invalid_escapes(cand),
            _strip_trailing_commas(_repair_invalid_escapes(cand)),
        ):
            try:
                obj = json.loads(variant)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                # accept {"items": [...]} wrappers
                for k in ("items", "list", "results", "data", "rows", "issues"):
                    if isinstance(obj.get(k), list):
                        return obj[k]
    return None


def llm_json(llm: Any, system: str, user: str, *, temperature: float = 0.2) -> Any:
    """Call the LLM and parse a JSON value out of the response."""
    raw = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    obj = parse_json_obj(raw)
    if obj is not None:
        return obj
    arr = parse_json_array(raw)
    if arr is not None:
        return arr
    raise ValueError(f"specialist could not parse JSON; raw: {raw[:400]}")
