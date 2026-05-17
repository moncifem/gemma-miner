"""Consolidator: merges the Curator's proposal with the Adversary's critique
into a final, locked codebook. Deterministic — no LLM call needed.

Applies:
  drop      → remove variable
  merge     → keep one, drop the other (with merged extraction_hint)
  retype    → change type / enum_values
  tighten   → append negative_examples + extraction_hint
"""

from __future__ import annotations

from typing import Any


def consolidate_codebook(proposal: dict, critique: dict) -> dict:
    """Apply the adversary's critique to the proposal. Returns a new codebook dict."""
    variables = list(proposal.get("variables") or [])
    by_name = {v["name"]: v for v in variables if isinstance(v, dict) and "name" in v}

    # 1. drop
    for d in critique.get("drop", []) or []:
        n = d.get("name")
        if n in by_name:
            by_name[n]["_adversary_note"] = d.get("reason", "dropped")
            del by_name[n]

    # 2. merge — keep `keep`, drop `into`
    for m in critique.get("merge", []) or []:
        keep = m.get("keep")
        into = m.get("into")
        if keep in by_name and into in by_name and keep != into:
            kv = by_name[keep]
            iv = by_name[into]
            # combine extraction hints
            hints = [h for h in (kv.get("extraction_hint"), iv.get("extraction_hint")) if h]
            if hints:
                kv["extraction_hint"] = " | ".join(hints)
            scars = kv.get("adversary_notes") or []
            scars.append(f"merged-from:{into}: {m.get('reason','')}")
            kv["adversary_notes"] = scars
            del by_name[into]

    # 3. retype
    for r in critique.get("retype", []) or []:
        n = r.get("name")
        if n in by_name:
            new_type = r.get("to")
            if new_type:
                by_name[n]["type"] = new_type
                if new_type == "enum" and r.get("enum_values"):
                    by_name[n]["enum_values"] = list(r["enum_values"])

    # 4. tighten
    for t in critique.get("tighten", []) or []:
        n = t.get("name")
        if n in by_name:
            v = by_name[n]
            if t.get("add_negative_examples"):
                ne = list(v.get("negative_examples") or [])
                ne += [s for s in t["add_negative_examples"] if isinstance(s, str)]
                v["negative_examples"] = ne[:6]
            if t.get("add_extraction_hint"):
                h = v.get("extraction_hint") or ""
                v["extraction_hint"] = (h + "  " + t["add_extraction_hint"]).strip()
            scars = v.get("adversary_notes") or []
            scars.append("tightened by adversary")
            v["adversary_notes"] = scars

    # Rebuild in original order, minus drops/merges.
    survivors = [v for v in variables if v.get("name") in by_name]
    survivors_names = {v["name"] for v in survivors}
    # If the adversary referenced a variable we don't have, ignore silently.
    for k, v in by_name.items():
        if k not in survivors_names:
            survivors.append(v)

    out = dict(proposal)
    out["variables"] = survivors
    if "name" in out and not out["name"].endswith("_v2"):
        out["name"] = out["name"]   # keep
    out.setdefault("description", "")
    return out
