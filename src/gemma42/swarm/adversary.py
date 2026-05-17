"""Adversary: critiques a proposed codebook. Finds variables likely to be
useless (null-most-of-the-time, redundant, ambiguous, mistyped).

Output:
  {
    "drop":    [{"name": "...", "reason": "..."}],
    "merge":   [{"keep": "...", "into": "...", "reason": "..."}],
    "retype":  [{"name": "...", "from": "string", "to": "enum", "enum_values": [...]}],
    "tighten": [{"name": "...", "add_negative_examples": [...], "add_extraction_hint": "..."}],
    "approve": ["...names that survive critique unchanged..."]
  }
"""

from __future__ import annotations

import json
from typing import Any

from gemma42.swarm.base import llm_json


_SYS = """You are the ADVERSARY. A junior data scientist proposed a codebook of variables. Your job is to RUTHLESSLY critique it. You have ONE goal: prevent useless variables from polluting the dataset.

Attack patterns to look for:
  1. NULL-MOST: the variable will be null in 80%+ of items because the info rarely appears.
  2. REDUNDANT: the variable duplicates another (e.g. n_violations vs n_total_breaches).
  3. AMBIGUOUS: the description is vague; two readers would extract differently.
  4. MISTYPED: a string variable that should be an enum (small known set), an integer that should be a boolean, etc.
  5. UNGROUNDED: no anchor in the sample texts — the agent will hallucinate.
  6. LEAKY: the variable encodes the label of something you want to predict (no leakage in a stats dataset).
  7. CHRONO: the variable name is fine but the description is incompatible with the dates in the sample.

For each attack, propose a remediation: drop, merge, retype, or tighten with negative examples.

Output ONE JSON object (no prose, no fences):
{
  "drop":   [{"name": "...", "reason": "..."}],
  "merge":  [{"keep": "...", "into": "...", "reason": "..."}],
  "retype": [{"name": "...", "from": "string", "to": "enum", "enum_values": [...]}],
  "tighten":[{"name": "...", "add_negative_examples": ["..."], "add_extraction_hint": "..."}],
  "approve": ["...names you would keep AS-IS..."]
}
"""


def critique_codebook(llm: Any, codebook: dict, samples: list[tuple[str, str]]) -> dict:
    sample_blocks = "\n\n".join(
        f"=== ITEM {iid} (truncated) ===\n{text[:6_000]}"
        for iid, text in samples[:3]
    )
    user = (
        "PROPOSED CODEBOOK:\n"
        + json.dumps(codebook, ensure_ascii=False, indent=2)[:30_000]
        + "\n\nSAMPLE ITEMS for grounding:\n"
        + sample_blocks
    )
    return llm_json(llm, _SYS, user, temperature=0.3)
