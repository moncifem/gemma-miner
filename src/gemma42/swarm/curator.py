"""Curator: proposes a 30-60 variable codebook from sampled item texts.

Output is a list of VariableSpec-shaped dicts. The orchestrator can then
pass them through the Adversary for critique.
"""

from __future__ import annotations

from typing import Any

from gemma42.swarm.base import llm_json


_SYS = """You are the CURATOR, designing a research-grade codebook of 30-60 typed variables.

OBJECTIVE: turn a corpus of textual documents into a tabular dataset useful for STATISTICS and MACHINE LEARNING. Therefore at least 60% of variables MUST be numeric (integer/float) or boolean. Strings are reserved for IDs and short labels.

NAMING CONVENTIONS:
  n_*       integer count
  pct_*     percentage 0-100 (float)
  amount_*  monetary amount (float, with `unit`)
  is_*      boolean fact
  has_*     boolean fact
  cat_*     categorical / enum  (define `enum_values`)
  dn_*      date in YYYY-MM-DD

For every variable include:
  - name              snake_case
  - type              one of {boolean, integer, float, string, enum, date, array}
  - description       one clear sentence
  - enum_values       (for enum)
  - unit              (for amount_*, e.g. "euros")
  - min_value/max_value when applicable
  - positive_examples  2-3 short anchor quotes from the sample text
  - negative_examples  2-3 short examples of what this variable is NOT (very useful for small models)
  - pass_              1 (default) or 2 if the variable depends on others
  - extraction_hint    optional one-line note

Output ONE JSON object:
{
  "name": "<dataset name, snake_case>",
  "domain": "<short label>",
  "description": "<one-paragraph summary>",
  "variables": [ {...}, {...}, ... ]
}
NO prose around it, NO markdown fences.
"""


def propose_codebook(llm: Any, samples: list[tuple[str, str]], *,
                     domain_hint: str | None = None,
                     min_variables: int = 30) -> dict:
    blocks: list[str] = []
    for iid, text in samples:
        blocks.append(f"\n=== SAMPLE ITEM {iid} ===\n{text[:12_000]}\n")
    if domain_hint:
        blocks.insert(0, f"DOMAIN HINT: {domain_hint}\n")
    blocks.insert(0, f"TARGET: at least {min_variables} variables, ≥60% numeric/boolean.\n")
    user = "\n".join(blocks)
    return llm_json(llm, _SYS, user, temperature=0.2)
