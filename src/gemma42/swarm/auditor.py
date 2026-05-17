"""Auditor: takes a sample of dataset rows and judges them against the source text.

Output:
  {
    "audits": [
      {"id": "...", "confidence": 0.0..1.0,
       "wrong_fields": ["field_a", ...],
       "comments": ["..."]},
      ...
    ],
    "overall_confidence": 0.0..1.0,
    "themes": ["common issue 1", "common issue 2"]
  }
"""

from __future__ import annotations

import json
from typing import Any

from gemma42.swarm.base import llm_json


_SYS = """You are the AUDITOR. For each (source_text, extracted_row) pair, judge whether the extraction is correct.

For every audit produce:
  - id: the row id
  - confidence: float in [0,1] — your trust in the row overall
  - wrong_fields: array of field names that look wrong, with a brief reason in `comments`
  - comments: array of short notes

Look for:
  - dates that don't appear in the text
  - amounts off by 10x (e.g. millions vs thousands)
  - booleans flipped
  - enum values not appearing in the source
  - hallucinated entities

Be HONEST. Confidence ~0.95 means everything checks. ~0.5 means partial. ~0.2 means contradicted by the source.

Output ONE JSON object (no prose, no fences):
{
  "audits": [...],
  "overall_confidence": 0.0..1.0,
  "themes": ["theme 1", "theme 2"]
}
"""


def audit_rows(llm: Any, pairs: list[tuple[dict, str]], *,
               max_text_chars: int = 8000) -> dict:
    """`pairs` = list of (row, source_text)."""
    blocks: list[str] = []
    for row, text in pairs[:8]:
        rid = row.get("id", "?")
        snippet = (text or "")[:max_text_chars]
        blocks.append(
            f"--- ROW id={rid} ---\n"
            f"EXTRACTED: {json.dumps({k:v for k,v in row.items() if not k.startswith('_')}, ensure_ascii=False)[:6000]}\n"
            f"SOURCE TEXT (first {max_text_chars} chars):\n{snippet}\n"
        )
    user = "\n\n".join(blocks)
    return llm_json(llm, _SYS, user, temperature=0.1)
