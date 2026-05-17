"""Statistician: writes a 3-paragraph plain-English findings report.

Input: the codebook + the per-variable stats. The Statistician produces a
short, intelligent narrative that a researcher would put in a paper.

Output:
  {
    "headline": "<one sentence>",
    "findings": ["finding 1 (numeric)", "finding 2", "finding 3"],
    "caveats":  ["caveat 1", ...],
    "narrative": "<one paragraph>"
  }
"""

from __future__ import annotations

import json
from typing import Any

from gemma42.swarm.base import llm_json


_SYS = """You are a SENIOR DATA SCIENTIST writing the findings section of a short report.

You receive a CODEBOOK (variable definitions) and a STATS dict (coverage, distributions, summary stats per variable).

Produce a JSON object:
{
  "headline":   "<one striking, factually-supported sentence>",
  "findings":   ["<finding 1 with a number>", "<finding 2>", "<finding 3>"],
  "caveats":    ["<data-quality caveat 1>", "<caveat 2>"],
  "narrative":  "<one to two paragraphs of plain English>"
}

Rules:
  - Every finding must cite a number that EXISTS in the stats.
  - No speculation beyond the stats.
  - Caveats: low-coverage variables, single-value enums, suspiciously narrow ranges.
  - Tone: tight, professional, no hype.
"""


def write_report(llm: Any, codebook: dict, stats: dict) -> dict:
    user = (
        "CODEBOOK:\n"
        + json.dumps({
            "name": codebook.get("name"),
            "description": codebook.get("description"),
            "domain": codebook.get("domain"),
            "variables": [
                {"name": v.get("name"), "type": v.get("type"),
                 "description": v.get("description")[:200]}
                for v in (codebook.get("variables") or [])
            ],
        }, ensure_ascii=False)[:15_000]
        + "\n\nSTATS:\n"
        + json.dumps(stats, ensure_ascii=False, default=str)[:20_000]
    )
    return llm_json(llm, _SYS, user, temperature=0.3)
