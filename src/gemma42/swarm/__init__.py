"""The Specialist Swarm — focused mini-agents for each pipeline phase.

Each specialist:
  - has a narrow system prompt
  - sees only the inputs relevant to its task
  - returns a structured result (JSON)
  - is independent — invoked as a single LLM call, not a multi-turn loop

The orchestrating agent calls specialists as tools. They coordinate via
the shared workspace (memory + dataset).
"""

from gemma42.swarm.adversary import critique_codebook
from gemma42.swarm.auditor import audit_rows
from gemma42.swarm.cartographer import map_site
from gemma42.swarm.consolidator import consolidate_codebook
from gemma42.swarm.curator import propose_codebook
from gemma42.swarm.statistician import write_report

__all__ = [
    "audit_rows", "critique_codebook", "consolidate_codebook",
    "map_site", "propose_codebook", "write_report",
]
