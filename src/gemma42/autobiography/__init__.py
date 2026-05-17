"""The agent's persistent autobiography — 4-level memory hierarchy.

  L1  working memory       (in-process; AgentState)
  L2  run memory           (in-process; AgentState.memory + Dataset)
  L3  project memory       (sqlite at <workdir>/autobiography.db)
  L4  global memory        (sqlite at ~/.gemma42/autobiography.db)
  [L5 federated cloud      (opt-in; see gemma42.cloud)]

Tables (schema in `store.py`):
  - sites        sites we've visited (domain + fingerprint)
  - recipes      reusable extractor specs per site
  - codebooks    reusable codebooks per domain
  - episodes     past runs (goal, outcome, summary)
  - lessons      distilled insights per episode
  - skills       promoted Python helpers
"""

from gemma42.autobiography.store import (
    Autobiography,
    Codebook as CodebookRow,
    Episode,
    Lesson,
    Recipe,
    Site,
    Skill,
)

__all__ = ["Autobiography", "CodebookRow", "Episode", "Lesson", "Recipe", "Site", "Skill"]
