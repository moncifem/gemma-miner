"""Active self-doubt + adaptive quality gating.

Wraps the Auditor specialist with bookkeeping:
  - per-row confidence score
  - aggregate "what fraction of the dataset would I trust at threshold T?"
  - hooks for differential coverage boosting (re-extract low-confidence rows)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gemma42.swarm.auditor import audit_rows


@dataclass
class AuditReport:
    n_audited: int
    overall_confidence: float
    per_id_confidence: dict[str, float] = field(default_factory=dict)
    wrong_fields: dict[str, list[str]] = field(default_factory=dict)
    themes: list[str] = field(default_factory=list)

    def quality_at(self, threshold: float) -> int:
        return sum(1 for c in self.per_id_confidence.values() if c >= threshold)

    def to_dict(self) -> dict:
        return {
            "n_audited": self.n_audited,
            "overall_confidence": self.overall_confidence,
            "per_id_confidence": self.per_id_confidence,
            "wrong_fields": self.wrong_fields,
            "themes": self.themes,
        }


def _row_text(row: dict, workdir: str | Path) -> str:
    """Best-effort: read the row's source text from text_path/pdf_path or
    pdf_text. Falls back to concatenating other in-row string fields so
    structured-only datasets (e.g. {date, org_type, decision}) still have
    something to extract from."""
    from gemma42.tools.codebook_tool import _row_text as _rt
    return _rt(row, str(workdir))


def stratified_sample(rows: list[dict], k: int) -> list[dict]:
    if len(rows) <= k:
        return rows
    idxs = set()
    idxs.add(0)
    idxs.add(len(rows) - 1)
    rnd = random.Random(42)
    while len(idxs) < k:
        idxs.add(rnd.randrange(0, len(rows)))
    return [rows[i] for i in sorted(idxs)]


def audit_dataset(llm: Any, rows: list[dict], workdir: str | Path,
                  *, sample_size: int = 8) -> AuditReport:
    sample = stratified_sample(rows, sample_size)
    pairs: list[tuple[dict, str]] = []
    for r in sample:
        pairs.append((r, _row_text(r, workdir)))
    pairs = [p for p in pairs if p[1]]
    if not pairs:
        return AuditReport(n_audited=0, overall_confidence=0.0, themes=["no source text available"])
    raw = audit_rows(llm, pairs)
    per_id: dict[str, float] = {}
    wrong: dict[str, list[str]] = {}
    for a in (raw.get("audits") or []):
        rid = str(a.get("id", "?"))
        try:
            per_id[rid] = float(a.get("confidence", 0.5))
        except (TypeError, ValueError):
            per_id[rid] = 0.5
        wf = a.get("wrong_fields") or []
        if wf:
            wrong[rid] = list(wf)
    return AuditReport(
        n_audited=len(pairs),
        overall_confidence=float(raw.get("overall_confidence", 0.5) or 0.5),
        per_id_confidence=per_id,
        wrong_fields=wrong,
        themes=list(raw.get("themes") or []),
    )


def format_report(rep: AuditReport) -> str:
    if rep.n_audited == 0:
        return "(no rows audited)"
    lines = [
        f"audited:           {rep.n_audited}",
        f"overall confidence: {rep.overall_confidence:.0%}",
        f"≥90% confidence:    {rep.quality_at(0.9)}",
        f"≥80% confidence:    {rep.quality_at(0.8)}",
        f"<60% confidence:    {sum(1 for c in rep.per_id_confidence.values() if c < 0.6)}",
    ]
    if rep.themes:
        lines.append("themes:")
        for t in rep.themes[:5]:
            lines.append(f"  - {t}")
    if rep.wrong_fields:
        lines.append("wrong-field hits:")
        for rid, fs in list(rep.wrong_fields.items())[:5]:
            lines.append(f"  {rid}: {fs}")
    return "\n".join(lines)
