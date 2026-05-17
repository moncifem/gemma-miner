"""Constitutional verification — declared and inferred domain rules.

A *rule* is a typed cross-row or per-row predicate the dataset MUST satisfy.
The user (or the inference engine) writes rules; the validator runs them on
every row + cross-row check; failures route to a fix-up agent or get
reported in the dataset card.

Rule kinds:
  - per-row    : f(row) → bool
  - cross-row  : f(rows) → list[(row, reason)]   (e.g., uniqueness, monotonicity)

Each rule is JSON-defined for portability + reproducibility (goes into the
gemma42.lock manifest). A small DSL covers the common cases without giving
the LLM arbitrary Python.

Examples (declared in natural language → compiled to JSON):
  - "every row must have a decision date in 2020-2026"
        {kind: "per_row", op: "in_range", field: "dn_decision",
         min: "2020-01-01", max: "2026-12-31"}
  - "if outcome = SANCT, fine must be > 0"
        {kind: "per_row", op: "implies",
         when: {field: "outcome", eq: "SANCT"},
         then: {field: "fine", gt: 0}}
  - "(date, organization) must be unique across rows"
        {kind: "cross_row", op: "unique", fields: ["date", "organization"]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── rule predicates ─────────────────────────────────────────────────────────


def _get(row: dict, field: str) -> Any:
    return row.get(field)


def _is_truthy(v: Any) -> bool:
    return v is not None and v != "" and v != [] and v != {}


def _cmp(a: Any, b: Any, op: str) -> bool:
    try:
        if op == "eq":  return a == b
        if op == "ne":  return a != b
        if op == "gt":  return a is not None and b is not None and a > b
        if op == "ge":  return a is not None and b is not None and a >= b
        if op == "lt":  return a is not None and b is not None and a < b
        if op == "le":  return a is not None and b is not None and a <= b
    except TypeError:
        return False
    return False


def _check_constraint(row: dict, c: dict) -> bool:
    """Evaluate a single constraint dict against a row."""
    field = c.get("field")
    v = _get(row, field) if field else None
    if "exists" in c:
        return _is_truthy(v) == bool(c["exists"])
    if "eq" in c:
        return _cmp(v, c["eq"], "eq")
    if "ne" in c:
        return _cmp(v, c["ne"], "ne")
    if "gt" in c:
        return _cmp(v, c["gt"], "gt")
    if "ge" in c:
        return _cmp(v, c["ge"], "ge")
    if "lt" in c:
        return _cmp(v, c["lt"], "lt")
    if "le" in c:
        return _cmp(v, c["le"], "le")
    if "in" in c:
        return v in (c["in"] or [])
    if "not_in" in c:
        return v not in (c["not_in"] or [])
    if "match" in c:
        if not isinstance(v, str):
            return False
        try:
            return bool(re.search(c["match"], v))
        except re.error:
            return False
    if "min" in c or "max" in c:
        try:
            if "min" in c and v is not None and v < c["min"]:
                return False
            if "max" in c and v is not None and v > c["max"]:
                return False
            return True
        except TypeError:
            return False
    return True


# ── rule kinds ──────────────────────────────────────────────────────────────


@dataclass
class Rule:
    name: str
    kind: str       # 'per_row' | 'cross_row'
    op: str         # 'check' | 'implies' | 'in_range' | 'unique' | ...
    spec: dict      # the operands
    description: str = ""
    severity: str = "error"   # 'error' | 'warning'
    inferred: bool = False    # True if proposed by the inference engine

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "op": self.op,
            "spec": self.spec, "description": self.description,
            "severity": self.severity, "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(
            name=d.get("name", "rule"),
            kind=d.get("kind", "per_row"),
            op=d.get("op", "check"),
            spec=d.get("spec", {}) or {},
            description=d.get("description", ""),
            severity=d.get("severity", "error"),
            inferred=bool(d.get("inferred", False)),
        )

    # ── per-row evaluation ───────────────────────────────────────────────

    def eval_row(self, row: dict) -> str | None:
        """For per_row rules: return failure message or None if OK."""
        if self.kind != "per_row":
            return None
        op = self.op
        spec = self.spec
        if op == "check":
            return None if _check_constraint(row, spec) else self._fail_msg(row)
        if op == "implies":
            when = spec.get("when") or {}
            then = spec.get("then") or {}
            if not _check_constraint(row, when):
                return None        # antecedent false → rule trivially satisfied
            return None if _check_constraint(row, then) else self._fail_msg(row)
        if op == "in_range":
            field = spec.get("field")
            v = _get(row, field) if field else None
            if v is None:
                return None  # null is not a violation by itself
            lo, hi = spec.get("min"), spec.get("max")
            if isinstance(v, str) and lo and isinstance(lo, str):
                # dates as strings YYYY-MM-DD compare lexically
                if v < lo or (hi and v > hi):
                    return f"{field}={v!r} outside [{lo}, {hi}]"
                return None
            try:
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    return f"{field}={v!r} outside [{lo}, {hi}]"
            except TypeError:
                return f"{field}={v!r} not comparable with [{lo}, {hi}]"
            return None
        if op == "enum_in":
            field = spec.get("field")
            v = _get(row, field) if field else None
            if v is None:
                return None
            allowed = spec.get("values") or []
            return None if v in allowed else f"{field}={v!r} ∉ {allowed}"
        return None

    def _fail_msg(self, row: dict) -> str:
        sample = {k: row.get(k) for k in (
            self.spec.get("field"),
            (self.spec.get("when") or {}).get("field"),
            (self.spec.get("then") or {}).get("field"),
        ) if k}
        return f"{self.description or self.name} ({sample})"

    # ── cross-row evaluation ─────────────────────────────────────────────

    def eval_rows(self, rows: list[dict]) -> list[tuple[Any, str]]:
        """For cross_row rules: return list of (offender, reason)."""
        if self.kind != "cross_row":
            return []
        op = self.op
        spec = self.spec
        if op == "unique":
            fields = spec.get("fields") or ["id"]
            seen: dict[tuple, Any] = {}
            failures: list[tuple[Any, str]] = []
            for r in rows:
                key = tuple(r.get(f) for f in fields)
                if all(x is None for x in key):
                    continue
                if key in seen:
                    failures.append((r, f"duplicate {fields}={key}"))
                seen[key] = r
            return failures
        if op == "monotonic":
            field = spec.get("field")
            order = spec.get("order", "asc")
            prev = None
            failures = []
            for r in rows:
                v = r.get(field)
                if v is None or prev is None:
                    prev = v
                    continue
                if order == "asc" and v < prev:
                    failures.append((r, f"{field} not ascending: {prev} → {v}"))
                if order == "desc" and v > prev:
                    failures.append((r, f"{field} not descending: {prev} → {v}"))
                prev = v
            return failures
        return []


# ── constitution = collection of rules ──────────────────────────────────────


@dataclass
class Constitution:
    rules: list[Rule] = field(default_factory=list)

    def add(self, rule: Rule) -> None:
        self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)

    def remove(self, name: str) -> bool:
        n = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        return len(self.rules) < n

    def evaluate(self, rows: list[dict]) -> dict:
        """Run every rule. Return a structured report."""
        per_row_failures: list[dict] = []
        cross_row_failures: list[dict] = []
        for r in self.rules:
            if r.kind == "per_row":
                for i, row in enumerate(rows):
                    msg = r.eval_row(row)
                    if msg:
                        per_row_failures.append({
                            "rule": r.name, "severity": r.severity,
                            "row_index": i, "id": row.get("id"),
                            "message": msg,
                        })
            else:
                for offender, reason in r.eval_rows(rows):
                    cross_row_failures.append({
                        "rule": r.name, "severity": r.severity,
                        "id": offender.get("id") if isinstance(offender, dict) else None,
                        "message": reason,
                    })
        n_errors = sum(
            1 for f in (per_row_failures + cross_row_failures)
            if f["severity"] == "error"
        )
        return {
            "n_rules": len(self.rules),
            "per_row_failures": per_row_failures,
            "cross_row_failures": cross_row_failures,
            "n_errors": n_errors,
            "ok": n_errors == 0,
        }

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.rules]

    @classmethod
    def from_list(cls, items: list[dict]) -> "Constitution":
        return cls(rules=[Rule.from_dict(d) for d in items])


# ── inference: propose rules from observed data ─────────────────────────────


def infer_rules(rows: list[dict], *, min_rows: int = 8) -> list[Rule]:
    """Look at the data and propose rules the corpus *appears* to follow.

    Heuristics (each gated on confidence):
      - a numeric field has min/max within a narrow range
      - a string field's values are all from a small set (→ enum_in)
      - a date field is always in a year/range window
      - a key combination is always unique
    """
    if not rows or len(rows) < min_rows:
        return []
    proposed: list[Rule] = []

    # Collect per-column non-null values.
    cols: dict[str, list] = {}
    for r in rows:
        for k, v in r.items():
            if k.startswith("_"):
                continue
            cols.setdefault(k, []).append(v)

    for k, values in cols.items():
        non_null = [v for v in values if v is not None and v != ""]
        if len(non_null) < min_rows // 2:
            continue
        # numeric range
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            mn, mx = min(non_null), max(non_null)
            proposed.append(Rule(
                name=f"range_{k}",
                kind="per_row",
                op="in_range",
                spec={"field": k, "min": mn, "max": mx},
                description=f"`{k}` is always in [{mn}, {mx}]",
                inferred=True,
            ))
        # date range
        elif all(isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in non_null):
            mn, mx = min(non_null), max(non_null)
            proposed.append(Rule(
                name=f"range_{k}",
                kind="per_row",
                op="in_range",
                spec={"field": k, "min": mn[:10], "max": mx[:10]},
                description=f"`{k}` is always between {mn[:10]} and {mx[:10]}",
                inferred=True,
            ))
        # small-set enum
        elif all(isinstance(v, str) for v in non_null):
            unique = sorted(set(non_null))
            if 2 <= len(unique) <= 10 and len(unique) < len(non_null) / 2:
                proposed.append(Rule(
                    name=f"enum_{k}",
                    kind="per_row",
                    op="enum_in",
                    spec={"field": k, "values": unique},
                    description=f"`{k}` is always one of {unique}",
                    inferred=True,
                ))

    # Uniqueness candidates: any column whose non-null values are all distinct.
    for k, values in cols.items():
        non_null = [v for v in values if v is not None and v != ""]
        if len(non_null) < min_rows:
            continue
        if len(set(map(str, non_null))) == len(non_null):
            proposed.append(Rule(
                name=f"unique_{k}",
                kind="cross_row",
                op="unique",
                spec={"fields": [k]},
                description=f"`{k}` is unique across rows",
                inferred=True,
            ))

    return proposed
