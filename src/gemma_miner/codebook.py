"""Codebook: typed variable definitions for structured extraction.

A `Codebook` is a list of variables (columns) that together describe the
structured information to be extracted from each item. Every variable has a
declared type, an optional unit/range, and a human description. The codebook
is the *contract* for the extraction phase — every row in the final dataset
must conform.

Naming conventions (encouraged, not enforced):

  n_*       integer count
  pct_*     percentage 0–100 (float)
  amount_*  monetary amount (float)
  is_*      boolean fact
  has_*     boolean fact
  cat_*     categorical / enum
  dn_*      date (YYYY-MM-DD)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

VAR_TYPES = ("boolean", "integer", "float", "string", "enum", "date", "array")


@dataclass
class VariableSpec:
    name: str
    type: str
    description: str
    enum_values: list[str] | None = None      # required for type='enum'
    unit: str | None = None                    # 'euros', 'percent', 'days'
    min_value: float | None = None
    max_value: float | None = None
    item_schema: dict | None = None            # for type='array' — JSON Schema per item
    required: bool = False
    extraction_hint: str | None = None         # extra LLM guidance per variable

    # Living-codebook upgrades:
    # - Negative examples: "this is NOT a valid value" — improves extraction
    #   sharply on small models because they often confuse adjacent concepts.
    negative_examples: list[str] = field(default_factory=list)
    # - Positive examples (verbatim quotes from the source). Optional anchors.
    positive_examples: list[str] = field(default_factory=list)
    # - Pass: variables can be flagged as pass-2 (only extract them when all
    #   pass-1 variables are already filled, with pass-1 context available).
    pass_: int = 1                              # 1 = first read, 2 = re-read with context
    # - Adversarial scars: previous adversary critiques that the variable has
    #   already survived. Used to remember "we already addressed X".
    adversary_notes: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name or not isinstance(self.name, str):
            errs.append("name must be a non-empty string")
        if self.type not in VAR_TYPES:
            errs.append(f"type must be one of {VAR_TYPES}, got {self.type!r}")
        if self.type == "enum" and not self.enum_values:
            errs.append(f"variable {self.name!r}: type='enum' requires enum_values")
        if self.type == "array" and not isinstance(self.item_schema, dict):
            errs.append(f"variable {self.name!r}: type='array' requires item_schema (dict)")
        return errs

    def to_json_schema(self) -> dict:
        """Render as a JSON Schema fragment (nullable unless required)."""
        if self.type == "boolean":
            schema: dict = {"type": ["boolean", "null"]}
        elif self.type == "integer":
            schema = {"type": ["integer", "null"]}
        elif self.type == "float":
            schema = {"type": ["number", "null"]}
        elif self.type == "date":
            schema = {"type": ["string", "null"], "description": "YYYY-MM-DD"}
        elif self.type == "enum":
            schema = {"type": ["string", "null"], "enum": [*self.enum_values, None] if self.enum_values else None}
        elif self.type == "array":
            schema = {"type": "array", "items": self.item_schema or {"type": "object"}}
        else:  # string
            schema = {"type": ["string", "null"]}
        descr = self.description
        if self.unit:
            descr = f"{descr} (unit: {self.unit})"
        if self.extraction_hint:
            descr = f"{descr} HINT: {self.extraction_hint}"
        if self.min_value is not None:
            descr = f"{descr} (min {self.min_value})"
        if self.max_value is not None:
            descr = f"{descr} (max {self.max_value})"
        if self.positive_examples:
            descr = f"{descr} EXAMPLES: " + " | ".join(self.positive_examples[:3])
        if self.negative_examples:
            descr = f"{descr} NOT: " + " | ".join(self.negative_examples[:3])
        schema["description"] = descr
        return schema


@dataclass
class Codebook:
    name: str
    description: str
    variables: list[VariableSpec] = field(default_factory=list)
    version: str = "0.1.0"
    domain: str | None = None
    primary_source_field: str = "text_path"   # which row field holds the text path
    fallback_text_field: str = "pdf_text"     # else inline text from this row field

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Codebook":
        import dataclasses as _dc
        allowed = {f.name for f in _dc.fields(VariableSpec)}
        vars_ = []
        for v in d.get("variables", []):
            # Drop any unknown keys (curator/adversary may attach internal
            # `_adversary_note` or other scratch fields). Keep what's valid.
            clean = {k: val for k, val in v.items() if k in allowed}
            vars_.append(VariableSpec(**clean))
        cb = cls(
            name=d.get("name", "codebook"),
            description=d.get("description", ""),
            version=d.get("version", "0.1.0"),
            domain=d.get("domain"),
            primary_source_field=d.get("primary_source_field", "text_path"),
            fallback_text_field=d.get("fallback_text_field", "pdf_text"),
            variables=vars_,
        )
        return cb

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "domain": self.domain,
            "primary_source_field": self.primary_source_field,
            "fallback_text_field": self.fallback_text_field,
            "variables": [asdict(v) for v in self.variables],
        }

    @classmethod
    def load(cls, path: str | Path) -> "Codebook":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def variable(self, name: str) -> VariableSpec | None:
        for v in self.variables:
            if v.name == name:
                return v
        return None

    def upsert_variable(self, var: VariableSpec) -> None:
        for i, existing in enumerate(self.variables):
            if existing.name == var.name:
                self.variables[i] = var
                return
        self.variables.append(var)

    def remove_variable(self, name: str) -> bool:
        n = len(self.variables)
        self.variables = [v for v in self.variables if v.name != name]
        return len(self.variables) < n

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.variables:
            errs.append("codebook has no variables")
        seen: set[str] = set()
        for v in self.variables:
            errs.extend(v.validate())
            if v.name in seen:
                errs.append(f"duplicate variable name: {v.name}")
            seen.add(v.name)
        return errs

    def to_json_schema(self) -> dict:
        """Build a JSON Schema object the LLM can be instructed against."""
        properties = {v.name: v.to_json_schema() for v in self.variables}
        required = [v.name for v in self.variables if v.required]
        return {
            "type": "object",
            "title": self.name,
            "description": self.description,
            "properties": properties,
            "required": required,
        }

    # ── type variety analysis ──────────────────────────────────────────────

    def type_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.variables:
            out[v.type] = out.get(v.type, 0) + 1
        return out

    def extraction_signature(self) -> str:
        """Stable hash over the EXTRACTION-relevant subset of the codebook.

        Used by the phase machine to decide whether silver rows are stale.
        Cosmetic changes that don't affect what the LLM sees (e.g. flipping
        `required`, adding adversary notes, renaming the codebook) must NOT
        force a full re-extract.
        """
        import hashlib as _hl

        extracted_fields_per_var = (
            "name", "type", "description", "extraction_hint",
            "enum_values", "unit", "min_value", "max_value",
            "item_schema", "positive_examples", "negative_examples",
            "pass_",
        )
        payload = [
            {k: getattr(v, k, None) for k in extracted_fields_per_var}
            for v in self.variables
        ]
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return _hl.sha256(blob.encode("utf-8")).hexdigest()

    # ── duplicate / near-duplicate detection ───────────────────────────────

    def find_duplicate_groups(self) -> list[list[str]]:
        """Return groups of variables whose names share a semantic stem.

        Detects the codebook-drift artefacts we saw in production runs:
          • `has_llm` and `is_mentions_llm`
          • `is_visa_sponsorship` and `is_visa_sponsorship_available`
          • `has_vector_database` and `has_vector_db`
          • `n_years_experience_required` and `n_years_experience_req`
        """
        import re as _re

        # Stems we strip to compute a canonical key for each variable.
        # Order matters: longer / more specific prefixes/suffixes first.
        PREFIXES = (
            "is_mentions_", "has_mentions_", "mentions_",
            "is_", "has_", "n_", "num_", "amount_", "pct_", "cat_", "dn_",
        )
        SUFFIXES = (
            "_available", "_required", "_req", "_min", "_max",
            "_count", "_total", "_num", "_number",
        )
        # Word-level synonym normalisation (the stem is rebuilt after this).
        SYN = {
            "database": "db", "databases": "db",
            "experience": "exp",
            "years": "yr", "year": "yr",
            "required": "req",
            "language": "lang",
            "configuration": "config",
            "specification": "spec",
        }

        def _key(name: str) -> str:
            n = name.lower()
            changed = True
            while changed:
                changed = False
                for p in PREFIXES:
                    if n.startswith(p):
                        n = n[len(p):]
                        changed = True
                        break
            changed = True
            while changed:
                changed = False
                for s in SUFFIXES:
                    if n.endswith(s):
                        n = n[: -len(s)]
                        changed = True
                        break
            # word-level synonym squash
            parts = [SYN.get(p, p) for p in _re.split(r"[_\W]+", n) if p]
            return "_".join(parts).strip("_")

        groups: dict[str, list[str]] = {}
        for v in self.variables:
            k = _key(v.name)
            if not k:
                continue
            groups.setdefault(k, []).append(v.name)
        return [names for names in groups.values() if len(names) > 1]

    def numeric_or_boolean_ratio(self) -> float:
        if not self.variables:
            return 0.0
        n = sum(
            1 for v in self.variables
            if v.type in ("integer", "float", "boolean")
        )
        return n / len(self.variables)

    # ── two-pass extraction ────────────────────────────────────────────────

    def pass1_variables(self) -> list[VariableSpec]:
        return [v for v in self.variables if (v.pass_ or 1) == 1]

    def pass2_variables(self) -> list[VariableSpec]:
        return [v for v in self.variables if (v.pass_ or 1) == 2]

    def to_json_schema_for_pass(self, pass_n: int) -> dict:
        """JSON Schema covering only variables flagged for this pass."""
        if pass_n == 1:
            vars_ = self.pass1_variables()
        else:
            vars_ = self.pass2_variables()
        properties = {v.name: v.to_json_schema() for v in vars_}
        required = [v.name for v in vars_ if v.required]
        return {
            "type": "object",
            "title": f"{self.name}_pass{pass_n}",
            "description": self.description,
            "properties": properties,
            "required": required,
        }
