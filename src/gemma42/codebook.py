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
        vars_ = []
        for v in d.get("variables", []):
            vars_.append(VariableSpec(**v))
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

    def numeric_or_boolean_ratio(self) -> float:
        if not self.variables:
            return 0.0
        n = sum(
            1 for v in self.variables
            if v.type in ("integer", "float", "boolean")
        )
        return n / len(self.variables)
