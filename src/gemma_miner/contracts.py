"""Contracts: declarative requirements the agent must satisfy before finishing.

A contract has a `check(dataset)` method that returns (ok, message). The agent
loop refuses to terminate until every contract returns ok=True. New contracts
can be added mid-run via the `add_contract` tool, which lets the user (or the
agent itself) tighten the spec during execution — "actually I need 200 rows,
not 100" is just an addContract call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class Contract(ABC):
    name: str
    description: str

    @abstractmethod
    def check(self, dataset: Any) -> tuple[bool, str]:
        """Return (ok, human-readable status)."""

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}


@dataclass
class MinRowsContract(Contract):
    """Dataset must contain at least N rows."""

    min_rows: int
    name: str = "min_rows"

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"Dataset must contain at least {self.min_rows} rows."

    def check(self, dataset: Any) -> tuple[bool, str]:
        n = len(dataset)
        if n >= self.min_rows:
            return True, f"OK ({n}/{self.min_rows} rows)"
        return False, f"need more rows ({n}/{self.min_rows})"


def _field_variants(name: str) -> list[str]:
    """Acceptable variants for a required field name.

    Lets the contract tolerate small naming drift between what the user said
    in the prompt and what the model named the column. e.g. "comments" /
    "n_comments" / "num_comments" / "comments_count" all map to the same
    semantic field.
    """
    import re as _re

    base = name.strip()
    if not base:
        return [name]
    snake = _re.sub(r"[^\w]+", "_", base.lower()).strip("_")
    variants: set[str] = {base, snake, snake.replace("__", "_")}

    # Strip count-like prefixes/suffixes to find the bare noun.
    bare = snake
    bare = _re.sub(r"^(?:n_|num_|number_of_|nb_|count_of_|amount_of_)", "", bare)
    bare = _re.sub(r"_(?:count|total|amount|num|number)$", "", bare)
    bare = bare.strip("_") or snake

    # Generate the family.
    for stem in {snake, bare}:
        if not stem:
            continue
        variants.update({
            stem,
            f"n_{stem}", f"num_{stem}", f"number_of_{stem}", f"nb_{stem}",
            f"{stem}_count", f"{stem}_total", f"{stem}_num", f"{stem}_number",
        })
    return [v for v in variants if v]


def _is_meaningful(v: Any) -> bool:
    """A field counts as populated when it carries non-empty content.

    Language-agnostic by design: we do NOT maintain a list of "placeholder
    strings" — that approach is biased to a few languages and easy for any
    model to bypass with a synonym. Detection of placeholder-stuffing is
    handled separately (and statistically) by `_low_cardinality_signals`
    below: a required field whose mode covers most rows is surfaced as
    evidence in the contract detail, so the agent can read the data and
    decide.
    """
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def _low_cardinality_signals(
    rows: list[dict],
    fields: list[str],
) -> dict[str, tuple[str, int, int]]:
    """For each required field, report (mode_value, mode_count, total) when
    the field is populated but its modal value covers ≥ 60% of populated
    rows AND the row sample is ≥ 10. Returns nothing for high-cardinality
    fields. Language-agnostic; works on any source.
    """
    from collections import Counter

    flagged: dict[str, tuple[str, int, int]] = {}
    if len(rows) < 10:
        return flagged
    for f in fields:
        variants = _field_variants(f)
        values: list[str] = []
        for row in rows:
            for v in variants:
                val = row.get(v)
                if _is_meaningful(val):
                    values.append(str(val)[:120])
                    break
        if not values:
            continue
        c = Counter(values)
        mode_value, mode_count = c.most_common(1)[0]
        # Only flag if the populated field is mostly one value AND that
        # value is short enough to plausibly be a placeholder/constant.
        if mode_count / len(values) >= 0.6 and len(mode_value) <= 80:
            flagged[f] = (mode_value, mode_count, len(values))
    return flagged


@dataclass
class FieldsContract(Contract):
    """Every row in the dataset must have these fields populated (non-null)."""

    required_fields: list[str]
    name: str = "required_fields"

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"Every row must have non-null fields: {', '.join(self.required_fields)}."

    def check(self, dataset: Any) -> tuple[bool, str]:
        # For each required field, accept ANY of its naming variants.
        missing: dict[str, int] = {}
        rows = dataset.rows() if hasattr(dataset, "rows") else list(dataset)
        for f in self.required_fields:
            variants = _field_variants(f)
            for row in rows:
                if not any(_is_meaningful(row.get(v)) for v in variants):
                    missing[f] = missing.get(f, 0) + 1

        # Low-cardinality evidence — surface "this field is the same value
        # on most rows" so the agent can audit. Likely placeholder stuffing
        # OR a legitimate constant; the agent reads the value and decides.
        suspicious = _low_cardinality_signals(rows, self.required_fields)

        if not missing and not suspicious:
            return True, "OK (all rows have required fields)"
        parts: list[str] = []
        if missing:
            parts.append(f"missing fields: {missing}")
        if suspicious:
            sus_str = ", ".join(
                f"{f}={value!r} on {c}/{n} rows"
                for f, (value, c, n) in suspicious.items()
            )
            parts.append(
                f"low-cardinality (suspect constant or placeholder): {sus_str}"
            )
        # Low-cardinality alone is a WARNING, not a failure — the agent
        # needs to see the value to decide. Pass only when no `missing`.
        ok = not missing
        return ok, " · ".join(parts)


@dataclass
class UniqueFieldContract(Contract):
    """No two rows may share the same value for this field."""

    field: str
    name: str = "unique_field"

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"Field '{self.field}' must be unique across rows."

    def check(self, dataset: Any) -> tuple[bool, str]:
        seen: set = set()
        dupes = 0
        for row in dataset.rows() if hasattr(dataset, "rows") else dataset:
            v = row.get(self.field)
            if v is None:
                continue
            if v in seen:
                dupes += 1
            else:
                seen.add(v)
        if dupes == 0:
            return True, f"OK ({len(seen)} unique values)"
        return False, f"{dupes} duplicate {self.field}(s)"


@dataclass
class CodebookContract(Contract):
    """The dataset must have a codebook with at least N variables AND a
    minimum numeric/boolean ratio. Operates on the workdir's codebook.json."""

    min_variables: int = 20
    min_numeric_or_boolean_ratio: float = 0.5
    name: str = "codebook"

    @property
    def description(self) -> str:  # type: ignore[override]
        return (
            f"Codebook must define at least {self.min_variables} variables, "
            f"with ≥{self.min_numeric_or_boolean_ratio:.0%} numeric/boolean."
        )

    def check(self, dataset: Any) -> tuple[bool, str]:
        # We read the codebook from the dataset's sibling workdir.
        from pathlib import Path
        from gemma_miner.codebook import Codebook

        workdir = Path(getattr(dataset, "path", Path("."))).parent
        path = workdir / "codebook.json"
        if not path.exists():
            return False, "codebook.json not found"
        try:
            cb = Codebook.load(path)
        except Exception as e:  # noqa: BLE001
            return False, f"codebook is invalid: {e}"
        if len(cb.variables) < self.min_variables:
            return False, f"only {len(cb.variables)} variables (need {self.min_variables})"
        ratio = cb.numeric_or_boolean_ratio()
        if ratio < self.min_numeric_or_boolean_ratio:
            return False, (
                f"numeric/boolean ratio {ratio:.0%} < target {self.min_numeric_or_boolean_ratio:.0%}"
            )
        return True, f"OK ({len(cb.variables)} variables, {ratio:.0%} numeric/boolean)"


@dataclass
class CoverageContract(Contract):
    """Each codebook variable must have ≥ min_coverage non-null values."""

    min_coverage: float = 0.5
    name: str = "coverage"

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"Every codebook variable must have ≥{self.min_coverage:.0%} non-null coverage."

    def check(self, dataset: Any) -> tuple[bool, str]:
        from pathlib import Path
        from gemma_miner.codebook import Codebook

        workdir = Path(getattr(dataset, "path", Path("."))).parent
        path = workdir / "codebook.json"
        if not path.exists():
            return False, "codebook.json not found"
        try:
            cb = Codebook.load(path)
        except Exception as e:  # noqa: BLE001
            return False, f"codebook invalid: {e}"
        rows = dataset.rows() if hasattr(dataset, "rows") else list(dataset)
        if not rows:
            return False, "0 rows"
        below: list[tuple[str, float]] = []
        for v in cb.variables:
            n = sum(1 for r in rows if r.get(v.name) not in (None, ""))
            cov = n / len(rows)
            if cov < self.min_coverage:
                below.append((v.name, cov))
        if not below:
            return True, f"OK (all variables ≥{self.min_coverage:.0%})"
        sample = ", ".join(f"{n}={c:.0%}" for n, c in below[:5])
        return False, f"{len(below)} variables below threshold: {sample}"


@dataclass
class CustomContract(Contract):
    """Wrap a callable as a contract. `fn(dataset) -> (bool, str)`."""

    name: str
    description: str
    fn: Any  # Callable[[Any], tuple[bool, str]]

    def check(self, dataset: Any) -> tuple[bool, str]:
        return self.fn(dataset)


class ContractBook:
    """Mutable collection of contracts attached to a run."""

    def __init__(self, contracts: list[Contract] | None = None):
        self._contracts: list[Contract] = list(contracts or [])

    def add(self, contract: Contract) -> None:
        # replace any existing contract with the same name
        self._contracts = [c for c in self._contracts if c.name != contract.name]
        self._contracts.append(contract)

    def remove(self, name: str) -> bool:
        n = len(self._contracts)
        self._contracts = [c for c in self._contracts if c.name != name]
        return len(self._contracts) < n

    def list(self) -> list[Contract]:
        return list(self._contracts)

    def status(self, dataset: Any) -> list[dict]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "ok": (res := c.check(dataset))[0],
                "detail": res[1],
            }
            for c in self._contracts
        ]

    def all_satisfied(self, dataset: Any) -> bool:
        return all(c.check(dataset)[0] for c in self._contracts)
