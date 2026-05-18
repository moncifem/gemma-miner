"""Deterministic value coercion — eats the variance of small-model outputs.

A small open model will write "Oui", "yes", "1", 'TRUE' for the same boolean.
It will write "27 millions d'euros", "27,000,000", "27000000 €" for the same
integer. It will write "16 avril 2026", "16/04/2026", "2026-04-16T12:00:00Z"
for the same date.

This module converts any of those to the canonical representation specified
by a `VariableSpec`. Failure returns None — the validator decides whether
that violates a `required` contract.

Everything is stdlib regex. No external libraries.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from gemma_miner.codebook import VariableSpec


# ── booleans ───────────────────────────────────────────────────────────────


_BOOL_TRUE = {
    "true", "t", "yes", "y",
    "oui", "vrai", "v",
    "sí", "si", "verdadero",
    "ja",
    "正", "是",
}
_BOOL_FALSE = {
    "false", "f", "no", "n",
    "non", "faux",
    "nein",
    "否", "非",
}
# Ambiguous tokens the LLM emits when it didn't actually find a signal.
# We treat these as `None` to preserve the null-not-false discipline:
# 0/1 are NOT booleans here, "unknown"/"n/a"/"unstated"/"non précisé" → None.
_BOOL_UNKNOWN = {
    "", "null", "none", "n/a", "na", "unknown", "unspecified",
    "unstated", "not stated", "not specified", "not mentioned",
    "non précisé", "non specifie", "inconnu", "indéterminé",
}


def _coerce_boolean(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    # 0/1 (and floats) are AMBIGUOUS: the LLM might mean "false/true" or it
    # might be a count it got the type wrong on. Refuse to guess — return None.
    if isinstance(v, (int, float)):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _BOOL_UNKNOWN:
            return None
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
    return None


# ── numbers ────────────────────────────────────────────────────────────────


_NUMBER_RE = re.compile(r"-?\d[\d\s ,.]*")
_MULTIPLIERS_FR = {
    "millier": 1_000, "milliers": 1_000, "mille": 1_000,
    "million": 1_000_000, "millions": 1_000_000,
    "milliard": 1_000_000_000, "milliards": 1_000_000_000,
    "billion": 1_000_000_000,  # French "billion" = 10^12, but US "billion" = 10^9; we go with 10^9 (US/scientific) by default
}
_MULTIPLIERS_EN = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000, "mn": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000, "bn": 1_000_000_000,
    "t": 1_000_000_000_000, "trillion": 1_000_000_000_000,
}


def _normalize_number_str(s: str) -> str:
    """Strip currency symbols, spaces, narrow nbsp; convert French decimal comma."""
    s = s.replace(" ", " ").replace(" ", " ")
    s = re.sub(r"[€$£¥₹]", " ", s)
    # If looks like "1 234,56" → "1234.56"
    if "," in s and "." not in s:
        if re.fullmatch(r"[\d\s,. \-]+", s.strip()):
            s = s.replace(" ", "").replace(".", "").replace(",", ".")
    elif "," in s and "." in s:
        # Locale-ambiguous: pick the rightmost as decimal separator.
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(" ", "")
    return s


def _detect_multiplier(text: str) -> int:
    """Look for 'million(s)', 'thousand', 'k', 'M', etc. in the surrounding text."""
    low = text.lower()
    for word, mult in {**_MULTIPLIERS_FR, **_MULTIPLIERS_EN}.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            return mult
    return 1


def _coerce_integer(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None

    # Handle textual 'un'/'one' million etc.
    word_repl = {
        "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
        "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six_en": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    lowered = s.lower()
    leading_word = re.match(r"^([a-zéè]+)\s+", lowered)
    if leading_word:
        word = leading_word.group(1)
        if word in word_repl and _detect_multiplier(lowered):
            return word_repl[word] * _detect_multiplier(lowered)

    # Standard pattern: pull the numeric token.
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    num_str = _normalize_number_str(m.group(0))
    try:
        val = float(num_str)
    except ValueError:
        return None
    mult = _detect_multiplier(s)
    val *= mult
    return int(round(val))


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    try:
        val = float(_normalize_number_str(m.group(0)))
    except ValueError:
        return None
    val *= _detect_multiplier(s)
    return val


# ── dates ──────────────────────────────────────────────────────────────────


_FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}
_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _coerce_date(v: Any) -> str | None:
    """Return ISO YYYY-MM-DD, or None."""
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.date().isoformat()
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None

    # ISO formats first: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ, etc.
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None

    # DD/MM/YYYY or DD-MM-YYYY (French / European). Tolerate trailing junk
    # (e.g. " 14h32", " · 2025"). Detect ambiguous DD/MM vs MM/DD by checking
    # whether the FIRST component is > 12 (then it must be a day).
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        # Decide day/month order. Default to DD/MM (Europe/French) unless the
        # first component clearly can't be a day OR the second clearly can't
        # be a month.
        candidates: list[tuple[int, int]] = []
        if 1 <= a <= 31 and 1 <= b <= 12:
            candidates.append((a, b))  # DD/MM
        if a > 12 and 1 <= b <= 31:
            candidates = [(b, a)]  # forced DD/MM (a is the year? no, a>12 so day)
        if 1 <= a <= 12 and b > 12 and b <= 31:
            candidates = [(b, a)]  # MM/DD detected (US)
        for d, mo in candidates:
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                continue

    # "16 avril 2026" or "April 16, 2026"
    low = unicodedata.normalize("NFKD", s.lower())
    m = re.match(r"^(\d{1,2})\s+([a-zéèû]+)\s+(\d{4})$", low)
    if m:
        d, mname, y = int(m.group(1)), m.group(2), int(m.group(3))
        mo = _FR_MONTHS.get(mname) or _EN_MONTHS.get(mname)
        if mo:
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                return None
    m = re.match(r"^([a-z]+)\s+(\d{1,2}),?\s+(\d{4})$", low)
    if m:
        mname, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        mo = _FR_MONTHS.get(mname) or _EN_MONTHS.get(mname)
        if mo:
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                return None
    return None


# ── enums ──────────────────────────────────────────────────────────────────


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _edit_distance(a: str, b: str, *, cap: int) -> int:
    """Standard Levenshtein with early termination beyond `cap`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            v = min(ins, dele, sub)
            cur.append(v)
            if v < best:
                best = v
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _coerce_enum(v: Any, values: list[str]) -> str | None:
    if v is None or not values:
        return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    # Exact.
    if s in values:
        return s
    # Case-insensitive, accent-insensitive.
    norm = _strip_accents(s).lower()
    norm_map = {_strip_accents(c).lower(): c for c in values}
    if norm in norm_map:
        return norm_map[norm]
    # Substring containment (handles "data-security-failure: missing logging"
    # → "data_security_failure" when the LLM appends commentary).
    for n, original in norm_map.items():
        if n and (n in norm or norm in n):
            # Containment must cover ≥60% of the shorter side to count.
            short = min(len(n), len(norm))
            longer = max(len(n), len(norm))
            if short / max(1, longer) >= 0.6:
                return original
    # Edit distance snap. Cap proportional to the candidate length.
    best_dist = None
    best_match: str | None = None
    for n, original in norm_map.items():
        cap = max(2, len(n) // 4)
        d = _edit_distance(norm, n, cap=cap)
        if d <= cap and (best_dist is None or d < best_dist):
            best_dist = d
            best_match = original
    return best_match


# ── arrays ─────────────────────────────────────────────────────────────────


def _coerce_array(v: Any) -> list | None:
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        if not v.strip():
            return []
        # split on common separators
        return [x.strip() for x in re.split(r"[;|,]\s*", v) if x.strip()]
    return None


# ── dispatch ───────────────────────────────────────────────────────────────


def coerce(value: Any, var: VariableSpec) -> Any:
    """Coerce `value` to the type declared by `var`. Returns None on failure."""
    if value is None:
        return None
    if var.type == "boolean":
        return _coerce_boolean(value)
    if var.type == "integer":
        out = _coerce_integer(value)
        return _bound(out, var)
    if var.type == "float":
        out = _coerce_float(value)
        return _bound(out, var)
    if var.type == "date":
        return _coerce_date(value)
    if var.type == "enum":
        return _coerce_enum(value, var.enum_values or [])
    if var.type == "array":
        return _coerce_array(value)
    # string
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def _bound(value: Any, var: VariableSpec) -> Any:
    if value is None:
        return None
    if var.min_value is not None and value < var.min_value:
        return None
    if var.max_value is not None and value > var.max_value:
        return None
    return value


def coerce_row(row: dict, variables: list[VariableSpec]) -> tuple[dict, dict[str, str]]:
    """Coerce every codebook field of `row`. Returns (cleaned_row, warnings)."""
    cleaned = dict(row)
    warnings: dict[str, str] = {}
    for v in variables:
        if v.name in cleaned:
            raw = cleaned[v.name]
            new = coerce(raw, v)
            if raw is not None and new is None:
                warnings[v.name] = f"coercion lost value {raw!r}"
            cleaned[v.name] = new
        elif v.required:
            warnings[v.name] = "required but missing"
            cleaned[v.name] = None
        else:
            cleaned.setdefault(v.name, None)
    return cleaned, warnings
