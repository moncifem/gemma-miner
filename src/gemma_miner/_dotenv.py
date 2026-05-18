"""Tiny dotenv loader — zero-dependency.

Looks for `.env` upward from the cwd; if found, loads `KEY=VALUE` lines into
`os.environ` without overwriting anything already set. Quotes (single and
double) are stripped from values. Lines starting with `#` and blanks are
ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: str | Path | None = None) -> bool:
    here = Path(start or os.getcwd()).resolve()
    for d in [here, *here.parents]:
        p = d / ".env"
        if p.is_file():
            _apply(p)
            return True
    return False


def _apply(path: Path) -> None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass
