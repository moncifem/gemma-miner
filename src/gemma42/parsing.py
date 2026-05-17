"""Robust parsing of model output into a tool call.

Small models often produce JSON that is *almost* right: wrapped in ```json fences,
followed by chatter, missing a comma, etc. We try several strategies in order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    thought: str
    tool: str
    args: dict


class ParseError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json|tool|tool_call)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


# JSON's valid escape characters after `\`. Anything else is a syntax error.
_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def _repair_invalid_escapes(s: str) -> str:
    r"""Escape `\X` sequences that JSON would reject (e.g. `\s`, `\d`).

    Small models love writing regex strings as JSON values, e.g.
    `{"regex": "\\s*([^<]+)"}`. The model often forgets to double the
    backslash and writes `{"regex": "\s*([^<]+)"}`. The latter is invalid
    JSON (the `\s` is not a recognized escape). We walk the string and
    convert any `\X` where X is not a valid JSON escape into `\\X`.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt not in _VALID_JSON_ESCAPES:
                out.append("\\\\")
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _looks_truncated(s: str) -> bool:
    """Heuristic: text appears cut off mid-JSON (max_tokens hit)."""
    s = s.rstrip()
    if not s:
        return False
    # Doesn't end with a closing brace/bracket of the top-level value.
    open_b = s.count("{") - s.count("}")
    open_s = s.count("[") - s.count("]")
    if open_b > 0 or open_s > 0:
        return True
    # Ends mid-string literal (rough check: odd number of unescaped quotes
    # since the last newline).
    return False


def _try_recover_truncated(s: str) -> str | None:
    """Attempt to repair JSON that was cut off mid-string.

    Strategy: find the largest prefix that parses, then synthesise a partial
    object that at least contains the tool name.
    """
    # Try cutting after the tool name; ignore args entirely.
    m = re.search(r'"(?:tool|name|action)"\s*:\s*"([^"]+)"', s)
    if not m:
        return None
    tool = m.group(1)
    # Try to extract a thought if present.
    t = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    thought = t.group(1) if t else ""
    return json.dumps({"thought": thought, "tool": tool, "args": {}})


def _candidates(text: str) -> list[str]:
    """Yield candidate JSON-ish substrings, best-first."""
    out: list[str] = []
    for m in _TAG_RE.findall(text):
        out.append(m.strip())
    for m in _FENCE_RE.findall(text):
        out.append(m.strip())
    # Largest balanced {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
    out.append(text.strip())
    return out


def parse_tool_call(text: str) -> ToolCall:
    """Parse a model reply into a (thought, tool, args) triple.

    The model is instructed to output JSON like:
        {"thought": "...", "tool": "...", "args": {...}}
    We tolerate a fair amount of malformedness.
    """
    last_err: Exception | None = None
    for cand in _candidates(text):
        variants = (
            cand,
            _strip_trailing_commas(cand),
            _repair_invalid_escapes(cand),
            _strip_trailing_commas(_repair_invalid_escapes(cand)),
        )
        for variant in variants:
            try:
                obj = json.loads(variant)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if not isinstance(obj, dict):
                continue
            tool = obj.get("tool") or obj.get("name") or obj.get("action")
            if not tool:
                continue
            args = obj.get("args") or obj.get("arguments") or obj.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_value": args}
            thought = obj.get("thought") or obj.get("reasoning") or ""
            return ToolCall(thought=str(thought), tool=str(tool), args=args)

    # Truncation rescue: if the reply LOOKS like an unfinished JSON tool call,
    # extract at least the tool name so we can surface a useful error.
    if _looks_truncated(text):
        repaired = _try_recover_truncated(text)
        if repaired:
            try:
                obj = json.loads(repaired)
                raise ParseError(
                    f"TRUNCATED: your previous reply was cut off mid-JSON "
                    f"(your tool was '{obj['tool']}'). This happens when you "
                    f"inline very large content into args. Use $file references "
                    f'instead: {{"some_field": {{"$file": "items/.../x.txt"}}}}'
                )
            except json.JSONDecodeError:
                pass

    raise ParseError(f"could not parse a tool call from model output: {last_err}")
