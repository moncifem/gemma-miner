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


def _python_strings_to_json(s: str) -> str:
    """Convert Python-style single-quoted string LITERALS to JSON double-quoted.

    Small models (especially 7-8B) often write tool args like:
        {"regex": '.*?<a>'}
    which is invalid JSON. We walk the text in a state machine and convert
    every Python-style single-quoted string to a JSON double-quoted string,
    while leaving alone:
      - apostrophes inside double-quoted strings
      - already-valid JSON
    """
    out: list[str] = []
    n = len(s)
    i = 0
    in_dq = False     # inside a JSON double-quoted string
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            # preserve the next char verbatim
            out.append(c)
            out.append(s[i + 1])
            i += 2
            continue
        if c == '"' and not in_dq:
            in_dq = True
            out.append(c)
            i += 1
            continue
        if c == '"' and in_dq:
            in_dq = False
            out.append(c)
            i += 1
            continue
        if c == "'" and not in_dq:
            # Find the closing single quote (respecting escapes).
            j = i + 1
            while j < n:
                if s[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if s[j] == "'":
                    break
                j += 1
            if j >= n:
                # unterminated; leave as-is
                out.append(c)
                i += 1
                continue
            inner = s[i + 1 : j]
            # Inside the inner content, escape any unescaped double quotes
            # because they're about to live inside a JSON string literal.
            repaired = inner.replace("\\'", "'")           # unescape Python \' → '
            repaired = re.sub(r'(?<!\\)"', r'\\"', repaired)  # escape unescaped "
            out.append('"')
            out.append(repaired)
            out.append('"')
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


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


def _bracket_balance(s: str) -> tuple[int, int]:
    """Return (unclosed_braces, unclosed_brackets) ignoring chars inside strings.

    Walks the source with a small state machine so braces appearing inside
    JSON string literals (e.g. inside a regex value) don't throw off the
    count.
    """
    open_b = 0
    open_s = 0
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and in_str:
            i += 2
            continue
        if c == '"':
            in_str = not in_str
            i += 1
            continue
        if not in_str:
            if c == "{":
                open_b += 1
            elif c == "}":
                open_b -= 1
            elif c == "[":
                open_s += 1
            elif c == "]":
                open_s -= 1
        i += 1
    return max(0, open_b), max(0, open_s)


def _looks_truncated(s: str) -> bool:
    """Heuristic: text appears cut off mid-JSON (max_tokens hit)."""
    s = s.rstrip()
    if not s:
        return False
    b, sb = _bracket_balance(s)
    return b > 0 or sb > 0


def _autoclose(s: str) -> str:
    """Append the closing braces/brackets needed to balance `s`.

    Handles the common case where the model emits valid JSON except for the
    final closing braces (e.g. Ollama drops the last 1-3 chars). Append-only
    repair; if the JSON had unbalanced internal structure, downstream parse
    still fails and we fall through to truncation rescue.
    """
    s = s.rstrip()
    if not s:
        return s
    # If the last char is a comma, drop it (likely trailing comma before cut).
    s = re.sub(r",\s*$", "", s)
    # If the last meaningful char is inside an open string, close the string.
    open_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            i += 2
            continue
        if c == '"':
            open_str = not open_str
        i += 1
    if open_str:
        s += '"'
    # Now balance braces/brackets.
    b, sb = _bracket_balance(s)
    s += "}" * b + "]" * sb
    return s


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
        # Try increasingly aggressive repairs. Small models (Gemma 4 8B/E4B
        # in particular) routinely emit Python-style single-quoted strings,
        # invalid `\X` regex escapes, trailing commas, and missing final
        # closing braces. We try each repair (and combinations) in order.
        sq = _python_strings_to_json(cand)
        repaired = _repair_invalid_escapes(cand)
        repaired_sq = _repair_invalid_escapes(sq)
        ac = _autoclose(cand)
        ac_sq = _autoclose(sq)
        ac_rep = _autoclose(repaired)
        ac_rep_sq = _autoclose(repaired_sq)
        variants = (
            cand,
            _strip_trailing_commas(cand),
            repaired,
            _strip_trailing_commas(repaired),
            sq,
            _strip_trailing_commas(sq),
            repaired_sq,
            _strip_trailing_commas(repaired_sq),
            # Autoclose variants — handle Ollama dropping the last few } chars.
            ac,
            _strip_trailing_commas(ac),
            ac_sq,
            ac_rep,
            ac_rep_sq,
            _strip_trailing_commas(ac_rep_sq),
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
