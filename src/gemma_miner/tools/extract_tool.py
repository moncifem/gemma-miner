"""LLM-as-a-tool: extract structured fields from a chunk of text against a JSON schema.

This is how the agent handles complex extraction tasks like the French
competition-authority decisions: it points us at the decision text and a JSON
schema, and we drive the same LLM under a strict extraction prompt to return
one validated object. The dataset's append step validates it again against
its own schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gemma_miner.dataset import _validate_against_schema
from gemma_miner.parsing import _candidates, _strip_trailing_commas
from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.llm import LLMClient
    from gemma_miner.state import AgentState


_SYSTEM = (
    "You are a strict information-extraction engine. Read the provided TEXT "
    "and produce a single JSON object that conforms exactly to the provided "
    "JSON Schema. Rules:\n"
    " - Output JSON only. No prose, no fences, no commentary.\n"
    " - If a field is not stated in the text, return null (not a guess).\n"
    " - Respect every type, enum, and 'required' list in the schema.\n"
    " - For arrays, return [] when none are mentioned.\n"
    " - Dates must be ISO 8601 (YYYY-MM-DD) when the schema specifies a date.\n"
)


class ExtractStructuredTool(Tool):
    name = "extract_structured"
    description = (
        "Extract structured fields from a text blob using a JSON schema. "
        "Provide either `text` directly OR a `path` to a file in the workdir. "
        "Returns the extracted JSON object and a validation report. Use this "
        "when the data you need is buried in prose (PDF transcript, legal "
        "decision, article body) rather than in a clean HTML table."
    )
    args_schema = {
        "schema": {
            "type": "object",
            "description": "JSON Schema (object) describing the fields to extract.",
        },
        "text": {"type": "string", "description": "Source text. Optional if 'path' given."},
        "path": {"type": "string", "description": "Path to a text file in the workdir."},
        "max_input_chars": {
            "type": "integer",
            "default": 24000,
            "description": "Truncate source text to this many chars before calling the model.",
        },
    }

    def __init__(self, llm: "LLMClient"):
        self.llm = llm

    def run(self, args: dict, state: "AgentState") -> ToolResult:
        schema = args.get("schema")
        if not isinstance(schema, dict):
            return ToolResult(output="ERROR: 'schema' (object) is required", error=True)

        text = args.get("text")
        if not text and args.get("path"):
            p = Path(args["path"])
            if not p.is_absolute():
                p = Path(state.workdir) / p
            if not p.exists():
                return ToolResult(output=f"ERROR: not found: {p}", error=True)
            text = p.read_text(encoding="utf-8", errors="replace")
        if not text:
            return ToolResult(output="ERROR: provide 'text' or 'path'", error=True)

        max_chars = int(args.get("max_input_chars") or 24000)
        if len(text) > max_chars:
            text = text[:max_chars]

        user = (
            "JSON Schema:\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
            "TEXT:\n<<<\n"
            f"{text}\n>>>\n\n"
            "Return one JSON object."
        )
        raw = self.llm.chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        obj = None
        for cand in _candidates(raw):
            for variant in (cand, _strip_trailing_commas(cand)):
                try:
                    obj = json.loads(variant)
                    break
                except Exception:  # noqa: BLE001
                    obj = None
            if obj is not None:
                break
        if obj is None:
            return ToolResult(
                output=f"ERROR: model returned unparseable JSON. raw:\n{raw[:1500]}",
                error=True,
            )

        ok, err = _validate_against_schema(obj, schema)
        report = "valid" if ok else f"INVALID: {err}"
        out = (
            f"validation: {report}\n"
            "--- extracted ---\n"
            f"{json.dumps(obj, indent=2, ensure_ascii=False)}"
        )
        return ToolResult(output=out, artifact=obj, error=not ok)
