"""Tool registry & default tool set.

Only generic, reusable tools live here. The set is deliberately small so the
agent can compose them for ANY scraping / dataset-construction task — no
domain assumptions, no meta-features, no bloat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_miner.refs import resolve_refs
from gemma_miner.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma_miner.state import AgentState


# Tools that OPT IN to $file reference resolution. Everything else passes args
# through verbatim — resolving $file on a path-taking tool would corrupt it.
_REF_RESOLVE_ALLOWED = {
    "dataset_append",
    "extract_structured",
    "write_file",
    "codebook_propose",
    "codebook_test",
    "extract_items",
}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def dispatch(self, name: str, args: dict, state: "AgentState") -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys())
            return ToolResult(
                output=f"ERROR: unknown tool '{name}'. Available: {available}",
                error=True,
            )
        if name in _REF_RESOLVE_ALLOWED:
            try:
                args = resolve_refs(args, state.workdir)
            except Exception as e:  # noqa: BLE001
                return ToolResult(
                    output=f"ERROR resolving $file reference: {e}",
                    error=True,
                )
        try:
            return tool.run(args, state)
        except Exception as e:  # noqa: BLE001
            return ToolResult(output=f"ERROR: {type(e).__name__}: {e}", error=True)


def default_registry(llm=None, extraction_llm=None) -> ToolRegistry:
    """Build the default tool set.

    `llm` drives planning/codebook/scrape tools. `extraction_llm` is used only
    for schema-constrained extraction; defaults to `llm` if not provided.
    """
    from gemma_miner.tools.codebook_tool import (
        CodebookEditTool,
        CodebookProposeTool,
        CodebookShowTool,
        CodebookTestTool,
    )
    from gemma_miner.tools.dataset_tool import (
        DatasetAppendTool,
        DatasetFromQueueTool,
        DatasetSampleTool,
        DatasetStatsTool,
    )
    from gemma_miner.tools.export_tool import (
        DatasetExportTool,
        DatasetValidateTool,
        HFPushTool,
    )
    from gemma_miner.tools.extract_items_tool import ExtractItemsTool
    from gemma_miner.tools.extract_text_tool import ExtractTextTool
    from gemma_miner.tools.extract_tool import ExtractStructuredTool
    from gemma_miner.tools.extractor_tool import (
        ExtractorDefineTool,
        ProcessQueueTool,
        ScrapePaginatedTool,
    )
    from gemma_miner.tools.file_tool import ListDirTool, ReadFileTool, WriteFileTool
    from gemma_miner.tools.finish_tool import FinishTool
    from gemma_miner.tools.html_tool import HtmlExtractTool, HtmlFindTool, HtmlInspectTool
    from gemma_miner.tools.http_tool import HttpGetTool
    from gemma_miner.tools.item_tool import SaveAttachmentTool
    from gemma_miner.tools.memory_tool import MemoryGetTool, MemoryListTool, MemorySetTool
    from gemma_miner.tools.plan_tool import SetPlanTool, ShowPlanTool
    from gemma_miner.tools.queue_tool import (
        QueueAddTool,
        QueueMarkDoneTool,
        QueueNextTool,
        QueueStatusTool,
    )
    from gemma_miner.tools.shell_tool import BashTool, PythonExecTool

    tools: list[Tool] = [
        # Fetch & inspect
        HttpGetTool(),
        HtmlInspectTool(),
        HtmlExtractTool(),
        HtmlFindTool(),
        # Declarative extraction (preferred path for clean repeating HTML)
        ExtractorDefineTool(),
        ScrapePaginatedTool(),
        ProcessQueueTool(),
        # Code escape hatches (generic — any API, any format)
        PythonExecTool(),
        BashTool(),
        # Files
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        ExtractTextTool(),
        SaveAttachmentTool(),
        # Dataset
        DatasetAppendTool(),
        DatasetFromQueueTool(),
        DatasetStatsTool(),
        DatasetSampleTool(),
        # Queue
        QueueAddTool(),
        QueueNextTool(),
        QueueMarkDoneTool(),
        QueueStatusTool(),
        # Memory
        MemoryGetTool(),
        MemorySetTool(),
        MemoryListTool(),
        # Plan
        SetPlanTool(),
        ShowPlanTool(),
        # Validation, export, publish
        DatasetValidateTool(),
        DatasetExportTool(),
        HFPushTool(),
        # Codebook (non-LLM editor tools)
        CodebookShowTool(),
        CodebookEditTool(),
        # Finish
        FinishTool(),
    ]
    if llm is not None:
        from gemma_miner.tools.llm_scrape_tool import LLMScrapeTool

        extractor_llm = extraction_llm or llm
        tools.append(ExtractStructuredTool(llm=extractor_llm))
        tools.append(CodebookProposeTool(llm=llm))
        tools.append(CodebookTestTool(llm=extractor_llm))
        tools.append(ExtractItemsTool(llm=extractor_llm))
        tools.append(LLMScrapeTool(llm=llm))
    return ToolRegistry(tools)
