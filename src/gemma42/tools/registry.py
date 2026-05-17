"""Tool registry & default tool set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma42.refs import resolve_refs
from gemma42.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from gemma42.state import AgentState


# Tools that OPT IN to $file reference resolution. For everything else, the
# model's arg strings pass through verbatim (because most tools take paths or
# code, not file content, and resolving $file would corrupt them).
_REF_RESOLVE_ALLOWED = {
    "dataset_append",
    "extract_structured",
    "write_file",
    "codebook_propose",
    "codebook_test",
    "extract_items",
    "codebook_design",
    "audit_dataset",
    "dataset_report",
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
        # Expand {"$file": "..."} references — but only for tools that ASKED
        # for it. Most tools take paths/code, and resolving $file would
        # corrupt those args (a 96KB HTML body would replace a one-line path).
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

    `llm` drives the agentic/reconnaissance tools. `extraction_llm` is used
    only for schema-constrained extraction tools; when omitted we preserve the
    historical single-client behavior.
    """
    from gemma42.tools.contract_tool import AddContractTool, ContractStatusTool
    from gemma42.tools.dataset_tool import (
        DatasetAppendTool,
        DatasetFromQueueTool,
        DatasetSampleTool,
        DatasetStatsTool,
    )
    from gemma42.tools.autobiography_tool import (
        AutobiographyLessonTool,
        AutobiographyRecallTool,
        AutobiographyStatsTool,
    )
    from gemma42.tools.cloud_tool import CloudPullTool, CloudPushTool, CloudSearchTool
    from gemma42.tools.codebook_tool import (
        CodebookEditTool,
        CodebookProposeTool,
        CodebookShowTool,
        CodebookTestTool,
    )
    from gemma42.tools.constitution_tool import (
        DatasetVerifyTool,
        RuleAddTool,
        RuleListTool,
        RulesInferTool,
    )
    from gemma42.tools.dashboard_tool import DashboardStartTool
    from gemma42.tools.fingerprint_tool import FingerprintCheckTool, RecipeSaveTool
    from gemma42.tools.gdt_tool import GoalTreeTool
    from gemma42.tools.kernel_tool import KernelResetTool, PyKernelTool, SkillPromoteTool
    from gemma42.tools.manifest_tool import DatasetDiffTool, ManifestWriteTool
    from gemma42.tools.swarm_tool import (
        AuditDatasetTool,
        CodebookDesignTool,
        DatasetReportTool,
    )
    from gemma42.tools.export_tool import (
        DatasetExportTool,
        DatasetValidateTool,
        HFPushTool,
    )
    from gemma42.tools.extract_items_tool import ExtractItemsTool
    from gemma42.tools.extract_text_tool import ExtractTextTool
    from gemma42.tools.extract_tool import ExtractStructuredTool
    from gemma42.tools.extractor_tool import (
        ExtractorDefineTool,
        ProcessQueueTool,
        ScrapePaginatedTool,
    )
    from gemma42.tools.file_tool import ListDirTool, ReadFileTool, WriteFileTool
    from gemma42.tools.item_tool import SaveAttachmentTool
    from gemma42.tools.queue_tool import (
        QueueAddTool,
        QueueMarkDoneTool,
        QueueNextTool,
        QueueStatusTool,
    )
    from gemma42.tools.finish_tool import FinishTool
    from gemma42.tools.assess_sample_tool import AssessSampleTool
    from gemma42.tools.discover_assets_tool import DiscoverAssetsTool
    from gemma42.tools.html_tool import HtmlExtractTool, HtmlFindTool, HtmlInspectTool
    from gemma42.tools.http_tool import HttpGetTool
    from gemma42.tools.memory_tool import MemoryGetTool, MemoryListTool, MemorySetTool
    from gemma42.tools.plan_tool import SetPlanTool, ShowPlanTool
    from gemma42.tools.shell_tool import BashTool, PythonExecTool

    tools: list[Tool] = [
        # Fetch & inspect
        HttpGetTool(),
        HtmlInspectTool(),
        HtmlExtractTool(),
        HtmlFindTool(),
        DiscoverAssetsTool(),
        # Declarative extraction (the preferred path for scraping)
        ExtractorDefineTool(),
        ScrapePaginatedTool(),
        ProcessQueueTool(),
        # Code escape hatches
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
        AssessSampleTool(),
        # Queue
        QueueAddTool(),
        QueueNextTool(),
        QueueMarkDoneTool(),
        QueueStatusTool(),
        # Memory
        MemoryGetTool(),
        MemorySetTool(),
        MemoryListTool(),
        # Plan (set once after discovery; followed for the rest of the run)
        SetPlanTool(),
        ShowPlanTool(),
        # Validation & export
        DatasetValidateTool(),
        DatasetExportTool(),
        HFPushTool(),
        # Constitution
        RuleAddTool(),
        RuleListTool(),
        RulesInferTool(),
        DatasetVerifyTool(),
        # Goal tree
        GoalTreeTool(),
        # Autobiography (L3 + L4)
        AutobiographyStatsTool(),
        AutobiographyRecallTool(),
        AutobiographyLessonTool(),
        # Fingerprint + recipe cache
        FingerprintCheckTool(),
        RecipeSaveTool(),
        # Persistent Python kernel
        PyKernelTool(),
        KernelResetTool(),
        SkillPromoteTool(),
        # Provenance + manifest + diff
        ManifestWriteTool(),
        DatasetDiffTool(),
        # Federated cloud
        CloudSearchTool(),
        CloudPushTool(),
        CloudPullTool(),
        # Live dashboard
        DashboardStartTool(),
        # Contracts & finish
        AddContractTool(),
        ContractStatusTool(),
        FinishTool(),
    ]
    if llm is not None:
        from gemma42.tools.llm_scrape_tool import LLMScrapeTool

        extractor_llm = extraction_llm or llm
        tools.append(ExtractStructuredTool(llm=extractor_llm))
        tools.append(CodebookProposeTool(llm=llm))
        tools.append(CodebookTestTool(llm=extractor_llm))
        tools.append(ExtractItemsTool(llm=extractor_llm))
        tools.append(LLMScrapeTool(llm=llm))
        # Specialist swarm (LLM-driven)
        tools.append(CodebookDesignTool(llm=llm))
        tools.append(AuditDatasetTool(llm=llm))
        tools.append(DatasetReportTool(llm=llm))
    # The non-LLM codebook tools
    tools.append(CodebookShowTool())
    tools.append(CodebookEditTool())
    return ToolRegistry(tools)
