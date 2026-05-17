"""gemma42 — structured scraping & dataset-construction agent for small open models."""

from gemma42.agent import Agent, run_agent
from gemma42.codebook import Codebook, VariableSpec
from gemma42.contracts import (
    CodebookContract,
    Contract,
    CoverageContract,
    FieldsContract,
    MinRowsContract,
    UniqueFieldContract,
)
from gemma42.dataset import Dataset
from gemma42.llm import LLMClient
from gemma42.memory import Memory
from gemma42.providers import PRESETS, list_providers, make_llm
from gemma42.state import AgentState

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentState",
    "Contract",
    "Dataset",
    "FieldsContract",
    "LLMClient",
    "Codebook",
    "CodebookContract",
    "CoverageContract",
    "Memory",
    "MinRowsContract",
    "PRESETS",
    "UniqueFieldContract",
    "VariableSpec",
    "list_providers",
    "make_llm",
    "run_agent",
]
