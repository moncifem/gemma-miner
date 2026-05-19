"""gemma-miner — structured scraping & dataset-construction agent for small open models."""

from gemma_miner.agent import Agent, run_agent
from gemma_miner.codebook import Codebook, VariableSpec
from gemma_miner.contracts import (
    CodebookContract,
    Contract,
    CoverageContract,
    FieldsContract,
    MinRowsContract,
    UniqueFieldContract,
)
from gemma_miner.dataset import Dataset
from gemma_miner.llm import LLMClient
from gemma_miner.memory import Memory
from gemma_miner.providers import PRESETS, list_providers, make_llm
from gemma_miner.state import AgentState

__version__ = "0.1.3"

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
