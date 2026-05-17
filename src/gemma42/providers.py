"""Provider factory.

Every backend supported here speaks the OpenAI chat-completions protocol, so
they all share `LLMClient`; this module is just a thin layer of presets and
sensible defaults for each provider (base URL, default model id, API-key env
var, auth quirks).

Usage:

    from gemma42 import make_llm
    llm = make_llm("ollama", model="gemma3:27b")
    llm = make_llm("together", model="google/gemma-3n-E4B-it")
    llm = make_llm("groq", model="llama-3.1-70b-versatile")

Or by URL — anything OpenAI-compatible just works:

    llm = make_llm("openai-compatible",
                   base_url="http://my-vllm:8000/v1",
                   model="meta-llama/Meta-Llama-3.1-8B-Instruct")
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from gemma42.llm import LLMClient, LLMConfig


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    default_model: str
    api_key_env: str | None
    api_key_required: bool = True
    default_temperature: float = 0.2


PRESETS: dict[str, ProviderPreset] = {
    "together": ProviderPreset(
        name="together",
        base_url="https://api.together.xyz/v1",
        default_model="google/gemma-4-31B-it",
        api_key_env="TOGETHER_API_KEY",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        # Use a 27B-30B class local model by default; user can override
        # ("ollama pull gemma3:27b" or "qwen2.5:32b" etc.).
        default_model="gemma4:31b",
        api_key_env=None,
        api_key_required=False,
    ),
    "groq": ProviderPreset(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.1-70b-versatile",
        api_key_env="GROQ_API_KEY",
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemma-3-27b-it",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "fireworks": ProviderPreset(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/gemma2-27b-it",
        api_key_env="FIREWORKS_API_KEY",
    ),
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    # Generic OpenAI-compatible endpoint. Caller must pass base_url & model.
    "openai-compatible": ProviderPreset(
        name="openai-compatible",
        base_url="",
        default_model="",
        api_key_env="OPENAI_API_KEY",
        api_key_required=False,
    ),
}


def list_providers() -> list[str]:
    return list(PRESETS.keys())


def make_llm(
    provider: str = "together",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> LLMClient:
    """Create an `LLMClient` from a provider name and optional overrides.

    Args:
      provider: one of `list_providers()`.
      model: model id; falls back to the provider's default.
      base_url: override the provider's base URL (use for self-hosted forks).
      api_key: explicit key; otherwise the provider's env var is read.
      temperature, max_tokens, timeout: passed to LLMConfig.
      extra_headers: optional headers (e.g. for OpenRouter HTTP-Referer).
    """
    key = provider.lower().replace("_", "-")
    if key not in PRESETS:
        raise ValueError(
            f"unknown provider {provider!r}. Available: {', '.join(list_providers())}"
        )
    preset = PRESETS[key]

    final_base = base_url or preset.base_url
    if not final_base:
        raise ValueError(
            f"provider {provider!r} requires base_url (it has no preset URL)."
        )
    final_model = model or preset.default_model
    if not final_model:
        raise ValueError(f"provider {provider!r} requires a model id.")

    final_key = api_key
    if final_key is None and preset.api_key_env:
        final_key = os.getenv(preset.api_key_env)
    if final_key is None and preset.api_key_required:
        raise RuntimeError(
            f"provider {provider!r} requires an API key. "
            f"Set ${preset.api_key_env} or pass api_key=..."
        )
    # Some local servers (ollama, vllm) reject empty Authorization; use a
    # placeholder so the header is well-formed.
    if final_key is None:
        final_key = "not-needed"

    config = LLMConfig(
        model=final_model,
        base_url=final_base,
        api_key=final_key,
        temperature=preset.default_temperature if temperature is None else temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_headers=extra_headers or {},
    )
    return LLMClient(config)
