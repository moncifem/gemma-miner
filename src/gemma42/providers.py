"""Provider factory.

Every backend supported here speaks the OpenAI chat-completions protocol, so
they all share `LLMClient`; this module is just a thin layer of presets and
sensible defaults for each provider (base URL, default model id, API-key env
var, auth quirks).

Usage:

    from gemma42 import make_llm
    llm = make_llm("ollama", model="gemma4:latest")
    llm = make_llm("openrouter", model="google/gemini-3.1-pro-preview")
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
        default_model="gemma4:latest",
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
        default_model="google/gemini-3.1-flash-lite",
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


def auto_provider() -> str:
    """Pick a sensible default provider based on the environment.

    Order:
      1. OPENROUTER_API_KEY set     → 'openrouter'
      2. TOGETHER_API_KEY set       → 'together'
      3. ollama daemon reachable    → 'ollama'
      4. GROQ_API_KEY set           → 'groq'
      5. OPENAI_API_KEY set         → 'openai'
      6. fallback                   → 'ollama' (will fail loudly if not running)
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("TOGETHER_API_KEY"):
        return "together"
    # Cheap reachability probe for the local ollama daemon.
    try:
        import socket

        with socket.create_connection(("localhost", 11434), timeout=0.2):
            return "ollama"
    except OSError:
        pass
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


# Known context windows per model family. Used for Ollama's `options.num_ctx`
# and as a hint for chunking. Tokens, not characters.
_CONTEXT_WINDOWS: dict[str, int] = {
    # Gemma 4 family (per Google: 128K for the 8B/E variants, 256K for 31B)
    "gemma4:latest":        128_000,
    "gemma4:8b":            128_000,
    "gemma4:e4b":           128_000,
    "gemma4:26b":           128_000,
    "gemma4:31b":           256_000,
    "google/gemma-4-31b-it": 256_000,
    "google/gemma-4-e4b-it": 128_000,
    "google/gemma-4-26b-a4b-it": 128_000,
    # Common alternatives so users don't have to pass it manually
    "llama-3.1-70b-versatile": 128_000,
    "llama-3.1-8b-instant":    128_000,
    "gpt-4o-mini":            128_000,
    "gpt-4o":                 128_000,
    "google/gemini-3.1-pro-preview": 1_048_576,
    "google/gemini-3.1-pro-preview-customtools": 1_048_576,
    "google/gemini-3.1-flash-lite": 1_048_576,
    "google/gemini-3.1-flash": 1_048_576,
    "google/gemini-3.1-pro": 1_048_576,
}


def _context_window_for(model: str) -> int:
    m = (model or "").lower()
    if m in _CONTEXT_WINDOWS:
        return _CONTEXT_WINDOWS[m]
    # Substring match for family detection.
    for k, v in _CONTEXT_WINDOWS.items():
        if k in m or m in k:
            return v
    return 128_000   # sane default for modern models


def make_llm(
    provider: str = "together",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 16384,
    context_window: int | None = None,
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

    resolved_ctx = context_window or _context_window_for(final_model)
    config = LLMConfig(
        model=final_model,
        base_url=final_base,
        api_key=final_key,
        temperature=preset.default_temperature if temperature is None else temperature,
        max_tokens=max_tokens,
        context_window=resolved_ctx,
        timeout=timeout,
        extra_headers=extra_headers or {},
    )
    return LLMClient(config)
