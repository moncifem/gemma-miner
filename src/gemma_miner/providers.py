"""Provider factory — Gemma + Gemini only.

Gemma Miner is opinionated: it ships first-class presets only for providers
that serve **Gemma** (open) or **Gemini** (Google's hosted) models, because
those are the families the agent is tuned and tested against. Anything else
is still reachable through the generic `openai-compatible` escape hatch.

Supported providers:

  • ollama       — local Gemma (no API key)
  • openrouter   — Gemini 3.1 Flash (cheap, fast)
  • together     — Gemma 4 31B (paid)
  • featherless  — Gemma 4 31B (paid, serverless GPU)

Usage:

    from gemma_miner import make_llm
    llm = make_llm("ollama",      model="gemma4:31b")
    llm = make_llm("openrouter",  model="google/gemini-3.1-flash-lite")
    llm = make_llm("together",    model="google/gemma-4-31b-it")
    llm = make_llm("featherless", model="google/gemma-4-31b-it")

Any other OpenAI-compatible endpoint works through `openai-compatible`:

    llm = make_llm("openai-compatible",
                   base_url="http://my-vllm:8000/v1",
                   model="google/gemma-4-31b-it")
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from gemma_miner.llm import LLMClient, LLMConfig


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    default_model: str
    api_key_env: str | None
    api_key_required: bool = True
    default_temperature: float = 0.2


PRESETS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        # 31B local; the wizard offers a live picker from `ollama tags` and
        # `/gemma-full-local` switches to whichever Gemma the user has pulled.
        default_model="gemma4:31b",
        api_key_env=None,
        api_key_required=False,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3.1-flash-lite",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "together": ProviderPreset(
        name="together",
        base_url="https://api.together.xyz/v1",
        default_model="google/gemma-4-31b-it",
        api_key_env="TOGETHER_API_KEY",
    ),
    "featherless": ProviderPreset(
        name="featherless",
        base_url="https://api.featherless.ai/v1",
        # Featherless is case-sensitive on model ids — use the exact slug
        # they publish on featherless.ai (capital B in "31B").
        default_model="google/gemma-4-31B-it",
        api_key_env="FEATHERLESS_API_KEY",
    ),
    # Generic OpenAI-compatible escape hatch — caller must pass base_url+model.
    "openai-compatible": ProviderPreset(
        name="openai-compatible",
        base_url="",
        default_model="",
        api_key_env=None,
        api_key_required=False,
    ),
}


def list_providers() -> list[str]:
    return list(PRESETS.keys())


def auto_provider() -> str:
    """Pick a sensible default provider based on the environment.

    Order:
      1. OPENROUTER_API_KEY set        → 'openrouter'
      2. TOGETHER_API_KEY set          → 'together'
      3. FEATHERLESS_API_KEY set       → 'featherless'
      4. ollama daemon reachable       → 'ollama'
      5. fallback                      → 'ollama' (will fail loudly if not running)
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("TOGETHER_API_KEY"):
        return "together"
    if os.getenv("FEATHERLESS_API_KEY"):
        return "featherless"
    try:
        import socket

        with socket.create_connection(("localhost", 11434), timeout=0.2):
            return "ollama"
    except OSError:
        pass
    return "ollama"


# Known context windows per model family — Gemma + Gemini only. Used for
# Ollama's `options.num_ctx` and as a chunking hint. Values are TOKENS.
_CONTEXT_WINDOWS: dict[str, int] = {
    # Gemma 4 family — 128K on 8B/26B/E variants, 256K on 31B.
    "gemma4:latest":             128_000,
    "gemma4:8b":                 128_000,
    "gemma4:e4b":                128_000,
    "gemma4:26b":                128_000,
    "gemma4:31b":                256_000,
    "google/gemma-4-31b-it":     256_000,
    "google/gemma-4-e4b-it":     128_000,
    "google/gemma-4-26b-a4b-it": 128_000,
    # Gemini 3.1 family — 1M tokens.
    "google/gemini-3.1-pro-preview":             1_048_576,
    "google/gemini-3.1-pro-preview-customtools": 1_048_576,
    "google/gemini-3.1-flash-lite":              1_048_576,
    "google/gemini-3.1-flash":                   1_048_576,
    "google/gemini-3.1-pro":                     1_048_576,
}


def _context_window_for(model: str) -> int:
    m = (model or "").lower()
    if m in _CONTEXT_WINDOWS:
        return _CONTEXT_WINDOWS[m]
    # Substring family detection (covers e.g. `gemma4:31b-instruct-q4_K_M`).
    for k, v in _CONTEXT_WINDOWS.items():
        if k in m or m in k:
            return v
    return 128_000   # sane default for modern Gemma/Gemini models


def make_llm(
    provider: str = "openrouter",
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
