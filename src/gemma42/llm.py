"""LLM client. OpenAI-compatible — works with Together AI, Groq, OpenRouter, vLLM, etc."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from gemma42._dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    model: str = "google/gemma-3n-E4B-it"
    base_url: str = "https://api.together.xyz/v1"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3
    extra_headers: dict[str, str] = field(default_factory=dict)


class LLMClient:
    """Thin OpenAI-compatible chat client.

    Default points at Together AI. Override `base_url` for any other provider
    that speaks the OpenAI chat-completions protocol.
    """

    def __init__(self, config: LLMConfig | None = None, **overrides: Any):
        self.config = config or LLMConfig()
        for k, v in overrides.items():
            setattr(self.config, k, v)
        if self.config.api_key is None:
            self.config.api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.config.api_key:
            raise RuntimeError(
                "No API key. Pass api_key=... or set TOGETHER_API_KEY / OPENAI_API_KEY."
            )
        self._client = httpx.Client(timeout=self.config.timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stop: list[str] | None = None,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                r = self._client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    content=json.dumps(payload),
                )
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}: {r.text[:200]}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_err}")

    def close(self) -> None:
        self._client.close()
