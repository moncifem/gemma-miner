"""LLM client. OpenAI-compatible — works with Together AI, Groq, OpenRouter, vLLM, etc."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from gemma_miner._dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    model: str = "google/gemma-3n-E4B-it"
    base_url: str = "https://api.together.xyz/v1"
    api_key: str | None = None
    temperature: float = 0.2
    # OUTPUT budget per call. We rarely need more than ~16K tokens of output
    # in a single tool call (specs / row arrays), so we cap here even though
    # the model can produce more.
    max_tokens: int = 16384
    # FULL context window (input + output) the model supports. For Ollama
    # we pass this as `options.num_ctx` so the local daemon allocates a
    # KV cache big enough for the page content + system prompt + brief.
    # 128K covers gemma4:latest (8B) and is a reasonable default for the
    # 31B (256K supported, but 128K is enough for everything we do).
    context_window: int = 128_000
    # Per-call timeout. The previous 120s, combined with 5 retries + exp
    # backoff, could keep the agent hanging for 10+ minutes on a single slow
    # turn. 90s is a more sensible upper bound; the underlying httpx call
    # uses sub-timeouts so we fail FAST on a stalled connection (15s connect,
    # 75s read) instead of waiting for the worst case.
    timeout: float = 90.0
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
        # Use sub-timeouts so a stalled connection fails in ~15s instead of
        # waiting the full read budget. read=timeout-15 lets the model take
        # most of the budget to actually produce tokens.
        connect = min(15.0, self.config.timeout / 4)
        read = max(30.0, self.config.timeout - connect)
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                self.config.timeout,
                connect=connect, read=read,
                write=connect, pool=connect,
            ),
        )

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
        # Ollama's OpenAI-compat layer ignores `max_tokens` for some models
        # and falls back to a small `num_predict` default. Pass the native
        # `options` block so we actually get the full output we ask for,
        # and set `num_ctx` to the model's real context window (128K for
        # gemma4:latest, 256K for gemma4:31b) so the daemon allocates a
        # KV cache that can hold the whole page + brief.
        if "11434" in self.config.base_url or "ollama" in self.config.base_url.lower():
            payload["options"] = {
                "num_predict": self.config.max_tokens,
                "num_ctx":     self.config.context_window,
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
        # Retry policy:
        #   - 5xx + 429 + connection/timeout errors  → retryable (exp. backoff)
        #   - 4xx (401/402/403/404/422 …)            → NOT retryable, fail fast.
        #     These are auth/quota/bad-request errors that retrying won't fix
        #     (the previous code burned 5 retries on every 402 Payment Required
        #      response, wasting ~3 minutes per failed call).
        import random as _r

        for attempt in range(self.config.max_retries):
            try:
                r = self._client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    content=json.dumps(payload),
                )
                if r.status_code >= 500 or r.status_code == 429:
                    # Retryable. Respect Retry-After header if present.
                    retry_after = r.headers.get("retry-after")
                    if retry_after:
                        try:
                            time.sleep(min(60.0, float(retry_after)))
                            continue
                        except ValueError:
                            pass
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}: {r.text[:200]}", request=r.request, response=r
                    )
                if 400 <= r.status_code < 500:
                    # Non-retryable. Build a specific error message and bail.
                    body = (r.text or "")[:400]
                    msg = f"HTTP {r.status_code} from {self.config.base_url} ({body})"
                    if r.status_code == 401:
                        msg += "  →  check your API key env var."
                    elif r.status_code == 402:
                        msg += "  →  out of credits / payment required on the provider."
                    elif r.status_code == 404:
                        msg += "  →  the model id may not exist on this provider."
                    raise RuntimeError(msg)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except RuntimeError:
                raise  # already a final, helpful error
            except (httpx.HTTPError, KeyError, httpx.ReadTimeout, httpx.ConnectError) as e:
                last_err = e
                wait = min(30.0, (2 ** attempt) + _r.uniform(0, 1.0))
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_err}")

    def close(self) -> None:
        self._client.close()
