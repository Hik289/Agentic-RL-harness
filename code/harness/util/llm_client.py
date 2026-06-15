"""Azure OpenAI gpt-5.4-mini client wrapper for the harness project.

All LLM calls in this project go through this wrapper.

Reads credentials from env:
  AZURE_OPENAI_ENDPOINT  (e.g. https://<your-resource>.services.ai.azure.com/openai/v1)
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_DEPLOYMENT  (default: gpt-5.4-mini)
  AZURE_OPENAI_API_VERSION (default: 2024-12-01-preview; falls back to /openai/v1 path)

The endpoint provided ends in '/openai/v1' -> we use the OpenAI-compatible
v1 path (the modern Azure 'Responses API' / 'v1' surface). We therefore use
the regular `openai.OpenAI` client with base_url, not AzureOpenAI.

Pricing (approx, for cost tracking only; verify against Azure portal):
  Input  : 0.25 USD / 1M tokens
  Output : 2.00 USD / 1M tokens
(gpt-5.4-mini placeholder rate; refine when real pricing confirmed.)
"""
from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError

log = logging.getLogger(__name__)


DEFAULT_ENDPOINT = "https://<your-resource>.services.ai.azure.com/openai/v1"
DEFAULT_DEPLOYMENT = "gpt-5.4-mini"

# Placeholder pricing for cost estimation. Refine when Azure billing confirmed.
PRICE_INPUT_PER_1M = float(os.getenv("AZURE_PRICE_INPUT_PER_1M", "0.25"))
PRICE_OUTPUT_PER_1M = float(os.getenv("AZURE_PRICE_OUTPUT_PER_1M", "2.00"))


@dataclass
class CallResult:
    ok: bool
    text: str = ""
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    attempts: int = 1
    error: Optional[str] = None
    raw: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


class LLMClient:
    """Thin wrapper around Azure gpt-5.4-mini via OpenAI v1-compatible API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.endpoint = (endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY env var or api_key arg required.")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_DEPLOYMENT
        self.timeout = timeout
        self.max_retries = max_retries

        # The endpoint ends in /openai/v1 — point OpenAI SDK at it directly.
        self._client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,  # we do our own retry
        )

    @staticmethod
    def _estimate_cost(in_tokens: int, out_tokens: int) -> float:
        return (in_tokens / 1_000_000.0) * PRICE_INPUT_PER_1M + (
            out_tokens / 1_000_000.0
        ) * PRICE_OUTPUT_PER_1M

    def chat(
        self,
        messages: list[dict],
        max_completion_tokens: int = 256,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> CallResult:
        """Call chat.completions with retry. Returns CallResult."""
        params: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if response_format is not None:
            params["response_format"] = response_format
        if seed is not None:
            params["seed"] = seed
        if extra:
            params.update(extra)

        attempt = 0
        last_err: Optional[str] = None
        t0 = time.monotonic()
        while attempt < self.max_retries:
            attempt += 1
            try:
                resp = self._client.chat.completions.create(**params)
                latency = time.monotonic() - t0
                choice = resp.choices[0]
                txt = choice.message.content or ""
                usage = resp.usage
                in_t = getattr(usage, "prompt_tokens", 0) or 0
                out_t = getattr(usage, "completion_tokens", 0) or 0
                tot_t = getattr(usage, "total_tokens", in_t + out_t) or (in_t + out_t)
                return CallResult(
                    ok=True,
                    text=txt,
                    finish_reason=choice.finish_reason,
                    prompt_tokens=in_t,
                    completion_tokens=out_t,
                    total_tokens=tot_t,
                    cost_usd=self._estimate_cost(in_t, out_t),
                    latency_s=latency,
                    attempts=attempt,
                    raw=None,
                )
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_err = f"{type(e).__name__}: {e}"
                log.warning("chat retry %d/%d: %s", attempt, self.max_retries, last_err)
                time.sleep(min(2 ** attempt, 8))
            except APIError as e:
                last_err = f"{type(e).__name__}: {e}"
                log.warning("chat APIError attempt %d: %s", attempt, last_err)
                time.sleep(min(2 ** attempt, 8))
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                log.exception("chat unexpected error")
                time.sleep(min(2 ** attempt, 8))
        return CallResult(
            ok=False,
            error=last_err,
            latency_s=time.monotonic() - t0,
            attempts=attempt,
        )


def get_default_client() -> LLMClient:
    return LLMClient()
