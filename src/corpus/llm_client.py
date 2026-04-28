"""Async OpenAI wrapper with retry, JSON-mode, and explicit env loading.

Looks for the API key in this order:
  1. rankforge/.env                                    (gitignored; preferred)
  2. ~/Documents/saas-app-profiler/.env.local          (legacy fallback)
  3. process environment (e.g. exported in the shell)

The key never lands in this repo's git history because `.env` is gitignored.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, RateLimitError, APITimeoutError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CANDIDATES = (
    REPO_ROOT / ".env",
    Path("/Users/avinashsingh/Documents/saas-app-profiler/.env.local"),
)
DEFAULT_MODEL = "gpt-4o-mini"
PLACEHOLDER_PREFIX = "sk-proj-REPLACE_ME"

_client: AsyncOpenAI | None = None
_model: str | None = None


def _load_env() -> tuple[str, str]:
    """Return (api_key, model). Raises if API key missing or still a placeholder."""
    loaded_from: Path | None = None
    for candidate in ENV_CANDIDATES:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            if loaded_from is None:
                loaded_from = candidate

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Looked in: "
            + ", ".join(str(p) for p in ENV_CANDIDATES)
            + " and the process environment."
        )
    if api_key.startswith(PLACEHOLDER_PREFIX):
        raise RuntimeError(
            f"OPENAI_API_KEY is still the placeholder. Edit {loaded_from} and "
            "replace sk-proj-REPLACE_ME with your real key."
        )

    model = os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    if loaded_from:
        logger.info("loaded env from %s (model=%s)", loaded_from, model)
    return api_key, model


def get_client(model_override: str | None = None) -> tuple[AsyncOpenAI, str]:
    """Lazily build a singleton AsyncOpenAI client + return resolved model name."""
    global _client, _model
    if _client is None:
        api_key, env_model = _load_env()
        _client = AsyncOpenAI(api_key=api_key)
        _model = env_model
    model = model_override or _model or DEFAULT_MODEL
    return _client, model


_RETRYABLE = (RateLimitError, APITimeoutError, APIStatusError)


async def chat_json(
    *,
    system: str,
    user: str,
    required_keys: list[str],
    model_override: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Call OpenAI chat completion in JSON mode and return the parsed dict.

    Retries on rate limits, timeouts, and 5xx with exponential backoff (max 5 attempts).
    Validates that all `required_keys` are present in the parsed response.
    """
    client, model = get_client(model_override)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(_RETRYABLE + (json.JSONDecodeError, ValueError)),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    ):
        with attempt:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"Missing required keys {missing} in LLM output: {raw[:200]}")
            return data

    raise RuntimeError("unreachable: AsyncRetrying did not return")
