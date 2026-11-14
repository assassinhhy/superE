"""Secrets come from the environment, never from the repository.

Binance market data does not use a key. DeepSeek, used in a later phase, is
read here so callers have one place that cannot be logged by accident.
"""

from __future__ import annotations

import os

DEEPSEEK_ENV = "DEEPSEEK_API_KEY"
LLM_BASE_URL_ENV = "SUPERE_LLM_BASE_URL"
LLM_API_KEY_ENV = "SUPERE_LLM_API_KEY"
LLM_MODEL_ENV = "SUPERE_LLM_MODEL"
DEFAULT_LLM_MODEL = "deepseek-v4.1-flash"


def deepseek_api_key() -> str | None:
    value = os.environ.get(DEEPSEEK_ENV, "").strip()
    return value or None


def deepseek_configured() -> bool:
    return deepseek_api_key() is not None


def llm_base_url() -> str | None:
    value = os.environ.get(LLM_BASE_URL_ENV, "").strip().rstrip("/")
    return value or None


def llm_api_key() -> str | None:
    """Return the configured key, or None. Callers must not log the result."""
    value = os.environ.get(LLM_API_KEY_ENV, "").strip()
    return value or None


def llm_model() -> str:
    value = os.environ.get(LLM_MODEL_ENV, "").strip()
    return value or DEFAULT_LLM_MODEL


def llm_configured() -> bool:
    return bool(llm_base_url() and llm_api_key())
