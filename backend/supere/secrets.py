"""Secrets come from the environment, never from the repository.

Binance market data does not use a key. DeepSeek, used in a later phase, is
read here so callers have one place that cannot be logged by accident.
"""

from __future__ import annotations

import os

DEEPSEEK_ENV = "DEEPSEEK_API_KEY"


def deepseek_api_key() -> str | None:
    value = os.environ.get(DEEPSEEK_ENV, "").strip()
    return value or None


def deepseek_configured() -> bool:
    return deepseek_api_key() is not None
