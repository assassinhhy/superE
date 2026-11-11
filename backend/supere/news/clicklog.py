"""Append-only click log for later calibration of the news overlay.

One JSON object per user click.  Never contains an API key, request headers or
credentials; only news hashes, scores, source outcomes and quantiles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def log_path(data_root: Path) -> Path:
    return Path(data_root) / "logs" / "predict_clicks.jsonl"


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def append(data_root: Path, record: dict) -> Path | None:
    path = log_path(data_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = _finite(record)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
        return path
    except (OSError, TypeError, ValueError):
        # Logging must not fail a click.
        return None
