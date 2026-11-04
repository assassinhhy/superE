"""Timestamps are epoch milliseconds. Display and log dates use UTC+8.

Kline buckets are not redefined in UTC+8. Use :mod:`supere.intervals` for
Binance's UTC open times, and this module only to render them.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone

UTC8 = timezone(timedelta(hours=8))


def local_now_ms() -> int:
    return int(time.time() * 1000)


def to_millis(ts: int) -> int:
    """Normalize seconds, milliseconds, or microseconds to milliseconds.

    Spot archive files switched to microseconds on 2025-01-01. UM archives
    are still milliseconds. The unit is decided from the digit count.
    """
    value = int(ts)
    digits = len(str(abs(value)))
    if digits >= 16:
        return value // 1000
    if digits <= 10:
        return value * 1000
    return value


def format_utc8(ms: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC8).strftime(fmt)


def format_utc8_ms(ms: int) -> str:
    base = format_utc8(ms)
    return f"{base}.{ms % 1000:03d}"


def utc8_date(ms: int) -> str:
    """Partition and log date in UTC+8 (``YYYY-MM-DD``)."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC8).strftime("%Y-%m-%d")


def parse_utc_datetime(text: str) -> int:
    """Parse ``YYYY-MM-DD HH:MM:SS`` as UTC and return epoch milliseconds.

    Metrics archive ``create_time`` uses this form and is the bar start.
    """
    dt = datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def utc_date_from_ms(ms: int):
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return dt.date()
