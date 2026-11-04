"""Binance interval definitions.

Kline boundaries stay on Binance's UTC grid. 3-day candles are not a plain
modulo of the Unix epoch: the 2020-01 archive opens on 2020-01-02 and then
every 3 days, which is UTC day-index % 3 == 2.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

UM_INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
)

# Weekly and monthly histories are small. The monthly archive folder stops at
# 2023-06, so both are fetched over REST (see the plan, section 5.1).
REST_ONLY_INTERVALS: frozenset[str] = frozenset({"1w", "1M"})

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "6h": 6 * 3_600_000,
    "8h": 8 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": 86_400_000,
    "3d": 3 * 86_400_000,
    "1w": 7 * 86_400_000,
}

DAY_MS = 86_400_000
MINUTE_MS = 60_000

# ETHUSDT USDT-M perpetual onboardDate from exchangeInfo (CC review T2).
UM_ONBOARD_MS = 1_574_840_700_000


def archive_interval(interval: str) -> str:
    """Monthly archive folder uses ``1mo`` so case-insensitive disks stay unique."""
    if interval == "1M":
        return "1mo"
    return interval


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _stamp(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def align_open(interval: str, ts_ms: int) -> int:
    """Open time of the candle that contains ``ts_ms`` (UTC boundaries)."""
    if interval == "1M":
        dt = _utc(ts_ms)
        return _stamp(datetime(dt.year, dt.month, 1, tzinfo=UTC))
    if interval == "1w":
        dt = _utc(ts_ms)
        monday = datetime(dt.year, dt.month, dt.day, tzinfo=UTC) - timedelta(days=dt.weekday())
        return _stamp(monday)
    if interval == "3d":
        day_index = ts_ms // DAY_MS
        delta = (day_index - 2) % 3
        return (day_index - delta) * DAY_MS
    step = INTERVAL_MS[interval]
    return ts_ms - (ts_ms % step)


def next_open(interval: str, open_ms: int) -> int:
    if interval == "1M":
        dt = _utc(open_ms)
        if dt.month == 12:
            nxt = datetime(dt.year + 1, 1, 1, tzinfo=UTC)
        else:
            nxt = datetime(dt.year, dt.month + 1, 1, tzinfo=UTC)
        return _stamp(nxt)
    if interval == "1w":
        return open_ms + INTERVAL_MS["1w"]
    if interval == "3d":
        return open_ms + INTERVAL_MS["3d"]
    return open_ms + INTERVAL_MS[interval]


def previous_open(interval: str, open_ms: int) -> int:
    if interval == "1M":
        dt = _utc(open_ms)
        if dt.month == 1:
            prev = datetime(dt.year - 1, 12, 1, tzinfo=UTC)
        else:
            prev = datetime(dt.year, dt.month - 1, 1, tzinfo=UTC)
        return _stamp(prev)
    if interval == "1w":
        return open_ms - INTERVAL_MS["1w"]
    if interval == "3d":
        return open_ms - INTERVAL_MS["3d"]
    return open_ms - INTERVAL_MS[interval]


def close_time_of(interval: str, open_ms: int) -> int:
    return next_open(interval, open_ms) - 1


def is_closed(interval: str, open_ms: int, now_ms: int) -> bool:
    """A bar is closed when exchange time has reached the next open."""
    return next_open(interval, open_ms) <= now_ms


def last_closed_open(interval: str, now_ms: int) -> int | None:
    """Open time of the most recent fully closed candle, if any."""
    current = align_open(interval, now_ms)
    if is_closed(interval, current, now_ms):
        return current
    if current <= 0:
        return None
    return previous_open(interval, current)


def iter_opens(interval: str, start_open: int, end_open: int):
    cur = align_open(interval, start_open)
    if cur < start_open:
        cur = next_open(interval, cur)
    while cur <= end_open:
        yield cur
        nxt = next_open(interval, cur)
        if nxt <= cur:
            break
        cur = nxt
