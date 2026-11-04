"""Data access used by the API and by later training code.

The read-only instance refuses writers by opening SQLite with ``mode=ro``.
Analytical scans go through DuckDB and do not take a write lock.
"""

from __future__ import annotations

from supere.storage.db import Database
from supere.storage.duck import DuckReader
from supere.timeutil import format_utc8


class DataAccess:
    def __init__(self, db: Database | None, duck: DuckReader | None = None) -> None:
        self.db = db
        self.duck = duck

    @property
    def ready(self) -> bool:
        return self.db is not None

    def klines(
        self,
        market: str,
        symbol: str,
        interval: str,
        *,
        start: int | None = None,
        end: int | None = None,
        limit: int = 500,
    ) -> list[dict]:
        if self.db is None:
            return []
        rows = self.db.fetch_klines(
            market, symbol, interval, start=start, end=end, limit=limit
        )
        return [_kline_dict(row) for row in rows]

    def coverage(self) -> list[dict]:
        if self.db is None:
            return []
        return [dict(row) for row in self.db.list_coverage()]

    def events(self, limit: int = 100) -> list[dict]:
        if self.db is None:
            return []
        out = []
        for row in self.db.list_events(limit=limit):
            item = dict(row)
            if item.get("ts") is not None:
                item["ts_utc8"] = format_utc8(int(item["ts"]))
            out.append(item)
        return out

    def anomalies(self, limit: int = 100) -> list[dict]:
        if self.db is None:
            return []
        return [dict(row) for row in self.db.list_anomalies(limit)]

    def duck_kline_count(self, market: str, symbol: str, interval: str) -> int | None:
        if self.duck is None or not self.ready:
            return None
        return self.duck.kline_count(market, symbol, interval)


def _kline_dict(row) -> dict:
    item = dict(row)
    item["open_time_utc8"] = format_utc8(int(item["open_time"]))
    item["close_time_utc8"] = format_utc8(int(item["close_time"]))
    item["closed"] = True
    return item
