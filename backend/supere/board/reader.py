"""Read-only view of the collector database. No writes, no schema changes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from supere.intervals import INTERVAL_MS
from supere.storage.db import Database

BOARD_INTERVALS = ("5m", "15m", "1h", "4h", "1d")
METRIC_MS = 300_000


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bar_from_row(row) -> dict:
    open_time = int(row["open_time"])
    interval = row["interval"]
    close_time = int(row["close_time"]) if row["close_time"] is not None else open_time + INTERVAL_MS[interval] - 1
    return {
        "open_time": open_time,
        "close_time": close_time,
        "open": float(row["o"]),
        "high": float(row["h"]),
        "low": float(row["l"]),
        "close": float(row["c"]),
        "volume": float(row["volume"]),
        "taker_buy": float(row["taker_buy_base"] or 0),
    }


def metric_from_row(row) -> dict:
    ts = int(row["ts"])
    return {
        "ts": ts,
        "known_at": ts + METRIC_MS,
        "oi": _num(row["oi"]),
        "oi_value": _num(row["oi_value"]),
        "top_pos_lsr": _num(row["top_pos_lsr"]),
        "top_acc_lsr": _num(row["top_acc_lsr"]),
        "global_lsr": _num(row["global_lsr"]),
        "taker_lsr": _num(row["taker_lsr"]),
    }


class MainReader:
    """Opens ``superE.sqlite`` with ``mode=ro`` and ``query_only``. Missing file yields empty reads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.db: Database | None = None
        if self.path.is_file():
            self.db = Database(self.path, read_only=True)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    @property
    def conn(self) -> sqlite3.Connection | None:
        return None if self.db is None else self.db.conn

    def klines(self, symbol: str, interval: str, *, limit: int, as_of_ms: int | None = None) -> list[dict]:
        if self.db is None:
            return []
        end = None if as_of_ms is None else int(as_of_ms)
        rows = self.db.fetch_klines("um", symbol, interval, end=end, limit=limit)
        bars = [bar_from_row(row) for row in rows]
        if as_of_ms is not None:
            bars = [bar for bar in bars if bar["close_time"] <= int(as_of_ms)]
        return bars

    def klines_between(self, symbol: str, interval: str, start_ms: int, end_ms: int, *, limit: int = 5000) -> list[dict]:
        """Closed bars with ``open_time`` in ``[start_ms, end_ms]``. Read only."""
        if self.db is None:
            return []
        try:
            rows = self.db.conn.execute(
                """
                SELECT * FROM klines
                WHERE market='um' AND symbol=? AND interval=?
                  AND open_time>=? AND open_time<=?
                ORDER BY open_time ASC
                LIMIT ?
                """,
                (symbol, interval, int(start_ms), int(end_ms), int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [bar_from_row(row) for row in rows]

    def metrics(self, symbol: str, *, limit: int, as_of_ms: int | None = None) -> list[dict]:
        if self.db is None:
            return []
        end = None if as_of_ms is None else int(as_of_ms) - METRIC_MS
        rows = self.db.metrics_between(symbol, None, end, limit=limit)
        items = [metric_from_row(row) for row in rows]
        if as_of_ms is not None:
            items = [item for item in items if item["known_at"] <= int(as_of_ms)]
        return items

    def funding_live(self, symbol: str) -> dict | None:
        if self.db is None:
            return None
        try:
            row = self.db.conn.execute(
                "SELECT rate, next_funding_time, mark_price, index_price, updated_at FROM funding_live WHERE symbol=?",
                (symbol,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return {
            "rate": _num(row["rate"]),
            "next_funding_time": None if row["next_funding_time"] is None else int(row["next_funding_time"]),
            "mark": _num(row["mark_price"]),
            "index": _num(row["index_price"]),
            "updated_at": int(row["updated_at"]),
        }

    def events(self, limit: int = 50) -> list[dict]:
        if self.db is None:
            return []
        try:
            rows = self.db.conn.execute(
                """
                SELECT ts_event, kind, title, importance, pause_before_min, pause_after_min
                FROM events ORDER BY ts_event DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "ts_event": int(row["ts_event"]),
                "kind": row["kind"],
                "title": row["title"],
                "importance": int(row["importance"]),
                "pause_before_min": int(row["pause_before_min"]),
                "pause_after_min": int(row["pause_after_min"]),
            }
            for row in rows
        ]
