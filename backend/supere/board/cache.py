"""Board cache. This file is the only database the board process writes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS points (
  kind TEXT NOT NULL,
  symbol TEXT NOT NULL,
  ts INTEGER NOT NULL,
  payload TEXT NOT NULL,
  fetched_at INTEGER NOT NULL,
  source TEXT NOT NULL,
  PRIMARY KEY (kind, symbol, ts)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL
);
"""


class BoardCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert_point(self, kind: str, symbol: str, ts: int, payload: dict, fetched_at: int, source: str) -> None:
        self.conn.execute(
            """
            INSERT INTO points (kind, symbol, ts, payload, fetched_at, source)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(kind, symbol, ts) DO UPDATE SET
              payload=excluded.payload,
              fetched_at=excluded.fetched_at,
              source=excluded.source
            """,
            (kind, symbol, int(ts), json.dumps(payload, ensure_ascii=False), int(fetched_at), source),
        )
        self.conn.commit()

    def latest(self, kind: str, symbol: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT ts, payload, fetched_at, source FROM points
            WHERE kind=? AND symbol=? ORDER BY ts DESC LIMIT 1
            """,
            (kind, symbol),
        ).fetchone()
        if row is None:
            return None
        return {
            "ts": int(row["ts"]),
            "payload": json.loads(row["payload"]),
            "fetched_at": int(row["fetched_at"]),
            "source": row["source"],
        }

    def points(self, kind: str, symbol: str, *, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT ts, payload, fetched_at, source FROM (
              SELECT ts, payload, fetched_at, source FROM points
              WHERE kind=? AND symbol=? ORDER BY ts DESC LIMIT ?
            ) ORDER BY ts ASC
            """,
            (kind, symbol, int(limit)),
        ).fetchall()
        out = []
        seen: set[int] = set()
        for row in rows:
            ts = int(row["ts"])
            if ts in seen:
                continue
            seen.add(ts)
            out.append(
                {
                    "ts": ts,
                    "payload": json.loads(row["payload"]),
                    "fetched_at": int(row["fetched_at"]),
                    "source": row["source"],
                }
            )
        return out

    def klines_between(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT payload FROM points
            WHERE kind=? AND symbol=? AND ts>=? AND ts<=?
            ORDER BY ts ASC
            """,
            (f"kline:{interval}", symbol, int(start_ms), int(end_ms)),
        ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["payload"])
            if isinstance(payload, dict) and payload.get("open_time") is not None:
                out.append(payload)
        return out

    def add_event(self, ts: int, kind: str, detail: dict) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
            (int(ts), kind, json.dumps(detail, ensure_ascii=False)),
        )
        self.conn.commit()
