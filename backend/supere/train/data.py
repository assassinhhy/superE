"""Read-only access to ``superE.sqlite`` for training.

The database is opened with ``mode=ro`` so training can never write to it, even
while the collector holds the write lock. Values stay as Binance decimal
strings in the table; they are converted to float64 here. The float copies are
for modelling only. Exact price comparisons happen in SQL or in the collector,
never here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS

# Extra history loaded before the first decision bar so rolling features and
# EMAs are warm. 300 bars covers the 26-period EMA and the 20-bar windows.
WARMUP_BARS = 300


@dataclass
class Bars:
    interval: str
    open_time: np.ndarray  # int64 ms, UTC boundaries (plan section 5.5)
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741 - matches the OHLC naming used everywhere
    c: np.ndarray
    volume: np.ndarray
    quote_volume: np.ndarray
    trades: np.ndarray
    taker_buy_base: np.ndarray

    @property
    def n(self) -> int:
        return int(self.open_time.shape[0])

    @property
    def interval_ms(self) -> int:
        return INTERVAL_MS[self.interval]

    @property
    def close_time(self) -> np.ndarray:
        """Decision moment of each bar: the close, equal to the next open."""
        return self.open_time + self.interval_ms

    def slice_to(self, end: int) -> Bars:
        """Bars ``[0, end)``. Used by the no-lookahead tests."""
        return Bars(
            interval=self.interval,
            open_time=self.open_time[:end],
            o=self.o[:end],
            h=self.h[:end],
            l=self.l[:end],
            c=self.c[:end],
            volume=self.volume[:end],
            quote_volume=self.quote_volume[:end],
            trades=self.trades[:end],
            taker_buy_base=self.taker_buy_base[:end],
        )


@dataclass
class Funding:
    time: np.ndarray  # settlement ms, sorted
    rate: np.ndarray

    @classmethod
    def empty(cls) -> Funding:
        return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))


@dataclass
class Metrics:
    ts: np.ndarray  # 5-minute bar start, ms, sorted
    oi: np.ndarray
    top_pos_lsr: np.ndarray
    top_acc_lsr: np.ndarray
    global_lsr: np.ndarray
    taker_lsr: np.ndarray

    @classmethod
    def empty(cls) -> Metrics:
        z = np.zeros(0, dtype=np.float64)
        return cls(np.zeros(0, dtype=np.int64), z, z, z, z, z)


def open_readonly(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到数据库 {path}；先运行采集器或 supere backfill")
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _to_float(value) -> float:
    """NULL, empty and textual NaN all become NaN. Malformed text also does."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("nan", "null", "n/a"):
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_bars(
    conn: sqlite3.Connection,
    market: str,
    symbol: str,
    interval: str,
    limit_bars: int | None = None,
) -> Bars:
    sql = """
        SELECT open_time, o, h, l, c, volume, quote_volume, trades, taker_buy_base
        FROM klines
        WHERE market = ? AND symbol = ? AND interval = ?
        ORDER BY open_time
    """
    rows = conn.execute(sql, (market, symbol, interval)).fetchall()
    if limit_bars is not None:
        rows = rows[-(limit_bars + WARMUP_BARS) :]
    if not rows:
        raise ValueError(f"库里没有 {market} {symbol} {interval} 的 K 线")

    open_time = np.array([r[0] for r in rows], dtype=np.int64)
    cols = list(zip(*rows, strict=True))

    def f64(i: int) -> np.ndarray:
        return np.array([_to_float(v) for v in cols[i]], dtype=np.float64)

    bars = Bars(
        interval=interval,
        open_time=open_time,
        o=f64(1),
        h=f64(2),
        l=f64(3),
        c=f64(4),
        volume=f64(5),
        quote_volume=f64(6),
        trades=f64(7),
        taker_buy_base=f64(8),
    )
    for name in ("o", "h", "l", "c", "volume"):
        bad = int(np.sum(~np.isfinite(getattr(bars, name))))
        if bad:
            raise ValueError(f"{interval} 的 {name} 有 {bad} 个非有限值（空或 inf），库可能损坏")
    if bars.n >= 2 and np.any(np.diff(bars.open_time) <= 0):
        raise ValueError("open_time 没有严格递增，库可能损坏")
    return bars


def gap_after(bars: Bars) -> np.ndarray:
    """``gap[i]`` is True when bar ``i+1`` does not directly follow bar ``i``.

    A missing bar makes ``n bars back`` mean a different span of time, so the
    runner drops any sample whose window crosses one of these.
    """
    if bars.n < 2:
        return np.zeros(bars.n, dtype=bool)
    gaps = np.diff(bars.open_time) != bars.interval_ms
    return np.concatenate([gaps, [False]])


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def load_funding(conn: sqlite3.Connection, symbol: str) -> Funding:
    if not _has_table(conn, "funding"):
        return Funding.empty()
    rows = conn.execute(
        "SELECT funding_time, rate FROM funding WHERE symbol = ? ORDER BY funding_time",
        (symbol,),
    ).fetchall()
    if not rows:
        return Funding.empty()
    return Funding(
        time=np.array([r[0] for r in rows], dtype=np.int64),
        rate=np.array([_to_float(r[1]) for r in rows], dtype=np.float64),
    )


def load_metrics(conn: sqlite3.Connection, symbol: str) -> Metrics:
    if not _has_table(conn, "um_metrics"):
        return Metrics.empty()
    rows = conn.execute(
        """
        SELECT ts, oi, top_pos_lsr, top_acc_lsr, global_lsr, taker_lsr
        FROM um_metrics WHERE symbol = ? ORDER BY ts
        """,
        (symbol,),
    ).fetchall()
    if not rows:
        return Metrics.empty()
    cols = list(zip(*rows, strict=True))
    return Metrics(
        ts=np.array(cols[0], dtype=np.int64),
        oi=np.array([_to_float(v) for v in cols[1]], dtype=np.float64),
        top_pos_lsr=np.array([_to_float(v) for v in cols[2]], dtype=np.float64),
        top_acc_lsr=np.array([_to_float(v) for v in cols[3]], dtype=np.float64),
        global_lsr=np.array([_to_float(v) for v in cols[4]], dtype=np.float64),
        taker_lsr=np.array([_to_float(v) for v in cols[5]], dtype=np.float64),
    )
