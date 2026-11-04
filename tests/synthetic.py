"""Deterministic synthetic market database for training tests.

Builds a small ``superE.sqlite`` with the real schema (``init_schema``), so the
training code is exercised against the same tables the collector writes. No
network, no full library needed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.storage.schema import init_schema

# 2020-01-01 00:00 UTC. It is a multiple of 12h, 1h and 15m, so every interval
# used in these tests starts on a Binance boundary.
START_MS = 1_577_836_800_000


def make_bar_arrays(n: int, seed: int = 7, interval: str = "12h") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    step = INTERVAL_MS[interval]
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    prev = np.concatenate([[2000.0], close[:-1]])
    opn = prev * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    volume = rng.uniform(1e3, 5e3, n)
    return {
        "open_time": START_MS + np.arange(n, dtype=np.int64) * step,
        "o": opn,
        "h": high,
        "l": low,
        "c": close,
        "volume": volume,
        "quote_volume": volume * close,
        "trades": rng.integers(500, 5000, n).astype(np.float64),
        "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
    }


def build_synthetic_db(
    path: Path,
    n: int = 600,
    seed: int = 7,
    interval: str = "12h",
    with_metrics: bool = True,
    with_funding: bool = True,
    symbol: str = "ETHUSDT",
) -> Path:
    arr = make_bar_arrays(n, seed=seed, interval=interval)
    conn = sqlite3.connect(path)
    try:
        init_schema(conn)
        rows = []
        for i in range(n):
            rows.append(
                (
                    "um",
                    symbol,
                    interval,
                    int(arr["open_time"][i]),
                    int(arr["open_time"][i]) + INTERVAL_MS[interval] - 1,
                    f"{arr['o'][i]:.4f}",
                    f"{arr['h'][i]:.4f}",
                    f"{arr['l'][i]:.4f}",
                    f"{arr['c'][i]:.4f}",
                    f"{arr['volume'][i]:.4f}",
                    f"{arr['quote_volume'][i]:.4f}",
                    int(arr["trades"][i]),
                    f"{arr['taker_buy_base'][i]:.4f}",
                    f"{arr['taker_buy_base'][i] * arr['c'][i]:.4f}",
                    "test",
                    0,
                    0,
                )
            )
        conn.executemany("INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

        if with_funding:
            rng = np.random.default_rng(seed + 1)
            span = n * INTERVAL_MS[interval]
            fr = [
                (symbol, START_MS + k * 8 * 3_600_000, f"{rng.normal(0.0001, 0.0002):.8f}", None, "rest")
                for k in range(span // (8 * 3_600_000))
            ]
            conn.executemany("INSERT INTO funding VALUES (?,?,?,?,?)", fr)

        if with_metrics:
            rng = np.random.default_rng(seed + 2)
            span = n * INTERVAL_MS[interval]
            count = span // 300_000
            oi = 1e5 * np.exp(np.cumsum(rng.normal(0.0, 0.002, count)))
            mrows = []
            for k in range(count):
                def maybe(v: float) -> str | None:
                    return None if rng.random() < 0.02 else f"{v:.6f}"

                mrows.append(
                    (
                        symbol,
                        START_MS + k * 300_000,
                        maybe(oi[k]),
                        None,
                        maybe(1.0 + rng.normal(0, 0.1)),
                        maybe(1.0 + rng.normal(0, 0.1)),
                        maybe(1.0 + rng.normal(0, 0.1)),
                        maybe(1.0 + rng.normal(0, 0.1)),
                        "test",
                    )
                )
            conn.executemany("INSERT INTO um_metrics VALUES (?,?,?,?,?,?,?,?,?)", mrows)
        conn.commit()
    finally:
        conn.close()
    return path
