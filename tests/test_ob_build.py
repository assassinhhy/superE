from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from supere.hist.schema import HIST_SCHEMAS, partition_dir
from supere.ob.build import build
from supere.ob.features import FEATURE_COLUMNS

DAY_MS = 86_400_000
DAY0 = 1_700_000_000_000 // DAY_MS * DAY_MS


def _write(raw: Path, kind: str, date: str, table: pa.Table, symbol: str = "ETHUSDT") -> None:
    folder = partition_dir(raw, kind, date, symbol)
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, folder / "part-000.parquet")


def _trades(day: int) -> pa.Table:
    rows = []
    for minute in range(1440):
        for offset, maker in ((5, False), (20, True), (40, False)):
            ts = DAY0 + day * DAY_MS + minute * 60_000 + offset * 1000
            rows.append(
                {
                    "agg_id": len(rows) + 1,
                    "price": "2000.0",
                    "qty": "1.5",
                    "first_id": len(rows) + 1,
                    "last_id": len(rows) + 1,
                    "ts": ts,
                    "is_buyer_maker": maker,
                    "recv_ms": ts,
                    "recv_local_ms": ts,
                    "source": "test",
                }
            )
    return pa.Table.from_pylist(rows, schema=HIST_SCHEMAS["aggTrades_hist"])


def _book(day: int) -> pa.Table:
    rows = []
    for minute in range(1440):
        ts = DAY0 + day * DAY_MS + minute * 60_000 + 30_000
        for pct in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
            rows.append(
                {
                    "ts": ts,
                    "percentage": pct,
                    "depth": "10.0",
                    "notional": str(1000.0 + abs(pct)),
                    "source": "test",
                }
            )
    return pa.Table.from_pylist(rows, schema=HIST_SCHEMAS["bookDepth_hist"])


def _metrics(day: int) -> pa.Table:
    rows = []
    for minute in range(0, 1440, 5):
        rows.append(
            {
                "ts": DAY0 + day * DAY_MS + minute * 60_000,
                "symbol": "ETHUSDT",
                "oi": "1000000.0",
                "oi_value": None,
                "top_pos_lsr": "1.5",
                "top_acc_lsr": "0.9",
                "global_lsr": "1.2",
                "taker_lsr": "0.8",
                "source": "test",
            }
        )
    return pa.Table.from_pylist(rows, schema=HIST_SCHEMAS["metrics_hist"])


def test_build_two_days(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "features"
    _write(raw, "aggTrades", "2023-11-14", _trades(0))
    _write(raw, "aggTrades", "2023-11-15", _trades(1))
    _write(raw, "bookDepth", "2023-11-14", _book(0))
    _write(raw, "bookDepth", "2023-11-15", _book(1))
    _write(raw, "metrics", "2023-11-14", _metrics(0))
    _write(raw, "metrics", "2023-11-15", _metrics(1))

    stats = build(raw, out, start="2023-11-14", end="2023-11-15", warmup_days=0)
    assert stats.days_written == 2
    assert stats.rows == 2880
    assert stats.missing_days == []

    day1 = pq.read_table(out / "um" / "ETHUSDT" / "date=2023-11-14" / "part-000.parquet")
    day2 = pq.read_table(out / "um" / "ETHUSDT" / "date=2023-11-15" / "part-000.parquet")
    assert day1.num_rows == day2.num_rows == 1440
    assert day1.schema.names == ["minute_ms", *FEATURE_COLUMNS]
    row = day1.to_pylist()[0]
    assert row["ob_quote"] == 3 * 1.5 * 2000.0
    assert row["ob_trade_count"] == 3
    assert row["ob_book_stale"] == 0.0
    assert math.isclose(row["ob_bid_notional_1"], 1001.0)
    assert math.isclose(row["ob_ask_notional_1"], 1001.0)
    # metrics bar 00:00 is only known at 00:05.
    assert math.isnan(row["ob_oi"])
    assert day1.to_pylist()[4]["ob_oi"] == 1_000_000.0
    # oi_value is a NULL string and must become NaN, not crash.
    assert math.isnan(day1.to_pylist()[4]["ob_oi_value"])

    meta = json.loads((out / "_meta.json").read_text(encoding="utf-8"))
    assert meta["rows"] == 2880
    assert meta["per_year"]["2023"] == 2880


def test_build_reads_the_requested_symbol(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "features"
    _write(raw, "aggTrades", "2023-11-14", _trades(0), symbol="BTCUSDT")
    _write(raw, "bookDepth", "2023-11-14", _book(0), symbol="BTCUSDT")
    _write(raw, "metrics", "2023-11-14", _metrics(0), symbol="BTCUSDT")
    stats = build(raw, out, symbol="btcusdt", start="2023-11-14", end="2023-11-14", warmup_days=0)
    assert stats.rows == 1440
    table = pq.read_table(out / "um" / "BTCUSDT" / "date=2023-11-14" / "part-000.parquet")
    assert table.num_rows == 1440
    assert table.to_pylist()[0]["ob_trade_count"] == 3.0
