"""Build the ``data/features/ob_1m`` table from raw order-book archives.

One row per UTC minute for every day that has aggTrades data.  The builder is a
single process and reuses the exact same ``ObFeatureState`` as the live
recorder.  Book and metrics days are fed before their minutes are emitted;
state carries across day boundaries so a 23:59 snapshot is still visible at
00:00 and the rolling 60-minute windows do not reset at midnight.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from supere.hist.schema import list_partition_dates, partition_dir
from supere.ob.features import (
    FEATURE_COLUMNS,
    MINUTE_MS,
    group_book_snapshots,
)
from supere.ob.stream import StreamingObEngine

log = logging.getLogger("supere.ob.build")

DAY_MS = 86_400_000
OUTPUT_SCHEMA = pa.schema(
    [("minute_ms", pa.int64())] + [(name, pa.float64()) for name in FEATURE_COLUMNS]
)


@dataclass
class BuildStats:
    days_written: int = 0
    rows: int = 0
    days_skipped: int = 0
    days_skipped_rows: int = 0
    missing_days: list[str] = field(default_factory=list)
    first_minute: int | None = None
    last_minute: int | None = None
    per_year: dict[str, int] = field(default_factory=dict)

    def add(self, date: str, rows: int, first: int, last: int) -> None:
        self.days_written += 1
        self.rows += rows
        self.first_minute = first if self.first_minute is None else min(self.first_minute, first)
        self.last_minute = last if self.last_minute is None else max(self.last_minute, last)
        year = date[:4]
        self.per_year[year] = self.per_year.get(year, 0) + rows


def _read_columns(
    raw_root: Path, kind: str, date: str, columns: list[str], symbol: str = "ETHUSDT"
) -> pa.Table | None:
    folder = partition_dir(raw_root, kind, date, symbol)
    files = sorted(folder.glob("part-*.parquet"))
    if not files:
        return None
    tables = [pq.read_table(path, columns=columns) for path in files]
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


def _to_float(table: pa.Table, name: str) -> np.ndarray:
    column = table[name]
    if pa.types.is_string(column.type):
        # Metrics archives contain empty quoted fields on some days.
        column = pc.if_else(pc.equal(column, ""), pa.scalar(None, pa.string()), column)
    try:
        return np.asarray(pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        values = column.to_pylist()
        out = np.full(len(values), np.nan, dtype=np.float64)
        for i, value in enumerate(values):
            if value is None:
                continue
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                pass
        return out


def _to_int(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(pc.cast(table[name], pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)


def _day_start_ms(date: str) -> int:
    return int(np.datetime64(date, "D").astype("datetime64[ms]").astype(np.int64))


def _feed_book(engine: StreamingObEngine, raw_root: Path, date: str, symbol: str) -> bool:
    table = _read_columns(
        raw_root, "bookDepth", date, ["ts", "percentage", "notional"], symbol
    )
    if table is None or table.num_rows == 0:
        return False
    ts = _to_int(table, "ts")
    pct = np.asarray(pc.cast(table["percentage"], pa.int64()).to_numpy(zero_copy_only=False))
    notional = _to_float(table, "notional")
    snap_ts, bid, ask, _bid_depth, _ask_depth = group_book_snapshots(ts, pct, notional)
    engine.state.book.add_day(snap_ts, bid, ask)
    return True


def _feed_metrics(engine: StreamingObEngine, raw_root: Path, date: str, symbol: str) -> bool:
    table = _read_columns(
        raw_root,
        "metrics",
        date,
        ["ts", "oi", "oi_value", "top_pos_lsr", "top_acc_lsr", "global_lsr", "taker_lsr"],
        symbol,
    )
    if table is None or table.num_rows == 0:
        return False
    ts = _to_int(table, "ts")
    columns = {name: _to_float(table, name) for name in (
        "oi", "oi_value", "top_pos_lsr", "top_acc_lsr", "global_lsr", "taker_lsr"
    )}
    order = np.argsort(ts, kind="stable")
    for idx in order:
        engine.state.metrics.push(
            int(ts[idx]),
            float(columns["oi"][idx]),
            float(columns["oi_value"][idx]),
            float(columns["top_pos_lsr"][idx]),
            float(columns["top_acc_lsr"][idx]),
            float(columns["global_lsr"][idx]),
            float(columns["taker_lsr"][idx]),
        )
    return True


def _row_table(rows: list[dict[str, float]], minutes: list[int]) -> pa.Table:
    arrays: list[pa.Array] = [pa.array(minutes, type=pa.int64())]
    for name in FEATURE_COLUMNS:
        arrays.append(pa.array([row.get(name, float("nan")) for row in rows], type=pa.float64()))
    return pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA)


def write_day(out_root: Path, date: str, table: pa.Table, symbol: str = "ETHUSDT", market: str = "um") -> Path:
    folder = Path(out_root) / "um" / symbol.upper() / f"date={date}"
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "part-000.parquet"
    tmp = folder / "part-000.parquet.tmp"
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, final)
    return final


def _raw_newer(raw_root: Path, date: str, symbol: str, existing: Path) -> bool:
    """True when a raw partition was rewritten after the feature partition."""
    try:
        built_at = existing.stat().st_mtime
    except OSError:
        return True
    for kind in ("aggTrades", "bookDepth", "metrics"):
        folder = partition_dir(raw_root, kind, date, symbol)
        for path in folder.glob("part-*.parquet"):
            try:
                if path.stat().st_mtime > built_at:
                    return True
            except OSError:
                continue
    return False


def build(
    raw_root: Path,
    out_root: Path,
    *,
    symbol: str = "ETHUSDT",
    start: str | None = None,
    end: str | None = None,
    warmup_days: int = 2,
    overwrite: bool = False,
    on_progress=None,
) -> BuildStats:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    symbol = symbol.upper()
    all_dates = list_partition_dates(raw_root, "aggTrades", symbol)
    if not all_dates:
        raise FileNotFoundError(f"{raw_root} 下没有 aggTrades_hist 分区")
    if start is None:
        start = all_dates[0]
    if end is None:
        end = all_dates[-1]
    dates = [d for d in all_dates if start <= d <= end]
    if not dates:
        return BuildStats()

    warmup = [d for d in all_dates if d < start][-warmup_days:] if warmup_days else []

    engine = StreamingObEngine()
    stats = BuildStats()

    for date in warmup:
        _build_day(engine, raw_root, None, date, stats, symbol=symbol, write=False, count=False)
        log.info("预热 %s", date)

    for index, date in enumerate(dates, 1):
        existing = out_root / "um" / symbol.upper() / f"date={date}" / "part-000.parquet"
        skip = False
        if not overwrite and existing.exists():
            if _raw_newer(raw_root, date, symbol, existing):
                log.warning("%s 的 raw 比特征分区新，重建该日", date)
            else:
                skip = True
        if skip:
            stats.days_skipped += 1
            # State still needs the data for later days.
            _build_day(engine, raw_root, None, date, stats, symbol=symbol, write=False, count=False)
            meta_row = pq.read_metadata(existing)
            if meta_row.num_rows:
                stats.rows += meta_row.num_rows
                stats.days_skipped_rows += meta_row.num_rows
                stats.per_year[date[:4]] = stats.per_year.get(date[:4], 0) + meta_row.num_rows
                day_start = _day_start_ms(date)
                stats.first_minute = day_start if stats.first_minute is None else min(stats.first_minute, day_start)
                stats.last_minute = day_start + DAY_MS - MINUTE_MS if stats.last_minute is None else max(
                    stats.last_minute, day_start + DAY_MS - MINUTE_MS
                )
        else:
            _build_day(engine, raw_root, out_root, date, stats, symbol=symbol, write=True, count=True)
        if on_progress and (index % 10 == 0 or index == len(dates)):
            on_progress(
                f"进度：特征 {index}/{len(dates)} 日，累计 {stats.rows:,} 行，"
                f"最新 {date}，跳过 {stats.days_skipped}，缺日 {len(stats.missing_days)}"
            )
    meta = {
        "symbol": symbol,
        "rows": stats.rows,
        "days": stats.days_written,
        "skipped_days": stats.days_skipped,
        "missing_days": stats.missing_days,
        "first_minute": stats.first_minute,
        "last_minute": stats.last_minute,
        "per_year": dict(sorted(stats.per_year.items())),
        "columns": ["minute_ms", *FEATURE_COLUMNS],
        "decision_lag_ms": MINUTE_MS,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _build_day(
    engine: StreamingObEngine,
    raw_root: Path,
    out_root: Path | None,
    date: str,
    stats: BuildStats,
    *,
    symbol: str,
    write: bool,
    count: bool,
) -> None:
    # Book and metrics are independent of aggTrades; feed them even when the
    # trade partition is missing so later days inherit the right state.
    _feed_book(engine, raw_root, date, symbol)
    _feed_metrics(engine, raw_root, date, symbol)

    table = _read_columns(
        raw_root, "aggTrades", date, ["ts", "price", "qty", "is_buyer_maker"], symbol
    )
    if table is None or table.num_rows == 0:
        if count:
            stats.missing_days.append(date)
        return
    ts = _to_int(table, "ts")
    price = _to_float(table, "price")
    qty = _to_float(table, "qty")
    maker = np.asarray(table["is_buyer_maker"].to_numpy(zero_copy_only=False), dtype=bool)
    del table

    if ts.size > 1 and not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts, kind="stable")
        ts, price, qty, maker = ts[order], price[order], qty[order], maker[order]

    day_start = _day_start_ms(date)
    day_end = day_start + DAY_MS
    if ts.size == 0 or ts[-1] < day_start or ts[0] >= day_end:
        if count:
            stats.missing_days.append(date)
        return

    pointer = int(np.searchsorted(ts, day_start, side="left"))
    rows: list[dict[str, float]] = []
    minutes: list[int] = []
    minute = day_start
    while minute < day_end:
        end = int(np.searchsorted(ts, minute + MINUTE_MS, side="left"))
        row = engine.push_minute_arrays(
            minute,
            ts[pointer:end],
            price[pointer:end],
            qty[pointer:end],
            maker[pointer:end],
        )
        rows.append(row)
        minutes.append(minute)
        pointer = end
        minute += MINUTE_MS

    if write and out_root is not None:
        table_out = _row_table(rows, minutes)
        write_day(out_root, date, table_out, symbol=symbol)
    if count:
        stats.add(date, len(minutes), minutes[0], minutes[-1])
