"""Convert data.binance.vision zip archives to daily Parquet partitions.

The CSV is extracted to a temporary file first.  pyarrow's CSV reader needs a
seekable input, and an extracted month (a few GB) is much cheaper than a
per-row Python parser.  Parsing runs single-threaded to stay out of the
training process's way.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from supere.hist.schema import HIST_SCHEMAS, partition_dir

AGG_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]
BOOK_COLUMNS = ["timestamp", "percentage", "depth", "notional"]
METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

_AGG_TYPES = {
    "agg_trade_id": pa.int64(),
    "price": pa.string(),
    "quantity": pa.string(),
    "first_trade_id": pa.int64(),
    "last_trade_id": pa.int64(),
    "transact_time": pa.int64(),
    "is_buyer_maker": pa.bool_(),
}
_BOOK_TYPES = {
    "timestamp": pa.string(),
    # Some 2026 archives write "-5.00" instead of "-5".
    "percentage": pa.float64(),
    "depth": pa.string(),
    "notional": pa.string(),
}
_METRICS_TYPES = {
    "create_time": pa.string(),
    "symbol": pa.string(),
    "sum_open_interest": pa.string(),
    "sum_open_interest_value": pa.string(),
    "count_toptrader_long_short_ratio": pa.string(),
    "sum_toptrader_long_short_ratio": pa.string(),
    "count_long_short_ratio": pa.string(),
    "sum_taker_long_short_vol_ratio": pa.string(),
}

_BLOCK = 64 << 20
_DAY_MS = 86_400_000


@dataclass
class ConvertResult:
    kind: str
    rows: int
    date_counts: dict[str, int]
    parquet: list[str]
    min_ts: int | None
    max_ts: int | None


class _PartitionWriters:
    def __init__(self, raw_root: Path, kind: str, symbol: str, market: str = "um") -> None:
        self.raw_root = Path(raw_root)
        self.kind = kind
        self.symbol = symbol.upper()
        self.market = market
        self.schema = HIST_SCHEMAS[_dataset(kind)]
        self._writers: dict[str, tuple[Path, Path, pq.ParquetWriter]] = {}
        self._written: list[str] = []
        self.counts: dict[str, int] = {}
        self._current_date: str | None = None

    def write(self, table: pa.Table, days: np.ndarray) -> None:
        if table.num_rows == 0:
            return
        start = 0
        n = table.num_rows
        for i in range(1, n + 1):
            if i == n or days[i] != days[start]:
                date = str(days[start])
                if self._current_date is not None and date != self._current_date:
                    # Rows are time-sorted, so an earlier date is finished.
                    self.close_before(date)
                self._write_group(table.slice(start, i - start), date)
                self._current_date = date
                start = i

    def close_before(self, date: str) -> None:
        for old_date in sorted(self._writers):
            if old_date >= date:
                break
            tmp, final, writer = self._writers.pop(old_date)
            writer.close()
            os.replace(tmp, final)
            self._written.append(str(final))

    def _write_group(self, table: pa.Table, date: str) -> None:
        tmp, final, writer = self._writer(date)
        writer.write_table(table, row_group_size=262_144)
        self.counts[date] = self.counts.get(date, 0) + table.num_rows

    def _writer(self, date: str) -> tuple[Path, Path, pq.ParquetWriter]:
        got = self._writers.get(date)
        if got is not None:
            return got
        folder = partition_dir(self.raw_root, self.kind, date, self.symbol, self.market)
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / "part-000.parquet"
        tmp = folder / "part-000.parquet.tmp"
        writer = pq.ParquetWriter(
            tmp,
            self.schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        )
        entry = (tmp, final, writer)
        self._writers[date] = entry
        return entry

    def close(self) -> list[str]:
        self.close_before("9999-12-31")
        written = list(self._written)
        self._written.clear()
        return written

    def abort(self) -> None:
        for tmp, _final, writer in self._writers.values():
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
            tmp.unlink(missing_ok=True)
        self._writers.clear()


def _dataset(kind: str) -> str:
    return {
        "aggTrades": "aggTrades_hist",
        "bookDepth": "bookDepth_hist",
        "metrics": "metrics_hist",
    }[kind]


def _first_line(path: Path) -> str:
    with Path(path).open("rb") as handle:
        chunk = handle.read(4096)
    return chunk.split(b"\n", 1)[0].decode("utf-8", "replace").strip()


def has_header(kind: str, path: Path) -> bool:
    line = _first_line(path)
    if kind == "aggTrades":
        return line.startswith("agg_trade_id")
    if kind == "bookDepth":
        return line.startswith("timestamp")
    if kind == "metrics":
        return line.startswith("create_time")
    raise ValueError(f"未知存档类型 {kind!r}")


def _day_strings(ts: pa.Array) -> np.ndarray:
    days = pc.cast(pc.cast(ts, pa.timestamp("ms")), pa.date32())
    return np.asarray(pc.cast(days, pa.string()).to_pylist(), dtype=object)


def _sorted(table: pa.Table) -> pa.Table:
    ts = table.column("ts").combine_chunks()
    if len(ts) == 0:
        return table
    values = np.asarray(ts.to_numpy(zero_copy_only=False))
    if values.size > 1 and not np.all(np.diff(values) >= 0):
        order = pc.sort_indices(table, sort_keys=[("ts", "ascending")])
        table = pc.take(table, order)
    return table


def _agg_table(batch: pa.RecordBatch, source: str) -> pa.Table:
    n = batch.num_rows
    ts = batch.column("transact_time")
    source_arr = pa.array([source] * n, type=pa.string())
    arrays = [
        batch.column("agg_trade_id"),
        batch.column("price"),
        batch.column("quantity"),
        batch.column("first_trade_id"),
        batch.column("last_trade_id"),
        ts,
        batch.column("is_buyer_maker"),
        ts,
        ts,
        source_arr,
    ]
    return pa.Table.from_arrays(arrays, schema=HIST_SCHEMAS["aggTrades_hist"])


def _book_table(batch: pa.RecordBatch, source: str) -> pa.Table:
    n = batch.num_rows
    ts = pc.strptime(batch.column("timestamp"), format="%Y-%m-%d %H:%M:%S", unit="ms")
    percentage = pc.cast(pc.round(pc.cast(batch.column("percentage"), pa.float64())), pa.int8())
    arrays = [
        ts,
        percentage,
        batch.column("depth"),
        batch.column("notional"),
        pa.array([source] * n, type=pa.string()),
    ]
    return pa.Table.from_arrays(arrays, schema=HIST_SCHEMAS["bookDepth_hist"])


def _metrics_table(batch: pa.RecordBatch, source: str) -> pa.Table:
    n = batch.num_rows
    ts = pc.strptime(batch.column("create_time"), format="%Y-%m-%d %H:%M:%S", unit="ms")
    arrays = [
        ts,
        batch.column("symbol"),
        batch.column("sum_open_interest"),
        batch.column("sum_open_interest_value"),
        # Binance: count = top-trader account ratio, sum = top-trader position
        # ratio.  Match supere.binance.parse.parse_metrics_csv.
        batch.column("sum_toptrader_long_short_ratio"),
        batch.column("count_toptrader_long_short_ratio"),
        batch.column("count_long_short_ratio"),
        batch.column("sum_taker_long_short_vol_ratio"),
        pa.array([source] * n, type=pa.string()),
    ]
    return pa.Table.from_arrays(arrays, schema=HIST_SCHEMAS["metrics_hist"])


def convert_csv(
    csv_path: Path,
    kind: str,
    raw_root: Path,
    *,
    symbol: str = "ETHUSDT",
    source: str = "binance_vision",
) -> ConvertResult:
    """Convert one extracted CSV into daily partitions under ``raw_root``."""
    csv_path = Path(csv_path)
    if kind == "aggTrades":
        columns, types = AGG_COLUMNS, _AGG_TYPES
    elif kind == "bookDepth":
        columns, types = BOOK_COLUMNS, _BOOK_TYPES
    elif kind == "metrics":
        columns, types = METRICS_COLUMNS, _METRICS_TYPES
    else:
        raise ValueError(f"未知存档类型 {kind!r}")

    skip = 1 if has_header(kind, csv_path) else 0
    reader = pacsv.open_csv(
        str(csv_path),
        read_options=pacsv.ReadOptions(
            block_size=_BLOCK,
            use_threads=False,
            column_names=columns,
            skip_rows=skip,
        ),
        convert_options=pacsv.ConvertOptions(column_types=types),
    )
    writers = _PartitionWriters(raw_root, kind, symbol)
    total = 0
    min_ts: int | None = None
    max_ts: int | None = None
    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows == 0:
                continue
            if kind == "aggTrades":
                table = _agg_table(batch, source)
            elif kind == "bookDepth":
                table = _book_table(batch, source)
            else:
                table = _metrics_table(batch, source)
            table = _sorted(table)
            ts = np.asarray(table.column("ts").to_numpy(zero_copy_only=False))
            if ts.size:
                batch_min = int(ts.min())
                batch_max = int(ts.max())
                min_ts = batch_min if min_ts is None else min(min_ts, batch_min)
                max_ts = batch_max if max_ts is None else max(max_ts, batch_max)
            days = _day_strings(table.column("ts"))
            writers.write(table, days)
            total += table.num_rows
        if total == 0:
            raise ValueError(f"空存档 {csv_path.name}")
        parquet = writers.close()
    except Exception:
        writers.abort()
        raise
    return ConvertResult(kind, total, dict(writers.counts), parquet, min_ts, max_ts)


def convert_archive(
    zip_path: Path,
    kind: str,
    raw_root: Path,
    *,
    symbol: str = "ETHUSDT",
    period: str = "daily",
    tmp_root: Path | None = None,
) -> ConvertResult:
    """Extract the first file of ``zip_path`` and convert it."""
    zip_path = Path(zip_path)
    raw_root = Path(raw_root)
    source = f"binance_vision:{period}"
    tmp_root = Path(tmp_root) if tmp_root is not None else raw_root / "_hist_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="conv-", dir=tmp_root))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [m for m in archive.namelist() if not m.endswith("/")]
            if not members:
                raise ValueError(f"{zip_path.name} 里没有文件")
            member = members[0]
            member_name = os.path.basename(member)
            target = work / member_name
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=_BLOCK)
        return convert_csv(target, kind, raw_root, symbol=symbol, source=source)
    finally:
        shutil.rmtree(work, ignore_errors=True)
