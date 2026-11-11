"""Parquet schemas for archives under ``data/raw/<dataset>_hist``.

``aggTrades_hist`` is byte-for-byte schema compatible with the live
``aggTrades`` dataset.  ``recv_ms`` and ``recv_local_ms`` carry the exchange
trade timestamp because no local receive time exists for an archive.

Partition dates are the archive's own UTC date.  The live collector partitions
by UTC+8 session date; historical archives are UTC-day files, so keeping UTC
avoids moving rows of a single archive across partitions.  Consumers should
join on ``ts``, not on the partition name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa

AGG_TRADES_HIST = pa.schema(
    [
        ("agg_id", pa.int64()),
        ("price", pa.string()),
        ("qty", pa.string()),
        ("first_id", pa.int64()),
        ("last_id", pa.int64()),
        ("ts", pa.int64()),
        ("is_buyer_maker", pa.bool_()),
        ("recv_ms", pa.int64()),
        ("recv_local_ms", pa.int64()),
        ("source", pa.string()),
    ]
)

BOOK_DEPTH_HIST = pa.schema(
    [
        ("ts", pa.int64()),
        ("percentage", pa.int8()),
        ("depth", pa.string()),
        ("notional", pa.string()),
        ("source", pa.string()),
    ]
)

METRICS_HIST = pa.schema(
    [
        ("ts", pa.int64()),
        ("symbol", pa.string()),
        ("oi", pa.string()),
        ("oi_value", pa.string()),
        ("top_pos_lsr", pa.string()),
        ("top_acc_lsr", pa.string()),
        ("global_lsr", pa.string()),
        ("taker_lsr", pa.string()),
        ("source", pa.string()),
    ]
)

HIST_SCHEMAS: dict[str, pa.Schema] = {
    "aggTrades_hist": AGG_TRADES_HIST,
    "bookDepth_hist": BOOK_DEPTH_HIST,
    "metrics_hist": METRICS_HIST,
}

# Archive family -> output dataset name.
KIND_DATASET = {
    "aggTrades": "aggTrades_hist",
    "bookDepth": "bookDepth_hist",
    "metrics": "metrics_hist",
}

# Calendar availability on data.binance.vision for ETHUSDT.
KIND_FIRST_DAY = {
    "aggTrades": "2019-12-31",
    "bookDepth": "2023-01-01",
    "metrics": "2021-12-01",
}

DATASET_ROOT_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def dataset_name(kind: str) -> str:
    try:
        return KIND_DATASET[kind]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ValueError(f"未知盘口存档类型 {kind!r}") from exc


def partition_dir(raw_root: Path, kind: str, date: str, symbol: str = "ETHUSDT", market: str = "um") -> Path:
    return Path(raw_root) / dataset_name(kind) / market / symbol.upper() / f"date={date}"


def list_partition_dates(raw_root: Path, kind: str, symbol: str = "ETHUSDT", market: str = "um") -> list[str]:
    root = Path(raw_root) / dataset_name(kind) / market / symbol.upper()
    if not root.exists():
        return []
    dates: list[str] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        match = DATASET_ROOT_RE.match(folder.name)
        if match and any(folder.glob("part-*.parquet")):
            dates.append(match.group(1))
    return sorted(dates)
