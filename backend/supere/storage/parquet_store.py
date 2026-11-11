"""Append-only JSONL, flushed to daily zstd Parquet.

A crash loses at most the data still sitting in the JSONL buffer since the
last fsync (about one second). Startup replays leftover JSONL.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASETS = ("aggTrades", "depth", "forceOrder")

_SCHEMAS: dict[str, pa.Schema] = {
    "aggTrades": pa.schema(
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
    ),
    "depth": pa.schema(
        [
            ("recv_ms", pa.int64()),
            ("recv_local_ms", pa.int64()),
            ("event_ms", pa.int64()),
            ("transact_ms", pa.int64()),
            ("update_id", pa.int64()),
            ("bids_px", pa.list_(pa.string())),
            ("bids_qty", pa.list_(pa.string())),
            ("asks_px", pa.list_(pa.string())),
            ("asks_qty", pa.list_(pa.string())),
            ("source", pa.string()),
        ]
    ),
    "forceOrder": pa.schema(
        [
            ("event_ms", pa.int64()),
            ("recv_ms", pa.int64()),
            ("recv_local_ms", pa.int64()),
            ("trade_time", pa.int64()),
            ("symbol", pa.string()),
            ("side", pa.string()),
            ("order_type", pa.string()),
            ("tif", pa.string()),
            ("qty", pa.string()),
            ("price", pa.string()),
            ("avg_price", pa.string()),
            ("last_qty", pa.string()),
            ("accum_qty", pa.string()),
            ("status", pa.string()),
            ("source", pa.string()),
        ]
    ),
}


def partition_dir(root: Path, dataset: str, market: str, symbol: str, date: str) -> Path:
    return root / dataset / market / symbol / f"date={date}"


class ParquetStore:
    def __init__(self, raw_root: Path) -> None:
        self.root = Path(raw_root)
        self.staging = self.root / "_staging"
        self.staging.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handles: dict[str, object] = {}

    def staging_name(self, dataset: str, market: str, symbol: str, date: str) -> str:
        return f"{dataset}__{market}__{symbol}__{date}.jsonl"

    def append(self, dataset: str, market: str, symbol: str, date: str, row: dict) -> None:
        if dataset not in _SCHEMAS:
            raise ValueError(f"未知数据集 {dataset}")
        name = self.staging_name(dataset, market, symbol, date)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                path = self.staging / name
                handle = path.open("a", encoding="utf-8")
                self._handles[name] = handle
            handle.write(line + "\n")

    def fsync(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())

    def rotate(self) -> list[Path]:
        """Close open logs and rename them so a crash can finish the convert."""
        with self._lock:
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            self._handles.clear()
            ready: list[Path] = []
            for path in sorted(self.staging.glob("*.jsonl")):
                if path.stat().st_size == 0:
                    path.unlink()
                    continue
                dest = path.with_suffix(".jsonl.processing")
                if dest.exists():
                    dest = path.with_name(path.stem + f".{path.stat().st_mtime_ns}.jsonl.processing")
                path.rename(dest)
                ready.append(dest)
            ready.extend(sorted(self.staging.glob("*.jsonl.processing")))
            # unique
            seen: set[Path] = set()
            unique: list[Path] = []
            for path in ready:
                if path in seen or not path.exists():
                    continue
                seen.add(path)
                unique.append(path)
            return unique

    def convert_many(self, paths: list[Path]) -> list[Path]:
        written: list[Path] = []
        for path in paths:
            out = self.convert_file(path)
            if out is not None:
                written.append(out)
        return written

    def convert_file(self, processing: Path) -> Path | None:
        name = processing.name
        if name.endswith(".jsonl.processing"):
            stem = name[: -len(".jsonl.processing")]
        else:
            stem = processing.stem
        dataset, market, symbol, date = stem.split("__", 3)
        rows = []
        text = processing.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if not rows:
            processing.unlink(missing_ok=True)
            return None
        table = table_from_rows(dataset, rows)
        folder = partition_dir(self.root, dataset, market, symbol, date)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"part-{rows[0].get('recv_ms') or rows[0].get('ts') or 0}.parquet"
        if out.exists():
            out = folder / f"part-{rows[-1].get('recv_ms') or rows[-1].get('ts')}-{len(rows)}.parquet"
        pq.write_table(table, out, compression="zstd", compression_level=3)
        processing.unlink(missing_ok=True)
        return out

    def replay_pending(self) -> list[Path]:
        paths = self.rotate()
        return self.convert_many(paths)

    def merge_parts(self, today: str, min_parts_today: int = 12) -> int:
        merged = 0
        if not self.root.exists():
            return 0
        for folder in self.root.glob("*/*/*/date=*"):
            date = folder.name.split("=", 1)[1]
            parts = sorted(p for p in folder.glob("part-*.parquet") if p.is_file())
            if len(parts) <= 1:
                continue
            if date >= today and len(parts) < min_parts_today:
                continue
            tables = [pq.read_table(path) for path in parts]
            table = pa.concat_tables(tables, promote_options="default")
            tmp = folder / "part-merged.parquet.tmp"
            final = folder / "part-000.parquet"
            pq.write_table(table, tmp, compression="zstd", compression_level=3)
            tmp.replace(final)
            for path in parts:
                if path.resolve() != final.resolve() and path.exists():
                    path.unlink()
            merged += 1
        return merged


def table_from_rows(dataset: str, rows: list[dict]) -> pa.Table:
    schema = _SCHEMAS[dataset]
    columns: dict[str, list] = {field.name: [] for field in schema}
    for row in rows:
        for field in schema:
            columns[field.name].append(row.get(field.name))
    arrays = []
    for field in schema:
        arrays.append(pa.array(columns[field.name], type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)
