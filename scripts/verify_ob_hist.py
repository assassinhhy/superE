"""Coverage / integrity report for the historical order-book raw datasets.

Reads Parquet metadata only (row counts, min/max ts, file sizes), so it is
cheap enough to run while the backfill is still going.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pyarrow.parquet as pq  # noqa: E402

from supere.hist.schema import KIND_DATASET, list_partition_dates, partition_dir  # noqa: E402

DEFAULT_ROOT = Path(r"D:\Tool\cg\superE\data\raw")
KINDS = ("aggTrades", "bookDepth", "metrics")


def _calendar_gaps(dates: list[str]) -> list[str]:
    if not dates:
        return []
    present = set(dates)
    start = dt.date.fromisoformat(dates[0])
    end = dt.date.fromisoformat(dates[-1])
    missing: list[str] = []
    cursor = start
    while cursor <= end:
        stamp = cursor.isoformat()
        if stamp not in present:
            missing.append(stamp)
        cursor += dt.timedelta(days=1)
    return missing


def _partition_stats(folder: Path, ts_column: str) -> dict:
    rows = 0
    files = 0
    size = 0
    min_ts = None
    max_ts = None
    for path in sorted(folder.glob("part-*.parquet")):
        meta = pq.read_metadata(path)
        files += 1
        rows += meta.num_rows
        size += path.stat().st_size
        for group in range(meta.num_row_groups):
            row_group = meta.row_group(group)
            for col in range(row_group.num_columns):
                column = row_group.column(col)
                if column.path_in_schema != ts_column:
                    continue
                stats = column.statistics
                if stats is None or not stats.has_min_max:
                    continue
                low = int(stats.min)
                high = int(stats.max)
                min_ts = low if min_ts is None else min(min_ts, low)
                max_ts = high if max_ts is None else max(max_ts, high)
    return {
        "files": files,
        "rows": rows,
        "bytes": size,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def report(raw_root: Path, symbol: str = "ETHUSDT") -> dict:
    out: dict = {"symbol": symbol, "datasets": {}}
    for kind in KINDS:
        dataset = KIND_DATASET[kind]
        dates = list_partition_dates(raw_root, kind, symbol)
        gaps = _calendar_gaps(dates)
        total_rows = total_bytes = 0
        min_ts = max_ts = None
        per_year: dict[str, int] = {}
        zero_days: list[str] = []
        for date in dates:
            folder = partition_dir(raw_root, kind, date, symbol)
            stats = _partition_stats(folder, "ts")
            total_rows += stats["rows"]
            total_bytes += stats["bytes"]
            if stats["rows"] == 0:
                zero_days.append(date)
            per_year[date[:4]] = per_year.get(date[:4], 0) + stats["rows"]
            if stats["min_ts"] is not None:
                min_ts = stats["min_ts"] if min_ts is None else min(min_ts, stats["min_ts"])
                max_ts = stats["max_ts"] if max_ts is None else max(max_ts, stats["max_ts"])
        out["datasets"][dataset] = {
            "days": len(dates),
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "rows": total_rows,
            "bytes": total_bytes,
            "gib": round(total_bytes / 1024**3, 3),
            "min_ts": min_ts,
            "max_ts": max_ts,
            "calendar_gap_days": gaps,
            "gap_count": len(gaps),
            "zero_row_days": zero_days,
            "per_year": dict(sorted(per_year.items())),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘口历史数据覆盖与完整性报告")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    payload = report(args.root, args.symbol)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, item in payload["datasets"].items():
        print(
            f"进度：{name} {item['days']} 日 {item['first']}..{item['last']} "
            f"行 {item['rows']:,} 占用 {item['gib']} GiB 缺日 {item['gap_count']} 零行日 {len(item['zero_row_days'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
