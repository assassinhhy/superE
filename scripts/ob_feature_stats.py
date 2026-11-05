"""Sample statistics for ``data/features/ob_1m``.

One pass over the feature partitions.  For every column it reports the NaN
rate, mean, standard deviation, min and max.  The output is the "特征样本统计"
section of the DONE report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from supere.hist.schema import list_partition_dates  # noqa: E402

DEFAULT_OUT = Path(r"D:\Tool\cg\superE\data\features\ob_1m")
DEFAULT_RAW = Path(r"D:\Tool\cg\superE\data\raw")


def stats(root: Path, raw_root: Path | None = None) -> dict:
    files = sorted(root.glob("um/*/date=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root} 下没有特征分区")
    columns: dict[str, dict] = {}
    per_year: dict[str, int] = {}
    rows = 0
    total_bytes = 0
    first = last = None
    feature_dates: set[str] = set()
    for path in files:
        table = pq.read_table(path)
        total_bytes += path.stat().st_size
        rows += table.num_rows
        date = path.parent.name.split("=", 1)[1]
        feature_dates.add(date)
        per_year[date[:4]] = per_year.get(date[:4], 0) + table.num_rows
        minute = table.column("minute_ms").to_numpy(zero_copy_only=False)
        if minute.size:
            low, high = int(minute.min()), int(minute.max())
            first = low if first is None else min(first, low)
            last = high if last is None else max(last, high)
        for name in table.schema.names:
            if name == "minute_ms":
                continue
            values = np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
            valid = values[np.isfinite(values)]
            entry = columns.setdefault(
                name,
                {"n": 0, "nan": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None},
            )
            entry["n"] += int(valid.size)
            entry["nan"] += int(values.size - valid.size)
            if valid.size:
                entry["sum"] += float(valid.sum())
                entry["sumsq"] += float((valid * valid).sum())
                low, high = float(valid.min()), float(valid.max())
                entry["min"] = low if entry["min"] is None else min(entry["min"], low)
                entry["max"] = high if entry["max"] is None else max(entry["max"], high)
    out_columns = {}
    for name, item in columns.items():
        total = item["n"] + item["nan"]
        mean = item["sum"] / item["n"] if item["n"] else None
        variance = item["sumsq"] / item["n"] - mean * mean if item["n"] else None
        out_columns[name] = {
            "nan_rate": item["nan"] / total if total else None,
            "mean": mean,
            "std": float(np.sqrt(max(variance, 0.0))) if variance is not None else None,
            "min": item["min"],
            "max": item["max"],
        }
    return {
        "rows": rows,
        "days": len(files),
        "first_minute": first,
        "last_minute": last,
        "bytes": total_bytes,
        "gib": round(total_bytes / 1024**3, 3),
        "per_year": dict(sorted(per_year.items())),
        "missing_raw_days": (
            sorted(set(list_partition_dates(raw_root, "aggTrades")) - feature_dates)
            if raw_root is not None
            else []
        ),
        "columns": out_columns,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1m 盘口特征样本统计")
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    payload = stats(args.root, args.raw)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"进度：特征统计 {payload['days']} 日 {payload['rows']:,} 行 "
        f"{payload['gib']} GiB 区间 {payload['first_minute']}..{payload['last_minute']} "
        f"缺日 {len(payload['missing_raw_days'])}"
    )
    for name, item in payload["columns"].items():
        nan = item["nan_rate"]
        nan_text = f"{nan:.3f}" if nan is not None else "n/a"
        print(
            f"  {name}: nan={nan_text} mean={item['mean']} std={item['std']} "
            f"min={item['min']} max={item['max']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
