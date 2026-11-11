"""Build ``data/features/ob_1m`` from ``data/raw/*_hist``.

Run after the backfill stages.  One process, one row per UTC minute.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from supere.ob.build import build  # noqa: E402

DEFAULT_RAW = Path(r"D:\Tool\cg\superE\data\raw")
DEFAULT_OUT = Path(r"D:\Tool\cg\superE\data\features\ob_1m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 1m 盘口特征表")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--warmup-days", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    stats = build(
        args.raw,
        args.out,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        warmup_days=args.warmup_days,
        overwrite=args.overwrite,
        on_progress=lambda message: print(message, flush=True),
    )
    print(
        f"进度：特征表完成 写 {stats.days_written} 日 / 跳过 {stats.days_skipped} 日，"
        f"行 {stats.rows:,}，缺日 {len(stats.missing_days)}，"
        f"区间 {stats.first_minute}..{stats.last_minute}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
