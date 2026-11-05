"""Backfill one symbol's UM klines into the 1a SQLite, reusing the collector code.

The 1a collector only backfills its configured symbol. This script runs the same
``Backfill.backfill_klines`` path (monthly archives -> daily archives -> REST
tail) for another symbol, so the extra series land in the same database with
the same ``source``/``ingested_at`` bookkeeping. Read the plan section
"追加 2026-11-05 15:00" (a).

It only writes K lines. It never trades, never reads keys, and it does not
touch the running collector: SQLite WAL serialises the two writers.

    python scripts/backfill_klines.py --config superE-1b/config.toml \
        --data-dir D:\\Tool\\cg\\superE\\data --symbol BTCUSDT \
        --intervals 5m,15m,30m,1h,4h
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from supere.binance.rest import BinanceRest
from supere.config import data_dir, load_config
from supere.ingest.backfill import Backfill
from supere.ingest.gapfill import recompute_coverage
from supere.ratelimit import WeightLimiter
from supere.storage.db import Database

PRODUCT_INTERVALS = ("5m", "15m", "30m", "1h", "4h")
log = logging.getLogger("backfill_klines")


def _stamp() -> tuple[int, int]:
    now = int(time.time() * 1000)
    return now, now


async def _run(args) -> int:
    cfg, config_path = load_config(args.config)
    root = Path(args.data_dir).resolve() if args.data_dir else data_dir(cfg, config_path)
    cfg._archive_dir = root / "archive_cache"  # type: ignore[attr-defined]
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    unknown = [i for i in intervals if i not in PRODUCT_INTERVALS]
    if unknown:
        print(f"不支持的周期 {unknown}", file=sys.stderr)
        return 2
    db_path = root / "superE.sqlite"
    if not db_path.exists():
        print(f"找不到数据库 {db_path}", file=sys.stderr)
        return 2

    limits = cfg.rate_limit
    limiter = WeightLimiter(
        "futures",
        limits.futures_weight_per_min,
        budget_ratio=limits.futures_budget_ratio,
        header_slowdown_ratio=limits.header_slowdown_ratio,
    )
    rest = BinanceRest(cfg, limiter, limiter, limiter)

    async def on_event(level: str, component: str, code: str, msg: str, detail: dict | None = None) -> None:
        line = f"[{level}] {component} {code} {msg}"
        print(line, flush=True)
        log.info(line)

    db = Database(db_path, read_only=False)
    try:
        backfill = Backfill(cfg, rest, db, _stamp, on_event)
        for interval in intervals:
            started = datetime.now(UTC)
            print(f"== {args.symbol} {interval} 开始 {started:%Y-%m-%d %H:%M:%S} UTC", flush=True)
            await backfill.backfill_klines("um", args.symbol.upper(), interval)
            server_ms, _ = _stamp()
            gaps = recompute_coverage(db, "um", args.symbol.upper(), interval, server_ms)
            bounds = db.min_max_open("um", args.symbol.upper(), interval)
            elapsed = (datetime.now(UTC) - started).total_seconds()
            print(
                f"== {args.symbol} {interval} 完成 {elapsed:.0f}s rows={db.count_klines('um')} "
                f"bounds={bounds} gaps={len(gaps)}",
                flush=True,
            )
    finally:
        db.close()
        await rest.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill_klines")
    parser.add_argument("--config", required=True, help="1a/1b 的 config.toml（只读用它取端点与目录）")
    parser.add_argument("--data-dir", default=None, help="数据目录；默认按 config 解析")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--intervals", default=",".join(PRODUCT_INTERVALS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
