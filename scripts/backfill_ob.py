"""Run the order-book archive backfill.

Example::

    python scripts/backfill_ob.py --stage all --workers 4

Stages run in the plan's order: bookDepth, recent aggTrades, metrics, then the
older aggTrades months.  Re-running is safe: verified zips are reused, written
partitions are skipped when the manifest and the Parquet file both exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import portalocker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from supere.hist.archives import VisionClient  # noqa: E402
from supere.hist.backfill import STAGES, build_items, monthly_gap_items, run_stage  # noqa: E402

DEFAULT_ROOT = Path(r"D:\Tool\cg\superE\data\raw")
DEFAULT_PROXY = "http://127.0.0.1:7890"


def _print(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘口类历史回填（不训练）")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--end", default=None, help="最后一天 YYYY-MM-DD（默认取存档最新日）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - console may not support it
        pass
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )

    proxy = None if args.no_proxy else args.proxy
    client = VisionClient(proxy=proxy, timeout=120.0, retries=5)
    end = dt.date.fromisoformat(args.end) if args.end else None
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    if args.dry_run:
        for stage in stages:
            items, missing = (
                monthly_gap_items(args.root, client, args.symbol)
                if stage == "aggtrades-gapfill"
                else build_items(stage, symbol=args.symbol, client=client, end=end)
            )
            monthly = sum(1 for item in items if item.period == "monthly")
            daily = len(items) - monthly
            _print(
                f"进度：dry-run {stage} 共 {len(items)} 个存档（monthly {monthly} / daily {daily}），"
                f"缺失槽位 {len(missing)}"
            )
            if missing[:5]:
                _print(f"进度：{stage} 缺失样例 {missing[:5]}")
        return 0

    reports = []
    failed_total = 0
    lock_path = args.root / "_hist" / "backfill.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = portalocker.Lock(str(lock_path), mode="a+", timeout=0)
        lock.acquire()
    except portalocker.exceptions.LockException:
        _print(f"进度：已有回填进程持有锁 {lock_path}，退出")
        return 2
    try:
        for stage in stages:
            _print(f"进度：开始阶段 {stage}")
            report = run_stage(
                stage,
                client=client,
                raw_root=args.root,
                symbol=args.symbol,
                workers=args.workers,
                end=end,
                on_progress=_print,
            )
            reports.append(report.__dict__)
            failed_total += report.failed
            _print(
                f"进度：阶段 {stage} 结束 完成 {report.converted} 跳过 {report.skipped} "
                f"缺失 {report.missing} 失败 {report.failed} 行 {report.rows:,}"
            )
    finally:
        try:
            lock.release()
        except Exception:  # noqa: BLE001
            pass

    summary = args.root / "_hist" / "stage_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(f"进度：回填阶段完成，摘要 {summary}，失败 {failed_total}")
    return 1 if failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
