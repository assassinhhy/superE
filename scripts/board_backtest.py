"""Three-month board backtest. Reads public klines only and does not trade.

From the repository root, with a network path to data.binance.vision:

    python scripts/board_backtest.py --download --start 2026-08-16 --end 2026-11-13 --out board-backtest.md

If this machine cannot fetch history, point it at CSVs exported on the home PC
(same columns as tests/fixtures/board):

    python scripts/board_backtest.py --fixture D:\\Tool\\cg\\superE\\data\\board\\fixture --out board-backtest.md

`--fixture` needs klines_15m.csv, klines_5m.csv, klines_1h.csv, klines_4h.csv,
klines_1d.csv and metrics_5m.csv. The report sweeps ATR1h stop factors
0.25, 0.35 and 0.5, each with fees only and with fees plus 1 tick.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from supere.board.backtest import render_report, walk, window_label  # noqa: E402
from supere.board.replay import load_fixture  # noqa: E402
from supere.board.settings import load_analysis  # noqa: E402

UTC = timezone.utc
VISION = "https://data.binance.vision/data/futures/um/daily"


def _days(start: str, end: str) -> list[str]:
    cursor = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    while cursor <= last:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "superE-backtest"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _zip_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = archive.namelist()[0]
        return archive.read(name).decode("utf-8")


def download(directory: Path, start: str, end: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    days = _days(start, end)
    kline_rows = {interval: [] for interval in ("5m", "15m", "1h", "4h", "1d")}
    # Daily RSI needs history before the window.
    warmup_start = (datetime.strptime(start, "%Y-%m-%d").date() - timedelta(days=250)).isoformat()
    for day in _days(warmup_start, end):
        url = f"{VISION}/klines/ETHUSDT/1d/ETHUSDT-1d-{day}.zip"
        try:
            text = _zip_text(_fetch(url))
        except Exception:
            continue
        for row in csv.reader(io.StringIO(text)):
            if not row or row[0] == "open_time":
                continue
            kline_rows["1d"].append(row)
    metrics = []
    for index, day in enumerate(days, start=1):
        for interval in ("5m", "15m", "1h", "4h"):
            url = f"{VISION}/klines/ETHUSDT/{interval}/ETHUSDT-{interval}-{day}.zip"
            text = _zip_text(_fetch(url))
            for row in csv.reader(io.StringIO(text)):
                if not row or row[0] == "open_time":
                    continue
                kline_rows[interval].append(row)
        url = f"{VISION}/metrics/ETHUSDT/ETHUSDT-metrics-{day}.zip"
        text = _zip_text(_fetch(url))
        for row in csv.DictReader(io.StringIO(text)):
            stamp = datetime.strptime(row["create_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            ts = int(stamp.timestamp() * 1000)
            metrics.append(
                {
                    "ts": ts,
                    "known_at": ts + 300_000,
                    "oi": row.get("sum_open_interest") or "",
                    "oi_value": row.get("sum_open_interest_value") or "",
                    "global_lsr": row.get("count_long_short_ratio") or "",
                    "top_pos_lsr": row.get("sum_toptrader_long_short_ratio") or "",
                    "taker_lsr": row.get("sum_taker_long_short_vol_ratio") or "",
                }
            )
        print(f"downloaded {index}/{len(days)} {day}", flush=True)
    for interval, rows in kline_rows.items():
        rows.sort(key=lambda item: int(item[0]))
        seen = set()
        with (directory / f"klines_{interval}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy"])
            for item in rows:
                if item[0] in seen:
                    continue
                seen.add(item[0])
                writer.writerow([item[0], item[6], item[1], item[2], item[3], item[4], item[5], item[9]])
    metrics.sort(key=lambda item: item["ts"])
    with (directory / "metrics_5m.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ts", "known_at", "oi", "oi_value", "global_lsr", "top_pos_lsr", "taker_lsr"],
        )
        writer.writeheader()
        seen = set()
        for item in metrics:
            if item["ts"] in seen:
                continue
            seen.add(item["ts"])
            writer.writerow(item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘面情景计划回测。不下单。")
    parser.add_argument("--fixture", help="已导出的 CSV 目录")
    parser.add_argument("--download", action="store_true", help="从 data.binance.vision 拉日包")
    parser.add_argument("--start", default="2026-08-16")
    parser.add_argument("--end", default="2026-11-13")
    parser.add_argument("--cache", default="data/board/backtest-cache", help="下载缓存目录，不提交")
    parser.add_argument("--out", default="board-backtest.md")
    parser.add_argument("--analysis", default=str(ROOT / "analysis.toml"))
    args = parser.parse_args(argv)
    directory = Path(args.fixture) if args.fixture else ROOT / args.cache
    if args.download or not args.fixture:
        if not args.download and not (directory / "klines_15m.csv").is_file():
            print("没有现成的 CSV。加上 --download，或用 --fixture 指向本机导出的目录。", file=sys.stderr)
            return 2
        if args.download:
            download(directory, args.start, args.end)
    fixture = load_fixture(directory)
    bars = fixture["frames"].get("15m") or []
    if len(bars) < 200:
        print(f"15m K 线只有 {len(bars)} 根，不够做回测。", file=sys.stderr)
        return 2
    analysis, _path = load_analysis(args.analysis)
    rows = []
    for factor in (0.25, 0.35, 0.5):
        for ticks in (0, 1):
            print(f"running b={factor} slippage={ticks}", flush=True)
            rows.append(walk(fixture, analysis, factor=factor, slippage_ticks=ticks))
    start, end = window_label(bars)
    report = render_report(rows, start=start, end=end, bars=len(bars))
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
