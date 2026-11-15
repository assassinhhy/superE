"""Three-month board backtest. Reads public klines only and does not trade.

From the repository root. Klines are read from www.binance.com first
(fapi.binance.com answers 451 from some networks) and data.binance.vision
is the fallback:

    python scripts/board_backtest.py --compare --end "2026-11-15 18:00"

That compares the fixed-entry touch fill with the old range fill at b=0.35,
for the 48 hours ending then and for the 90 days ending then, under taker
0.05%×2 and maker 0.02%×2.

The factor sweep is unchanged:

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
import json
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from supere.board.backtest import (  # noqa: E402
    MAKER_FEE,
    TAKER_FEE,
    render_report,
    reprice,
    slice_window,
    walk,
    walk_compare,
    window_label,
)
from supere.board.replay import load_fixture  # noqa: E402
from supere.board.settings import load_analysis  # noqa: E402
from supere.timeutil import UTC8  # noqa: E402

UTC = timezone.utc
VISION = "https://data.binance.vision/data/futures/um/daily"
REST_KLINES = "https://www.binance.com/fapi/v1/klines"


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


def _utc8_ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC8).timestamp() * 1000)


def _rest_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = int(start_ms)
    while cursor < int(end_ms):
        query = urlencode(
            {
                "symbol": "ETHUSDT",
                "interval": interval,
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": 1500,
            }
        )
        payload = json.loads(_fetch(f"{REST_KLINES}?{query}"))
        if isinstance(payload, dict):
            raise RuntimeError(payload.get("msg") or payload)
        if not payload:
            break
        rows.extend(payload)
        nxt = int(payload[-1][0]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(payload) < 1500:
            break
        time.sleep(0.12)
    return rows


def _write_klines(directory: Path, interval: str, rows: list[list]) -> None:
    rows = sorted(rows, key=lambda item: int(item[0]))
    seen: set[str] = set()
    with (directory / f"klines_{interval}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy"])
        for item in rows:
            if str(item[0]) in seen:
                continue
            seen.add(str(item[0]))
            writer.writerow([item[0], item[6], item[1], item[2], item[3], item[4], item[5], item[9]])


def download_rest(directory: Path, start_ms: int, end_ms: int) -> None:
    """Public klines via www.binance.com. Vision daily zips cover a failed interval."""
    directory.mkdir(parents=True, exist_ok=True)
    warmup = {
        "5m": 20 * 24 * 3_600_000,
        "15m": 20 * 24 * 3_600_000,
        "1h": 30 * 24 * 3_600_000,
        "4h": 50 * 24 * 3_600_000,
        "1d": 250 * 24 * 3_600_000,
    }
    for interval, pad in warmup.items():
        print(f"klines {interval}", flush=True)
        try:
            rows = _rest_klines(interval, start_ms - pad, end_ms)
            if len(rows) < 50:
                raise RuntimeError(f"only {len(rows)} rows")
            source = "www.binance.com"
        except Exception as exc:
            print(f"  rest failed ({exc}); falling back to data.binance.vision", flush=True)
            source = "data.binance.vision"
            rows = _vision_interval(interval, start_ms - pad, end_ms)
        _write_klines(directory, interval, rows)
        print(f"  {len(rows)} from {source}", flush=True)


def _vision_interval(interval: str, start_ms: int, end_ms: int) -> list[list]:
    start = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date()
    end = datetime.fromtimestamp(end_ms / 1000, tz=UTC).date()
    rows: list[list] = []
    day = start
    while day <= end:
        url = f"{VISION}/klines/ETHUSDT/{interval}/ETHUSDT-{interval}-{day.isoformat()}.zip"
        try:
            text = _zip_text(_fetch(url))
        except Exception:
            day += timedelta(days=1)
            continue
        for item in csv.reader(io.StringIO(text)):
            if not item or item[0] == "open_time":
                continue
            rows.append(item)
        day += timedelta(days=1)
    return rows


def _line(label: str, fill: str, fee: str, summary: dict) -> str:
    return (
        f"| {label} | {fill} | {fee} | {summary['signals']} | {summary['counted']} | "
        f"{summary['hits']} | {summary['fails']} | {summary['invalid']} | {summary.get('unfilled', 0)} | "
        f"{summary['hit_rate']:.1%} | {summary['avg_rr']:.2f} | {summary['max_loss_streak']} |"
    )


def compare_windows(fixture: dict, analysis, *, end_text: str) -> str:
    end_ms = _utc8_ms(end_text)
    start_90 = end_ms - 90 * 24 * 3_600_000
    start_48 = end_ms - 48 * 3_600_000
    print("replaying 90d once, both fill rules", flush=True)
    compared = walk_compare(
        fixture,
        analysis,
        factor=0.35,
        slippage_ticks=0,
        fee_rate=0.0,
        start_ms=start_90 - 1,
        end_ms=end_ms - 1,
    )
    lines = [
        "# 固定进场 vs 区间进场",
        "",
        f"- 截止 {end_text} UTC+8。48 小时从那时往前推，90 天同样。b=0.35，滑点 0。",
        "- 固定进场：计划出来之后的 4 根 15m 内，价格碰到 `entry` 才按该价成交。碰不到算未成交。",
        "- 区间进场（旧）：收盘落在区间里就按收盘成交，否则按区间中点，不等触价。",
        "- 命中规则不变：先到 T1 算命中，同一根 K 碰到止损和目标算止损。T1 平一半。无效单不进分母。",
        "- taker 是单边 0.05%×2，maker 是单边 0.02%×2，都从价格盈亏里扣。",
        "",
        "| 窗口 | 进场 | 费用 | 触发 | 计入 | 命中 | 失败 | 无效 | 未成交 | 命中率 | 平均 R | 最大连亏 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, start in (("48h", start_48), ("90d", start_90)):
        for fill, summary in compared.items():
            window = slice_window(summary, start - 1, end_ms - 1)
            for fee_name, rate in (("taker 0.05%×2", TAKER_FEE), ("maker 0.02%×2", MAKER_FEE)):
                lines.append(_line(label, fill, fee_name, reprice(window, rate)))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘面情景计划回测。不下单。")
    parser.add_argument("--fixture", help="已导出的 CSV 目录")
    parser.add_argument("--download", action="store_true", help="从 data.binance.vision 拉日包")
    parser.add_argument("--compare", action="store_true", help="固定进场和区间进场，48 小时与 90 天，taker/maker")
    parser.add_argument("--start", default="2026-08-16")
    parser.add_argument("--end", default="2026-11-13")
    parser.add_argument("--cache", default="data/board/backtest-cache", help="下载缓存目录，不提交")
    parser.add_argument("--out", default="board-backtest.md")
    parser.add_argument("--analysis", default=str(ROOT / "analysis.toml"))
    args = parser.parse_args(argv)
    directory = Path(args.fixture) if args.fixture else ROOT / args.cache
    if args.compare:
        if " " not in args.end:
            print("对比需要 --end \"YYYY-MM-DD HH:MM\"（UTC+8）", file=sys.stderr)
            return 2
        end_ms = _utc8_ms(args.end)
        if args.download or not (directory / "klines_15m.csv").is_file():
            # One extra day so a signal at the window end can still reach its 24h path.
            download_rest(directory, end_ms - 90 * 24 * 3_600_000, end_ms + 24 * 3_600_000)
        fixture = load_fixture(directory)
        analysis, _path = load_analysis(args.analysis)
        report = compare_windows(fixture, analysis, end_text=args.end)
        Path(args.out).write_text(report, encoding="utf-8")
        print(report)
        print(f"wrote {args.out}")
        return 0
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
