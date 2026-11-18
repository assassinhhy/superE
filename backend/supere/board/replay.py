"""Build a snapshot from fixture files using only data known at ``at_ms``."""

from __future__ import annotations

import bisect
import csv
from pathlib import Path

from supere.board.engine import analyze
from supere.board.fresh import crosscheck, judge, kline_status
from supere.board.settings import AnalysisConfig
from supere.board.snapshot import ticker_from_bars
from supere.intervals import INTERVAL_MS
from supere.timeutil import format_utc8

INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d")


def _times(fixture: dict, rows: list[dict], field: str, name: str) -> list[int]:
    """Extract a time column once per fixture, not once per snapshot.

    The index lives on ``fixture`` (so it dies with the fixture) and is keyed
    by ``(interval, field)``. ``id(rows)`` is deliberately not used: ids get
    reused after garbage collection and would return a stale index.
    """
    index = fixture.setdefault("_time_index", {})
    key = (name, field)
    cached = index.get(key)
    if cached is None or cached[0] is not rows or len(cached[1]) != len(rows):
        cached = (rows, [int(row[field]) for row in rows])
        index[key] = cached
    return cached[1]


def _num(value):
    if value is None or value == "":
        return None
    return float(value)


def load_bars(path: Path) -> list[dict]:
    bars = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bars.append(
                {
                    "open_time": int(row["open_time"]),
                    "close_time": int(row["close_time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "taker_buy": float(row["taker_buy"]),
                }
            )
    bars.sort(key=lambda bar: bar["open_time"])
    return bars


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "ts": int(row["ts"]),
                    "known_at": int(row["known_at"]),
                    "oi": _num(row.get("oi")),
                    "oi_value": _num(row.get("oi_value")),
                    "global_lsr": _num(row.get("global_lsr")),
                    "top_pos_lsr": _num(row.get("top_pos_lsr")),
                    "taker_lsr": _num(row.get("taker_lsr")),
                }
            )
    rows.sort(key=lambda item: item["ts"])
    return rows


def load_fixture(directory: Path) -> dict:
    root = Path(directory)
    frames = {}
    for interval in INTERVALS:
        path = root / f"klines_{interval}.csv"
        if path.is_file():
            frames[interval] = load_bars(path)
    metrics = load_metrics(root / "metrics_5m.csv") if (root / "metrics_5m.csv").is_file() else []
    return {"frames": frames, "metrics": metrics}


def snapshot_at(fixture: dict, at_ms: int, analysis: AnalysisConfig) -> dict:
    """Drop every bar that has not closed and every metric that is not yet knowable."""
    limit = analysis.board.warmup_bars
    grace = analysis.board.stale.kline_grace_seconds * 1000
    klines = {}
    for interval, bars in fixture["frames"].items():
        if interval == "1m":
            continue
        # Bars are ordered by open time, and close time moves with them.
        cut = bisect.bisect_right(_times(fixture, bars, "close_time", interval), at_ms)
        closed = bars[max(0, cut - limit) : cut]
        latest = closed[-1]["close_time"] if closed else None
        status = kline_status(latest, at_ms, INTERVAL_MS[interval], grace)
        klines[interval] = {
            "status": status,
            "source": "fixture",
            "data_ts": latest,
            "utc8": None if latest is None else format_utc8(latest),
            "label": "" if latest is None else format_utc8(latest, "%H:%M:%S"),
            "bars": closed if status == "ok" else closed,
        }
    metric_cut = bisect.bisect_right(_times(fixture, fixture["metrics"], "known_at", "metrics"), at_ms)
    metrics = fixture["metrics"][max(0, metric_cut - limit) : metric_cut]
    if metrics:
        known = metrics[-1]["known_at"]
        status = "ok" if at_ms - known <= analysis.board.stale.metrics_minutes * 60_000 else "stale"
    else:
        known = None
        status = "missing"
    price = None
    price_ts = None
    minute = [bar for bar in fixture["frames"].get("1m", []) if bar["close_time"] <= at_ms]
    if minute:
        price = minute[-1]["close"]
        price_ts = minute[-1]["close_time"]
    elif klines.get("5m", {}).get("bars"):
        price = klines["5m"]["bars"][-1]["close"]
        price_ts = klines["5m"]["bars"][-1]["close_time"]
    price_view = judge(price, price_ts, price_ts, "K 线", at_ms, INTERVAL_MS["5m"] + grace)
    base = klines.get("5m", {}).get("bars") or klines.get("15m", {}).get("bars") or []
    ticker = ticker_from_bars(base, at_ms) if base else None
    ticker_status = klines["5m"]["status"] if klines.get("5m", {}).get("bars") else (
        klines["15m"]["status"] if klines.get("15m", {}).get("bars") else "missing"
    )
    return {
        "symbol": analysis.board.symbol,
        "now_ms": int(at_ms),
        "now_utc8": format_utc8(int(at_ms)),
        "price": price_view,
        "ticker": {
            "value": ticker,
            "status": "missing" if ticker is None else ticker_status,
            "source": "K 线" if ticker else "",
            "data_ts": None if ticker is None else ticker["data_ts"],
            "utc8": None if ticker is None else format_utc8(ticker["data_ts"], "%H:%M:%S"),
        },
        "klines": klines,
        "metrics": {
            "status": status,
            "source": "fixture um_metrics",
            "data_ts": known,
            "rows": metrics if status == "ok" else metrics,
            "value": None if not metrics else metrics[-1]["oi"],
            "label": "未获取" if status != "ok" else format_utc8(known, "%H:%M:%S"),
        },
        "oi": judge(None, None, None, "币安 /fapi/v1/openInterest", at_ms, 180_000),
        "funding": judge(None, None, None, "markPrice", at_ms, 120_000),
        "okx": judge(None, None, None, "OKX", at_ms, 180_000),
        "crosscheck": crosscheck(None, None, analysis.board.crosscheck.max_diff_pct, binance_ok=False, okx_ok=False),
        "limit": {"active": False, "until_ms": None, "message": ""},
        "events": [],
        "calendar": [],
    }


def replay_at(directory: Path, at_ms: int, analysis: AnalysisConfig) -> dict:
    return analyze(snapshot_at(load_fixture(directory), at_ms, analysis), analysis)
