"""Refresh tests/fixtures/board from public www.binance.com klines. No API key.

The unit tests already carry a snapshot, so this is only for refreshing it
from a machine that can reach www.binance.com. It never calls an order route.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC8 = timezone(timedelta(hours=8))
BASE = "https://www.binance.com"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "board"


def _get(path: str):
    request = urllib.request.Request(path, headers={"User-Agent": "superE-fixture"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(datetime(2026, 11, 14, 16, 30, tzinfo=UTC8).timestamp() * 1000)
    spans = {
        "1d": int(datetime(2025, 1, 1, tzinfo=UTC8).timestamp() * 1000),
        "4h": end - 400 * 4 * 3600 * 1000,
        "1h": end - 500 * 3600 * 1000,
        "15m": end - 1500 * 15 * 60 * 1000,
        "5m": end - 1500 * 5 * 60 * 1000,
        "1m": int(datetime(2026, 11, 14, 9, 0, tzinfo=UTC8).timestamp() * 1000),
    }
    for interval, start in spans.items():
        rows = _get(
            f"{BASE}/fapi/v1/klines?symbol=ETHUSDT&interval={interval}&startTime={start}&endTime={end}&limit=1500"
        )
        with (OUT / f"klines_{interval}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["open_time", "close_time", "open", "high", "low", "close", "volume", "taker_buy"])
            for item in rows:
                writer.writerow([item[0], item[6], item[1], item[2], item[3], item[4], item[5], item[9]])
    start_m = int(datetime(2026, 11, 13, 0, 0, tzinfo=UTC8).timestamp() * 1000)
    by = {}
    for path, field in (
        ("/futures/data/openInterestHist", "oi"),
        ("/futures/data/globalLongShortAccountRatio", "global_lsr"),
        ("/futures/data/topLongShortPositionRatio", "top_pos_lsr"),
        ("/futures/data/takerlongshortRatio", "taker_lsr"),
    ):
        rows = _get(f"{BASE}{path}?symbol=ETHUSDT&period=5m&startTime={start_m}&endTime={end}&limit=500")
        for row in rows:
            known = int(row["timestamp"])
            item = by.setdefault(
                known,
                {"ts": known - 300_000, "known_at": known, "oi": "", "oi_value": "", "global_lsr": "", "top_pos_lsr": "", "taker_lsr": ""},
            )
            if field == "oi":
                item["oi"] = row.get("sumOpenInterest") or ""
                item["oi_value"] = row.get("sumOpenInterestValue") or ""
            elif field == "taker_lsr":
                item["taker_lsr"] = row.get("buySellRatio") or ""
            else:
                item[field] = row.get("longShortRatio") or ""
    with (OUT / "metrics_5m.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ts", "known_at", "oi", "oi_value", "global_lsr", "top_pos_lsr", "taker_lsr"],
        )
        writer.writeheader()
        for known in sorted(by):
            writer.writerow(by[known])


if __name__ == "__main__":
    main()
