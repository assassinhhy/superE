"""Build the US macro release calendar (FOMC / CPI / NFP / PCE) from a free feed.

TradingView's public economic-calendar JSON is free, keyless and has history
back to 2019, so the plan's "免费源或手工维护 CSV" (追加 2026-11-05 15:00 (c))
becomes a reproducible download instead of a hand-maintained file. Release
times are already UTC. The four event types are the ones the plan names.

FOMC decisions show up as title ``Interest Rate`` (importance 1) at 18:00Z
during EDT and 19:00Z during EST. CPI is title ``CPI``; NFP is ``Non Farm
Payrolls``; PCE is ``Core PCE Price Index MoM``. Duplicates per (type, time)
are collapsed.

    python scripts/fetch_macro_calendar.py --data-dir D:\\Tool\\cg\\superE\\data
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

URL = "https://economic-calendar.tradingview.com/events"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}
# title -> canonical type. Order matters: first match wins.
RULES: tuple[tuple[str, str], ...] = (
    ("Interest Rate", "FOMC"),
    ("CPI", "CPI"),
    ("Non Farm Payrolls", "NFP"),
    ("Core PCE Price Index MoM", "PCE"),
)


def _month(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m")


def _iter_months(start: str, end: str):
    year, month = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (year, month) <= (ey, em):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def _iso_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)


def _next_iso_ms(year: int, month: int) -> int:
    return _iso_ms(year + (month == 12), 1 if month == 12 else month + 1)


def classify(title: str, importance) -> str | None:
    for text, kind in RULES:
        if title == text:
            return kind
    return None


def fetch_month(client: httpx.Client, year: int, month: int, attempts: int = 4) -> list[dict]:
    params = {
        "from": datetime.fromtimestamp(_iso_ms(year, month) / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": datetime.fromtimestamp(_next_iso_ms(year, month) / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "countries": "US",
    }
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(URL, params=params)
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("result") or [])
        except Exception as exc:  # noqa: BLE001 - network retries
            last = exc
            if attempt < attempts:
                time.sleep(min(20.0, 2.0 ** attempt))
    raise RuntimeError(f"日历拉取失败 {year}-{month:02d}: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_macro_calendar")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", default="2019-06")
    parser.add_argument("--end", default=None, help="UTC YYYY-MM; default 14 months ahead")
    parser.add_argument("--out", default=None)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    ahead = now.year * 12 + (now.month - 1) + 14
    end = args.end or f"{ahead // 12}-{ahead % 12 + 1:02d}"
    sy, sm = (int(x) for x in args.start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out_path = Path(args.out) if args.out else Path(args.data_dir) / "macro_calendar.csv"

    events: dict[tuple[str, int], dict] = {}
    with httpx.Client(proxy=args.proxy, timeout=40.0, headers=HEADERS) as client:
        for year, month in _iter_months(args.start, end):
            rows = fetch_month(client, year, month)
            kept = 0
            for row in rows:
                title = (row.get("title") or "").strip()
                kind = classify(title, row.get("importance"))
                if kind is None:
                    continue
                stamp = row.get("date") or ""
                try:
                    event_ms = int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)
                except ValueError:
                    continue
                events[(kind, event_ms)] = {
                    "event_time": event_ms,
                    "type": kind,
                    "title": title,
                    "importance": row.get("importance"),
                    "source": "tradingview",
                }
                kept += 1
            print(f"{year}-{month:02d}: 原始 {len(rows)}，保留 {kept}", flush=True)

    ordered = sorted(events.values(), key=lambda e: (e["event_time"], e["type"]))
    tmp = out_path.with_suffix(".csv.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_time", "type", "title", "importance", "source"])
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(out_path)
    counts: dict[str, int] = {}
    for event in ordered:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    print(f"calendar n={len(ordered)} {counts} -> {out_path}", flush=True)
    print(f"首个 {ordered[0]['event_time']} 末尾 {ordered[-1]['event_time']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
