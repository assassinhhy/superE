"""Download the Deribit ETH DVOL index history as an hourly CSV.

Deribit public endpoint ``public/get_volatility_index_data`` needs no key. The
index (DVOL) is an annualised implied-volatility level; the plan uses it as a
feature for the point/interval forecast (plan section 追加 2026-11-05 15:00 (b)).

Candles are hourly. ``ts`` is the bucket open (UTC ms); a candle's ``close`` is
only known at ``ts + resolution``, so features must use that as the as-of time.

    python scripts/fetch_dvol.py --data-dir D:\\Tool\\cg\\superE\\data --currency ETH
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
MAX_CANDLES = 1000


def _ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp() * 1000)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")


def fetch_range(client: httpx.Client, currency: str, start_ms: int, end_ms: int, resolution: str,
                attempts: int = 5) -> list[list]:
    params = {
        "currency": currency.upper(),
        "start_timestamp": int(start_ms),
        "end_timestamp": int(end_ms),
        "resolution": resolution,
    }
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(URL, params=params)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return list(payload.get("result", {}).get("data") or [])
        except Exception as exc:  # noqa: BLE001 - network retries
            last = exc
            if attempt < attempts:
                time.sleep(min(30.0, 2.0 ** attempt))
    raise RuntimeError(f"DVOL 拉取失败 {_iso(start_ms)}..{_iso(end_ms)}: {last}")


def download(client: httpx.Client, currency: str, start_ms: int, end_ms: int, step_ms: int) -> dict[int, list]:
    """Hourly candles keyed by bucket open; pages forward until ``end_ms``."""
    out: dict[int, list] = {}
    cursor = start_ms
    rounds = 0
    while cursor <= end_ms:
        rounds += 1
        page = fetch_range(client, currency, cursor, min(end_ms, cursor + (MAX_CANDLES - 1) * step_ms), "3600")
        if not page:
            break
        for row in page:
            out[int(row[0])] = row
        last = int(page[-1][0])
        nxt = last + step_ms
        if nxt <= cursor:
            break
        cursor = nxt
        if rounds % 5 == 0:
            print(f"  ... {_iso(cursor)} 已取 {len(out)} 条", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_dvol")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--currency", default="ETH")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None, help="UTC ISO date; default: now")
    parser.add_argument("--out", default=None)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = parser.parse_args(argv)

    start_ms = _ms(args.start)
    end_ms = _ms(args.end) if args.end else int(time.time() * 1000)
    out_path = Path(args.out) if args.out else Path(args.data_dir) / f"dvol_{args.currency.upper()}_1h.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    old: dict[int, list] = {}
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                old[int(row["ts"])] = [int(row["ts"]), row["open"], row["high"], row["low"], row["close"]]

    with httpx.Client(proxy=args.proxy, timeout=60.0, headers={"User-Agent": "superE/0.1 research"}) as client:
        fresh = download(client, args.currency, start_ms, end_ms, 3_600_000)

    merged = {**old, **fresh}
    ordered = [merged[k] for k in sorted(merged)]
    tmp = out_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close"])
        writer.writerows(ordered)
    tmp.replace(out_path)
    print(
        f"DVOL {args.currency.upper()} {_iso(ordered[0][0])}..{_iso(ordered[-1][0])} "
        f"n={len(ordered)} -> {out_path} (新增 {len(fresh)}，旧 {len(old)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
