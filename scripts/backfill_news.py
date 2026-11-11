"""Backfill crypto news from the free GDELT DOC 2.0 ArtList API (plan v3, stage 5).

No key is needed.  GDELT asks for at most one request every five seconds; this
script sleeps between requests and retries 429/5xx with backoff.  Historical
items have no true collector ingest time, so ``available_at`` is the GDELT
``seendate`` and ``timing_quality='proxy_gdelt_seen'``.  That is a conservative
proxy: an article is never treated as visible before GDELT saw it.

Usage (from the train worktree)::

    python scripts/backfill_news.py --data-dir D:\\Tool\\cg\\superE\\data \
        --start 2025-01-01 --end 2026-11-05 --chunk-hours 24 --sleep 6.5

The script writes a coverage report to ``data/news_backfill_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from supere.news.dedup import simhash64, url_hash
from supere.storage.db import Database

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = "(ethereum OR ETH OR ether OR bitcoin OR crypto) sourcelang:english"
PROXY = "http://127.0.0.1:7890"


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_seen(text: str) -> int | None:
    text = (text or "").strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return None


def fetch_chunk(client: httpx.Client, start: datetime, end: datetime, query: str,
                maxrecords: int = 250, retries: int = 6) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(maxrecords),
        "sort": "datedesc",
        "startdatetime": _stamp(start),
        "enddatetime": _stamp(end),
    }
    for attempt in range(max(1, retries)):
        try:
            response = client.get(GDELT_URL, params=params)
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    # An empty result is sometimes returned as an empty body.
                    return []
                return list(payload.get("articles") or [])
            if response.status_code == 429 and retries <= 1:
                return []
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(15 + 10 * attempt)
                continue
            print(f"  GDELT {response.status_code}: {response.text[:120]}", flush=True)
            return []
        except Exception as exc:  # noqa: BLE001
            print(f"  GDELT error {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(10 + 5 * attempt)
    return []


def insert_articles(db: Database, articles: list[dict], existing: set[str]) -> int:
    inserted = 0
    for art in articles:
        url = str(art.get("url") or "").strip()
        title = str(art.get("title") or "").strip()
        if not url or not title:
            continue
        seen = _parse_seen(str(art.get("seendate") or ""))
        if seen is None:
            continue
        digest = url_hash(url)
        if digest in existing:
            continue
        existing.add(digest)
        db.insert_news({
            "ts_published": None,
            "ts_ingested": seen,
            "ts_local": seen,
            "source": "gdelt",
            "source_tier": 2,
            "lang": str(art.get("language") or "en")[:16],
            "title": title[:1000],
            "body": "",
            "url": url[:2000],
            "url_hash": digest,
            "text_simhash": str(simhash64(title)),
            "status": "new",
            "available_at": seen,
            "timing_quality": "proxy_gdelt_seen",
            "direction": 0,
        })
        inserted += 1
    return inserted


def coverage(db: Database) -> dict:
    row = db.conn.execute(
        "SELECT COUNT(*), MIN(available_at), MAX(available_at) FROM news WHERE source='gdelt'"
    ).fetchone()
    months = db.conn.execute(
        "SELECT strftime('%Y-%m', available_at/1000, 'unixepoch') AS ym, COUNT(*) "
        "FROM news WHERE source='gdelt' AND available_at IS NOT NULL GROUP BY ym ORDER BY ym"
    ).fetchall()
    return {
        "count": int(row[0] or 0),
        "first_ms": int(row[1]) if row[1] is not None else None,
        "last_ms": int(row[2]) if row[2] is not None else None,
        "months": {str(r[0]): int(r[1]) for r in months},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill_news")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", default="2025-01-01", help="UTC date")
    parser.add_argument("--end", default=None, help="UTC date; default now")
    parser.add_argument("--chunk-hours", type=int, default=24)
    parser.add_argument("--sleep", type=float, default=6.5)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--proxy", default=PROXY)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--split-saturated", action="store_true", default=True)
    args = parser.parse_args(argv)

    data_root = Path(args.data_dir).resolve()
    db = Database(data_root / "superE.sqlite")
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    existing = {row[0] for row in db.conn.execute(
        "SELECT url_hash FROM news WHERE url_hash IS NOT NULL")}
    print(f"开始回填 GDELT {args.start} .. {end.date()}，已有 {len(existing)} 条", flush=True)
    requests = 0
    inserted = 0
    chunk = timedelta(hours=max(1, args.chunk_hours))
    cursor = start
    try:
        with httpx.Client(timeout=90, proxy=args.proxy) as client:
            while cursor < end and requests < args.max_requests:
                nxt = min(cursor + chunk, end)
                articles = fetch_chunk(client, cursor, nxt, args.query)
                requests += 1
                added = insert_articles(db, articles, existing)
                inserted += added
                if added or requests % 10 == 0:
                    print(f"{cursor:%Y-%m-%d %H:%M} +{added} (total {inserted}, req {requests})", flush=True)
                # A saturated response means the window likely has more than
                # maxrecords articles. Split it once into two halves.
                if args.split_saturated and len(articles) >= 250 and (nxt - cursor) >= timedelta(hours=2):
                    mid = cursor + (nxt - cursor) / 2
                    for sub_start, sub_end in ((cursor, mid), (mid, nxt)):
                        if requests >= args.max_requests:
                            break
                        sub = fetch_chunk(client, sub_start, sub_end, args.query)
                        requests += 1
                        sub_added = insert_articles(db, sub, existing)
                        inserted += sub_added
                        print(f"  split {sub_start:%m-%d %H:%M} +{sub_added}", flush=True)
                        time.sleep(args.sleep)
                cursor = nxt
                time.sleep(args.sleep)
    finally:
        report = coverage(db)
        report["inserted_this_run"] = inserted
        report["requests"] = requests
        report["query"] = args.query
        report["timing_quality"] = "proxy_gdelt_seen"
        (data_root / "news_backfill_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        db.close()
        print(f"完成：新增 {inserted}，请求 {requests}，覆盖 {report['count']} 条", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
