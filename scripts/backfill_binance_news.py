"""Backfill historical Binance announcements (real timestamps, no key).

GDELT DOC returned HTTP 429 for this host on 2026-11-05 even after long waits.
Binance's public announcement API is reachable, paginates by catalog and gives
an exact ``releaseDate``; it is a narrower source (exchange announcements, not
general crypto media) but it is real history with a precise availability time.

The script records the actual coverage and never pretends to be a complete
news archive.  GDELT can be added later with ``scripts/backfill_news.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from supere.news.dedup import simhash64, url_hash
from supere.storage.db import Database

API = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
CATALOGS = {
    48: "new_listing",
    49: "binance_news",
    50: "activities",
    51: "fiat_listing",
    52: "api_update",
    53: "airdrop",
    54: "maintenance",
    55: "delisting",
}
PROXY = "http://127.0.0.1:7890"


def _get(client: httpx.Client, catalog: int, page: int, size: int = 50) -> list[dict]:
    for attempt in range(4):
        try:
            r = client.get(API, params={"type": 1, "pageNo": page, "pageSize": size,
                                        "catalogId": catalog})
            if r.status_code == 200:
                try:
                    cats = (r.json().get("data") or {}).get("catalogs") or []
                except ValueError:
                    return []
                return list(cats[0].get("articles") or []) if cats else []
            if r.status_code in (429, 500, 502, 503):
                time.sleep(5 + 5 * attempt)
                continue
            return []
        except Exception:
            time.sleep(3 + 3 * attempt)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill_binance_news")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--proxy", default=PROXY)
    parser.add_argument("--max-pages", type=int, default=400)
    args = parser.parse_args(argv)
    data_root = Path(args.data_dir).resolve()
    start_ms = int(datetime.strptime(args.start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    db = Database(data_root / "superE.sqlite")
    existing = {row[0] for row in db.conn.execute(
        "SELECT url_hash FROM news WHERE url_hash IS NOT NULL")}
    inserted = 0
    pages = 0
    with httpx.Client(timeout=30, proxy=args.proxy,
                      headers={"User-Agent": "superE-news/1.0"}) as client:
        for catalog, label in CATALOGS.items():
            stop = False
            for page in range(1, args.max_pages + 1):
                if stop:
                    break
                articles = _get(client, catalog, page)
                pages += 1
                if not articles:
                    break
                for art in articles:
                    release = art.get("releaseDate") or art.get("publishDate")
                    title = str(art.get("title") or "").strip()
                    code = str(art.get("code") or "").strip()
                    if not release or not title or not code:
                        continue
                    ts = int(release)
                    if ts < start_ms:
                        stop = True
                        continue
                    url = f"https://www.binance.com/en/support/announcement/{code}"
                    digest = url_hash(url)
                    if digest in existing:
                        continue
                    existing.add(digest)
                    db.insert_news({
                        "ts_published": ts,
                        "ts_ingested": ts,
                        "ts_local": ts,
                        "source": "binance_announcement",
                        "source_tier": 1,
                        "lang": "en",
                        "title": title[:1000],
                        "body": "",
                        "url": url,
                        "url_hash": digest,
                        "text_simhash": str(simhash64(title)),
                        "status": "new",
                        "available_at": ts,
                        "timing_quality": "published_time",
                        "direction": 0,
                        "category": label,
                    })
                    inserted += 1
                if inserted and inserted % 200 == 0:
                    print(f"catalog {catalog} page {page}: total {inserted}", flush=True)
                time.sleep(0.8)
    report = {
        "inserted": inserted, "pages": pages,
        "coverage": _coverage(db),
        "source": "binance_announcement",
        "timing_quality": "published_time",
        "note": "GDELT DOC was HTTP 429 from this host; Binance announcements are real but narrow.",
    }
    (data_root / "news_binance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    db.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _coverage(db: Database) -> dict:
    row = db.conn.execute(
        "SELECT COUNT(*), MIN(available_at), MAX(available_at) FROM news "
        "WHERE source='binance_announcement'"
    ).fetchone()
    months = db.conn.execute(
        "SELECT strftime('%Y-%m', available_at/1000, 'unixepoch') ym, COUNT(*) "
        "FROM news WHERE source='binance_announcement' GROUP BY ym ORDER BY ym"
    ).fetchall()
    return {"count": int(row[0] or 0),
            "first_ms": int(row[1]) if row[1] else None,
            "last_ms": int(row[2]) if row[2] else None,
            "months": {str(r[0]): int(r[1]) for r in months}}


if __name__ == "__main__":
    raise SystemExit(main())
