"""Realtime news collector (plan v3, stage 5).

Sources:
* GDELT DOC ArtList for the last window (free, no key), with ``seendate`` as a
  conservative visibility proxy.
* Binance announcement API as a short-latency supplement (failure tolerated).
* RSS feeds are optional; a source that fails never stops the loop.

The collector writes to ``superE.sqlite`` and runs the scorer (DeepSeek when
configured, rule fallback otherwise).  ``/api/predict`` never depends on it.

Run once for acceptance or continuously:

    python scripts/collect_news.py --data-dir D:\\Tool\\cg\\superE\\data --once
    python scripts/collect_news.py --data-dir D:\\Tool\\cg\\superE\\data --interval 300
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from supere.news.dedup import simhash64, url_hash
from supere.news.scoring import PROXY, score_pending
from supere.storage.db import Database

try:
    from scripts.backfill_news import DEFAULT_QUERY, fetch_chunk, insert_articles
except ImportError:  # running as ``python scripts/collect_news.py``
    from backfill_news import DEFAULT_QUERY, fetch_chunk, insert_articles

BINANCE_ANNOUNCEMENTS = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    "?type=1&pageNo=1&pageSize=20&catalogId=48"
)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def collect_gdelt(db: Database, existing: set[str], *, hours: int = 12,
                  proxy: str = PROXY) -> int:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    with httpx.Client(timeout=60, proxy=proxy) as client:
        articles = fetch_chunk(client, start, end, DEFAULT_QUERY, retries=1)
    return insert_articles(db, articles, existing)


def collect_binance(db: Database, existing: set[str], *, proxy: str = PROXY) -> int:
    try:
        with httpx.Client(timeout=25, proxy=proxy,
                          headers={"User-Agent": "superE-news/1.0"}) as client:
            response = client.get(BINANCE_ANNOUNCEMENTS)
        if response.status_code != 200:
            return 0
        payload = response.json()
        catalogs = (payload.get("data") or {}).get("catalogs") or []
        articles: list[dict] = []
        for catalog in catalogs:
            for article in catalog.get("articles") or []:
                title = str(article.get("title") or "").strip()
                code = str(article.get("code") or "")
                if not title or not code:
                    continue
                release = article.get("releaseDate") or article.get("publishDate")
                articles.append({
                    "title": title,
                    "url": f"https://www.binance.com/en/support/announcement/{code}",
                    "seendate": datetime.fromtimestamp(
                        int(release) / 1000, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    if release else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                    "language": "en",
                })
        inserted = 0
        for art in articles:
            url = art["url"]
            digest = url_hash(url)
            if digest in existing:
                continue
            seen = None
            try:
                seen = int(datetime.strptime(art["seendate"], "%Y%m%dT%H%M%SZ")
                           .replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                seen = _now_ms()
            existing.add(digest)
            db.insert_news({
                "ts_published": seen, "ts_ingested": _now_ms(), "ts_local": _now_ms(),
                "source": "binance_announcement", "source_tier": 1, "lang": "en",
                "title": art["title"][:1000], "body": "", "url": url[:2000],
                "url_hash": digest, "text_simhash": str(simhash64(art["title"])),
                "status": "new", "available_at": max(seen, _now_ms()),
                "timing_quality": "ingested", "direction": 0,
            })
            inserted += 1
        return inserted
    except Exception as exc:  # noqa: BLE001
        print(f"binance announcements skipped: {type(exc).__name__}", flush=True)
        return 0


def cycle(db: Database, existing: set[str], *, proxy: str = PROXY,
          score_limit: int = 500) -> dict:
    added = collect_gdelt(db, existing, proxy=proxy)
    added += collect_binance(db, existing, proxy=proxy)
    counts = score_pending(Path(db.path), limit=score_limit, use_deepseek=True, proxy=proxy)
    return {"added": added, "score": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collect_news")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--proxy", default=PROXY)
    args = parser.parse_args(argv)
    path = Path(args.data_dir).resolve() / "superE.sqlite"
    db = Database(path)
    existing = {row[0] for row in db.conn.execute(
        "SELECT url_hash FROM news WHERE url_hash IS NOT NULL")}
    try:
        while True:
            started = time.time()
            try:
                result = cycle(db, existing, proxy=args.proxy)
                print(f"{datetime.now().strftime('%H:%M:%S')} collector {json.dumps(result, ensure_ascii=False)}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not stop the loop
                print(f"{datetime.now().strftime('%H:%M:%S')} collector error {type(exc).__name__}: {exc}",
                      flush=True)
            if args.once:
                return 0
            time.sleep(max(10.0, args.interval - (time.time() - started)))
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
