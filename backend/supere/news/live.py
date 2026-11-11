"""Click-time news fetch.  Runs only when the user clicks 预测.

The collector may cache news in the background, but this module is the
foreground path: fetch now -> score -> aggregate -> overlay -> model call.
Every source is isolated: a timeout, HTTP error or malformed feed is recorded
and skipped, never raised into the forecast path.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from supere.news.dedup import url_hash
from supere.news.scoring import PROXY, content_hash
from supere.storage.db import Database

BINANCE_URL = (
    "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query"
    "?type=1&pageNo=1&pageSize=20"
)
BINANCE_CATALOGS = {48, 49, 51, 128, 157, 161}
RSS_SOURCES = (
    ("google_news_eth", "https://news.google.com/rss/search?q=ethereum+OR+ETH+crypto&hl=en-US&gl=US&ceid=US:en"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("decrypt", "https://decrypt.co/feed"),
)
PROXY_DEFAULT = PROXY
MAX_AGE_MS = 48 * 3_600_000
MAX_ITEMS = 80

ETH_WORDS = {
    "eth", "ethereum", "ether", "vitalik", "beacon", "staking", "staker", "layer 2",
    "layer2", "l2", "defi", "erc-20", "erc20", "eip", "ethusdt", "现货", "以太坊",
}
# Broad crypto/market-wide items are kept only when they can plausibly move ETH
# through a macro, regulatory, exchange or liquidity channel.  Generic "crypto"
# headlines from the Google feed were too noisy for an ETH panel.
MACRO_WORDS = {
    "fed", "fomc", "cpi", "inflation", "rate", "tariff", "sec", "etf", "hack",
    "exploit", "security incident", "liquidation", "regulation", "lawsuit",
    "crash", "plunge", "rally", "surge",
}
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass
class FetchReport:
    sources: dict[str, dict] = field(default_factory=dict)

    def record(self, name: str, count: int, error: str | None = None) -> None:
        self.sources[name] = {
            "ok": error is None,
            "count": int(count),
            "error": (error or "")[:160] or None,
        }


def _text(value: Any, limit: int = 2000) -> str:
    raw = html.unescape(str(value or ""))
    raw = TAG_RE.sub(" ", raw)
    return SPACE_RE.sub(" ", raw).strip()[:limit]


def _parse_time(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _relevant(title: str, body: str) -> bool:
    text = f"{title} {body}".lower()
    tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]+", text))
    return bool(tokens & ETH_WORDS) or bool(tokens & MACRO_WORDS)


def _client(proxy: str | None, timeout: float) -> httpx.Client:
    return httpx.Client(
        proxy=proxy,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/131.0 Safari/537.36"},
    )


def _fetch_binance(proxy: str | None, timeout: float, now_ms: int) -> list[dict]:
    out: list[dict] = []
    with _client(proxy, timeout) as client:
        response = client.get(BINANCE_URL)
        response.raise_for_status()
        payload = response.json()
    catalogs = (payload.get("data") or {}).get("catalogs") or []
    for catalog in catalogs:
        try:
            catalog_id = int(catalog.get("catalogId"))
        except (TypeError, ValueError):
            catalog_id = 0
        if catalog_id and catalog_id not in BINANCE_CATALOGS:
            continue
        for article in catalog.get("articles") or []:
            title = _text(article.get("title"), 500)
            code = str(article.get("code") or "").strip()
            if not title or not code:
                continue
            published = None
            release = article.get("releaseDate") or article.get("publishDate")
            try:
                published = int(release) if release else None
            except (TypeError, ValueError):
                published = None
            if published is not None and now_ms - published > MAX_AGE_MS:
                continue
            if not _relevant(title, ""):
                continue
            out.append({
                "source": "binance_announcement",
                "source_tier": 1,
                "lang": "en",
                "title": title,
                "body": "",
                "url": f"https://www.binance.com/en/support/announcement/{code}",
                "ts_published": published,
            })
    return out


def _rss_items(name: str, raw: bytes, now_ms: int) -> list[dict]:
    root = ET.fromstring(raw)
    out: list[dict] = []
    nodes = root.findall(".//item")
    if not nodes:
        nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for node in nodes:
        def child(*tags: str) -> str:
            for tag in tags:
                found = node.find(tag)
                if found is not None and (found.text or "").strip():
                    return found.text.strip()
                for path in (f"{{http://www.w3.org/2005/Atom}}{tag}",):
                    found = node.find(path)
                    if found is not None and (found.text or "").strip():
                        return found.text.strip()
            return ""

        title = _text(child("title"), 500)
        link = child("link")
        if not link:
            link_node = node.find("{http://www.w3.org/2005/Atom}link")
            if link_node is not None:
                link = link_node.get("href") or ""
        body = _text(child("description", "summary", "content"), 800)
        published = _parse_time(child("pubDate", "published", "updated"))
        source_node = node.find("source")
        source_name = _text(source_node.text if source_node is not None else "", 60)
        if not title or not link:
            continue
        if published is not None and now_ms - published > MAX_AGE_MS:
            continue
        if not _relevant(title, body):
            continue
        if name == "google_news_eth" and source_name:
            # Google titles carry a " - Publisher" suffix; keep the title readable.
            if title.endswith(f" - {source_name}"):
                title = title[: -(len(source_name) + 3)].strip()
        out.append({
            "source": name if name != "google_news_eth" else "google_news",
            "source_tier": 2,
            "lang": "en",
            "title": title,
            "body": body,
            "url": link.strip(),
            "ts_published": published,
        })
    return out


def _fetch_rss(name: str, url: str, proxy: str | None, timeout: float, now_ms: int) -> list[dict]:
    with _client(proxy, timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        return _rss_items(name, response.content, now_ms)


def _dedupe(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    urls: set[str] = set()
    titles: set[str] = set()
    for item in sorted(items, key=lambda row: (-(row.get("ts_published") or 0), row.get("title") or "")):
        url = str(item.get("url") or "").strip()
        title = re.sub(r"\W+", "", str(item.get("title") or "").lower())[:120]
        if url and url in urls:
            continue
        if title and title in titles:
            continue
        if url:
            urls.add(url)
        if title:
            titles.add(title)
        out.append(item)
    return out


def fetch_now(now_ms: int, *, proxy: str | None = PROXY_DEFAULT, timeout: float = 7.0,
              max_workers: int = 6) -> dict:
    """Fetch all sources concurrently.  Never raises for a single bad source."""
    report = FetchReport()
    collected: list[dict] = []
    tasks: list[tuple[str, Any]] = [("binance_announcement", lambda: _fetch_binance(proxy, timeout, now_ms))]
    tasks += [(name, lambda n=name, u=url: _fetch_rss(n, u, proxy, timeout, now_ms))
              for name, url in RSS_SOURCES]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks}
        done, pending = wait(futures, timeout=timeout + 3.0)
        for future in pending:
            report.record(futures[future], 0, "timeout")
            future.cancel()
        for future in done:
            name = futures[future]
            try:
                rows = future.result() or []
                collected.extend(rows)
                report.record(name, len(rows))
            except Exception as exc:  # noqa: BLE001 - one source must not block the rest
                report.record(name, 0, f"{type(exc).__name__}: {exc}")
    items = _dedupe(collected)[:MAX_ITEMS]
    failed = [name for name, meta in report.sources.items() if not meta.get("ok")]
    status = "ok" if items and not failed else ("partial" if items else ("error" if failed else "empty"))
    return {"items": items, "sources": report.sources, "fetched_at": int(now_ms), "status": status}


def persist(items: list[dict], db_path: Path, now_ms: int) -> list[dict]:
    """Best-effort cache into the news table; failures keep the in-memory rows."""
    if not items:
        return []
    try:
        db = Database(Path(db_path))
    except Exception:  # noqa: BLE001 - news must not block the forecast
        return items
    try:
        existing = {row[0] for row in db.conn.execute(
            "SELECT url_hash FROM news WHERE url_hash IS NOT NULL"
        )}
        for item in items:
            url = str(item.get("url") or "").strip()
            digest = url_hash(url) if url else None
            row = db.conn.execute(
                "SELECT id, summary_zh, sentiment_label FROM news WHERE url_hash=? ORDER BY id DESC LIMIT 1",
                (digest,),
            ).fetchone() if digest else None
            if row is not None:
                item["id"] = int(row["id"])
                item["summary_zh"] = row["summary_zh"] or ""
                item["sentiment_label"] = row["sentiment_label"] or ""
                continue
            if digest:
                existing.add(digest)
            published = item.get("ts_published")
            title = str(item.get("title") or "")[:1000]
            body = str(item.get("body") or "")[:3000]
            news_id = db.insert_news({
                "ts_published": published,
                "ts_ingested": int(now_ms),
                "ts_local": int(now_ms),
                "source": item.get("source") or "unknown",
                "source_tier": int(item.get("source_tier") or 2),
                "lang": item.get("lang") or "en",
                "title": title,
                "body": body,
                "url": url[:2000],
                "url_hash": digest,
                "text_simhash": "0",
                "status": "new",
                # Event time is used for decay; available_at is the conservative
                # "seen by us" time and is never earlier than the first fetch.
                "available_at": int(now_ms),
                "timing_quality": "click_fetch",
                "content_hash": content_hash(title, body),
                "direction": 0,
            })
            item["id"] = int(news_id)
            item["available_at"] = int(now_ms)
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass
    return items


def snapshot(now_ms: int, db_path: Path, *, proxy: str | None = PROXY_DEFAULT,
             timeout: float = 7.0) -> dict:
    fetched = fetch_now(now_ms, proxy=proxy, timeout=timeout)
    items = persist(fetched["items"], db_path, now_ms)
    # Persisted rows carry available_at; fresh in-memory rows use now_ms.
    for item in items:
        item.setdefault("available_at", int(now_ms))
        item.setdefault("ts_published", None)
        item.setdefault("summary_zh", "")
        item.setdefault("sentiment_label", "")
    return {
        "items": items,
        "sources": fetched["sources"],
        "status": fetched["status"],
        "fetched_at": fetched["fetched_at"],
    }
