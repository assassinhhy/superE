"""News fetch for the hourly refresh and the manual refresh endpoint.

Prediction reads the cache this module fills. A single bad source is recorded
and skipped; it is never raised into the forecast path. GDELT is not on this
path (historical backfill only).
"""

from __future__ import annotations

import html
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

import httpx

from supere.news.dedup import same_text, simhash64, url_hash
from supere.news.scoring import PROXY, content_hash
from supere.storage.db import Database

BINANCE_URL = (
    "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query"
    "?type=1&pageNo=1&pageSize=20"
)
BINANCE_CATALOGS = {48, 49, 51, 128, 157, 161}
_GOOGLE_QUERY = '("ethereum" OR "ether" OR "ETH ETF" OR vitalik OR "spot ether") when:1d'
GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search?q="
    + quote_plus(_GOOGLE_QUERY)
    + "&hl=en-US&gl=US&ceid=US:en"
)
# Crypto-native feeds first. Google is a capped supplement, not the main source.
RSS_SOURCES = (
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("google_news_eth", GOOGLE_NEWS_URL),
)
PROXY_DEFAULT = PROXY
MAX_AGE_MS = 24 * 3_600_000
MAX_ITEMS = 60
GOOGLE_MAX_ITEMS = 12
_PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")

ETH_WORDS = {
    "eth", "ethereum", "ether", "vitalik", "beacon", "staking", "staker", "layer 2",
    "layer2", "l2", "defi", "erc-20", "erc20", "eip", "ethusdt", "现货", "以太坊",
}
ETH_ONLY_TOKENS = {"eth", "ethereum", "ether", "vitalik", "ethusdt"}
OTHER_ASSETS = {
    "bitcoin", "btc", "solana", "sol", "xrp", "ripple", "doge", "dogecoin", "bnb",
    "cardano", "ada", "avalanche", "avax",
}
_DROP_QUERY = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id", "oc", "ref"}
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


def resolve_proxy(explicit: str | None = None) -> str | None:
    """Proxy for news HTTP calls.

    An explicit empty string forces a direct connection. Otherwise the first
    non-empty ``HTTPS_PROXY`` / ``HTTP_PROXY`` (either case) wins, then
    ``explicit``, then the local default ``http://127.0.0.1:7890``.
    """
    if explicit == "":
        return None
    for name in _PROXY_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if explicit:
        return explicit
    return PROXY


def _eth_hit(text: str, tokens: set[str], *, strict: bool) -> bool:
    if strict:
        if tokens & ETH_ONLY_TOKENS:
            return True
    elif tokens & ETH_WORDS or tokens & ETH_ONLY_TOKENS:
        return True
    if "以太坊" in text or "以太币" in text:
        return True
    return any(phrase in text for phrase in ("eip-", "erc-20", "erc20", "ethusdt"))


def _relevant(title: str, body: str, *, eth_only: bool = False) -> bool:
    text = f"{title} {body}".lower()
    tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]+", text))
    eth = _eth_hit(text, tokens, strict=eth_only)
    if eth_only:
        return eth
    if eth:
        return True
    if tokens & OTHER_ASSETS:
        return False
    return bool(tokens & MACRO_WORDS)


def _client(proxy: str | None, timeout: float) -> httpx.Client:
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout),
        "follow_redirects": True,
        "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/131.0 Safari/537.36"},
    }
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def canonical_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in _DROP_QUERY and not key.lower().startswith("utm_")
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


def limit_source(name: str, rows: list[dict]) -> list[dict]:
    if name == "google_news_eth":
        return rows[:GOOGLE_MAX_ITEMS]
    return rows


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
            if not _relevant(title, "", eth_only=True):
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


def _rss_items(name: str, raw: bytes, now_ms: int, *, eth_only: bool = False) -> list[dict]:
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
        if not _relevant(title, body, eth_only=eth_only or name == "google_news_eth"):
            continue
        if name == "google_news_eth" and source_name:
            # Google titles carry a " - Publisher" suffix; keep the title readable.
            if title.endswith(f" - {source_name}"):
                title = title[: -(len(source_name) + 3)].strip()
        out.append({
            "source": name if name != "google_news_eth" else "google_news",
            "source_tier": 3 if name == "google_news_eth" else 2,
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
        rows = _rss_items(name, response.content, now_ms, eth_only=name == "google_news_eth")
        return limit_source(name, rows)


def _title_tokens(title: str) -> set[str]:
    text = re.sub(r"\s+-\s+[^-]+$", "", (title or "").lower())
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]{2,}|[一-鿿]{2,}", text):
        tokens.add(token)
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            tokens.add(token[:-1])
    return tokens


def titles_similar(left: str, right: str) -> bool:
    """Same event: high token overlap, or a close simhash plus shared tokens."""
    a = _title_tokens(left)
    b = _title_tokens(right)
    if not a or not b:
        return False
    overlap = len(a & b) / len(a | b)
    if overlap >= 0.72:
        return True
    if overlap >= 0.5 and same_text(simhash64(left), simhash64(right)):
        return True
    return False


def _dedupe(items: list[dict]) -> list[dict]:
    """Keep one row per URL and per similar title. Prefer the higher-tier source."""
    ordered = sorted(items, key=lambda row: (
        int(row.get("source_tier") or 9),
        -len(str(row.get("body") or "")),
        -(row.get("ts_published") or 0),
    ))
    kept: list[dict] = []
    urls: set[str] = set()
    for item in ordered:
        url = canonical_url(str(item.get("url") or ""))
        title = str(item.get("title") or "")
        if url and url in urls:
            continue
        if any(titles_similar(title, str(prev.get("title") or "")) for prev in kept):
            continue
        if url:
            urls.add(url)
        kept.append(item)
    kept.sort(key=lambda row: (-(row.get("ts_published") or 0), str(row.get("title") or "")))
    return kept


def fetch_now(now_ms: int, *, proxy: str | None = None, timeout: float = 8.0,
              max_workers: int = 6) -> dict:
    """Fetch all sources concurrently.  Never raises for a single bad source."""
    proxy = resolve_proxy(proxy)
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
