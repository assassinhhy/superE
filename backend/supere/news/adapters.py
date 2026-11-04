"""Source adapters. Phase 1a ships the interface and the fear-and-greed source.

Other sources in the plan (Odaily, JinSe, RSS, calendars) are phase 2.
Enable a source by name in ``[news] enabled_sources``. Nothing is polled
until it is listed there.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from supere.news.models import FearGreedPoint, RawItem

FEAR_GREED_URL = "https://api.alternative.me/fng/"
DAY_MS = 86_400_000


class Collector(Protocol):
    name: str
    source_tier: int

    async def collect(self, client: httpx.AsyncClient) -> list[RawItem]:
        """Return new items. Implementations must not require a Binance key."""

    async def healthcheck(self, client: httpx.AsyncClient) -> bool:
        """True when the source answers. Failures are surfaced, not hidden."""


class FearGreedCollector:
    """Alternative.me daily index. ``known_at`` is the index day plus one day."""

    name = "alternative_me"
    source_tier = 2

    async def fetch_points(self, client: httpx.AsyncClient, limit: int = 1) -> list[FearGreedPoint]:
        response = await client.get(FEAR_GREED_URL, params={"limit": limit})
        response.raise_for_status()
        payload = response.json()
        points: list[FearGreedPoint] = []
        for item in payload.get("data") or []:
            ts_ms = int(item["timestamp"]) * 1000
            points.append(
                FearGreedPoint(
                    ts_ms=ts_ms,
                    value=int(item["value"]),
                    classification=str(item.get("value_classification") or ""),
                    known_at_ms=ts_ms + DAY_MS,
                )
            )
        return points

    async def collect(self, client: httpx.AsyncClient) -> list[RawItem]:
        points = await self.fetch_points(client, limit=1)
        items: list[RawItem] = []
        for point in points:
            items.append(
                RawItem(
                    source=self.name,
                    source_tier=self.source_tier,
                    lang="en",
                    title=f"Fear and Greed {point.value} {point.classification}".strip(),
                    body="",
                    url=FEAR_GREED_URL,
                    ts_published_ms=point.ts_ms,
                    external_id=str(point.ts_ms),
                )
            )
        return items

    async def healthcheck(self, client: httpx.AsyncClient) -> bool:
        response = await client.get(FEAR_GREED_URL, params={"limit": 1})
        return response.status_code == 200


def default_registry() -> dict[str, Collector]:
    collector = FearGreedCollector()
    return {collector.name: collector}


def persist_news(db, item: RawItem, ts_ingested: int, ts_local: int) -> int | None:
    from supere.news.dedup import simhash64, url_hash

    identity = item.url or item.title
    if item.external_id:
        identity = f"{identity}#{item.external_id}"
    digest = url_hash(identity) if identity else None
    if digest and db.news_by_url_hash(digest) is not None:
        return None
    return db.insert_news(
        {
            "ts_published": item.ts_published_ms,
            "ts_ingested": ts_ingested,
            "ts_local": ts_local,
            "source": item.source,
            "source_tier": item.source_tier,
            "lang": item.lang,
            "title": item.title,
            "body": item.body,
            "url": item.url,
            "url_hash": digest,
            "text_simhash": str(simhash64(f"{item.title}\n{item.body}")),
            "status": "new",
        }
    )
