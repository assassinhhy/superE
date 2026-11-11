"""Initial history: vision archives, then a REST tail.

``1w`` and ``1M`` are REST-only. The monthly ``1mo`` archive stops in 2023-06,
and both series are small enough for a couple of requests. Weekly zips do
still exist; they are skipped on purpose so there is one code path.

Full aggTrade history is not downloaded here. That archive is about a
gigabyte a month of zip files. Live capture plus gap fill covers it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from supere.binance.archive import zip_text
from supere.binance.parse import funding_from_rest, parse_funding_csv, parse_kline_csv, parse_metrics_csv
from supere.binance.urls import funding_zip, kline_zip, metrics_zip
from supere.ingest.gapfill import (
    _series,
    default_start,
    fill_kline_range,
    recompute_coverage,
    rest_path,
)
from supere.intervals import (
    REST_ONLY_INTERVALS,
    UM_INTERVALS,
    align_open,
    last_closed_open,
    next_open,
    previous_open,
)

log = logging.getLogger("supere.backfill")


def span_bounds(interval: str, start_ms: int, end_ms_exclusive: int) -> tuple[int, int] | None:
    """First and last candle open times that fall inside ``[start, end)``."""
    start_open = align_open(interval, start_ms)
    if start_open < start_ms:
        start_open = next_open(interval, start_open)
    last_open = align_open(interval, end_ms_exclusive - 1)
    if last_open < start_open:
        return None
    return start_open, last_open


def span_is_complete(db, market: str, symbol: str, interval: str, start_open: int, end_open: int) -> bool:
    """True when every expected open in the span is already stored.

    A rerun can then skip that archive. UPSERT still makes a partial span safe
    to import again.
    """
    if end_open < start_open:
        return False
    expected = 0
    cursor = start_open
    while cursor <= end_open:
        expected += 1
        nxt = next_open(interval, cursor)
        if nxt <= cursor:
            break
        cursor = nxt
        if expected > 200_000:
            break
    if expected == 0:
        return False
    return db.count_klines_between(market, symbol, interval, start_open, end_open) >= expected


def month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    start = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
    if month == 12:
        end = int(datetime(year + 1, 1, 1, tzinfo=UTC).timestamp() * 1000)
    else:
        end = int(datetime(year, month + 1, 1, tzinfo=UTC).timestamp() * 1000)
    return start, end


def iter_months(start, end):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


class Backfill:
    def __init__(self, cfg, rest, db, stamp, on_event, *, months=None, days=None) -> None:
        self.cfg = cfg
        self.rest = rest
        self.db = db
        self.stamp = stamp
        self.on_event = on_event
        self.months = months
        self.days = days

    async def run(self) -> None:
        await self.on_event("info", "backfill", "BACKFILL_START", "开始历史回填", {})
        symbol = self.cfg.symbols.um
        for market, sym, intervals in (
            ("um", symbol, UM_INTERVALS),
            ("um_mark", symbol, UM_INTERVALS),
            ("um_index", symbol, UM_INTERVALS),
            ("um_premium", symbol, UM_INTERVALS),
            ("spot", self.cfg.symbols.spot, ("1m",)),
        ):
            for interval in intervals:
                await self.backfill_klines(market, sym, interval)
        await self.backfill_funding(symbol)
        await self.backfill_metrics(symbol)
        server_ms, _local = self.stamp()
        for market, sym, intervals in (
            ("um", symbol, UM_INTERVALS),
            ("spot", self.cfg.symbols.spot, ("1m",)),
        ):
            for interval in intervals:
                recompute_coverage(self.db, market, sym, interval, server_ms)
        await self.on_event("info", "backfill", "BACKFILL_DONE", "历史回填结束", {})

    async def backfill_klines(self, market: str, symbol: str, interval: str) -> None:
        server_ms, _local = self.stamp()
        start = align_open(interval, default_start(market))
        last = last_closed_open(interval, server_ms)
        if last is None or last < start:
            return
        if interval in REST_ONLY_INTERVALS:
            await fill_kline_range(self.rest, self.db, self.stamp, market, symbol, interval, start, last)
            return
        covered = await self._monthly(market, symbol, interval)
        await self._daily(market, symbol, interval, covered)
        bounds = self.db.min_max_open(market, symbol, interval)
        if bounds is None:
            await self.on_event(
                "warning",
                "backfill",
                "ARCHIVE_MISSING",
                f"{market} {interval} 没有导入到存档，只补最近一页，避免用 REST 拉全历史。",
                {},
            )
            from supere.ingest.gapfill import fill_recent_tail

            await fill_recent_tail(
                self.rest, self.db, self.stamp, market, symbol, interval, self.cfg.ingest.startup_tail_limit
            )
            return
        tail_from = next_open(interval, bounds[1])
        if tail_from <= last:
            await fill_kline_range(
                self.rest, self.db, self.stamp, market, symbol, interval, tail_from, last
            )
        # UM lists on 2019-11-27; 1m archives start at 2020-01. Fill that edge.
        if bounds[0] > start:
            early_end = previous_open(interval, bounds[0])
            if early_end >= start:
                await fill_kline_range(
                    self.rest, self.db, self.stamp, market, symbol, interval, start, early_end
                )

    async def _monthly(self, market: str, symbol: str, interval: str) -> set[tuple[int, int]]:
        covered: set[tuple[int, int]] = set()
        today = datetime.now(UTC).date()
        first = datetime(2020, 1, 1, tzinfo=UTC).date() if market != "spot" else datetime(2017, 8, 1, tzinfo=UTC).date()
        months = self.months if self.months is not None else list(iter_months(first, today))
        venue = "spot" if market == "spot" else "um"
        series = _series(market)
        for year, month in months:
            stamp = f"{year:04d}-{month:02d}"
            start_ms, end_ms = month_bounds_ms(year, month)
            bounds = span_bounds(interval, start_ms, end_ms)
            if bounds is not None and span_is_complete(
                self.db, market, symbol.upper(), interval, bounds[0], bounds[1]
            ):
                covered.add((year, month))
                log.info("已有 %s %s %s，跳过月度存档", market, interval, stamp)
                continue
            url = kline_zip(self.cfg.endpoints.vision, venue, symbol.upper(), interval, "monthly", stamp, series)
            dest = rest_path(self.cfg, "klines", url.rsplit("/", 1)[-1])
            try:
                ok = await self.rest.download_verified(url, dest)
            except Exception as exc:
                log.warning("月度存档失败 %s %s %s: %s", market, interval, stamp, exc)
                continue
            if not ok:
                continue
            self._import_kline_file(dest, market, symbol, interval)
            covered.add((year, month))
            log.info("已导入 %s %s %s", market, interval, stamp)
        return covered

    async def _daily(self, market: str, symbol: str, interval: str, covered: set[tuple[int, int]]) -> None:
        today = datetime.now(UTC).date()
        yesterday = today - timedelta(days=1)
        first = datetime(2020, 1, 1, tzinfo=UTC).date() if market != "spot" else datetime(2017, 8, 17, tzinfo=UTC).date()
        if self.days is not None:
            days = self.days
        else:
            days = []
            day = first
            while day <= yesterday:
                if (day.year, day.month) not in covered:
                    days.append(day)
                day += timedelta(days=1)
        venue = "spot" if market == "spot" else "um"
        series = _series(market)
        for day in days:
            stamp = day.isoformat()
            start_ms = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
            bounds = span_bounds(interval, start_ms, start_ms + 86_400_000)
            if bounds is not None and span_is_complete(
                self.db, market, symbol.upper(), interval, bounds[0], bounds[1]
            ):
                log.info("已有 %s %s %s，跳过日度存档", market, interval, stamp)
                continue
            url = kline_zip(self.cfg.endpoints.vision, venue, symbol.upper(), interval, "daily", stamp, series)
            dest = rest_path(self.cfg, "klines", url.rsplit("/", 1)[-1])
            try:
                ok = await self.rest.download_verified(url, dest)
            except Exception as exc:
                log.warning("日度存档失败 %s %s %s: %s", market, interval, stamp, exc)
                continue
            if not ok:
                continue
            self._import_kline_file(dest, market, symbol, interval)

    def _import_kline_file(self, dest, market: str, symbol: str, interval: str) -> None:
        server_ms, local_ms = self.stamp()
        rows = parse_kline_csv(
            zip_text(dest.read_bytes()), market, symbol, interval, "archive", server_ms, local_ms
        )
        self.db.upsert_klines(rows)

    async def backfill_funding(self, symbol: str) -> None:
        today = datetime.now(UTC).date()
        months = self.months if self.months is not None else list(iter_months(datetime(2020, 1, 1).date(), today))
        for year, month in months:
            stamp = f"{year:04d}-{month:02d}"
            url = funding_zip(self.cfg.endpoints.vision, symbol.upper(), stamp)
            dest = rest_path(self.cfg, "funding", url.rsplit("/", 1)[-1])
            try:
                ok = await self.rest.download_verified(url, dest)
            except Exception:
                log.warning("资金费率存档失败 %s", stamp, exc_info=True)
                continue
            if not ok:
                continue
            rows = parse_funding_csv(zip_text(dest.read_bytes()), symbol, "archive")
            self.db.upsert_funding(rows)
        server_ms, _local = self.stamp()
        start = int(datetime(2019, 11, 27, tzinfo=UTC).timestamp() * 1000)
        cursor = start
        while cursor < server_ms:
            raw = await self.rest.funding_rate(symbol, start=cursor, end=server_ms, limit=1000)
            if not raw:
                break
            self.db.upsert_funding([funding_from_rest(item) for item in raw])
            last = int(raw[-1]["fundingTime"])
            nxt = last + 1
            if nxt <= cursor or len(raw) < 1000:
                break
            cursor = nxt

    async def backfill_metrics(self, symbol: str) -> None:
        today = datetime.now(UTC).date()
        first = datetime(2021, 12, 1, tzinfo=UTC).date()
        if self.days is not None:
            days = [day for day in self.days if day >= first]
        else:
            days = []
            day = first
            yesterday = today - timedelta(days=1)
            while day <= yesterday:
                days.append(day)
                day += timedelta(days=1)
        for day in days:
            stamp = day.isoformat()
            start_ms = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
            # A full UTC day of 5-minute metrics is 288 rows. Already-imported
            # days are left in place; a day that crashed before commit is retried.
            if self.db.count_metrics_between(symbol.upper(), start_ms, start_ms + 86_400_000) >= 288:
                log.info("已有 metrics %s，跳过", stamp)
                continue
            url = metrics_zip(self.cfg.endpoints.vision, symbol.upper(), stamp)
            dest = rest_path(self.cfg, "metrics", url.rsplit("/", 1)[-1])
            try:
                ok = await self.rest.download_verified(url, dest)
            except Exception:
                log.warning("metrics 存档失败 %s", stamp, exc_info=True)
                continue
            if not ok:
                continue
            try:
                rows = parse_metrics_csv(zip_text(dest.read_bytes()), "archive")
                self.db.upsert_metrics(rows)
            except Exception:
                log.warning("metrics 导入失败 %s", stamp, exc_info=True)
                continue
