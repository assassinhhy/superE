"""Market-move detector. It explains price; it does not emit a trade signal.

"过去 24 小时同一时段的 P99" is read as the 99th percentile of 1m quote
volume over the previous 24 hours. A same-minute-of-day reading inside a
single 24h window would be one sample, which cannot be a percentile.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from supere.config import AnomalyConfig
from supere.storage.db import Database
from supere.storage.models import Kline


@dataclass(slots=True)
class AnomalyHit:
    ts: int
    kind: str
    value: float
    threshold: float
    detail: dict


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]


def median(values: list[float]) -> float | None:
    return percentile(values, 50)


def wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        true_ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def detect_volume(
    quote: float,
    previous_60: list[float],
    previous_24h: list[float],
    mult: float,
) -> AnomalyHit | None:
    if len(previous_60) < 30 or len(previous_24h) < 60:
        return None
    med = median(previous_60)
    p99 = percentile(previous_24h, 99)
    if med is None or p99 is None or med <= 0:
        return None
    if quote > med * mult and quote >= p99:
        return AnomalyHit(
            ts=0,
            kind="volume",
            value=quote,
            threshold=max(med * mult, p99),
            detail={"median_60": med, "p99_24h": p99, "mult": mult},
        )
    return None


def detect_price_jump(
    close: float,
    prev_close: float,
    atr: float | None,
    mult: float,
) -> AnomalyHit | None:
    if atr is None or atr <= 0 or prev_close <= 0:
        return None
    move = abs(close - prev_close)
    threshold = mult * atr
    if move > threshold:
        return AnomalyHit(
            ts=0,
            kind="price_jump",
            value=move,
            threshold=threshold,
            detail={"prev_close": prev_close, "close": close, "atr": atr, "mult": mult},
        )
    return None


class AnomalyEngine:
    def __init__(self, cfg: AnomalyConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db
        self._liqs: deque[tuple[int, Decimal]] = deque()
        self._ticks: deque[tuple[int, Decimal]] = deque()
        self._last_rate: Decimal | None = None
        self._last_oi: Decimal | None = None
        self._last_10s_alert = 0

    def _save(self, hit: AnomalyHit) -> None:
        linked = self.db.news_ids_between(hit.ts - 15 * 60_000, hit.ts + 15 * 60_000)
        self.db.insert_anomaly(hit.ts, hit.kind, hit.value, hit.threshold, hit.detail, linked)

    def on_closed_1m(self, bar: Kline) -> list[AnomalyHit]:
        history = self.db.fetch_klines(
            bar.market, bar.symbol, "1m", end=bar.open_time - 1, limit=1440
        )
        quotes = [float(row["quote_volume"]) for row in history]
        hits: list[AnomalyHit] = []
        volume_hit = detect_volume(
            float(bar.quote_volume),
            quotes[-60:],
            quotes,
            self.cfg.volume_median_mult,
        )
        if volume_hit:
            volume_hit.ts = bar.open_time
            hits.append(volume_hit)
        closes = [float(row["c"]) for row in history]
        highs = [float(row["h"]) for row in history]
        lows = [float(row["l"]) for row in history]
        atr = wilder_atr(highs, lows, closes, self.cfg.atr_period)
        prev = closes[-1] if closes else None
        if prev is not None:
            jump = detect_price_jump(float(bar.c), prev, atr, self.cfg.price_atr_mult)
            if jump:
                jump.ts = bar.open_time
                hits.append(jump)
        for hit in hits:
            self._save(hit)
        return hits

    def on_agg_price(self, ts: int, price: str) -> AnomalyHit | None:
        px = Decimal(price)
        self._ticks.append((ts, px))
        cutoff = ts - 10_000
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()
        if len(self._ticks) < 2:
            return None
        old_ts, old_px = self._ticks[0]
        if old_px <= 0:
            return None
        move = abs((px - old_px) / old_px) * Decimal(100)
        if move > Decimal(str(self.cfg.price_10s_pct)) and ts - self._last_10s_alert > 10_000:
            self._last_10s_alert = ts
            hit = AnomalyHit(
                ts=ts,
                kind="price_jump_10s",
                value=float(move),
                threshold=self.cfg.price_10s_pct,
                detail={"from": str(old_px), "to": str(px), "from_ts": old_ts},
            )
            self._save(hit)
            return hit
        return None

    def on_liquidation(self, ts: int, price: str, qty: str) -> list[AnomalyHit]:
        notional = Decimal(price) * Decimal(qty)
        hits: list[AnomalyHit] = []
        if notional >= Decimal(str(self.cfg.liq_single_usd)):
            hits.append(
                AnomalyHit(
                    ts=ts,
                    kind="liquidation",
                    value=float(notional),
                    threshold=self.cfg.liq_single_usd,
                    detail={"price": price, "qty": qty, "scope": "single"},
                )
            )
        self._liqs.append((ts, notional))
        cutoff = ts - 60_000
        while self._liqs and self._liqs[0][0] < cutoff:
            self._liqs.popleft()
        total = sum((item[1] for item in self._liqs), Decimal(0))
        if total >= Decimal(str(self.cfg.liq_minute_usd)):
            hits.append(
                AnomalyHit(
                    ts=ts,
                    kind="liquidation_minute",
                    value=float(total),
                    threshold=self.cfg.liq_minute_usd,
                    detail={"scope": "1m"},
                )
            )
        for hit in hits:
            self._save(hit)
        return hits

    def on_oi(self, ts: int, oi: str) -> AnomalyHit | None:
        current = Decimal(oi)
        previous = self._last_oi
        self._last_oi = current
        if previous is None or previous == 0:
            return None
        change = abs((current - previous) / previous) * Decimal(100)
        if change > Decimal(str(self.cfg.oi_change_pct)):
            hit = AnomalyHit(
                ts=ts,
                kind="open_interest",
                value=float(change),
                threshold=self.cfg.oi_change_pct,
                detail={"previous": str(previous), "oi": oi},
            )
            self._save(hit)
            return hit
        return None

    def on_funding(self, ts: int, rate: str) -> AnomalyHit | None:
        current = Decimal(rate)
        previous = self._last_rate
        self._last_rate = current
        if previous is None:
            return None
        jump = abs(current - previous)
        if jump > Decimal(str(self.cfg.funding_jump_abs)):
            hit = AnomalyHit(
                ts=ts,
                kind="funding",
                value=float(jump),
                threshold=self.cfg.funding_jump_abs,
                detail={"previous": str(previous), "rate": rate},
            )
            self._save(hit)
            return hit
        return None
