"""Row types stored in SQLite. Prices stay decimal strings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Kline:
    market: str
    symbol: str
    interval: str
    open_time: int
    close_time: int
    o: str
    h: str
    l: str
    c: str
    volume: str
    quote_volume: str
    trades: int
    taker_buy_base: str
    taker_buy_quote: str
    source: str
    ingested_at: int
    ingested_local: int


@dataclass(slots=True)
class MetricRow:
    symbol: str
    ts: int
    oi: str | None = None
    oi_value: str | None = None
    top_pos_lsr: str | None = None
    top_acc_lsr: str | None = None
    global_lsr: str | None = None
    taker_lsr: str | None = None
    source: str = "rest"


@dataclass(slots=True)
class FundingRow:
    symbol: str
    funding_time: int
    rate: str
    mark_price: str | None = None
    source: str = "rest"
