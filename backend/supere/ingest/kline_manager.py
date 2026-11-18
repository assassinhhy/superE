"""Closed candles are written. The forming candle stays in memory."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from supere.binance.parse import dec_str, kline_from_list
from supere.storage.db import Database
from supere.storage.models import Kline


@dataclass(slots=True)
class _Forming:
    bar: Kline
    kline_event_ms: int
    last_agg_id: int
    asof: int


class KlineManager:
    def __init__(self, db: Database, stamp) -> None:
        self.db = db
        self.stamp = stamp  # () -> (server_ms, local_ms)
        self.forming: dict[tuple[str, str, str], _Forming] = {}

    def snapshot(self, market: str, symbol: str, interval: str) -> dict | None:
        item = self.forming.get((market, symbol, interval))
        if item is None:
            return None
        bar = item.bar
        return {
            "market": bar.market,
            "symbol": bar.symbol,
            "interval": bar.interval,
            "open_time": bar.open_time,
            "close_time": bar.close_time,
            "o": bar.o,
            "h": bar.h,
            "l": bar.l,
            "c": bar.c,
            "volume": bar.volume,
            "quote_volume": bar.quote_volume,
            "trades": bar.trades,
            "taker_buy_base": bar.taker_buy_base,
            "taker_buy_quote": bar.taker_buy_quote,
            "closed": False,
            "asof": item.asof,
        }

    def snapshots(self) -> list[dict]:
        return [
            self.snapshot(market, symbol, interval)
            for (market, symbol, interval) in list(self.forming)
            if self.snapshot(market, symbol, interval) is not None
        ]

    async def on_kline(self, market: str, parsed: dict) -> Kline | None:
        """Apply a kline event. Returns the bar when it has just closed."""
        server_ms, local_ms = self.stamp()
        bar = kline_from_list(
            market,
            parsed["symbol"],
            parsed["interval"],
            parsed["row"],
            "ws",
            server_ms,
            local_ms,
        )
        key = (market, bar.symbol, bar.interval)
        event_ms = int(parsed.get("event_ms") or bar.open_time)
        if parsed.get("closed"):
            self.forming.pop(key, None)
            self.db.upsert_klines([bar])
            return bar
        current = self.forming.get(key)
        if current is not None and event_ms < current.asof:
            return None
        # Updating the same forming bar keeps the aggTrade watermark: zeroing it
        # would re-apply trades already folded into this bar. Only a new bar starts
        # from 0.
        last_agg_id = 0
        if current is not None and current.bar.open_time == bar.open_time:
            last_agg_id = current.last_agg_id
        self.forming[key] = _Forming(bar=bar, kline_event_ms=event_ms,
                                     last_agg_id=last_agg_id, asof=event_ms)
        return None

    async def on_agg_trade(self, market: str, parsed: dict) -> None:
        trade_ts = int(parsed["ts"])
        price = Decimal(parsed["price"])
        qty = Decimal(parsed["qty"])
        agg_id = int(parsed["agg_id"])
        symbol = parsed["symbol"]
        for key, forming in list(self.forming.items()):
            bar_market, bar_symbol, _interval = key
            if bar_market != market or bar_symbol != symbol:
                continue
            bar = forming.bar
            if trade_ts < bar.open_time or trade_ts > bar.close_time:
                continue
            if trade_ts < forming.kline_event_ms or agg_id <= forming.last_agg_id:
                continue
            quote = price * qty
            taker = not parsed.get("is_buyer_maker")
            bar.c = dec_str(price)
            bar.h = dec_str(max(Decimal(bar.h), price))
            bar.l = dec_str(min(Decimal(bar.l), price))
            bar.volume = dec_str(Decimal(bar.volume) + qty)
            bar.quote_volume = dec_str(Decimal(bar.quote_volume) + quote)
            bar.trades += 1
            if taker:
                bar.taker_buy_base = dec_str(Decimal(bar.taker_buy_base) + qty)
                bar.taker_buy_quote = dec_str(Decimal(bar.taker_buy_quote) + quote)
            forming.last_agg_id = agg_id
            forming.asof = trade_ts
