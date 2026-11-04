"""Stream names for the two futures sockets and the spot reference socket."""

from __future__ import annotations

from supere.intervals import UM_INTERVALS


def market_streams(symbol: str) -> list[str]:
    base = symbol.lower()
    names = [f"{base}@kline_{interval}" for interval in UM_INTERVALS]
    names.extend(
        [
            f"{base}@aggTrade",
            f"{base}@markPrice@1s",
            f"{base}@forceOrder",
            f"{base}@miniTicker",
        ]
    )
    return names


def public_streams(symbol: str) -> list[str]:
    return [f"{symbol.lower()}@depth20@100ms"]


def spot_streams(symbol: str) -> list[str]:
    return [f"{symbol.lower()}@kline_1m"]
