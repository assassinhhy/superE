"""data.binance.vision paths. The monthly candle folder for 1M is ``1mo``."""

from __future__ import annotations

from supere.intervals import archive_interval


def _join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(parts)


def kline_zip(
    vision: str,
    venue: str,
    symbol: str,
    interval: str,
    period: str,
    stamp: str,
    series: str = "klines",
) -> str:
    """``venue`` is ``um`` or ``spot``. ``series`` is klines, markPriceKlines, ...

    ``period`` is ``monthly`` or ``daily``. ``stamp`` is ``YYYY-MM`` or ``YYYY-MM-DD``.
    """
    folder = archive_interval(interval)
    name = f"{symbol}-{folder}-{stamp}.zip"
    if venue == "spot":
        return _join(vision, "spot", period, "klines", symbol, folder, name)
    return _join(vision, "futures", "um", period, series, symbol, folder, name)


def funding_zip(vision: str, symbol: str, stamp: str) -> str:
    name = f"{symbol}-fundingRate-{stamp}.zip"
    return _join(vision, "futures", "um", "monthly", "fundingRate", symbol, name)


def metrics_zip(vision: str, symbol: str, day: str) -> str:
    name = f"{symbol}-metrics-{day}.zip"
    return _join(vision, "futures", "um", "daily", "metrics", symbol, name)


def agg_trades_zip(vision: str, symbol: str, day: str) -> str:
    name = f"{symbol}-aggTrades-{day}.zip"
    return _join(vision, "futures", "um", "daily", "aggTrades", symbol, name)


def checksum_url(zip_url: str) -> str:
    return zip_url + ".CHECKSUM"
