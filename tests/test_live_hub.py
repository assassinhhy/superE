"""LiveHub keeps exactly one forming bar per market/symbol/interval."""

from __future__ import annotations

from supere.ipc.bus import LiveHub


def _bar(open_time: int, *, symbol: str = "ETHUSDT", interval: str = "1m", market: str = "um") -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "open_time": open_time,
        "closed": False,
        "c": "100",
    }


def _kline(bar: dict, *, closed: bool = False) -> dict:
    return {"type": "kline", "closed": closed, "bar": bar}


def test_newer_open_time_replaces_stale_forming():
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000)))
    hub.apply(_kline(_bar(120_000)))

    assert [item["open_time"] for item in hub.forming] == [120_000]


def test_closed_event_drops_zombie_forming():
    # A missed ``closed`` event used to leave the previous bar in ``forming``
    # forever, freezing the chart on a stale candle.
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000)))
    hub.apply(_kline(_bar(120_000), closed=True))

    assert hub.forming == []


def test_closed_event_without_prior_forming_stays_empty():
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000), closed=True))

    assert hub.forming == []


def test_other_symbols_and_intervals_are_untouched():
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000)))
    hub.apply(_kline(_bar(60_000, symbol="BTCUSDT")))
    hub.apply(_kline(_bar(60_000, interval="5m")))

    keys = sorted((item["symbol"], item["interval"]) for item in hub.forming)
    assert keys == [("BTCUSDT", "1m"), ("ETHUSDT", "1m"), ("ETHUSDT", "5m")]

    hub.apply(_kline(_bar(120_000)))
    keys = sorted((item["symbol"], item["interval"], item["open_time"]) for item in hub.forming)
    assert keys == [
        ("BTCUSDT", "1m", 60_000),
        ("ETHUSDT", "1m", 120_000),
        ("ETHUSDT", "5m", 60_000),
    ]


def test_other_market_is_untouched():
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000, market="um")))
    hub.apply(_kline(_bar(60_000, market="spot")))

    assert sorted(item["market"] for item in hub.forming) == ["spot", "um"]


def test_kline_without_bar_is_ignored():
    hub = LiveHub()
    hub.apply({"type": "kline", "closed": False})

    assert hub.forming == []


def test_non_kline_messages_do_not_touch_forming():
    hub = LiveHub()
    hub.apply(_kline(_bar(60_000)))
    hub.apply({"type": "miniTicker", "market": "um", "symbol": "ETHUSDT"})

    assert [item["open_time"] for item in hub.forming] == [60_000]
