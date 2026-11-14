"""Entry confirmation on a closed 5m or 15m bar. No volume requirement for engulfing."""

from __future__ import annotations


def _body(bar: dict) -> tuple[float, float]:
    return min(bar["open"], bar["close"]), max(bar["open"], bar["close"])


def wick_confirms(bar: dict, *, side: str, atr: float, body_mult: float, atr_mult: float, zone: tuple[float, float]) -> bool:
    low_body, high_body = _body(bar)
    body = high_body - low_body
    upper = bar["high"] - high_body
    lower = low_body - bar["low"]
    inside = zone[0] <= bar["close"] <= zone[1]
    if not inside or atr <= 0:
        return False
    if side == "short":
        return upper >= body * body_mult and upper >= atr * atr_mult
    return lower >= body * body_mult and lower >= atr * atr_mult


def engulf_confirms(prev: dict, bar: dict, *, side: str, zone: tuple[float, float]) -> bool:
    if not (zone[0] <= bar["close"] <= zone[1]):
        return False
    prev_low, prev_high = _body(prev)
    low, high = _body(bar)
    covers = low <= prev_low and high >= prev_high and high > low
    if not covers:
        return False
    if side == "short":
        return bar["close"] < bar["open"]
    return bar["close"] > bar["open"]


def rsi_turn(values: list[float | None], *, side: str, lookback: int, short_from: float, long_from: float) -> bool:
    usable = [value for value in values if value is not None]
    if len(usable) < 2:
        return False
    window = usable[-(lookback + 1):]
    latest, previous = window[-1], window[-2]
    if side == "short":
        return any(value > short_from for value in window[:-1]) and latest < previous
    return any(value < long_from for value in window[:-1]) and latest > previous


def confirm_bar(
    bars: list[dict],
    rsi: list[float | None],
    *,
    side: str,
    atr: float,
    zone: tuple[float, float],
    params,
) -> list[str]:
    if len(bars) < 2 or atr is None:
        return []
    names = []
    bar = bars[-1]
    if wick_confirms(
        bar,
        side=side,
        atr=atr,
        body_mult=params.wick_body_mult,
        atr_mult=params.wick_atr_mult,
        zone=zone,
    ):
        names.append("长影线")
    if engulf_confirms(bars[-2], bar, side=side, zone=zone):
        names.append("吞没")
    if rsi_turn(
        rsi,
        side=side,
        lookback=params.rsi_lookback_bars,
        short_from=params.rsi_short_from,
        long_from=params.rsi_long_from,
    ) and zone[0] <= bar["close"] <= zone[1]:
        names.append("RSI拐头")
    return names
