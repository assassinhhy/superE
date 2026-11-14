"""EMA, Wilder RSI, MACD and Wilder ATR.

RSI and MACD use the same seed as ``frontend/src/signals.js``. ATR uses the
same seed as ``frontend/src/indicators.js``. EMA is the standard SMA seed,
k = 2/(n+1), which is also how that MACD seeds its fast and slow lines.
"""

from __future__ import annotations


def ema_series(closes: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if period < 1 or len(closes) < period:
        return out
    prev = sum(closes[:period]) / period
    out[period - 1] = prev
    k = 2 / (period + 1)
    for i in range(period, len(closes)):
        prev = closes[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi_series(closes: list[float], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    gain_sum = 0.0
    loss_sum = 0.0
    avg_gain = None
    avg_loss = None
    for i, close in enumerate(closes):
        change = 0.0 if i == 0 else close - closes[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        gain_sum += gain
        loss_sum += loss
        if i < period:
            continue
        if avg_gain is None:
            avg_gain = gain_sum / period
            avg_loss = loss_sum / period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[i] = 100.0
        elif avg_gain == 0:
            out[i] = 0.0
        else:
            out[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def macd_series(
    closes: list[float], short: int = 12, long: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    dif: list[float | None] = [None] * len(closes)
    dea: list[float | None] = [None] * len(closes)
    close_sum = 0.0
    ema_short = 0.0
    ema_long = 0.0
    dif_sum = 0.0
    dea_value = 0.0
    max_period = max(short, long)
    for i, close in enumerate(closes):
        close_sum += close
        if i >= short - 1:
            ema_short = (
                (2 * close + (short - 1) * ema_short) / (short + 1) if i > short - 1 else close_sum / short
            )
        if i >= long - 1:
            ema_long = (
                (2 * close + (long - 1) * ema_long) / (long + 1) if i > long - 1 else close_sum / long
            )
        if i >= max_period - 1:
            dif_value = ema_short - ema_long
            dif[i] = dif_value
            dif_sum += dif_value
            if i >= max_period + signal - 2:
                if i > max_period + signal - 2:
                    dea_value = (dif_value * 2 + dea_value * (signal - 1)) / (signal + 1)
                else:
                    dea_value = dif_sum / signal
                dea[i] = dea_value
    hist = [
        None if dif[i] is None or dea[i] is None else dif[i] - dea[i]
        for i in range(len(closes))
    ]
    return dif, dea, hist


def atr_series(bars: list[dict], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(bars)
    if period < 1:
        return out
    prev_close = None
    seed = 0.0
    value = None
    for i, bar in enumerate(bars):
        tr = bar["high"] - bar["low"]
        if prev_close is not None:
            tr = max(tr, abs(bar["high"] - prev_close), abs(bar["low"] - prev_close))
        prev_close = bar["close"]
        if i < period:
            seed += tr
        if i == period - 1:
            value = seed / period
            out[i] = value
        elif i >= period and value is not None:
            value = (value * (period - 1) + tr) / period
            out[i] = value
    return out


def last_value(series: list[float | None]) -> float | None:
    for value in reversed(series):
        if value is not None:
            return value
    return None


def pack_bars(bars: list[dict], *, ema_periods: list[int], rsi_period: int, macd: list[int], atr_period: int) -> dict:
    closes = [bar["close"] for bar in bars]
    dif, dea, hist = macd_series(closes, macd[0], macd[1], macd[2])
    return {
        "bars": bars,
        "closes": closes,
        "ema": {period: ema_series(closes, period) for period in ema_periods},
        "rsi": rsi_series(closes, rsi_period),
        "dif": dif,
        "dea": dea,
        "hist": hist,
        "atr": atr_series(bars, atr_period),
    }
