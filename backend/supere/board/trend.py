"""1h / 4h direction. Tangle is decided before bullish or bearish order."""

from __future__ import annotations

from supere.board.indicators import last_value


def classify_frame(close: float, ema20: float, ema50: float, tangle_pct: float) -> str:
    if close == 0:
        return "震荡"
    if abs(ema20 - ema50) / close * 100 < tangle_pct:
        return "震荡"
    if close < ema20 and ema20 < ema50:
        return "空"
    if close > ema20 and ema20 > ema50:
        return "多"
    return "震荡"


def synthesize(frame_4h: str, frame_1h: str) -> tuple[str, str]:
    """Return ``(bias, direction)``."""
    if frame_4h == "空" and frame_1h == "空":
        return "只做空", "空头"
    if frame_4h == "多" and frame_1h == "多":
        return "只做多", "多头"
    if frame_4h == "空" and frame_1h in {"震荡", "多"}:
        return "偏空", "空头"
    if frame_4h == "多" and frame_1h in {"震荡", "空"}:
        return "偏多", "多头"
    if frame_4h == "震荡":
        return "震荡，只做区间两头或不做", "震荡"
    return "震荡，只做区间两头或不做", "震荡"


def macd_fading(hist: list[float | None], *, bullish: bool, bars: int) -> bool:
    """Histogram shrinking against the trend for ``bars`` consecutive closes."""
    values = [value for value in hist if value is not None]
    need = bars + 1
    if len(values) < need:
        return False
    window = values[-need:]
    for older, newer in zip(window, window[1:]):
        if bullish:
            if newer <= 0 or abs(newer) >= abs(older):
                return False
        else:
            if newer >= 0 or abs(newer) >= abs(older):
                return False
    return True


def daily_rsi_tag(rsi: float | None, *, overbought: float, oversold: float) -> str | None:
    if rsi is None:
        return None
    if rsi < oversold:
        return "大级别超卖，注意反抽"
    if rsi > overbought:
        return "大级别超买，注意回调"
    return None


def frame_view(pack: dict | None, tangle_pct: float) -> str | None:
    if not pack:
        return None
    close = pack["closes"][-1]
    ema20 = last_value(pack["ema"].get(20, []))
    ema50 = last_value(pack["ema"].get(50, []))
    if ema20 is None or ema50 is None:
        return None
    return classify_frame(close, ema20, ema50, tangle_pct)
