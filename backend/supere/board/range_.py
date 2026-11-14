"""Recent 15m range, volume shrink, and swing structure."""

from __future__ import annotations


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def swings(bars: list[dict], wing: int = 3) -> tuple[list[float], list[float]]:
    highs: list[float] = []
    lows: list[float] = []
    if len(bars) < wing * 2 + 1:
        return highs, lows
    for i in range(wing, len(bars) - wing):
        high = bars[i]["high"]
        low = bars[i]["low"]
        left = bars[i - wing:i]
        right = bars[i + 1:i + 1 + wing]
        if all(high > bar["high"] for bar in left) and all(high >= bar["high"] for bar in right):
            highs.append(high)
        if all(low < bar["low"] for bar in left) and all(low <= bar["low"] for bar in right):
            lows.append(low)
    return highs, lows


def build_range(bars: list[dict], price: float | None, params) -> dict | None:
    if len(bars) < params.bars or price is None:
        return None
    window = bars[-params.bars:]
    high = max(bar["high"] for bar in window)
    low = min(bar["low"] for bar in window)
    width = high - low
    recent = bars[-params.shrink_recent:]
    base = bars[-(params.shrink_recent + params.shrink_base):-params.shrink_recent]
    recent_avg = _mean([bar["volume"] for bar in recent])
    base_avg = _mean([bar["volume"] for bar in base])
    shrink = False
    ratio = None
    if recent_avg is not None and base_avg and base_avg > 0:
        ratio = recent_avg / base_avg
        shrink = ratio < params.shrink_ratio
    if width <= 0:
        position = "middle"
        pos_pct = 50.0
    else:
        pos_pct = (price - low) / width * 100
        if pos_pct >= 100 - params.upper_pct:
            position = "upper"
        elif pos_pct <= params.lower_pct:
            position = "lower"
        else:
            position = "middle"
    labels = {"upper": "上25%", "middle": "中间50%", "lower": "下25%"}
    high_points, low_points = swings(bars[-96:], 3)
    structure = []
    if len(high_points) >= params.structure_swings:
        last = high_points[-params.structure_swings:]
        if all(last[i] > last[i + 1] for i in range(len(last) - 1)):
            structure.append("高点下移 " + "→".join(f"{value:.0f}" for value in last))
    if len(low_points) >= params.structure_swings:
        last = low_points[-params.structure_swings:]
        if all(last[i] < last[i + 1] for i in range(len(last) - 1)):
            structure.append("低点上移 " + "→".join(f"{value:.0f}" for value in last))
    shrink_text = ""
    if shrink and ratio is not None:
        shrink_text = f"缩量横盘：近{params.shrink_recent}根均量 = 前{params.shrink_base}根 × {ratio:.2f}"
    return {
        "high": high,
        "low": low,
        "width": width,
        "shrink": shrink,
        "shrink_ratio": ratio,
        "shrink_text": shrink_text,
        "position": position,
        "position_pct": pos_pct,
        "position_text": labels[position],
        "blocked": position == "middle",
        "structure": structure,
    }
