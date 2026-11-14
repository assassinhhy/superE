"""15m key levels. 5m pins are included so a wick diluted by the 15m bar is kept."""

from __future__ import annotations

import math
from datetime import datetime

from supere.board.indicators import last_value
from supere.timeutil import UTC8

_RANK = {
    "插针扫过": 0,
    "前日高": 1,
    "前日低": 1,
    "摆动高": 2,
    "摆动低": 2,
    "当日高": 3,
    "当日低": 3,
    "均线": 4,
    "整数": 5,
}


def day_start_ms(moment_ms: int) -> int:
    dt = datetime.fromtimestamp(moment_ms / 1000, tz=UTC8).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


def _wick(bar: dict, *, body_mult: float, atr: float, atr_mult: float) -> list[dict]:
    body = abs(bar["close"] - bar["open"])
    lower = min(bar["open"], bar["close"]) - bar["low"]
    upper = bar["high"] - max(bar["open"], bar["close"])
    found = []
    if lower >= body * body_mult and lower >= atr * atr_mult:
        found.append({"price": bar["low"], "kind": "插针扫过", "dynamic": False})
    if upper >= body * body_mult and upper >= atr * atr_mult:
        found.append({"price": bar["high"], "kind": "插针扫过", "dynamic": False})
    return found


def _swings(bars: list[dict], wing: int, lookback: int) -> list[dict]:
    if wing < 1 or len(bars) < wing * 2 + 1:
        return []
    start = max(wing, len(bars) - lookback)
    found = []
    for i in range(start, len(bars) - wing):
        high = bars[i]["high"]
        low = bars[i]["low"]
        left = bars[i - wing:i]
        right = bars[i + 1:i + 1 + wing]
        if all(high > bar["high"] for bar in left) and all(high >= bar["high"] for bar in right):
            found.append({"price": high, "kind": "摆动高", "dynamic": False})
        if all(low < bar["low"] for bar in left) and all(low <= bar["low"] for bar in right):
            found.append({"price": low, "kind": "摆动低", "dynamic": False})
    return found


def _touches(bars: list[dict], price: float, band: float) -> int:
    runs = 0
    inside = False
    for bar in bars:
        hit = bar["low"] <= price + band and bar["high"] >= price - band
        if hit and not inside:
            runs += 1
        inside = hit
    return runs


def _union_touches(bars: list[dict], prices: list[float], band: float) -> int:
    runs = 0
    inside = False
    for bar in bars:
        hit = any(bar["low"] <= price + band and bar["high"] >= price - band for price in prices)
        if hit and not inside:
            runs += 1
        inside = hit
    return runs


def _merge(raw: list[dict], atr: float, merge_atr: float, bars: list[dict], touch_atr: float) -> list[dict]:
    if not raw:
        return []
    ordered = sorted(raw, key=lambda item: item["price"])
    gap = atr * merge_atr
    clusters: list[list[dict]] = [[ordered[0]]]
    for item in ordered[1:]:
        if item["price"] - clusters[-1][-1]["price"] < gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    band = atr * touch_atr
    merged = []
    for cluster in clusters:
        best = min(cluster, key=lambda item: (_RANK.get(item["kind"], 9), item["price"]))
        kinds = []
        for item in sorted(cluster, key=lambda row: _RANK.get(row["kind"], 9)):
            label = item.get("label") or item["kind"]
            if label not in kinds:
                kinds.append(label)
        merged.append(
            {
                "price": best["price"],
                "kind": best["kind"],
                "label": best.get("label") or best["kind"],
                "sources": kinds,
                "dynamic": any(item.get("dynamic") for item in cluster),
                "touches": _union_touches(bars, [item["price"] for item in cluster], band),
            }
        )
    return merged


def build_levels(
    *,
    bars_15: list[dict],
    bars_5: list[dict],
    packs: dict,
    price: float | None,
    now_ms: int,
    atr: float | None,
    params,
) -> dict:
    empty = {"resistance": [], "support": [], "all": []}
    if not bars_15 or atr is None or atr <= 0 or price is None:
        return empty
    raw: list[dict] = []
    start = day_start_ms(now_ms)
    today = [bar for bar in bars_15 if start <= bar["open_time"] < start + 86_400_000]
    prev = [bar for bar in bars_15 if start - 86_400_000 <= bar["open_time"] < start]
    if today:
        raw.append({"price": max(bar["high"] for bar in today), "kind": "当日高", "dynamic": False})
        raw.append({"price": min(bar["low"] for bar in today), "kind": "当日低", "dynamic": False})
    if prev:
        raw.append({"price": max(bar["high"] for bar in prev), "kind": "前日高", "dynamic": False})
        raw.append({"price": min(bar["low"] for bar in prev), "kind": "前日低", "dynamic": False})
    raw.extend(_swings(bars_15, params.fractal_wing, params.swing_lookback))
    window = bars_15[-params.touch_window_bars:]
    for bar in window:
        raw.extend(_wick(bar, body_mult=params.wick_body_mult, atr=atr, atr_mult=params.wick_atr_mult))
    if params.include_5m_wicks:
        for bar in bars_5[-params.wick_5m_lookback:]:
            raw.extend(_wick(bar, body_mult=params.wick_body_mult, atr=atr, atr_mult=params.wick_atr_mult))
    low = min(bar["low"] for bar in window)
    high = max(bar["high"] for bar in window)
    step = params.round_step
    cursor = math.floor((low - atr) / step) * step
    limit = high + atr
    while cursor <= limit:
        raw.append({"price": float(cursor), "kind": "整数", "dynamic": False})
        cursor += step
    for spec in params.ma_levels:
        frame, name = spec.split(":")
        period = int(name.replace("EMA", ""))
        pack = packs.get(frame)
        if not pack:
            continue
        value = last_value(pack["ema"].get(period, []))
        if value is None:
            continue
        raw.append({"price": value, "kind": "均线", "label": f"{frame} {name}", "dynamic": True})
    touch_bars = bars_15[-params.touch_window_bars:]
    merged = _merge(raw, atr, params.merge_atr, touch_bars, params.touch_atr)
    above = sorted((item for item in merged if item["price"] > price), key=lambda item: item["price"])
    below = sorted((item for item in merged if item["price"] < price), key=lambda item: item["price"], reverse=True)
    return {
        "resistance": above[: params.nearest_each_side],
        "support": below[: params.nearest_each_side],
        "all": merged,
    }
