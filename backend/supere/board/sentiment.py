"""Open-interest quadrants and the crowding / funding / volume notes."""

from __future__ import annotations


def price_side(dp_pct: float, threshold: float) -> str:
    if dp_pct <= -threshold:
        return "down"
    if dp_pct >= threshold:
        return "up"
    return "flat"


def oi_side(doi_pct: float, threshold: float) -> str:
    if doi_pct <= -threshold:
        return "down"
    if doi_pct >= threshold:
        return "up"
    return "flat"


def quadrant(dp_pct: float, doi_pct: float, dp_threshold: float, doi_threshold: float) -> str:
    price = price_side(dp_pct, dp_threshold)
    oi = oi_side(doi_pct, doi_threshold)
    table = {
        ("down", "down"): "多单在爆仓/平仓",
        ("down", "up"): "新空单进场，空头主动",
        ("up", "down"): "空头回补",
        ("up", "up"): "新多进场，上涨较健康",
        ("flat", "up"): "横盘堆仓，后续突破波动会放大",
        ("flat", "down"): "横盘去杠杆，中性",
    }
    if (price, oi) in table:
        return table[(price, oi)]
    if price != "flat" and oi == "flat":
        return "仓位无明显变化"
    return "仓位无明显变化"


def crowding(global_lsr: float | None, long_at: float, short_at: float) -> str | None:
    if global_lsr is None:
        return None
    if global_lsr >= long_at:
        return "散户多头拥挤，向下破位易连环爆多"
    if global_lsr <= short_at:
        return "散户空头拥挤，向上挤压易连环爆空"
    return None


def funding_heat(rate: float | None, hot: float, neutral: float) -> str | None:
    if rate is None:
        return None
    if abs(rate) >= hot:
        return "费率偏热"
    if abs(rate) <= neutral:
        return "中性"
    return None


def taker_pressure(ratio: float | None, sell_at: float, buy_at: float) -> str | None:
    if ratio is None:
        return None
    if ratio < sell_at:
        return "主动卖压"
    if ratio > buy_at:
        return "主动买盘"
    return None


def volume_spike(volumes: list[float], mult: float, window: int) -> bool:
    if len(volumes) < window + 1:
        return False
    base = volumes[-(window + 1):-1]
    if not base:
        return False
    avg = sum(base) / len(base)
    if avg <= 0:
        return False
    return volumes[-1] >= avg * mult


def aligned_oi(rows: list[dict], at_ms: int) -> dict | None:
    """Latest 5m metric whose ``known_at`` is already knowable at ``at_ms``."""
    chosen = None
    for row in rows:
        if row.get("oi") is None:
            continue
        if int(row["known_at"]) <= int(at_ms):
            chosen = row
    return chosen
