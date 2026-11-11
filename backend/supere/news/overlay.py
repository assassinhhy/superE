"""Click-time news aggregation and the display-only overlay on a price forecast.

The overlay never rewrites the model's quantiles; it returns a separate adjusted
copy.  All formulas are deterministic and documented so a click can be recomputed:

* visible news only: ``available_at <= decision time``
* window 24h; per-horizon half-life = ``max(horizon, 1h)``
* weight ``w_i = 0.5 ** (age / half_life)``
* ``R = sum(w*r)/sum(w)``, ``U = sum(w*r*u)/sum(w*r)``
* ``D = sum(w*r*(0.4+0.6*u)*d) / sum(w*r*(0.4+0.6*u))``
* gate ``U*R < 0.15`` -> no change
* width ``scale = 1 + 0.35*U*R`` (never below 1)
* median ``delta = 0.25*D*R*(p75-p25)/2``, clipped to ``0.5*(p75-p25)/2``
* every quantile is shifted by ``delta`` and scaled around the raw median.

No position advice is produced here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from supere.timeutil import UTC8

GATE = 0.15
ALPHA = 0.35
BETA = 0.25
MAX_WIDTH_MULTIPLIER = 1.0 + ALPHA
WINDOW_HOURS = 24
MIN_HALF_LIFE_HOURS = 1.0
HIGH_URGENCY = 0.7
WATCH_URGENCY = 0.6

EVENT_WHY = {
    "regulation": "监管预期可能变化，影响 ETH 的风险溢价",
    "security": "交易所或链上安全信心可能受影响",
    "macro": "利率预期和风险偏好可能变化",
    "flows": "资金流向和买盘可能变化",
    "tech": "以太坊网络预期和市场情绪可能变化",
    "general": "市场情绪和资金流可能受到影响",
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _visible(item: dict) -> bool:
    return item.get("available_at") is not None


def sentiment_word(direction: float) -> str:
    d = _num(direction)
    if d > 0.12:
        return "利多"
    if d < -0.12:
        return "利空"
    return "中性"


def _event_why(event_type: str) -> str:
    return EVENT_WHY.get(str(event_type or "general"), EVENT_WHY["general"])


def _fmt_ms(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")


def _top_news(items: list[dict], limit: int = 3) -> list[dict]:
    return [
        {
            "id": item.get("id"),
            "content_hash": item.get("content_hash") or "",
            "title": item.get("title") or "",
            "source": item.get("source") or "unknown",
            "url": item.get("url") or "",
            "time_utc8": (item.get("ts_published_utc8") or item.get("available_at_utc8")
                          or _fmt_ms(item.get("ts_published")) or _fmt_ms(item.get("available_at"))),
            "sentiment_label": item.get("sentiment_label") or sentiment_word(item.get("direction", 0)),
            "summary_zh": item.get("summary_zh") or "",
            "direction": round(_num(item.get("direction")), 4),
            "urgency": round(_num(item.get("urgency")), 4),
            "relevance": round(_num(item.get("relevance")), 4),
            "contribution": round(_num(item.get("contribution")), 6),
            "event_type": item.get("event_type") or "general",
            "scorer": item.get("scored_by") or item.get("scorer") or "",
            "score_version": item.get("score_version") or "",
        }
        for item in items[:limit]
    ]


def aggregate(items: list[dict], now_ms: int, horizon_ms: int) -> dict:
    """Time-decayed U/D/R plus counts, top contributors and a Chinese alert."""
    half_life_ms = max(int(horizon_ms), int(MIN_HALF_LIFE_HOURS * 3_600_000))
    window_ms = WINDOW_HOURS * 3_600_000
    usable: list[dict] = []
    for item in items or []:
        available = item.get("available_at")
        if available is None or int(available) > int(now_ms):
            continue
        event_ms = item.get("ts_published")
        if event_ms is None or int(event_ms) > int(now_ms):
            event_ms = available
        age = int(now_ms) - int(event_ms)
        if age < 0 or age > window_ms:
            continue
        decay = 0.5 ** (age / half_life_ms)
        relevance = _clip(_num(item.get("relevance")), 0.0, 1.0)
        urgency = _clip(_num(item.get("urgency")), 0.0, 1.0)
        direction = _clip(_num(item.get("direction")), -1.0, 1.0)
        row = dict(item)
        row.update({
            "age_ms": age,
            "decay": decay,
            "relevance": relevance,
            "urgency": urgency,
            "direction": direction,
            "sentiment_label": item.get("sentiment_label") or sentiment_word(direction),
            "contribution": decay * relevance * (0.35 + 0.65 * urgency) * (0.25 + 0.75 * abs(direction)),
        })
        usable.append(row)

    total_w = sum(row["decay"] for row in usable)
    wr = sum(row["decay"] * row["relevance"] for row in usable)
    relevance = (wr / total_w) if total_w else 0.0
    urgency = (sum(row["decay"] * row["relevance"] * row["urgency"] for row in usable) / wr) if wr else 0.0
    dir_den = sum(row["decay"] * row["relevance"] * (0.4 + 0.6 * row["urgency"]) for row in usable)
    direction = (
        sum(row["decay"] * row["relevance"] * (0.4 + 0.6 * row["urgency"]) * row["direction"]
            for row in usable) / dir_den
    ) if dir_den else 0.0

    counts_1h = sum(1 for row in usable if row["age_ms"] <= 3_600_000 and row["urgency"] >= HIGH_URGENCY)
    counts_4h = sum(1 for row in usable if row["age_ms"] <= 4 * 3_600_000 and row["urgency"] >= HIGH_URGENCY)
    counts_24h = sum(1 for row in usable if row["urgency"] >= HIGH_URGENCY)
    positive = sum(1 for row in usable if row["direction"] > 0.12)
    negative = sum(1 for row in usable if row["direction"] < -0.12)
    neutral = len(usable) - positive - negative
    sources: dict[str, int] = {}
    for row in usable:
        sources[str(row.get("source") or "unknown")] = sources.get(str(row.get("source") or "unknown"), 0) + 1
    ranked = sorted(usable, key=lambda row: (-row["contribution"], -row["urgency"], -row["age_ms"]))
    top = _top_news(ranked)

    agg = {
        "window_hours": WINDOW_HOURS,
        "half_life_hours": round(half_life_ms / 3_600_000, 2),
        "decay_formula": "w_i = 0.5 ** (age_i / half_life)",
        "empty_defaults": {"u": 0.0, "d": 0.0, "r": 0.0, "gate_triggered": False},
        "u": round(_clip(urgency, 0, 1), 6),
        "d": round(_clip(direction, -1, 1), 6),
        "r": round(_clip(relevance, 0, 1), 6),
        "count": len(usable),
        "count_1h": sum(1 for row in usable if row["age_ms"] <= 3_600_000),
        "count_4h": sum(1 for row in usable if row["age_ms"] <= 4 * 3_600_000),
        "high_urgency_1h": counts_1h,
        "high_urgency_4h": counts_4h,
        "high_urgency_24h": counts_24h,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "sources": sources,
        "top_news": top,
        "gate": {
            "threshold": GATE,
            "u_r": round(_clip(urgency, 0, 1) * _clip(relevance, 0, 1), 6),
            "triggered": bool(usable) and (urgency * relevance) >= GATE,
        },
    }
    agg["alert"] = build_alert(agg)
    agg["summary"] = {
        "label": sentiment_word(direction),
        "net_sentiment": round(direction, 4),
        "count": len(usable),
        "high_urgency": counts_24h,
        "window_hours": WINDOW_HOURS,
        "line": (f"近 {WINDOW_HOURS}h ETH 相关新闻整体{sentiment_word(direction)}"
                 + ("，有高紧急消息" if counts_24h else "")),
    }
    return agg


def build_alert(agg: dict) -> dict:
    """Plain-Chinese alert built only from observed headlines and scores."""
    top = agg.get("top_news") or []
    if not top:
        return {
            "level": "none",
            "title": "暂无显著异常",
            "what": "近期没有抓取到 ETH 相关新闻，或新闻源暂时不可用。",
            "why": "没有新闻输入时，预测只展示模型原值，不做消息面调整。",
            "urgency": "低",
            "watch_window": "无需特别关注",
            "uncertain": False,
            "items": [],
        }
    urgent = [row for row in top if _num(row.get("urgency")) >= WATCH_URGENCY]
    triggered = bool(agg.get("gate", {}).get("triggered"))
    positive = int(agg.get("positive", 0))
    negative = int(agg.get("negative", 0))
    if positive and negative:
        consistency = "多空消息同时存在，方向存在分歧"
    elif positive:
        consistency = "消息整体偏利多，方向较一致"
    elif negative:
        consistency = "消息整体偏利空，方向较一致"
    else:
        consistency = "消息方向整体中性"
    if not urgent and not triggered:
        return {
            "level": "none",
            "title": "暂无显著异常",
            "what": (f"近 {agg.get('window_hours', WINDOW_HOURS)} 小时有 ETH 相关消息，"
                     "但没有达到紧急阈值的事件。"),
            "why": "当前消息强度不足以调整模型区间。",
            "consistency": consistency,
            "sources": "",
            "recency": "",
            "urgency": "低",
            "watch_window": "常规关注即可",
            "uncertain": False,
            "items": [],
        }

    level = "high" if any(_num(row.get("urgency")) >= HIGH_URGENCY for row in urgent) else "watch"
    label = sentiment_word(agg.get("d", 0))
    headlines: list[str] = []
    seen: set[str] = set()
    for row in urgent[:3]:
        title = (row.get("title") or "").strip()
        key = title[:24].lower()
        if not title or key in seen:
            continue
        seen.add(key)
        headlines.append(f"「{title}」（{row.get('sentiment_label') or '中性'}）")
    what = "；".join(headlines) if headlines else "近期有消息达到关注阈值。"
    event_types = [str(row.get("event_type") or "general") for row in top]
    why = "；".join(dict.fromkeys(_event_why(event) for event in event_types))
    max_urgency = max((_num(row.get("urgency")) for row in top), default=0.0)
    watch = "未来 1–4 小时" if max_urgency >= HIGH_URGENCY else "未来 1–2 小时"
    if int(agg.get("high_urgency_1h", 0)):
        recency = "高紧急消息集中在最近 1 小时"
    elif int(agg.get("high_urgency_4h", 0)):
        recency = "高紧急消息集中在最近 4 小时"
    else:
        recency = "高紧急消息分布在近 24 小时"
    source_names = []
    for row in top:
        name = str(row.get("source") or "").strip()
        if name and name not in source_names:
            source_names.append(name)
    uncertain = any(not row.get("summary_zh") for row in urgent) or any(
        row.get("scorer") == "rule_fallback" for row in urgent
    )
    return {
        "level": level,
        "title": ("出现高紧急 ETH 相关消息" if level == "high" else "有值得关注的 ETH 相关消息"),
        "what": what,
        "why": why,
        "consistency": consistency,
        "sources": "、".join(source_names[:3]),
        "recency": recency,
        "urgency": "高" if level == "high" else "中",
        "watch_window": watch,
        "uncertain": bool(uncertain),
        "note": ("当前部分条目只有标题和规则打分，细节有限。" if uncertain else ""),
        "items": top[:3],
    }


def apply_overlay(forecast: dict | None, agg: dict) -> dict | None:
    """Return the display-only adjusted quantiles; never mutates ``forecast``."""
    if not forecast:
        return None
    keys = ("median", "p10", "p25", "p75", "p90")
    try:
        raw = {key: float(forecast[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return None
    if any(value != value for value in raw.values()):
        return None

    gate = agg.get("gate", {})
    triggered = bool(gate.get("triggered"))
    u = _clip(_num(agg.get("u")), 0.0, 1.0)
    d = _clip(_num(agg.get("d")), -1.0, 1.0)
    r = _clip(_num(agg.get("r")), 0.0, 1.0)
    scale = 1.0
    delta = 0.0
    cap = 0.0
    if triggered:
        scale = _clip(1.0 + ALPHA * u * r, 1.0, MAX_WIDTH_MULTIPLIER)
        width50 = raw["p75"] - raw["p25"]
        delta_raw = BETA * d * r * (width50 / 2.0)
        cap = 0.5 * (width50 / 2.0)
        delta = _clip(delta_raw, -cap, cap)
    median_adj = raw["median"] + delta
    adjusted = {key: median_adj + (raw[key] - raw["median"]) * scale for key in keys}
    # Keep the invariant explicit even if a bad artifact has crossed quantiles.
    ordered = sorted((adjusted["p10"], adjusted["p25"], adjusted["median"],
                      adjusted["p75"], adjusted["p90"]))
    adjusted = dict(zip(("p10", "p25", "median", "p75", "p90"), ordered))
    width50 = max(raw["p75"] - raw["p25"], 0.0)
    width80 = max(raw["p90"] - raw["p10"], 0.0)
    ref_close = _num(forecast.get("ref_close"), raw["median"])
    return {
        "applied": triggered,
        "u": round(u, 6),
        "d": round(d, 6),
        "r": round(r, 6),
        "gate": {
            "threshold": GATE,
            "u_r": round(u * r, 6),
            "triggered": triggered,
        },
        "window_hours": agg.get("window_hours", WINDOW_HOURS),
        "half_life_hours": agg.get("half_life_hours"),
        "decay_formula": agg.get("decay_formula"),
        "alpha": ALPHA,
        "beta": BETA,
        "width_multiplier": round(scale, 6),
        "median_delta": round(delta, 6),
        "median_delta_cap": round(cap, 6),
        "width50_bps": round(width50 / ref_close * 1e4, 4) if ref_close else None,
        "width80_bps": round(width80 / ref_close * 1e4, 4) if ref_close else None,
        "adjusted": {key: round(value, 6) for key, value in adjusted.items()},
        "model": {key: round(value, 6) for key, value in raw.items()},
        "top_news": agg.get("top_news", [])[:3],
        "counts": {
            "count": agg.get("count", 0),
            "high_urgency_1h": agg.get("high_urgency_1h", 0),
            "high_urgency_4h": agg.get("high_urgency_4h", 0),
            "high_urgency_24h": agg.get("high_urgency_24h", 0),
            "positive": agg.get("positive", 0),
            "negative": agg.get("negative", 0),
            "neutral": agg.get("neutral", 0),
        },
        "sources": agg.get("sources", {}),
    }
