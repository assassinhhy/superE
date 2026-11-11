"""Chinese news headlines built from scored items. Counts are never invented."""

from __future__ import annotations

import re
from typing import Any

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
LABELS = ("利多", "利空", "中性")
# Below this, a scored item leaves the main list and sits in the collapsed "其它" group.
PANEL_RELEVANCE_MIN = 0.4
_MARKET_TOKENS = {
    "eth", "ethereum", "ether", "vitalik", "ethusdt",
    "bitcoin", "btc", "etf", "liquidation", "liquidations",
    "fed", "fomc", "cpi", "inflation", "usd", "dollar", "dxy",
}
_MARKET_PHRASES = ("以太坊", "以太币", "比特币", "美联储", "通胀", "美元", "清算", "资金流")


def market_channel(title: str, body: str = "") -> bool:
    """ETH, macro (Fed / inflation / USD) or broad crypto (BTC, ETF flows, liquidations)."""
    text = f"{title} {body}".lower()
    tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]+", text))
    if tokens & _MARKET_TOKENS:
        return True
    return any(phrase in text for phrase in _MARKET_PHRASES)


def relevance_of(item: dict) -> float:
    for key in ("relevance", "eth_relevance"):
        try:
            value = item.get(key)
            if value is None:
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return 0.0


def in_main_feed(item: dict) -> bool:
    """Main list vs collapsed 其它. Low LLM relevance stays out of the overlay too."""
    return relevance_of(item) >= PANEL_RELEVANCE_MIN


def label_of(item: dict) -> str:
    raw = str(item.get("sentiment_label") or item.get("label") or "").strip()
    if raw in LABELS:
        return raw
    try:
        direction = float(item.get("direction") or 0.0)
    except (TypeError, ValueError):
        direction = 0.0
    if direction > 0.12:
        return "利多"
    if direction < -0.12:
        return "利空"
    return "中性"


def _event_ms(item: dict) -> int | None:
    for key in ("ts_published", "available_at", "ts_ingested"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def label_counts(items: list[dict], now_ms: int, window_ms: int) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    key = {"利多": "positive", "利空": "negative", "中性": "neutral"}
    for item in items or []:
        event = _event_ms(item)
        if event is None:
            continue
        age = int(now_ms) - event
        if age < 0 or age > int(window_ms):
            continue
        counts[key[label_of(item)]] += 1
    return counts


def bias_word(counts: dict[str, int]) -> str:
    """偏多 / 偏空 / 分歧 / 中性 from the label counts in one window."""
    pos = int(counts.get("positive") or 0)
    neg = int(counts.get("negative") or 0)
    if pos >= 2 and neg >= 2 and min(pos, neg) * 2 >= max(pos, neg):
        return "分歧"
    if pos > neg:
        return "偏多"
    if neg > pos:
        return "偏空"
    if pos and neg:
        return "分歧"
    return "中性"


def is_important(item: dict) -> bool:
    try:
        if int(item.get("importance") or 0) >= 2:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return float(item.get("urgency") or 0.0) >= 0.7
    except (TypeError, ValueError):
        return False


def _top_impact(items: list[dict], now_ms: int, window_ms: int) -> str:
    ranked: list[tuple[int, float, str]] = []
    for item in items or []:
        if not is_important(item):
            continue
        event = _event_ms(item)
        if event is None:
            continue
        age = int(now_ms) - event
        if age < 0 or age > int(window_ms):
            continue
        text = str(item.get("impact_zh") or item.get("summary_zh") or "").strip()
        if not text:
            continue
        try:
            importance = int(item.get("importance") or 0)
        except (TypeError, ValueError):
            importance = 0
        try:
            urgency = float(item.get("urgency") or 0.0)
        except (TypeError, ValueError):
            urgency = 0.0
        ranked.append((importance, urgency, text))
    if not ranked:
        return ""
    ranked.sort(key=lambda row: (-row[0], -row[1]))
    text = ranked[0][2]
    return text if text.endswith("。") else text + "。"


def rule_prose(window: str, bias: str, counts: dict[str, int], impact: str) -> str:
    total = int(counts.get("positive") or 0) + int(counts.get("negative") or 0) + int(counts.get("neutral") or 0)
    if total == 0:
        return f"{window}没有以太坊相关消息。"
    sentence = (
        f"{window}整体{bias}，利多 {counts.get('positive', 0)} 条、"
        f"利空 {counts.get('negative', 0)} 条、中性 {counts.get('neutral', 0)} 条。"
    )
    extra = (impact or "").strip()
    if extra and not extra.endswith("。"):
        extra += "。"
    return sentence + extra


def headline(bias: str, counts: dict[str, int]) -> str:
    return (
        f"最近 1 小时{bias} · 利多 {counts.get('positive', 0)} / "
        f"利空 {counts.get('negative', 0)} / 中性 {counts.get('neutral', 0)}"
    )


def compose_summary(items: list[dict], now_ms: int, base: dict | None = None,
                    digest: dict | None = None) -> dict:
    """Merge the overlay summary with a 1h headline and 1h/24h prose."""
    counts_1h = label_counts(items, now_ms, HOUR_MS)
    counts_24h = label_counts(items, now_ms, DAY_MS)
    bias_1h = bias_word(counts_1h)
    bias_24h = bias_word(counts_24h)
    prose_1h = (digest or {}).get("prose_1h") or rule_prose(
        "最近 1 小时", bias_1h, counts_1h, _top_impact(items, now_ms, HOUR_MS),
    )
    prose_24h = (digest or {}).get("prose_24h") or rule_prose(
        "最近 24 小时", bias_24h, counts_24h, _top_impact(items, now_ms, DAY_MS),
    )
    out = dict(base or {})
    out.update({
        "headline": headline(bias_1h, counts_1h),
        "bias_1h": bias_1h,
        "bias_24h": bias_24h,
        "counts_1h": counts_1h,
        "counts_24h": counts_24h,
        "prose_1h": str(prose_1h),
        "prose_24h": str(prose_24h),
    })
    return out


def highlight_items(items: list[dict], limit: int = 5) -> list[dict]:
    ranked = [item for item in items or [] if is_important(item)]
    ranked.sort(key=lambda item: (
        -int(item.get("importance") or 0),
        -float(item.get("urgency") or 0.0),
        -(_event_ms(item) or 0),
    ))
    return ranked[:limit]


def digest_payload(items: list[dict], now_ms: int) -> dict[str, Any]:
    """Facts the digest model is allowed to mention."""
    counts_1h = label_counts(items, now_ms, HOUR_MS)
    counts_24h = label_counts(items, now_ms, DAY_MS)
    important = []
    for item in highlight_items(items, 5):
        important.append({
            "label": label_of(item),
            "summary_zh": str(item.get("summary_zh") or "")[:180],
            "impact_zh": str(item.get("impact_zh") or "")[:180],
        })
    return {
        "hour_1": {"bias": bias_word(counts_1h), **counts_1h},
        "hour_24": {"bias": bias_word(counts_24h), **counts_24h},
        "important": important,
    }
