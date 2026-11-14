"""News JSON for the panel. This module does not import the forecast model."""

from __future__ import annotations

import time
from pathlib import Path

from supere import secrets
from supere.news import live, overlay
from supere.news.refresh import next_slot_ms
from supere.news.scoring import (
    content_hash,
    list_news,
    rule_score,
    rule_summary_zh,
    score_items,
    score_version,
    sentiment_label,
)
from supere.news.summary import compose_summary, highlight_items, in_main_feed
from supere.timeutil import format_utc8


def display_item(item: dict) -> dict:
    """Add Chinese display fields without changing the stored scores."""
    out = dict(item)
    title = str(out.get("title") or "")
    direction = out.get("direction")
    if direction is None:
        direction = out.get("direction_score")
    direction = float(direction or 0.0)
    urgency = float(out.get("urgency") or 0.0)
    relevance = float(out.get("relevance") or out.get("eth_relevance") or 0.0)
    event_type = str(out.get("event_type") or "general")
    out["direction"] = round(direction, 4)
    out["urgency"] = round(urgency, 4)
    out["relevance"] = round(relevance, 4)
    out["sentiment_label"] = out.get("sentiment_label") or sentiment_label(direction)
    if not out.get("summary_zh"):
        out["summary_zh"] = rule_summary_zh(title, event_type, direction, urgency, relevance)
    if not out.get("content_hash"):
        out["content_hash"] = content_hash(title, str(out.get("body") or ""))
    try:
        importance = int(out.get("importance") or 0)
    except (TypeError, ValueError):
        importance = 0
    if importance not in (1, 2, 3):
        importance = 3 if urgency >= 0.7 else (2 if urgency >= 0.45 else 1)
    out["importance"] = importance
    out["reason_zh"] = str(out.get("reason_zh") or "")
    out["impact_zh"] = str(out.get("impact_zh") or "")
    if importance >= 2 and not out["impact_zh"]:
        out["impact_zh"] = out.get("summary_zh") or ""
    available = out.get("available_at")
    published = out.get("ts_published")
    if available:
        out["available_at_utc8"] = format_utc8(int(available))
    if published:
        out["ts_published_utc8"] = format_utc8(int(published))
    if out.get("ts_ingested"):
        out["ts_ingested_utc8"] = format_utc8(int(out["ts_ingested"]))
    return out


def news_payload(data_root: Path, limit: int, since_hours: int,
                 *, items_override: list[dict] | None = None, now_ms: int | None = None,
                 sources: dict | None = None, status: str | None = None,
                 scorer_version: str | None = None, meta: dict | None = None) -> dict:
    """Latest cached news plus a Chinese aggregate. Does not fetch."""
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    meta = dict(meta or {})
    window = min(int(since_hours), 24)
    if items_override is not None:
        rows = [row for row in items_override if str(row.get("source") or "") != "gdelt"]
    else:
        since_ms = now_ms - window * 3_600_000
        try:
            rows = list_news(
                Path(data_root) / "superE.sqlite", limit=limit, since_ms=since_ms,
                sources=tuple(sorted(live.PANEL_SOURCES)),
            )
        except Exception as exc:  # noqa: BLE001 - a news failure is only a news failure
            nxt = int(meta.get("next_refresh_at") or next_slot_ms(now_ms))
            return {"items": [], "summary": None, "alert": None, "highlights": [],
                    "status": "error", "message": f"{type(exc).__name__}: {exc}",
                    "timezone": "UTC+8", "last_refresh_at": meta.get("last_refresh_at"),
                    "next_refresh_at": nxt,
                    "refreshing": bool(meta.get("refreshing")),
                    "cooldown_remaining_s": int(meta.get("cooldown_remaining_s") or 0),
                    "sources": sources if sources is not None else (meta.get("sources") or {})}
    shown = [display_item(dict(row)) for row in rows]
    shown.sort(key=lambda row: (row.get("ts_published") or row.get("available_at") or 0,
                                row.get("id") or 0), reverse=True)
    shown = shown[:limit]
    items = [row for row in shown if in_main_feed(row)]
    other = [row for row in shown if not in_main_feed(row)]
    agg = overlay.aggregate(items, now_ms, 3_600_000)
    digest = meta.get("digest") if isinstance(meta.get("digest"), dict) else None
    summary = compose_summary(items, now_ms, agg["summary"], digest)
    nxt = int(meta.get("next_refresh_at") or next_slot_ms(now_ms))
    return {
        "items": items,
        "other": other,
        "summary": summary,
        "alert": agg["alert"],
        "highlights": highlight_items(items),
        "aggregate": {
            "u": agg["u"], "d": agg["d"], "r": agg["r"],
            "count": agg["count"],
            "high_urgency_1h": agg["high_urgency_1h"],
            "high_urgency_4h": agg["high_urgency_4h"],
            "high_urgency_24h": agg["high_urgency_24h"],
            "positive": agg["positive"], "negative": agg["negative"], "neutral": agg["neutral"],
            "sources": agg["sources"],
        },
        "sources": sources if sources is not None else (meta.get("sources") or {}),
        "score_counts": meta.get("score_counts") or {},
        "scorer_version": scorer_version or score_version(),
        "status": status or ("ok" if (items or other) else "empty"),
        "timezone": "UTC+8",
        "degraded": not bool(items),
        "last_refresh_at": meta.get("last_refresh_at"),
        "last_refresh_at_utc8": meta.get("last_refresh_at_utc8"),
        "next_refresh_at": nxt,
        "next_refresh_at_utc8": meta.get("next_refresh_at_utc8") or format_utc8(nxt),
        "refreshing": bool(meta.get("refreshing")),
        "cooldown_remaining_s": int(meta.get("cooldown_remaining_s") or 0),
    }


def score_fetched(db_path: Path, snapshot: dict) -> dict:
    """Score newly fetched items. LLM when configured, else DeepSeek or rules.

    Rows already scored for this content hash are reused and are not sent to the model.
    """
    items = snapshot.get("items") or []
    if not items:
        snapshot["scored"] = True
        return snapshot
    use_llm = secrets.llm_configured()
    try:
        if use_llm:
            from supere.news import llm
            result = llm.score_panel_items(db_path, items, proxy=live.resolve_proxy())
        else:
            result = score_items(db_path, items, use_deepseek=True, timeout=15.0)
        snapshot["score_counts"] = result["counts"]
        snapshot["scorer_version"] = result["version"]
        scores = result["scores"]
    except Exception as exc:  # noqa: BLE001 - keep the fetched titles
        snapshot["score_counts"] = {"error": type(exc).__name__}
        snapshot["scorer_version"] = score_version()
        scores = {}
    for item in items:
        title = str(item.get("title") or "")
        body = str(item.get("body") or "")
        ch = str(item.get("content_hash") or content_hash(title, body))
        item["content_hash"] = ch
        score = scores.get(ch)
        if score is None:
            score = rule_score(title, body)
            if use_llm:
                score.scorer = "rule"
            if not score.summary_zh:
                score.summary_zh = rule_summary_zh(
                    title, score.event_type, score.direction, score.urgency, score.eth_relevance
                )
        item["relevance"] = score.eth_relevance
        item["direction"] = score.direction
        item["urgency"] = score.urgency
        item["magnitude"] = score.magnitude
        item["event_type"] = score.event_type
        item["confidence"] = score.confidence
        item["scored_by"] = score.scorer
        item["score_version"] = snapshot.get("scorer_version") or score_version()
        item["summary_zh"] = score.summary_zh
        item["sentiment_label"] = score.label or sentiment_label(score.direction)
        item["reason_zh"] = score.reason_zh
        item["importance"] = int(score.importance or 1)
        item["impact_zh"] = score.impact_zh
    snapshot["scored"] = True
    return snapshot
