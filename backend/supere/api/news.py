"""Read-only news payload for the 8766 API (independent of the forecast model)."""

from __future__ import annotations

import sqlite3
import time

from supere.timeutil import format_utc8


def news_payload(conn: sqlite3.Connection, limit: int = 50, since_hours: int = 72) -> dict:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(news)")}
    if not cols:
        return {"items": [], "summary": None, "status": "empty", "timezone": "UTC+8"}
    avail = "COALESCE(available_at, ts_ingested)" if "available_at" in cols else "ts_ingested"
    direction = ("COALESCE(direction_score, direction, 0.0)"
                 if "direction_score" in cols else "COALESCE(direction, 0.0)")
    urgency = "COALESCE(urgency, 0.0)" if "urgency" in cols else "0.0"
    magnitude = "COALESCE(magnitude, 0.0)" if "magnitude" in cols else "0.0"
    event = "COALESCE(event_type, '')" if "event_type" in cols else "''"
    timing = "COALESCE(timing_quality, '')" if "timing_quality" in cols else "''"
    since = int(time.time() * 1000) - int(since_hours) * 3_600_000
    rows = conn.execute(
        f"SELECT id, {avail} AS available_at, ts_published, ts_ingested, source, title, url, "
        f"COALESCE(eth_relevance, 0.0), {direction}, {urgency}, {magnitude}, {event}, {timing}, "
        f"COALESCE(scored_by, ''), COALESCE(score_version, '') "
        f"FROM news WHERE {avail} >= ? ORDER BY available_at DESC, id DESC LIMIT ?",
        (since, int(limit)),
    ).fetchall()
    keys = ["id", "available_at", "ts_published", "ts_ingested", "source", "title", "url",
            "relevance", "direction", "urgency", "magnitude", "event_type",
            "timing_quality", "scored_by", "score_version"]
    items = []
    for row in rows:
        item = dict(zip(keys, tuple(row), strict=True))
        if item.get("available_at"):
            item["available_at_utc8"] = format_utc8(int(item["available_at"]))
        if item.get("ts_published"):
            item["ts_published_utc8"] = format_utc8(int(item["ts_published"]))
        if item.get("ts_ingested"):
            item["ts_ingested_utc8"] = format_utc8(int(item["ts_ingested"]))
        items.append(item)
    total = 0.0
    net = 0.0
    high = 0
    for item in items:
        w = max(0.05, float(item.get("relevance") or 0.0))
        total += w
        net += float(item.get("direction") or 0.0) * w
        if float(item.get("urgency") or 0.0) >= 0.7:
            high += 1
    score = net / total if total else 0.0
    label = "偏多" if score > 0.12 else ("偏空" if score < -0.12 else "中性")
    summary = {"label": label, "net_sentiment": round(score, 4), "count": len(items),
               "high_urgency": high, "window_hours": int(since_hours),
               "line": f"近 {since_hours}h 新闻 {len(items)} 条，整体{label}"
                       + (f"，高紧急度 {high} 条" if high else "")}
    return {"items": items, "summary": summary, "status": "ok" if items else "empty",
            "timezone": "UTC+8", "degraded": not bool(items)}
