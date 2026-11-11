"""Debug prediction API on 127.0.0.1:8768. Never trades.

Why a separate process: the running 1a API on 8766 has no prediction route and
is not restarted for this. The UI proxies ``/api/predict`` to this port (see
``frontend/vite.config.js``); every other ``/api`` path still goes to 8766.

    python -m supere.train.serve --data-dir D:\\Tool\\cg\\superE\\data

Routes:

* ``POST /api/predict``              user-click path: fetch/score current ETH news,
                                       run the model, attach a display-only news
                                       overlay, append a click log. The model's
                                       own quantiles are never overwritten.
* ``GET /api/predict``               the five product intervals, no news fetch
* ``GET /api/predict?interval=15m``  one interval
* ``GET /api/news``                  cached news, Chinese summaries and the alert
* ``GET /api/health``                liveness only

The only writable side effect is the news cache and ``logs/predict_clicks.jsonl``;
K-line/model data stays read-only.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from supere.intervals import INTERVAL_MS
from supere.news import clicklog, live, overlay
from supere.news.scoring import (content_hash, list_news, rule_score, rule_summary_zh,
                                 score_items, score_version, sentiment_label)
from supere.train.config import PRODUCT_INTERVALS, TRAINABLE_INTERVALS
from supere.train.predict import QuantilePredictService as PredictService
from supere.timeutil import format_utc8

DEFAULT_PORT = 8768


def _display_item(item: dict) -> dict:
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
    available = out.get("available_at")
    published = out.get("ts_published")
    if available:
        out["available_at_utc8"] = format_utc8(int(available))
    if published:
        out["ts_published_utc8"] = format_utc8(int(published))
    if out.get("ts_ingested"):
        out["ts_ingested_utc8"] = format_utc8(int(out["ts_ingested"]))
    return out


def _news_payload(data_root: Path, limit: int, since_hours: int,
                  *, items_override: list[dict] | None = None, now_ms: int | None = None,
                  sources: dict | None = None, status: str | None = None,
                  scorer_version: str | None = None) -> dict:
    """Latest news plus a Chinese aggregate. Never blocks /api/predict."""
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    if items_override is not None:
        rows = items_override
    else:
        since_ms = now_ms - int(since_hours) * 3_600_000
        try:
            rows = list_news(Path(data_root) / "superE.sqlite", limit=limit, since_ms=since_ms)
        except Exception as exc:  # noqa: BLE001 - news is independent of the forecast
            return {"items": [], "summary": None, "alert": None, "status": "error",
                    "message": f"{type(exc).__name__}: {exc}", "timezone": "UTC+8"}
    items = [_display_item(dict(row)) for row in rows]
    items.sort(key=lambda row: (row.get("ts_published") or row.get("available_at") or 0,
                                row.get("id") or 0), reverse=True)
    items = items[:limit]
    agg = overlay.aggregate(items, now_ms, 3_600_000)
    summary = agg["summary"]
    return {
        "items": items,
        "summary": summary,
        "alert": agg["alert"],
        "aggregate": {
            "u": agg["u"], "d": agg["d"], "r": agg["r"],
            "count": agg["count"],
            "high_urgency_1h": agg["high_urgency_1h"],
            "high_urgency_4h": agg["high_urgency_4h"],
            "high_urgency_24h": agg["high_urgency_24h"],
            "positive": agg["positive"], "negative": agg["negative"], "neutral": agg["neutral"],
            "sources": agg["sources"],
        },
        "sources": sources if sources is not None else {},
        "scorer_version": scorer_version or score_version(),
        "status": status or ("ok" if items else "empty"),
        "timezone": "UTC+8",
        "degraded": not bool(items),
    }


def _score_fetched(db_path: Path, snapshot: dict) -> dict:
    """Score newly fetched items; rule fallback is handled inside score_items."""
    items = snapshot.get("items") or []
    if not items:
        return snapshot
    try:
        result = score_items(db_path, items, use_deepseek=True, timeout=15.0)
        snapshot["score_counts"] = result["counts"]
        snapshot["scorer_version"] = result["version"]
        scores = result["scores"]
    except Exception as exc:  # noqa: BLE001 - never block the forecast
        snapshot["score_counts"] = {"error": f"{type(exc).__name__}: {exc}"}
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
        item["sentiment_label"] = sentiment_label(score.direction)
    return snapshot

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def _click_record(data_root: Path, now_ms: int, payload: dict, snapshot: dict,
                  news_payload: dict) -> dict:
    horizons = []
    for item in payload.get("items") or []:
        forecast = item.get("price_forecast") or {}
        ov = item.get("news_overlay") or {}
        cal = dict(forecast.get("calibration") or {})
        # Keep the historical ``before``/``after`` keys untouched and add the
        # reproducible v6 chain.  ``before`` is the calibrated (served) model
        # interval; the raw v5 block is explicit below.
        horizons.append({
            "interval": item.get("interval"),
            "model_id": item.get("model_id"),
            "status": item.get("status"),
            "before": {key: forecast.get(key) for key in ("median", "p10", "p25", "p75", "p90")},
            "after": ov.get("adjusted"),
            "u": ov.get("u"),
            "d": ov.get("d"),
            "r": ov.get("r"),
            "gate": ov.get("gate"),
            "width_multiplier": ov.get("width_multiplier"),
            "median_delta": ov.get("median_delta"),
            "top_news": ov.get("top_news") or [],
            "asof_utc8": forecast.get("asof_utc8"),
            "decision_ms": forecast.get("decision_ms"),
            "horizon_end_ms": forecast.get("horizon_end_ms"),
            "raw_price_forecast": forecast.get("raw_price_forecast"),
            "calibrated_price_forecast": forecast.get("calibrated_price_forecast"),
            "final_price_forecast": item.get("final_price_forecast"),
            "calibration": {
                "calib": cal.get("calib"),
                "policy": cal.get("policy"),
                "calib_version": cal.get("calib_version"),
                "candidate": cal.get("candidate"),
                "window_days": cal.get("window_days"),
                "min_samples": cal.get("min_samples"),
                "n_mature": cal.get("n_mature"),
                "active": cal.get("active"),
                "reason": cal.get("reason"),
                "model_hash": cal.get("model_hash"),
                "model_id": cal.get("model_id"),
                "feature_hash": cal.get("feature_hash"),
                "source": cal.get("source"),
                "bootstrap": cal.get("bootstrap"),
                "alert": cal.get("alert"),
                "delay_ms": cal.get("delay_ms"),
            },
        })
    news_items = news_payload.get("items") or []
    aggregate = news_payload.get("aggregate") or {}
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ts_ms": int(now_ms),
        "trigger": "user_click",
        "news": {
            "status": news_payload.get("status"),
            "fetched_at": snapshot.get("fetched_at"),
            "sources": snapshot.get("sources") or {},
            "score_counts": snapshot.get("score_counts") or {},
            "scorer_version": news_payload.get("scorer_version"),
            "u": aggregate.get("u"),
            "d": aggregate.get("d"),
            "r": aggregate.get("r"),
            "items": [
                {
                    "id": row.get("id"),
                    "content_hash": row.get("content_hash"),
                    "title": row.get("title"),
                    "source": row.get("source"),
                    "url": row.get("url"),
                    "direction": row.get("direction"),
                    "urgency": row.get("urgency"),
                    "relevance": row.get("relevance"),
                    "summary_zh": row.get("summary_zh"),
                    "scored_by": row.get("scored_by"),
                    "score_version": row.get("score_version"),
                }
                for row in news_items[:50]
            ],
        },
        "horizons": horizons,
    }


def create_app(data_root: Path, *, news_snapshot=None, clock=None) -> FastAPI:
    root = Path(data_root)
    service = PredictService(root, clock=clock) if clock is not None else PredictService(root)
    db_path = root / "superE.sqlite"
    snapshotter = news_snapshot or (lambda now_ms: live.snapshot(now_ms, db_path))
    app = FastAPI(title="superE debug predict", default_response_class=UTF8JSONResponse)

    @app.get("/api/health")
    def health():
        out = {"ok": True, "debug": True, "data_dir": str(root)}
        try:
            out["calibration"] = service.calibration_status()
        except Exception as exc:  # noqa: BLE001 - liveness first
            out["calibration"] = {"enabled": True, "loaded": False,
                                  "error": f"{type(exc).__name__}: {exc}"}
        return out

    @app.get("/api/predict")
    def predict(interval: str | None = Query(default=None, max_length=8)):
        if interval is None:
            return service.predict_all(PRODUCT_INTERVALS)
        if interval not in TRAINABLE_INTERVALS:
            return UTF8JSONResponse(
                status_code=400,
                content={"debug": True, "status": "bad_interval", "direction": None,
                         "message": f"不支持的周期 {interval}；可选 {', '.join(TRAINABLE_INTERVALS)}"},
            )
        return service.predict(interval)

    @app.post("/api/predict")
    def predict_on_click():
        """Explicit user action: fetch news -> score -> model -> overlay -> log."""
        now_ms = int((clock() if clock is not None else time.time()) * 1000)
        try:
            snapshot = snapshotter(now_ms)
        except Exception as exc:  # noqa: BLE001 - news failure must not block the model
            snapshot = {"items": [], "sources": {}, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}", "fetched_at": now_ms}
        snapshot = _score_fetched(db_path, snapshot)
        payload = service.predict_all(PRODUCT_INTERVALS)
        try:
            news_payload = _news_payload(
                root, 60, 72, items_override=snapshot.get("items") or [], now_ms=now_ms,
                sources=snapshot.get("sources") or {}, status=snapshot.get("status"),
                scorer_version=snapshot.get("scorer_version"),
            )
        except Exception as exc:  # noqa: BLE001 - news must never block the model
            empty = overlay.aggregate([], now_ms, 3_600_000)
            news_payload = {"items": [], "summary": empty["summary"], "alert": empty["alert"],
                            "aggregate": {"u": 0.0, "d": 0.0, "r": 0.0, "count": 0},
                            "sources": snapshot.get("sources") or {}, "status": "error",
                            "message": f"{type(exc).__name__}: {exc}", "timezone": "UTC+8"}
        for item in payload.get("items") or []:
            interval = item.get("interval")
            horizon_ms = INTERVAL_MS.get(interval, 3_600_000)
            try:
                agg = overlay.aggregate(news_payload.get("items") or [], now_ms, horizon_ms)
                item["news_overlay"] = overlay.apply_overlay(item.get("price_forecast"), agg)
            except Exception:  # noqa: BLE001 - keep the raw forecast
                item["news_overlay"] = None
            # Explicit chain for the archive: raw -> calibrated -> final
            # (final == calibrated when the news gate did not fire).
            forecast = item.get("price_forecast") or {}
            ov = item.get("news_overlay") or {}
            item["final_price_forecast"] = (ov.get("adjusted")
                                            or forecast.get("calibrated_price_forecast")
                                            or None)
        payload["news"] = news_payload
        payload["trigger"] = "user_click"
        payload["click_at_utc8"] = format_utc8(now_ms)
        clicklog.append(root, _click_record(root, now_ms, payload, snapshot, news_payload))
        return payload

    @app.get("/api/news")
    def news(limit: int = Query(default=50, ge=1, le=200),
             since_hours: int = Query(default=72, ge=1, le=720)):
        return _news_payload(root, limit, since_hours)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m supere.train.serve")
    parser.add_argument("--data-dir", required=True, help="数据目录（含 superE.sqlite 和 models/）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        print("只允许本机回环地址", file=sys.stderr)
        return 2

    import uvicorn

    uvicorn.run(create_app(Path(args.data_dir).resolve()), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
