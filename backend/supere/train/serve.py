"""Debug prediction API on 127.0.0.1:8768. Never trades.

The desk does not call this process. News lives on 127.0.0.1:8769
(``python -m supere news``) and does not load a model. These routes remain so
older checks and the click-time overlay still run if this process is started.

    python -m supere.train.serve --data-dir D:\\Tool\\cg\\superE\\data

Routes:

* ``POST /api/predict``              user-click path. The model quantiles are never overwritten.
* ``GET /api/predict``               the five product intervals, no news fetch
* ``GET /api/predict?interval=15m``  one interval
* ``GET /api/news``                  cached news only; does not fetch
* ``POST /api/news/refresh``         fetch now, with a 60s cooldown
* ``GET /api/health``                liveness only

The only writable side effect is the news cache and ``logs/predict_clicks.jsonl``;
K-line/model data stays read-only.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from supere.intervals import INTERVAL_MS
from supere.news import clicklog, overlay
from supere.news.panel import news_payload as _news_payload
from supere.news.panel import score_fetched as _score_fetched
from supere.news.serve import make_refresher, mount_news
from supere.train.config import PRODUCT_INTERVALS, TRAINABLE_INTERVALS
from supere.train.predict import QuantilePredictService as PredictService
from supere.timeutil import format_utc8

DEFAULT_PORT = 8768


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def _click_record(data_root: Path, now_ms: int, payload: dict, snapshot: dict,
                  news_payload: dict) -> dict:
    horizons = []
    for item in payload.get("items") or []:
        forecast = item.get("price_forecast") or {}
        ov = item.get("news_overlay") or {}
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


def create_app(data_root: Path, *, news_snapshot=None, clock=None, schedule: bool = False) -> FastAPI:
    root = Path(data_root)
    service = PredictService(root, clock=clock) if clock is not None else PredictService(root)
    db_path = root / "superE.sqlite"

    def now_ms() -> int:
        if clock is None:
            return int(time.time() * 1000)
        return int(clock() * 1000)

    refresher = make_refresher(root, news_snapshot=news_snapshot, now_ms=now_ms)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if schedule:
            refresher.start()
        try:
            yield
        finally:
            refresher.stop()

    app = FastAPI(title="superE debug predict", default_response_class=UTF8JSONResponse, lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"ok": True, "debug": True, "data_dir": str(root)}

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
        """Explicit user action: cached news (refetch if older than 1h) -> model -> overlay."""
        at_ms = now_ms()
        try:
            snapshot = refresher.for_predict(at_ms)
        except Exception as exc:  # noqa: BLE001 - news failure must not block the model
            snapshot = {"items": [], "sources": {}, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}", "fetched_at": at_ms, "scored": True}
        if not snapshot.get("scored"):
            snapshot = _score_fetched(db_path, snapshot)
        payload = service.predict_all(PRODUCT_INTERVALS)
        try:
            news_payload = _news_payload(
                root, 80, 24, items_override=snapshot.get("items") or [], now_ms=at_ms,
                sources=snapshot.get("sources") or {}, status=snapshot.get("status"),
                scorer_version=snapshot.get("scorer_version"),
                meta=refresher.public_state(at_ms),
            )
        except Exception as exc:  # noqa: BLE001 - news must never block the model
            empty = overlay.aggregate([], at_ms, 3_600_000)
            news_payload = {"items": [], "summary": empty["summary"], "alert": empty["alert"],
                            "aggregate": {"u": 0.0, "d": 0.0, "r": 0.0, "count": 0},
                            "sources": snapshot.get("sources") or {}, "status": "error",
                            "message": f"{type(exc).__name__}: {exc}", "timezone": "UTC+8"}
        for item in payload.get("items") or []:
            interval = item.get("interval")
            horizon_ms = INTERVAL_MS.get(interval, 3_600_000)
            try:
                agg = overlay.aggregate(news_payload.get("items") or [], at_ms, horizon_ms)
                item["news_overlay"] = overlay.apply_overlay(item.get("price_forecast"), agg)
            except Exception:  # noqa: BLE001 - keep the raw forecast
                item["news_overlay"] = None
        payload["news"] = news_payload
        payload["trigger"] = "user_click"
        payload["click_at_utc8"] = format_utc8(at_ms)
        clicklog.append(root, _click_record(root, at_ms, payload, snapshot, news_payload))
        return payload

    mount_news(app, root, refresher, now_ms)
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

    uvicorn.run(
        create_app(Path(args.data_dir).resolve(), schedule=True),
        host=args.host, port=args.port, log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
