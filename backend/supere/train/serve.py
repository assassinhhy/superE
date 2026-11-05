"""Debug prediction API on 127.0.0.1:8768. Read-only: never writes SQLite, never trades.

Why a separate process: the running 1a API on 8766 has no prediction route and
is not restarted for this. The UI proxies ``/api/predict`` to this port (see
``frontend/vite.config.js``); every other ``/api`` path still goes to 8766.

    python -m supere.train.serve --data-dir D:\\Tool\\cg\\superE\\data

Routes:

* ``GET /api/predict``               the five product intervals, in one call
                                       each item has ``probs``, ``direction`` and
                                       ``top_features`` (SHAP contributions, see predict.py)
* ``GET /api/predict?interval=15m``  one interval
* ``GET /api/health``                liveness only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from supere.train.config import PRODUCT_INTERVALS, TRAINABLE_INTERVALS
from supere.train.predict import QuantilePredictService as PredictService
from supere.timeutil import format_utc8

DEFAULT_PORT = 8768


def _news_payload(data_root: Path, limit: int, since_hours: int) -> dict:
    """Latest news plus a one-line aggregate. Never blocks /api/predict."""
    import time
    from supere.news.scoring import list_news
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - int(since_hours) * 3_600_000
    try:
        rows = list_news(Path(data_root) / "superE.sqlite", limit=limit, since_ms=since_ms)
    except Exception as exc:  # noqa: BLE001 - news is independent of the forecast
        return {"items": [], "summary": None, "status": "error",
                "message": f"{type(exc).__name__}: {exc}", "timezone": "UTC+8"}
    items = []
    for row in rows:
        item = dict(row)
        if item.get("available_at"):
            item["available_at_utc8"] = format_utc8(int(item["available_at"]))
        if item.get("ts_published"):
            item["ts_published_utc8"] = format_utc8(int(item["ts_published"]))
        if item.get("ts_ingested"):
            item["ts_ingested_utc8"] = format_utc8(int(item["ts_ingested"]))
        items.append(item)
    total_w = 0.0
    net = 0.0
    high = 0
    for item in items:
        w = max(0.05, float(item.get("relevance") or 0.0))
        total_w += w
        net += float(item.get("direction") or 0.0) * w
        if float(item.get("urgency") or 0.0) >= 0.7:
            high += 1
    net_score = (net / total_w) if total_w else 0.0
    label = "偏多" if net_score > 0.12 else ("偏空" if net_score < -0.12 else "中性")
    summary = {
        "label": label,
        "net_sentiment": round(float(net_score), 4),
        "count": len(items),
        "high_urgency": high,
        "window_hours": int(since_hours),
        "line": f"近 {since_hours}h 新闻 {len(items)} 条，整体{label}"
                + (f"，高紧急度 {high} 条" if high else ""),
    }
    return {"items": items, "summary": summary, "status": "ok" if items else "empty",
            "timezone": "UTC+8", "degraded": not bool(items),
            "note": "direction 为 DeepSeek/规则打分，仅辅助解释，不构成交易建议。"}

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def create_app(data_root: Path) -> FastAPI:
    service = PredictService(data_root)
    app = FastAPI(title="superE debug predict", default_response_class=UTF8JSONResponse)

    @app.get("/api/health")
    def health():
        return {"ok": True, "debug": True, "data_dir": str(Path(data_root))}

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

    @app.get("/api/news")
    def news(limit: int = Query(default=50, ge=1, le=200),
             since_hours: int = Query(default=72, ge=1, le=720)):
        return _news_payload(Path(data_root), limit, since_hours)

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
