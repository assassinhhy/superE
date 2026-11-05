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

DEFAULT_PORT = 8768

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
