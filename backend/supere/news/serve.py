"""News service on 127.0.0.1:8769. No forecast model, no orders.

The chart API (8766) and the collector (8767) stay up. This process only
fetches, scores and serves news. Start it from the repo root:

    python -m supere news
    python -m supere.news.serve --data-dir D:\\Tool\\cg\\superE\\data

Routes:

* ``GET /api/health``
* ``GET /api/news``           cached rows; does not fetch
* ``POST /api/news/refresh``  fetch now, 60s cooldown, one refresh at a time

The page on 8766 or 5173 calls this port directly. CORS allows loopback only.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supere import secrets
from supere.news import live
from supere.news.panel import news_payload, score_fetched
from supere.news.refresh import NewsRefresher

DEFAULT_PORT = 8769


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def make_refresher(data_root: Path, *, news_snapshot, now_ms) -> NewsRefresher:
    root = Path(data_root)
    db_path = root / "superE.sqlite"

    def fetcher(at_ms: int) -> dict:
        if news_snapshot is not None:
            return news_snapshot(at_ms)
        return live.snapshot(at_ms, db_path, proxy=live.resolve_proxy())

    def digest(items: list[dict], at_ms: int) -> dict | None:
        if not secrets.llm_configured():
            return None
        from supere.news import llm
        return llm.digest_prose(items, at_ms, proxy=live.resolve_proxy())

    return NewsRefresher(
        db_path,
        fetcher=fetcher,
        scorer=lambda snap: score_fetched(db_path, snap),
        now_ms=now_ms,
        digest_fn=digest,
    )


def mount_news(app: FastAPI, data_root: Path, refresher: NewsRefresher, now_ms) -> None:
    root = Path(data_root)

    @app.get("/api/news")
    def news(limit: int = Query(default=80, ge=1, le=200),
             since_hours: int = Query(default=24, ge=1, le=720)):
        at_ms = now_ms()
        state = refresher.public_state(at_ms)
        return news_payload(root, limit, since_hours, now_ms=at_ms, meta=state,
                            sources=state.get("sources") or {})

    @app.post("/api/news/refresh")
    def refresh_news():
        """Manual refresh. Cooling down returns the remaining seconds with HTTP 200."""
        at_ms = now_ms()
        result = refresher.refresh(at_ms, manual=True)
        state = refresher.public_state(at_ms)
        state["cooldown_remaining_s"] = int(result.get("cooldown_remaining_s") or 0)
        state["refreshing"] = bool(result.get("refreshing"))
        snap = result.get("snapshot") or {}
        # A failed fetch must not blank the panel. Keep the scored rows already on disk.
        failed = bool(result.get("accepted")) and snap.get("status") == "error"
        payload = news_payload(
            root, 80, 24,
            items_override=None if failed or not result.get("accepted") else snap.get("items"),
            now_ms=at_ms,
            sources=state.get("sources") if failed or not result.get("accepted") else snap.get("sources"),
            status=None if failed else (snap.get("status") if result.get("accepted") else None),
            scorer_version=snap.get("scorer_version"),
            meta=state,
        )
        if failed:
            payload["status"] = "error"
            payload["message"] = "这次没有抓到新消息，下面仍是上次的缓存。"
        payload["accepted"] = bool(result.get("accepted"))
        payload["cooling"] = bool(result.get("cooling"))
        payload["cooldown_remaining_s"] = int(result.get("cooldown_remaining_s") or 0)
        payload["refreshing"] = bool(result.get("refreshing"))
        return payload


def create_app(data_root: Path, *, news_snapshot=None, clock=None, schedule: bool = False) -> FastAPI:
    """News-only app. Does not construct a prediction service."""
    root = Path(data_root)

    def now_ms() -> int:
        if clock is None:
            import time
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

    app = FastAPI(title="superE news", default_response_class=UTF8JSONResponse, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "role": "news", "data_dir": str(root)}

    mount_news(app, root, refresher, now_ms)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m supere.news.serve")
    parser.add_argument("--data-dir", required=True, help="数据目录（含 superE.sqlite）")
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
