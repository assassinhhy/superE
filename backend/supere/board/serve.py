"""Board service on 127.0.0.1:8770. Read-only main database, no orders.

The chart API (8766), the collector (8767) and the news service (8769) stay up.
This process only reads them and serves ``GET /api/board``.

    python -m supere board
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supere.board.engine import analyze
from supere.board.fetcher import BoardFetcher
from supere.board.risk import load_calendar
from supere.board.settings import AnalysisConfig, load_analysis
from supere.config import apply_no_proxy, load_config, proxy_or_none

log = logging.getLogger("supere.board")


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def create_app(
    data_root: Path,
    analysis: AnalysisConfig,
    *,
    proxy: str | None = None,
    schedule: bool = False,
    fetcher: BoardFetcher | None = None,
    clock=None,
) -> FastAPI:
    root = Path(data_root)
    owns_fetcher = fetcher is None
    if fetcher is None:
        fetcher = BoardFetcher(analysis, root, proxy=proxy, clock=clock)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()
        thread = None
        if schedule:
            def loop() -> None:
                while not stop.wait(1.0):
                    try:
                        fetcher.poll_once()
                    except Exception:
                        log.exception("盘面拉取失败")

            thread = threading.Thread(target=loop, name="board-fetch", daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=2)
            if owns_fetcher:
                fetcher.close()

    app = FastAPI(title="superE board", default_response_class=UTF8JSONResponse, lifespan=lifespan)
    app.state.fetcher = fetcher
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "role": "board", "symbol": analysis.board.symbol, "data_dir": str(root)}

    @app.get("/api/board/data")
    def board_data():
        return fetcher.view()

    @app.get("/api/board")
    def board():
        snap = fetcher.view()
        snap["calendar"] = load_calendar(root / "board" / "calendar.csv")
        return analyze(snap, analysis)

    @app.get("/api/board/config")
    def board_config():
        return analysis.model_dump()

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m supere board")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--analysis")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("只允许本机回环地址", file=sys.stderr)
        return 2

    cfg, config_path = load_config(args.config)
    analysis, _path = load_analysis(args.analysis, config_path)
    port = int(args.port or analysis.board.port)
    root = Path(args.data_dir)
    apply_no_proxy(cfg.proxy.no_proxy)
    from supere.logging_setup import setup_logging

    setup_logging(root / "logs", "board")
    pid = root / "run" / "board.pid"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(str(os.getpid()), encoding="utf-8")

    import uvicorn

    try:
        uvicorn.run(
            create_app(
                root,
                analysis,
                proxy=proxy_or_none(cfg.proxy.url),
                schedule=True,
            ),
            host=args.host,
            port=port,
            log_level="info",
        )
    finally:
        try:
            pid.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
