"""HTTP API on 127.0.0.1:8766. It never writes SQLite."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supere import __version__
from supere.config import AppConfig
from supere.ipc.bus import LiveHub
from supere.storage.dal import DataAccess
from supere.timeutil import format_utc8


class UTF8JSONResponse(JSONResponse):
    """Starlette only appends ``charset`` for ``text/*``. Windows clients then
    guess GBK and split the last byte of a CJK character."""

    media_type = "application/json; charset=utf-8"


def create_app(cfg: AppConfig, dal: DataAccess, hub: LiveHub, *, connect_ipc: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = None
        if connect_ipc:
            url = f"ws://{cfg.server.ipc_host}:{cfg.server.ipc_port}"
            task = asyncio.create_task(hub.run(url, stop))
        app.state.stop = stop
        yield
        stop.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="superE",
        version=__version__,
        lifespan=lifespan,
        default_response_class=UTF8JSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.cfg = cfg
    app.state.dal = dal
    app.state.hub = hub

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": __version__,
            "db": dal.ready,
            "api_port": cfg.server.api_port,
            "ipc_connected": hub.connected,
        }

    @app.get("/api/status")
    def status():
        body = dict(hub.status)
        body["db"] = dal.ready
        body["proxy"] = cfg.proxy.url
        body["timezone"] = "UTC+8"
        if body.get("ts"):
            body["ts_utc8"] = format_utc8(int(body["ts"]))
        return body

    @app.get("/api/klines")
    def klines(
        market: str = "um",
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start: int | None = None,
        end: int | None = None,
        limit: int = Query(default=500, ge=1, le=1500),
        include_forming: bool = False,
    ):
        if not dal.ready:
            raise HTTPException(503, "数据库还没准备好")
        rows = dal.klines(market, symbol.upper(), interval, start=start, end=end, limit=limit)
        if include_forming:
            for bar in hub.forming:
                if (
                    bar.get("market") == market
                    and bar.get("symbol") == symbol.upper()
                    and bar.get("interval") == interval
                ):
                    rows.append({**bar, "open_time_utc8": format_utc8(int(bar["open_time"]))})
        return {
            "market": market,
            "symbol": symbol.upper(),
            "interval": interval,
            "timezone": "UTC+8",
            "bars": rows,
        }

    @app.get("/api/coverage")
    def coverage():
        return {"rows": dal.coverage()}

    @app.get("/api/events")
    def events(limit: int = Query(default=100, ge=1, le=1000)):
        return {"rows": dal.events(limit)}

    @app.get("/api/anomalies")
    def anomalies(limit: int = Query(default=100, ge=1, le=1000)):
        return {"rows": dal.anomalies(limit)}

    @app.get("/api/funding")
    def funding(symbol: str = "ETHUSDT", limit: int = Query(default=50, ge=1, le=1000)):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        rows = dal.db.conn.execute(
            "SELECT * FROM funding WHERE symbol=? ORDER BY funding_time DESC LIMIT ?",
            (symbol.upper(), limit),
        ).fetchall()
        live = dal.db.conn.execute(
            "SELECT * FROM funding_live WHERE symbol=?", (symbol.upper(),)
        ).fetchone()
        return {
            "settled": [dict(row) for row in rows],
            "predicted": dict(live) if live else None,
        }

    @app.get("/api/metrics")
    def metrics(symbol: str = "ETHUSDT", limit: int = Query(default=50, ge=1, le=1000)):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        rows = dal.db.recent_metrics(symbol.upper(), limit)
        return {"rows": [dict(row) for row in rows], "note": "ts 是 5 分钟 bar 起点，数值在 ts+5m 才可观察"}

    @app.get("/api/fear-greed")
    def fear_greed(limit: int = 30):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        return {"rows": [dict(row) for row in dal.db.list_fear_greed(limit)]}

    @app.post("/api/shutdown")
    async def shutdown():
        # Drop the read-only connection before the ingestor checkpoints, so
        # the WAL can actually truncate.
        if dal.duck is not None:
            dal.duck.close()
            dal.duck = None
        if dal.db is not None:
            dal.db.close()
            dal.db = None
        await hub.send({"type": "shutdown"})
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True
        return {"ok": True}

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(
                    {
                        "forming": hub.forming,
                        "depth": hub.depth,
                        "ticker": hub.ticker,
                        "mark": hub.mark,
                        "status": hub.status,
                    }
                )
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    return app


async def serve(cfg: AppConfig, config_path) -> None:
    import os
    from pathlib import Path

    import uvicorn

    from supere.config import apply_no_proxy, assert_loopback, data_dir
    from supere.logging_setup import setup_logging
    from supere.storage.db import Database
    from supere.storage.duck import DuckReader

    apply_no_proxy(cfg.proxy.no_proxy)
    assert_loopback(cfg.server.api_host)
    root = data_dir(cfg, config_path)
    setup_logging(root / "logs", "api")
    pid = root / "run" / "api.pid"
    pid.parent.mkdir(parents=True, exist_ok=True)
    pid.write_text(str(os.getpid()), encoding="utf-8")
    db = None
    for _ in range(50):
        sqlite_path = root / "superE.sqlite"
        if sqlite_path.exists():
            try:
                db = Database(sqlite_path, read_only=True)
                break
            except Exception:
                await asyncio.sleep(0.2)
        else:
            await asyncio.sleep(0.2)
    duck = DuckReader(root / "superE.sqlite", root / "raw") if db is not None else None
    dal = DataAccess(db, duck)
    hub = LiveHub()
    app = create_app(cfg, dal, hub, connect_ipc=True)
    config = uvicorn.Config(
        app,
        host=cfg.server.api_host,
        port=cfg.server.api_port,
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    app.state.server = server
    import logging

    from supere.logging_setup import flush_logs

    log = logging.getLogger("supere")
    try:
        await server.serve()
    finally:
        log.info("API 开始关闭")
        flush_logs()
        if db is not None:
            db.close()
        if duck is not None:
            duck.close()
        Path(pid).unlink(missing_ok=True)
        log.info("API 已关闭")
        flush_logs()
