"""HTTP API on 127.0.0.1:8766. It never writes SQLite."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from supere import __version__
from supere.api.read import MISSING_UI_HTML, aggregate_check, catalog, disk_view, settings_view
from supere.api.news import news_payload
from supere.config import AppConfig
from supere.ipc.bus import LiveHub
from supere.storage.dal import DataAccess
from supere.timeutil import format_utc8


class UTF8JSONResponse(JSONResponse):
    """Starlette only appends ``charset`` for ``text/*``. Windows clients then
    guess GBK and split the last byte of a CJK character."""

    media_type = "application/json; charset=utf-8"


def create_app(
    cfg: AppConfig,
    dal: DataAccess,
    hub: LiveHub,
    *,
    connect_ipc: bool = True,
    static_dir: Path | None = None,
    data_path: Path | None = None,
) -> FastAPI:
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
    app.state.data_path = data_path
    app.state.preview = False
    ui_ready = static_dir is not None and (static_dir / "index.html").is_file()
    app.state.ui = ui_ready

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": __version__,
            "db": dal.ready,
            "api_port": cfg.server.api_port,
            "ipc_connected": hub.connected,
            "preview": bool(app.state.preview),
            "ui": bool(app.state.ui),
            "timezone": "UTC+8",
        }

    @app.get("/api/status")
    def status():
        body = dict(hub.status)
        body["db"] = dal.ready
        body["proxy"] = cfg.proxy.url
        body["timezone"] = "UTC+8"
        body["preview"] = bool(app.state.preview)
        if body.get("ts"):
            body["ts_utc8"] = format_utc8(int(body["ts"]))
        if body.get("self_weight_minute") is None and dal.db is not None:
            row = dal.db.latest_runtime("futures")
            if row is not None:
                body["self_weight_minute"] = row["self_weight_minute"]
                body["self_weight_peak"] = row["self_weight_peak"]
                body["header_weight"] = row["header_weight"]
                body["rest_halted"] = bool(row["halted"])
                body.setdefault("ts", row["ts"])
                body["ts_utc8"] = format_utc8(int(row["ts"]))
        body["weight_budget"] = int(cfg.rate_limit.futures_weight_per_min * cfg.rate_limit.futures_budget_ratio)
        disk = disk_view(data_path, cfg.disk.warn_free_gb, cfg.disk.critical_free_gb)
        if disk is not None:
            body["disk_free_gb"] = disk["free_gb"]
            body["disk_level"] = disk["level"]
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
    def funding(
        symbol: str = "ETHUSDT",
        start: int | None = None,
        end: int | None = None,
        limit: int = Query(default=50, ge=1, le=2000),
    ):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        if start is None and end is None:
            rows = dal.db.conn.execute(
                "SELECT * FROM funding WHERE symbol=? ORDER BY funding_time DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
            settled = []
            for row in rows:
                item = dict(row)
                item["funding_time_utc8"] = format_utc8(int(item["funding_time"]))
                settled.append(item)
        else:
            settled = dal.funding_between(symbol.upper(), start, end, limit)
        live = dal.db.conn.execute(
            "SELECT * FROM funding_live WHERE symbol=?", (symbol.upper(),)
        ).fetchone()
        predicted = dict(live) if live else None
        return {
            "settled": settled,
            "predicted": predicted,
            "note": "settled 是已结算费率。predicted 是实时预测，不进历史，也不进图上的资金费率。",
        }

    @app.get("/api/metrics")
    def metrics(
        symbol: str = "ETHUSDT",
        start: int | None = None,
        end: int | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
    ):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        if start is None and end is None:
            rows = []
            for row in dal.db.recent_metrics(symbol.upper(), limit):
                item = dict(row)
                item["ts_utc8"] = format_utc8(int(item["ts"]))
                item["known_at"] = int(item["ts"]) + 300_000
                rows.append(item)
            rows.reverse()
        else:
            rows = dal.metrics_between(symbol.upper(), start, end, limit)
        return {
            "rows": rows,
            "note": "ts 是 5 分钟 bar 起点，数值在 ts+5 分钟才可观察",
        }

    @app.get("/api/catalog")
    def data_catalog():
        if not dal.ready:
            raise HTTPException(503, "数据库还没准备好")
        return catalog(dal)

    @app.get("/api/settings")
    def settings():
        return settings_view(cfg, data_path)

    @app.get("/api/signals")
    def signals(limit: int = Query(default=50, ge=1, le=200)):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        return {
            "phase": 2,
            "rows": [dict(row) for row in dal.db.list_signals(limit)],
            "note": "预测信号第二阶段才写入。这个接口不计算方向，也不填占位概率。",
        }

    @app.get("/api/jobs")
    def jobs(limit: int = Query(default=50, ge=1, le=200)):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        return {"phase": 2, "rows": [dict(row) for row in dal.db.list_jobs(limit)]}

    @app.get("/api/models")
    def models(limit: int = Query(default=50, ge=1, le=200)):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        return {"phase": 2, "rows": [dict(row) for row in dal.db.list_models(limit)]}

    @app.get("/api/checks/aggregate")
    def checks_aggregate(
        interval: str = "5m",
        bars: int = Query(default=4, ge=1, le=12),
    ):
        if not dal.ready:
            raise HTTPException(503, "数据库还没准备好")
        return aggregate_check(dal, interval, bars)

    @app.get("/api/liquidations")
    def liquidations(
        start: int,
        end: int,
        symbol: str = "ETHUSDT",
    ):
        rows: list[dict] = []
        if dal.duck is not None:
            try:
                rows.extend(dal.duck.force_orders("um", symbol.upper(), start, end))
            except Exception:
                rows = []
        seen = {(item.get("ts"), str(item.get("price")), str(item.get("qty"))) for item in rows}
        for item in hub.liquidations:
            ts = int(item.get("event_ms") or item.get("trade_time") or 0)
            if ts < start or ts > end:
                continue
            key = (ts, str(item.get("price")), str(item.get("qty")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "ts": ts,
                    "price": item.get("price"),
                    "qty": item.get("qty"),
                    "side": item.get("side"),
                }
            )
        rows.sort(key=lambda item: int(item["ts"]))
        return {"rows": rows, "note": "只包含自录强平，历史补不回来，而且推送是抽样"}

    @app.get("/api/fear-greed")
    def fear_greed(limit: int = 30):
        if not dal.ready or dal.db is None:
            raise HTTPException(503, "数据库还没准备好")
        return {"rows": [dict(row) for row in dal.db.list_fear_greed(limit)]}

    @app.get("/api/news")
    def news(limit: int = Query(default=50, ge=1, le=200),
             since_hours: int = Query(default=72, ge=1, le=720)):
        if not dal.ready or dal.db is None:
            return {"items": [], "summary": None, "status": "empty", "timezone": "UTC+8"}
        try:
            return news_payload(dal.db.conn, limit, since_hours)
        except Exception as exc:  # noqa: BLE001 - news must not break the API
            return {"items": [], "summary": None, "status": "error", "timezone": "UTC+8",
                    "message": f"{type(exc).__name__}: {exc}", "degraded": True}

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
                        "closed": list(hub.closed),
                        "depth": hub.depth,
                        "ticker": hub.ticker,
                        "mark": hub.mark,
                        "trades": list(hub.trades)[:40],
                        "liquidations": list(hub.liquidations)[:30],
                        "status": hub.status,
                    }
                )
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    if ui_ready:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    else:

        @app.get("/")
        def ui_missing():
            return HTMLResponse(MISSING_UI_HTML)

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
    setup_logging(root / "logs")
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
    dist = Path(config_path).resolve().parent / "frontend" / "dist"
    app = create_app(cfg, dal, hub, connect_ipc=True, static_dir=dist, data_path=root)
    config = uvicorn.Config(
        app,
        host=cfg.server.api_host,
        port=cfg.server.api_port,
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    app.state.server = server
    try:
        await server.serve()
    finally:
        if db is not None:
            db.close()
        if duck is not None:
            duck.close()
        Path(pid).unlink(missing_ok=True)
