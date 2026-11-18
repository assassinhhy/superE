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
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from supere.board.engine import analyze
from supere.board.fetcher import BoardFetcher
from supere.board.journal import PlanJournal, journal_path, parse_bound
from supere.board.risk import load_calendar
from supere.board.settings import AnalysisConfig, load_analysis
from supere.config import apply_no_proxy, load_config, proxy_or_none
from supere.netproxy import resolve_proxy

log = logging.getLogger("supere.board")

# The 10-second panel poll re-resolves at most once a minute. The plans, stats
# and CSV endpoints force a resolve so a reviewer never reads a stale outcome.
RESOLVE_THROTTLE_MS = 60_000


class NotePayload(BaseModel):
    followed: bool | None = None
    note: str | None = None


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
    frozen: dict | None = None,
) -> FastAPI:
    root = Path(data_root)
    owns_fetcher = fetcher is None and frozen is None
    if fetcher is None and frozen is None:
        fetcher = BoardFetcher(analysis, root, proxy=proxy, clock=clock)
    journal = PlanJournal(journal_path(root))
    symbol = analysis.board.symbol

    def load_bars(interval: str, start_ms: int, end_ms: int) -> list[dict]:
        if fetcher is None:
            return []
        bars = fetcher.reader.klines_between(symbol, interval, start_ms, end_ms)
        if bars:
            return bars
        return fetcher.cache.klines_between(symbol, interval, start_ms, end_ms)

    resolve_lock = threading.Lock()
    last_resolve: dict[str, int | None] = {"ms": None}

    def resolve_now(now_ms: int | None = None, *, force: bool = False) -> None:
        moment = now_ms
        if moment is None and fetcher is not None:
            moment = fetcher.now()
        if moment is None:
            return
        moment = int(moment)
        with resolve_lock:
            previous = last_resolve["ms"]
            if not force and previous is not None and 0 <= moment - previous < RESOLVE_THROTTLE_MS:
                return
            last_resolve["ms"] = moment
            try:
                journal.resolve_pending(load_bars, moment)
            except Exception:
                log.exception("计划结果回填失败")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()
        thread = None
        try:
            resolve_now(force=True)
        except Exception:
            log.exception("启动时回填计划结果失败")
        if schedule and fetcher is not None:
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
            if owns_fetcher and fetcher is not None:
                fetcher.close()
            journal.close()

    app = FastAPI(title="superE board", default_response_class=UTF8JSONResponse, lifespan=lifespan)
    app.state.fetcher = fetcher
    app.state.journal = journal
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "role": "board", "symbol": analysis.board.symbol, "data_dir": str(root)}

    @app.get("/api/board/data")
    def board_data():
        if frozen is not None:
            return {"frozen": True, "as_of_utc8": frozen.get("as_of_utc8")}
        return fetcher.view()

    def _bounds(start: str | None, end: str | None) -> tuple[int | None, int | None]:
        return parse_bound(start), parse_bound(end, end=True)

    _BAD_BOUND = {"error": "时间格式应为 UTC+8 的 YYYY-MM-DD HH:MM 或毫秒"}

    @app.get("/api/board/plans")
    def board_plans(
        start: str | None = None,
        end: str | None = None,
        limit: int = 200,
        status: str | None = None,
        kind: str | None = None,
        followed: bool | None = None,
    ):
        try:
            start_ms, end_ms = _bounds(start, end)
        except ValueError:
            return UTF8JSONResponse(_BAD_BOUND, status_code=400)
        resolve_now(force=True)
        rows = journal.list_plans(
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            status=status,
            kind=kind,
            followed=followed,
        )
        return {"plans": rows, "count": len(rows)}

    @app.get("/api/board/plans.csv")
    def board_plans_csv(
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        followed: bool | None = None,
    ):
        try:
            start_ms, end_ms = _bounds(start, end)
        except ValueError:
            return UTF8JSONResponse(_BAD_BOUND, status_code=400)
        resolve_now(force=True)
        rows = journal.list_plans(
            start_ms=start_ms,
            end_ms=end_ms,
            limit=5000,
            status=status,
            kind=kind,
            followed=followed,
        )
        body = "\ufeff" + journal.to_csv(rows)
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=board-plans.csv"},
        )

    @app.get("/api/board/stats")
    def board_stats(
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        followed: bool | None = None,
    ):
        try:
            start_ms, end_ms = _bounds(start, end)
        except ValueError:
            return UTF8JSONResponse(_BAD_BOUND, status_code=400)
        resolve_now(force=True)
        return journal.summarize_plans(
            start_ms=start_ms,
            end_ms=end_ms,
            status=status,
            kind=kind,
            followed=followed,
        )

    @app.post("/api/board/plans/{plan_id}/note")
    def board_plan_note(plan_id: str, payload: NotePayload):
        row = journal.update_note(plan_id, followed=payload.followed, note=payload.note)
        if row is None:
            return UTF8JSONResponse({"error": "计划不存在"}, status_code=404)
        return {"plan": row}

    @app.get("/api/board")
    def board():
        if frozen is not None:
            return frozen
        snap = fetcher.view()
        snap["calendar"] = load_calendar(root / "board" / "calendar.csv")
        result = analyze(snap, analysis)
        try:
            journal.record_board(result, analysis)
        except Exception:
            log.exception("计划日志写入失败")
        resolve_now(result.get("as_of"))
        return result

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
    parser.add_argument("--fixture", help="回放 CSV 目录。和 --at 一起用时不访问外网")
    parser.add_argument("--at", help="回放时刻，UTC+8，格式 YYYY-MM-DD HH:MM")
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

    frozen = None
    schedule = True
    proxy = resolve_proxy(proxy_or_none(cfg.proxy.url))
    if args.fixture or args.at:
        if not args.fixture or not args.at:
            print("回放需要同时给出 --fixture 和 --at", file=sys.stderr)
            return 2
        from datetime import datetime

        from supere.board.replay import replay_at
        from supere.timeutil import UTC8

        moment = datetime.strptime(args.at, "%Y-%m-%d %H:%M").replace(tzinfo=UTC8)
        frozen = replay_at(Path(args.fixture), int(moment.timestamp() * 1000), analysis)
        schedule = False
        proxy = None
    try:
        uvicorn.run(
            create_app(
                root,
                analysis,
                proxy=proxy,
                schedule=schedule,
                frozen=frozen,
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
