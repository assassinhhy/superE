"""Ingestor process: the only SQLite writer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

from supere.binance.parse import (
    dec_str,
    funding_from_rest,
    kline_from_list,
    metric_ts_from_rest,
    metrics_from_oi_hist,
    metrics_from_ratio,
    parse_ws_message,
)
from supere.binance.rest import BinanceRest
from supere.binance.streams import market_streams, public_streams, spot_streams
from supere.clock import ClockState, TimeSample, choose_offset
from supere.config import AppConfig, apply_no_proxy, assert_loopback, data_dir, proxy_or_none
from supere.diskmon import check_disk
from supere.ingest.anomaly import AnomalyEngine
from supere.ingest.backfill import Backfill
from supere.ingest.coverage_book import CoverageBook
from supere.ingest.gapfill import fill_recent_tail, startup_gapfill
from supere.ingest.kline_manager import KlineManager
from supere.ingest.overlap import OverlapGate
from supere.ipc.bus import IpcServer
from supere.lockfile import SingleInstance
from supere.logging_setup import setup_logging
from supere.news.adapters import default_registry, persist_news
from supere.ratelimit import WeightLimiter
from supere.storage.backup import backup_sqlite, copy_new_parquet, prune_old_backups
from supere.storage.db import Database
from supere.storage.models import MetricRow
from supere.storage.parquet_store import ParquetStore
from supere.timeutil import UTC8, local_now_ms, utc8_date

log = logging.getLogger("supere.live")

_DATA_KINDS = {"kline", "aggTrade", "depth", "forceOrder", "markPrice", "miniTicker"}


def _write_pid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


async def _close_ws(ws) -> None:
    if ws is None:
        return
    with contextlib.suppress(Exception):
        await ws.close()


class Ingestor:
    def __init__(self, cfg: AppConfig, config_path: Path, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.transport = transport
        self.root = data_dir(cfg, config_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg._archive_dir = self.root / "archive_cache"  # type: ignore[attr-defined]
        self.stop = asyncio.Event()
        self.clock = ClockState()
        self.status: dict = {"market": "down", "public": "down", "spot": "down"}
        self.db: Database | None = None
        self.store: ParquetStore | None = None
        self.rest: BinanceRest | None = None
        self.ipc: IpcServer | None = None
        self.klines: KlineManager | None = None
        self.anomaly: AnomalyEngine | None = None
        self.coverage = CoverageBook()
        self.lock = SingleInstance(self.root / "superE.lock")
        self._agg_state: tuple[int, int] | None = None
        self._last_depth_ms: int | None = None
        self._last_force_ms: int | None = None
        self._ticker: dict | None = None
        self._sub_id = 0

    def stamp(self) -> tuple[int, int]:
        local = local_now_ms()
        return self.clock.server_ms(local), local

    async def event(self, level: str, component: str, code: str, msg: str, detail: dict | None = None) -> None:
        server_ms, local_ms = self.stamp()
        logging.getLogger("supere").log(getattr(logging, level.upper(), logging.INFO), "%s %s", code, msg)
        if self.db is not None:
            self.db.add_event(server_ms, level, component, code, msg, detail, ts_local=local_ms)

    async def run(self, *, backfill: bool = False, live: bool = True) -> None:
        apply_no_proxy(self.cfg.proxy.no_proxy)
        assert_loopback(self.cfg.server.api_host)
        assert_loopback(self.cfg.server.ipc_host)
        setup_logging(self.root / "logs")
        self.lock.acquire()
        pid_path = self.root / "run" / "ingestor.pid"
        _write_pid(pid_path)
        self.db = Database(self.root / "superE.sqlite", read_only=False)
        self.store = ParquetStore(self.root / "raw")
        self.store.replay_pending()
        limits = self.cfg.rate_limit
        futures = WeightLimiter(
            "futures",
            limits.futures_weight_per_min,
            budget_ratio=limits.futures_budget_ratio,
            header_slowdown_ratio=limits.header_slowdown_ratio,
        )
        spot = WeightLimiter(
            "spot",
            limits.spot_weight_per_min,
            budget_ratio=limits.spot_budget_ratio,
            header_slowdown_ratio=limits.header_slowdown_ratio,
        )
        futures_data = WeightLimiter(
            "futures_data",
            limits.futures_data_requests_per_5min,
            budget_ratio=limits.futures_data_budget_ratio,
            header_slowdown_ratio=1.1,
            window_s=300,
        )
        self.rest = BinanceRest(
            self.cfg,
            futures,
            spot,
            futures_data,
            transport=self.transport,
            on_event=self.event,
        )
        self._futures = futures
        self._spot = spot
        self.klines = KlineManager(self.db, self.stamp)
        self.anomaly = AnomalyEngine(self.cfg.anomaly, self.db)
        self.ipc = IpcServer(self.cfg.server.ipc_host, self.cfg.server.ipc_port)
        self.ipc.on_shutdown = self.stop.set
        try:
            await self.ipc.start()
            await self.event(
                "info",
                "ingest",
                "INGESTOR_START",
                f"采集器已启动，IPC {self.ipc.host}:{self.ipc.port}",
                {"pid": os.getpid()},
            )
            await self.calibrate()
            disk = check_disk(self.root, self.cfg.disk.warn_free_gb, self.cfg.disk.critical_free_gb)
            if disk.level != "ok":
                await self.event(
                    "warning" if disk.level == "warn" else "error",
                    "disk",
                    "DISK_" + disk.level.upper(),
                    f"数据盘剩余 {disk.free_gb:.1f} GB",
                    {"free_gb": disk.free_gb},
                )
            if backfill:
                await Backfill(self.cfg, self.rest, self.db, self.stamp, self.event).run()
            if live:
                if not backfill:
                    await startup_gapfill(self.rest, self.db, self.store, self.stamp, self.cfg, self.event)
                await self._live()
        finally:
            await self._shutdown(pid_path)

    async def _shutdown(self, pid_path: Path) -> None:
        self.stop.set()
        if self.store is not None:
            with contextlib.suppress(Exception):
                paths = self.store.rotate()
                self.store.convert_many(paths)
            with contextlib.suppress(Exception):
                self.coverage.flush(self.db)
            self._flush_agg_state()
        if self.db is not None:
            with contextlib.suppress(Exception):
                self.db.checkpoint()
            self.db.close()
            self.db = None
        if self.rest is not None:
            await self.rest.aclose()
        if self.ipc is not None:
            await self.ipc.close()
        pid_path.unlink(missing_ok=True)
        self.lock.release()

    async def _live(self) -> None:
        tasks = [
            asyncio.create_task(self._socket_loop("market", self.cfg.endpoints.fstream_market, market_streams(self.cfg.symbols.um), self._on_market), name="market"),
            asyncio.create_task(self._socket_loop("public", self.cfg.endpoints.fstream_public, public_streams(self.cfg.symbols.um), self._on_public), name="public"),
            asyncio.create_task(self._socket_loop("spot", self.cfg.endpoints.spot_ws, spot_streams(self.cfg.symbols.spot), self._on_spot), name="spot"),
            asyncio.create_task(self._clock_loop(), name="clock"),
            asyncio.create_task(self._flush_loop(), name="flush"),
            asyncio.create_task(self._metrics_loop(), name="metrics"),
            asyncio.create_task(self._mark_loop(), name="mark"),
            asyncio.create_task(self._disk_loop(), name="disk"),
            asyncio.create_task(self._stats_loop(), name="stats"),
            asyncio.create_task(self._backup_loop(), name="backup"),
            asyncio.create_task(self._status_loop(), name="status"),
            asyncio.create_task(self._news_loop(), name="news"),
        ]
        await self.stop.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sleep(self, seconds: float) -> bool:
        if seconds < 0:
            seconds = 0
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def calibrate(self) -> None:
        samples: list[TimeSample] = []
        for _ in range(self.cfg.clock.samples):
            t_send = local_now_ms()
            try:
                server = await self.rest.server_time()
            except Exception as exc:
                await self.event("warning", "clock", "CLOCK_UNAVAILABLE", f"时钟校准失败: {exc}", {})
                return
            t_recv = local_now_ms()
            samples.append(TimeSample(t_send, t_recv, server))
        chosen = choose_offset(samples)
        kind = self.clock.apply(
            chosen,
            warn_abs_ms=self.cfg.clock.warn_abs_ms,
            change_alert_ms=self.cfg.clock.change_alert_ms,
        )
        detail = {
            "offset_ms": chosen.offset_ms,
            "rtt_ms": chosen.rtt_ms,
            "samples": [sample.offset_ms for sample in samples],
            "warn": self.clock.warn,
        }
        await self.event("info", "clock", "CLOCK_SYNC", f"时钟偏移 {chosen.offset_ms} ms", detail)
        if kind == "changed":
            await self.event("warning", "clock", "CLOCK_DRIFT", "偏移变化超过 500 ms", detail)
        elif kind == "skew":
            await self.event(
                "warning",
                "clock",
                "CLOCK_SKEW",
                "偏移超过 2 秒，请检查 Windows 时间同步",
                detail,
            )

    async def _clock_loop(self) -> None:
        while not self.stop.is_set():
            if await self._sleep(self.cfg.clock.recalibrate_seconds):
                return
            await self.calibrate()

    async def _connect(self, url: str):
        return await connect(
            url,
            proxy=proxy_or_none(self.cfg.proxy.url),
            ping_interval=self.cfg.ingest.client_ping_seconds,
            ping_timeout=120,
            open_timeout=20,
            close_timeout=2,
            max_size=4 * 1024 * 1024,
            compression=None,
            user_agent_header="superE/0.1.0",
        )

    async def _subscribe(self, ws, streams: list[str]) -> None:
        self._sub_id += 1
        await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": self._sub_id}))

    async def _apply(self, raw, conn_id: str, handler, gate: OverlapGate) -> None:
        parsed = parse_ws_message(raw)
        if not parsed:
            return
        if parsed.get("kind") == "error":
            await self.event("warning", "ws", "WS_ERROR", str(parsed.get("error")), {})
            return
        kind = parsed.get("kind")
        if kind not in _DATA_KINDS:
            return
        if gate.observe(conn_id, kind, parsed["dedupe_key"]):
            await handler(parsed)

    async def _socket_loop(self, name: str, url: str, streams: list[str], handler) -> None:
        backoff = 1.0
        gate = OverlapGate(self.cfg.ingest.overlap_confirm_messages)
        while not self.stop.is_set():
            try:
                await self._run_socket(name, url, streams, handler, gate)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status[name] = "down"
                await self.event("warning", "ws", "WS_DISCONNECT", f"{name} 断开: {exc}", {"socket": name})
                self._note_drop(name)
                if await self._sleep(backoff):
                    return
                backoff = min(backoff * 2, 30.0)
                if name in {"market", "spot"}:
                    with contextlib.suppress(Exception):
                        await self._refill_after_drop(name)

    async def _run_socket(self, name: str, url: str, streams: list[str], handler, gate: OverlapGate) -> None:
        ws = await self._connect(url)
        try:
            await self._subscribe(ws, streams)
            self.status[name] = "connected"
            await self.event("info", "ws", "WS_CONNECTED", f"{name} 已连接", {"url": url})
            started = time.monotonic()
            limit = max(1.0, self.cfg.ingest.reconnect_hours * 3600)
            while not self.stop.is_set():
                remaining = limit - (time.monotonic() - started)
                if remaining <= 0:
                    self.status[name] = "overlap"
                    ws = await self._overlap_switch(name, ws, url, streams, handler, gate)
                    started = time.monotonic()
                    self.status[name] = "connected"
                    continue
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(30.0, remaining))
                except TimeoutError:
                    continue
                await self._apply(raw, "active", handler, gate)
        finally:
            await _close_ws(ws)

    async def _overlap_switch(self, name, old, url, streams, handler, gate: OverlapGate):
        new = await self._connect(url)
        try:
            await self._subscribe(new, streams)
            gate.start("new")
            deadline = time.monotonic() + 30
            await self.event("info", "ws", "WS_OVERLAP_BEGIN", f"{name} 开始重叠切换", {})
            while not gate.promoted:
                if self.stop.is_set() or time.monotonic() > deadline:
                    await _close_ws(new)
                    await self.event("warning", "ws", "WS_OVERLAP_TIMEOUT", f"{name} 新连接没有及时出数据，保留旧连接", {})
                    return old
                await self._read_either(old, new, handler, gate)
            await _close_ws(old)
            await self.event("info", "ws", "WS_OVERLAP", f"{name} 已切到新连接", {})
            return new
        except Exception:
            await _close_ws(new)
            raise

    async def _read_either(self, old, new, handler, gate: OverlapGate) -> None:
        old_task = asyncio.create_task(old.recv())
        new_task = asyncio.create_task(new.recv())
        done, pending = await asyncio.wait({old_task, new_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(Exception):
                await task
        for task in done:
            raw = task.result()
            conn_id = "old" if task is old_task else "new"
            await self._apply(raw, conn_id, handler, gate)

    def _note_drop(self, name: str) -> None:
        server_ms, _local = self.stamp()
        if name == "public" and self._last_depth_ms is not None:
            if server_ms - self._last_depth_ms >= self.cfg.ingest.depth_gap_ms:
                self.coverage.gap("um:depth20", utc8_date(server_ms), self._last_depth_ms, server_ms, "disconnect")
        if name == "market" and self._last_force_ms is not None:
            self.coverage.gap("um:forceOrder", utc8_date(server_ms), self._last_force_ms, server_ms, "disconnect")

    async def _refill_after_drop(self, name: str) -> None:
        symbol = self.cfg.symbols.um if name == "market" else self.cfg.symbols.spot
        market = "um" if name == "market" else "spot"
        intervals = (
            __import__("supere.intervals", fromlist=["UM_INTERVALS"]).UM_INTERVALS
            if market == "um"
            else ("1m",)
        )
        for interval in intervals:
            await fill_recent_tail(
                self.rest, self.db, self.stamp, market, symbol, interval, self.cfg.ingest.startup_tail_limit
            )
        if market == "um":
            from supere.ingest.gapfill import fill_agg_gap

            await fill_agg_gap(self.rest, self.db, self.store, self.stamp, self.cfg, symbol)
        await self.event("info", "ingest", "GAP_FILL", f"{name} 断线后已补 K 线", {"socket": name})

    async def _on_market(self, parsed: dict) -> None:
        server_ms, local_ms = self.stamp()
        self.status["last_market_ms"] = server_ms
        kind = parsed["kind"]
        symbol = parsed.get("symbol") or self.cfg.symbols.um
        if kind == "kline":
            closed = await self.klines.on_kline("um", parsed)
            bar = None if closed is None else _public_bar(closed, closed=True)
            if bar is None:
                bar = self.klines.snapshot("um", symbol, parsed["interval"])
            await self.ipc.broadcast({"type": "kline", "closed": closed is not None, "bar": bar})
            if closed is not None and closed.interval == "1m":
                self.anomaly.on_closed_1m(closed)
            if closed is not None:
                asyncio.create_task(self._verify_later("um", closed))
        elif kind == "aggTrade":
            await self.klines.on_agg_trade("um", parsed)
            self.anomaly.on_agg_price(parsed["ts"], parsed["price"])
            self.store.append(
                "aggTrades",
                "um",
                symbol,
                utc8_date(parsed["ts"]),
                {
                    "agg_id": parsed["agg_id"],
                    "price": parsed["price"],
                    "qty": parsed["qty"],
                    "first_id": parsed["first_id"],
                    "last_id": parsed["last_id"],
                    "ts": parsed["ts"],
                    "is_buyer_maker": parsed["is_buyer_maker"],
                    "recv_ms": server_ms,
                    "recv_local_ms": local_ms,
                    "source": "ws",
                },
            )
            self.coverage.add("um:aggTrade", utc8_date(parsed["ts"]), parsed["ts"])
            self._agg_state = (parsed["agg_id"], parsed["ts"])
            await self.ipc.broadcast({"type": "aggTrade", "symbol": symbol, "price": parsed["price"], "qty": parsed["qty"], "ts": parsed["ts"], "agg_id": parsed["agg_id"]})
        elif kind == "forceOrder":
            self._last_force_ms = parsed["event_ms"] or server_ms
            row = {
                "event_ms": parsed["event_ms"],
                "recv_ms": server_ms,
                "recv_local_ms": local_ms,
                "trade_time": parsed["trade_time"],
                "symbol": symbol,
                "side": parsed["side"],
                "order_type": parsed["order_type"],
                "tif": parsed["tif"],
                "qty": parsed["qty"],
                "price": parsed["price"],
                "avg_price": parsed["avg_price"],
                "last_qty": parsed["last_qty"],
                "accum_qty": parsed["accum_qty"],
                "status": parsed["status"],
                "source": "ws",
            }
            self.store.append("forceOrder", "um", symbol, utc8_date(self._last_force_ms), row)
            self.coverage.add("um:forceOrder", utc8_date(self._last_force_ms), self._last_force_ms)
            self.anomaly.on_liquidation(self._last_force_ms, parsed["price"], parsed["qty"])
            await self.ipc.broadcast({"type": "forceOrder", **row})
        elif kind == "markPrice":
            self.db.upsert_funding_live(
                symbol,
                parsed["rate"],
                parsed["next_funding_time"],
                parsed["mark_price"],
                parsed["index_price"],
                server_ms,
                local_ms,
            )
            if parsed["rate"] is not None:
                self.anomaly.on_funding(parsed["event_ms"] or server_ms, parsed["rate"])
            await self.ipc.broadcast({"type": "markPrice", **{k: parsed[k] for k in ("symbol", "mark_price", "index_price", "rate", "next_funding_time", "event_ms")}})
        elif kind == "miniTicker":
            self._ticker = parsed
            await self.ipc.broadcast({"type": "miniTicker", "symbol": symbol, "close": parsed["close"], "quote_volume": parsed["quote_volume"], "event_ms": parsed["event_ms"]})

    async def _on_public(self, parsed: dict) -> None:
        if parsed["kind"] != "depth":
            return
        server_ms, local_ms = self.stamp()
        self.status["last_public_ms"] = server_ms
        if self._last_depth_ms is not None and server_ms - self._last_depth_ms > self.cfg.ingest.depth_gap_ms:
            self.coverage.gap("um:depth20", utc8_date(server_ms), self._last_depth_ms, server_ms, "interval")
        self._last_depth_ms = server_ms
        symbol = parsed["symbol"] or self.cfg.symbols.um
        row = {
            "recv_ms": server_ms,
            "recv_local_ms": local_ms,
            "event_ms": parsed["event_ms"],
            "transact_ms": parsed["transact_ms"],
            "update_id": parsed["update_id"],
            "bids_px": parsed["bids_px"],
            "bids_qty": parsed["bids_qty"],
            "asks_px": parsed["asks_px"],
            "asks_qty": parsed["asks_qty"],
            "source": "ws",
        }
        self.store.append("depth", "um", symbol, utc8_date(server_ms), row)
        self.coverage.add("um:depth20", utc8_date(server_ms), server_ms)
        await self.ipc.broadcast({"type": "depth", "symbol": symbol, "update_id": parsed["update_id"], "bids_px": parsed["bids_px"], "asks_px": parsed["asks_px"], "recv_ms": server_ms})

    async def _on_spot(self, parsed: dict) -> None:
        if parsed["kind"] != "kline":
            return
        server_ms, _local = self.stamp()
        self.status["last_spot_ms"] = server_ms
        closed = await self.klines.on_kline("spot", parsed)
        bar = self.klines.snapshot("spot", parsed["symbol"], parsed["interval"])
        if closed is not None:
            bar = _public_bar(closed, closed=True)
            asyncio.create_task(self._verify_later("spot", closed))
        await self.ipc.broadcast({"type": "kline", "closed": closed is not None, "bar": bar})

    async def _verify_later(self, market: str, bar) -> None:
        if await self._sleep(self.cfg.ingest.kline_verify_delay_seconds):
            return
        try:
            raw = await self.rest.klines(
                market, bar.symbol, bar.interval, start=bar.open_time, end=bar.close_time, limit=5
            )
        except Exception:
            log.warning("收盘回读失败 %s %s", market, bar.interval, exc_info=True)
            return
        server_ms, local_ms = self.stamp()
        rows = []
        for item in raw:
            parsed = kline_from_list(market, bar.symbol, bar.interval, item, "rest_verify", server_ms, local_ms)
            if parsed.open_time == bar.open_time:
                rows.append(parsed)
        if rows:
            self.db.upsert_klines(rows)

    def _flush_agg_state(self) -> None:
        if self._agg_state is None or self.db is None:
            return
        server_ms, _local = self.stamp()
        self.db.set_stream_state("um:aggTrade", self._agg_state[0], self._agg_state[1], server_ms)

    async def _flush_loop(self) -> None:
        elapsed = 0.0
        step = min(1.0, self.cfg.ingest.jsonl_fsync_seconds)
        while not self.stop.is_set():
            if await self._sleep(step):
                return
            elapsed += step
            self.store.fsync()
            self._flush_agg_state()
            if elapsed < self.cfg.ingest.parquet_flush_seconds:
                continue
            elapsed = 0.0
            paths = self.store.rotate()
            await asyncio.to_thread(self.store.convert_many, paths)
            self.coverage.flush(self.db)
            today = utc8_date(self.stamp()[0])
            await asyncio.to_thread(self.store.merge_parts, today)

    async def _metrics_loop(self) -> None:
        while not self.stop.is_set():
            try:
                await self._poll_metrics()
            except Exception as exc:
                log.warning("合约指标轮询失败: %s", exc)
            if await self._sleep(self.cfg.ingest.metrics_poll_seconds):
                return

    async def _poll_metrics(self) -> None:
        symbol = self.cfg.symbols.um
        params = {"symbol": symbol, "period": "5m", "limit": 10}
        oi_hist = await self.rest.futures_data("/futures/data/openInterestHist", params)
        top_pos = await self.rest.futures_data("/futures/data/topLongShortPositionRatio", params)
        top_acc = await self.rest.futures_data("/futures/data/topLongShortAccountRatio", params)
        global_ratio = await self.rest.futures_data("/futures/data/globalLongShortAccountRatio", params)
        taker = await self.rest.futures_data("/futures/data/takerlongshortRatio", params)
        rows: dict[int, MetricRow] = {}
        for row in metrics_from_oi_hist(oi_hist):
            rows[row.ts] = row
        for row in metrics_from_ratio(top_pos, "top_pos_lsr"):
            rows.setdefault(row.ts, MetricRow(symbol=symbol, ts=row.ts)).top_pos_lsr = row.top_pos_lsr
        for row in metrics_from_ratio(top_acc, "top_acc_lsr"):
            rows.setdefault(row.ts, MetricRow(symbol=symbol, ts=row.ts)).top_acc_lsr = row.top_acc_lsr
        for row in metrics_from_ratio(global_ratio, "global_lsr"):
            rows.setdefault(row.ts, MetricRow(symbol=symbol, ts=row.ts)).global_lsr = row.global_lsr
        for row in metrics_from_ratio(taker, "taker_lsr"):
            rows.setdefault(row.ts, MetricRow(symbol=symbol, ts=row.ts)).taker_lsr = row.taker_lsr
        self.db.upsert_metrics(list(rows.values()))
        try:
            current = await self.rest.open_interest(symbol)
        except Exception as exc:
            log.warning("当前持仓量失败: %s", exc)
            current = {}
        if current.get("openInterest") is not None:
            ts = metric_ts_from_rest(int(current.get("time") or self.stamp()[0]))
            self.anomaly.on_oi(ts, dec_str(current["openInterest"]))
        try:
            settled = await self.rest.funding_rate(symbol, limit=20)
            self.db.upsert_funding([funding_from_rest(item) for item in settled])
        except Exception as exc:
            log.warning("已结算资金费率失败: %s", exc)

    async def _mark_loop(self) -> None:
        from supere.intervals import UM_INTERVALS

        while not self.stop.is_set():
            for market in ("um_mark", "um_index", "um_premium"):
                for interval in UM_INTERVALS:
                    try:
                        await fill_recent_tail(self.rest, self.db, self.stamp, market, self.cfg.symbols.um, interval, 2)
                    except Exception as exc:
                        log.warning("标记价格 K 线失败 %s %s: %s", market, interval, exc)
                        break
            if await self._sleep(self.cfg.ingest.mark_kline_poll_seconds):
                return

    async def _disk_loop(self) -> None:
        while not self.stop.is_set():
            disk = check_disk(self.root, self.cfg.disk.warn_free_gb, self.cfg.disk.critical_free_gb)
            if disk.level != "ok":
                await self.event(
                    "warning" if disk.level == "warn" else "error",
                    "disk",
                    "DISK_" + disk.level.upper(),
                    f"数据盘剩余 {disk.free_gb:.1f} GB",
                    {"free_gb": round(disk.free_gb, 2)},
                )
            if await self._sleep(60):
                return

    async def _stats_loop(self) -> None:
        while not self.stop.is_set():
            server_ms, _local = self.stamp()
            for venue, limiter, limit in (
                ("futures", self._futures, self.cfg.rate_limit.futures_weight_per_min),
                ("spot", self._spot, self.cfg.rate_limit.spot_weight_per_min),
            ):
                minute, peak, header, halted = limiter.snapshot()
                self.db.add_runtime_stat(server_ms, venue, minute, peak, header, limit, halted)
            if await self._sleep(60):
                return

    async def _backup_loop(self) -> None:
        while not self.stop.is_set():
            now = datetime.now(UTC8)
            if now.hour == self.cfg.backup.hour_utc8:
                dest = self.root / "backups" / f"superE-{now:%Y%m%d}.sqlite"
                if not dest.exists():
                    await asyncio.to_thread(backup_sqlite, self.db.path, dest)
                    copied = await asyncio.to_thread(
                        copy_new_parquet,
                        self.root / "raw",
                        self.root / "backups" / "parquet",
                        self.root / "backups" / "parquet-manifest.json",
                    )
                    prune_old_backups(self.root / "backups", self.cfg.backup.keep_days, now)
                    await self.event("info", "backup", "BACKUP", f"已备份，新增 Parquet {copied} 个", {"path": str(dest)})
            if await self._sleep(600):
                return

    async def _status_loop(self) -> None:
        while not self.stop.is_set():
            server_ms, _local = self.stamp()
            minute, peak, header, halted = self._futures.snapshot()
            payload = {
                "type": "status",
                "ts": server_ms,
                "sockets": dict(self.status),
                "clock_offset_ms": self.clock.offset_ms if self.clock.calibrated else None,
                "clock_warn": self.clock.warn,
                "rest_halted": halted,
                "self_weight_minute": minute,
                "self_weight_peak": peak,
                "header_weight": header,
                "proxy": self.cfg.proxy.url,
                "ticker": self._ticker["close"] if self._ticker else None,
            }
            await self.ipc.broadcast(payload)
            if await self._sleep(2):
                return

    async def _news_loop(self) -> None:
        enabled = set(self.cfg.news.enabled_sources)
        if not enabled:
            return
        registry = default_registry()
        proxy = proxy_or_none(self.cfg.proxy.url)
        async with httpx.AsyncClient(proxy=proxy, timeout=30.0, follow_redirects=True) as client:
            while not self.stop.is_set():
                for name in enabled:
                    collector = registry.get(name)
                    if collector is None:
                        await self.event("warning", "news", "NEWS_UNKNOWN", f"未知消息源 {name}", {})
                        continue
                    try:
                        if name == "alternative_me" and hasattr(collector, "fetch_points"):
                            points = await collector.fetch_points(client, limit=1)
                            server_ms, local_ms = self.stamp()
                            for point in points:
                                self.db.upsert_fear_greed(
                                    point.ts_ms,
                                    point.value,
                                    point.classification,
                                    server_ms,
                                    local_ms,
                                    point.known_at_ms,
                                )
                        items = await collector.collect(client)
                        server_ms, local_ms = self.stamp()
                        for item in items:
                            persist_news(self.db, item, server_ms, local_ms)
                    except Exception as exc:
                        await self.event("warning", "news", "NEWS_FAIL", f"{name} 采集失败: {exc}", {})
                if await self._sleep(self.cfg.news.fear_greed_poll_seconds):
                    return


def _public_bar(bar, *, closed: bool) -> dict:
    return {
        "market": bar.market,
        "symbol": bar.symbol,
        "interval": bar.interval,
        "open_time": bar.open_time,
        "close_time": bar.close_time,
        "o": bar.o,
        "h": bar.h,
        "l": bar.l,
        "c": bar.c,
        "volume": bar.volume,
        "quote_volume": bar.quote_volume,
        "trades": bar.trades,
        "closed": closed,
    }
