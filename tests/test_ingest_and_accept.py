from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from websockets.asyncio.server import serve

from supere.acceptance import offline_checks
from supere.binance.rest import BinanceRest
from supere.config import AppConfig
from supere.ingest.backfill import Backfill
from supere.ingest.live import Ingestor
from supere.ratelimit import WeightLimiter
from supere.storage.db import Database


def _zip(csv_text: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ETHUSDT-1m-2024-01.csv", csv_text)
    blob = buffer.getvalue()
    return blob, hashlib.sha256(blob).hexdigest()


async def test_backfill_imports_archive_and_dedups(tmp_path: Path):
    csv = "1704067200000,1,2,0.5,1.5,10,1704070799999,15,4,3,4,0\n"
    blob, digest = _zip(csv)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/ETHUSDT-1m-2024-01.zip"):
            return httpx.Response(200, content=blob)
        if url.endswith("/ETHUSDT-1m-2024-01.zip.CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  ETHUSDT-1m-2024-01.zip\n")
        if request.url.path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": 1_700_000_000_000})
        if "klines" in request.url.path.lower() or request.url.path.endswith("fundingRate"):
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    cfg = AppConfig()
    cfg.data.dir = str(tmp_path)
    cfg.proxy.url = ""
    cfg._archive_dir = tmp_path / "archive_cache"
    db = Database(tmp_path / "superE.sqlite")
    rest = BinanceRest(
        cfg,
        WeightLimiter("futures", 2400),
        WeightLimiter("spot", 6000),
        WeightLimiter("futures_data", 1000, window_s=300),
        transport=httpx.MockTransport(handler),
    )

    async def on_event(*_args, **_kwargs):
        return None

    # A clock after the January 2024 archive, so the tail is a real closed range.
    job = Backfill(
        cfg,
        rest,
        db,
        lambda: (1_736_000_000_000, 1_736_000_000_000),
        on_event,
        months=[(2024, 1)],
        days=[],
    )
    await job.run()
    await job.run()
    assert db.count_klines("um") == 1
    row = db.fetch_klines("um", "ETHUSDT", "1m")[0]
    assert row["source"] == "archive"
    assert row["o"] == "1"
    await rest.aclose()
    db.close()


def _live_config(tmp_path: Path, port: int) -> AppConfig:
    cfg = AppConfig()
    cfg.data.dir = str(tmp_path / "data")
    cfg.proxy.url = ""
    cfg.server.ipc_port = 0
    base = f"ws://127.0.0.1:{port}"
    cfg.endpoints.fstream_market = base + "/market"
    cfg.endpoints.fstream_public = base + "/public"
    cfg.endpoints.spot_ws = base + "/spot"
    cfg.endpoints.fapi = "https://fapi.test"
    cfg.endpoints.spot_rest = "https://api.test"
    cfg.endpoints.spot_rest_fallback = "https://data.test"
    cfg.ingest.reconnect_hours = 24
    cfg.ingest.parquet_flush_seconds = 0.2
    cfg.ingest.jsonl_fsync_seconds = 0.05
    cfg.ingest.kline_verify_delay_seconds = 3600
    cfg.ingest.metrics_poll_seconds = 3600
    cfg.ingest.mark_kline_poll_seconds = 3600
    cfg.ingest.startup_tail_limit = 2
    cfg.clock.samples = 1
    cfg.clock.recalibrate_seconds = 3600
    cfg.backup.hour_utc8 = 25
    return cfg


async def _handler(ws):
    raw = json.loads(await ws.recv())
    await ws.send(json.dumps({"result": None, "id": raw.get("id", 1)}))
    params = raw.get("params") or []
    joined = " ".join(params)
    if "depth20" in joined:
        await ws.send(
            json.dumps(
                {
                    "e": "depthUpdate",
                    "E": 1_700_000_100_000,
                    "T": 1_700_000_100_000,
                    "s": "ETHUSDT",
                    "u": 50,
                    "b": [["2000", "1"]],
                    "a": [["2001", "2"]],
                }
            )
        )
    elif len(params) == 1:
        await ws.send(
            json.dumps(
                {
                    "e": "kline",
                    "E": 1_700_000_060_000,
                    "s": "ETHUSDT",
                    "k": {
                        "t": 1_700_000_040_000,
                        "T": 1_700_000_099_999,
                        "s": "ETHUSDT",
                        "i": "1m",
                        "o": "10",
                        "c": "11",
                        "h": "12",
                        "l": "9",
                        "v": "3",
                        "n": 2,
                        "x": True,
                        "q": "30",
                        "V": "1",
                        "Q": "10",
                        "B": "0",
                    },
                }
            )
        )
    else:
        open_time = 1_700_000_040_000
        await ws.send(
            json.dumps(
                {
                    "e": "kline",
                    "E": 1_700_000_060_000,
                    "s": "ETHUSDT",
                    "k": {
                        "t": open_time,
                        "T": open_time + 59_999,
                        "s": "ETHUSDT",
                        "i": "1m",
                        "o": "100",
                        "c": "110",
                        "h": "120",
                        "l": "90",
                        "v": "5",
                        "n": 4,
                        "x": True,
                        "q": "500",
                        "V": "2",
                        "Q": "200",
                        "B": "0",
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "e": "aggTrade",
                    "E": 1_700_000_060_100,
                    "s": "ETHUSDT",
                    "a": 42,
                    "p": "111",
                    "q": "0.1",
                    "f": 1,
                    "l": 1,
                    "T": 1_700_000_060_100,
                    "m": False,
                }
            )
        )
    try:
        await ws.wait_closed()
    except Exception:
        return


async def test_live_socket_writes_sqlite_and_parquet(tmp_path: Path):
    server = await serve(_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = _live_config(tmp_path, port)
    config_path = tmp_path / "config.toml"
    config_path.write_text("proxy = {}\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": 1_700_000_200_000})
        if "klines" in request.url.path.lower():
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"msg": "no"})

    ing = Ingestor(cfg, config_path, transport=httpx.MockTransport(handler))
    task = asyncio.create_task(ing.run(live=True))
    db_path = tmp_path / "data" / "superE.sqlite"
    found = False
    try:
        for _ in range(80):
            await asyncio.sleep(0.1)
            if not db_path.exists():
                continue
            probe = Database(db_path, read_only=True)
            try:
                found = probe.count_klines("um") >= 1
            finally:
                probe.close()
            if found:
                break
            if task.done() and task.exception() is not None:
                raise task.exception()
        assert found, "采集器没有把收盘 K 线写入 SQLite"
        await asyncio.sleep(0.6)
        depth_files = list((tmp_path / "data" / "raw").rglob("depth/**/*.parquet"))
        agg_files = list((tmp_path / "data" / "raw").rglob("aggTrades/**/*.parquet"))
        assert depth_files, "depth20 没有落成 Parquet"
        assert agg_files, "aggTrade 没有落成 Parquet"
        probe = Database(db_path, read_only=True)
        try:
            clocks = probe.count_events("CLOCK_SYNC")
            assert clocks >= 1
        finally:
            probe.close()
    finally:
        ing.stop.set()
        await asyncio.wait_for(task, timeout=10)
        server.close()
        await server.wait_closed()


async def test_live_socket_stops_cleanly(tmp_path: Path):
    server = await serve(_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = _live_config(tmp_path, port)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[proxy]\nurl=''\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/time"):
            return httpx.Response(200, json={"serverTime": 1_700_000_200_000})
        if "klines" in request.url.path.lower():
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    ing = Ingestor(cfg, config_path, transport=httpx.MockTransport(handler))
    task = asyncio.create_task(ing.run(live=True))
    await asyncio.sleep(0.5)
    ing.stop.set()
    await asyncio.wait_for(task, timeout=10)
    server.close()
    await server.wait_closed()
    assert not (tmp_path / "data" / "run" / "ingestor.pid").exists()


def test_offline_acceptance(tmp_path: Path):
    checks = offline_checks(tmp_path)
    failed = [item for item in checks if item.status == "FAIL"]
    assert not failed, [item.line() for item in failed]
    assert any(item.code == "A1" and item.status == "PASS" for item in checks)
    assert any(item.code == "A6" and item.status == "PASS" for item in checks)


async def test_pause_proxy_stops_bytes():
    from supere.forwarder import PauseProxy

    async def echo(reader, writer):
        data = await reader.read(1024)
        writer.write(data)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    upstream = server.sockets[0].getsockname()[1]
    proxy = PauseProxy("127.0.0.1", upstream, listen_port=0)
    await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.listen_port)
        writer.write(b"ping")
        await writer.drain()
        assert await reader.read(4) == b"ping"
        writer.close()
        await proxy.pause()
        with pytest.raises(Exception):
            await asyncio.wait_for(asyncio.open_connection("127.0.0.1", proxy.listen_port), timeout=1)
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()
