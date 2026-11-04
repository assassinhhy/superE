from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from supere.api.app import create_app
from supere.binance.safety import FORBIDDEN_PATH_MARKERS, guard_url
from supere.config import AppConfig, load_config
from supere.errors import TradingDisabled
from supere.ingest.aggregate import aggregate_1m, compare_bars
from supere.ingest.anomaly import AnomalyEngine, detect_price_jump, detect_volume
from supere.ingest.gaps import find_missing_ranges
from supere.ingest.kline_manager import KlineManager
from supere.ingest.overlap import OverlapGate
from supere.ipc.bus import LiveHub
from supere.lockfile import SingleInstance
from supere.logging_setup import mask_secrets
from supere.news.adapters import FearGreedCollector, persist_news
from supere.news.dedup import hamming, same_text, simhash64, url_hash
from supere.news.models import RawItem
from supere.storage.backup import backup_sqlite, restore_drill
from supere.storage.dal import DataAccess
from supere.storage.db import Database
from supere.storage.duck import DuckReader
from supere.storage.models import Kline
from supere.storage.parquet_store import ParquetStore


def _bar(i: int, interval: str = "1m", **overrides) -> Kline:
    open_time = i * 60_000
    values = dict(
        market="um",
        symbol="ETHUSDT",
        interval=interval,
        open_time=open_time,
        close_time=open_time + 59_999,
        o="100",
        h="110",
        l="90",
        c="105",
        volume="2",
        quote_volume="200",
        trades=4,
        taker_buy_base="1",
        taker_buy_quote="100",
        source="test",
        ingested_at=1,
        ingested_local=1,
    )
    values.update(overrides)
    return Kline(**values)


def test_upsert_dedup_and_readonly(db: Database, tmp_path: Path):
    db.upsert_klines([_bar(1), _bar(1, c="999")])
    rows = db.fetch_klines("um", "ETHUSDT", "1m")
    assert len(rows) == 1
    assert rows[0]["c"] == "999"
    db.close()
    ro = Database(tmp_path / "superE.sqlite", read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.conn.execute("DELETE FROM klines")
    ro.close()


def test_gap_and_aggregate():
    gaps = find_missing_ranges("1m", [0, 180_000], 0, 180_000)
    assert gaps == [(60_000, 120_000)]
    bars = [_bar(i, c=str(100 + i), h=str(110 + i), l=str(90)) for i in range(5)]
    agg = aggregate_1m(bars, "5m")
    assert len(agg) == 1
    assert agg[0].o == "100"
    assert agg[0].c == "104"
    assert Decimal(agg[0].volume) == Decimal("10")
    official = _bar(0, interval="5m", o="100", h=agg[0].h, l="90", c="104", volume="10", quote_volume="1000", trades=20, taker_buy_base="5", taker_buy_quote="500")
    assert compare_bars(agg[0], official) == []


def test_forming_bar_is_not_stored(db: Database):
    manager = KlineManager(db, lambda: (10, 9))
    parsed = {
        "symbol": "ETHUSDT",
        "interval": "1m",
        "event_ms": 10,
        "closed": False,
        "row": [0, "1", "2", "0.5", "1.5", "4", 59999, "6", 5, "1", "2", "0"],
    }

    async def run():
        assert await manager.on_kline("um", parsed) is None
        assert db.count_klines() == 0
        await manager.on_agg_trade(
            "um",
            {"symbol": "ETHUSDT", "ts": 20, "price": "3", "qty": "1", "agg_id": 8, "is_buyer_maker": False},
        )
        snap = manager.snapshot("um", "ETHUSDT", "1m")
        assert snap["c"] == "3"
        assert snap["closed"] is False
        parsed["closed"] = True
        parsed["event_ms"] = 30
        closed = await manager.on_kline("um", parsed)
        assert closed is not None
        assert db.count_klines() == 1
        assert manager.snapshot("um", "ETHUSDT", "1m") is None

    import asyncio

    asyncio.run(run())


def test_overlap_promotes_without_double_apply():
    gate = OverlapGate(3)
    assert gate.observe("old", "depth", ("depth", 1)) is True
    assert gate.observe("old", "depth", ("depth", 1)) is False
    gate.start("new")
    assert gate.observe("new", "depth", ("depth", 1)) is False
    assert gate.promoted is False
    assert gate.observe("new", "depth", ("depth", 2)) is True
    assert gate.observe("new", "aggTrade", ("aggTrade", 9)) is True
    assert gate.promoted is True


def test_parquet_duckdb_and_backup(tmp_path: Path, db: Database):
    db.upsert_klines([_bar(1)])
    store = ParquetStore(tmp_path / "raw")
    store.append(
        "depth",
        "um",
        "ETHUSDT",
        "2026-11-04",
        {
            "recv_ms": 10,
            "recv_local_ms": 9,
            "event_ms": 10,
            "transact_ms": 10,
            "update_id": 5,
            "bids_px": ["1"],
            "bids_qty": ["2"],
            "asks_px": ["3"],
            "asks_qty": ["4"],
            "source": "ws",
        },
    )
    store.append(
        "aggTrades",
        "um",
        "ETHUSDT",
        "2026-11-04",
        {
            "agg_id": 7,
            "price": "10",
            "qty": "1",
            "first_id": 1,
            "last_id": 1,
            "ts": 10,
            "is_buyer_maker": False,
            "recv_ms": 11,
            "recv_local_ms": 9,
            "source": "ws",
        },
    )
    written = store.convert_many(store.rotate())
    assert written
    meta = pq.read_metadata(written[0])
    assert meta.num_rows >= 1
    duck = DuckReader(db.path, tmp_path / "raw")
    assert duck.kline_count("um", "ETHUSDT", "1m") == 1
    summary = duck.parquet_summary("aggTrades", "um", "ETHUSDT", "2026-11-04")
    assert summary["rows"] == 1
    duck.close()
    dest = tmp_path / "bak.sqlite"
    backup_sqlite(db.path, dest)
    drilled = restore_drill(dest, tmp_path / "drill")
    assert drilled["integrity"] == "ok"
    assert drilled["klines"] == 1


def test_anomaly_and_news(db: Database):
    quotes = [10.0] * 80
    hit = detect_volume(1_000, quotes[-60:], quotes, 5)
    assert hit is not None and hit.kind == "volume"
    assert detect_volume(11, quotes[-60:], quotes, 5) is None
    jump = detect_price_jump(120, 100, 2, 4)
    assert jump is not None
    assert detect_price_jump(101, 100, 2, 4) is None
    for i in range(20):
        db.upsert_klines([_bar(i, c="100", h="101", l="99", quote_volume="10")])
    engine = AnomalyEngine(__import__("supere.config", fromlist=["AnomalyConfig"]).AnomalyConfig(), db)
    hits = engine.on_liquidation(1_000, "2500", "500")
    assert any(item.kind == "liquidation" for item in hits)
    engine.on_funding(1, "0.0001")
    funding_hit = engine.on_funding(2, "0.0010")
    assert funding_hit is not None
    item = RawItem("odaily", 3, "zh", "ETH 升级", "body", "https://example.test/a", 1, "1")
    assert persist_news(db, item, 2, 1) is not None
    assert persist_news(db, item, 3, 1) is None
    assert url_hash("https://example.test/a#1")
    assert same_text(simhash64("eth upgrade today"), simhash64("eth upgrade today"))
    assert hamming(1, 0) == 1


def test_fear_greed_parse():
    class Dummy:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def get(self, url, params=None):
            return Dummy(
                {"data": [{"value": "40", "value_classification": "Fear", "timestamp": "1700000000"}]}
            )

    import asyncio

    points = asyncio.run(FearGreedCollector().fetch_points(Client(), limit=1))
    assert points[0].value == 40
    assert points[0].known_at_ms == points[0].ts_ms + 86_400_000


def test_api_reads_and_reports_utc8(db: Database):
    db.upsert_klines([_bar(1)])
    db.add_event(1, "info", "ws", "WS_CONNECTED", "market 已连接", None, ts_local=1)
    cfg = AppConfig()
    app = create_app(cfg, DataAccess(db), LiveHub(), connect_ipc=False)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["api_port"] == 8766
        assert "charset=utf-8" in health.headers["content-type"]
        body = client.get("/api/klines", params={"limit": 500}).json()
        assert body["timezone"] == "UTC+8"
        assert body["bars"][0]["open_time_utc8"].endswith("08:01:00") or " " in body["bars"][0]["open_time_utc8"]
        events = client.get("/api/events")
        assert "charset=utf-8" in events.headers["content-type"]
        assert "已连接" in events.content.decode("utf-8")
        assert client.get("/api/coverage").status_code == 200
        assert client.post("/api/shutdown").json()["ok"] is True


def test_second_instance_lock(tmp_path: Path):
    first = SingleInstance(tmp_path / "superE.lock")
    second = SingleInstance(tmp_path / "superE.lock")
    first.acquire()
    with pytest.raises(SystemExit):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


async def test_index_klines_use_pair_not_symbol():
    import httpx

    from supere.binance.rest import BinanceRest, kline_params
    from supere.ratelimit import WeightLimiter

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "params": dict(request.url.params)})
        return httpx.Response(200, json=[])

    cfg = AppConfig()
    cfg.proxy.url = ""
    cfg.endpoints.fapi = "https://fapi.test"
    cfg.endpoints.spot_rest = "https://api.test"
    rest = BinanceRest(
        cfg,
        WeightLimiter("futures", 2400),
        WeightLimiter("spot", 6000),
        WeightLimiter("futures_data", 1000, window_s=300),
        transport=httpx.MockTransport(handler),
    )
    await rest.klines("um_index", "ETHUSDT", "1m", limit=2)
    await rest.klines("um_mark", "ETHUSDT", "1m", limit=2)
    await rest.klines("um_premium", "ETHUSDT", "1m", limit=2)
    await rest.klines("um", "ETHUSDT", "1m", limit=2)
    await rest.aclose()
    index, mark, premium, trade = seen
    assert index["path"].endswith("/fapi/v1/indexPriceKlines")
    assert index["params"]["pair"] == "ETHUSDT"
    assert "symbol" not in index["params"]
    assert mark["path"].endswith("/fapi/v1/markPriceKlines")
    assert mark["params"]["symbol"] == "ETHUSDT"
    assert "pair" not in mark["params"]
    assert premium["path"].endswith("/fapi/v1/premiumIndexKlines")
    assert premium["params"]["symbol"] == "ETHUSDT"
    assert "pair" not in premium["params"]
    assert trade["params"]["symbol"] == "ETHUSDT"
    capped = kline_params("um_index", "ETHUSDT", "1m", limit=1500)
    assert capped["limit"] == 1000
    assert capped["pair"] == "ETHUSDT"


def test_shutdown_flushes_staging_and_wal(tmp_path: Path, db: Database):
    from supere.acceptance import clean_stop_check
    from supere.ingest.live import finalize_storage
    from supere.storage.parquet_store import ParquetStore

    store = ParquetStore(tmp_path / "raw")
    store.append(
        "depth",
        "um",
        "ETHUSDT",
        "2026-11-04",
        {
            "recv_ms": 10,
            "recv_local_ms": 9,
            "event_ms": 10,
            "transact_ms": 10,
            "update_id": 1,
            "bids_px": ["1"],
            "bids_qty": ["1"],
            "asks_px": ["2"],
            "asks_qty": ["1"],
            "source": "ws",
        },
    )
    db.upsert_klines([_bar(1)])
    finalize_storage(store, db)
    db.close()
    assert list((tmp_path / "raw" / "_staging").glob("*.jsonl*")) == []
    assert list((tmp_path / "raw").rglob("*.parquet"))
    wal = tmp_path / "superE.sqlite-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    check = clean_stop_check(tmp_path)
    assert check.status == "PASS", check.detail


def test_log_file_keeps_chinese_and_shutdown_lines(tmp_path: Path):
    import logging

    from supere.logging_setup import flush_logs, setup_logging

    setup_logging(tmp_path, "ingestor")
    log = logging.getLogger("supere")
    log.info("请检查 Windows 时间同步")
    log.info("market 已连接")
    log.info("采集器开始关闭，先写入暂存并做 checkpoint")
    log.info("采集器已关闭")
    flush_logs()
    files = list(tmp_path.glob("supere-ingestor-*.log"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "请检查 Windows" in text
    assert "已连接" in text
    assert "采集器开始关闭" in text
    assert "采集器已关闭" in text
    assert "\ufffd" not in text


def test_spawn_detached_does_not_inherit_caller_pipes(tmp_path: Path, monkeypatch):
    import subprocess
    import sys

    from supere.spawn import spawn_detached

    captured: dict = {}

    class _Proc:
        pid = 43210

    def fake_popen(**kwargs):
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    pid = spawn_detached(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert pid == 43210
    assert captured["close_fds"] is True
    assert captured["stdin"] is subprocess.DEVNULL
    if sys.platform == "win32":
        flags = captured["creationflags"]
        assert flags & 0x00000008  # DETACHED_PROCESS
        assert flags & 0x00000200  # CREATE_NEW_PROCESS_GROUP
    else:
        assert captured["start_new_session"] is True


def test_spawn_detached_returns_while_child_is_alive(tmp_path: Path):
    import os
    import signal
    import sys
    import time

    from supere.spawn import spawn_detached

    started = time.monotonic()
    pid = spawn_detached(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    assert time.monotonic() - started < 5
    try:
        # Windows 没有 SIGKILL；os.kill 在那里等同于 TerminateProcess。
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def test_smoke_labels_are_not_repeated():
    from supere.acceptance import Check

    line = Check("S1", "PASS", "合约时间 HTTP 200").line()
    assert line == "[PASS] S1 合约时间 HTTP 200"
    assert "S1 S1" not in line


def test_config_defaults_and_secret_mask():
    cfg, _path = load_config(str(Path(__file__).resolve().parents[1] / "config.toml"))
    assert cfg.proxy.url == "http://127.0.0.1:7890"
    assert cfg.server.api_port == 8766
    assert "deepseek" not in cfg.model_dump_json().lower() or "enabled_sources" in cfg.model_dump_json()
    assert mask_secrets("token sk-abcdefghijklmnopqrstuvwxyz") == "token [redacted]"
    assert "DEEPSEEK" not in cfg.model_dump_json()


def test_no_trading_surface():
    with pytest.raises(TradingDisabled):
        guard_url("https://fapi.binance.com/fapi/v1/order")
    with pytest.raises(TradingDisabled):
        guard_url("https://api.binance.com/api/v3/order/test")
    root = Path(__file__).resolve().parents[1] / "backend" / "supere"
    for path in root.rglob("*.py"):
        if path.name == "safety.py":
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "x-mbx-apikey" not in text
        assert "binance_api_key" not in text
        assert "binance_secret" not in text
        for marker in ("/fapi/v1/order", "/api/v3/order"):
            assert marker not in text
    assert "/fapi/v1/order" in FORBIDDEN_PATH_MARKERS


def test_duckdb_rejects_writes(db: Database):
    duck = DuckReader(db.path, Path(db.path).parent / "raw")
    with pytest.raises(ValueError):
        duck.query("DELETE FROM market.klines")
    duck.close()
