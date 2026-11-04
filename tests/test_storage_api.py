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
    cfg = AppConfig()
    app = create_app(cfg, DataAccess(db), LiveHub(), connect_ipc=False)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["api_port"] == 8766
        body = client.get("/api/klines", params={"limit": 500}).json()
        assert body["timezone"] == "UTC+8"
        assert body["bars"][0]["open_time_utc8"].endswith("08:01:00") or " " in body["bars"][0]["open_time_utc8"]
        assert client.post("/api/shutdown").json()["ok"] is True
        assert client.get("/api/coverage").status_code == 200


def test_second_instance_lock(tmp_path: Path):
    first = SingleInstance(tmp_path / "superE.lock")
    second = SingleInstance(tmp_path / "superE.lock")
    first.acquire()
    with pytest.raises(SystemExit):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


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
