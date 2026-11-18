"""Read-only API and preview book used by the 1b UI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from supere.api.app import create_app
from supere.config import AppConfig
from supere.desktop import origin_of
from supere.ingest.kline_manager import KlineManager
from supere.ipc.bus import LiveHub
from supere.preview import load_preview_hub, seed_preview
from supere.storage.dal import DataAccess
from supere.storage.db import Database
from supere.storage.duck import DuckReader
from supere.storage.parquet_store import ParquetStore


def test_metrics_and_funding_accept_the_chart_page_size(db: Database):
    from supere.storage.models import FundingRow, MetricRow

    db.upsert_metrics(
        [MetricRow(symbol="ETHUSDT", ts=1_700_000_000_000, oi="123", global_lsr="1.2", source="test")]
    )
    db.upsert_funding([FundingRow("ETHUSDT", 1_700_000_000_000, "0.0001", "3500", "test")])
    app = create_app(AppConfig(), DataAccess(db), LiveHub(), connect_ipc=False)
    with TestClient(app) as client:
        metrics = client.get("/api/metrics", params={"symbol": "ETHUSDT", "limit": 1000})
        funding = client.get("/api/funding", params={"symbol": "ETHUSDT", "limit": 1000})
    assert metrics.status_code == 200
    assert metrics.json()["rows"][0]["oi"] == "123"
    assert metrics.json()["rows"][0]["global_lsr"] == "1.2"
    assert funding.status_code == 200
    assert funding.json()["settled"][0]["rate"] == "0.0001"


def test_metrics_and_funding_page_backwards_with_end(db: Database):
    from supere.storage.models import FundingRow, MetricRow

    base = 1_700_000_000_000
    step = 300_000
    db.upsert_metrics(
        [MetricRow(symbol="ETHUSDT", ts=base + k * step, oi=str(k), source="test") for k in range(6)]
    )
    eight_h = 28_800_000
    db.upsert_funding(
        [FundingRow("ETHUSDT", base + k * eight_h, str(k), "3500", "test") for k in range(6)]
    )
    app = create_app(AppConfig(), DataAccess(db), LiveHub(), connect_ipc=False)
    with TestClient(app) as client:
        # Newest page with no window: the last `limit` rows, ascending.
        newest = client.get("/api/metrics", params={"symbol": "ETHUSDT", "limit": 2}).json()["rows"]
        assert [int(r["ts"]) for r in newest] == [base + 4 * step, base + 5 * step]
        # `end` anchors the window: the two rows at or before base+3*step, nothing newer.
        window = client.get(
            "/api/metrics",
            params={"symbol": "ETHUSDT", "end": base + 3 * step, "limit": 2},
        ).json()["rows"]
        assert [int(r["ts"]) for r in window] == [base + 2 * step, base + 3 * step]
        # Paging back: the next page ends just before the oldest row we already have.
        older = client.get(
            "/api/metrics",
            params={"symbol": "ETHUSDT", "end": base + 2 * step - 1, "limit": 2},
        ).json()["rows"]
        assert [int(r["ts"]) for r in older] == [base, base + step]
        # start and end together bound both sides.
        bounded = client.get(
            "/api/metrics",
            params={"symbol": "ETHUSDT", "start": base + step, "end": base + 2 * step, "limit": 10},
        ).json()["rows"]
        assert [int(r["ts"]) for r in bounded] == [base + step, base + 2 * step]
        # Funding pages the same way.
        fund_window = client.get(
            "/api/funding",
            params={"symbol": "ETHUSDT", "end": base + 3 * eight_h, "limit": 2},
        ).json()["settled"]
        assert [int(r["funding_time"]) for r in fund_window] == [base + 2 * eight_h, base + 3 * eight_h]
        fund_older = client.get(
            "/api/funding",
            params={"symbol": "ETHUSDT", "end": base + 2 * eight_h - 1, "limit": 2},
        ).json()["settled"]
        assert [int(r["funding_time"]) for r in fund_older] == [base, base + eight_h]


def test_origin_stays_on_loopback():
    assert origin_of(AppConfig()) == "http://127.0.0.1:8766"


def test_hub_keeps_trades_depth_and_closed_bars():
    hub = LiveHub()
    hub.apply(
        {
            "type": "aggTrade",
            "symbol": "ETHUSDT",
            "price": "3500",
            "qty": "0.2",
            "ts": 10,
            "agg_id": 3,
            "is_buyer_maker": True,
        }
    )
    hub.apply(
        {
            "type": "depth",
            "bids_px": ["1"],
            "bids_qty": ["2"],
            "asks_px": ["3"],
            "asks_qty": ["4"],
            "update_id": 8,
        }
    )
    hub.apply(
        {
            "type": "kline",
            "closed": True,
            "bar": {"market": "um", "symbol": "ETHUSDT", "interval": "1m", "open_time": 0, "c": "9"},
        }
    )
    hub.apply(
        {
            "type": "kline",
            "closed": False,
            "bar": {"market": "um", "symbol": "ETHUSDT", "interval": "1m", "open_time": 60_000, "c": "10"},
        }
    )
    assert hub.trades[0]["is_buyer_maker"] is True
    assert hub.depth["bids_qty"] == ["2"]
    assert hub.closed[0]["c"] == "9"
    assert hub.forming[0]["c"] == "10"


def test_snapshot_exposes_taker_for_cvd(db: Database):
    manager = KlineManager(db, lambda: (10, 9))

    async def run():
        parsed = {
            "symbol": "ETHUSDT",
            "interval": "1m",
            "event_ms": 10,
            "closed": False,
            "row": [0, "1", "2", "0.5", "1.5", "4", 59999, "6", 5, "1.5", "2", "0"],
        }
        await manager.on_kline("um", parsed)
        snap = manager.snapshot("um", "ETHUSDT", "1m")
        assert snap["taker_buy_base"] == "1.5"
        assert snap["closed"] is False

    asyncio.run(run())


def test_preview_book_and_readonly_ui(tmp_path: Path):
    writer = Database(tmp_path / "superE.sqlite", read_only=False)
    summary = seed_preview(writer, days=2)
    writer.close()
    db = Database(tmp_path / "superE.sqlite", read_only=True)
    hub = LiveHub()
    cfg = AppConfig()
    load_preview_hub(hub, db, cfg, summary)
    app = create_app(cfg, DataAccess(db, DuckReader(db.path, tmp_path / "raw")), hub, connect_ipc=False, data_path=tmp_path)
    app.state.preview = True
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["preview"] is True
        assert health["timezone"] == "UTC+8"
        page = client.get("/")
        assert page.status_code == 200
        assert "pnpm build" in page.text
        bars = client.get("/api/klines", params={"interval": "5m", "limit": 10, "include_forming": 1}).json()
        assert bars["timezone"] == "UTC+8"
        assert bars["bars"][-1]["closed"] is False
        older = client.get(
            "/api/klines",
            params={"interval": "1m", "end": bars["bars"][0]["open_time"] - 1, "limit": 5},
        ).json()
        assert older["bars"]
        assert older["bars"][-1]["open_time"] < bars["bars"][0]["open_time"]
        check = client.get("/api/checks/aggregate", params={"interval": "5m", "bars": 3}).json()
        assert check["ok"] is True, check
        assert check["compared"] == 3
        catalog = client.get("/api/catalog").json()
        assert any(item["since"] == "2021-12-01" for item in catalog["floors"])
        assert any(item["id"] == "backfill" for item in catalog["commands"])
        settings = client.get("/api/settings").json()
        assert settings["editable"] is False
        assert settings["fees"]["taker"] == 0.0005
        assert settings["proxy"] == "http://127.0.0.1:7890"
        assert "key" not in settings["note"].lower() or "密钥" in settings["note"]
        signals = client.get("/api/signals").json()
        assert signals["rows"] == []
        assert signals["phase"] == 2
        metrics = client.get("/api/metrics", params={"start": summary["start_ms"], "limit": 5}).json()
        assert metrics["rows"][0]["known_at"] == metrics["rows"][0]["ts"] + 300_000
        anomalies = client.get("/api/anomalies").json()["rows"]
        assert anomalies
        assert anomalies[0]["ts_utc8"].startswith("2026-")
        status = client.get("/api/status").json()
        assert status["sockets"]["market"] == "preview"
        assert status["self_weight_minute"] == 12
        with client.websocket_connect("/ws/live") as ws:
            frame = ws.receive_json()
        assert frame["depth"]["asks_qty"]
        assert frame["trades"]
        assert any(bar["interval"] == "15m" for bar in frame["forming"])
    db.close()


def test_liquidations_read_parquet_and_hub(tmp_path: Path, db: Database):
    from supere.timeutil import utc8_date

    event_ms = 1_700_000_000_000
    store = ParquetStore(tmp_path / "raw")
    store.append(
        "forceOrder",
        "um",
        "ETHUSDT",
        utc8_date(event_ms),
        {
            "event_ms": event_ms,
            "recv_ms": event_ms,
            "recv_local_ms": event_ms,
            "trade_time": event_ms,
            "symbol": "ETHUSDT",
            "side": "SELL",
            "order_type": "LIMIT",
            "tif": "IOC",
            "qty": "2",
            "price": "3500",
            "avg_price": "3500",
            "last_qty": "2",
            "accum_qty": "2",
            "status": "FILLED",
            "source": "ws",
        },
    )
    store.convert_many(store.rotate())
    hub = LiveHub()
    hub.apply(
        {
            "type": "forceOrder",
            "event_ms": event_ms + 500,
            "price": "3501",
            "qty": "1",
            "side": "BUY",
        }
    )
    duck = DuckReader(db.path, tmp_path / "raw")
    app = create_app(AppConfig(), DataAccess(db, duck), hub, connect_ipc=False)
    with TestClient(app) as client:
        body = client.get(
            "/api/liquidations",
            params={"start": event_ms, "end": event_ms + 500},
        ).json()
    assert len(body["rows"]) == 2
    assert body["rows"][0]["qty"] == "2"
    duck.close()


def test_ui_does_not_invent_signals():
    root = Path(__file__).resolve().parents[1] / "frontend" / "src"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*") if path.suffix in {".vue", ".js", ".mjs"})
    assert "第二阶段" in text
    assert "0.58" not in text
    assert "偏涨" not in text
    # 允许的写请求只有：用户点「预测」时的 POST /api/predict、手动刷新消息的
    # POST /api/news/refresh、以及盘面复盘备注 POST /api/board/plans/.../note。
    lines = text.splitlines()
    post_contexts = [
        "\n".join(lines[max(0, idx - 2) : min(len(lines), idx + 3)])
        for idx, line in enumerate(lines)
        if 'method: "POST"' in line
    ]
    assert post_contexts, "预期存在点击预测触发的 POST"
    allowed = ("/api/predict", "/api/news/refresh", "/api/board/plans/")
    assert all(any(path in ctx for path in allowed) for ctx in post_contexts), post_contexts
    assert "/fapi/" not in text
