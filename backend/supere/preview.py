"""Synthetic market book for the UI when Binance cannot be reached.

The writer connection is closed before the API starts. The API opens the
file read-only. Forming candles and the tape live in memory only.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import uvicorn

from supere.api.app import create_app
from supere.config import AppConfig, data_dir
from supere.desktop import origin_of, run_window, wait_for_api
from supere.ingest.aggregate import aggregate_1m
from supere.intervals import UM_INTERVALS, close_time_of, next_open
from supere.ipc.bus import LiveHub
from supere.storage.dal import DataAccess
from supere.storage.db import Database
from supere.storage.duck import DuckReader
from supere.storage.models import FundingRow, Kline, MetricRow


def _text(value: Decimal) -> str:
    return format(value, "f")


def _shift(bar: Kline, market: str, delta: Decimal | None, flat: str | None) -> Kline:
    if flat is not None:
        prices = (flat, flat, flat, flat)
        volume = quote = taker = taker_quote = "0"
        trades = 0
    else:
        assert delta is not None
        prices = tuple(_text(Decimal(getattr(bar, name)) + delta) for name in ("o", "h", "l", "c"))
        volume, quote, taker, taker_quote = (
            bar.volume,
            bar.quote_volume,
            bar.taker_buy_base,
            bar.taker_buy_quote,
        )
        trades = bar.trades
    return Kline(
        market=market,
        symbol=bar.symbol,
        interval=bar.interval,
        open_time=bar.open_time,
        close_time=bar.close_time,
        o=prices[0],
        h=prices[1],
        l=prices[2],
        c=prices[3],
        volume=volume,
        quote_volume=quote,
        trades=trades,
        taker_buy_base=taker,
        taker_buy_quote=taker_quote,
        source="preview",
        ingested_at=bar.ingested_at,
        ingested_local=bar.ingested_local,
    )


def seed_preview(db: Database, *, days: int = 10) -> dict:
    """Build a deterministic ETH-like book starting Monday 2026-10-26 UTC."""
    if days < 1:
        raise ValueError("days 至少为 1")
    start = datetime(2026, 10, 26, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    price = Decimal("3500.00")
    um_1m: list[Kline] = []
    for i in range(days * 24 * 60):
        delta = Decimal((i * 17) % 11 - 5) * Decimal("0.15")
        opened = price
        closed = (price + delta).quantize(Decimal("0.01"))
        if closed < Decimal("100"):
            closed = Decimal("100.00")
        high = max(opened, closed) + Decimal("0.40")
        low = min(opened, closed) - Decimal("0.40")
        volume = Decimal(8 + (i % 9))
        taker = (volume * Decimal("0.55")).quantize(Decimal("0.0001"))
        mid = (opened + closed) / 2
        open_time = start_ms + i * 60_000
        um_1m.append(
            Kline(
                market="um",
                symbol="ETHUSDT",
                interval="1m",
                open_time=open_time,
                close_time=open_time + 59_999,
                o=_text(opened),
                h=_text(high),
                l=_text(low),
                c=_text(closed),
                volume=_text(volume),
                quote_volume=_text((volume * mid).quantize(Decimal("0.0001"))),
                trades=4 + (i % 5),
                taker_buy_base=_text(taker),
                taker_buy_quote=_text((taker * mid).quantize(Decimal("0.0001"))),
                source="preview",
                ingested_at=open_time,
                ingested_local=open_time,
            )
        )
        price = closed

    grouped: dict[tuple[str, str], list[Kline]] = {}

    def add(rows: list[Kline]) -> None:
        for bar in rows:
            grouped.setdefault((bar.market, bar.interval), []).append(bar)

    add(um_1m)
    for interval in UM_INTERVALS:
        if interval in {"1m", "1M"}:
            continue
        add(aggregate_1m(um_1m, interval, ingested_at=start_ms))
    for market, delta, flat in (
        ("spot", Decimal("-2"), None),
        ("um_mark", Decimal("0.30"), None),
        ("um_index", Decimal("-1.20"), None),
        ("um_premium", None, "0.000100"),
    ):
        for rows in list(grouped.values()):
            if rows[0].market != "um":
                continue
            add([_shift(bar, market, delta, flat) for bar in rows])

    for rows in grouped.values():
        db.upsert_klines(rows)
        first, last = rows[0], rows[-1]
        db.upsert_coverage(
            first.market,
            first.symbol,
            first.interval,
            first.open_time,
            last.open_time,
            [],
            last.close_time,
        )

    metrics = []
    for i in range(0, days * 24 * 60, 5):
        ts = start_ms + i * 60_000
        metrics.append(
            MetricRow(
                symbol="ETHUSDT",
                ts=ts,
                oi=str(2_000_000 + i),
                oi_value=str(7_000_000_000 + i * 100),
                top_pos_lsr="1.20",
                top_acc_lsr="1.10",
                global_lsr="1.05",
                taker_lsr="0.98",
                source="preview",
            )
        )
    db.upsert_metrics(metrics)
    fundings = [
        FundingRow(
            "ETHUSDT",
            start_ms + hour * 3_600_000,
            "0.000100",
            "3500",
            "preview",
        )
        for hour in range(0, days * 24, 8)
    ]
    db.upsert_funding(fundings)
    db.insert_anomaly(
        start_ms + 86_400_000,
        "volume",
        8.2,
        5,
        {"note": "preview"},
        [],
    )
    db.insert_anomaly(
        start_ms + 2 * 86_400_000,
        "price_jump",
        1.6,
        4,
        {"note": "preview"},
        [],
    )
    db.add_event(
        start_ms,
        "info",
        "preview",
        "BACKFILL_REQUIRED",
        "预览库不回填币安",
        None,
        ts_local=start_ms,
    )
    db.apply_raw_coverage(
        "um:depth20",
        "2026-10-26",
        start_ms,
        start_ms + 3_600_000,
        10,
        [{"from": start_ms + 1_000, "to": start_ms + 4_000, "reason": "preview"}],
    )
    last = um_1m[-1]
    return {
        "start_ms": start_ms,
        "end_ms": last.close_time,
        "last_close": last.c,
        "minutes": len(um_1m),
    }


def load_preview_hub(hub: LiveHub, db: Database, cfg: AppConfig, summary: dict) -> None:
    last_close = Decimal(summary["last_close"])
    end_ms = int(summary["end_ms"])
    for interval in UM_INTERVALS:
        rows = db.fetch_klines("um", "ETHUSDT", interval, limit=1)
        if not rows:
            continue
        last = rows[-1]
        open_time = next_open(interval, int(last["open_time"]))
        hub.forming.append(
            {
                "market": "um",
                "symbol": "ETHUSDT",
                "interval": interval,
                "open_time": open_time,
                "close_time": close_time_of(interval, open_time),
                "o": last["c"],
                "h": last["c"],
                "l": last["c"],
                "c": last["c"],
                "volume": "1",
                "quote_volume": last["c"],
                "trades": 1,
                "taker_buy_base": "0.6",
                "taker_buy_quote": "0.6",
                "closed": False,
            }
        )
    spot_rows = db.fetch_klines("spot", "ETHUSDT", "1m", limit=1)
    if spot_rows:
        last = spot_rows[-1]
        open_time = next_open("1m", int(last["open_time"]))
        hub.forming.append(
            {
                "market": "spot",
                "symbol": "ETHUSDT",
                "interval": "1m",
                "open_time": open_time,
                "close_time": close_time_of("1m", open_time),
                "o": last["c"],
                "h": last["c"],
                "l": last["c"],
                "c": last["c"],
                "volume": "1",
                "quote_volume": last["c"],
                "trades": 1,
                "taker_buy_base": "0.6",
                "taker_buy_quote": "0.6",
                "closed": False,
            }
        )
    bids_px, bids_qty, asks_px, asks_qty = [], [], [], []
    for level in range(20):
        bids_px.append(_text(last_close - Decimal("0.01") * (level + 1)))
        asks_px.append(_text(last_close + Decimal("0.01") * (level + 1)))
        bids_qty.append(_text(Decimal(20 - level) / Decimal(2)))
        asks_qty.append(_text(Decimal(18 - level if level < 18 else 1) / Decimal(2)))
    hub.depth = {
        "type": "depth",
        "symbol": "ETHUSDT",
        "update_id": 1,
        "bids_px": bids_px,
        "bids_qty": bids_qty,
        "asks_px": asks_px,
        "asks_qty": asks_qty,
        "recv_ms": end_ms,
    }
    for i in range(12):
        hub.trades.appendleft(
            {
                "type": "aggTrade",
                "symbol": "ETHUSDT",
                "price": _text(last_close - Decimal("0.01") * i),
                "qty": "0.4",
                "ts": end_ms - i * 400,
                "agg_id": 1000 + i,
                "is_buyer_maker": i % 2 == 0,
            }
        )
    hub.ticker = {
        "type": "miniTicker",
        "symbol": "ETHUSDT",
        "open": "3500.00",
        "high": _text(last_close + Decimal("20")),
        "low": "3400.00",
        "close": _text(last_close),
        "volume": "1000",
        "quote_volume": "3500000",
        "event_ms": end_ms,
    }
    hub.mark = {
        "type": "markPrice",
        "symbol": "ETHUSDT",
        "mark_price": _text(last_close + Decimal("0.30")),
        "index_price": _text(last_close - Decimal("1.20")),
        "rate": "0.000080",
        "next_funding_time": end_ms + 3_600_000,
        "event_ms": end_ms,
    }
    hub.status = {
        "type": "status",
        "ts": end_ms,
        "sockets": {
            "market": "preview",
            "public": "preview",
            "spot": "preview",
            "last_market_ms": end_ms,
            "last_public_ms": end_ms,
            "last_spot_ms": end_ms,
        },
        "clock_offset_ms": 0,
        "clock_warn": False,
        "rest_halted": False,
        "self_weight_minute": 12,
        "self_weight_peak": 40,
        "header_weight": 80,
        "proxy": cfg.proxy.url,
    }
    hub.connected = True


async def _animate(hub: LiveHub, stop: asyncio.Event) -> None:
    step = 0
    while not stop.is_set():
        step += 1
        delta = Decimal("0.05") if step % 2 else Decimal("-0.03")
        live_price = None
        for bar in list(hub.forming):
            price = Decimal(bar["c"]) + delta
            bar["c"] = _text(price)
            if price > Decimal(bar["h"]):
                bar["h"] = bar["c"]
            if price < Decimal(bar["l"]):
                bar["l"] = bar["c"]
            bar["volume"] = _text(Decimal(bar["volume"]) + Decimal("0.1"))
            if bar.get("market") == "um" and bar.get("interval") == "1m":
                live_price = price
        if live_price is not None and hub.depth is not None:
            bids_px, bids_qty, asks_px, asks_qty = [], [], [], []
            for level in range(20):
                bids_px.append(_text(live_price - Decimal("0.01") * (level + 1)))
                asks_px.append(_text(live_price + Decimal("0.01") * (level + 1)))
                bids_qty.append(_text(Decimal(20 - level) / Decimal(2)))
                asks_qty.append(_text(Decimal(18 - level if level < 18 else 1) / Decimal(2)))
            hub.depth["bids_px"] = bids_px
            hub.depth["bids_qty"] = bids_qty
            hub.depth["asks_px"] = asks_px
            hub.depth["asks_qty"] = asks_qty
        if live_price is not None and hub.mark is not None:
            hub.mark["mark_price"] = _text(live_price + Decimal("0.30"))
            hub.mark["index_price"] = _text(live_price - Decimal("1.20"))
        if hub.forming:
            sample = hub.forming[0]
            hub.trades.appendleft(
                {
                    "type": "aggTrade",
                    "symbol": "ETHUSDT",
                    "price": sample["c"],
                    "qty": "0.2",
                    "ts": int(sample["open_time"]) + step,
                    "agg_id": 5000 + step,
                    "is_buyer_maker": step % 2 == 0,
                }
            )
            if hub.ticker is not None:
                hub.ticker["close"] = sample["c"]
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.3)
        except TimeoutError:
            continue


def _build(cfg: AppConfig, config_path: Path, *, days: int, port: int | None):
    if port is not None:
        cfg.server.api_port = port
    root = data_dir(cfg, config_path) / "preview"
    root.mkdir(parents=True, exist_ok=True)
    sqlite_path = root / "superE.sqlite"
    sqlite_path.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(sqlite_path) + suffix).unlink(missing_ok=True)
    writer = Database(sqlite_path, read_only=False)
    summary = seed_preview(writer, days=days)
    writer.close()
    db = Database(sqlite_path, read_only=True)
    duck = DuckReader(sqlite_path, root / "raw")
    hub = LiveHub()
    load_preview_hub(hub, db, cfg, summary)
    dist = Path(config_path).resolve().parent / "frontend" / "dist"
    app = create_app(
        cfg,
        DataAccess(db, duck),
        hub,
        connect_ipc=False,
        static_dir=dist,
        data_path=root,
    )
    app.state.preview = True
    return app, hub, db, duck


def serve_preview(
    cfg: AppConfig,
    config_path: Path,
    *,
    open_window: bool = False,
    port: int | None = None,
    days: int = 10,
) -> None:
    app, hub, db, duck = _build(cfg, config_path, days=days, port=port)
    config = uvicorn.Config(
        app,
        host=cfg.server.api_host,
        port=cfg.server.api_port,
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    origin = origin_of(cfg)
    print(f"预览数据，没有连接币安。打开 {origin}/")

    async def run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(_animate(hub, stop))
        try:
            await server.serve()
        finally:
            stop.set()
            task.cancel()
            db.close()
            duck.close()

    if not open_window:
        asyncio.run(run())
        return

    def _serve() -> None:
        asyncio.run(run())

    thread = threading.Thread(target=_serve, name="supere-preview", daemon=True)
    thread.start()
    if not wait_for_api(origin, timeout=30):
        raise SystemExit("预览 API 没有起来")
    run_window(origin + "/")
