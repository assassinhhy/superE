"""Fill kline and aggTrade holes. Depth and liquidations are only recorded."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from supere.binance.archive import zip_text
from supere.binance.parse import dec_str, kline_from_list, parse_kline_csv
from supere.binance.rest import KLINE_PAGE
from supere.binance.urls import agg_trades_zip, kline_zip
from supere.ingest.gaps import find_missing_ranges
from supere.intervals import (
    UM_INTERVALS,
    UM_ONBOARD_MS,
    align_open,
    close_time_of,
    is_closed,
    last_closed_open,
    next_open,
)
from supere.storage.db import Database
from supere.storage.parquet_store import ParquetStore
from supere.timeutil import utc8_date

log = logging.getLogger("supere.gapfill")

PRICE_MARKETS = ("um", "um_mark", "um_index", "um_premium")


def recompute_coverage(db: Database, market: str, symbol: str, interval: str, now_ms: int) -> list[tuple[int, int]]:
    bounds = db.min_max_open(market, symbol, interval)
    last = last_closed_open(interval, now_ms)
    if bounds is None or last is None:
        db.upsert_coverage(market, symbol, interval, None, None, [], now_ms)
        return []
    start, end = bounds
    scan_end = max(end, last)
    opens = db.iter_opens(market, symbol, interval, start, scan_end)
    gaps = find_missing_ranges(interval, opens, start, scan_end)
    db.upsert_coverage(market, symbol, interval, start, scan_end, gaps, now_ms)
    return gaps


async def fill_kline_range(
    rest,
    db: Database,
    stamp,
    market: str,
    symbol: str,
    interval: str,
    start_open: int,
    end_open: int,
    page: int = KLINE_PAGE,
) -> int:
    """REST-fill closed candles in ``[start_open, end_open]``. Returns rows written."""
    if end_open < start_open:
        return 0
    written = 0
    cursor = start_open
    end_time = close_time_of(interval, end_open)
    page = max(1, min(KLINE_PAGE, page))
    while cursor <= end_open:
        server_ms, local_ms = stamp()
        raw = await rest.klines(
            market,
            symbol,
            interval,
            start=cursor,
            end=end_time,
            limit=page,
        )
        if not raw:
            break
        rows = []
        for item in raw:
            bar = kline_from_list(market, symbol, interval, item, "rest", server_ms, local_ms)
            if bar.open_time < cursor or bar.open_time > end_open:
                continue
            if not is_closed(interval, bar.open_time, server_ms):
                continue
            rows.append(bar)
        written += db.upsert_klines(rows)
        last_open = int(raw[-1][0])
        # Spot archives can be microseconds; REST is milliseconds, but be tolerant.
        from supere.timeutil import to_millis

        last_open = to_millis(last_open)
        nxt = next_open(interval, last_open)
        if nxt <= cursor or len(raw) < page:
            break
        cursor = nxt
    return written


async def fill_recent_tail(rest, db, stamp, market: str, symbol: str, interval: str, limit: int) -> int:
    server_ms, _local = stamp()
    last = last_closed_open(interval, server_ms)
    if last is None:
        return 0
    start = last
    # Walk back ``limit`` bars without materialising a huge list for 1M.
    for _ in range(max(0, limit - 1)):
        prev = start
        from supere.intervals import previous_open

        start = previous_open(interval, start)
        if start >= prev:
            break
    return await fill_kline_range(
        rest, db, stamp, market, symbol, interval, start, last, page=min(KLINE_PAGE, max(1, limit))
    )


def _day_start(day) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


async def fill_agg_gap(rest, db, store: ParquetStore, stamp, cfg, symbol: str) -> int:
    """Fill aggTrade ids after the last stored id. Long holes prefer the archive."""
    state = db.get_stream_state("um:aggTrade")
    if state is None or state["last_id"] is None or state["last_ts"] is None:
        return 0
    last_id = int(state["last_id"])
    last_ts = int(state["last_ts"])
    server_ms, local_ms = stamp()
    gap_ms = server_ms - last_ts
    threshold = int(cfg.ingest.aggtrade_archive_gap_hours * 3_600_000)
    written = 0
    if gap_ms > threshold:
        written += await _import_agg_archive_days(rest, store, cfg, symbol, last_ts, server_ms, local_ms)
    cursor = last_id + 1
    pages = 0
    while pages < 2000:
        batch = await rest.agg_trades(symbol, cursor, limit=1000)
        pages += 1
        if not batch:
            break
        server_ms, local_ms = stamp()
        max_id = cursor - 1
        for item in batch:
            agg_id = int(item["a"])
            ts = int(item["T"])
            row = {
                "agg_id": agg_id,
                "price": dec_str(item["p"]),
                "qty": dec_str(item["q"]),
                "first_id": int(item["f"]),
                "last_id": int(item["l"]),
                "ts": ts,
                "is_buyer_maker": bool(item.get("m")),
                "recv_ms": server_ms,
                "recv_local_ms": local_ms,
                "source": "rest",
            }
            store.append("aggTrades", "um", symbol.upper(), utc8_date(ts), row)
            max_id = max(max_id, agg_id)
        db.set_stream_state("um:aggTrade", max_id, int(batch[-1]["T"]), server_ms)
        if len(batch) < 1000:
            written += len(batch)
            break
        written += len(batch)
        nxt = max_id + 1
        if nxt <= cursor:
            break
        cursor = nxt
    return written


async def _import_agg_archive_days(rest, store, cfg, symbol, start_ms, end_ms, local_ms) -> int:
    start_day = datetime.fromtimestamp(start_ms / 1000, tz=UTC).date() + timedelta(days=1)
    end_day = datetime.fromtimestamp(end_ms / 1000, tz=UTC).date() - timedelta(days=1)
    today = datetime.fromtimestamp(end_ms / 1000, tz=UTC).date()
    written = 0
    day = start_day
    while day <= end_day and day < today:
        stamp = day.isoformat()
        marker_dir = store.root / "aggTrades" / "um" / symbol.upper() / f"date={stamp}"
        marker = marker_dir / "archive.done"
        if marker.exists():
            day += timedelta(days=1)
            continue
        url = agg_trades_zip(cfg.endpoints.vision, symbol.upper(), stamp)
        dest = rest_path(cfg, "aggTrades", f"{symbol}-aggTrades-{stamp}.zip")
        try:
            ok = await rest.download_verified(url, dest)
        except Exception:
            log.warning("aggTrade 存档下载失败 %s", stamp, exc_info=True)
            break
        if not ok:
            break
        text = zip_text(dest.read_bytes())
        rows = []
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if not parts or not parts[0].isdigit():
                continue
            ts = int(parts[5])
            rows.append(
                {
                    "agg_id": int(parts[0]),
                    "price": dec_str(parts[1]),
                    "qty": dec_str(parts[2]),
                    "first_id": int(parts[3]),
                    "last_id": int(parts[4]),
                    "ts": ts,
                    "is_buyer_maker": parts[6].lower() == "true",
                    "recv_ms": end_ms,
                    "recv_local_ms": local_ms,
                    "source": "archive",
                }
            )
            if len(rows) >= 50_000:
                _write_rows(store, symbol, stamp, rows)
                written += len(rows)
                rows = []
        if rows:
            _write_rows(store, symbol, stamp, rows)
            written += len(rows)
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
        day += timedelta(days=1)
    return written


def _write_rows(store: ParquetStore, symbol: str, date: str, rows: list[dict]) -> None:
    from supere.storage.parquet_store import table_from_rows
    import pyarrow.parquet as pq

    folder = store.root / "aggTrades" / "um" / symbol.upper() / f"date={date}"
    folder.mkdir(parents=True, exist_ok=True)
    table = table_from_rows("aggTrades", rows)
    out = folder / f"part-archive-{rows[0]['agg_id']}.parquet"
    pq.write_table(table, out, compression="zstd", compression_level=3)


def rest_path(cfg, kind: str, name: str):
    from pathlib import Path

    root = Path(cfg.data.dir)
    # cfg.data.dir may be relative; callers pass an absolute archive dir via cfg is awkward.
    return Path(getattr(cfg, "_archive_dir", Path(cfg.data.dir) / "archive_cache")) / kind / name


async def import_daily_klines(rest, db, stamp_fn, cfg, market: str, symbol: str, interval: str, day) -> bool:
    series = _series(market)
    venue = "spot" if market == "spot" else "um"
    url = kline_zip(
        cfg.endpoints.vision,
        venue,
        symbol.upper(),
        interval,
        "daily",
        day.isoformat(),
        series=series,
    )
    dest = rest_path(cfg, "klines", url.rsplit("/", 1)[-1])
    ok = await rest.download_verified(url, dest)
    if not ok:
        return False
    server_ms, local_ms = stamp_fn()
    rows = parse_kline_csv(
        zip_text(dest.read_bytes()), market, symbol, interval, "archive", server_ms, local_ms
    )
    db.upsert_klines(rows)
    return True


def _series(market: str) -> str:
    return {
        "um": "klines",
        "spot": "klines",
        "um_mark": "markPriceKlines",
        "um_index": "indexPriceKlines",
        "um_premium": "premiumIndexKlines",
    }[market]


def default_start(market: str) -> int:
    if market == "spot":
        return int(datetime(2017, 8, 17, tzinfo=UTC).timestamp() * 1000)
    return UM_ONBOARD_MS


async def startup_gapfill(rest, db, store, stamp, cfg, on_event) -> None:
    """On an empty database, keep a short tail and ask for ``supere backfill``.

    Existing databases are scanned. Holes are filled from REST, and kline holes
    longer than ``archive_gap_days`` are pulled from daily archives first.
    """
    server_ms, _local = stamp()
    symbol = cfg.symbols.um
    empty = db.min_max_open("um", symbol, "1m") is None
    if empty:
        await on_event(
            "warning",
            "ingest",
            "BACKFILL_REQUIRED",
            "本地还没有历史 K 线。先只补最近一页，完整历史请运行 supere backfill。",
            {},
        )
        for interval in UM_INTERVALS:
            await fill_recent_tail(rest, db, stamp, "um", symbol, interval, cfg.ingest.startup_tail_limit)
        await fill_recent_tail(
            rest, db, stamp, "spot", cfg.symbols.spot, "1m", cfg.ingest.startup_tail_limit
        )
        for interval in UM_INTERVALS:
            recompute_coverage(db, "um", symbol, interval, server_ms)
        return

    limit_ms = int(cfg.ingest.archive_gap_days * 86_400_000)
    for market, sym, intervals in (
        ("um", symbol, UM_INTERVALS),
        ("spot", cfg.symbols.spot, ("1m",)),
    ):
        for interval in intervals:
            gaps = recompute_coverage(db, market, sym, interval, server_ms)
            for start_open, end_open in gaps:
                span = close_time_of(interval, end_open) - start_open
                if span > limit_ms and interval not in {"1w", "1M"}:
                    await _fill_gap_with_archive(rest, db, stamp, cfg, market, sym, interval, start_open, end_open)
                await fill_kline_range(rest, db, stamp, market, sym, interval, start_open, end_open)
            recompute_coverage(db, market, sym, interval, server_ms)
    await fill_agg_gap(rest, db, store, stamp, cfg, symbol)


async def _fill_gap_with_archive(rest, db, stamp, cfg, market, symbol, interval, start_open, end_open) -> None:
    start_day = datetime.fromtimestamp(start_open / 1000, tz=UTC).date()
    end_day = datetime.fromtimestamp(end_open / 1000, tz=UTC).date()
    today = datetime.now(UTC).date()
    day = start_day
    while day <= end_day and day < today:
        try:
            await import_daily_klines(rest, db, stamp, cfg, market, symbol, interval, day)
        except Exception:
            log.warning("日度 K 线存档失败 %s %s %s", market, interval, day, exc_info=True)
            return
        day += timedelta(days=1)
    # align_open is used by callers; reference it so the import stays meaningful for 3d/1w gaps.
    _ = align_open
