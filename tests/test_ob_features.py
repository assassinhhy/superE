from __future__ import annotations

import math

import numpy as np

from supere.ob.features import (
    FEATURE_COLUMNS,
    MINUTE_MS,
    MinuteSummary,
    group_book_snapshots,
    live_depth_to_percentage_levels,
)
from supere.ob.stream import StreamingObEngine

DAY = 1_700_000_000_000 // 86_400_000 * 86_400_000


def _trades(minutes: int = 180, per_minute: int = 40, seed: int = 7):
    rng = np.random.default_rng(seed)
    ts, price, qty, maker = [], [], [], []
    level = 2000.0
    for minute in range(minutes):
        level *= math.exp(rng.normal(0, 0.0005))
        offsets = np.sort(rng.integers(0, MINUTE_MS, per_minute))
        for offset in offsets:
            level *= math.exp(rng.normal(0, 0.0002))
            ts.append(DAY + minute * MINUTE_MS + int(offset))
            price.append(level)
            qty.append(float(rng.lognormal(mean=-2.0, sigma=1.0)))
            maker.append(bool(rng.integers(0, 2)))
    return (
        np.asarray(ts, dtype=np.int64),
        np.asarray(price, dtype=np.float64),
        np.asarray(qty, dtype=np.float64),
        np.asarray(maker, dtype=bool),
    )


def _books(minutes: int = 180, seed: int = 11):
    rng = np.random.default_rng(seed)
    rows = []
    for minute in range(minutes):
        for offset in (5, 35):
            ts = DAY + minute * MINUTE_MS + offset * 1000
            base = float(rng.uniform(1e6, 3e6))
            for pct in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
                rows.append((ts, pct, base * (1.0 + abs(pct) * 0.2), base * 10 * (1.0 + abs(pct) * 0.2)))
    ts = np.asarray([r[0] for r in rows], dtype=np.int64)
    pct = np.asarray([r[1] for r in rows], dtype=np.int64)
    notional = np.asarray([r[3] for r in rows], dtype=np.float64)
    return ts, pct, notional


def _metrics(minutes: int = 180, seed: int = 13):
    rng = np.random.default_rng(seed)
    rows = []
    oi = 1e6
    for minute in range(0, minutes, 5):
        oi *= math.exp(rng.normal(0, 0.001))
        rows.append(
            (
                DAY + minute * MINUTE_MS,
                oi,
                oi * 2000.0,
                1.5 + rng.normal(0, 0.01),
                0.9 + rng.normal(0, 0.01),
                1.2 + rng.normal(0, 0.01),
                0.8 + rng.normal(0, 0.01),
            )
        )
    return rows


def _batch(ts, price, qty, maker, book_ts, book_pct, book_notional, metrics, minutes):
    engine = StreamingObEngine()
    snap_ts, bid, ask, _, _ = group_book_snapshots(book_ts, book_pct, book_notional)
    engine.state.book.add_day(snap_ts, bid, ask)
    for row in metrics:
        engine.state.metrics.push(*row)
    rows = []
    start = 0
    for minute in range(minutes):
        begin = DAY + minute * MINUTE_MS
        end = begin + MINUTE_MS
        while start < len(ts) and ts[start] < begin:
            start += 1
        stop = start
        while stop < len(ts) and ts[stop] < end:
            stop += 1
        rows.append(
            engine.push_minute_arrays(
                begin, ts[start:stop], price[start:stop], qty[start:stop], maker[start:stop]
            )
        )
        start = stop
    return rows


def _incremental(ts, price, qty, maker, book_ts, book_pct, book_notional, metrics, minutes):
    engine = StreamingObEngine()
    events = []
    for t in np.unique(book_ts):
        events.append((int(t), 0, "book", int(t)))
    for i, row in enumerate(metrics):
        events.append((int(row[0]), 1, "metrics", i))
    for i, t in enumerate(ts):
        events.append((int(t), 2, "trade", i))
    events.sort()
    rows = []
    for t, _order, kind, index in events:
        if kind == "book":
            mask = book_ts == t
            levels = {}
            for pct, notional in zip(book_pct[mask], book_notional[mask]):
                levels[int(pct)] = float(notional)
            engine.on_book(t, levels)
        elif kind == "metrics":
            engine.on_metrics(t, oi=metrics[index][1], oi_value=metrics[index][2],
                              top_pos_lsr=metrics[index][3], top_acc_lsr=metrics[index][4],
                              global_lsr=metrics[index][5], taker_lsr=metrics[index][6])
        else:
            engine.on_trade(t, price[index], qty[index], maker[index])
    if engine.current_minute is not None:
        engine.close_minute()
    return engine.drain_rows()


def test_batch_equals_incremental():
    minutes = 180
    trades = _trades(minutes)
    books = _books(minutes)
    metrics = _metrics(minutes)
    batch = _batch(*trades, *books, metrics, minutes)
    stream = _incremental(*trades, *books, metrics, minutes)
    assert len(batch) == len(stream) == minutes
    for i, (left, right) in enumerate(zip(batch, stream)):
        for name in FEATURE_COLUMNS:
            a = left[name]
            b = right[name]
            if math.isnan(a) and math.isnan(b):
                continue
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (i, name, a, b)


def test_no_lookahead_truncation():
    minutes = 180
    trades = _trades(minutes)
    books = _books(minutes)
    metrics = _metrics(minutes)
    full = _batch(*trades, *books, metrics, minutes)

    cut = 90
    cut_ms = DAY + cut * MINUTE_MS
    ts, price, qty, maker = trades
    keep = ts < cut_ms
    book_ts, book_pct, book_notional = books
    book_keep = book_ts < cut_ms
    metrics_keep = [row for row in metrics if row[0] + 300_000 <= cut_ms]
    partial = _batch(
        ts[keep], price[keep], qty[keep], maker[keep],
        book_ts[book_keep], book_pct[book_keep], book_notional[book_keep],
        metrics_keep, cut,
    )
    assert len(partial) == cut
    for i in range(cut):
        for name in FEATURE_COLUMNS:
            a = full[i][name]
            b = partial[i][name]
            if math.isnan(a) and math.isnan(b):
                continue
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (i, name, a, b)


def test_metrics_publication_delay():
    engine = StreamingObEngine()
    engine.state.metrics.push(DAY, 100.0, 1000.0, 1.0, 1.0, 1.0, 1.0)
    engine.state.metrics.push(DAY + 300_000, 200.0, 2000.0, 1.0, 1.0, 1.0, 1.0)
    rows = []
    empty = MinuteSummary(0, 0, 0.0, 0.0, 0.0, 0.0)
    for minute in range(11):
        minute_ms = DAY + minute * MINUTE_MS
        row = engine.state.push_minute(minute_ms, MinuteSummary(minute_ms, 0, 0.0, 0.0, 0.0, 0.0), np.zeros(0))
        rows.append(row)
    assert math.isnan(rows[0]["ob_oi"])  # decision 00:01, bar 00:00 known at 00:05
    assert rows[4]["ob_oi"] == 100.0  # decision 00:05
    assert rows[9]["ob_oi"] == 200.0  # decision 00:10
    assert math.isclose(rows[9]["ob_oi_chg_5m"], math.log(2.0))


def test_book_staleness_cap():
    engine = StreamingObEngine()
    bid = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ask = np.array([11.0, 21.0, 31.0, 41.0, 51.0])
    engine.state.book.on_snapshot(DAY, bid, ask)
    rows = []
    for minute in range(7):
        minute_ms = DAY + minute * MINUTE_MS
        rows.append(
            engine.state.push_minute(
                minute_ms, MinuteSummary(minute_ms, 0, 0.0, 0.0, 0.0, 0.0), np.zeros(0)
            )
        )
    assert rows[4]["ob_book_stale"] == 0.0
    assert rows[4]["ob_bid_notional_1"] == 10.0
    assert rows[5]["ob_book_stale"] == 1.0
    assert math.isnan(rows[5]["ob_bid_notional_1"])


def test_book_snapshot_at_exact_close_belongs_to_next_minute():
    engine = StreamingObEngine()
    bid = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ask = np.array([11.0, 21.0, 31.0, 41.0, 51.0])
    engine.state.book.on_snapshot(DAY + MINUTE_MS, bid, ask)  # exactly 00:01:00
    first = engine.state.push_minute(
        DAY, MinuteSummary(DAY, 0, 0.0, 0.0, 0.0, 0.0), np.zeros(0)
    )
    second = engine.state.push_minute(
        DAY + MINUTE_MS, MinuteSummary(DAY + MINUTE_MS, 0, 0.0, 0.0, 0.0, 0.0), np.zeros(0)
    )
    assert first["ob_book_stale"] == 1.0
    assert second["ob_bid_notional_1"] == 10.0


def test_idempotent_close_and_empty_minutes():
    engine = StreamingObEngine()
    engine.on_trade(DAY + 1_000, 2000.0, 1.0, False)
    first = engine.close_minute()
    again = engine.close_minute()
    assert first is again
    engine.on_trade(DAY + 2 * MINUTE_MS + 1_000, 2000.0, 1.0, False)  # minute 1 had no trades
    engine.on_trade(DAY + 3 * MINUTE_MS + 1_000, 2000.0, 1.0, False)
    engine.close_minute()
    rows = engine.drain_rows()
    assert len(rows) == 4
    assert rows[0]["ob_trade_count"] == 1.0
    assert rows[1]["ob_trade_count"] == 0.0
    assert rows[2]["ob_trade_count"] == 1.0
    assert rows[3]["ob_trade_count"] == 1.0

    engine.on_trade(DAY + 500, 2000.0, 1.0, False)  # late trade for minute 0
    assert engine.late_trades == 1


def test_group_book_snapshots_marks_missing_levels():
    ts = np.array([100, 100, 100, 200, 200], dtype=np.int64)
    pct = np.array([-1, 1, -2, 1, 2], dtype=np.int64)
    notional = np.array([10.0, 11.0, 12.0, 21.0, 22.0])
    snap_ts, bid, ask, _, _ = group_book_snapshots(ts, pct, notional)
    assert list(snap_ts) == [100, 200]
    assert bid[0, 0] == 10.0 and bid[0, 1] == 12.0
    assert math.isnan(bid[0, 4])
    assert math.isnan(bid[1, 0])
    assert ask[1, 0] == 21.0


def test_live_depth_bands_are_cumulative():
    levels = live_depth_to_percentage_levels(
        bids=[(99.5, 10.0), (98.0, 10.0), (94.0, 1.0)],  # 0.5%, 2%, 6%
        asks=[(101.0, 5.0), (104.5, 1.0)],
        mid=100.0,
    )
    assert levels[-1] == 995.0
    assert levels[-2] == 995.0 + 980.0
    assert math.isnan(levels[-3])  # deepest bid is 2%; deeper bands are unknown
    assert math.isnan(levels[-5])
    assert levels[1] == 505.0
    assert levels[4] == 505.0
    assert math.isnan(levels[5])  # deepest ask is 4.5%
