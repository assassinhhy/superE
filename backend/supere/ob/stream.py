"""Incremental driver for the live recorder.

The recorder already receives aggTrade, depth and metrics messages.  Feeding
them here produces exactly the same row as the batch builder in
``supere.ob.build`` because both call ``ObFeatureState.push_minute``.

Usage from a live loop (all timestamps are epoch milliseconds UTC)::

    engine = StreamingObEngine()
    engine.on_book(snapshot_ts, levels)        # levels: {percentage: notional}
    engine.on_metrics(ts, oi=..., oi_value=..., ...)
    for trade in trades:
        engine.on_trade(ts, price, qty, is_buyer_maker, auto_close=False)

    # At the wall-clock minute boundary, after every book/metrics event with
    # ts < boundary has been fed:
    rows = engine.close_minute(previous_minute)   # + engine.drain_rows()

``close_minute`` is idempotent for the current minute: a scheduler close and
the auto-close triggered by the first trade of the next minute produce one row,
and intermediate empty minutes are kept in ``drain_rows()`` rather than
dropped.  Late trades for an already-closed minute are counted in
``late_trades`` and ignored; they must be reconciled by the gap filler.
"""

from __future__ import annotations

import numpy as np

from supere.ob.features import (
    FEATURE_COLUMNS,
    MINUTE_MS,
    NAN,
    MinuteSummary,
    ObFeatureState,
    summarize_trades,
)


class StreamingObEngine:
    def __init__(self) -> None:
        self.state = ObFeatureState()
        self.current_minute: int | None = None
        self._closed_minute: int | None = None
        self._closed_row: dict[str, float] | None = None
        self._closed_rows: list[dict[str, float]] = []
        self.late_trades = 0
        self._ts: list[int] = []
        self._price: list[float] = []
        self._qty: list[float] = []
        self._maker: list[bool] = []
        self.last_row: dict[str, float] | None = None

    # -- feeds -------------------------------------------------------------
    def on_book(self, ts: int, levels: dict[int, float]) -> None:
        bid = np.full(5, NAN)
        ask = np.full(5, NAN)
        for percentage, notional in levels.items():
            level = abs(int(percentage)) - 1
            if 0 <= level < 5:
                if percentage < 0:
                    bid[level] = float(notional)
                elif percentage > 0:
                    ask[level] = float(notional)
        self.state.book.on_snapshot(int(ts), bid, ask)

    def on_metrics(
        self,
        ts: int,
        *,
        oi: float,
        oi_value: float,
        top_pos_lsr: float,
        top_acc_lsr: float,
        global_lsr: float,
        taker_lsr: float,
    ) -> None:
        self.state.metrics.push(ts, oi, oi_value, top_pos_lsr, top_acc_lsr, global_lsr, taker_lsr)

    def on_trade(
        self,
        ts: int,
        price: float,
        qty: float,
        is_buyer_maker: bool,
        *,
        auto_close: bool = True,
    ) -> dict[str, float] | None:
        minute = (int(ts) // MINUTE_MS) * MINUTE_MS
        closed: dict[str, float] | None = None
        if self.current_minute is None:
            self.current_minute = minute
        elif minute < self.current_minute:
            self.late_trades += 1
            return None
        elif minute > self.current_minute:
            if auto_close:
                closed = self.close_minute(self.current_minute)
                while self.current_minute + MINUTE_MS < minute:
                    self.current_minute += MINUTE_MS
                    self._closed_minute = None
                    self.close_minute(self.current_minute)
            self.current_minute = minute
            self._closed_minute = None
        elif self._closed_minute == minute:
            # The scheduler already closed this minute; a straggler cannot
            # change the emitted row.
            self.late_trades += 1
            return None
        self._ts.append(int(ts))
        self._price.append(float(price))
        self._qty.append(float(qty))
        self._maker.append(bool(is_buyer_maker))
        return closed

    # -- minute close ------------------------------------------------------
    def _emit(self, row: dict[str, float]) -> dict[str, float]:
        self._closed_rows.append(row)
        self.last_row = row
        return row

    def close_minute(self, minute_ms: int | None = None) -> dict[str, float]:
        if self.current_minute is None:
            raise RuntimeError("还没有收到任何 aggTrade")
        minute = self.current_minute if minute_ms is None else int(minute_ms)
        if minute != self.current_minute:
            raise ValueError("close_minute 只能关闭当前分钟")
        if self._closed_minute == minute and not self._ts:
            assert self._closed_row is not None
            return self._closed_row
        ts = np.asarray(self._ts, dtype=np.int64)
        price = np.asarray(self._price, dtype=np.float64)
        qty = np.asarray(self._qty, dtype=np.float64)
        maker = np.asarray(self._maker, dtype=bool)
        summary, notionals = summarize_trades(minute, ts, price, qty, maker)
        row = self.state.push_minute(minute, summary, notionals)
        self._ts.clear()
        self._price.clear()
        self._qty.clear()
        self._maker.clear()
        self._closed_minute = minute
        self._closed_row = row
        return self._emit(row)

    def drain_rows(self) -> list[dict[str, float]]:
        rows = self._closed_rows
        self._closed_rows = []
        return rows

    def push_minute_arrays(
        self,
        minute_ms: int,
        ts: np.ndarray,
        price: np.ndarray,
        qty: np.ndarray,
        is_buyer_maker: np.ndarray,
    ) -> dict[str, float]:
        """Batch entry point used by the historical builder."""
        summary, notionals = summarize_trades(minute_ms, ts, price, qty, is_buyer_maker)
        return self.state.push_minute(minute_ms, summary, notionals)


def row_to_list(row: dict[str, float]) -> list[float]:
    return [row.get(name, NAN) for name in FEATURE_COLUMNS]
