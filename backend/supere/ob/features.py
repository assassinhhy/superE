"""Causal 1-minute order-book features.

The same code runs in three places:

* the historical builder, over daily Parquet partitions;
* the backfill tests, batch vs incremental;
* the live recorder (``StreamingObEngine``), so training and serving cannot
  drift apart.

Causality rules
---------------
* A feature row is keyed by ``minute_ms`` and its decision moment is
  ``minute_ms + 60_000``.
* Only trades with ``transact_time < decision`` contribute to the row.
* A book snapshot counts when ``snapshot_ts < decision``.  Strict inequality:
  a snapshot stamped exactly at the minute close belongs to the next minute,
  which also keeps a day-partitioned batch build from needing tomorrow's file
  for today's last row.  Values older than five minutes are treated as
  missing; the age is reported separately.
* A metrics bar with ``create_time = t`` is known at ``t + 5 min``.  It counts
  only when ``t + 5 min <= decision``.  A bar older than one hour is treated as
  missing so a data gap does not silently reuse a day-old value.
* Rolling windows only look backwards.  The large-trade threshold is the p99
  of the *previous* 60 completed minutes; it never uses the minute it scores.

All functions are deterministic and hold no I/O.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

MINUTE_MS = 60_000
METRICS_INTERVAL_MS = 300_000
METRICS_DELAY_MS = 300_000
METRICS_MAX_AGE_MS = 3_600_000
BOOK_STALE_MS = 300_000
ROLL_MINUTES = 60
BOOK_HISTORY_MINUTES = 20
MIN_LARGE_OBS = 1000
MIN_PRIOR_MINUTES = 30

LOG_LO = -4.0
LOG_HI = 10.0
BIN_WIDTH = 0.1
N_BINS = int(round((LOG_HI - LOG_LO) / BIN_WIDTH))

FEATURE_COLUMNS: tuple[str, ...] = (
    # aggTrades
    "ob_quote",
    "ob_qty",
    "ob_vwap",
    "ob_buy_quote",
    "ob_sell_quote",
    "ob_signed_quote",
    "ob_signed_ratio",
    "ob_trade_count",
    "ob_log_trade_count",
    "ob_avg_trade_notional",
    "ob_trade_intensity_z60",
    "ob_large_trade_count",
    "ob_large_trade_quote",
    "ob_large_trade_ratio",
    "ob_vwap_dev_60m",
    # bookDepth
    "ob_bid_notional_1",
    "ob_ask_notional_1",
    "ob_bid_notional_2",
    "ob_ask_notional_2",
    "ob_bid_notional_5",
    "ob_ask_notional_5",
    "ob_imb_1",
    "ob_imb_2",
    "ob_imb_5",
    "ob_imb_1_chg_1m",
    "ob_imb_5_chg_1m",
    "ob_imb_5_chg_5m",
    "ob_imb_5_chg_15m",
    "ob_bid_notional_5_chg_1m",
    "ob_ask_notional_5_chg_1m",
    "ob_bid_notional_5_chg_5m",
    "ob_ask_notional_5_chg_5m",
    "ob_bid_notional_5_chg_15m",
    "ob_ask_notional_5_chg_15m",
    "ob_depth_volume_ratio",
    "ob_book_age_s",
    "ob_book_stale",
    # metrics
    "ob_oi",
    "ob_oi_log",
    "ob_oi_value",
    "ob_oi_chg_5m",
    "ob_oi_chg_15m",
    "ob_oi_chg_1h",
    "ob_oi_value_chg_5m",
    "ob_top_pos_lsr",
    "ob_top_acc_lsr",
    "ob_global_lsr",
    "ob_taker_lsr",
    "ob_global_lsr_chg_5m",
    "ob_taker_lsr_chg_5m",
)

NAN = float("nan")


def _safe_log_ratio(current: float, previous: float) -> float:
    if not (math.isfinite(current) and math.isfinite(previous)):
        return NAN
    if current <= 0.0 or previous <= 0.0:
        return NAN
    return math.log(current / previous)


def _ratio_change(current: float, previous: float) -> float:
    if not (math.isfinite(current) and math.isfinite(previous)) or previous == 0.0:
        return NAN
    return current / previous - 1.0


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    if not (math.isfinite(total) and total > 0.0):
        return NAN
    return (bid - ask) / total


# ---------------------------------------------------------------------------
# Book snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookSnapshot:
    ts: int
    bid_notional: np.ndarray  # index 0..4 => -1%..-5%
    ask_notional: np.ndarray
    bid_depth: np.ndarray | None = None
    ask_depth: np.ndarray | None = None

    @classmethod
    def empty(cls, ts: int = -1) -> "BookSnapshot":
        nan5 = np.full(5, NAN)
        return cls(ts, nan5.copy(), nan5.copy(), nan5.copy(), nan5.copy())


def group_book_snapshots(
    ts: np.ndarray,
    percentage: np.ndarray,
    notional: np.ndarray,
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Group the ten ±1..±5 rows per snapshot.

    Returns ``(snap_ts, bid_notional[n,5], ask_notional[n,5], bid_depth,
    ask_depth)``.  A level missing from a snapshot is NaN, not zero.
    """
    ts = np.asarray(ts, dtype=np.int64)
    pct = np.asarray(percentage, dtype=np.int64)
    notional = np.asarray(notional, dtype=np.float64)
    if ts.size == 0:
        empty = np.zeros((0, 5), dtype=np.float64)
        return np.zeros(0, dtype=np.int64), empty, empty.copy(), None, None
    uniq, inv = np.unique(ts, return_inverse=True)
    level = np.clip(np.abs(pct) - 1, 0, 4)
    n = uniq.size
    bid_notional = _group_levels(inv, level, notional, pct < 0, n)
    ask_notional = _group_levels(inv, level, notional, pct > 0, n)
    bid_depth = ask_depth = None
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64)
        bid_depth = _group_levels(inv, level, depth, pct < 0, n)
        ask_depth = _group_levels(inv, level, depth, pct > 0, n)
    return uniq, bid_notional, ask_notional, bid_depth, ask_depth


def _group_levels(inv, level, values, mask, n) -> np.ndarray:
    out = np.full((n, 5), NAN, dtype=np.float64)
    counts = np.zeros((n, 5), dtype=np.int64)
    flat = inv * 5 + level
    sums = np.bincount(flat[mask], weights=values[mask], minlength=n * 5).reshape(n, 5)
    counts = np.bincount(flat[mask], minlength=n * 5).reshape(n, 5)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def live_depth_to_percentage_levels(
    bids: list[tuple[float, float]] | np.ndarray,
    asks: list[tuple[float, float]] | np.ndarray,
    mid: float,
) -> dict[int, float]:
    """Turn a live top-N book into the archive's cumulative ±1..±5% notional.

    The archive's ``depth``/``notional`` at ``percentage = k`` is cumulative
    out to ``k`` percent from the mark price.  The live ``depth20`` stream only
    has the top twenty levels, so bands deeper than the deepest visible level
    are returned as NaN rather than zero.  Filling them with zero would make
    live book features systematically smaller than the archive's.
    """
    out = {pct: NAN for pct in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)}
    if not math.isfinite(mid) or mid <= 0.0:
        return out
    for sign, levels in ((-1, bids), (1, asks)):
        parsed: list[tuple[float, float]] = []
        for price, qty in levels:
            price = float(price)
            qty = float(qty)
            if price <= 0.0 or qty <= 0.0:
                continue
            distance = (mid - price) / mid * 100.0 if sign < 0 else (price - mid) / mid * 100.0
            if distance <= 0.0 or distance > 5.0:
                continue
            parsed.append((distance, price * qty))
        if not parsed:
            continue
        covered = min(5, int(math.floor(max(distance for distance, _ in parsed))))
        for level in range(1, covered + 1):
            out[sign * level] = sum(notional for distance, notional in parsed if distance <= level)
    return out


# ---------------------------------------------------------------------------
# Trade minute aggregation
# ---------------------------------------------------------------------------


@dataclass
class MinuteSummary:
    minute_ms: int
    n: int
    quote: float
    qty: float
    buy_quote: float
    sell_quote: float

    @property
    def vwap(self) -> float:
        return self.quote / self.qty if self.qty > 0.0 else NAN


def summarize_trades(
    minute_ms: int,
    ts: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    is_buyer_maker: np.ndarray,
) -> tuple[MinuteSummary, np.ndarray]:
    """Summarise one minute and return ``(summary, trade_notional)``."""
    n = int(ts.size)
    if n == 0:
        return MinuteSummary(int(minute_ms), 0, 0.0, 0.0, 0.0, 0.0), np.zeros(0)
    notionals = price * qty
    maker = np.asarray(is_buyer_maker, dtype=bool)
    sell = float(notionals[maker].sum())
    buy = float(notionals[~maker].sum())
    return (
        MinuteSummary(
            minute_ms=int(minute_ms),
            n=n,
            quote=float(notionals.sum()),
            qty=float(qty.sum()),
            buy_quote=buy,
            sell_quote=sell,
        ),
        notionals,
    )


def notional_hist(notionals: np.ndarray) -> np.ndarray:
    if notionals.size == 0:
        return np.zeros(N_BINS, dtype=np.int64)
    values = np.maximum(np.asarray(notionals, dtype=np.float64), 1e-12)
    idx = np.floor((np.log10(values) - LOG_LO) / BIN_WIDTH).astype(np.int64)
    np.clip(idx, 0, N_BINS - 1, out=idx)
    return np.bincount(idx, minlength=N_BINS).astype(np.int64, copy=False)


def hist_quantile(hist: np.ndarray, q: float = 0.99) -> float:
    total = int(hist.sum())
    if total < MIN_LARGE_OBS:
        return NAN
    cumulative = np.cumsum(hist)
    target = q * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(max(idx, 0), N_BINS - 1)
    return float(10.0 ** (LOG_LO + idx * BIN_WIDTH))


@dataclass
class _MinuteEntry:
    minute_ms: int
    n: int
    quote: float
    qty: float
    hist: np.ndarray


class TradeRolling:
    """Rolling minute state for aggTrade features."""

    def __init__(self) -> None:
        self._entries: deque[_MinuteEntry] = deque()
        self._hist_sum = np.zeros(N_BINS, dtype=np.int64)
        self._last_minute: int | None = None

    def _evict(self, minute_ms: int) -> None:
        cutoff = minute_ms - ROLL_MINUTES * MINUTE_MS
        while self._entries and self._entries[0].minute_ms < cutoff:
            old = self._entries.popleft()
            self._hist_sum -= old.hist

    def push(self, summary: MinuteSummary, notionals: np.ndarray) -> dict[str, float]:
        minute = summary.minute_ms
        self._evict(minute)
        prior = list(self._entries)

        threshold = hist_quantile(self._hist_sum, 0.99) if len(prior) >= MIN_PRIOR_MINUTES else NAN
        if math.isfinite(threshold) and notionals.size:
            large = notionals > threshold
            large_count = float(np.count_nonzero(large))
            large_quote = float(notionals[large].sum())
        elif math.isfinite(threshold):
            large_count = 0.0
            large_quote = 0.0
        else:
            large_count = NAN
            large_quote = NAN
        large_ratio = large_quote / summary.quote if summary.quote > 0.0 and math.isfinite(large_quote) else NAN

        if len(prior) >= MIN_PRIOR_MINUTES:
            counts = np.fromiter((e.n for e in prior), dtype=np.float64, count=len(prior))
            mean = float(counts.mean())
            std = float(counts.std(ddof=0))
            intensity_z = (summary.n - mean) / std if std > 1e-12 else NAN
        else:
            intensity_z = NAN

        window = [e for e in self._entries if e.minute_ms >= minute - (ROLL_MINUTES - 1) * MINUTE_MS]
        roll_quote = summary.quote + sum(e.quote for e in window)
        roll_qty = summary.qty + sum(e.qty for e in window)
        roll_vwap = roll_quote / roll_qty if roll_qty > 0.0 else NAN
        vwap = summary.vwap
        vwap_dev = vwap / roll_vwap - 1.0 if math.isfinite(vwap) and math.isfinite(roll_vwap) and roll_vwap != 0 else NAN

        hist = notional_hist(notionals)
        entry = _MinuteEntry(minute, summary.n, summary.quote, summary.qty, hist)
        self._entries.append(entry)
        self._hist_sum += hist
        self._last_minute = minute

        buy, sell = summary.buy_quote, summary.sell_quote
        total = buy + sell
        signed = buy - sell
        return {
            "ob_quote": summary.quote,
            "ob_qty": summary.qty,
            "ob_vwap": vwap,
            "ob_buy_quote": buy,
            "ob_sell_quote": sell,
            "ob_signed_quote": signed,
            "ob_signed_ratio": signed / total if total > 0.0 else NAN,
            "ob_trade_count": float(summary.n),
            "ob_log_trade_count": math.log1p(summary.n),
            "ob_avg_trade_notional": summary.quote / summary.n if summary.n else NAN,
            "ob_trade_intensity_z60": intensity_z,
            "ob_large_trade_count": large_count,
            "ob_large_trade_quote": large_quote,
            "ob_large_trade_ratio": large_ratio,
            "ob_vwap_dev_60m": vwap_dev,
        }


# ---------------------------------------------------------------------------
# Book tracker
# ---------------------------------------------------------------------------


class BookTracker:
    """Most recent book snapshot at or before the decision moment."""

    def __init__(self) -> None:
        self._days: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque()
        self._pending: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._pos = 0
        self.last = BookSnapshot.empty()

    def add_day(self, snap_ts: np.ndarray, bid_notional: np.ndarray, ask_notional: np.ndarray) -> None:
        if len(snap_ts):
            order = np.argsort(snap_ts, kind="stable")
            self._days.append(
                (np.asarray(snap_ts)[order], np.asarray(bid_notional)[order], np.asarray(ask_notional)[order])
            )

    def on_snapshot(self, ts: int, bid_notional, ask_notional) -> None:
        snapshot = BookSnapshot(
            int(ts), np.asarray(bid_notional, dtype=float), np.asarray(ask_notional, dtype=float)
        )
        # Live feeds are ordered.  A late snapshot older than the last one is
        # ignored; a snapshot ahead of the current decision waits in _pending.
        if snapshot.ts < self.last.ts:
            return
        if self._pending and snapshot.ts < self._pending[-1][0]:
            return
        self._pending.append((snapshot.ts, snapshot.bid_notional, snapshot.ask_notional))

    def advance(self, decision_ms: int) -> None:
        while self._pending and self._pending[0][0] < decision_ms:
            ts, bid, ask = self._pending.popleft()
            self.last = BookSnapshot(ts, bid, ask)
        while self._days:
            ts, bid, ask = self._days[0]
            idx = int(np.searchsorted(ts, decision_ms, side="left"))
            if idx > self._pos:
                self.last = BookSnapshot(int(ts[idx - 1]), bid[idx - 1], ask[idx - 1])
                self._pos = idx
            if self._pos >= len(ts):
                self._days.popleft()
                self._pos = 0
            else:
                break

    def asof(self, decision_ms: int) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.last.ts < 0:
            for key in (
                "bid1", "ask1", "bid2", "ask2", "bid5", "ask5", "imb1", "imb2", "imb5", "age_s",
            ):
                out[key] = NAN
            out["stale"] = 1.0
            return out
        age_ms = decision_ms - self.last.ts
        out["age_s"] = age_ms / 1000.0
        stale = age_ms > BOOK_STALE_MS
        out["stale"] = 1.0 if stale else 0.0
        names = ("bid1", "ask1", "bid2", "ask2", "bid5", "ask5")
        if stale:
            for name in names:
                out[name] = NAN
            out["imb1"] = out["imb2"] = out["imb5"] = NAN
            return out
        for level, name in ((0, "1"), (1, "2"), (4, "5")):
            bid = float(self.last.bid_notional[level])
            ask = float(self.last.ask_notional[level])
            out[f"bid{name}"] = bid
            out[f"ask{name}"] = ask
            out[f"imb{name}"] = _imbalance(bid, ask)
        return out


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------


_METRIC_KEYS = (
    "oi",
    "oi_value",
    "top_pos_lsr",
    "top_acc_lsr",
    "global_lsr",
    "taker_lsr",
)


class MetricsTracker:
    """5-minute metrics as of a decision, with a 5-minute publication delay.

    Only bars that are not yet known plus the previous hour are retained, so a
    multi-year batch build does not accumulate every historical metrics row.
    """

    def __init__(self) -> None:
        self._pending: deque[tuple[int, dict[str, float]]] = deque()
        self._recent: dict[int, dict[str, float]] = {}
        self._last_pushed: int | None = None
        self._last: tuple[int, dict[str, float]] | None = None

    def push(
        self,
        ts: int,
        oi: float,
        oi_value: float,
        top_pos_lsr: float,
        top_acc_lsr: float,
        global_lsr: float,
        taker_lsr: float,
    ) -> None:
        ts = int(ts)
        if self._last_pushed is not None and ts <= self._last_pushed:
            return
        current = {
            "oi": oi,
            "oi_value": oi_value,
            "top_pos_lsr": top_pos_lsr,
            "top_acc_lsr": top_acc_lsr,
            "global_lsr": global_lsr,
            "taker_lsr": taker_lsr,
        }
        row = dict(current)
        for lag, tag in ((1, "5m"), (3, "15m"), (12, "1h")):
            prev = self._recent.get(ts - lag * METRICS_INTERVAL_MS)
            row[f"oi_chg_{tag}"] = _safe_log_ratio(oi, prev["oi"]) if prev else NAN
            row[f"oi_value_chg_{tag}"] = (
                _safe_log_ratio(oi_value, prev["oi_value"]) if prev else NAN
            )
            row[f"global_lsr_chg_{tag}"] = (
                _safe_log_ratio(global_lsr, prev["global_lsr"]) if prev else NAN
            )
            row[f"taker_lsr_chg_{tag}"] = (
                _safe_log_ratio(taker_lsr, prev["taker_lsr"]) if prev else NAN
            )
        self._last_pushed = ts
        self._pending.append((ts, row))
        self._recent[ts] = row
        cutoff = ts - 12 * METRICS_INTERVAL_MS
        for old in [key for key in self._recent if key < cutoff]:
            del self._recent[old]

    def advance(self, decision_ms: int) -> None:
        while self._pending and self._pending[0][0] + METRICS_DELAY_MS <= decision_ms:
            self._last = self._pending.popleft()

    def asof(self, decision_ms: int) -> dict[str, float]:
        out = {
            "oi": NAN,
            "oi_log": NAN,
            "oi_value": NAN,
            "oi_chg_5m": NAN,
            "oi_chg_15m": NAN,
            "oi_chg_1h": NAN,
            "oi_value_chg_5m": NAN,
            "top_pos_lsr": NAN,
            "top_acc_lsr": NAN,
            "global_lsr": NAN,
            "taker_lsr": NAN,
            "global_lsr_chg_5m": NAN,
            "taker_lsr_chg_5m": NAN,
        }
        if self._last is None:
            return out
        bar_ts, row = self._last
        known_at = bar_ts + METRICS_DELAY_MS
        if decision_ms - known_at > METRICS_MAX_AGE_MS:
            return out
        out["oi"] = row["oi"]
        out["oi_log"] = math.log(row["oi"]) if math.isfinite(row["oi"]) and row["oi"] > 0 else NAN
        out["oi_value"] = row["oi_value"]
        for name in (
            "oi_chg_5m",
            "oi_chg_15m",
            "oi_chg_1h",
            "oi_value_chg_5m",
            "top_pos_lsr",
            "top_acc_lsr",
            "global_lsr",
            "taker_lsr",
            "global_lsr_chg_5m",
            "taker_lsr_chg_5m",
        ):
            out[name] = row[name]
        return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


class ObFeatureState:
    """Combine the three trackers into one feature row per minute."""

    def __init__(self) -> None:
        self.trades = TradeRolling()
        self.book = BookTracker()
        self.metrics = MetricsTracker()
        self._book_history: dict[int, dict[str, float]] = {}
        self._book_minutes: deque[int] = deque(maxlen=BOOK_HISTORY_MINUTES)

    def _book_feature(self, minute: int) -> dict[str, float]:
        values = self.book.asof(minute + MINUTE_MS)
        lagged: dict[int, dict[str, float] | None] = {}
        for lag in (1, 5, 15):
            lagged[lag] = self._book_history.get(minute - lag * MINUTE_MS)
        prior = lagged[1]
        prior5 = lagged[5]
        prior15 = lagged[15]

        def chg(name: str, lag_value: dict[str, float] | None) -> float:
            if lag_value is None:
                return NAN
            return _ratio_change(values[name], lag_value[name])

        def diff(name: str, lag_value: dict[str, float] | None) -> float:
            if lag_value is None:
                return NAN
            current = values[name]
            previous = lag_value[name]
            if not (math.isfinite(current) and math.isfinite(previous)):
                return NAN
            return current - previous

        out = dict(values)
        out["imb1_chg_1m"] = diff("imb1", prior)
        out["imb5_chg_1m"] = diff("imb5", prior)
        out["imb5_chg_5m"] = diff("imb5", prior5)
        out["imb5_chg_15m"] = diff("imb5", prior15)
        out["bid5_chg_1m"] = chg("bid5", prior)
        out["ask5_chg_1m"] = chg("ask5", prior)
        out["bid5_chg_5m"] = chg("bid5", prior5)
        out["ask5_chg_5m"] = chg("ask5", prior5)
        out["bid5_chg_15m"] = chg("bid5", prior15)
        out["ask5_chg_15m"] = chg("ask5", prior15)
        return out

    def push_minute(
        self,
        minute_ms: int,
        summary: MinuteSummary,
        notionals: np.ndarray,
        *,
        book_ready: bool = True,
        metrics_ready: bool = True,
    ) -> dict[str, float]:
        minute = int(minute_ms)
        trade = self.trades.push(summary, notionals)

        if book_ready:
            self.book.advance(minute + MINUTE_MS)
        if metrics_ready:
            self.metrics.advance(minute + MINUTE_MS)

        book = self._book_feature(minute)
        met = self.metrics.asof(minute + MINUTE_MS) if metrics_ready else MetricsTracker().asof(minute + MINUTE_MS)

        depth_ratio = NAN
        bid5 = book["bid5"]
        ask5 = book["ask5"]
        if math.isfinite(bid5) and math.isfinite(ask5) and summary.quote > 0.0:
            depth_ratio = (bid5 + ask5) / summary.quote

        row: dict[str, float] = {
            **trade,
            "ob_bid_notional_1": book["bid1"],
            "ob_ask_notional_1": book["ask1"],
            "ob_bid_notional_2": book["bid2"],
            "ob_ask_notional_2": book["ask2"],
            "ob_bid_notional_5": book["bid5"],
            "ob_ask_notional_5": book["ask5"],
            "ob_imb_1": book["imb1"],
            "ob_imb_2": book["imb2"],
            "ob_imb_5": book["imb5"],
            "ob_imb_1_chg_1m": book["imb1_chg_1m"],
            "ob_imb_5_chg_1m": book["imb5_chg_1m"],
            "ob_imb_5_chg_5m": book["imb5_chg_5m"],
            "ob_imb_5_chg_15m": book["imb5_chg_15m"],
            "ob_bid_notional_5_chg_1m": book["bid5_chg_1m"],
            "ob_ask_notional_5_chg_1m": book["ask5_chg_1m"],
            "ob_bid_notional_5_chg_5m": book["bid5_chg_5m"],
            "ob_ask_notional_5_chg_5m": book["ask5_chg_5m"],
            "ob_bid_notional_5_chg_15m": book["bid5_chg_15m"],
            "ob_ask_notional_5_chg_15m": book["ask5_chg_15m"],
            "ob_depth_volume_ratio": depth_ratio,
            "ob_book_age_s": book["age_s"],
            "ob_book_stale": book["stale"],
            "ob_oi": met["oi"],
            "ob_oi_log": met["oi_log"],
            "ob_oi_value": met["oi_value"],
            "ob_oi_chg_5m": met["oi_chg_5m"],
            "ob_oi_chg_15m": met["oi_chg_15m"],
            "ob_oi_chg_1h": met["oi_chg_1h"],
            "ob_oi_value_chg_5m": met["oi_value_chg_5m"],
            "ob_top_pos_lsr": met["top_pos_lsr"],
            "ob_top_acc_lsr": met["top_acc_lsr"],
            "ob_global_lsr": met["global_lsr"],
            "ob_taker_lsr": met["taker_lsr"],
            "ob_global_lsr_chg_5m": met["global_lsr_chg_5m"],
            "ob_taker_lsr_chg_5m": met["taker_lsr_chg_5m"],
        }

        stored = {
            "bid5": book["bid5"],
            "ask5": book["ask5"],
            "imb1": book["imb1"],
            "imb5": book["imb5"],
        }
        if minute in self._book_history:
            self._book_minutes.remove(minute)
        elif len(self._book_minutes) == self._book_minutes.maxlen:
            old = self._book_minutes.popleft()
            self._book_history.pop(old, None)
        self._book_history[minute] = stored
        self._book_minutes.append(minute)
        return row


def empty_row(minute_ms: int) -> dict[str, float]:
    row = {name: NAN for name in FEATURE_COLUMNS}
    row["ob_quote"] = 0.0
    row["ob_qty"] = 0.0
    row["ob_trade_count"] = 0.0
    row["ob_log_trade_count"] = 0.0
    row["ob_buy_quote"] = 0.0
    row["ob_sell_quote"] = 0.0
    row["ob_signed_quote"] = 0.0
    row["ob_signed_ratio"] = NAN
    return row
