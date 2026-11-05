"""Trade-outcome labels per side, in R units, after fees and funding (plan 2a).

Round-1 scheme of the 2026-11-05 iteration (``计划-训练口径迭代``):

* decision bar ``i`` (its close is the decision moment), entry = open of ``i+1``
  (same as the triple-barrier label and the backtest);
* R = ``k_sl * ATR14_i``. A long has its stop at ``entry - R``; a short at
  ``entry + R``. The target is ``tp_r * R`` on the favourable side (1.5R / 2R);
* the window is bars ``i+1 .. i+H``. The first of stop / target touched decides
  the trade. A bar that touches both is a stop (conservative, no 1m data). If
  neither is touched, the trade exits at the close of bar ``i+H``;
* a stop fill is the stop price, or the bar's open when it gapped through the
  stop (worse for the trader). A target fill is the target price (no gap bonus);
* ``net_taker`` = gross R - taker fee on entry and exit - funding paid in the
  holding window, in R. ``net_maker`` replaces the entry fee with the maker fee.
  The label ``y`` is ``net_taker > 0``: a trade that does not clear fees is not a
  win, even when the price moved the right way.

Funding follows ``backtest.simulate``: settlements in ``(entry_time, exit_time]``
are charged to the side, sign ``d * sum(rate)``.

Everything here is built from bars ``<= i+H`` for row ``i``'s outcome and bars
``<= i`` for its ATR. Only the outcome (the label) looks forward, which is what a
label is for; the features in ``trade_features`` never do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from supere.train.data import Bars, Funding
from supere.train.features import atr

FEE_TAKER = 0.0005  # per side, VIP0 public rate (owner's account rate takes precedence)
FEE_MAKER = 0.0002

SIDES = {"long": 1, "short": -1}


@dataclass
class SideOutcome:
    side: int  # +1 long, -1 short
    gross: np.ndarray  # R, before fees and funding
    net_taker: np.ndarray  # R, taker entry and taker exit, after funding
    net_maker: np.ndarray  # R, maker entry and taker exit, after funding
    outcome: np.ndarray  # int8: +1 target, -1 stop, 0 timeout
    exit_bar: np.ndarray  # int64 bar whose close (or fill) ends the trade

    @property
    def y(self) -> np.ndarray:
        return (self.net_taker > 0).astype(np.int64)


@dataclass
class TradeLabels:
    idx: np.ndarray  # decision bar of each sample, sorted
    entry: np.ndarray  # open of bar idx+1
    r_price: np.ndarray  # R in price units = k_sl * ATR14_idx
    horizon: int
    tp_r: float
    sides: dict[str, SideOutcome] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.idx.shape[0])


def build(
    bars: Bars,
    horizon: int,
    tp_r: float,
    k_sl: float = 1.0,
    atr_period: int = 14,
    funding: Funding | None = None,
) -> TradeLabels:
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    if tp_r <= 0 or k_sl <= 0:
        raise ValueError("tp_r 和 k_sl 必须为正")
    n = bars.n
    empty = TradeLabels(
        idx=np.zeros(0, dtype=np.int64),
        entry=np.zeros(0),
        r_price=np.zeros(0),
        horizon=horizon,
        tp_r=float(tp_r),
    )
    if n <= horizon + 1:
        return empty

    a = atr(bars, atr_period)
    # Decision i needs bars i+1..i+H, so i runs over 0..n-H-1.
    i_all = np.arange(n - horizon, dtype=np.int64)
    width = a[: n - horizon]
    ok = np.isfinite(width) & (width > 0.0)
    i = i_all[ok]
    if i.shape[0] == 0:
        return empty
    r_price = k_sl * width[ok]
    entry_bar = i + 1
    entry = bars.o[entry_bar]

    # Row s of each view is bars s..s+H-1, so row i+1 is the window of decision i.
    hw = sliding_window_view(bars.h, horizon)[entry_bar]
    lw = sliding_window_view(bars.l, horizon)[entry_bar]
    ow = sliding_window_view(bars.o, horizon)[entry_bar]
    close_end = bars.c[i + horizon]

    fund_cs = None
    if funding is not None and funding.time.shape[0]:
        fund_cs = np.concatenate([[0.0], np.cumsum(funding.rate)])
    entry_time = bars.open_time[entry_bar]
    exit_time_at = lambda eb: bars.open_time[eb] + bars.interval_ms  # noqa: E731

    out = TradeLabels(idx=i, entry=entry, r_price=r_price, horizon=horizon, tp_r=float(tp_r))
    fee_taker_r = 2.0 * FEE_TAKER * entry / r_price
    fee_maker_r = (FEE_TAKER + FEE_MAKER) * entry / r_price

    for name, d in SIDES.items():
        stop = entry - d * r_price
        tgt = entry + d * tp_r * r_price
        if d > 0:
            stop_hit = lw <= stop[:, None]
            tgt_hit = hw >= tgt[:, None]
        else:
            stop_hit = hw >= stop[:, None]
            tgt_hit = lw <= tgt[:, None]
        any_stop = stop_hit.any(axis=1)
        any_tgt = tgt_hit.any(axis=1)
        k_stop = np.argmax(stop_hit, axis=1)
        k_tgt = np.argmax(tgt_hit, axis=1)
        rows = np.arange(i.shape[0])

        # Same bar touching both: the stop is taken (conservative tie).
        stop_first = any_stop & (~any_tgt | (k_stop <= k_tgt))
        tgt_first = any_tgt & ~stop_first
        timeout = ~any_stop & ~any_tgt

        open_at_stop = ow[rows, k_stop]
        if d > 0:
            stop_fill = np.minimum(open_at_stop, stop)
        else:
            stop_fill = np.maximum(open_at_stop, stop)

        exit_px = np.where(stop_first, stop_fill, np.where(tgt_first, tgt, close_end))
        exit_bar = np.where(
            stop_first,
            entry_bar + k_stop,
            np.where(tgt_first, entry_bar + k_tgt, i + horizon),
        ).astype(np.int64)
        outcome = np.where(stop_first, -1, np.where(tgt_first, 1, 0)).astype(np.int8)
        gross = d * (exit_px - entry) / r_price

        if fund_cs is not None:
            exit_time = exit_time_at(exit_bar)
            lo = np.searchsorted(funding.time, entry_time, side="right")
            hi = np.searchsorted(funding.time, exit_time, side="right")
            fund_sum = fund_cs[hi] - fund_cs[lo]
            fund_r = d * fund_sum * entry / r_price
        else:
            fund_r = np.zeros_like(gross)

        out.sides[name] = SideOutcome(
            side=d,
            gross=gross,
            net_taker=gross - fee_taker_r - fund_r,
            net_maker=gross - fee_maker_r - fund_r,
            outcome=outcome,
            exit_bar=exit_bar,
        )
    return out
