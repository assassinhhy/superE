"""Limit-entry pullback trades with a partial take-profit (2026-11-05 batch 2, plan idea 1).

Scheme, per decision bar ``i`` (its close is the decision moment) and side ``d`` (+1 long, -1 short):

* **Limit price** ``L = close_i - d * pb_atr * ATR14_i``: a buy limit below the close for a long,
  a sell limit above it for a short. The order rests for ``wait`` bars, bars ``i+1 .. i+wait``.
* **Fill rule (strict).** A long fills only when a bar's low goes *through* ``L`` (``low < L``);
  a low that merely equals ``L`` does not fill. A short fills when ``high > L``. A gap through the
  limit fills at the bar's open (the better price), otherwise at ``L``. Not filled = no trade.
* **Risk unit** ``R = k_sl * ATR14_i``. From the fill price ``F``: stop at ``F - d*R``,
  first target at ``F + d*tp1_r*R`` (1R), final target at ``F + d*tp_r*R``.
* **Management** (from the fill bar ``j``): a bar that touches the stop closes the position. After
  the first target (``tp1_frac`` of the size, 50% at 1R) the rest moves its stop to breakeven ``F``
  and waits for the final target. Timeout: the rest exits at the close of bar ``i+H``.
  Conservative ordering, no 1-minute data: in the fill bar only the stop is checked; the 1R target
  can only be taken from the bar after the fill; within a bar the stop is checked before any target,
  and a target reached in the bar where the first target was set cannot be cashed until the next bar.
* **Gaps.** A stop fill is the stop, or the bar's open when it gaps through (worse). Targets fill at
  the target price (no gap bonus).
* **Costs** (R units). Main口径: taker on entry and on exit (0.05% per side). The maker-entry
  variant uses 0.02% on entry. Exit fees use the notional-weighted exit price, so the partial exit
  pays on its own part. Funding is charged on the fill price over ``(fill_time, exit_time]``.
* **Decision-level value.** An unfilled candidate counts as 0R and is not a trade; ``y`` is
  ``filled and net_taker > 0``. Its exit bar is set to ``i+H`` so purge stays conservative.

Everything is built from bars ``<= i+H`` for the outcome and bars ``<= i`` for ATR and the limit.
Only the label looks forward.
"""

from __future__ import annotations

import numpy as np

from supere.train.data import Bars, Funding
from supere.train.features import atr
from supere.train.trade_labels import FEE_MAKER, FEE_TAKER, SIDES, SideOutcome, TradeLabels

OUT_STOP = -1  # full stop before the first target
OUT_TIME = 0  # timeout (full or the rest of a partial position)
OUT_TP2 = 1  # final target after the first target
OUT_BE = 2  # breakeven stop after the first target
OUT_NOFILL = -9  # limit never filled: no trade


def build(
    bars: Bars,
    horizon: int,
    tp_r: float,
    *,
    k_sl: float = 1.0,
    pb_atr: float = 0.5,
    wait: int = 2,
    tp1_r: float = 1.0,
    tp1_frac: float = 0.5,
    atr_period: int = 14,
    funding: Funding | None = None,
) -> TradeLabels:
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    if tp_r <= tp1_r or k_sl <= 0 or pb_atr < 0 or wait < 1:
        raise ValueError("需要 tp_r > tp1_r > 0，k_sl > 0，pb_atr >= 0，wait >= 1")
    if not 0.0 < tp1_frac < 1.0:
        raise ValueError("tp1_frac 必须在 (0, 1) 内")
    wait = min(int(wait), int(horizon))
    n = bars.n
    empty = TradeLabels(idx=np.zeros(0, dtype=np.int64), entry=np.zeros(0), r_price=np.zeros(0),
                        horizon=horizon, tp_r=float(tp_r))
    if n <= horizon + 1:
        return empty

    a = atr(bars, atr_period)
    i_all = np.arange(n - horizon, dtype=np.int64)
    width = a[: n - horizon]
    ok = np.isfinite(width) & (width > 0.0)
    i = i_all[ok]
    if i.shape[0] == 0:
        return empty
    r_price = k_sl * width[ok]
    atr_i = width[ok]
    end = i + horizon  # last bar of the window, inclusive
    close_end = bars.c[end]
    entry_ref = bars.o[i + 1]
    out = TradeLabels(idx=i, entry=entry_ref, r_price=r_price, horizon=horizon, tp_r=float(tp_r))
    m = i.shape[0]
    rel = np.arange(1, wait + 1, dtype=np.int64)
    win_idx = i[:, None] + rel[None, :]  # bars i+1 .. i+wait
    t_range = np.arange(horizon, dtype=np.int64)  # management offsets from the fill bar

    for name, d in SIDES.items():
        limit = bars.c[i] - d * pb_atr * atr_i
        lo_w = bars.l[win_idx]
        hi_w = bars.h[win_idx]
        touch = (lo_w < limit[:, None]) if d > 0 else (hi_w > limit[:, None])
        filled = touch.any(axis=1)
        k_fill = np.argmax(touch, axis=1)
        fill_bar = np.where(filled, i + 1 + k_fill, -1).astype(np.int64)
        fb = np.where(filled, fill_bar, 0)
        op_fill = bars.o[fb]
        # Gap through the limit fills at the open (better); otherwise at the limit.
        fill_px = np.where(d * (op_fill - limit) < 0, op_fill, limit)
        fill_px = np.where(filled, fill_px, np.nan)

        # Management, filled rows only. Arrays cover all rows; unfilled rows stay at their defaults.
        F = np.where(filled, fill_px, 0.0)
        R = r_price
        s0 = F - d * R
        t1_lvl = F + d * tp1_r * R
        t2_lvl = F + d * tp_r * R
        active = filled.copy()
        tp1 = np.zeros(m, dtype=bool)
        notional = np.zeros(m)  # weighted exit notional, filled when the position closes
        outcome = np.full(m, OUT_TIME, dtype=np.int8)
        exit_bar = end.copy()
        for t in t_range:
            b = fill_bar + t
            act = active & (b <= end)
            if not act.any():
                break
            bb = np.where(act, b, 0)
            op_b = bars.o[bb]
            lo_b = bars.l[bb]
            hi_b = bars.h[bb]
            adv = lo_b if d > 0 else hi_b
            fav = hi_b if d > 0 else lo_b
            # Stop level: the original stop, or breakeven once the first target is in.
            level = np.where(tp1, F, s0)
            stop_hit = act & ((adv - level) * d <= 0)
            stop_px = np.where(d * (op_b - level) < 0, op_b, level)
            # Full stop (no first target yet).
            full = stop_hit & ~tp1
            notional[full] = stop_px[full]
            outcome[full] = OUT_STOP
            exit_bar[full] = bb[full]
            active &= ~full
            # Breakeven stop of the rest, after the first target.
            be = stop_hit & tp1
            notional[be] = tp1_frac * t1_lvl[be] + (1.0 - tp1_frac) * stop_px[be]
            outcome[be] = OUT_BE
            exit_bar[be] = bb[be]
            active &= ~be
            # Final target of the rest, after the first target (stop checked first in the bar).
            if t > 0:
                tgt2 = act & tp1 & ((fav - t2_lvl) * d >= 0)
                notional[tgt2] = tp1_frac * t1_lvl[tgt2] + (1.0 - tp1_frac) * t2_lvl[tgt2]
                outcome[tgt2] = OUT_TP2
                exit_bar[tgt2] = bb[tgt2]
                active &= ~tgt2
            # First target: taken from the bar after the fill (conservative), once only.
            if t > 0:
                first = act & ~tp1 & ((fav - t1_lvl) * d >= 0)
                tp1 |= first
        # Rows still open at the end of the window: timeout at the close of bar i+H.
        rest = active
        if rest.any():
            cl = close_end
            notional[rest] = np.where(tp1[rest],
                                      tp1_frac * t1_lvl[rest] + (1.0 - tp1_frac) * cl[rest],
                                      cl[rest])
            outcome[rest] = OUT_TIME
            exit_bar[rest] = end[rest]

        # Exit notional: the first-target part, when not already included above, is included here.
        # (Rows that stopped at breakeven or took the final target already added it.)
        gross = np.where(filled, d * (notional - F) / R, 0.0)

        fund_r = np.zeros(m)
        if funding is not None and funding.time.shape[0]:
            fund_cs = np.concatenate([[0.0], np.cumsum(funding.rate)])
            entry_time = bars.open_time[fb]
            exit_time = bars.open_time[np.where(filled, exit_bar, 0)] + bars.interval_ms
            lo = np.searchsorted(funding.time, entry_time, side="right")
            hi = np.searchsorted(funding.time, exit_time, side="right")
            fund_sum = fund_cs[hi] - fund_cs[lo]
            fund_r = np.where(filled, d * fund_sum * F / R, 0.0)

        exit_fee_r = np.where(filled, FEE_TAKER * notional / R, 0.0)
        net_taker = np.where(filled, gross - FEE_TAKER * F / R - exit_fee_r - fund_r, 0.0)
        net_maker = np.where(filled, gross - FEE_MAKER * F / R - exit_fee_r - fund_r, 0.0)
        outcome = np.where(filled, outcome, OUT_NOFILL).astype(np.int8)
        exit_bar = np.where(filled, exit_bar, end).astype(np.int64)

        out.sides[name] = SideOutcome(
            side=d,
            gross=gross,
            net_taker=net_taker,
            net_maker=net_maker,
            outcome=outcome,
            exit_bar=exit_bar,
            filled=filled,
            fill_bar=fill_bar,
        )
    return out
