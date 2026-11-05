"""Breakout-pullback candidates (plan 2b, meta-labelling gate).

A long candidate at decision bar ``i`` (its close):

* a breakout happened at bar ``b`` in ``i-recent .. i-1``: close above the highest
  high of the ``lookback`` bars before ``b``. The most recent such ``b`` is used;
* the pullback touched the breakout level: the lowest low over ``b+1 .. i`` is at
  or below ``level + touch_atr * ATR_i``;
* the retest held: close ``i`` is above the level.

A short is the mirror image. The model then only decides on candidate bars; the
candidate rule itself is fixed in advance (not tuned on the test blocks).

Everything uses bars ``<= i``. The breakout-level distance is a feature, so the
model can see how far the close sits from the level.
"""

from __future__ import annotations

import numpy as np

from supere.train import features as F
from supere.train.data import Bars

NAMES = ("bp_long_dist_atr", "bp_short_dist_atr", "bp_long_age", "bp_short_age")


def _prev_extreme(x: np.ndarray, lookback: int, fn) -> np.ndarray:
    """``out[b]`` = fn over ``x[b-lookback .. b-1]`` (bars strictly before ``b``)."""
    roll = F._roll(x, lookback, fn)
    out = np.full(x.shape[0], np.nan)
    out[1:] = roll[:-1]
    return out


def masks_and_features(bars: Bars, lookback: int = 20, recent: int = 10, touch_atr: float = 0.5):
    """Return ``(long_mask, short_mask, names, feature_block)`` for every bar."""
    n = bars.n
    a = F.atr(bars, 14)
    hi_prev = _prev_extreme(bars.h, lookback, np.max)
    lo_prev = _prev_extreme(bars.l, lookback, np.min)
    brk_up = bars.c > hi_prev  # NaN compares False
    brk_dn = bars.c < lo_prev

    i_all = np.arange(n, dtype=np.int64)
    out_long = np.zeros(n, dtype=bool)
    out_short = np.zeros(n, dtype=bool)
    lev_up = np.full(n, np.nan)
    lev_dn = np.full(n, np.nan)
    age_up = np.full(n, np.nan)
    age_dn = np.full(n, np.nan)

    found_up = np.zeros(n, dtype=bool)
    found_dn = np.zeros(n, dtype=bool)
    # Pullback extremes over (b, i], where b is the breakout bar. Captured once, when
    # the breakout is first found, so the window is exactly the bars after the break.
    min_low = np.full(n, np.inf)
    max_high = np.full(n, -np.inf)
    low_at_found = np.full(n, np.nan)
    high_at_found = np.full(n, np.nan)
    for k in range(1, recent + 1):
        j = i_all - k + 1  # newest bar added to the window (b, i] for this k
        valid = j >= 0
        jj = np.where(valid, j, 0)
        min_low = np.where(valid, np.minimum(min_low, bars.l[jj]), min_low)
        max_high = np.where(valid, np.maximum(max_high, bars.h[jj]), max_high)
        b = i_all - k  # breakout candidate, most recent first
        ok_b = b >= 0
        bb = np.where(ok_b, b, 0)
        new_up = ok_b & brk_up[bb] & ~found_up
        new_dn = ok_b & brk_dn[bb] & ~found_dn
        lev_up = np.where(new_up, hi_prev[bb], lev_up)
        lev_dn = np.where(new_dn, lo_prev[bb], lev_dn)
        age_up = np.where(new_up, float(k), age_up)
        age_dn = np.where(new_dn, float(k), age_dn)
        low_at_found = np.where(new_up, min_low, low_at_found)
        high_at_found = np.where(new_dn, max_high, high_at_found)
        found_up |= new_up
        found_dn |= new_dn

    with np.errstate(invalid="ignore"):
        retest_up = found_up & (low_at_found <= lev_up + touch_atr * a) & (bars.c > lev_up)
        retest_dn = found_dn & (high_at_found >= lev_dn - touch_atr * a) & (bars.c < lev_dn)
    out_long[:] = retest_up
    out_short[:] = retest_dn

    with np.errstate(invalid="ignore", divide="ignore"):
        dist_up = np.where(found_up, (bars.c - lev_up) / a, np.nan)
        dist_dn = np.where(found_dn, (lev_dn - bars.c) / a, np.nan)
    block = np.column_stack([dist_up, dist_dn, age_up, age_dn])
    return out_long, out_short, list(NAMES), block
