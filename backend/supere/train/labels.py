"""Triple-barrier labels (plan 8.1).

For decision bar ``i`` (its close is the decision moment):

* entry = open of bar ``i+1`` (same expression as the backtest, plan 7.4);
* barriers = entry ± k * ATR_i, where ATR_i only uses bars ``<= i``;
* the window is bars ``i+1 .. i+H``. The first barrier touched decides the
  class: +1 for the upper one, -1 for the lower one. If none is touched by bar
  ``i+H``, the class is 0 ("flat / watch") and the exit is that bar's close.
* A bar that touches both barriers is ``ambiguous``. Without 1m data to order
  the touches, the conservative branch is taken: class -1 at the lower
  barrier. The share of ambiguous samples is reported, so the bias is visible.

``exit_bar`` is the bar whose close ends the label. It drives purging (plan 8.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from supere.train.data import Bars
from supere.train.features import atr


@dataclass
class Labels:
    idx: np.ndarray  # decision bar index of each sample (sorted)
    y: np.ndarray  # -1, 0, +1
    exit_bar: np.ndarray  # bar index whose close ends the label
    entry: np.ndarray  # open of bar idx+1
    exit_price: np.ndarray  # barrier price, or close at horizon
    ambiguous: np.ndarray  # bool
    fwd_logret: np.ndarray  # log(close at horizon / entry), auxiliary target

    @property
    def size(self) -> int:
        return int(self.idx.shape[0])


def build(bars: Bars, horizon: int, k_atr: float, atr_period: int = 14) -> Labels:
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    n = bars.n
    a = atr(bars, atr_period)

    idx: list[int] = []
    y: list[int] = []
    exit_bar: list[int] = []
    entry: list[float] = []
    exit_price: list[float] = []
    amb: list[bool] = []
    fwd: list[float] = []

    for i in range(n - horizon):
        width = a[i]
        if not np.isfinite(width) or width <= 0.0:
            continue
        e = i + 1
        px = bars.o[e]
        up = px + k_atr * width
        dn = px - k_atr * width
        last = i + horizon

        label = 0
        ex_bar = last
        ex_px = bars.c[last]
        was_amb = False
        for j in range(e, last + 1):
            hit_up = bars.h[j] >= up
            hit_dn = bars.l[j] <= dn
            if hit_up and hit_dn:
                label, ex_bar, ex_px, was_amb = -1, j, dn, True
                break
            if hit_up:
                label, ex_bar, ex_px = 1, j, up
                break
            if hit_dn:
                label, ex_bar, ex_px = -1, j, dn
                break

        idx.append(i)
        y.append(label)
        exit_bar.append(ex_bar)
        entry.append(px)
        exit_price.append(ex_px)
        amb.append(was_amb)
        fwd.append(float(np.log(bars.c[last] / px)))

    return Labels(
        idx=np.asarray(idx, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        exit_bar=np.asarray(exit_bar, dtype=np.int64),
        entry=np.asarray(entry, dtype=np.float64),
        exit_price=np.asarray(exit_price, dtype=np.float64),
        ambiguous=np.asarray(amb, dtype=bool),
        fwd_logret=np.asarray(fwd, dtype=np.float64),
    )


def class_shares(y: np.ndarray) -> dict[str, float]:
    total = max(int(y.shape[0]), 1)
    return {
        "up": float(np.sum(y == 1)) / total,
        "flat": float(np.sum(y == 0)) / total,
        "down": float(np.sum(y == -1)) / total,
    }
