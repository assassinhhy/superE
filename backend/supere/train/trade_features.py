"""Round-1 feature set (plan section 3), built on top of ``features.build``.

* calendar features (``hour_utc8``, ``dow_utc8``) are dropped by default. The plan
  keeps them only if an ablation shows a consistent gain; none is run yet.
* structure on the own timeframe: distance from the close to the 20-bar high and
  low, in ATR units (bars ``<= i`` only).
* metrics changes over one hour: log change of the global and taker long/short
  ratios (the OI change is already in ``features.build`` as ``um_oi_chg_1h``).
  Same visibility rule as the metrics levels: usable only at
  ``ts + 5 min + interval <= decision``.
* higher-timeframe context: for the closed higher-timeframe bar whose close is at
  or before the decision moment, its EMA20 slope over 3 bars, RSI14, and the
  distance of its close to its 20-bar high and low in its own ATR units.

Rows whose higher-timeframe bar has not closed yet get NaN, never a forward
fill. LightGBM handles NaN.
"""

from __future__ import annotations

import numpy as np

from supere.train import features as F
from supere.train.data import Bars, Funding, Metrics

CALENDAR = ("hour_utc8", "dow_utc8")
HIGHER_TF = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "12h"}
LSR_LAG = 12  # five-minute rows: one hour
MTF_NAMES = ("htf_ema20_slope3", "htf_rsi_14", "htf_dist_hi20_atr", "htf_dist_lo20_atr")


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return F._roll(x, w, np.max)


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    return F._roll(x, w, np.min)


def structure(bars: Bars) -> tuple[list[str], np.ndarray]:
    a = F.atr(bars, 14)
    hi20 = _roll_max(bars.h, 20)
    lo20 = _roll_min(bars.l, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_hi = (hi20 - bars.c) / a
        dist_lo = (bars.c - lo20) / a
    return ["dist_hi20_atr", "dist_lo20_atr"], np.column_stack([dist_hi, dist_lo])


def higher_context(higher: Bars) -> np.ndarray:
    """Per bar of the higher timeframe: the four ``MTF_NAMES`` values, using that bar and earlier only."""
    c = higher.c
    ema20 = F._ema(c, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.full(c.shape[0], np.nan)
        if c.shape[0] > 3:
            slope[3:] = ema20[3:] / ema20[:-3] - 1.0
    rsi = F._rsi(c, 14)
    a = F.atr(higher, 14)
    hi20 = _roll_max(higher.h, 20)
    lo20 = _roll_min(higher.l, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_hi = (hi20 - c) / a
        dist_lo = (c - lo20) / a
    return np.column_stack([slope, rsi, dist_hi, dist_lo])


def _asof(sorted_close: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the last bar whose close is ``<= query``; -1 when none."""
    return np.searchsorted(sorted_close, query, side="right") - 1


def _metrics_lsr_change(bars: Bars, metrics: Metrics) -> tuple[list[str], np.ndarray]:
    n = bars.n
    names = ["um_global_lsr_chg_1h", "um_taker_lsr_chg_1h"]
    out = np.full((n, 2), np.nan)
    if metrics.ts.shape[0] == 0:
        return names, out
    known = metrics.ts + F.METRICS_DELAY_MS + bars.interval_ms
    m = _asof(known, bars.close_time)
    prev = m - LSR_LAG
    ok = (m >= 0) & (prev >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        for col, arr in enumerate((metrics.global_lsr, metrics.taker_lsr)):
            out[ok, col] = np.log(arr[m[ok]]) - np.log(arr[prev[ok]])
    out[~np.isfinite(out)] = np.nan  # a zero or missing ratio is "not available", not inf
    return names, out


def build(
    bars: Bars,
    funding: Funding | None,
    metrics: Metrics | None,
    higher: Bars | None,
    *,
    drop_calendar: bool = True,
    use_structure: bool = True,
    use_lsr: bool = True,
    use_multi_tf: bool = True,
) -> tuple[list[str], np.ndarray]:
    names, X = F.build(bars, funding, metrics)
    cols = [X]
    names = list(names)
    if drop_calendar:
        keep = [i for i, nm in enumerate(names) if nm not in CALENDAR]
        X = X[:, keep]
        names = [names[i] for i in keep]
        cols = [X]
    if use_structure:
        nm, block = structure(bars)
        names += nm
        cols.append(block)
    if use_lsr:
        nm, block = _metrics_lsr_change(bars, metrics) if metrics is not None else (
            ["um_global_lsr_chg_1h", "um_taker_lsr_chg_1h"], np.full((bars.n, 2), np.nan))
        names += nm
        cols.append(block)
    if use_multi_tf:
        if higher is None:
            raise ValueError("多周期特征需要更高周期的 K 线")
        ctx = higher_context(higher)
        j = _asof(higher.close_time, bars.close_time)
        ok = j >= 0
        block = np.full((bars.n, len(MTF_NAMES)), np.nan)
        block[ok] = ctx[j[ok]]
        names += list(MTF_NAMES)
        cols.append(block)
    return names, np.hstack(cols).astype(np.float64)


# bp_* are the breakout-pullback distances and ages: NaN when no setup, which is a valid state.
OPTIONAL_PREFIXES = ("funding_", "um_", "bp_")


def required_columns(names: list[str]) -> list[int]:
    """Columns that must be finite for a row to be used. Funding and metrics may be NaN."""
    return [i for i, nm in enumerate(names) if not nm.startswith(OPTIONAL_PREFIXES)]
