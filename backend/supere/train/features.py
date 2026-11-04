"""Causal features. Row ``i`` may only use information known at bar ``i``'s close.

Rules (plan 8.2 and 8.3):

* Price features read bars ``0..i``. Rolling windows never look forward.
* The decision moment for row ``i`` is its close, ``open_time + interval``.
  The entry for a signal is the next bar's open, which is the same instant.
* Funding is the last settled rate with ``funding_time < decision``. Strict
  inequality, so a settlement exactly at the decision moment is not used.
* Metrics bars carry the bar start. A value only counts as known at
  ``ts + 5 min`` (plan 5.5 / 8.2), and training adds one more interval as a
  conservative delay. So a metrics row is usable when
  ``ts + 5min + interval <= decision``.
* NaN means "not available". LightGBM handles NaN natively. Nothing is
  forward-filled across a missing value.

``truncation`` tests in ``tests/test_train_features.py`` check these rules.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from supere.train.data import Bars, Funding, Metrics

METRICS_DELAY_MS = 5 * 60_000

# Names that must be finite for a row to be used. Funding and metrics are
# optional inputs and may be NaN.
CORE_FEATURES: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "rsi_14",
    "macd_hist_pct",
    "atr_pct",
    "rv_20",
    "parkinson_20",
    "donchian_pos_20",
    "bb_pos_20",
    "bb_width_20",
    "vol_ratio_20",
    "taker_buy_ratio",
    "log_trades",
    "log_avg_trade_quote",
    "body_pct",
    "range_pct",
    "hour_utc8",
    "dow_utc8",
)


def _roll(x: np.ndarray, w: int, fn) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    if x.shape[0] >= w:
        out[w - 1 :] = fn(sliding_window_view(x, w), axis=1)
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    prev = x[0]
    for i in range(x.shape[0]):
        prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full(c.shape[0], np.nan)
    if c.shape[0] <= n:
        return out
    d = np.diff(c)
    gains = np.maximum(d, 0.0)
    losses = np.maximum(-d, 0.0)
    ag = gains[:n].mean()
    al = losses[:n].mean()
    out[n] = _rsi_value(ag, al)
    for i in range(n + 1, c.shape[0]):
        ag = (ag * (n - 1) + gains[i - 1]) / n
        al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = _rsi_value(ag, al)
    return out


def _rsi_value(ag: float, al: float) -> float:
    if al == 0.0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def true_range(bars: Bars) -> np.ndarray:
    prev_c = np.concatenate([[bars.c[0]], bars.c[:-1]])
    return np.maximum.reduce(
        [bars.h - bars.l, np.abs(bars.h - prev_c), np.abs(bars.l - prev_c)]
    )


def atr(bars: Bars, period: int = 14) -> np.ndarray:
    """Simple average of true range over ``period`` bars ending at ``i``."""
    return _roll(true_range(bars), period, np.mean)


def _asof_index(sorted_times: np.ndarray, query: np.ndarray, side: str) -> np.ndarray:
    """Index of the last element satisfying the comparison; -1 when none."""
    if sorted_times.shape[0] == 0:
        return np.full(query.shape[0], -1, dtype=np.int64)
    if side == "left":  # element < query
        return np.searchsorted(sorted_times, query, side="left") - 1
    return np.searchsorted(sorted_times, query, side="right") - 1  # element <= query


def build(
    bars: Bars,
    funding: Funding | None = None,
    metrics: Metrics | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return ``(names, X)`` with ``X`` shaped ``(bars.n, len(names))``."""
    n = bars.n
    c, h, low, o = bars.c, bars.h, bars.l, bars.o
    feats: dict[str, np.ndarray] = {}

    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.log(c)
        r1 = np.full(n, np.nan)
        r1[1:] = np.diff(log_c)
        for k in (1, 3, 5, 10):
            lagged = np.full(n, np.nan)
            if n > k:
                lagged[k:] = log_c[:-k]
            feats[f"ret_{k}"] = log_c - lagged
        feats["rsi_14"] = _rsi(c, 14)

        macd = _ema(c, 12) - _ema(c, 26)
        signal = _ema(macd, 9)
        feats["macd_hist_pct"] = (macd - signal) / c

        tr_atr = atr(bars, 14)
        feats["atr_pct"] = tr_atr / c
        feats["rv_20"] = _roll(r1, 20, np.std)

        hl = np.log(h / low)
        feats["parkinson_20"] = np.sqrt(_roll(hl * hl, 20, np.mean) / (4.0 * np.log(2.0)))

        hi20 = _roll(h, 20, np.max)
        lo20 = _roll(low, 20, np.min)
        feats["donchian_pos_20"] = (c - lo20) / (hi20 - lo20)

        ma20 = _roll(c, 20, np.mean)
        sd20 = _roll(c, 20, np.std)
        feats["bb_pos_20"] = (c - ma20) / (2.0 * sd20)
        feats["bb_width_20"] = (4.0 * sd20) / ma20

        vol_ma20 = _roll(bars.volume, 20, np.mean)
        feats["vol_ratio_20"] = bars.volume / vol_ma20
        feats["taker_buy_ratio"] = bars.taker_buy_base / bars.volume
        feats["log_trades"] = np.log1p(bars.trades)
        feats["log_avg_trade_quote"] = np.log1p(bars.quote_volume / np.maximum(bars.trades, 1.0))

        feats["body_pct"] = (c - o) / o
        feats["range_pct"] = (h - low) / c

    # Time features use the decision moment (close), shown in UTC+8.
    decision = bars.close_time
    utc8_ms = decision + 8 * 3_600_000
    feats["hour_utc8"] = ((utc8_ms // 3_600_000) % 24).astype(np.float64)
    days = utc8_ms // 86_400_000
    feats["dow_utc8"] = ((days + 3) % 7).astype(np.float64)  # 1970-01-01 was a Thursday

    # Funding: last settlement strictly before the decision moment.
    if funding is not None and funding.time.shape[0]:
        j = _asof_index(funding.time, decision, side="left")
        ok = j >= 0
        rate = np.full(n, np.nan)
        since = np.full(n, np.nan)
        rate[ok] = funding.rate[j[ok]]
        since[ok] = (decision[ok] - funding.time[j[ok]]) / 3_600_000.0
        feats["funding_last"] = rate
        feats["funding_hours_since"] = since
    else:
        feats["funding_last"] = np.full(n, np.nan)
        feats["funding_hours_since"] = np.full(n, np.nan)

    # Metrics: latest row whose conservative availability time is <= decision.
    if metrics is not None and metrics.ts.shape[0]:
        known = metrics.ts + METRICS_DELAY_MS + bars.interval_ms
        m = _asof_index(known, decision, side="right")
        ok = m >= 0
        oi_chg = np.full(n, np.nan)
        # One hour of metrics history (12 five-minute rows) before the usable row.
        prev = m - 12
        both = ok & (prev >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            oi_chg[both] = np.log(metrics.oi[m[both]]) - np.log(metrics.oi[prev[both]])
        feats["um_oi_chg_1h"] = oi_chg
        for name, arr in (
            ("um_top_pos_lsr", metrics.top_pos_lsr),
            ("um_top_acc_lsr", metrics.top_acc_lsr),
            ("um_global_lsr", metrics.global_lsr),
            ("um_taker_lsr", metrics.taker_lsr),
        ):
            col = np.full(n, np.nan)
            col[ok] = arr[m[ok]]
            feats[name] = col
    else:
        for name in ("um_oi_chg_1h", "um_top_pos_lsr", "um_top_acc_lsr", "um_global_lsr", "um_taker_lsr"):
            feats[name] = np.full(n, np.nan)

    names = list(feats)
    X = np.column_stack([feats[k] for k in names]).astype(np.float64)
    return names, X


def feature_hash(names: list[str], extra: dict | None = None) -> str:
    """Hash of the feature list and the settings that shaped it (plan 7.2)."""
    import json

    payload = json.dumps({"names": names, "extra": extra or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
