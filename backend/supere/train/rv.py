"""Causal realized measures from 1-minute bars (plan v3, stage 1).

Everything here is computed from bars that have *closed* at or before a decision
moment.  For a decision at ``t`` (the close of an interval bar) the 1-minute bar
with ``open_time == t - 60s`` is the last one included; a bar opening at ``t``
is still forming and is never read.

Labels (future realized variance over ``[t, t + h]``) are produced only for
training and are never used as features.

Windows are expressed in minutes and include the model horizon plus fixed
6h/24h/7d scales.  Seasonal factors use only the same hour-of-day / day-of-week
from the *past* 28 days, never the full sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

from supere.train import data, features

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
EPS = 1e-12
SEASON_DAYS = 28


@dataclass
class RVFeatures:
    names: list[str]
    X: np.ndarray                 # (n_decisions, n_features), aligned to decision_ms
    rv_future: np.ndarray         # future interval RV (label), NaN when unavailable
    rv_past: np.ndarray           # past interval RV
    decision_ms: np.ndarray
    first_1m_ms: int
    last_1m_ms: int
    n_1m: int

    @property
    def m(self) -> int:
        return int(self.decision_ms.shape[0])


def _cum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])


def _window_sum(cum: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """``cum[hi] - cum[lo]`` with clamped, sorted-in-time indices."""
    return cum[np.clip(hi, 0, cum.shape[0] - 1)] - cum[np.clip(lo, 0, cum.shape[0] - 1)]


def _rolling_mean_std(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean/std of a 1-minute series (NaN treated as missing)."""
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    c1 = _cum(x0)
    c2 = _cum(x0 * x0)
    cn = _cum(ok.astype(np.float64))
    idx = np.arange(x.shape[0] + 1)
    lo = np.maximum(idx - w, 0)
    n = cn[idx] - cn[lo]
    s1 = c1[idx] - c1[lo]
    s2 = c2[idx] - c2[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = s2 / n - mean * mean
    var = np.where((n > 1) & (var > 0), var, np.nan)
    return mean, np.sqrt(var)


def _hhi_window_sum(cum: np.ndarray, idx: np.ndarray, w: int) -> np.ndarray:
    lo = np.maximum(idx - w, 0)
    return _window_sum(cum, idx, lo)


def _seasonal_median(hour_rv: np.ndarray, hour_idx: np.ndarray, lag_hours: int,
                     repeats: int) -> np.ndarray:
    """Median of the same slot over the previous ``repeats`` cycles (causal)."""
    n = hour_idx.shape[0]
    vals = np.full((repeats, n), np.nan)
    for k in range(1, repeats + 1):
        j = hour_idx - lag_hours * k
        good = (j >= 0) & (j < hour_rv.shape[0])
        vals[k - 1, good] = hour_rv[j[good]]
    with np.errstate(invalid="ignore"):
        out = np.nanmedian(vals, axis=0)
    # Require at least a week of history, otherwise the factor is not estimated.
    count = np.sum(np.isfinite(vals), axis=0)
    return np.where(count >= min(7, repeats), out, np.nan)


def _micro_columns(bars_1m: data.Bars) -> dict[str, np.ndarray]:
    """1-minute taker/quote imbalance, trade count and average trade size."""
    vol = np.maximum(bars_1m.volume, EPS)
    taker = (2.0 * bars_1m.taker_buy_base - bars_1m.volume) / vol
    taker = np.clip(taker, -1.0, 1.0)
    cols: dict[str, np.ndarray] = {
        "taker_imb": taker,
        "log_trades": np.log1p(np.maximum(bars_1m.trades, 0.0)),
        "log_avg_trade": np.log1p(bars_1m.quote_volume / np.maximum(bars_1m.trades, 1.0)),
        "log_volume": np.log1p(np.maximum(bars_1m.quote_volume, 0.0)),
    }
    if bars_1m.taker_buy_quote is not None:
        qv = np.maximum(bars_1m.quote_volume, EPS)
        q = (2.0 * bars_1m.taker_buy_quote - bars_1m.quote_volume) / qv
        cols["quote_imb"] = np.clip(q, -1.0, 1.0)
    else:
        cols["quote_imb"] = np.full(bars_1m.n, np.nan)
    return cols


def build(bars_1m: data.Bars, decision_ms: np.ndarray, interval_ms: int,
          windows_min: tuple[int, ...] | None = None) -> RVFeatures:
    """Realized measures at each decision moment.

    ``decision_ms`` must be sorted and every value must be a bar close.  The
    returned ``X`` has one row per decision, in the same order.
    """
    if bars_1m.interval != "1m":
        raise ValueError("rv.build 需要 1 分钟 K 线")
    n1 = bars_1m.n
    decision_ms = np.asarray(decision_ms, dtype=np.int64)
    n = decision_ms.shape[0]
    if n1 < 2 or n == 0:
        names = ["rv_empty"]
        return RVFeatures(names, np.full((n, 1), np.nan), np.full(n, np.nan),
                          np.full(n, np.nan), decision_ms, 0, 0, n1)

    open_t = bars_1m.open_time
    close_t = open_t + MINUTE_MS
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(bars_1m.c)
        r = np.full(n1, np.nan)
        r[1:] = np.diff(logc)
    gap = np.zeros(n1, dtype=bool)
    if n1 >= 2:
        gap[1:] = np.diff(open_t) != MINUTE_MS
    valid = np.isfinite(r) & ~gap
    r0 = np.where(valid, r, 0.0)
    prev_valid = np.concatenate([[False], valid[:-1]])

    # Index of the first 1m bar opening at/after the decision: bars before it
    # have open_time < t and close <= t, so they are known.
    idx = np.searchsorted(open_t, decision_ms, side="left")
    idx = np.clip(idx, 0, n1)
    h = max(1, int(round(interval_ms / MINUTE_MS)))
    if windows_min is None:
        windows = sorted({h, 60, 360, 1440, 10080})
    else:
        windows = sorted(set(int(w) for w in windows_min) | {h})
    windows = [w for w in windows if w > 0]

    r2 = r0 * r0
    r4 = r2 * r2
    bpv_pair = np.abs(r0) * np.abs(np.concatenate([[0.0], r0[:-1]]))
    bpv_pair[~valid] = 0.0
    bpv_pair[~prev_valid] = 0.0
    park = np.where(valid, np.log(np.maximum(bars_1m.h, EPS) / np.maximum(bars_1m.l, EPS)) ** 2
                    / (4.0 * np.log(2.0)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gk = 0.5 * np.log(np.maximum(bars_1m.h, EPS) / np.maximum(bars_1m.l, EPS)) ** 2 \
             - (2.0 * np.log(2.0) - 1.0) * np.log(np.maximum(bars_1m.c, EPS)
                                                  / np.maximum(bars_1m.o, EPS)) ** 2
    gk = np.where(valid & np.isfinite(gk), gk, 0.0)
    pos = np.where(valid & (r0 > 0), r2, 0.0)
    neg = np.where(valid & (r0 < 0), r2, 0.0)
    cnt1 = valid.astype(np.float64)

    cum_r2 = _cum(r2)
    cum_r4 = _cum(r4)
    cum_bpv = _cum(bpv_pair)
    cum_park = _cum(park)
    cum_gk = _cum(gk)
    cum_pos = _cum(pos)
    cum_neg = _cum(neg)
    cum_cnt = _cum(cnt1)

    cols: dict[str, np.ndarray] = {}

    # ---- realized measures per window
    for w in windows:
        lo = np.maximum(idx - w, 0)
        rv = _window_sum(cum_r2, idx, lo)
        bpv = 1.5707963267948966 * _window_sum(cum_bpv, idx, lo)
        rq = _window_sum(cum_r4, idx, lo)
        cnt = _window_sum(cum_cnt, idx, lo)
        rsp = _window_sum(cum_pos, idx, lo)
        rsn = _window_sum(cum_neg, idx, lo)
        n_eff = np.maximum(cnt, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            jump = np.maximum(0.0, 1.0 - bpv / np.maximum(rv, EPS))
            rs_ratio = np.log((rsp + EPS) / (rsn + EPS))
            signed_jump = (rsp - rsn) / np.maximum(rv, EPS)
            park_m = _window_sum(cum_park, idx, lo) / n_eff
            gk_m = _window_sum(cum_gk, idx, lo) / n_eff
            rq_scaled = (n_eff / 3.0) * rq
        label = f"{w}m"
        cols[f"rv_{label}"] = rv
        cols[f"logrv_{label}"] = np.log(rv + EPS)
        cols[f"bpv_{label}"] = bpv
        cols[f"log_bpv_{label}"] = np.log(bpv + EPS)
        cols[f"rq_{label}"] = np.log(rq_scaled + EPS)
        cols[f"jump_{label}"] = jump
        cols[f"rsp_{label}"] = rsp
        cols[f"rsn_{label}"] = rsn
        cols[f"log_rsp_{label}"] = np.log(rsp + EPS)
        cols[f"log_rsn_{label}"] = np.log(rsn + EPS)
        cols[f"rs_ratio_{label}"] = rs_ratio
        cols[f"signed_jump_{label}"] = signed_jump
        cols[f"park_{label}"] = park_m
        cols[f"gk_{label}"] = gk_m

    # ---- short/long ratios and a causal z-score of the horizon log RV
    lo_h = np.maximum(idx - h, 0)
    rv_h = _window_sum(cum_r2, idx, lo_h)
    lo_long = np.maximum(idx - 1440, 0)
    rv_long = _window_sum(cum_r2, idx, lo_long)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["rv_ratio_h_1d"] = np.log((rv_h + EPS) / (rv_long + EPS))
        cols["rv_ratio_h_6h"] = np.log((rv_h + EPS) / (_window_sum(cum_r2, idx, np.maximum(idx - 360, 0)) + EPS))
    logrv_h = np.log(rv_h + EPS)
    # z-score against the previous 24h of the same causal series
    logrv_series = np.full(n1, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logrv_series[:] = np.log(r2 + EPS)
    mean1, std1 = _rolling_mean_std(logrv_series, 1440)
    # Use the value at idx-1 so the current minute is not in its own baseline.
    base_i = np.clip(idx - 1, 0, n1 - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["rv_z_1d"] = (logrv_h - mean1[base_i]) / std1[base_i]

    # ---- UTC hour-of-day / day-of-week seasonality (past only)
    hour_id = (open_t // HOUR_MS).astype(np.int64)
    h0 = int(hour_id.min())
    n_hours = int(hour_id.max() - h0 + 1)
    hour_rv = np.bincount(hour_id - h0, weights=r2, minlength=n_hours).astype(np.float64)
    hour_cnt = np.bincount(hour_id - h0, weights=cnt1, minlength=n_hours).astype(np.float64)
    hour_rv = np.where(hour_cnt > 0, hour_rv, np.nan)
    decision_hour = ((decision_ms // HOUR_MS) - h0).astype(np.int64)
    seas_hour = _seasonal_median(hour_rv, decision_hour, 24, SEASON_DAYS)
    seas_dow = _seasonal_median(hour_rv, decision_hour, 168, 4)
    hour_of_day = ((decision_ms // HOUR_MS) % 24).astype(np.float64)
    dow = ((decision_ms // DAY_MS) + 3).astype(np.float64) % 7.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["seas_hour_med_28d"] = seas_hour
        cols["seas_dow_med_4w"] = seas_dow
        cols["seas_ratio_hour"] = np.log((rv_h + EPS) / (seas_hour + EPS))
        cols["seas_ratio_dow"] = np.log((rv_h + EPS) / (seas_dow + EPS))
    cols["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
    cols["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)
    cols["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    cols["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    # ---- microstructure (1m taker flow, trade counts, average size)
    micro = _micro_columns(bars_1m)
    micro_windows = [w for w in (h, 60, 360, 1440) if w in windows or w == h]
    for key, series in micro.items():
        series = np.asarray(series, dtype=np.float64)
        s0 = np.where(np.isfinite(series), series, 0.0)
        c = _cum(s0)
        cn = _cum(np.isfinite(series).astype(np.float64))
        for w in sorted(set(micro_windows)):
            lo = np.maximum(idx - w, 0)
            n_eff = np.maximum(_window_sum(cn, idx, lo), 1.0)
            mean = _window_sum(c, idx, lo) / n_eff
            cols[f"micro_{key}_{w}m"] = mean
    # Trade-count / quote-volume surprise versus the prior 24h of 1m means.
    mean_t, std_t = _rolling_mean_std(micro["log_trades"], 1440)
    mean_v, std_v = _rolling_mean_std(micro["log_volume"], 1440)
    base_i = np.clip(idx - 1, 0, n1 - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for w in sorted(set(micro_windows)):
            lo = np.maximum(idx - w, 0)
            n_eff = np.maximum(_window_sum(_cum(np.isfinite(micro["log_trades"]).astype(float)), idx, lo), 1.0)
            mt = _window_sum(_cum(np.where(np.isfinite(micro["log_trades"]), micro["log_trades"], 0.0)), idx, lo) / n_eff
            mv = _window_sum(_cum(np.where(np.isfinite(micro["log_volume"]), micro["log_volume"], 0.0)), idx, lo) / n_eff
            cols[f"micro_trades_z_{w}m"] = (mt - mean_t[base_i]) / std_t[base_i]
            cols[f"micro_volume_z_{w}m"] = (mv - mean_v[base_i]) / std_v[base_i]
    # Interactions: signed flow is more informative when volatility is high.
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["micro_imb_x_rv_h"] = cols[f"micro_taker_imb_{h}m"] * rv_h
        cols["micro_imb_x_trade_z_h"] = cols[f"micro_taker_imb_{h}m"] * cols[f"micro_trades_z_{h}m"]
        cols["micro_jump_x_imb_h"] = cols.get(f"jump_{h}m", np.nan) * cols[f"micro_taker_imb_{h}m"]

    # ---- labels: past and future interval RV (future is a label only)
    fut_idx = np.searchsorted(open_t, decision_ms + interval_ms, side="left")
    fut_idx = np.clip(fut_idx, 0, n1)
    rv_past = _window_sum(cum_r2, idx, np.maximum(idx - h, 0))
    rv_future = _window_sum(cum_r2, fut_idx, idx)
    rv_future = np.where((fut_idx > idx) & (fut_idx <= n1), rv_future, np.nan)

    names = list(cols)
    X = np.column_stack([cols[k] for k in names]).astype(np.float64)
    return RVFeatures(names=names, X=X, rv_future=rv_future, rv_past=rv_past,
                      decision_ms=decision_ms, first_1m_ms=int(open_t[0]),
                      last_1m_ms=int(close_t[-1]), n_1m=n1)


# ---------------------------------------------------------------- cache

def cache_file(data_root: Path, interval: str, snapshot_ms: int, limit_bars: int | None) -> Path:
    tag = f"{interval}_{snapshot_ms}_{limit_bars or 'all'}"
    return Path(data_root) / "cache_v3" / f"rv_{tag}.npz"


def save_cache(path: Path, rvf: RVFeatures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            names=np.asarray(rvf.names, dtype=object),
            X=rvf.X, rv_future=rvf.rv_future, rv_past=rvf.rv_past,
            decision_ms=rvf.decision_ms,
            meta=np.asarray([rvf.first_1m_ms, rvf.last_1m_ms, rvf.n_1m], dtype=np.int64),
        )
    os.replace(tmp, path)


def load_cache(path: Path) -> RVFeatures | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        names = [str(x) for x in z["names"].tolist()]
        meta = z["meta"]
        return RVFeatures(names=names, X=z["X"], rv_future=z["rv_future"],
                          rv_past=z["rv_past"], decision_ms=z["decision_ms"],
                          first_1m_ms=int(meta[0]), last_1m_ms=int(meta[1]), n_1m=int(meta[2]))


def build_cached(data_root: Path, bars_1m: data.Bars, decision_ms: np.ndarray,
                 interval_ms: int, interval: str, snapshot_ms: int,
                 limit_bars: int | None, *, use_cache: bool = True) -> RVFeatures:
    path = cache_file(data_root, interval, snapshot_ms, limit_bars)
    if use_cache:
        cached = load_cache(path)
        if cached is not None and cached.m == decision_ms.shape[0] and np.array_equal(cached.decision_ms, decision_ms):
            return cached
    rvf = build(bars_1m, decision_ms, interval_ms)
    if use_cache:
        save_cache(path, rvf)
    return rvf


# ---------------------------------------------------------------- HAR / QLIKE

HAR_CORE: tuple[str, ...] = ("logrv_{h}m", "logrv_360m", "logrv_1440m")
HAR_RS: tuple[str, ...] = ("logrv_{h}m", "logrv_360m", "logrv_1440m",
                           "log_rsp_{h}m", "log_rsn_{h}m",
                           "log_rsp_360m", "log_rsn_360m")
HAR_CJ: tuple[str, ...] = ("logrv_{h}m", "logrv_360m", "logrv_1440m",
                           "log_bpv_{h}m", "jump_{h}m", "log_bpv_360m", "jump_360m")


def har_matrix(rvf: RVFeatures, kind: str, h: int) -> tuple[list[str], np.ndarray]:
    """Small, stable log-HAR design matrix. ``kind``: har | har_rs | har_cj."""
    names = {"har": HAR_CORE, "har_rs": HAR_RS, "har_cj": HAR_CJ}[kind]
    out_names = [n.format(h=h) for n in names]
    missing = [n for n in out_names if n not in rvf.names]
    if missing:
        raise ValueError(f"HAR 特征缺失: {missing}")
    cols = [rvf.X[:, rvf.names.index(n)] for n in out_names]
    return out_names, np.column_stack(cols)


def qlike(rv: np.ndarray, sigma2: np.ndarray) -> float:
    """QLIKE on variance forecasts (lower is better)."""
    ok = np.isfinite(rv) & np.isfinite(sigma2) & (rv > 0) & (sigma2 > 0)
    if not ok.any():
        return float("nan")
    r = rv[ok] / sigma2[ok]
    return float(np.mean(r - np.log(r) - 1.0))
