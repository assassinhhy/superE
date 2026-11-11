"""Version 3 point/interval forecast (plan 追加 2026-11-05 17:30).

Decomposes the forecast into *location*, *scale* and *shape*:

* scale: EWMA / GARCH / log-HAR / HAR-RS / HAR-CJ / a parsimonious
  Realized-GARCH-style regression, all causal and fitted inside each CV fold;
* shape: LightGBM quantile regressors on ``y / sigma`` or a scale-only
  empirical/Student-t residual distribution;
* calibration: split CQR, normalized CQR, exponentially weighted normalized
  conformal, and normalized CQR + ACI;
* median: if the model's q50 does not beat "current price" on CV, the served
  median falls back to the current price and the output says so.

The module reuses :mod:`supere.train.quantile` for folds, LightGBM helpers,
baselines and metrics; it does not rewrite the v2 pipeline.  Every selection is
made on purged walk-forward CV.  The 15% holdout is evaluated once.

Read-only: opens SQLite with ``mode=ro``.  Never places orders, never reads keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

from supere.intervals import INTERVAL_MS
from supere.timeutil import UTC8 as _UTC8
from supere.train import altdata, data, features, newsfeat, quantile as q, rv

log = logging.getLogger("supere.train.quantile_v3")

PRODUCT_INTERVALS = q.PRODUCT_INTERVALS
ALPHAS = q.ALPHAS
AKEY = q.AKEY
FEATURE_MODES = ("base", "rv", "rv_season", "rv_micro", "all")
SIGMA_KINDS = ("ewma", "garch", "har", "har_rs", "har_cj", "rgarch")
SHAPES = ("quantile", "empirical", "student_t", "blend")
CONFORMALS = ("none", "cqr", "norm", "weighted", "norm_aci")
RV_PREFIXES = ("rv_", "logrv_", "bpv_", "log_bpv_", "rq_", "jump_", "rsp_", "rsn_",
               "rs_ratio_", "signed_jump_", "park_", "gk_")
SEASON_PREFIXES = ("seas_", "hour_", "dow_")
MICRO_PREFIXES = ("micro_",)


@dataclass(frozen=True)
class V3Config:
    interval: str
    symbol: str = "ETHUSDT"
    market: str = "um"
    limit_bars: int | None = None
    snapshot_ms: int | None = q.SNAPSHOT_MS
    features: str = "base"
    sigma_model: str = "ewma"
    shape: str = "quantile"
    normalize: bool = True
    conformal: str = "none"
    calib_frac: float = 0.2
    half_life_days: float = 30.0
    median_fallback: bool = True
    feature_groups: str = ""
    btc_symbol: str = "BTCUSDT"
    use_news: bool = False
    news_common_slice: bool = False
    use_dvol_exog: bool = False
    ewma_lambda: float = 0.94
    baseline_window: int = 500
    n_folds: int = 5
    holdout_frac: float = 0.15
    embargo: int = 3
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 42
    n_jobs: int = 4
    shape_window: int = 5000
    dist_attempt: str = "none"  # none | ngboost | lightgbmlss


def default_config(interval: str, overrides: dict | None = None) -> V3Config:
    if interval not in INTERVAL_MS:
        raise ValueError(f"未知周期 {interval}")
    cfg = V3Config(interval=interval, limit_bars=q.DEFAULT_LIMIT.get(interval))
    if overrides:
        cfg = replace(cfg, **overrides)
    validate(cfg)
    return cfg


def validate(cfg: V3Config) -> None:
    if cfg.features not in FEATURE_MODES:
        raise ValueError(f"features 只能是 {FEATURE_MODES}")
    if cfg.sigma_model not in SIGMA_KINDS:
        raise ValueError(f"sigma_model 只能是 {SIGMA_KINDS}")
    if cfg.shape not in SHAPES:
        raise ValueError(f"shape 只能是 {SHAPES}")
    if cfg.shape in ("empirical", "student_t") and not cfg.normalize:
        raise ValueError("empirical/student_t 形状需要 normalize=true")
    if cfg.conformal in ("norm", "weighted", "norm_aci") and not cfg.normalize:
        raise ValueError("归一化共形需要 normalize=true")
    if cfg.conformal not in CONFORMALS:
        raise ValueError(f"conformal 只能是 {CONFORMALS}")
    if cfg.dist_attempt not in ("none", "ngboost", "lightgbmlss"):
        raise ValueError("dist_attempt bad")
    if not 0.0 < cfg.calib_frac < 0.5:
        raise ValueError("calib_frac 必须在 (0, 0.5)")
    if cfg.n_folds < 1 or not 0.0 < cfg.holdout_frac < 0.5:
        raise ValueError("折数或留存比例不合法")


def parse_overrides(pairs: list[str], base: V3Config) -> dict:
    types = {f.name: f.type for f in fields(V3Config)}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set 需要 key=value，收到 {pair}")
        key, text = pair.split("=", 1)
        if key not in types or key == "interval":
            raise ValueError(f"未知配置项 {key}")
        field = next(f for f in fields(V3Config) if f.name == key)
        if text.lower() == "none" and field.default is None:
            out[key] = None
        elif text.lower() == "none":
            out[key] = "none"
        elif types[key] in ("bool", bool):
            out[key] = text.lower() in ("1", "true", "yes")
        elif "int" in str(types[key]) and "float" not in str(types[key]):
            out[key] = int(text)
        elif "float" in str(types[key]):
            out[key] = float(text)
        else:
            out[key] = text
    return out


def _qcfg(cfg: V3Config) -> q.QConfig:
    return q.default_config(cfg.interval, {
        "limit_bars": cfg.limit_bars,
        "snapshot_ms": cfg.snapshot_ms,
        "feature_groups": cfg.feature_groups,
        "btc_symbol": cfg.btc_symbol,
        "ewma_lambda": cfg.ewma_lambda,
        "zscore": True,
        "use_funding": True,
        "use_metrics": True,
        "n_folds": cfg.n_folds,
        "holdout_frac": cfg.holdout_frac,
        "embargo": cfg.embargo,
        "baseline_window": cfg.baseline_window,
        "n_jobs": cfg.n_jobs,
    })


# ---------------------------------------------------------------- features

def _selected_rv_names(rvf: rv.RVFeatures, mode: str) -> list[str]:
    out = []
    for name in rvf.names:
        keep = False
        if mode in ("rv", "rv_season", "rv_micro", "all") and name.startswith(RV_PREFIXES):
            keep = True
        if mode in ("rv_season", "all") and name.startswith(SEASON_PREFIXES):
            keep = True
        if mode in ("rv_micro", "all") and name.startswith(MICRO_PREFIXES):
            keep = True
        if keep:
            out.append(name)
    return out


def assemble(base_names: list[str], base_X: np.ndarray, rvf: rv.RVFeatures | None,
             cfg: V3Config, news_cols: dict[str, np.ndarray] | None) -> tuple[list[str], np.ndarray]:
    cols: list[tuple[str, np.ndarray]] = [(n, base_X[:, i]) for i, n in enumerate(base_names)]
    if rvf is not None and cfg.features != "base":
        wanted = _selected_rv_names(rvf, cfg.features)
        idx = {n: i for i, n in enumerate(rvf.names)}
        cols.extend((n, rvf.X[:, idx[n]]) for n in wanted)
    if news_cols:
        cols.extend((n, np.asarray(v, dtype=np.float64)) for n, v in news_cols.items())
    names = [n for n, _ in cols]
    X = np.column_stack([v for _, v in cols]).astype(np.float64)
    return names, X


def core_indices(names: list[str], cfg: V3Config) -> list[int]:
    required = set(features.CORE_FEATURES)
    h = max(1, int(round(INTERVAL_MS[cfg.interval] / 60_000)))
    required.add(f"rv_{h}m")
    return [i for i, n in enumerate(names) if n in required]


@dataclass
class SamplesV3:
    idx: np.ndarray
    X: np.ndarray
    y: np.ndarray
    sig_base: np.ndarray
    sig_garch: np.ndarray
    sig_base_full: np.ndarray
    sig_garch_full: np.ndarray
    sig_full: np.ndarray
    names: list[str]
    ref_close: np.ndarray
    decision_ms: np.ndarray
    bars_n: int
    r1: np.ndarray
    closes: np.ndarray
    rvf: rv.RVFeatures
    news: newsfeat.NewsData | None
    base_names: list[str]

    @property
    def m(self) -> int:
        return int(self.idx.shape[0])


def load_samples(cfg: V3Config, db_path: Path, data_root: Path) -> SamplesV3:
    qc = _qcfg(cfg)
    groups = q.feature_groups(qc)
    conn = data.open_readonly(db_path)
    try:
        bars = data.load_bars(conn, cfg.market, cfg.symbol, cfg.interval, cfg.limit_bars,
                              close_by_ms=cfg.snapshot_ms)
        funding = data.load_funding(conn, cfg.symbol)
        metrics = data.load_metrics(conn, cfg.symbol)
        mark = data.load_bars(conn, "um_mark", cfg.symbol, cfg.interval, cfg.limit_bars,
                              close_by_ms=cfg.snapshot_ms)
        index = data.load_bars(conn, "um_index", cfg.symbol, cfg.interval, cfg.limit_bars,
                               close_by_ms=cfg.snapshot_ms)
        premium = data.load_bars(conn, "um_premium", cfg.symbol, cfg.interval, cfg.limit_bars,
                                 close_by_ms=cfg.snapshot_ms)
        htf_interval = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d"}.get(cfg.interval)
        try:
            higher = data.load_bars(conn, "um", cfg.symbol, htf_interval, None,
                                    close_by_ms=cfg.snapshot_ms) if htf_interval else None
        except ValueError:
            higher = None
        btc_bars = None
        if "btc" in groups:
            try:
                btc_bars = data.load_bars(conn, cfg.market, cfg.btc_symbol, cfg.interval,
                                          cfg.limit_bars, close_by_ms=cfg.snapshot_ms)
            except ValueError as exc:
                log.warning("BTC 数据不可用，%s 组按缺失处理: %s", cfg.interval, exc)
        one_min = None
        try:
            one_min = data.load_bars(conn, cfg.market, cfg.symbol, "1m", None,
                                     close_by_ms=cfg.snapshot_ms)
        except ValueError:
            log.warning("%s 没有 1 分钟数据，RV/HAR 基线不可用", cfg.symbol)
    finally:
        conn.close()

    base_names, base_X, sig_base, r1 = q.build_features(
        bars, funding, metrics, qc, mark, index, premium, higher,
        altdata.load_alt(Path(db_path).parent, groups, btc=btc_bars), btc_bars)
    rvf = None
    if one_min is not None:
        rvf = rv.build_cached(Path(data_root), one_min, bars.close_time, bars.interval_ms,
                              cfg.interval, int(cfg.snapshot_ms or 0), cfg.limit_bars)
        if rvf.n_1m < 2:
            rvf = None
    news = None
    news_cols = None
    if cfg.use_news or cfg.news_common_slice:
        news = newsfeat.load_news(Path(db_path), end_ms=cfg.snapshot_ms)
        news_cols = newsfeat.build_columns(bars.close_time, bars.interval_ms, news)
    names, X = assemble(base_names, base_X, rvf, cfg,
                        news_cols if cfg.use_news else None)
    if rvf is None:
        # Keep the schema stable for datasets without 1m bars: RV columns are
        # NaN and HAR baselines are skipped by the caller.
        rvf = rv.RVFeatures(names=[], X=np.full((bars.n, 0), np.nan),
                            rv_future=np.full(bars.n, np.nan),
                            rv_past=np.full(bars.n, np.nan),
                            decision_ms=bars.close_time, first_1m_ms=0, last_1m_ms=0, n_1m=0)
    has_rv = rvf.n_1m > 0

    n = bars.n
    y = np.full(n, np.nan)
    if n > 1:
        y[:-1] = r1[1:]
    core = core_indices(names, cfg)
    core_ok = np.all(np.isfinite(X[:, core]), axis=1)
    gaps = data.gap_after(bars).astype(np.int64)
    gap_cum = np.concatenate([[0], np.cumsum(gaps)])
    lo = np.maximum(np.arange(n) - q.WINDOW_LOOKBACK, 0)
    hi = np.arange(n)
    gap_in_window = (gap_cum[hi + 1] - gap_cum[lo]) > 0
    ok = (core_ok & np.isfinite(y) & np.isfinite(sig_base) & (sig_base > 0)
          & ~gap_in_window)
    if has_rv:
        ok &= np.isfinite(rvf.rv_future) & np.isfinite(rvf.rv_past)
    ok[n - 1] = False
    if cfg.news_common_slice and news is not None and news.n:
        covered = (bars.close_time >= int(news.avail[0]) + 7 * 86400_000)
        ok &= covered & (news_cols["news_coverage"] > 0)
    idx = np.nonzero(ok)[0].astype(np.int64)
    if idx.shape[0] < 1000:
        raise ValueError(f"{cfg.interval} 的有效样本只有 {idx.shape[0]} 个，不够训练")
    sig_garch_full = q._garch_vol(r1)
    return SamplesV3(
        idx=idx, X=X[idx], y=y[idx], sig_base=sig_base[idx],
        sig_garch=sig_garch_full[idx],
        sig_base_full=sig_base, sig_garch_full=sig_garch_full, sig_full=sig_base,
        names=names,
        ref_close=bars.c[idx], decision_ms=bars.close_time[idx], bars_n=n,
        r1=r1, closes=bars.c, rvf=rvf, news=news, base_names=base_names,
    )


# ---------------------------------------------------------------- sigma models

@dataclass
class SigmaFit:
    kind: str
    feature_names: list[str] | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    clip_lo: float = 1e-6
    clip_hi: float = 0.5
    sigma_all: np.ndarray | None = None
    qlike: float | None = None


def _h(cfg: V3Config) -> int:
    return max(1, int(round(INTERVAL_MS[cfg.interval] / 60_000)))


def _sigma_design(S: SamplesV3, cfg: V3Config, kind: str) -> tuple[list[str], np.ndarray]:
    h = _h(cfg)
    if kind in ("har", "har_rs", "har_cj"):
        names, Z = rv.har_matrix(S.rvf, kind, h)
    elif kind == "rgarch":
        names, Z = rv.har_matrix(S.rvf, "har", h)
        idx = {n: i for i, n in enumerate(S.rvf.names)}
        past = S.rvf.X[:, idx[f"rv_{h}m"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            logpast = np.log(past + rv.EPS)
        # Feedback uses the interval return that ended at the decision bar; it
        # is known at that moment and is not the future label.
        stretch = np.column_stack([logpast, S.r1, np.abs(S.r1)])
        names = names + ["log_rv_past", "ret_past", "abs_ret_past"]
        Z = np.column_stack([Z, stretch])
    else:
        raise ValueError(f"未知 sigma kind {kind}")
    return names, Z


def fit_sigma(S: SamplesV3, train: np.ndarray, cfg: V3Config, kind: str | None = None) -> SigmaFit:
    kind = kind or cfg.sigma_model
    if kind in ("har", "har_rs", "har_cj", "rgarch") and S.rvf.n_1m == 0:
        return SigmaFit(kind="ewma", sigma_all=_clip_sigma(S.sig_base_full, S, train))
    if kind == "ewma":
        return SigmaFit(kind=kind, sigma_all=_clip_sigma(S.sig_base_full, S, train))
    if kind == "garch":
        return SigmaFit(kind=kind, sigma_all=_clip_sigma(S.sig_garch_full, S, train))
    names, Z = _sigma_design(S, cfg, kind)
    target = np.log(S.rvf.rv_future + rv.EPS)
    rows = S.idx[train]
    ok = np.all(np.isfinite(Z[rows]), axis=1) & np.isfinite(target[rows])
    if ok.sum() < 200:
        # Fall back to a causal HAR so a bad exogenous block never stops a run.
        names, Z = rv.har_matrix(S.rvf, "har", _h(cfg))
        ok = np.all(np.isfinite(Z[rows]), axis=1) & np.isfinite(target[rows])
    if ok.sum() < 200:
        sigma = S.sig_base
        return SigmaFit(kind=kind, sigma_all=_clip_sigma(sigma, S, train),
                        qlike=_qlike_at(S, sigma, train))
    model = Ridge(alpha=1.0)
    model.fit(Z[rows][ok], target[rows][ok])
    finite = np.all(np.isfinite(Z), axis=1)
    pred = np.full(Z.shape[0], np.nan, dtype=np.float64)
    if finite.any():
        pred[finite] = model.predict(Z[finite])
    sigma = np.sqrt(np.exp(np.clip(pred, -30.0, 5.0)))
    sigma = _clip_sigma(sigma, S, train)
    return SigmaFit(kind=kind, feature_names=names, coef=np.asarray(model.coef_, dtype=np.float64),
                    intercept=float(model.intercept_), clip_lo=float(np.nanmin(sigma)),
                    clip_hi=float(np.nanmax(sigma)), sigma_all=sigma,
                    qlike=_qlike_at(S, sigma, train))


def _clip_sigma(sigma: np.ndarray, S: SamplesV3, train: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        ref = np.sqrt(np.maximum(S.rvf.rv_future[S.idx[train]], 0.0))
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if ref.size == 0:
        lo, hi = 1e-6, 0.5
    else:
        lo = max(1e-8, float(np.quantile(ref, 0.001)) * 0.1)
        hi = min(2.0, float(np.quantile(ref, 0.999)) * 10.0)
    out = np.asarray(sigma, dtype=np.float64)
    return np.clip(out, lo, hi)


def _qlike_at(S: SamplesV3, sigma: np.ndarray, train: np.ndarray) -> float:
    rvf = S.rvf.rv_future[S.idx[train]]
    var = np.square(sigma[S.idx[train]])
    return rv.qlike(rvf, var)


def predict_sigma(S: SamplesV3, fit: SigmaFit, cfg: V3Config) -> np.ndarray:
    if fit.sigma_all is not None:
        return fit.sigma_all
    names, Z = _sigma_design(S, cfg, fit.kind)
    if fit.feature_names and names != fit.feature_names:
        # Keep column order stable even if the design builder adds a block.
        idx = {n: i for i, n in enumerate(names)}
        Z = np.column_stack([Z[:, idx[n]] for n in fit.feature_names])
    pred = float(fit.intercept) + Z @ np.asarray(fit.coef, dtype=np.float64)
    sigma = np.sqrt(np.exp(np.clip(pred, -30.0, 5.0)))
    return np.clip(sigma, fit.clip_lo, fit.clip_hi)


def sigma_from_meta(meta: dict, S: SamplesV3, row: int) -> float:
    kind = meta["sigma"]["kind"]
    if kind == "ewma":
        return float(S.sig_base[row]) if hasattr(S, "sig_base") else float("nan")
    if kind == "garch":
        return float(S.sig_garch[row]) if hasattr(S, "sig_garch") else float("nan")
    names, Z = _sigma_design(S, default_config(meta["interval"], meta["config"]), kind)
    idx = {n: i for i, n in enumerate(names)}
    x = np.asarray([Z[row, idx[n]] for n in meta["sigma"]["feature_names"]], dtype=np.float64)
    pred = float(meta["sigma"]["intercept"]) + float(x @ np.asarray(meta["sigma"]["coef"]))
    return float(np.clip(math.sqrt(math.exp(min(5.0, max(-30.0, pred)))),
                         meta["sigma"]["clip_lo"], meta["sigma"]["clip_hi"]))


# ---------------------------------------------------------------- shape models

@dataclass
class ShapeFit:
    kind: str
    boosters: list[lgb.Booster] | None = None
    zq: np.ndarray | None = None
    df: float | None = None
    scale: float | None = None
    blend_w: float | None = None


def _z(sigma_all: np.ndarray, S: SamplesV3, pos: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return S.y[pos] / sigma_all[S.idx[pos]]


def fit_shape(S: SamplesV3, sigma_all: np.ndarray, train: np.ndarray, cfg: V3Config) -> ShapeFit:
    z = _z(sigma_all, S, train)
    ok = np.isfinite(z) & (np.abs(z) < 50)
    z = z[ok]
    if z.shape[0] < 200:
        raise ValueError("形状样本太少")
    if cfg.shape == "empirical":
        window = z[-int(cfg.shape_window):]
        return ShapeFit(kind="empirical", zq=np.quantile(window, ALPHAS))
    if cfg.shape == "student_t":
        sample = z[-min(z.shape[0], 20_000):]
        try:
            df, loc, scale = stats.t.fit(sample, floc=0.0)
            if np.isfinite(df) and df > 1.0 and np.isfinite(scale) and scale > 0:
                return ShapeFit(kind="student_t", df=float(df), scale=float(scale))
        except Exception:
            pass
        window = z[-int(cfg.shape_window):]
        return ShapeFit(kind="empirical", zq=np.quantile(window, ALPHAS))
    if cfg.shape == "blend":
        # 50/50 ensemble of the conditional-shape GBM and a Student-t shape.
        # The weight is fixed a priori; it is not tuned on holdout.
        target = z if cfg.normalize else S.y[train][ok]
        boosters = q._fit_quantiles(S.X[train][ok], target, cfg)  # type: ignore[arg-type]
        sample = z[-min(z.shape[0], 20_000):]
        try:
            df, _, scale = stats.t.fit(sample, floc=0.0)
        except Exception:
            df, scale = 5.0, float(np.std(z))
        if not (np.isfinite(df) and df > 1.0 and np.isfinite(scale) and scale > 0):
            df, scale = 5.0, float(np.std(z) or 1.0)
        return ShapeFit(kind="blend", boosters=boosters, df=float(df),
                        scale=float(scale), blend_w=0.5)
    # quantile GBM on standardized (or raw, for an exact v2-style baseline)
    target = z if cfg.normalize else S.y[train][ok]
    boosters = q._fit_quantiles(S.X[train][ok], target, cfg)  # type: ignore[arg-type]
    return ShapeFit(kind="quantile", boosters=boosters)


def predict_shape(S: SamplesV3, shape: ShapeFit, sigma: np.ndarray, pos: np.ndarray,
                  cfg: V3Config | None = None) -> np.ndarray:
    if shape.kind == "quantile":
        Qz = q.predict_raw(shape.boosters, S.X[pos])
        if cfg is not None and not cfg.normalize:
            return q.sort_rows(Qz)
    elif shape.kind == "student_t":
        Qz = stats.t.ppf(np.asarray(ALPHAS), shape.df, loc=0.0, scale=shape.scale)
        Qz = np.tile(np.asarray(Qz, dtype=np.float64), (pos.shape[0], 1))
    elif shape.kind == "blend":
        Qg = q.predict_raw(shape.boosters, S.X[pos])
        if cfg is not None and not cfg.normalize:
            Qt = np.tile(np.asarray(stats.t.ppf(np.asarray(ALPHAS), shape.df, loc=0.0,
                                                scale=shape.scale), dtype=np.float64),
                         (pos.shape[0], 1))
            return q.sort_rows(shape.blend_w * Qg + (1.0 - shape.blend_w) * Qt)
        Qt = np.tile(np.asarray(stats.t.ppf(np.asarray(ALPHAS), shape.df, loc=0.0,
                                            scale=shape.scale), dtype=np.float64),
                     (pos.shape[0], 1))
        Qz = shape.blend_w * Qg + (1.0 - shape.blend_w) * Qt
    else:
        Qz = np.tile(np.asarray(shape.zq, dtype=np.float64), (pos.shape[0], 1))
    return q.sort_rows(Qz * sigma[S.idx[pos]][:, None])


# ---------------------------------------------------------------- conformal

@dataclass
class ConformalFit:
    mode: str
    lo50: float = 0.0
    hi50: float = 0.0
    lo80: float = 0.0
    hi80: float = 0.0
    half_life_days: float = 30.0

    @property
    def c50(self) -> float:
        return 0.5 * (self.lo50 + self.hi50)

    @property
    def c80(self) -> float:
        return 0.5 * (self.lo80 + self.hi80)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(values, level, method="higher"))
    target = level * cw[-1]
    j = int(np.searchsorted(cw, target, side="left"))
    return float(v[min(j, v.shape[0] - 1)])


def fit_conformal(S: SamplesV3, shape: ShapeFit, sigma_all: np.ndarray,
                  part: np.ndarray, cal: np.ndarray, cfg: V3Config) -> ConformalFit:
    if cfg.conformal == "none" or cal.shape[0] < 50:
        return ConformalFit(mode=cfg.conformal, half_life_days=cfg.half_life_days)
    Q = predict_shape(S, shape, sigma_all, cal, cfg)
    y = S.y[cal]
    sig = sigma_all[S.idx[cal]]
    norm = cfg.conformal in ("norm", "weighted", "norm_aci")
    scale = sig if norm else np.ones_like(sig)
    # Two-sided split conformal: each tail is calibrated separately so a band
    # that is already too wide can be narrowed as well as widened.  This is not
    # a single global width scaling.
    lo50 = (Q[:, 1] - y) / scale
    hi50 = (y - Q[:, 3]) / scale
    lo80 = (Q[:, 0] - y) / scale
    hi80 = (y - Q[:, 4]) / scale
    if cfg.conformal == "weighted":
        age_days = (S.decision_ms[cal].max() - S.decision_ms[cal]) / 86_400_000.0
        w = np.exp(-math.log(2.0) * age_days / max(0.5, cfg.half_life_days))
        a, b = _weighted_quantile(lo50, w, 0.75), _weighted_quantile(hi50, w, 0.75)
        c, d = _weighted_quantile(lo80, w, 0.90), _weighted_quantile(hi80, w, 0.90)
    else:
        a = float(np.quantile(lo50, 0.75, method="higher"))
        b = float(np.quantile(hi50, 0.75, method="higher"))
        c = float(np.quantile(lo80, 0.90, method="higher"))
        d = float(np.quantile(hi80, 0.90, method="higher"))
    return ConformalFit(mode=cfg.conformal, lo50=float(a), hi50=float(b),
                        lo80=float(c), hi80=float(d),
                        half_life_days=cfg.half_life_days)


def apply_conformal(Q: np.ndarray, sigma: np.ndarray, conf: ConformalFit) -> np.ndarray:
    if conf.mode == "none":
        return q.sort_rows(Q)
    out = Q.copy()
    if conf.mode == "cqr":
        out[:, 1] -= conf.lo50
        out[:, 3] += conf.hi50
        out[:, 0] -= conf.lo80
        out[:, 4] += conf.hi80
    else:
        out[:, 1] -= conf.lo50 * sigma
        out[:, 3] += conf.hi50 * sigma
        out[:, 0] -= conf.lo80 * sigma
        out[:, 4] += conf.hi80 * sigma
    return q.sort_rows(out)


def apply_aci(Q: np.ndarray, y: np.ndarray, gamma: float = 0.02) -> np.ndarray:
    """Online ACI on a test segment; state starts fresh at the segment start."""
    out = Q.copy()
    s50 = 1.0
    s80 = 1.0
    for i in range(Q.shape[0]):
        center = float(Q[i, 2])
        h50 = float(Q[i, 3] - Q[i, 1]) / 2.0
        h80 = float(Q[i, 4] - Q[i, 0]) / 2.0
        lo50, hi50 = center - h50 * s50, center + h50 * s50
        lo80, hi80 = center - h80 * s80, center + h80 * s80
        row = np.sort(np.array([lo80, lo50, center, hi50, hi80]))
        out[i] = row
        s50 = min(3.0, max(0.5, s50 + gamma * ((0.0 if lo50 <= y[i] <= hi50 else 1.0) - 0.5)))
        s80 = min(3.0, max(0.5, s80 + gamma * ((0.0 if lo80 <= y[i] <= hi80 else 1.0) - 0.2)))
    return out


# ---------------------------------------------------------------- fit / predict

@dataclass
class V3Fit:
    sigma: SigmaFit
    shape: ShapeFit
    conformal: ConformalFit
    n_train: int
    n_calib: int


def fit_block(S: SamplesV3, train: np.ndarray, cfg: V3Config) -> V3Fit:
    if train.shape[0] < 500:
        raise ValueError(f"训练样本只有 {train.shape[0]} 个，太少")
    if cfg.conformal == "none":
        part = train
        cal = np.zeros(0, dtype=np.int64)
    else:
        cut = int(round(train.shape[0] * (1.0 - cfg.calib_frac)))
        cal = train[cut:]
        part = train[:cut]
        if part.shape[0] >= 500 and cal.shape[0] >= 50:
            part = part[S.idx[part] + 1 < S.idx[cal[0]] - cfg.embargo]
    sigma_fit = fit_sigma(S, part if part.shape[0] >= 500 else train, cfg)
    sigma_all = predict_sigma(S, sigma_fit, cfg)
    shape = fit_shape(S, sigma_all, part if part.shape[0] >= 500 else train, cfg)
    conf = fit_conformal(S, shape, sigma_all, part, cal, cfg)
    return V3Fit(sigma=sigma_fit, shape=shape, conformal=conf,
                 n_train=int(part.shape[0]), n_calib=int(cal.shape[0]))


def predict_block(S: SamplesV3, fit: V3Fit, pos: np.ndarray, cfg: V3Config) -> np.ndarray:
    sigma = predict_sigma(S, fit.sigma, cfg)
    Q = predict_shape(S, fit.shape, sigma, pos, cfg)
    Q = apply_conformal(Q, sigma[S.idx[pos]], fit.conformal)
    if cfg.conformal == "norm_aci":
        Q = apply_aci(Q, S.y[pos])
    return q.sort_rows(Q)


def center_quantiles(Q: np.ndarray) -> np.ndarray:
    """Median fallback: shift the whole distribution so q50 = current price."""
    return q.sort_rows(Q - Q[:, 2:3])


def score_v3(y: np.ndarray, Q: np.ndarray, sig: np.ndarray) -> dict:
    out = q.score(y, Q, sig)
    out["mae_zero_bps"] = round(float(np.mean(np.abs(y))) * 1e4, 4)
    out["crps_approx_bps"] = round(2.0 * out["pinball_combined_bps"], 4)
    below = (Q <= y[:, None]).mean(axis=1)
    hist = np.histogram(below, bins=np.linspace(0, 1, 6))[0]
    out["pit_mean"] = round(float(below.mean()), 4)
    out["pit_hist"] = [round(float(x / max(below.shape[0], 1)), 4) for x in hist]
    out["kupiec50_p"] = _kupiec(out["cov50"], y.shape[0], 0.50)
    out["kupiec80_p"] = _kupiec(out["cov80"], y.shape[0], 0.80)
    return out


def _kupiec(cov: float, n: int, nominal: float) -> float | None:
    if n <= 0:
        return None
    n1 = max(0.0, min(float(n), cov * n))
    n0 = n - n1
    phat = n1 / n
    if phat in (0.0, 1.0):
        return 0.0 if abs(phat - nominal) > 1e-12 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = -2.0 * ((n0 * math.log(1 - nominal) + n1 * math.log(nominal))
                     - (n0 * math.log(1 - phat) + n1 * math.log(phat)))
    return round(float(stats.chi2.sf(lr, 1)), 6)


# ---------------------------------------------------------------- baselines

def _normal_z() -> np.ndarray:
    from statistics import NormalDist
    nd = NormalDist()
    return np.array([nd.inv_cdf(a) for a in ALPHAS])


def _shape_from_train(S: SamplesV3, sigma_all: np.ndarray, train: np.ndarray,
                      kind: str, cfg: V3Config) -> tuple[np.ndarray | None, float | None, float | None]:
    z = _z(sigma_all, S, train)
    z = z[np.isfinite(z) & (np.abs(z) < 50)]
    if kind == "gauss":
        return None, None, None
    if z.shape[0] < 200:
        return (np.quantile(z, ALPHAS) if z.shape[0] else None), None, None
    if kind == "emp":
        return np.quantile(z[-int(cfg.shape_window):], ALPHAS), None, None
    if kind == "t":
        try:
            df, _, scale = stats.t.fit(z[-min(z.shape[0], 20_000):], floc=0.0)
            if np.isfinite(df) and df > 1 and np.isfinite(scale) and scale > 0:
                return None, float(df), float(scale)
        except Exception:
            pass
    return np.quantile(z[-int(cfg.shape_window):], ALPHAS), None, None


def _baseline_from_sigma(S: SamplesV3, sigma_all: np.ndarray, train: np.ndarray,
                         pos: np.ndarray, shape_kind: str, cfg: V3Config) -> np.ndarray:
    zq, df, scale = _shape_from_train(S, sigma_all, train, shape_kind, cfg)
    sig = sigma_all[S.idx[pos]][:, None]
    if zq is not None:
        return q.sort_rows(np.tile(zq, (pos.shape[0], 1)) * sig)
    if df is not None:
        return q.sort_rows(np.tile(stats.t.ppf(np.asarray(ALPHAS), df, loc=0, scale=scale),
                                   (pos.shape[0], 1)) * sig)
    return q.sort_rows(np.tile(_normal_z(), (pos.shape[0], 1)) * sig)


def _fold_baselines(S: SamplesV3, blk: q.Block, cfg: V3Config,
                    base_global: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {name: B[blk.test] for name, B in base_global.items()}
    garch = S.sig_garch_full
    out["garch_gauss"] = _baseline_from_sigma(S, garch, blk.train, blk.test, "gauss", cfg)
    out["garch_emp"] = _baseline_from_sigma(S, garch, blk.train, blk.test, "emp", cfg)
    out["garch_t"] = _baseline_from_sigma(S, garch, blk.train, blk.test, "t", cfg)
    for kind in (("har", "har_rs", "har_cj") if S.rvf.n_1m > 0 else ()):
        fit = fit_sigma(S, blk.train, cfg, kind=kind)
        sigma = predict_sigma(S, fit, cfg)
        tag = {"har": "har", "har_rs": "har_rs", "har_cj": "har_cj"}[kind]
        out[f"{tag}_gauss"] = _baseline_from_sigma(S, sigma, blk.train, blk.test, "gauss", cfg)
        out[f"{tag}_emp"] = _baseline_from_sigma(S, sigma, blk.train, blk.test, "emp", cfg)
        out[f"{tag}_t"] = _baseline_from_sigma(S, sigma, blk.train, blk.test, "t", cfg)
    return out


# ---------------------------------------------------------------- runs

def gates_v3(model: dict, folds: list[dict], base_scores: dict[str, dict],
             best_name: str) -> dict:
    best = base_scores[best_name]
    improve = 1.0 - model["pinball_combined_bps"] / best["pinball_combined_bps"]
    ok50 = sum(1 for f in folds if abs(f["cov50"] - 0.5) <= 0.02)
    ok80 = sum(1 for f in folds if abs(f["cov80"] - 0.8) <= 0.02)
    calibrated = [b for b in base_scores.values() if abs(b["cov50"] - 0.5) <= 0.03]
    ref = calibrated or list(base_scores.values())
    width_ref = min(b["w50_bps"] for b in ref)
    cov_pool_ok = abs(model["cov50"] - 0.5) <= 0.03 and abs(model["cov80"] - 0.8) <= 0.03
    fold_ok = ok50 >= 4 and ok80 >= 4
    main = bool(fold_ok and cov_pool_ok and improve >= 0.03 and model["w50_bps"] <= width_ref)
    secondary = bool(fold_ok and cov_pool_ok and improve >= 0.015
                     and model["w50_bps"] <= 1.01 * width_ref)
    return {
        "folds_cov50_ok": f"{ok50}/{len(folds)}",
        "folds_cov80_ok": f"{ok80}/{len(folds)}",
        "fold_cov_rule": fold_ok,
        "pooled_cov_rule": cov_pool_ok,
        "pinball_improve_vs_best_base": round(float(improve), 4),
        "pinball_rule_main": improve >= 0.03,
        "pinball_rule_secondary": improve >= 0.015,
        "width50_ref_bps": round(float(width_ref), 4),
        "width_rule_main": model["w50_bps"] <= width_ref,
        "width_rule_secondary": model["w50_bps"] <= 1.01 * width_ref,
        "best_baseline": best_name,
        "main_pass": main,
        "secondary_pass": secondary,
        "pass": main,
        "status": "达标" if main else "仅参考",
    }


def run_cv(S: SamplesV3, cfg: V3Config, base_global: dict[str, np.ndarray], log_fn) -> dict:
    qc = _qcfg(cfg)
    folds, _ = q.cv_blocks(S, qc)
    fold_out = []
    pooled_pos, pooled_Q = [], []
    base_pool: dict[str, list[np.ndarray]] = {}
    for blk in folds:
        fit = fit_block(S, blk.train, cfg)
        Q = predict_block(S, fit, blk.test, cfg)
        sc = score_v3(S.y[blk.test], Q, S.sig_base[blk.test])
        sc.update({"fold": blk.number, "n_train": fit.n_train, "n_calib": fit.n_calib,
                   "c50": round(fit.conformal.c50, 8), "c80": round(fit.conformal.c80, 8),
                   "sigma_kind": fit.sigma.kind, "shape_kind": fit.shape.kind,
                   "sigma_qlike": fit.sigma.qlike})
        fold_out.append(sc)
        pooled_pos.append(blk.test)
        pooled_Q.append(Q)
        for name, B in _fold_baselines(S, blk, cfg, base_global).items():
            base_pool.setdefault(name, []).append(B)
        log_fn(f"  fold {blk.number}: train={fit.n_train} cal={fit.n_calib} "
               f"cov50={sc['cov50']} cov80={sc['cov80']} pin={sc['pinball_combined_bps']} "
               f"w50={sc['w50_bps']} qlike={fit.sigma.qlike}")
    pos = np.concatenate(pooled_pos)
    Q = np.vstack(pooled_Q)
    y = S.y[pos]
    model = score_v3(y, Q, S.sig_base[pos])
    model_before = model
    median_fallback = bool(cfg.median_fallback
                           and model["mae_bps"] >= model["mae_zero_bps"] * 0.999)
    if median_fallback:
        Q = center_quantiles(Q)
        model = score_v3(y, Q, S.sig_base[pos])
    base_scores = {name: score_v3(y, np.vstack(parts), S.sig_base[pos])
                   for name, parts in base_pool.items()}
    best_name = min(base_scores, key=lambda k: base_scores[k]["pinball_combined_bps"])
    gates = gates_v3(model, fold_out, base_scores, best_name)
    return {
        "model": model,
        "model_before_median_fallback": model_before,
        "folds": fold_out,
        "baselines": base_scores,
        "best_baseline": best_name,
        "gates": gates,
        "median_fallback": median_fallback,
        "config": asdict(cfg),
        "samples": S.m,
        "span_utc8": [_fmt(int(S.decision_ms[0])), _fmt(int(S.decision_ms[-1]))],
        "news_coverage": _news_coverage(S),
    }


def run_holdout(S: SamplesV3, cfg: V3Config, base_global: dict[str, np.ndarray],
                median_fallback: bool) -> dict:
    qc = _qcfg(cfg)
    _, hold = q.cv_blocks(S, qc)
    fit = fit_block(S, hold.train, cfg)
    Q = predict_block(S, fit, hold.test, cfg)
    if median_fallback:
        Q = center_quantiles(Q)
    y = S.y[hold.test]
    model = score_v3(y, Q, S.sig_base[hold.test])
    base_scores = {name: score_v3(y, B, S.sig_base[hold.test])
                   for name, B in _fold_baselines(S, hold, cfg, base_global).items()}
    best_name = min(base_scores, key=lambda k: base_scores[k]["pinball_combined_bps"])
    gates = gates_v3(model, [{"cov50": model["cov50"], "cov80": model["cov80"]}],
                     base_scores, best_name)
    return {"model": model, "baselines": base_scores, "best_baseline": best_name,
            "gates_single_block": gates, "n_train": fit.n_train, "n_calib": fit.n_calib,
            "c50": fit.conformal.c50, "c80": fit.conformal.c80,
            "sigma_kind": fit.sigma.kind, "shape_kind": fit.shape.kind,
            "sigma_qlike": fit.sigma.qlike, "config": asdict(cfg)}


def _news_coverage(S: SamplesV3) -> dict | None:
    if S.news is None or S.news.n == 0:
        return None
    return {
        "n": S.news.n,
        "first_utc8": _fmt(int(S.news.avail[0])),
        "last_utc8": _fmt(int(S.news.avail[-1])),
        "sample_first_utc8": _fmt(int(S.decision_ms[0])),
        "sample_last_utc8": _fmt(int(S.decision_ms[-1])),
    }


# ---------------------------------------------------------------- artifacts

ARTIFACT_ROOT_NAME = "models_quantile"
META_FILE = "meta.json"
FEATURES_FILE = "features.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=_UTC8).strftime("%Y-%m-%d %H:%M")


def _backup_v2(data_root: Path, interval: str) -> None:
    src = Path(data_root) / ARTIFACT_ROOT_NAME / interval
    meta_path = src / META_FILE
    if not meta_path.exists():
        return
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if int(old.get("schema_version", 2)) >= 3:
        return
    dst = Path(data_root) / "models_quantile_v2_backup" / interval
    if dst.exists():
        return
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def save_artifacts(data_root: Path, S: SamplesV3, cfg: V3Config, fit: V3Fit,
                   meta_extra: dict, median_fallback: bool) -> Path:
    _backup_v2(data_root, cfg.interval)
    directory = Path(data_root) / ARTIFACT_ROOT_NAME / cfg.interval
    directory.mkdir(parents=True, exist_ok=True)
    shas = {}
    if fit.shape.kind in ("quantile", "blend"):
        for a, booster in zip(ALPHAS, fit.shape.boosters):
            target = directory / q.model_file(a)
            tmp = directory / (q.model_file(a) + ".tmp")
            booster.save_model(str(tmp))
            os.replace(tmp, target)
            shas[q.model_file(a)] = _sha(target)
    else:
        for a in ALPHAS:
            (directory / q.model_file(a)).unlink(missing_ok=True)
    feat_text = json.dumps(S.names, ensure_ascii=False, indent=2)
    (directory / (FEATURES_FILE + ".tmp")).write_text(feat_text, encoding="utf-8")
    os.replace(directory / (FEATURES_FILE + ".tmp"), directory / FEATURES_FILE)
    shas[FEATURES_FILE] = _sha(directory / FEATURES_FILE)

    sigma_meta = {"kind": fit.sigma.kind, "feature_names": fit.sigma.feature_names,
                  "coef": (fit.sigma.coef.tolist() if fit.sigma.coef is not None else None),
                  "intercept": fit.sigma.intercept, "clip_lo": fit.sigma.clip_lo,
                  "clip_hi": fit.sigma.clip_hi, "qlike": fit.sigma.qlike}
    shape_meta = {"kind": fit.shape.kind,
                  "zq": (fit.shape.zq.tolist() if fit.shape.zq is not None else None),
                  "df": fit.shape.df, "scale": fit.shape.scale,
                  "blend_w": fit.shape.blend_w}
    conf_meta = {"mode": fit.conformal.mode, "c50": fit.conformal.c50,
                 "c80": fit.conformal.c80, "lo50": fit.conformal.lo50,
                 "hi50": fit.conformal.hi50, "lo80": fit.conformal.lo80,
                 "hi80": fit.conformal.hi80,
                 "half_life_days": fit.conformal.half_life_days}
    cfg_dict = asdict(cfg)
    meta = {
        "schema_version": 3,
        "model_id": f"q3-{cfg.interval}-{hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:10]}",
        "interval": cfg.interval,
        "symbol": cfg.symbol,
        "market": cfg.market,
        "config": cfg_dict,
        "alphas": list(ALPHAS),
        "sigma": sigma_meta,
        "shape": shape_meta,
        "conformal": conf_meta,
        "c50": fit.conformal.c50,
        "c80": fit.conformal.c80,
        "median_fallback": bool(median_fallback),
        "median_source": "current_price" if median_fallback else "model",
        "n_features": len(S.names),
        "samples": S.m,
        "sample_span_utc8": [_fmt(int(S.decision_ms[0])), _fmt(int(S.decision_ms[-1]))],
        "trained_at_utc8": datetime.now(_UTC8).strftime("%Y-%m-%d %H:%M:%S"),
        "shas": shas,
        "news_coverage": _news_coverage(S),
        **meta_extra,
    }
    tmp = directory / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, directory / META_FILE)
    return directory


@dataclass
class V3Artifact:
    interval: str
    directory: Path
    meta: dict
    names: list[str]
    boosters: list[lgb.Booster] | None
    model_id: str


def load_v3(data_root: Path, interval: str) -> V3Artifact | None:
    directory = Path(data_root) / ARTIFACT_ROOT_NAME / interval
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("schema_version", 2)) < 3:
        return None
    names = json.loads((directory / FEATURES_FILE).read_text(encoding="utf-8"))
    boosters = None
    if meta.get("shape", {}).get("kind") in ("quantile", "blend"):
        boosters = [lgb.Booster(model_file=str(directory / q.model_file(a))) for a in ALPHAS]
    return V3Artifact(interval=interval, directory=directory, meta=meta, names=names,
                      boosters=boosters, model_id=str(meta.get("model_id")))


# ---------------------------------------------------------------- inference

def _load_inference_samples(data_root: Path, interval: str, cfg_dict: dict,
                            now_ms: int) -> tuple[SamplesV3, int]:
    cfg = default_config(interval, cfg_dict)
    db_path = Path(data_root) / "superE.sqlite"
    qc = _qcfg(cfg)
    groups = q.feature_groups(qc)
    conn = data.open_readonly(db_path)
    try:
        bars = data.load_bars(conn, cfg.market, cfg.symbol, interval, 1500)
        funding = data.load_funding(conn, cfg.symbol)
        metrics = data.load_metrics(conn, cfg.symbol)
        mark = data.load_bars(conn, "um_mark", cfg.symbol, interval, 1500)
        index = data.load_bars(conn, "um_index", cfg.symbol, interval, 1500)
        premium = data.load_bars(conn, "um_premium", cfg.symbol, interval, 1500)
        htf_interval = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d"}.get(interval)
        try:
            higher = data.load_bars(conn, "um", cfg.symbol, htf_interval, None) if htf_interval else None
        except ValueError:
            higher = None
        btc_bars = None
        if "btc" in groups:
            try:
                btc_bars = data.load_bars(conn, cfg.market, cfg.btc_symbol, interval, 1500)
            except ValueError:
                btc_bars = None
        one_min = None
        try:
            one_min = data.load_bars(conn, cfg.market, cfg.symbol, "1m", 60_000)
        except ValueError:
            one_min = None
    finally:
        conn.close()
    n_closed = q.closed_count(bars.close_time, now_ms)
    if n_closed == 0:
        raise ValueError("no_bars")
    closed = bars.slice_to(n_closed)
    if one_min is None:
        n1 = 0
        one_min = data.Bars(interval="1m", open_time=np.zeros(0, dtype=np.int64),
                            o=np.zeros(0), h=np.zeros(0), l=np.zeros(0), c=np.zeros(0),
                            volume=np.zeros(0), quote_volume=np.zeros(0),
                            trades=np.zeros(0), taker_buy_base=np.zeros(0))
    else:
        n1 = q.closed_count(one_min.close_time, now_ms)
        if n1 > 0:
            one_min = one_min.slice_to(n1)
    base_names, base_X, sig_base, r1 = q.build_features(
        closed, funding, metrics, qc, mark, index, premium, higher,
        altdata.load_alt(Path(data_root), groups, btc=btc_bars), btc_bars)
    rvf = rv.build(one_min, closed.close_time, closed.interval_ms)
    if rvf.n_1m < 2:
        rvf = rv.RVFeatures(names=[], X=np.full((closed.n, 0), np.nan),
                            rv_future=np.full(closed.n, np.nan),
                            rv_past=np.full(closed.n, np.nan),
                            decision_ms=closed.close_time, first_1m_ms=0,
                            last_1m_ms=0, n_1m=0)
    news = None
    news_cols = None
    if cfg.use_news:
        news = newsfeat.load_news(db_path, end_ms=now_ms)
        news_cols = newsfeat.build_columns(closed.close_time, closed.interval_ms, news)
    names, X = assemble(base_names, base_X, rvf, cfg, news_cols)
    sig_garch = q._garch_vol(r1)
    S = SamplesV3(idx=np.arange(closed.n), X=X, y=np.full(closed.n, np.nan),
                  sig_base=sig_base, sig_garch=sig_garch,
                  sig_base_full=sig_base, sig_garch_full=sig_garch, sig_full=sig_base,
                  names=names,
                  ref_close=closed.c, decision_ms=closed.close_time, bars_n=closed.n,
                  r1=r1, closes=closed.c, rvf=rvf, news=news, base_names=base_names)
    return S, n_closed - 1


def predict_latest(data_root: Path, interval: str, art: V3Artifact, now_ms: int) -> dict:
    S, last = _load_inference_samples(Path(data_root), interval, art.meta["config"], now_ms)
    if S.names != art.names:
        raise ValueError(f"{interval} 的特征列表和当前代码不一致，需要重新训练")
    row = S.X[last]
    core = core_indices(S.names, default_config(interval, art.meta["config"]))
    if not np.all(np.isfinite(row[core])):
        return {"status": "no_features", "message": "核心特征还不完整（历史不够长）",
                "price_forecast": None}
    meta = art.meta
    kind = meta["sigma"]["kind"]
    if kind == "ewma":
        sigma = float(S.sig_base[last])
    elif kind == "garch":
        sigma = float(S.sig_garch[last])
    else:
        cfg = default_config(interval, meta["config"])
        names, Z = _sigma_design(S, cfg, kind)
        idx = {n: i for i, n in enumerate(names)}
        x = np.asarray([Z[last, idx[n]] for n in meta["sigma"]["feature_names"]])
        pred = float(meta["sigma"]["intercept"]) + float(x @ np.asarray(meta["sigma"]["coef"]))
        sigma = float(np.clip(math.sqrt(math.exp(min(5.0, max(-30.0, pred)))),
                              meta["sigma"]["clip_lo"], meta["sigma"]["clip_hi"]))
    shape = meta["shape"]
    cfg_meta = default_config(interval, meta["config"])
    if shape["kind"] == "quantile":
        Qz = q.predict_raw(art.boosters, row.reshape(1, -1))
        Q = q.sort_rows(Qz * sigma) if cfg_meta.normalize else q.sort_rows(Qz)
    elif shape["kind"] == "student_t":
        Qz = np.asarray(stats.t.ppf(np.asarray(ALPHAS), shape["df"], loc=0.0,
                                    scale=shape["scale"]), dtype=np.float64).reshape(1, -1)
        Q = q.sort_rows(Qz * sigma)
    elif shape["kind"] == "blend":
        Qg = q.predict_raw(art.boosters, row.reshape(1, -1))
        Qt = np.asarray(stats.t.ppf(np.asarray(ALPHAS), shape["df"], loc=0.0,
                                    scale=shape["scale"]), dtype=np.float64).reshape(1, -1)
        w = float(shape.get("blend_w") or 0.5)
        Qz = w * Qg + (1.0 - w) * Qt
        Q = q.sort_rows(Qz * sigma) if cfg_meta.normalize else q.sort_rows(Qz)
    else:
        Qz = np.asarray(shape["zq"], dtype=np.float64).reshape(1, -1)
        Q = q.sort_rows(Qz * sigma)
    conf_meta = meta["conformal"]
    conf = ConformalFit(mode=conf_meta["mode"],
                        lo50=conf_meta.get("lo50", conf_meta.get("c50", 0.0)),
                        hi50=conf_meta.get("hi50", conf_meta.get("c50", 0.0)),
                        lo80=conf_meta.get("lo80", conf_meta.get("c80", 0.0)),
                        hi80=conf_meta.get("hi80", conf_meta.get("c80", 0.0)),
                        half_life_days=conf_meta.get("half_life_days", 30.0))
    Q = apply_conformal(Q, np.asarray([sigma]), conf)
    if meta.get("median_fallback"):
        Q = center_quantiles(Q)
    Q = q.sort_rows(Q)
    ref_close = float(S.ref_close[last])
    decision_ms = int(S.decision_ms[last])
    top = []
    explain = "尺度模型：" + str(kind)
    if shape["kind"] in ("quantile", "blend") and art.boosters:
        top = q._quantile_explain(art.boosters[2], art.names, row, 3)
        CN = getattr(q, "CN_LABEL", {})
        explain = "主要特征：" + "、".join(CN.get(f["name"], f["name"]) for f in top)
    horizon_ms = INTERVAL_MS[interval]
    pf = {
        "interval": interval,
        "asof_utc8": _fmt(decision_ms),
        "horizon_end_utc8": _fmt(decision_ms + horizon_ms),
        "horizon_end_time": _fmt(decision_ms + horizon_ms),
        "ref_close": ref_close,
        "median": float(ref_close * math.exp(float(Q[0, 2]))),
        "p10": float(ref_close * math.exp(float(Q[0, 0]))),
        "p25": float(ref_close * math.exp(float(Q[0, 1]))),
        "p75": float(ref_close * math.exp(float(Q[0, 3]))),
        "p90": float(ref_close * math.exp(float(Q[0, 4]))),
        "width50_bps": round(float((Q[0, 3] - Q[0, 1]) * 1e4), 4),
        "width80_bps": round(float((Q[0, 4] - Q[0, 0]) * 1e4), 4),
        "sigma_hat": round(float(sigma), 8),
        "sigma_model": kind,
        "shape_model": shape["kind"],
        "median_source": "current_price" if meta.get("median_fallback") else "model",
        "coverage_ok": bool(meta.get("gates", {}).get("pooled_cov_rule", False)),
        "status": meta.get("status", "仅参考"),
        "explain": explain,
        "top_features": top,
    }
    age_bars = (now_ms - decision_ms) / INTERVAL_MS[interval]
    status = "ok" if age_bars <= 2 else "stale"
    return {
        "interval": interval, "debug": True, "status": status,
        "price_forecast": pf if status == "ok" else None,
        "message": ("价格区间预测，样本外弱，勿实盘" if status == "ok"
                    else "数据已过期，采集器可能没在跑"),
        "model_id": art.model_id,
        "trained_at_utc8": meta.get("trained_at_utc8"),
    }


# ---------------------------------------------------------------- CLI

def _logger(path: Path | None):
    def log_fn(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if path is not None:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    return log_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m supere.train.quantile_v3")
    parser.add_argument("mode", choices=["cv", "holdout", "fit"])
    parser.add_argument("--intervals", default=",".join(PRODUCT_INTERVALS))
    parser.add_argument("--round", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    data_root = Path(args.data_dir).resolve()
    db_path = data_root / "superE.sqlite"
    out_dir = data_root / "cv_quantile_v3" / args.round
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fn = _logger(out_dir / "run.log")
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    summary: dict = {}
    for interval in intervals:
        base_cfg = default_config(interval)
        cfg = default_config(interval, parse_overrides(args.set, base_cfg))
        log_fn(f"== {interval} mode={args.mode} round={args.round} cfg={json.dumps(asdict(cfg), ensure_ascii=False)}")
        S = load_samples(cfg, db_path, data_root)
        log_fn(f"  samples={S.m} span={_fmt(int(S.decision_ms[0]))}..{_fmt(int(S.decision_ms[-1]))} "
               f"features={len(S.names)}")
        base_global = q.baseline_quantiles(S, _qcfg(cfg))
        if args.mode == "cv":
            res = run_cv(S, cfg, base_global, log_fn)
            (out_dir / f"{interval}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            summary[interval] = {"model": res["model"], "gates": res["gates"],
                                 "best_baseline": res["best_baseline"],
                                 "median_fallback": res["median_fallback"]}
            log_fn(f"  pass={res['gates']['pass']} cov50={res['model']['cov50']} "
                   f"cov80={res['model']['cov80']} pin={res['model']['pinball_combined_bps']} "
                   f"best={res['best_baseline']} "
                   f"{res['baselines'][res['best_baseline']]['pinball_combined_bps']} "
                   f"w50={res['model']['w50_bps']} gates={res['gates']}")
        elif args.mode == "holdout":
            cv_path = out_dir / f"{interval}.json"
            median_fallback = False
            if cv_path.exists():
                median_fallback = bool(json.loads(cv_path.read_text(encoding="utf-8")).get("median_fallback"))
            res = run_holdout(S, cfg, base_global, median_fallback)
            (out_dir / f"{interval}.holdout.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            summary[interval] = {"model": res["model"], "gates": res["gates_single_block"]}
            log_fn(f"  holdout {json.dumps(res['model'], ensure_ascii=False)}")
        else:
            all_pos = np.arange(S.m, dtype=np.int64)
            fit = fit_block(S, all_pos, cfg)
            cv_path = out_dir / f"{interval}.json"
            hold_path = out_dir / f"{interval}.holdout.json"
            extra = _round_verdict(cv_path, hold_path)
            median_fallback = bool(extra.get("cv", {}).get("median_fallback", False))
            directory = save_artifacts(data_root, S, cfg, fit, extra, median_fallback)
            log_fn(f"  saved {directory} sigma={fit.sigma.kind} shape={fit.shape.kind} "
                   f"conformal={fit.conformal.mode} median_fallback={median_fallback}")
            summary[interval] = {"saved": str(directory), "sigma": fit.sigma.kind,
                                 "shape": fit.shape.kind, "conformal": fit.conformal.mode,
                                 "median_fallback": median_fallback}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


def _round_verdict(cv_path: Path, hold_path: Path) -> dict:
    if not cv_path.exists():
        return {"status": "仅参考", "verdict_note": "没有这一轮的 CV 结果文件"}
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    g = cv["gates"]
    status = "达标" if g.get("main_pass") else "仅参考"
    extra = {
        "status": status,
        "coverage_ok": bool(g.get("pooled_cov_rule")),
        "gates": g,
        "cv": {"model": cv["model"], "baselines": cv["baselines"],
               "best_baseline": cv["best_baseline"], "folds": cv["folds"],
               "median_fallback": cv.get("median_fallback", False),
               "model_before_median_fallback": cv.get("model_before_median_fallback")},
    }
    if hold_path.exists():
        hold = json.loads(hold_path.read_text(encoding="utf-8"))
        extra["holdout"] = {"model": hold["model"], "baselines": hold["baselines"],
                            "best_baseline": hold["best_baseline"],
                            "gates": hold["gates_single_block"]}
        # A main pass must also improve on the holdout in the same direction.
        try:
            cv_imp = g["pinball_improve_vs_best_base"]
            hb = hold["baselines"][hold["best_baseline"]]["pinball_combined_bps"]
            h_imp = 1.0 - hold["model"]["pinball_combined_bps"] / hb
            extra["holdout_improve"] = round(float(h_imp), 4)
            if cv_imp > 0 and h_imp <= 0:
                extra["status"] = "仅参考"
        except Exception:
            pass
    return extra


if __name__ == "__main__":
    raise SystemExit(main())
