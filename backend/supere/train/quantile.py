"""Quantile price forecast for the next bar close (plan: 点位区间预测, 2026-11-05).

For every interval ``h`` in {5m, 15m, 30m, 1h, 4h} the model predicts the close
``h`` ahead of the newest closed bar:

* target ``y_i = log(C[i+1] / C[i])`` for decision bar ``i`` (its close is the
  decision moment; ``C[i+1]`` is the close one interval later);
* one LightGBM quantile regressor per alpha in ``ALPHAS``; the five outputs are
  sorted per row so they never cross;
* optional standardisation (``target='vol'``): ``y / sigma_i`` with the causal
  EWMA volatility ``sigma_i``, de-standardised after prediction;
* optional split conformal calibration (``calib='cqr'``): the last part of each
  training block is held out to widen/narrow the 50% and 80% bands so their
  coverage matches the nominal level (conformalised quantile regression).

Validation (plan section 验证): the first ``1 - holdout_frac`` of samples is the
CV region, split into ``n_folds`` expanding-window test blocks; a training
sample is kept only when its label has finished before the test block minus an
embargo. The newest ``holdout_frac`` is the final check, run once by ``final``.

Read-only: opens SQLite with ``mode=ro``. Never places orders, never reads keys.

CLI (run from the train worktree, see ``--help``)::

    python -m supere.train.quantile cv --intervals 1h --round r1 --set target=vol
    python -m supere.train.quantile holdout --intervals 1h --round r1 --set ...
    python -m supere.train.quantile fit --intervals 1h --round r1 --set ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import logging
import os
import sys
import time

log = logging.getLogger("supere.train.quantile")


def closed_count(close_time: np.ndarray, now_ms: int) -> int:
    return int(np.searchsorted(close_time, now_ms, side="right"))


from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
import lightgbm as lgb
import numpy as np

from supere.intervals import INTERVAL_MS
from supere.timeutil import UTC8 as _UTC8
from supere.train import altdata, data, features
from supere.train.runner import WINDOW_LOOKBACK

ALPHAS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
AKEY = {0.10: "p10", 0.25: "p25", 0.50: "median", 0.75: "p75", 0.90: "p90"}
PRODUCT_INTERVALS: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h")

# Snapshot used for every round so the numbers are reproducible:
# 2026-11-05 11:00 (UTC+8) = close of the newest 1h bar at the time of the run.
SNAPSHOT_MS = 1793847600000
# 5m and 15m keep only the newest bars (about 2 years and 2.6 years), as in the
# earlier debug batch; 30m, 1h and 4h use their full history.
DEFAULT_LIMIT: dict[str, int | None] = {"5m": 200_000, "15m": 200_000, "30m": None, "1h": None, "4h": None}

DROP_FEATURES = ()  # testing calendar features
ZSCORE_SOURCES = ("taker_buy_ratio", "log_trades", "um_oi_chg_1h", "um_global_lsr", "um_taker_lsr", "funding_last",                  "mark_basis", "index_basis", "premium_index")
# May be NaN (history starts later, or the series is optional); not required to be finite.
OPTIONAL_PREFIXES = ("um_", "funding_", "mark_", "index_", "premium_",
                     "btc_", "eth_btc_", "dvol_", "macro_")
FEATURE_GROUPS: tuple[str, ...] = ("btc", "dvol", "macro")
ZWIN = 100
MODEL_VERSION = 2

CN_LABEL = {
    "ret_1": "上一根涨跌", "ret_3": "3根涨跌", "ret_5": "5根涨跌", "ret_10": "10根涨跌",
    "ret_20": "20根涨跌", "ret_50": "50根涨跌", "rsi_14": "RSI", "macd_hist_pct": "MACD 柱",
    "atr_pct": "ATR 波动", "rv_20": "20根实现波动", "rv_5": "5根实现波动",
    "parkinson_20": "Parkinson 波动", "donchian_pos_20": "区间位置", "bb_pos_20": "布林位置",
    "bb_width_20": "布林宽度", "vol_ratio_20": "量能比", "taker_buy_ratio": "主动买入占比",
    "log_trades": "成交笔数", "log_avg_trade_quote": "单笔成交额", "body_pct": "实体幅度",
    "range_pct": "振幅", "funding_last": "资金费率", "funding_hours_since": "距上次资金费率",
    "um_oi_chg_1h": "持仓量1小时变化", "um_top_pos_lsr": "大户持仓多空比",
    "um_top_acc_lsr": "大户账户多空比", "um_global_lsr": "全局多空比", "um_taker_lsr": "主动买卖比",
    "ewma_vol_fast": "短期 EWMA 波动", "ewma_vol_slow": "长期 EWMA 波动",
    "vol_fast_slow": "短长波动比", "absret_1": "上一根绝对涨跌",
    "btc_ret_1": "BTC 上一根涨跌", "btc_ret_3": "BTC 3根涨跌", "btc_ret_5": "BTC 5根涨跌",
    "btc_ret_20": "BTC 20根涨跌", "btc_rv_20": "BTC 20根波动", "btc_rv_50": "BTC 50根波动",
    "eth_btc_rs_1": "ETH/BTC 相对强弱1根", "eth_btc_rs_3": "ETH/BTC 相对强弱3根",
    "eth_btc_rs_5": "ETH/BTC 相对强弱5根", "eth_btc_rs_20": "ETH/BTC 相对强弱20根",
    "eth_btc_corr_20": "ETH-BTC 20根相关", "eth_btc_corr_60": "ETH-BTC 60根相关",
    "eth_btc_beta_20": "ETH 对 BTC 的 beta", "eth_btc_vol_ratio_20": "ETH/BTC 波动比",
    "dvol_level": "DVOL 水平", "dvol_chg_1h": "DVOL 1小时变化", "dvol_chg_4h": "DVOL 4小时变化",
    "dvol_chg_1d": "DVOL 1天变化", "dvol_chg_7d": "DVOL 7天变化", "dvol_z_30d": "DVOL 30天 z",
    "dvol_rv_ratio": "DVOL/实现波动", "macro_min_to_next": "距下一宏观事件",
    "macro_min_since_prev": "距上一宏观事件", "macro_in_next_window": "下窗口有宏观事件",
    "macro_in_prev_window": "上窗口有宏观事件",
}


class QuantileError(RuntimeError):
    pass


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class QConfig:
    interval: str
    symbol: str = "ETHUSDT"
    market: str = "um"
    limit_bars: int | None = None
    snapshot_ms: int | None = SNAPSHOT_MS
    target: str = "raw"  # raw | vol
    calib: str = "none"  # none | cqr
    calib_frac: float = 0.2
    use_funding: bool = True
    use_metrics: bool = True
    zscore: bool = True
    n_folds: int = 5
    holdout_frac: float = 0.15
    embargo: int = 3
    feature_groups: str = ""  # comma list of btc,dvol,macro
    btc_symbol: str = "BTCUSDT"
    ewma_lambda: float = 0.94
    aci_gamma: float = 0.0
    vol_model: str = "ewma"  # ewma | garch
    baseline_window: int = 500
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 42
    n_jobs: int = 4


def default_config(interval: str, overrides: dict | None = None) -> QConfig:
    if interval not in INTERVAL_MS:
        raise QuantileError(f"未知周期 {interval}")
    cfg = QConfig(interval=interval, limit_bars=DEFAULT_LIMIT.get(interval))
    if overrides:
        cfg = replace(cfg, **overrides)
    validate(cfg)
    return cfg


def validate(cfg: QConfig) -> None:
    if cfg.target not in ("raw", "vol"):
        raise QuantileError("target 只能是 raw 或 vol")
    if getattr(cfg, "vol_model", "ewma") not in ("ewma", "garch"):
        raise QuantileError("vol_model bad")
    if cfg.calib not in ("none", "cqr"):
        raise QuantileError("calib 只能是 none 或 cqr")
    bad = feature_groups(cfg) - set(FEATURE_GROUPS)
    if bad:
        raise QuantileError(f"未知特征组 {sorted(bad)}；可选 {', '.join(FEATURE_GROUPS)}")
    if not 0.0 < cfg.calib_frac < 0.5:
        raise QuantileError("calib_frac 必须在 (0, 0.5)")
    if cfg.n_folds < 1 or not 0.0 < cfg.holdout_frac < 0.5:
        raise QuantileError("折数或留存比例不合法")


def feature_groups(cfg: QConfig) -> set[str]:
    text = (cfg.feature_groups or "").strip().lower()
    if text in ("all", "全部"):
        return set(FEATURE_GROUPS)
    return {part.strip() for part in text.split(",") if part.strip()}


def parse_overrides(pairs: list[str], base: QConfig) -> dict:
    """``key=value`` strings -> typed override dict, using the field types of QConfig."""
    types = {f.name: f.type for f in fields(QConfig)}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise QuantileError(f"--set 需要 key=value，收到 {pair}")
        key, text = pair.split("=", 1)
        if key not in types or key == "interval":
            raise QuantileError(f"未知配置项 {key}")
        field = next(f for f in fields(QConfig) if f.name == key)
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


# ---------------------------------------------------------------- features

def _ewma_vol(r: np.ndarray, lam: float) -> np.ndarray:
    """sigma_i from returns r_0..r_i only (EWMA of squared returns, seeded at the first return)."""
    out = np.full(r.shape[0], np.nan)
    v = np.nan
    for i in range(r.shape[0]):
        x = r[i]
        if not np.isfinite(x):
            continue
        v = x * x if not np.isfinite(v) else lam * v + (1.0 - lam) * x * x
        out[i] = math.sqrt(v)
    return out


def _garch_vol(r: np.ndarray, alpha: float = 0.05, beta: float = 0.94) -> np.ndarray:
    out = np.full(r.shape[0], np.nan)
    v = np.nan
    omega = 1e-6
    longvar = np.nanvar(r[np.isfinite(r)][:1000]) if np.isfinite(r).sum() > 10 else 1e-4
    omega = max(1e-8, (1.0 - alpha - beta) * longvar)
    for i in range(r.shape[0]):
        x = r[i]
        if not np.isfinite(x):
            continue
        v = longvar if not np.isfinite(v) else omega + alpha * x * x + beta * v
        out[i] = math.sqrt(max(v, 1e-12))
    return out


def _zscore(x: np.ndarray, w: int) -> np.ndarray:
    m = features._roll(x, w, np.mean)
    s = features._roll(x, w, np.std)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - m) / s
    return np.where(s > 0, z, np.nan)


def build_features(bars: data.Bars, funding: data.Funding | None, metrics: data.Metrics | None,
                    cfg: QConfig, mark_bars: data.Bars | None = None,
                    index_bars: data.Bars | None = None,
                    premium_bars: data.Bars | None = None,
                    higher_bars: data.Bars | None = None,
                    alt: altdata.AltData | None = None,
                    btc_bars: data.Bars | None = None) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:

    """Causal features for every bar. Returns ``(names, X, sigma, r1)``.

    ``sigma[i]`` is the EWMA volatility of log returns up to bar ``i``; ``r1[i]``
    is ``log(C[i]/C[i-1])``. Row ``i`` reads bars ``0..i`` only.
    """
    names, X = features.build(bars, funding if cfg.use_funding else None,
                              metrics if cfg.use_metrics else None)
    cols: dict[str, np.ndarray] = {n: X[:, i] for i, n in enumerate(names) if n not in DROP_FEATURES}
    c = bars.c

    def _align(extra, field):
        out = np.full(bars.n, np.nan)
        if extra is None or extra.n == 0:
            return out
        idx = np.searchsorted(extra.open_time, bars.open_time)
        ok = (idx >= 0) & (idx < extra.n) & (extra.open_time[np.clip(idx, 0, extra.n - 1)] == bars.open_time)
        if ok.any():
            out[ok] = getattr(extra, field)[idx[ok]]
        return out

    mc = _align(mark_bars, "c")
    ic = _align(index_bars, "c")
    pc = _align(premium_bars, "c")
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["mark_basis"] = np.log(mc / c)
        cols["index_basis"] = np.log(c / ic)
        cols["premium_index"] = pc
        cols["mark_index_spread"] = np.log(mc / ic)

    # Higher-timeframe context (causal: last closed higher bar at or before decision).
    if higher_bars is not None and higher_bars.n:
        hc = _align(higher_bars, "c")
        # higher bar index whose close <= decision
        hidx = np.searchsorted(higher_bars.close_time, bars.open_time + bars.interval_ms, side="right") - 1
        hret = np.full(bars.n, np.nan)
        hvol = np.full(bars.n, np.nan)
        for i in range(bars.n):
            k = hidx[i]
            if 0 <= k < higher_bars.n and k > 0:
                hret[i] = np.log(higher_bars.c[k] / higher_bars.c[k-1])
                # realized vol of last 20 higher bars
                lo = max(0, k - 19)
                if k - lo >= 5:
                    hvol[i] = np.std(np.diff(np.log(higher_bars.c[lo:k+1])))
        cols["htf_ret"] = hret
        cols["htf_vol"] = hvol
    with np.errstate(divide="ignore", invalid="ignore"):
        log_c = np.log(c)
        r1 = np.full(bars.n, np.nan)
        r1[1:] = np.diff(log_c)
        sig = _ewma_vol(r1, cfg.ewma_lambda)
        sig_slow = _ewma_vol(r1, 0.99)
        sig_garch = _garch_vol(r1)
        cols["ewma_vol_fast"] = sig
        cols["ewma_vol_slow"] = sig_slow
        cols["vol_fast_slow"] = np.log(sig / sig_slow)
        cols["garch_vol"] = sig_garch
        if getattr(cfg, "vol_model", "ewma") == "garch":
            sig = sig_garch
        cols["rv_5"] = features._roll(r1, 5, np.std)
        cols["rv_10"] = features._roll(r1, 10, np.std)
        cols["rv_50"] = features._roll(r1, 50, np.std)
        cols["absret_1"] = np.abs(r1)
        cols["vol_of_vol"] = features._roll(np.abs(r1), 20, np.std)
        cols["ret_skew_20"] = features._roll(r1, 20, lambda x, axis=1: np.mean((x - x.mean(axis=axis, keepdims=True))**3, axis=axis) / (np.std(x, axis=axis)**3 + 1e-12))
    if cfg.zscore:
        for name in ZSCORE_SOURCES:
            if name in cols:
                cols[f"{name}_z{ZWIN}"] = _zscore(cols[name], ZWIN)
    # Alternative inputs (plan 追加 2026-11-05 15:00): BTC context, DVOL, macro calendar.
    # Appended after the base columns so old feature lists keep their order.
    groups = feature_groups(cfg)
    if groups:
        if btc_bars is not None:
            alt = replace(alt, btc=btc_bars) if alt is not None else altdata.AltData(btc=btc_bars)
        extra = altdata.build_columns(bars, bars.close_time, bars.interval_ms, sig, alt, groups)
        cols.update(extra)
    out_names = list(cols)
    out_x = np.column_stack([cols[n] for n in out_names]).astype(np.float64)
    return out_names, out_x, sig, r1


@dataclass
class Samples:
    idx: np.ndarray  # decision bar index (abs, within the loaded bars)
    X: np.ndarray
    y: np.ndarray  # log(C[i+1]/C[i]), return units
    sig: np.ndarray
    names: list[str]
    ref_close: np.ndarray
    decision_ms: np.ndarray
    bars_n: int
    r1: np.ndarray  # full-length 1-bar log returns (for baselines)
    sig_full: np.ndarray  # full-length sigma (for baselines)
    closes: np.ndarray  # full-length closes

    @property
    def m(self) -> int:
        return int(self.idx.shape[0])


def core_index(names: list[str]) -> list[int]:
    """Columns that must be finite for a row to be used. Funding and metrics may be NaN."""
    out = []
    for i, nm in enumerate(names):
        base = nm.split("_z")[0] if nm.endswith(f"_z{ZWIN}") else nm
        if not base.startswith(OPTIONAL_PREFIXES):
            out.append(i)
    return out


def make_samples(bars: data.Bars, funding: data.Funding, metrics: data.Metrics, cfg: QConfig,
                 mark_bars: data.Bars | None = None, index_bars: data.Bars | None = None,
                 premium_bars: data.Bars | None = None,
                 higher_bars: data.Bars | None = None,
                 alt: altdata.AltData | None = None,
                 btc_bars: data.Bars | None = None) -> Samples:
    names, X, sig, r1 = build_features(bars, funding, metrics, cfg, mark_bars, index_bars, premium_bars,
                                       higher_bars, alt, btc_bars)

    n = bars.n
    y = np.full(n, np.nan)
    if n > 1:
        y[:-1] = r1[1:]
    core_ok = np.all(np.isfinite(X[:, core_index(names)]), axis=1)
    gaps = data.gap_after(bars).astype(np.int64)
    gap_cum = np.concatenate([[0], np.cumsum(gaps)])
    lo = np.maximum(np.arange(n) - WINDOW_LOOKBACK, 0)
    hi = np.arange(n)
    gap_in_window = (gap_cum[hi + 1] - gap_cum[lo]) > 0
    ok = core_ok & np.isfinite(y) & np.isfinite(sig) & (sig > 0) & ~gap_in_window
    ok[n - 1] = False
    idx = np.nonzero(ok)[0].astype(np.int64)
    if idx.shape[0] < 500:
        raise QuantileError(f"{cfg.interval} 的有效样本只有 {idx.shape[0]} 个，不够训练")
    close_time = bars.close_time
    return Samples(idx=idx, X=X[idx], y=y[idx], sig=sig[idx], names=names,
                   ref_close=bars.c[idx], decision_ms=close_time[idx], bars_n=n,
                   r1=r1, sig_full=sig, closes=bars.c)


def load_samples(cfg: QConfig, db_path: Path) -> Samples:
    groups = feature_groups(cfg)
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
        higher = data.load_bars(conn, "um", cfg.symbol, htf_interval, None,
                               close_by_ms=cfg.snapshot_ms) if htf_interval else None
        btc_bars = None
        if "btc" in groups:
            try:
                btc_bars = data.load_bars(conn, cfg.market, cfg.btc_symbol, cfg.interval, cfg.limit_bars,
                                          close_by_ms=cfg.snapshot_ms)
            except ValueError as exc:
                log.warning("BTC 数据不可用，%s 组按缺失处理: %s", cfg.interval, exc)
    finally:
        conn.close()
    alt = altdata.load_alt(Path(db_path).parent, groups, btc=btc_bars)
    return make_samples(bars, funding, metrics, cfg, mark, index, premium, higher, alt, btc_bars)




# ---------------------------------------------------------------- splits

@dataclass(frozen=True)
class Block:
    number: int
    train: np.ndarray  # positions into the Samples arrays
    test: np.ndarray


def cv_blocks(S: Samples, cfg: QConfig) -> tuple[list[Block], Block]:
    """Purged expanding-window folds on the CV region, plus the final holdout block.

    The first fold block is only training data. Fold ``k`` trains on everything
    before its block whose label ended ``embargo`` bars before the block starts.
    """
    m = S.m
    cut = int(round(m * (1.0 - cfg.holdout_frac)))
    edges = np.linspace(0, cut, cfg.n_folds + 2).astype(np.int64)
    exit_bar = S.idx + 1
    folds: list[Block] = []
    for k in range(1, cfg.n_folds + 1):
        s, e = int(edges[k]), int(edges[k + 1])
        if e <= s:
            continue
        keep = np.nonzero(exit_bar[:s] < S.idx[s] - cfg.embargo)[0]
        folds.append(Block(number=k, train=keep.astype(np.int64), test=np.arange(s, e, dtype=np.int64)))
    keep_h = np.nonzero(exit_bar[:cut] < S.idx[cut] - cfg.embargo)[0]
    hold = Block(number=0, train=keep_h.astype(np.int64), test=np.arange(cut, m, dtype=np.int64))
    return folds, hold


# ---------------------------------------------------------------- models

@dataclass
class Fitted:
    boosters: list[lgb.Booster]  # one per ALPHAS
    c50: float  # conformal half-widening for the 50% band (model units), 0 if none
    c80: float  # same for the 80% band
    n_train: int
    n_calib: int


def _model_target(S: Samples, pos: np.ndarray, cfg: QConfig) -> np.ndarray:
    y = S.y[pos]
    return y / S.sig[pos] if cfg.target == "vol" else y


def _fit_quantiles(X: np.ndarray, y: np.ndarray, cfg: QConfig) -> list[lgb.Booster]:
    out = []
    for a in ALPHAS:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=a,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )
        model.fit(X, y)
        out.append(model.booster_)
    return out


def sort_rows(Q: np.ndarray) -> np.ndarray:
    return np.sort(Q, axis=1)


def predict_raw(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    return sort_rows(np.column_stack([b.predict(X) for b in boosters]))


def conformal_half_width(E: np.ndarray, miscoverage: float) -> float:
    n = E.shape[0]
    if n == 0:
        return 0.0
    level = min(1.0, math.ceil((n + 1) * (1.0 - miscoverage)) / n)
    return float(np.quantile(E, level, method="higher"))


def apply_calibration(Q: np.ndarray, c50: float, c80: float) -> np.ndarray:
    """Widen (or narrow, if negative) the 50% and 80% bands around the model quantiles."""
    out = Q.copy()
    out[:, 1] -= c50
    out[:, 3] += c50
    out[:, 0] -= c80
    out[:, 4] += c80
    return sort_rows(out)


def fit_block(S: Samples, train: np.ndarray, cfg: QConfig) -> Fitted:
    """Fit on ``train`` positions. With ``calib='cqr'`` the newest part is held out for calibration."""
    if train.shape[0] < 200:
        raise QuantileError(f"训练样本只有 {train.shape[0]} 个，太少")
    if cfg.calib == "none":
        fit_pos = train
        boosters = _fit_quantiles(S.X[fit_pos], _model_target(S, fit_pos, cfg), cfg)
        return Fitted(boosters, 0.0, 0.0, int(fit_pos.shape[0]), 0)

    cut = int(round(train.shape[0] * (1.0 - cfg.calib_frac)))
    cal = train[cut:]
    part = train[:cut]
    # Purge again at the calibration boundary so no training label reaches into it.
    part = part[S.idx[part] + 1 < S.idx[cal[0]] - cfg.embargo]
    if part.shape[0] < 200 or cal.shape[0] < 50:
        raise QuantileError("校准切分后样本太少")
    boosters = _fit_quantiles(S.X[part], _model_target(S, part, cfg), cfg)
    Qc = predict_raw(boosters, S.X[cal])
    yc = _model_target(S, cal, cfg)
    c50 = conformal_half_width(np.maximum(Qc[:, 1] - yc, yc - Qc[:, 3]), 0.5)
    c80 = conformal_half_width(np.maximum(Qc[:, 0] - yc, yc - Qc[:, 4]), 0.2)
    return Fitted(boosters, c50, c80, int(part.shape[0]), int(cal.shape[0]))


def predict_block(S: Samples, fit: Fitted, pos: np.ndarray, cfg: QConfig) -> np.ndarray:
    """Quantile returns (log units) for the given sample positions."""
    Q = predict_raw(fit.boosters, S.X[pos])
    if fit.c50 or fit.c80:
        Q = apply_calibration(Q, fit.c50, fit.c80)
    if cfg.target == "vol":
        Q = Q * S.sig[pos][:, None]
    Q = sort_rows(Q)
    if getattr(cfg, "aci_gamma", 0.0) and pos.shape[0]:
        gamma = float(cfg.aci_gamma)
        y = S.y[pos]
        out = Q.copy()
        s50 = 1.0
        s80 = 1.0
        for i in range(pos.shape[0]):
            center = float(Q[i, 2])
            half50 = float(Q[i, 3] - Q[i, 1]) / 2.0
            half80 = float(Q[i, 4] - Q[i, 0]) / 2.0
            lo50 = center - half50 * s50
            hi50 = center + half50 * s50
            lo80 = center - half80 * s80
            hi80 = center + half80 * s80
            out[i] = np.array([lo80, lo50, center, hi50, hi80])
            out[i] = np.sort(out[i])
            hit50 = bool(lo50 <= y[i] <= hi50)
            hit80 = bool(lo80 <= y[i] <= hi80)
            e50 = 0.0 if hit50 else 1.0
            e80 = 0.0 if hit80 else 1.0
            s50 = min(3.0, max(0.5, s50 + gamma * (e50 - 0.5)))
            s80 = min(3.0, max(0.5, s80 + gamma * (e80 - 0.2)))
        Q = sort_rows(out)
    return Q

# ---------------------------------------------------------------- baselines

def _pinball(y: np.ndarray, q: np.ndarray, a: float) -> np.ndarray:
    d = y - q
    return np.maximum(a * d, (a - 1.0) * d)


def baseline_quantiles(S: Samples, cfg: QConfig) -> dict[str, np.ndarray]:
    """Three baselines, quantile returns per sample (same units as the target).

    * ``rw_emp``: median = current price; band = empirical quantiles of the last N one-bar returns.
    * ``ewma_gauss``: median = current price; band = Gaussian with EWMA sigma.
    * ``ewma_emp``: empirical quantiles of the last N EWMA-standardised returns, times sigma.
    Only data up to bar ``i`` is used for the sample at bar ``i``.
    """
    m = S.m
    N = cfg.baseline_window
    qs = np.array(ALPHAS)
    nd = NormalDist()
    z = np.array([nd.inv_cdf(a) for a in ALPHAS])
    out: dict[str, np.ndarray] = {}

    sig_s = S.sig_full[S.idx]
    out["ewma_gauss"] = sort_rows(sig_s[:, None] * z[None, :])

    z_std = np.full(S.bars_n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_std[:] = S.r1 / S.sig_full
    rw = np.empty((m, len(ALPHAS)))
    ew = np.empty((m, len(ALPHAS)))
    chunk = 20_000
    offsets = np.arange(N - 1, -1, -1)
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        I = S.idx[s:e][:, None] - offsets[None, :]
        valid = I >= 0
        Ic = np.where(valid, I, 0)
        W = np.where(valid, S.r1[Ic], np.nan)
        rw[s:e] = np.nanquantile(W, qs, axis=1).T
        Wz = np.where(valid, z_std[Ic], np.nan)
        ew[s:e] = np.nanquantile(Wz, qs, axis=1).T * sig_s[s:e, None]
    rw = rw.copy()
    rw[:, 2] = 0.0
    out["rw_emp"] = sort_rows(rw)
    out["ewma_emp"] = sort_rows(ew)
    return out


# ---------------------------------------------------------------- metrics

def score(y: np.ndarray, Q: np.ndarray, sig: np.ndarray) -> dict:
    """All numbers in bps of the reference price (log units x 1e4), coverage as fractions."""
    pin = [float(np.mean(_pinball(y, Q[:, k], a))) for k, a in enumerate(ALPHAS)]
    inside50 = (y >= Q[:, 1]) & (y <= Q[:, 3])
    inside80 = (y >= Q[:, 0]) & (y <= Q[:, 4])
    err = np.abs(y - Q[:, 2])
    nz = (y != 0) & (Q[:, 2] != 0)
    return {
        "n": int(y.shape[0]),
        "pinball_bps": {AKEY[a]: round(p * 1e4, 4) for a, p in zip(ALPHAS, pin)},
        "pinball_combined_bps": round(float(np.mean(pin)) * 1e4, 4),
        "cov50": round(float(inside50.mean()), 4),
        "cov80": round(float(inside80.mean()), 4),
        "w50_bps": round(float(np.mean(Q[:, 3] - Q[:, 1])) * 1e4, 4),
        "w80_bps": round(float(np.mean(Q[:, 4] - Q[:, 0])) * 1e4, 4),
        "mae_bps": round(float(np.mean(err)) * 1e4, 4),
        "mdae_bps": round(float(np.median(err)) * 1e4, 4),
        "dir_hit": round(float(np.mean(np.sign(y[nz]) == np.sign(Q[nz, 2]))), 4) if nz.any() else None,
        "regime": regime(y, Q, sig),
    }


def regime(y: np.ndarray, Q: np.ndarray, sig: np.ndarray) -> dict:
    """Coverage split by the volatility tercile of the sample (low, mid, high)."""
    if y.shape[0] < 30:
        return {}
    cuts = np.quantile(sig, [1 / 3, 2 / 3])
    bucket = np.digitize(sig, cuts)
    names = ("低波动", "中波动", "高波动")
    out = {}
    for b, name in enumerate(names):
        mask = bucket == b
        if not mask.any():
            continue
        out[name] = {
            "n": int(mask.sum()),
            "cov50": round(float(((y[mask] >= Q[mask, 1]) & (y[mask] <= Q[mask, 3])).mean()), 4),
            "cov80": round(float(((y[mask] >= Q[mask, 0]) & (y[mask] <= Q[mask, 4])).mean()), 4),
        }
    return out


def gates(model: dict, folds: list[dict], best_base: dict, base_scores: dict[str, dict]) -> dict:
    """The four acceptance rules of the plan, one per key. Each is True/False."""
    ok50 = sum(1 for f in folds if abs(f["cov50"] - 0.5) <= 0.02)
    ok80 = sum(1 for f in folds if abs(f["cov80"] - 0.8) <= 0.02)
    improve = 1.0 - model["pinball_combined_bps"] / best_base["pinball_combined_bps"]
    calibrated_bases = [b for b in base_scores.values() if abs(b["cov50"] - 0.5) <= 0.02]
    ref = calibrated_bases or list(base_scores.values())
    width_ref = min(b["w50_bps"] for b in ref)
    out = {
        "folds_cov50_ok": f"{ok50}/{len(folds)}",
        "folds_cov80_ok": f"{ok80}/{len(folds)}",
        "cov_rule": ok50 >= 4 and ok80 >= 4,
        "pinball_improve_vs_best_base": round(improve, 4),
        "pinball_rule": improve >= 0.03,
        "width50_ref_bps": round(width_ref, 4),
        "width_rule": model["w50_bps"] <= width_ref,
    }
    out["pass"] = bool(out["cov_rule"] and out["pinball_rule"] and out["width_rule"])
    return out


# ---------------------------------------------------------------- runs

def run_cv(S: Samples, cfg: QConfig, base: dict[str, np.ndarray], log) -> dict:
    folds, _ = cv_blocks(S, cfg)
    fold_out = []
    pooled_pos, pooled_Q = [], []
    for blk in folds:
        fit = fit_block(S, blk.train, cfg)
        Q = predict_block(S, fit, blk.test, cfg)
        y = S.y[blk.test]
        sc = score(y, Q, S.sig[blk.test])
        sc.update({"fold": blk.number, "n_train": fit.n_train, "n_calib": fit.n_calib,
                   "c50": round(fit.c50, 6), "c80": round(fit.c80, 6)})
        fold_out.append(sc)
        pooled_pos.append(blk.test)
        pooled_Q.append(Q)
        log(f"  fold {blk.number}: train={fit.n_train} test={blk.test.shape[0]} "
            f"cov50={sc['cov50']} cov80={sc['cov80']} pin={sc['pinball_combined_bps']} w50={sc['w50_bps']}")
    pos = np.concatenate(pooled_pos)
    Q = np.vstack(pooled_Q)
    y = S.y[pos]
    model = score(y, Q, S.sig[pos])
    base_scores = {}
    for name, B in base.items():
        base_scores[name] = score(y, B[pos], S.sig[pos])
    best_name = min(base_scores, key=lambda k: base_scores[k]["pinball_combined_bps"])
    g = gates(model, fold_out, base_scores[best_name], base_scores)
    return {"model": model, "folds": fold_out, "baselines": base_scores,
            "best_baseline": best_name, "gates": g}


def run_holdout(S: Samples, cfg: QConfig, base: dict[str, np.ndarray]) -> dict:
    _, hold = cv_blocks(S, cfg)
    fit = fit_block(S, hold.train, cfg)
    Q = predict_block(S, fit, hold.test, cfg)
    y = S.y[hold.test]
    model = score(y, Q, S.sig[hold.test])
    base_scores = {name: score(y, B[hold.test], S.sig[hold.test]) for name, B in base.items()}
    best_name = min(base_scores, key=lambda k: base_scores[k]["pinball_combined_bps"])
    g = gates(model, [{"cov50": model["cov50"], "cov80": model["cov80"]}], base_scores[best_name], base_scores)
    return {"model": model, "baselines": base_scores, "best_baseline": best_name,
            "gates_single_block": g, "n_train": fit.n_train, "n_calib": fit.n_calib,
            "c50": fit.c50, "c80": fit.c80}


# ---------------------------------------------------------------- artifacts

ARTIFACT_ROOT_NAME = "models_quantile"
META_FILE = "meta.json"
FEATURES_FILE = "features.json"


def model_file(alpha: float) -> str:
    return f"model_{AKEY[alpha]}.txt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_artifacts(data_root: Path, S: Samples, cfg: QConfig, fit: Fitted, names: list[str],
                   meta_extra: dict) -> Path:
    directory = Path(data_root) / ARTIFACT_ROOT_NAME / cfg.interval
    directory.mkdir(parents=True, exist_ok=True)
    shas = {}
    for a, booster in zip(ALPHAS, fit.boosters):
        target = directory / model_file(a)
        tmp = directory / (model_file(a) + ".tmp")
        booster.save_model(str(tmp))
        os.replace(tmp, target)
        shas[model_file(a)] = _sha(target)
    feat_text = json.dumps(names, ensure_ascii=False, indent=2)
    (directory / (FEATURES_FILE + ".tmp")).write_text(feat_text, encoding="utf-8")
    os.replace(directory / (FEATURES_FILE + ".tmp"), directory / FEATURES_FILE)
    shas[FEATURES_FILE] = _sha(directory / FEATURES_FILE)
    cfg_dict = asdict(cfg)
    meta = {
        "schema_version": MODEL_VERSION,
        "model_id": f"q-{cfg.interval}-{hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:10]}",
        "interval": cfg.interval,
        "symbol": cfg.symbol,
        "market": cfg.market,
        "config": cfg_dict,
        "alphas": list(ALPHAS),
        "c50": fit.c50,
        "c80": fit.c80,
        "n_features": len(names),
        "samples": S.m,
        "sample_span_utc8": [_fmt(int(S.decision_ms[0])), _fmt(int(S.decision_ms[-1]))],
        "trained_at_utc8": datetime.now(_UTC8).strftime("%Y-%m-%d %H:%M:%S"),
        "shas": shas,
        **meta_extra,
    }
    tmp = directory / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, directory / META_FILE)
    return directory


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=_UTC8).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- CLI

def _round_verdict(out_dir: Path, interval: str) -> dict:
    """Gate results of the round this model came from. Missing files mean 仅参考."""
    cv_path = out_dir / f"{interval}.json"
    if not cv_path.exists():
        return {"status": "仅参考", "verdict_note": "没有这一轮的 CV 结果文件"}
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    g = cv["gates"]
    extra = {
        "status": "达标" if g.get("pass") else "仅参考",
        "coverage_ok": bool(g.get("cov_rule")),
        "gates": g,
        "cv_model": cv["model"],
        "cv_baselines": cv["baselines"],
        "cv_best_baseline": cv["best_baseline"],
        "cv_folds": cv["folds"],
    }
    hold_path = out_dir / f"{interval}.holdout.json"
    if hold_path.exists():
        hold = json.loads(hold_path.read_text(encoding="utf-8"))
        extra["holdout"] = {"model": hold["model"], "baselines": hold["baselines"],
                            "best_baseline": hold["best_baseline"], "gates": hold["gates_single_block"]}
    return extra


def _logger(path: Path | None):
    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if path is not None:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m supere.train.quantile")
    parser.add_argument("mode", choices=["cv", "holdout", "fit"])
    parser.add_argument("--intervals", default=",".join(PRODUCT_INTERVALS))
    parser.add_argument("--round", required=True, help="轮次标签，用于输出文件名，例如 r1")
    parser.add_argument("--data-dir", required=True, help="数据目录（含 superE.sqlite）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--db", default=None, help="默认 <data-dir>/superE.sqlite")
    args = parser.parse_args(argv)

    data_root = Path(args.data_dir).resolve()
    db_path = Path(args.db).resolve() if args.db else data_root / "superE.sqlite"
    out_dir = data_root / "cv_quantile" / args.round
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _logger(out_dir / "run.log")
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    summary: dict = {}
    for interval in intervals:
        base_cfg = default_config(interval)
        overrides = parse_overrides(args.set, base_cfg)
        cfg = default_config(interval, overrides)
        log(f"== {interval} mode={args.mode} round={args.round} cfg={json.dumps(asdict(cfg), ensure_ascii=False)}")
        S = load_samples(cfg, db_path)
        log(f"  samples={S.m} span={_fmt(int(S.decision_ms[0]))}..{_fmt(int(S.decision_ms[-1]))}")
        base = baseline_quantiles(S, cfg)
        if args.mode == "cv":
            res = run_cv(S, cfg, base, log)
            res["config"] = asdict(cfg)
            res["samples"] = S.m
            res["span_utc8"] = [_fmt(int(S.decision_ms[0])), _fmt(int(S.decision_ms[-1]))]
            (out_dir / f"{interval}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            summary[interval] = {"model": res["model"], "gates": res["gates"],
                                 "best_baseline": res["best_baseline"]}
            log(f"  pass={res['gates']['pass']} cov50={res['model']['cov50']} cov80={res['model']['cov80']} "
                f"pin={res['model']['pinball_combined_bps']} best_base={res['best_baseline']} "
                f"{res['baselines'][res['best_baseline']]['pinball_combined_bps']} "
                f"w50={res['model']['w50_bps']} gates={res['gates']}")
        elif args.mode == "holdout":
            res = run_holdout(S, cfg, base)
            res["config"] = asdict(cfg)
            (out_dir / f"{interval}.holdout.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            summary[interval] = {"model": res["model"], "gates": res["gates_single_block"]}
            log(f"  holdout {json.dumps(res['model'], ensure_ascii=False)}")
        else:  # fit: final model on every sample, calibration tail kept inside fit_block
            all_pos = np.arange(S.m, dtype=np.int64)
            fit = fit_block(S, all_pos, cfg)
            extra = _round_verdict(out_dir, interval)
            directory = save_artifacts(data_root, S, cfg, fit, S.names, extra)
            log(f"  saved {directory}")
            summary[interval] = {"saved": str(directory), "c50": fit.c50, "c80": fit.c80}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0
import time as _time

def load_quantile(data_root, interval):
    import json
    from pathlib import Path
    import lightgbm as lgb
    directory = Path(data_root) / "models_quantile" / interval
    if not directory.exists():
        return None
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    names = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    boosters = []
    for a in ALPHAS:
        key = AKEY[a]
        boosters.append(lgb.Booster(model_file=str(directory / f"model_{key}.txt")))
    class _A:
        pass
    a = _A()
    a.interval = interval
    a.directory = directory
    a.boosters = boosters
    a.names = names
    a.meta = meta
    a.c50 = float(meta.get("c50", 0.0))
    a.c80 = float(meta.get("c80", 0.0))
    a.aci_gamma = float(meta.get("config", {}).get("aci_gamma", 0.0))
    @property
    def model_id(self):
        return str(self.meta.get("model_id"))
    a.model_id = a.meta.get("model_id")
    return a

def _quantile_explain(booster, names, row, top=3):
    import numpy as np
    n = len(names)
    flat = np.asarray(booster.predict(row.reshape(1, -1), pred_contrib=True), dtype=np.float64).reshape(-1)
    if flat.shape[0] != n + 1:
        return []
    seg = flat[:n]
    order = np.argsort(-np.abs(seg), kind="stable")[:top]
    return [{"name": names[i], "contribution": round(float(seg[i]), 6),
             "value": (float(row[i]) if np.isfinite(row[i]) else None)} for i in order]


if __name__ == "__main__":
    raise SystemExit(main())
