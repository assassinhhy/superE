"""Latest-bar prediction from a saved debug model. Read-only, never trades.

No-leak rules:

* Only bars whose close is at or before ``now`` are used. A forming bar, or any
  bar the collector has not closed yet, is cut off before features are built.
* Features come from ``features.build``, the same code as training. Row ``last``
  reads bars ``0..last`` only, so the prediction is for the entry at the next
  bar's open.
* Funding and metrics are read as of the decision moment, with the same delays
  as in training.

``PredictService`` keeps the artifacts and the auxiliary tables in memory for a
short time. It never writes to the database.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.timeutil import UTC8
from supere.train import altdata, artifacts, data, features
from supere.train.artifacts import DEBUG_NOTE, Artifact, ArtifactError
from supere.train.config import PRODUCT_INTERVALS, TRAINABLE_INTERVALS
from supere.train.runner import WINDOW_LOOKBACK

log = logging.getLogger("supere.train.predict")

AUX_TTL_SECONDS = 60.0  # funding and metrics re-read at most this often
DEFAULT_BAR_LIMIT = 1500  # newest bars loaded per call; EMA warm-up needs ~300


EXPLAIN_TOP = 3  # features shown per prediction
CLASS_ORDER = ("down", "flat", "up")  # booster column order: classes 0, 1, 2


@dataclass
class Latest:
    """Decision row and its probabilities, before any JSON shaping."""

    last: int
    decision_ms: int
    probs: dict[str, float]  # keys: down, flat, up
    core_ok: bool
    explain_for: str  # the class with the highest probability
    top_features: list[dict]  # [{name, contribution, value}], largest |contribution| first
    atr: float  # ATR(atr_period) at the decision bar, same as the label barrier width
    ref_close: float  # close of the decision bar; the entry reference


def closed_count(close_time: np.ndarray, now_ms: int) -> int:
    """Number of bars whose close is at or before ``now_ms``. Close times are sorted."""
    return int(np.searchsorted(close_time, now_ms, side="right"))


def direction(probs: dict[str, float], threshold: float) -> str:
    """Same rule as ``backtest.signal_dirs``: a side needs ``p >= threshold`` and beats the other side."""
    up, dn = probs["up"], probs["down"]
    if up >= threshold and up > dn:
        return "up"
    if dn >= threshold and dn > up:
        return "down"
    return "flat"


def latest_probs(art: Artifact, bars: data.Bars, funding: data.Funding, metrics_in: data.Metrics | None,
                 now_ms: int) -> Latest | str:
    """Probabilities for the newest closed bar, or a short reason string when there are none."""
    n_closed = closed_count(bars.close_time, now_ms)
    if n_closed == 0:
        return "no_bars"
    closed = bars.slice_to(n_closed)
    last = n_closed - 1

    # A gap inside the lookback means the rolling features would mix bars that are not adjacent.
    gaps = data.gap_after(closed)
    lo = max(last - WINDOW_LOOKBACK, 0)
    if np.any(gaps[lo:last]):
        return "gap"

    names, X = features.build(
        closed,
        funding if art.meta["use_funding"] else None,
        metrics_in if art.meta["use_metrics"] else None,
    )
    if names != art.names:
        raise ArtifactError(f"{art.interval} 的特征列表和当前代码不一致，需要重新训练")

    row = X[last]
    atr_now = float(features.atr(closed, int(art.meta["atr_period"]))[last])
    core_idx = [names.index(c) for c in features.CORE_FEATURES]
    if not np.all(np.isfinite(row[core_idx])):
        return "no_features"

    raw = art.booster.predict(row.reshape(1, -1))
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw.shape[0] != 3 or not np.all(np.isfinite(raw)):
        raise ArtifactError(f"{art.interval} 的模型输出不是三类概率")
    probs = {"down": float(raw[0]), "flat": float(raw[1]), "up": float(raw[2])}
    explain_for = max(CLASS_ORDER, key=lambda k: probs[k])
    top = explain_row(art.booster, names, row, CLASS_ORDER.index(explain_for))
    return Latest(last=last, decision_ms=int(bars.close_time[last]), probs=probs, core_ok=True,
                  explain_for=explain_for, top_features=top, atr=atr_now,
                  ref_close=float(closed.c[last]))


LEVEL_RULE = ("入场参考 = 决策 K 线收盘价（训练标签用的是下一根开盘，实盘会有开盘跳空偏差）；"
              "止损 = 入场 ∓ k×ATR14；止盈1 = 入场 ± k×ATR14（与训练标签的上下障碍相同，回测口径）；"
              "止盈2 = 入场 ± 2k×ATR14（未回测）。价格点位与杠杆无关。")


def trade_levels(direction_: str, ref: float, atr: float, k_atr: float) -> tuple[dict | None, str | None]:
    """Entry, stop and two targets as prices, from the same barrier rule as the labels.

    Returns ``(levels, None)`` for a long or short, and ``(None, reason)`` for flat or
    when the ATR is not usable. Stop and target 1 are the lower and upper barriers the
    labels and the backtest use, so the backtest checks these exact levels.
    """
    if direction_ not in ("up", "down"):
        return None, "观望：没有方向概率过阈值，不给点位"
    if not (np.isfinite(ref) and ref > 0 and np.isfinite(atr) and atr > 0):
        return None, "ATR 或参考价不可用，不给点位"
    sign = 1.0 if direction_ == "up" else -1.0
    width = k_atr * atr
    entry = ref
    stop = entry - sign * width
    t1 = entry + sign * width
    t2 = entry + sign * 2.0 * width
    levels = {
        "side": "long" if sign > 0 else "short",
        "ref_price": _num(ref),
        "entry": _num(entry),
        "stop": _num(stop),
        "targets": [_num(t1), _num(t2)],
        "k_atr": float(k_atr),
        "atr": _num(atr),
        "stop_pct": _num(stop / entry - 1.0),
        "target_pct": [_num(t1 / entry - 1.0), _num(t2 / entry - 1.0)],
        "backtested_targets": [True, False],
        "rule": LEVEL_RULE,
    }
    return levels, None


def explain_row(booster, names: list[str], row: np.ndarray, class_index: int,
                top: int = EXPLAIN_TOP) -> list[dict]:
    """Per-feature SHAP contributions (LightGBM ``pred_contrib``) for one class of one row.

    The contribution is on the booster's raw score for that class, not on the
    probability: positive pushes the class up, negative pushes it down. Per class the
    booster returns ``n_features + 1`` values, the last being the bias, and the
    segments sum to the raw score (checked in the tests).
    """
    n = len(names)
    flat = np.asarray(booster.predict(row.reshape(1, -1), pred_contrib=True), dtype=np.float64).reshape(-1)
    if flat.shape[0] != len(CLASS_ORDER) * (n + 1):
        raise ArtifactError("模型的 pred_contrib 输出形状和特征数对不上")
    seg = flat[class_index * (n + 1): class_index * (n + 1) + n]
    order = np.argsort(-np.abs(seg), kind="stable")[:top]
    return [
        {"name": names[i], "contribution": round(float(seg[i]), 6), "value": _num(row[i])}
        for i in order
    ]


def _num(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


NOT_TRADABLE_REASON = ("2026-11-05 训练口径迭代未达标（扣费后期望或折间稳定性不够），"
                       "不要据此下单")


def _base(interval: str) -> dict:
    # Every item says whether the model is tradable. Default: not tradable, until a
    # model whose meta.json has tradable=true (written only after the iteration passed) is served.
    return {"interval": interval, "debug": True, "debug_note": DEBUG_NOTE,
            "tradable": False, "tradable_status": "未达标",
            "tradable_reason": NOT_TRADABLE_REASON}


def _tradable_fields(meta: dict) -> dict:
    ok = meta.get("tradable") is True
    return {"tradable": ok,
            "tradable_status": "可用（已达标）" if ok else "未达标",
            "tradable_reason": meta.get("tradable_reason") or ("" if ok else NOT_TRADABLE_REASON)}


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")


def _quality_pass(meta: dict) -> bool:
    """Internal model-quality flag; never rendered as text in the UI."""
    return str(meta.get("status") or "") == "达标"


def _sanitize_forecast(pf: dict | None, meta: dict) -> dict | None:
    """Keep the old status field as a generic ok flag; never expose quality text."""
    if isinstance(pf, dict):
        pf["status"] = "ok"
        pf["quality_pass"] = _quality_pass(meta)
    return pf


CONTEXT_WINDOW = 20  # bars for the key high and low shown beside the prediction


def _context(bars: data.Bars, funding: data.Funding, last: int, decision_ms: int) -> dict:
    """Display-only numbers for the explanation text: no model input, no levels.

    Key high and low are over the last ``CONTEXT_WINDOW`` closed bars including the
    decision bar. The funding rate is the last settlement strictly before the decision.
    """
    lo = max(last - CONTEXT_WINDOW + 1, 0)
    k = int(np.searchsorted(funding.time, decision_ms, side="left")) - 1
    fund = None
    if k >= 0:
        fund = {"rate": _num(funding.rate[k]), "time_utc8": _fmt(int(funding.time[k]))}
    return {
        "ref_close": _num(bars.c[last]),
        "window_bars": int(last - lo + 1),
        "high": _num(np.max(bars.h[lo:last + 1])),
        "low": _num(np.min(bars.l[lo:last + 1])),
        "funding": fund,
    }


class PredictService:
    def __init__(self, data_root: Path, *, bar_limit: int = DEFAULT_BAR_LIMIT,
                 clock=time.time) -> None:
        self.data_root = Path(data_root)
        self.db_path = self.data_root / "superE.sqlite"
        self.bar_limit = bar_limit
        self._clock = clock
        # Funding and metrics per (symbol, market), each with the time it was read.
        self._aux: dict[tuple[str, str], tuple[float, data.Funding, data.Metrics]] = {}
        self._art_cache: dict[str, tuple[float, Artifact]] = {}

    # -- public -----------------------------------------------------------

    def predict(self, interval: str) -> dict:
        if interval not in TRAINABLE_INTERVALS:
            return {**_base(interval), "status": "bad_interval", "direction": None,
                    "message": f"不支持的周期 {interval}"}
        try:
            art = self._artifact(interval)
        except ArtifactError as exc:
            return {**_base(interval), "status": "bad_model", "direction": None, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.exception("读取 %s 的模型出错", interval)
            return {**_base(interval), "status": "error", "direction": None,
                    "message": f"{type(exc).__name__}: {exc}"}
        if art is None:
            return {**_base(interval), "status": "no_model", "direction": None,
                    "message": "这个周期还没有训练好的模型"}
        if not self.db_path.exists():
            return {**_base(interval), "status": "no_data", "direction": None,
                    "message": f"找不到数据库 {self.db_path}"}

        now_ms = int(self._clock() * 1000)
        try:
            bars, funding, metrics_in = self._load(art)
            got = latest_probs(art, bars, funding, metrics_in, now_ms)
        except ArtifactError as exc:
            return {**_base(interval), "status": "bad_model", "direction": None, "message": str(exc)}
        except (sqlite3.Error, ValueError) as exc:
            log.warning("预测 %s 读数据失败：%s", interval, exc)
            return {**_base(interval), "status": "no_data", "direction": None, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one interval must not fail the whole batch
            log.exception("预测 %s 出错", interval)
            return {**_base(interval), "status": "error", "direction": None,
                    "message": f"{type(exc).__name__}: {exc}"}

        if isinstance(got, str):
            return {**_base(interval), "status": got, "direction": None,
                    "message": _REASON[got], "model_id": art.model_id}

        age_bars = (now_ms - got.decision_ms) / INTERVAL_MS[interval]
        status = "ok" if age_bars <= 2 else "stale"
        threshold = float(art.meta["threshold"])
        dir_now = direction(got.probs, threshold) if status == "ok" else None
        if status != "ok":
            levels, levels_reason = None, "数据已过期，不给点位"
        else:
            levels, levels_reason = trade_levels(dir_now, got.ref_close, got.atr, float(art.meta["k_atr"]))
        return {
            **_base(interval),
            **_tradable_fields(art.meta),
            "status": status,
            "direction": dir_now,
            "levels": levels,
            "levels_reason": levels_reason,
            "probs": {k: round(v, 6) for k, v in got.probs.items()},
            "explain_for": got.explain_for,
            "top_features": got.top_features,
            "context": _context(bars, funding, got.last, got.decision_ms),
            "threshold": threshold,
            "asof_utc8": _fmt(got.decision_ms),
            "bar_open_utc8": _fmt(int(bars.open_time[got.last])),
            "model_id": art.model_id,
            "trained_at_utc8": art.meta["trained_at_utc8"],
            "horizon": art.meta["horizon"],
            "oos_accuracy": art.meta["oos"]["classification"].get("accuracy"),
            "message": DEBUG_NOTE if status == "ok" else "数据已过期，采集器可能没在跑",
        }

    def predict_all(self, intervals: tuple[str, ...] = PRODUCT_INTERVALS) -> dict:
        return {"debug": True, "debug_note": DEBUG_NOTE,
                "items": [self.predict(i) for i in intervals]}

    # -- internals ----------------------------------------------------------

    def _artifact(self, interval: str) -> Artifact | None:
        directory = artifacts.model_dir(self.data_root, interval)
        if not directory.exists():
            return None
        # *.tmp files are mid-save; ignore them so a save never races this stat().
        stamp = max((p.stat().st_mtime for p in directory.iterdir() if not p.name.endswith(".tmp")),
                    default=0.0)
        cached = self._art_cache.get(interval)
        if cached and cached[0] == stamp:
            return cached[1]
        art = artifacts.load(self.data_root, interval)
        if art is None:
            return None
        self._art_cache[interval] = (stamp, art)
        return art

    def _load(self, art: Artifact) -> tuple[data.Bars, data.Funding, data.Metrics | None]:
        symbol = art.meta["symbol"]
        market = art.meta["market"]
        now = self._clock()
        key = (symbol, market)
        cached = self._aux.get(key)
        if cached is None or now - cached[0] > AUX_TTL_SECONDS:
            conn = data.open_readonly(self.db_path)
            try:
                funding = data.load_funding(conn, symbol)
                metrics_in = data.load_metrics(conn, symbol)
            finally:
                conn.close()
            cached = (now, funding, metrics_in)
            self._aux[key] = cached
        _, funding, metrics_in = cached

        conn = data.open_readonly(self.db_path)
        try:
            bars = data.load_bars(conn, market, symbol, art.interval, self.bar_limit)
        finally:
            conn.close()
        return bars, funding, metrics_in


_REASON = {
    "no_bars": "库里还没有这个周期的已收盘 K 线",
    "gap": "最近的 K 线有缺口，特征不可靠",
    "no_features": "核心特征还不完整（历史不够长）",
}

import time as _t

class QuantilePredictService:
    def __init__(self, data_root, *, bar_limit=1500, clock=None, calibrator=None):
        from pathlib import Path
        self.data_root = Path(data_root)
        self.db_path = self.data_root / "superE.sqlite"
        self.bar_limit = bar_limit
        self._clock = clock or _t.time
        self._cache = {}
        self._v3_cache = {}
        # ``None`` -> build the live v6 CQR calibrator lazily; pass False to
        # disable it explicitly (tests / offline diagnostics).
        self._calibrator = calibrator
        self._calibrator_disabled = calibrator is False

    # -- v6 live calibration ------------------------------------------------

    def _get_calibrator(self):
        if self._calibrator_disabled:
            return None
        if self._calibrator is None:
            from supere.train.calib_live import LiveCalibrator
            self._calibrator = LiveCalibrator(self.data_root, clock=self._clock)
        return self._calibrator

    def calibration_status(self) -> dict:
        cal = self._calibrator
        if cal is None or cal is False:
            return {"enabled": not self._calibrator_disabled, "loaded": False,
                    "intervals": {}}
        try:
            out = cal.status()
            out["loaded"] = True
            return out
        except Exception as exc:  # noqa: BLE001 - health must stay simple
            return {"enabled": True, "loaded": False, "error": f"{type(exc).__name__}: {exc}"}

    def _artifact(self, interval):
        import supere.train.quantile as q
        if interval in self._cache:
            return self._cache[interval]
        art = q.load_quantile(self.data_root, interval)
        self._cache[interval] = art
        return art

    def _v3_artifact(self, interval):
        import supere.train.quantile_v3 as v3
        if interval not in self._v3_cache:
            self._v3_cache[interval] = v3.load_v3(self.data_root, interval)
        return self._v3_cache[interval]

    def predict(self, interval):
        import math
        import numpy as np
        from supere.intervals import INTERVAL_MS
        from supere.timeutil import UTC8
        import supere.train.quantile as q
        import supere.train.data as data
        PRODUCT = q.PRODUCT_INTERVALS
        if interval not in PRODUCT:
            return {"interval": interval, "debug": True, "status": "bad_interval",
                    "message": f"不支持的周期 {interval}", "price_forecast": None}
        # Version 3 artifacts decompose scale and shape.  They are served by the
        # same contract; v2 artifacts keep using the code below.
        try:
            from supere.train import quantile_v3 as v3
            v3_art = self._v3_artifact(interval)
            if v3_art is not None:
                now_ms = int(self._clock() * 1000)
                res = v3.predict_latest(self.data_root, interval, v3_art, now_ms)
                pf = _sanitize_forecast(
                    res.get("price_forecast"), getattr(v3_art, "meta", {}) or {}
                )
                # v6 order: v5 raw -> CQR / none -> (news overlay happens in
                # serve.py) -> UI.  Calibration never raises into the endpoint.
                if pf is not None:
                    try:
                        calibrator = self._get_calibrator()
                        if calibrator is not None:
                            pf = calibrator.decorate(interval, v3_art, pf, now_ms)
                    except Exception:  # noqa: BLE001 - keep the raw forecast
                        log.exception("v6 校准接入 %s 失败", interval)
                res["price_forecast"] = pf
                res["debug_note"] = "价格区间预测，样本外弱，勿实盘"
                return res
        except Exception as exc:  # noqa: BLE001 - fall back to an explicit error item
            log.exception("v3 预测 %s 出错", interval)
            return {"interval": interval, "debug": True, "status": "error",
                    "message": f"{type(exc).__name__}: {exc}", "price_forecast": None}
        art = self._artifact(interval)
        if art is None:
            return {"interval": interval, "debug": True, "status": "no_model",
                    "message": "这个周期还没有训练好的模型", "price_forecast": None}
        if not self.db_path.exists():
            return {"interval": interval, "debug": True, "status": "no_data",
                    "message": f"找不到数据库 {self.db_path}", "price_forecast": None}
        cfg = art.meta["config"]
        groups = q.feature_groups(q.default_config(interval, cfg))
        now_ms = int(self._clock() * 1000)
        try:
            conn = data.open_readonly(self.db_path)
            try:
                bars = data.load_bars(conn, cfg["market"], cfg["symbol"], interval, self.bar_limit)
                funding = data.load_funding(conn, cfg["symbol"])
                metrics_in = data.load_metrics(conn, cfg["symbol"])
                mark = data.load_bars(conn, "um_mark", cfg["symbol"], interval, self.bar_limit)
                index = data.load_bars(conn, "um_index", cfg["symbol"], interval, self.bar_limit)
                premium = data.load_bars(conn, "um_premium", cfg["symbol"], interval, self.bar_limit)
                btc_bars = None
                if "btc" in groups:
                    btc_bars = data.load_bars(conn, cfg["market"], cfg.get("btc_symbol", "BTCUSDT"),
                                              interval, self.bar_limit)
            finally:
                conn.close()
            if btc_bars is not None:
                # Only BTC bars already closed at ``now``; alignment is by open time.
                n_btc = int(np.searchsorted(btc_bars.close_time, now_ms, side="right"))
                btc_bars = btc_bars.slice_to(n_btc)
            htf_interval = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d"}.get(interval)
            higher = None
            if htf_interval:
                conn = data.open_readonly(self.db_path)
                try:
                    higher = data.load_bars(conn, "um", cfg["symbol"], htf_interval, None)
                finally:
                    conn.close()
            alt = altdata.load_alt(self.data_root, groups, btc=btc_bars)
            n_closed = int(np.searchsorted(bars.close_time, now_ms, side="right"))
            if n_closed == 0:
                return {"interval": interval, "debug": True, "status": "no_bars",
                        "message": "库里还没有这个周期的已收盘 K 线", "price_forecast": None}
            closed = bars.slice_to(n_closed)
            last = n_closed - 1
            names, X, sig, r1 = q.build_features(
                closed, funding if cfg.get("use_funding") else None,
                metrics_in if cfg.get("use_metrics") else None, q.default_config(interval, cfg),
                mark, index, premium, higher, alt=alt, btc_bars=btc_bars)
            if names != art.names:
                raise q.QuantileError(f"{interval} 的特征列表和当前代码不一致，需要重新训练")
            row = X[last]
            ci = q.core_index(names)
            if not np.all(np.isfinite(row[ci])):
                return {"interval": interval, "debug": True, "status": "no_features",
                        "message": "核心特征还不完整（历史不够长）", "price_forecast": None}
            Q = q.predict_raw(art.boosters, row.reshape(1, -1))
            if cfg.get("target") == "vol":
                Q = Q * float(sig[last])
            Q = q.sort_rows(Q)
            ref_close = float(closed.c[last])
            decision_ms = int(bars.close_time[last])
            top = q._quantile_explain(art.boosters[2], art.names, row, 3) if hasattr(q, "_quantile_explain") else []
            CN = getattr(q, "CN_LABEL", {})
            explain = "主要特征：" + "、".join(CN.get(f["name"], f["name"]) for f in top)
            pf = {
                "interval": interval,
                "asof_utc8": datetime.fromtimestamp(decision_ms / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M"),
                "horizon_end_utc8": datetime.fromtimestamp((decision_ms + INTERVAL_MS[interval]) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M"),
                "horizon_end_time": datetime.fromtimestamp((decision_ms + INTERVAL_MS[interval]) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M"),
                "ref_close": ref_close,
                "median": float(ref_close * math.exp(float(Q[0, 2]))),
                "p10": float(ref_close * math.exp(float(Q[0, 0]))),
                "p25": float(ref_close * math.exp(float(Q[0, 1]))),
                "p75": float(ref_close * math.exp(float(Q[0, 3]))),
                "p90": float(ref_close * math.exp(float(Q[0, 4]))),
                "width50_bps": round(float((Q[0, 3] - Q[0, 1]) * 1e4), 4),
                "width80_bps": round(float((Q[0, 4] - Q[0, 0]) * 1e4), 4),
                "coverage_ok": True,
                "status": "ok",
                "quality_pass": _quality_pass(art.meta),
                "explain": explain,
                "top_features": top,
            }
            age_bars = (now_ms - decision_ms) / INTERVAL_MS[interval]
            status = "ok" if age_bars <= 2 else "stale"
            if status != "ok":
                pf = None
            return {
                "interval": interval,
                "debug": True,
                "status": status,
                "price_forecast": pf,
                "message": ("调试预测，样本外弱，勿实盘" if status == "ok" else "数据已过期，采集器可能没在跑"),
                "model_id": art.model_id,
                "trained_at_utc8": art.meta.get("trained_at_utc8"),
            }
        except Exception as exc:
            return {"interval": interval, "debug": True, "status": "error",
                    "message": f"{type(exc).__name__}: {exc}", "price_forecast": None}

    def predict_all(self, intervals=None):
        if intervals is None:
            intervals = q.PRODUCT_INTERVALS
        return {"debug": True, "debug_note": "调试预测，样本外弱，勿实盘",
                "items": [self.predict(i) for i in intervals]}

# The product route serves the quantile service. The earlier classification
# debug service is kept under its own name for its tests and for callers that
# still expect ``direction`` + ``levels``.
ClassificationPredictService = PredictService
PredictService = QuantilePredictService
