"""Deployment-isomorphic offline calibration replay (stage 1).

Three pre-registered candidates on frozen raw predictions:

``none``
    the frozen v5 quantiles, unchanged.
``cqr``
    rolling two-sided *normalized* CQR.  Per tail the score is
    ``(q_lo - y) / max(sigma, eps)`` or ``(y - q_hi) / max(sigma, eps)`` and the
    bound is shifted by the finite-sample ``ceil((n+1)p)`` order statistic of
    the past scores (75% for the 50% band, 90% for the 80% band).  Negative
    corrections are allowed (narrowing).
``cqr_aci``
    the same rolling CQR, but the order-statistic level is driven by a per-tail
    ACI state: ``alpha_next = alpha + gamma * (alpha_target - miss)``.  The
    level is ``1 - alpha``; alpha is clipped to [1e-3, 1-1e-3] (a clipped /
    finite-width variant, so the paper's unrestricted-drift guarantee is not
    claimed).

Deployment properties enforced here:

* only predictions whose ``horizon_end + label_delay <= now`` feed back;
* one prediction id is consumed once; duplicate ids/requests do not learn;
* the median ``q50`` is never moved and row crossing falls back to ``none``;
* state is keyed by (symbol, horizon, model_hash, calib_version) and is reset
  at every fold/model change in the primary ``fold`` scope;
* the rolling window and the ACI state are updated event by event, not in
  batches, and are snapshot/restorable.

The module is pure numpy/pandas; the CLI reads the OOF parquet written by
:mod:`supere.train.oof_v6` and writes replay parquet + metrics JSON to the v6
output tree.  No service, database or model directory is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import v6proto

STATE_VERSION = "v6-calib-1"
ALPHA_CLIP = (1e-3, 1.0 - 1e-3)
DEFAULT_WINDOW_DAYS = v6proto.CALIB_WINDOW_DAYS
MIN_SAMPLES = v6proto.MIN_CALIB_SAMPLES


# --------------------------------------------------------------------- stats

def order_stats(values: np.ndarray, levels: list[float]) -> tuple[list[float], list[str]]:
    """Finite-sample conformal order statistics ``ceil((n+1)p)``.

    Returns the values and one flag per level: ``""`` normal, ``"low"`` clamped
    to the minimum (k < 1), ``"high"`` clamped to the maximum (k > n).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return [0.0] * len(levels), ["empty"] * len(levels)
    ks: list[int] = []
    flags: list[str] = []
    for level in levels:
        level = float(min(max(level, 0.0), 1.0))
        k = int(math.ceil((n + 1) * level))
        if k < 1:
            k, flag = 1, "low"
        elif k > n:
            k, flag = n, "high"
        else:
            flag = ""
        ks.append(k)
        flags.append(flag)
    uniq = sorted(set(ks))
    part = np.partition(arr, [k - 1 for k in uniq])
    return [float(part[k - 1]) for k in ks], flags


def _clip_alpha(a: float) -> tuple[float, bool]:
    lo, hi = ALPHA_CLIP
    b = min(max(a, lo), hi)
    return b, b != a


@dataclass
class _Band:
    decision: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray
    head: int = 0
    tail: int = 0

    def append(self, decision_ms: int, s_lo: float, s_hi: float) -> None:
        self.decision[self.tail] = decision_ms
        self.s_lo[self.tail] = s_lo
        self.s_hi[self.tail] = s_hi
        self.tail += 1

    def expire(self, cutoff_ms: int) -> None:
        while self.head < self.tail and self.decision[self.head] < cutoff_ms:
            self.head += 1

    @property
    def count(self) -> int:
        return self.tail - self.head

    def lo_slice(self) -> np.ndarray:
        return self.s_lo[self.head:self.tail]

    def hi_slice(self) -> np.ndarray:
        return self.s_hi[self.head:self.tail]


@dataclass
class EngineConfig:
    candidate: str
    window_days: float
    min_samples: int = MIN_SAMPLES
    gamma: float = 0.005
    delay_ms: int = 0
    state_scope: str = "fold"
    horizon_ms: int = 0
    symbol: str = "ETHUSDT"
    model_hash: str = ""
    calib_version: str = STATE_VERSION


class CalibrationEngine:
    """Event-driven rolling CQR (+ optional ACI) state machine.

    ``predict`` must be called in decision-time order.  ``set_label`` may be
    called at any time; the label is *read* only when the pending prediction
    matures (``horizon_end + delay <= now``).  ``advance`` is idempotent for an
    unchanged clock and duplicate prediction ids are ignored.
    """

    def __init__(self, cfg: EngineConfig, capacity: int, eps: float):
        self.cfg = cfg
        self.eps = float(max(eps, 1e-12))
        cap = max(int(capacity) + 2, 16)
        self.band50 = _Band(np.empty(cap, np.int64), np.empty(cap), np.empty(cap))
        self.band80 = _Band(np.empty(cap, np.int64), np.empty(cap), np.empty(cap))
        self.pending: deque = deque()
        self.labels: dict[str, float] = {}
        self.seen_predict: set[str] = set()
        self.seen_observe: set[str] = set()
        self.missing_labels: list[str] = []
        self.alpha = {50: [v6proto.TAIL_TARGET[50], v6proto.TAIL_TARGET[50]],
                      80: [v6proto.TAIL_TARGET[80], v6proto.TAIL_TARGET[80]]}
        self.alpha_clips = 0
        self.rows = 0
        self.fallbacks = {"cold": 0, "crossing": 0, "empty": 0, "duplicate": 0}
        self.quantile_clamp = 0
        self.window_days_eff = float(cfg.window_days)

    # ---------------------------------------------------------------- clock
    def advance(self, now_ms: int) -> None:
        now_ms = int(now_ms)
        window_ms = int(round(self.window_days_eff * 86_400_000))
        cutoff = now_ms - window_ms
        matured = []
        while self.pending and self.pending[0][0] <= now_ms:
            matured.append(self.pending.popleft())
        for item in matured:
            self._consume(item)
        self.band50.expire(cutoff)
        self.band80.expire(cutoff)

    def _consume(self, item) -> None:
        (avail, pred_id, decision_ms, raw_q, sigma,
         lo50, hi50, lo80, hi80, a_lo50, a_hi50, a_lo80, a_hi80) = item
        if pred_id in self.seen_observe:
            return
        y = self.labels.pop(pred_id, None)
        if y is None:
            self.missing_labels.append(pred_id)
            return
        self.seen_observe.add(pred_id)
        scale = max(float(sigma), self.eps)
        q10, q25, _q50, q75, q90 = (float(x) for x in raw_q)
        self.band50.append(decision_ms, (q25 - y) / scale, (y - q75) / scale)
        self.band80.append(decision_ms, (q10 - y) / scale, (y - q90) / scale)
        if self.cfg.candidate == "cqr_aci":
            for band, lo, hi, a_lo, a_hi, target in (
                    (50, lo50, hi50, a_lo50, a_hi50, v6proto.TAIL_TARGET[50]),
                    (80, lo80, hi80, a_lo80, a_hi80, v6proto.TAIL_TARGET[80])):
                miss_lo = 1.0 if y < lo else 0.0
                miss_hi = 1.0 if y > hi else 0.0
                a1, c1 = _clip_alpha(a_lo + self.cfg.gamma * (target - miss_lo))
                a2, c2 = _clip_alpha(a_hi + self.cfg.gamma * (target - miss_hi))
                self.alpha[band] = [a1, a2]
                self.alpha_clips += int(c1) + int(c2)

    # ------------------------------------------------------------- predict
    def predict(self, decision_ms: int, raw_q, sigma: float, pred_id: str,
                horizon_end_ms: int):
        self.advance(int(decision_ms))
        q = np.sort(np.asarray(raw_q, dtype=np.float64).copy())
        self.rows += 1
        if pred_id in self.seen_predict:
            self.fallbacks["duplicate"] += 1
            return q, {"active": False, "reason": "duplicate", "n_calib": 0,
                       "alpha": None, "clamped": 0}
        self.seen_predict.add(pred_id)
        n50 = self.band50.count
        if n50 < self.cfg.min_samples:
            out = q
            reason = "cold"
            alpha_used = None
            clamped = 0
            self.fallbacks["cold"] += 1
        else:
            out, reason, alpha_used, clamped = self._calibrate(q, float(sigma))
        avail = int(horizon_end_ms) + int(self.cfg.delay_ms)
        a = self.alpha
        self.pending.append((avail, pred_id, int(decision_ms), q, float(sigma),
                             float(out[1]), float(out[3]), float(out[0]), float(out[4]),
                             float(a[50][0]), float(a[50][1]),
                             float(a[80][0]), float(a[80][1])))
        diag = {"active": reason == "", "reason": reason or "ok", "n_calib": int(n50),
                "alpha": alpha_used, "clamped": int(clamped)}
        return out, diag

    def _calibrate(self, q: np.ndarray, sigma: float):
        scale = max(float(sigma), self.eps)
        if self.cfg.candidate == "cqr_aci":
            levels50 = [1.0 - self.alpha[50][0], 1.0 - self.alpha[50][1]]
            levels80 = [1.0 - self.alpha[80][0], 1.0 - self.alpha[80][1]]
        else:
            levels50 = [0.75, 0.75]
            levels80 = [0.90, 0.90]
        s50_lo, f1 = order_stats(self.band50.lo_slice(), [levels50[0]])
        s50_hi, f2 = order_stats(self.band50.hi_slice(), [levels50[1]])
        s80_lo, f3 = order_stats(self.band80.lo_slice(), [levels80[0]])
        s80_hi, f4 = order_stats(self.band80.hi_slice(), [levels80[1]])
        clamped = sum(1 for f in (f1[0], f2[0], f3[0], f4[0]) if f in ("low", "high"))
        self.quantile_clamp += clamped
        out = q.copy()
        out[1] = q[1] - s50_lo[0] * scale
        out[3] = q[3] + s50_hi[0] * scale
        out[0] = q[0] - s80_lo[0] * scale
        out[4] = q[4] + s80_hi[0] * scale
        # q50 is deliberately untouched.  Nested crossing falls back to raw.
        if not (out[0] <= out[1] <= q[2] <= out[3] <= out[4]):
            self.fallbacks["crossing"] += 1
            out = q.copy()
            reason = "crossing"
        else:
            reason = ""
        alpha_used = ({"50": list(self.alpha[50]), "80": list(self.alpha[80])}
                      if self.cfg.candidate == "cqr_aci" else None)
        return out, reason, alpha_used, clamped

    # -------------------------------------------------------------- labels
    def set_label(self, pred_id: str, y: float) -> None:
        if pred_id in self.labels or pred_id in self.seen_observe:
            return
        self.labels[pred_id] = float(y)

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict:
        return {
            "state_version": STATE_VERSION,
            "candidate": self.cfg.candidate,
            "window_days": self.window_days_eff,
            "gamma": self.cfg.gamma,
            "delay_ms": self.cfg.delay_ms,
            "alpha": {str(k): list(v) for k, v in self.alpha.items()},
            "alpha_clips": self.alpha_clips,
            "rows": self.rows,
            "bands": {
                "50": self._band_snapshot(self.band50),
                "80": self._band_snapshot(self.band80),
            },
            "pending": [list(x[:3]) + [np.asarray(x[3]).tolist()] + list(x[4:])
                        for x in self.pending],
            "labels": dict(self.labels),
            "seen_predict": sorted(self.seen_predict),
            "seen_observe": sorted(self.seen_observe),
            "missing_labels": list(self.missing_labels),
            "fallbacks": dict(self.fallbacks),
            "quantile_clamp": self.quantile_clamp,
        }

    @staticmethod
    def _band_snapshot(b: _Band) -> dict:
        return {"decision": b.decision[b.head:b.tail].tolist(),
                "s_lo": b.s_lo[b.head:b.tail].tolist(),
                "s_hi": b.s_hi[b.head:b.tail].tolist()}

    @classmethod
    def restore(cls, cfg: EngineConfig, capacity: int, eps: float,
                snap: dict) -> "CalibrationEngine":
        if snap.get("state_version") != STATE_VERSION:
            raise ValueError("state version mismatch")
        eng = cls(cfg, capacity, eps)
        eng.window_days_eff = float(snap["window_days"])
        eng.alpha = {int(k): list(v) for k, v in snap["alpha"].items()}
        eng.alpha_clips = int(snap["alpha_clips"])
        eng.rows = int(snap["rows"])
        for band, key in ((eng.band50, "50"), (eng.band80, "80")):
            data = snap["bands"][key]
            band.head, band.tail = 0, len(data["decision"])
            band.decision[:band.tail] = np.asarray(data["decision"], dtype=np.int64)
            band.s_lo[:band.tail] = np.asarray(data["s_lo"], dtype=np.float64)
            band.s_hi[:band.tail] = np.asarray(data["s_hi"], dtype=np.float64)
        eng.pending = deque(tuple(x) for x in snap["pending"])
        eng.labels = {str(k): float(v) for k, v in snap["labels"].items()}
        eng.seen_predict = set(snap["seen_predict"])
        eng.seen_observe = set(snap["seen_observe"])
        eng.missing_labels = list(snap["missing_labels"])
        eng.fallbacks = dict(snap["fallbacks"])
        eng.quantile_clamp = int(snap["quantile_clamp"])
        return eng


class ReadOnlyView:
    """Pure read view for an HTTP-style contract: a GET never advances learning.

    A real server would call this from its request handler; the prediction is
    produced by the scheduler, not by the request.  Tests assert that repeated
    reads leave the engine fingerprint unchanged.
    """

    def __init__(self, engine: CalibrationEngine, symbol: str = "ETHUSDT",
                 horizon: str = "1h"):
        self._engine = engine
        self.symbol = symbol
        self.horizon = horizon

    def get(self) -> dict:
        e = self._engine
        return {
            "state_version": STATE_VERSION,
            "symbol": self.symbol,
            "horizon": self.horizon,
            "model_hash": e.cfg.model_hash,
            "calib_version": e.cfg.calib_version,
            "candidate": e.cfg.candidate,
            "window_days": e.window_days_eff,
            "n_active_50": e.band50.count,
            "n_active_80": e.band80.count,
            "pending": len(e.pending),
            "alpha": {str(k): list(v) for k, v in e.alpha.items()},
            "rows_scored": e.rows,
            "fallbacks": dict(e.fallbacks),
            "read_only": True,
        }

    def fingerprint(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(self.get(), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ replay

def replay_interval(df, interval: str, candidate: str, *, window_days: float,
                    gamma: float = 0.005, delay_ms: int = 0,
                    state_scope: str = "fold", min_samples: int = MIN_SAMPLES,
                    log_fn=None) -> tuple[object, dict]:
    """Replay one candidate over one interval's OOF rows."""
    t0 = time.time()
    n = len(df)
    qkeys = ("10", "25", "50", "75", "90")
    cols = {f"q{k}_{candidate}": np.empty(n) for k in qkeys}
    active = np.zeros(n, dtype=bool)
    reason = np.array(["none"] * n, dtype=object)
    ncal = np.zeros(n, dtype=np.int32)
    alpha_lo50 = np.full(n, np.nan)
    alpha_hi50 = np.full(n, np.nan)
    alpha_lo80 = np.full(n, np.nan)
    alpha_hi80 = np.full(n, np.nan)

    if candidate == "none":
        arr = df[["q10", "q25", "q50", "q75", "q90"]].to_numpy(np.float64)
        for j, k in enumerate(qkeys):
            cols[f"q{k}_{candidate}"] = arr[:, j]
        active[:] = True
        reason[:] = "none"
        summary = {"candidate": candidate, "window_days": None, "gamma": None,
                   "delay_ms": delay_ms, "state_scope": "n/a",
                   "fallbacks": {"cold": 0, "crossing": 0, "empty": 0, "duplicate": 0},
                   "alpha_clips": 0, "quantile_clamp": 0, "max_alpha_drift": 0.0,
                   "final_alpha": None, "missing_labels": 0}
    else:
        decision = df["decision_ms"].to_numpy(np.int64)
        y = df["y_logret"].to_numpy(np.float64)
        raw = df[["q10", "q25", "q50", "q75", "q90"]].to_numpy(np.float64)
        sigma = df["sigma_model"].to_numpy(np.float64)
        horizon_end = df["horizon_end_ms"].to_numpy(np.int64)
        fold = df["fold"].to_numpy(np.int64)
        sample_pos = df["sample_pos"].to_numpy(np.int64)
        model_hash = df["model_hash"].to_numpy(object)
        eps_by_fold = {}
        for f in np.unique(fold):
            m = fold == f
            if "eps_train" in df.columns:
                vals = df.loc[m, "eps_train"].to_numpy(np.float64)
                eps_by_fold[int(f)] = max(1e-12, float(vals[0]))
            else:  # synthetic callers without a training-eps column
                eps_by_fold[int(f)] = max(1e-12, float(np.quantile(sigma[m], 0.001)))
        cfg = EngineConfig(candidate=candidate, window_days=window_days, gamma=gamma,
                           delay_ms=delay_ms, state_scope=state_scope,
                           min_samples=min_samples, horizon_ms=INTERVAL_MS[interval])
        eng = CalibrationEngine(cfg, n, eps_by_fold[int(fold[0])])
        cur_fold = int(fold[0])
        for i in range(n):
            if state_scope == "fold" and int(fold[i]) != cur_fold:
                cur_fold = int(fold[i])
                eng = CalibrationEngine(
                    EngineConfig(candidate=candidate, window_days=window_days,
                                 gamma=gamma, delay_ms=delay_ms,
                                 state_scope=state_scope, min_samples=min_samples,
                                 horizon_ms=INTERVAL_MS[interval],
                                 model_hash=str(model_hash[i])),
                    n, eps_by_fold[cur_fold])
            pred_id = f"{interval}:{cur_fold}:{int(sample_pos[i])}"
            eng.set_label(pred_id, y[i])
            out, diag = eng.predict(int(decision[i]), raw[i], float(sigma[i]),
                                    pred_id, int(horizon_end[i]))
            for j, k in enumerate(qkeys):
                cols[f"q{k}_{candidate}"][i] = out[j]
            active[i] = diag["active"]
            reason[i] = diag["reason"]
            ncal[i] = diag["n_calib"]
            if diag["alpha"]:
                alpha_lo50[i] = diag["alpha"]["50"][0]
                alpha_hi50[i] = diag["alpha"]["50"][1]
                alpha_lo80[i] = diag["alpha"]["80"][0]
                alpha_hi80[i] = diag["alpha"]["80"][1]
        drift = 0.0
        if candidate == "cqr_aci":
            for band, target in v6proto.TAIL_TARGET.items():
                drift = max(drift, abs(eng.alpha[band][0] - target),
                            abs(eng.alpha[band][1] - target))
        summary = {"candidate": candidate, "window_days": window_days, "gamma": gamma,
                   "delay_ms": delay_ms, "state_scope": state_scope,
                   "fallbacks": dict(eng.fallbacks), "alpha_clips": int(eng.alpha_clips),
                   "quantile_clamp": int(eng.quantile_clamp),
                   "max_alpha_drift": round(float(drift), 6),
                   "final_alpha": {str(k): list(v) for k, v in eng.alpha.items()},
                   "missing_labels": len(eng.missing_labels)}

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    out[f"active_{candidate}"] = active
    out[f"reason_{candidate}"] = reason
    out[f"ncal_{candidate}"] = ncal
    if candidate == "cqr_aci":
        out["alpha_lo50"] = alpha_lo50
        out["alpha_hi50"] = alpha_hi50
        out["alpha_lo80"] = alpha_lo80
        out["alpha_hi80"] = alpha_hi80
    summary["elapsed_s"] = round(time.time() - t0, 2)
    if log_fn:
        log_fn(f"[{interval}] {candidate} window={window_days}d gamma={gamma} "
               f"scope={state_scope} active={active.mean():.3f} "
               f"fallbacks={summary['fallbacks']} {summary['elapsed_s']}s")
    return out, summary


# ----------------------------------------------------------------- metrics

def _pinball(y_bps: np.ndarray, q_bps: np.ndarray, tau: float) -> np.ndarray:
    return (tau - (y_bps < q_bps).astype(np.float64)) * (y_bps - q_bps)


def metrics(y_log: np.ndarray, Q_log: np.ndarray, ref_close: np.ndarray,
            sigma: np.ndarray, rv_future: np.ndarray | None = None) -> dict:
    y = np.asarray(y_log, dtype=np.float64) * 1e4
    Ql = np.asarray(Q_log, dtype=np.float64)
    Q = Ql * 1e4
    taus = (0.1, 0.25, 0.5, 0.75, 0.9)
    pin = {f"p{int(t*100)}": float(np.mean(_pinball(y, Q[:, j], t)))
           for j, t in enumerate(taus)}
    inside50 = (y >= Q[:, 1]) & (y <= Q[:, 3])
    inside80 = (y >= Q[:, 0]) & (y <= Q[:, 4])
    left50 = float(np.mean(y < Q[:, 1]))
    right50 = float(np.mean(y > Q[:, 3]))
    left80 = float(np.mean(y < Q[:, 0]))
    right80 = float(np.mean(y > Q[:, 4]))

    def _is(alpha: float, lo: int, hi: int) -> float:
        l, u = Q[:, lo], Q[:, hi]
        val = (u - l) + (2.0 / alpha) * np.where(y < l, l - y, 0.0) \
            + (2.0 / alpha) * np.where(y > u, y - u, 0.0)
        return float(np.mean(val))

    price = np.asarray(ref_close, dtype=np.float64)
    w50_price = price * (np.exp(Ql[:, 3]) - np.exp(Ql[:, 1]))
    w80_price = price * (np.exp(Ql[:, 4]) - np.exp(Ql[:, 0]))
    out = {
        "n": int(y.shape[0]),
        "pinball_bps": {k: round(v, 4) for k, v in pin.items()},
        "pinball_combined_bps": round(float(np.mean(list(pin.values()))), 4),
        "pinball_equal_mean_v6_bps": round(float(np.mean(list(pin.values()))), 4),
        "cov50": round(float(inside50.mean()), 5),
        "cov80": round(float(inside80.mean()), 5),
        "left50": round(left50, 5), "right50": round(right50, 5),
        "left80": round(left80, 5), "right80": round(right80, 5),
        "w50_bps": round(float(np.mean(Q[:, 3] - Q[:, 1])), 4),
        "w80_bps": round(float(np.mean(Q[:, 4] - Q[:, 0])), 4),
        "w50_price": round(float(np.mean(w50_price)), 6),
        "w80_price": round(float(np.mean(w80_price)), 6),
        "IS50_bps": round(_is(0.5, 1, 3), 4),
        "IS80_bps": round(_is(0.2, 0, 4), 4),
        "q50_mae_bps": round(float(np.mean(np.abs(y - Q[:, 2]))), 4),
        "q50_mdae_bps": round(float(np.median(np.abs(y - Q[:, 2]))), 4),
    }
    if rv_future is not None:
        rv = np.asarray(rv_future, dtype=np.float64)
        var = np.square(np.asarray(sigma, dtype=np.float64))
        ok = np.isfinite(rv) & (rv > 0) & np.isfinite(var) & (var > 0)
        if ok.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                ql = np.log(var[ok]) + rv[ok] / var[ok]
            out["qlike"] = round(float(np.mean(ql)), 6)
            out["qlike_n"] = int(ok.sum())
    return out


def day_index(decision_ms: np.ndarray) -> np.ndarray:
    return (np.asarray(decision_ms, dtype=np.int64) + 8 * 3_600_000) // 86_400_000


def rolling_coverage_error(df, q_cols: tuple[str, str], window_days: float,
                           nominal: float = 0.5, min_n: int = 200,
                           mask: np.ndarray | None = None) -> dict:
    """Causal trailing realised coverage error (evaluation only)."""
    decision = df["decision_ms"].to_numpy(np.int64)
    y = df["y_logret"].to_numpy(np.float64)
    q = df[list(q_cols)].to_numpy(np.float64)
    inside = ((y >= q[:, 0]) & (y <= q[:, 1])).astype(np.float64)
    window_ms = int(round(window_days * 86_400_000))
    left = 0
    csum = 0.0
    errs = []
    for i in range(len(df)):
        csum += inside[i]
        while decision[left] < decision[i] - window_ms:
            csum -= inside[left]
            left += 1
        cnt = i - left + 1
        if cnt >= min_n and (mask is None or mask[i]):
            errs.append(abs(csum / cnt - nominal))
    if not errs:
        return {"n": 0, "mean_abs": None, "p90_abs": None, "window_days": window_days}
    e = np.asarray(errs)
    return {"n": int(e.size), "mean_abs": round(float(e.mean()), 5),
            "p90_abs": round(float(np.quantile(e, 0.9)), 5), "window_days": window_days}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def cells(df, q50_cols: tuple[str, str], q80_cols: tuple[str, str],
          macro: dict | None = None, mask: np.ndarray | None = None) -> list[dict]:
    """Pre-registered conditional coverage audit (no model is fitted)."""
    if mask is not None:
        df = df[mask].reset_index(drop=True)
    decision = df["decision_ms"].to_numpy(np.int64)
    y = df["y_logret"].to_numpy(np.float64)
    q50 = df[list(q50_cols)].to_numpy(np.float64)
    q80 = df[list(q80_cols)].to_numpy(np.float64)
    inside50 = (y >= q50[:, 0]) & (y <= q50[:, 1])
    inside80 = (y >= q80[:, 0]) & (y <= q80[:, 1])
    day = day_index(decision)
    hour = ((decision + 8 * 3_600_000) // 3_600_000) % 24
    weekday = (day + 3) % 7  # 1970-01-01 was a Thursday; 0=Mon
    session = np.where(hour < 8, "00-08", np.where(hour < 16, "08-16", "16-24"))
    weekend = weekday >= 5

    rv = df["rv_past"].to_numpy(np.float64)
    regime = np.array(["na"] * len(df), dtype=object)
    for d in np.unique(day):
        start = int(d) * 86_400_000 - 8 * 3_600_000
        prior = np.nonzero(decision < start)[0]
        if prior.size < 200:
            continue
        keep = prior[decision[prior] >= start - 90 * 86_400_000]
        vals = rv[keep]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 200:
            continue
        cuts = np.quantile(vals, [1 / 3, 2 / 3])
        m = day == d
        r = rv[m]
        regime[m] = np.where(~np.isfinite(r) | (r <= 0), "na",
                             np.where(r <= cuts[0], "low",
                                      np.where(r <= cuts[1], "mid", "high")))

    event = np.zeros(len(df), dtype=bool)
    nevents = 0
    if macro is not None and macro.get("times") is not None:
        times = np.asarray(macro["times"], dtype=np.int64)
        end = df["horizon_end_ms"].to_numpy(np.int64)
        lo = np.searchsorted(times, decision, side="left")
        hi = np.searchsorted(times, end, side="right")
        event = hi > lo
        nevents = int(len(times))

    groups = {
        "all": np.ones(len(df), dtype=bool),
        "rv_low": regime == "low", "rv_mid": regime == "mid", "rv_high": regime == "high",
        "weekday": ~weekend, "weekend": weekend,
        "session_00-08": session == "00-08", "session_08-16": session == "08-16",
        "session_16-24": session == "16-24",
        "macro_event": event, "macro_non_event": ~event,
    }
    out = []
    for name, m in groups.items():
        n = int(m.sum())
        if n == 0:
            continue
        days = int(np.unique(day[m]).size)
        k50 = int(inside50[m].sum())
        k80 = int(inside80[m].sum())
        lo50, hi50 = wilson(k50, n)
        lo80, hi80 = wilson(k80, n)
        cov50 = k50 / n
        cov80 = k80 / n
        out.append({
            "cell": name, "n": n, "days": days,
            "cov50": round(cov50, 5), "cov50_ci95": [round(lo50, 5), round(hi50, 5)],
            "cov80": round(cov80, 5), "cov80_ci95": [round(lo80, 5), round(hi80, 5)],
            "dev50_pp": round(100 * (cov50 - 0.5), 3),
            "dev80_pp": round(100 * (cov80 - 0.8), 3),
            "left50": round(float(np.mean(y[m] < q50[m, 0])), 5),
            "right50": round(float(np.mean(y[m] > q50[m, 1])), 5),
            "diagnosable": bool(n >= 200 and days >= 30),
        })
    if macro is not None:
        out.append({"cell": "_macro_meta", "calendar_events": nevents})
    return out


def worst_valid_deviation(cell_rows: list[dict], key: str = "dev50_pp") -> float | None:
    vals = [abs(r[key]) for r in cell_rows
            if r.get("diagnosable") and r.get("cell") not in ("all", "_macro_meta")]
    return max(vals) if vals else None


# ------------------------------------------------------------- bootstrap

def _resample_mean_ci(block_day: np.ndarray, sums: np.ndarray, counts: np.ndarray,
                      block_days: int, B: int, seed: int) -> dict:
    """Paired day(or block)-level bootstrap of a ratio-of-sums mean."""
    days = np.unique(block_day)
    day_pos = {d: i for i, d in enumerate(days)}
    ds = np.zeros(days.size)
    dc = np.zeros(days.size)
    for d, s, c in zip(block_day, sums, counts):
        j = day_pos[d]
        ds[j] += s
        dc[j] += c
    if block_days <= 1:
        groups = [np.array([j]) for j in range(days.size)]
    else:
        groups = [np.arange(j, min(j + block_days, days.size))
                  for j in range(0, days.size, block_days)]
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(groups), size=len(groups))
        js = np.concatenate([groups[j] for j in idx])
        vals[b] = ds[js].sum() / max(dc[js].sum(), 1.0)
    return {"mean": float(ds.sum() / max(dc.sum(), 1.0)),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "p_ge_0": float(np.mean(vals >= 0.0)), "B": int(B),
            "blocks": int(len(groups)), "block_days": int(block_days)}


def paired_bootstrap(df, cand_cols, base_cols, interval: str, B: int = 2000,
                     seed: int = 42) -> dict:
    """Day-block paired bootstrap of mean pinball and coverage differences."""
    decision = df["decision_ms"].to_numpy(np.int64)
    y = df["y_logret"].to_numpy(np.float64) * 1e4
    qc = df[list(cand_cols)].to_numpy(np.float64) * 1e4
    qb = df[list(base_cols)].to_numpy(np.float64) * 1e4
    taus = (0.1, 0.25, 0.5, 0.75, 0.9)
    loss_c = np.mean([_pinball(y, qc[:, j], t) for j, t in enumerate(taus)], axis=0)
    loss_b = np.mean([_pinball(y, qb[:, j], t) for j, t in enumerate(taus)], axis=0)
    d = loss_c - loss_b
    inside_c = ((y >= qc[:, 1]) & (y <= qc[:, 3])).astype(np.float64)
    inside_b = ((y >= qb[:, 1]) & (y <= qb[:, 3])).astype(np.float64)
    day = day_index(decision)
    ones = np.ones_like(d)
    out = {}
    for name, vals in (("pinball_diff_bps", d),
                       ("cov50_diff", inside_c - inside_b)):
        out[name] = {
            "day": _resample_mean_ci(day, vals, ones, 1, B, seed),
            "week": _resample_mean_ci(day, vals, ones, 7, B, seed + 1),
        }
    return out


def dm_hac(d: np.ndarray, lag: int) -> dict:
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return {"stat": None, "p": None, "n": int(n)}
    mean = float(d.mean())
    x = d - mean
    s = float(np.mean(x * x))
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(np.mean(x[l:] * x[:-l]))
    if s <= 0:
        return {"stat": None, "p": None, "n": int(n)}
    stat = mean / math.sqrt(s / n)
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return {"stat": round(stat, 4), "p": round(p, 8), "n": int(n), "lag": int(lag)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------- helpers

def load_macro(path: Path) -> dict:
    import csv
    times = []
    types = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row.get("importance", "1")) < 1:
                    continue
                times.append(int(row["event_time"]))
                types.append(row.get("type", ""))
            except (KeyError, ValueError):
                continue
    return {"times": np.asarray(sorted(times), dtype=np.int64),
            "types": types, "path": str(path)}


def _qcols(candidate: str, band: str) -> tuple[str, str]:
    if band == "50":
        return (f"q25_{candidate}", f"q75_{candidate}")
    return (f"q10_{candidate}", f"q90_{candidate}")


def evaluate_candidate(df, candidate: str, macro: dict | None,
                       interval: str, mask: np.ndarray | None = None) -> dict:
    sub = df if mask is None else df[mask].reset_index(drop=True)
    qc = [f"q{k}_{candidate}" for k in ("10", "25", "50", "75", "90")]
    Q = sub[qc].to_numpy(np.float64)
    y = sub["y_logret"].to_numpy(np.float64)
    m = metrics(y, Q, sub["ref_close"].to_numpy(np.float64),
                sub["sigma_model"].to_numpy(np.float64),
                sub["rv_future"].to_numpy(np.float64) if "rv_future" in sub else None)
    window = DEFAULT_WINDOW_DAYS[interval]
    m["rolling_cov50_error"] = rolling_coverage_error(df, _qcols(candidate, "50"),
                                                      window, 0.5, mask=mask)
    m["rolling_cov80_error"] = rolling_coverage_error(df, _qcols(candidate, "80"),
                                                      window, 0.8, mask=mask)
    m["cells"] = cells(df, _qcols(candidate, "50"), _qcols(candidate, "80"),
                       macro, mask=mask)
    m["worst_valid_dev50_pp"] = worst_valid_deviation(m["cells"], "dev50_pp")
    m["worst_valid_dev80_pp"] = worst_valid_deviation(m["cells"], "dev80_pp")
    if f"active_{candidate}" in sub:
        act = sub[f"active_{candidate}"].to_numpy(bool)
        m["active_fraction"] = round(float(act.mean()), 5)
        if act.any():
            m["active_metrics"] = metrics(
                y[act], Q[act], sub["ref_close"].to_numpy(np.float64)[act],
                sub["sigma_model"].to_numpy(np.float64)[act],
                sub["rv_future"].to_numpy(np.float64)[act] if "rv_future" in sub else None)
    return m


def bridge_v5(df, v5_flag: bool | None = None) -> dict:
    """Old/new protocol bridge on identical rows."""
    Q = df[["q10", "q25", "q50", "q75", "q90"]].to_numpy(np.float64)
    centered = Q - Q[:, 2:3]
    y = df["y_logret"].to_numpy(np.float64)
    ref = df["ref_close"].to_numpy(np.float64)
    sig = df["sigma_model"].to_numpy(np.float64)
    rv = df["rv_future"].to_numpy(np.float64)
    return {
        "v6_fixed_no_fallback": metrics(y, Q, ref, sig, rv),
        "v5_deployment_centered_bridge": metrics(y, centered, ref, sig, rv),
        "frozen_har_on_identical_rows": metrics(
            y, df[["har_q10", "har_q25", "har_q50", "har_q75", "har_q90"]].to_numpy(np.float64),
            ref, sig, rv),
        "v5_deployed_median_fallback_flag": v5_flag,
    }


def confirmatory_split(df, fraction: float = v6proto.GAMMA_SELECT_FRACTION) -> tuple:
    decision = df["decision_ms"].to_numpy(np.int64)
    cut = int(np.quantile(decision, fraction))
    inner = decision <= cut
    return inner, ~inner


def run_interval(df, interval: str, out_dir: Path, *, macro: dict | None = None,
                 v5_flag: bool | None = None,
                 scopes: tuple[str, ...] = ("fold",),
                 longer_window: bool = False, delay_ms: int = 0,
                 log_fn=print) -> dict:
    out_dir = v6proto.assert_experiment_output(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("decision_ms", kind="stable").reset_index(drop=True)
    window = DEFAULT_WINDOW_DAYS[interval]
    inner, outer = confirmatory_split(df)

    summary: dict = {
        "interval": interval, "n": int(len(df)),
        "span_utc8": [qfmt(int(df.decision_ms.iat[0])), qfmt(int(df.decision_ms.iat[-1]))],
        "window_days": window, "inner_n": int(inner.sum()), "outer_n": int(outer.sum()),
        "inner_span_utc8": [qfmt(int(df.decision_ms[inner].iat[0])),
                            qfmt(int(df.decision_ms[inner].iat[-1]))] if inner.any() else None,
        "outer_span_utc8": [qfmt(int(df.decision_ms[outer].iat[0])),
                            qfmt(int(df.decision_ms[outer].iat[-1]))] if outer.any() else None,
        "candidates": {}, "gamma_selection": None, "v5_deployed_median_fallback_flag": v5_flag,
        "label_delay_ms": int(delay_ms),
    }

    work = df.copy()
    work, s = replay_interval(work, interval, "none", window_days=window,
                              delay_ms=delay_ms)
    summary["candidates"]["none"] = s
    work, s = replay_interval(work, interval, "cqr", window_days=window,
                              delay_ms=delay_ms, state_scope="fold", log_fn=log_fn)
    summary["candidates"]["cqr"] = s

    inner_loss = {}
    for gamma in v6proto.GAMMA_CANDIDATES:
        tmp, _s = replay_interval(df, interval, "cqr_aci", window_days=window,
                                  gamma=gamma, delay_ms=delay_ms,
                                  state_scope="fold", log_fn=log_fn)
        Q = tmp[[f"q{k}_cqr_aci" for k in ("10", "25", "50", "75", "90")]].to_numpy(np.float64)
        y = tmp["y_logret"].to_numpy(np.float64)
        mm = metrics(y[inner], Q[inner], tmp["ref_close"].to_numpy(np.float64)[inner],
                     tmp["sigma_model"].to_numpy(np.float64)[inner])
        inner_loss[gamma] = mm["pinball_combined_bps"]
    gamma_best = min(inner_loss, key=lambda g: (inner_loss[g], g))
    summary["gamma_selection"] = {"inner_pinball_bps": inner_loss, "chosen": gamma_best,
                                  "rule": "lowest inner pinball, ties -> smaller gamma"}
    work, s = replay_interval(work, interval, "cqr_aci", window_days=window,
                              gamma=gamma_best, delay_ms=delay_ms,
                              state_scope="fold", log_fn=log_fn)
    summary["candidates"]["cqr_aci"] = s

    if longer_window:
        lw = window * v6proto.LONGER_WINDOW_FACTOR
        tmp, _s = replay_interval(df, interval, "cqr", window_days=lw,
                                  delay_ms=delay_ms, state_scope="fold")
        Q = tmp[[f"q{k}_cqr" for k in ("10", "25", "50", "75", "90")]].to_numpy(np.float64)
        y = tmp["y_logret"].to_numpy(np.float64)
        summary["longer_window_inner_only"] = {
            "window_days": lw, "candidate": "cqr",
            "inner_metrics": metrics(y[inner], Q[inner],
                                     tmp["ref_close"].to_numpy(np.float64)[inner],
                                     tmp["sigma_model"].to_numpy(np.float64)[inner]),
        }

    if "pooled" in scopes:
        tmp, sp = replay_interval(df, interval, "cqr", window_days=window,
                                  delay_ms=delay_ms, state_scope="pooled", log_fn=log_fn)
        summary["pooled_scope_sensitivity"] = {
            "summary": sp,
            "full_metrics": evaluate_candidate(tmp, "cqr", macro, interval),
            "outer_metrics": evaluate_candidate(tmp, "cqr", macro, interval, mask=outer)
            if outer.any() else None,
        }

    for c in ("none", "cqr", "cqr_aci"):
        summary["candidates"][c]["full"] = evaluate_candidate(work, c, macro, interval)
        if outer.any():
            summary["candidates"][c]["outer_confirmatory"] = evaluate_candidate(
                work, c, macro, interval, mask=outer)

    summary["bridge"] = bridge_v5(work, v5_flag)

    lag = {"5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6}[interval]
    boot = {}
    for c in ("cqr", "cqr_aci"):
        cols_c = tuple(f"q{k}_{c}" for k in ("10", "25", "50", "75", "90"))
        cols_b = tuple(f"q{k}_none" for k in ("10", "25", "50", "75", "90"))
        sub = work[outer] if outer.any() else work
        b = paired_bootstrap(sub, cols_c, cols_b, interval)
        y = sub["y_logret"].to_numpy(np.float64) * 1e4
        qc = sub[list(cols_c)].to_numpy(np.float64) * 1e4
        qb = sub[list(cols_b)].to_numpy(np.float64) * 1e4
        taus = (0.1, 0.25, 0.5, 0.75, 0.9)
        lc = np.mean([_pinball(y, qc[:, j], t) for j, t in enumerate(taus)], axis=0)
        lb = np.mean([_pinball(y, qb[:, j], t) for j, t in enumerate(taus)], axis=0)
        b["dm_hac_pinball"] = dm_hac(lc - lb, lag)
        boot[c] = b
    summary["bootstrap_outer"] = boot

    keep = (["interval", "fold", "decision_ms", "y_logret", "ref_close", "sigma_model",
             "q10", "q25", "q50", "q75", "q90",
             "har_q10", "har_q25", "har_q50", "har_q75", "har_q90"]
            + [c for c in work.columns if c.startswith(("q10_", "q25_", "q50_", "q75_", "q90_",
                                                        "active_", "reason_", "ncal_"))]
            + [c for c in ("alpha_lo50", "alpha_hi50", "alpha_lo80", "alpha_hi80")
               if c in work])
    path = out_dir / f"replay_{interval}.parquet"
    work[keep].to_parquet(path, index=False, compression="zstd")
    summary["replay_path"] = str(path)
    summary["peek_ledger"] = {
        "holdout_read": False,
        # none + cqr + two inner gamma runs + final cqr_aci
        # (+1 longer window, +1 pooled-scope sensitivity when enabled)
        "candidate_replays": 3 + len(v6proto.GAMMA_CANDIDATES)
        + (1 if longer_window else 0) + (1 if "pooled" in scopes else 0),
    }
    return summary


def qfmt(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000 + 8 * 3600))


def compare_to_none(summary: dict) -> dict:
    """Apply the pre-registered acceptance rules on the confirmatory set."""
    out = {}
    for c in ("cqr", "cqr_aci"):
        cand = summary["candidates"].get(c, {}).get("outer_confirmatory")
        base = summary["candidates"].get("none", {}).get("outer_confirmatory")
        if not cand or not base:
            continue
        pin_w = (cand["pinball_combined_bps"] - base["pinball_combined_bps"]) / base["pinball_combined_bps"]
        is_w = (cand["IS50_bps"] - base["IS50_bps"]) / base["IS50_bps"]
        w_w = (cand["w50_bps"] - base["w50_bps"]) / base["w50_bps"]
        rc_b = base["rolling_cov50_error"]["mean_abs"]
        rc_c = cand["rolling_cov50_error"]["mean_abs"]
        rc_improve = (rc_b - rc_c) / rc_b if rc_b else None
        wd_b = base.get("worst_valid_dev50_pp")
        wd_c = cand.get("worst_valid_dev50_pp")
        wd_improve = (wd_b - wd_c) / wd_b if (wd_b and wd_c is not None) else None
        cov_ok = (0.48 <= cand["cov50"] <= 0.52) and (0.78 <= cand["cov80"] <= 0.82)
        cost_ok = bool(pin_w is not None and pin_w <= 0.003
                       and is_w <= 0.005 and w_w <= 0.03)
        improve_ok = bool((rc_improve is not None and rc_improve >= 0.20)
                          or (wd_improve is not None and wd_improve >= 0.20))
        out[c] = {
            "pinball_rel_change": round(float(pin_w), 6),
            "IS50_rel_change": round(float(is_w), 6),
            "w50_rel_change": round(float(w_w), 6),
            "rolling_cov_error_rel_improve": None if rc_improve is None else round(float(rc_improve), 6),
            "worst_cell_rel_improve": None if wd_improve is None else round(float(wd_improve), 6),
            "coverage_ok": bool(cov_ok),
            "cost_ok": cost_ok,
            "improve_ok": improve_ok,
            "keep_none": bool(not (cov_ok and cost_ok and improve_ok)),
        }
    return out


def report_markdown(summaries: dict, out_dir: Path) -> Path:
    lines = ["# v6 stage 1 calibration replay", "",
             "All numbers are bps of the reference price unless stated; coverage is a fraction.",
             "`none` is the frozen v5 raw prediction on identical rows.", ""]
    for iv, s in summaries.items():
        if not isinstance(s, dict) or "candidates" not in s:
            continue
        lines.append(f"## {iv}")
        lines.append("")
        lines.append(f"- rows: {s['n']}  span: {s['span_utc8'][0]} .. {s['span_utc8'][1]} (UTC+8)")
        if s.get("outer_span_utc8"):
            lines.append(f"- confirmatory (outer 60%): {s['outer_n']} rows "
                         f"{s['outer_span_utc8'][0]} .. {s['outer_span_utc8'][1]}")
        lines.append("")
        lines.append("| candidate | set | n | pinball | cov50 | cov80 | w50 | IS50 | q50 MAE | left50 | right50 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in ("none", "cqr", "cqr_aci"):
            for label, key in (("full", "full"), ("outer", "outer_confirmatory")):
                m = s["candidates"][c].get(key)
                if not m:
                    continue
                lines.append(f"| {c} | {label} | {m['n']} | {m['pinball_combined_bps']} | "
                             f"{m['cov50']} | {m['cov80']} | {m['w50_bps']} | {m['IS50_bps']} | "
                             f"{m['q50_mae_bps']} | {m['left50']} | {m['right50']} |")
        lines.append("")
        for c, d in compare_to_none(s).items():
            def fmt(x, pct=True):
                if x is None:
                    return "n/a"
                return f"{x:+.4%}" if pct else f"{x:+.6f}"
            lines.append(f"- {c}: pinball {fmt(d['pinball_rel_change'])}, "
                         f"IS50 {fmt(d['IS50_rel_change'])}, w50 {fmt(d['w50_rel_change'])}, "
                         f"rolling-cov improve {fmt(d['rolling_cov_error_rel_improve'])}, "
                         f"worst-cell improve {fmt(d['worst_cell_rel_improve'])}, "
                         f"coverage_ok={d['coverage_ok']}, cost_ok={d['cost_ok']}, "
                         f"improve_ok={d['improve_ok']}, keep_none={d['keep_none']}")
        lines.append("")
        lines.append(f"- gamma selection: {json.dumps(s.get('gamma_selection'), ensure_ascii=False)}")
        lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser(description="offline calibration replay on v5 OOF")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--intervals", nargs="+", default=list(v6proto.PRODUCT_INTERVALS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--macro", default=r"D:\Tool\cg\superE\data\macro_calendar.csv")
    ap.add_argument("--scopes", nargs="+", default=["fold"])
    ap.add_argument("--longer-window", action="store_true")
    ap.add_argument("--delay-ms", type=int, default=0)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "calibration"
    macro = load_macro(Path(args.macro))
    manifest_path = run_dir / "oof_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summaries = {}
    for iv in args.intervals:
        df = pd.read_parquet(run_dir / f"oof_{iv}.parquet")
        v5_flag = None
        m = manifest.get(iv)
        if m:
            v5_flag = bool(m.get("v5_deployed_median_fallback"))
        summaries[iv] = run_interval(df, iv, out_dir, macro=macro, v5_flag=v5_flag,
                                     scopes=tuple(args.scopes),
                                     longer_window=args.longer_window,
                                     delay_ms=int(args.delay_ms))
        print(f"[{iv}] done")
    for c in ("cqr", "cqr_aci"):
        pv = {}
        for iv, s in summaries.items():
            b = s.get("bootstrap_outer", {}).get(c, {}).get("dm_hac_pinball", {})
            if b.get("p") is not None:
                pv[iv] = b["p"]
        if pv:
            summaries.setdefault("_holm", {})[c] = holm(pv)
    (out_dir / "metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    report_markdown(summaries, out_dir)
    print(f"report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
