"""Live v6 CQR calibration for the served quantile forecast.

This module attaches the *frozen* stage-1 calibration state machine
(:mod:`supere.train.calib_v6`) to the real prediction path.  It does not train
or change the v5 model: raw quantiles come from the deployed artifact, CQR only
shifts the four outer quantiles, and ``q50`` is never moved.

Policy (2026-11-11 deployment, plan ``计划-第六版上线-20261111.md``):

* rolling two-sided normalized CQR on 5m, 1h and 4h only (no ACI);
* window 28 days for 5m/1h, 90 days for 4h;
* 15m and 30m stay ``none`` by policy;
* fewer than 200 matured calibration scores falls back to ``none`` with an
  explicit reason;
* a label is consumed only after ``decision + horizon + label_delay``; the
  default publication delay is 60 s (4h included), so nothing is learned early;
* one prediction id is learned once across requests and restarts; duplicate
  requests return the cached calibrated output for the same decision bar;
* predictions are keyed by (symbol, horizon, model fingerprint, calib version).
  A changed model fingerprint or a damaged snapshot never mixes state: the old
  file is quarantined and the interval degrades to raw + an alert.

State is written only below ``<data_dir>/calib_v6`` (never ``models_quantile``)
with a temporary file plus :func:`os.replace` so a crash cannot leave a
half-written snapshot.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import calib_v6
from supere.train import v6proto

log = logging.getLogger("supere.train.calib_live")

LIVE_SCHEMA = "supere.calib_live/1"
LIVE_STATE_VERSION = "v6-live-1"
CALIB_VERSION = "v6-cqr-1"
CALIB_INTERVALS: tuple[str, ...] = ("5m", "1h", "4h")
NONE_INTERVALS: tuple[str, ...] = ("15m", "30m")
DEFAULT_LABEL_DELAY_MS = 60_000
# A label row that never shows up must not block the state machine forever.
MISSING_DEADLINE_MS = 24 * 3_600_000
# Duplicate protection / cached outputs only need to cover recent decisions.
RETAIN_MS = 7 * 86_400_000
MAX_OUTPUT_CACHE = 64
MAX_MISSING_RECORDS = 500
BPS = 1e4

QUANTILE_KEYS = ("p10", "p25", "median", "p75", "p90")


# ------------------------------------------------------------------ metadata

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_fingerprint(directory: Path | None, *, model_id: str = "",
                         meta: dict | None = None) -> dict:
    """Content fingerprint of the deployed artifact files.

    The live state is partitioned by this value.  Only metadata is derived;
    the model files themselves are never modified.
    """
    directory = Path(directory) if directory else None
    files = ("meta.json", "features.json", "model_p10.txt", "model_p25.txt",
             "model_median.txt", "model_p75.txt", "model_p90.txt")
    shas: dict[str, str] = {}
    if directory is not None and directory.exists():
        for name in files:
            path = directory / name
            if path.is_file():
                shas[name] = _sha256_file(path)
    if shas:
        h = hashlib.sha256()
        for name in sorted(shas):
            h.update(name.encode())
            h.update(shas[name].encode())
        digest = h.hexdigest()
        source = "artifact_files"
    else:  # test doubles / partially materialised artifacts
        payload = json.dumps({"model_id": model_id, "meta": meta or {}},
                             sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        source = "model_id_meta"
    return {
        "model_hash": digest,
        "model_hash_short": digest[:16],
        "model_id": str(model_id or ""),
        "feature_hash": shas.get("features.json", "")[:16],
        "meta_hash": shas.get("meta.json", "")[:16],
        "fingerprint_source": source,
        "files": shas,
    }


class _FingerprintCache:
    """Hash the artifact once per directory stamp; never per request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[str, float, str], dict] = {}

    def get(self, art) -> dict:
        directory = getattr(art, "directory", None)
        meta = getattr(art, "meta", {}) or {}
        model_id = str(getattr(art, "model_id", "") or meta.get("model_id") or "")
        stamp = 0.0
        if directory is not None:
            try:
                stamp = max((p.stat().st_mtime for p in Path(directory).iterdir()
                             if not p.name.endswith(".tmp")), default=0.0)
            except OSError:
                stamp = 0.0
        key = (str(directory), stamp, model_id)
        with self._lock:
            cached = self._values.get(key)
        if cached is not None:
            return cached
        info = artifact_fingerprint(Path(directory) if directory else None,
                                    model_id=model_id, meta=meta)
        with self._lock:
            self._values[key] = info
        return info


_FINGERPRINTS = _FingerprintCache()


# ------------------------------------------------------------------- labels

class SqliteLabelSource:
    """Realised return for a matured prediction, read from closed K-lines.

    ``y = log(close(decision + horizon) / close(decision))``.  Both bars must
    already exist, and ``decision + horizon + delay <= now``; a forming bar can
    never be used because its close time is in the future by construction.
    """

    def __init__(self, db_path: Path, delay_ms: int = DEFAULT_LABEL_DELAY_MS) -> None:
        self.db_path = Path(db_path)
        self.delay_ms = int(delay_ms)
        self._cache: dict[tuple[str, int], float | None] = {}

    def label(self, interval: str, decision_ms: int, now_ms: int) -> float | None:
        decision_ms = int(decision_ms)
        now_ms = int(now_ms)
        horizon = INTERVAL_MS[interval]
        # Maturity is checked here as well as by the engine; never early.
        if decision_ms + horizon + self.delay_ms > now_ms:
            return None
        key = (interval, decision_ms)
        if key in self._cache:
            return self._cache[key]
        value = self._read(interval, decision_ms)
        self._cache[key] = value
        return value

    def _read(self, interval: str, decision_ms: int) -> float | None:
        from supere.train import data as tdata
        if not self.db_path.exists():
            return None
        try:
            conn = tdata.open_readonly(self.db_path)
        except Exception:  # noqa: BLE001 - degrade, never block the forecast
            return None
        try:
            rows = conn.execute(
                "SELECT open_time, c FROM klines WHERE market = ? AND symbol = ? "
                "AND interval = ? AND open_time IN (?, ?)",
                ("um", "ETHUSDT", interval,
                 decision_ms - INTERVAL_MS[interval], decision_ms),
            ).fetchall()
        except Exception:  # noqa: BLE001
            return None
        finally:
            conn.close()
        closes = {int(r[0]): r[1] for r in rows}
        prev = closes.get(decision_ms - INTERVAL_MS[interval])
        nxt = closes.get(decision_ms)
        try:
            prev_f, next_f = float(prev), float(nxt)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(prev_f) and np.isfinite(next_f) and prev_f > 0 and next_f > 0):
            return None
        return float(math.log(next_f / prev_f))


# -------------------------------------------------------------------- state

def state_root_for(data_root: Path) -> Path:
    return Path(data_root) / "calib_v6"


def state_path(state_root: Path, symbol: str, horizon: str,
               model_hash: str, calib_version: str = CALIB_VERSION) -> Path:
    """Partitioned state path: symbol / horizon / model fingerprint / version."""
    return (Path(state_root) / str(symbol) / str(horizon) / str(model_hash)[:16]
            / str(calib_version) / "state.json")


def prediction_id(interval: str, model_hash: str, decision_ms: int) -> str:
    return f"{interval}:{str(model_hash)[:12]}:{int(decision_ms)}"


def _pid_decision(pid: str) -> int | None:
    try:
        return int(str(pid).rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return None


def _quarantine(path: Path, reason: str, now_ms: int) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(int(now_ms) / 1000))
    target = path.with_name(f"state.{reason}-{stamp}.json")
    try:
        os.replace(path, target)
        return target
    except OSError as exc:  # pragma: no cover - unusual filesystem failure
        log.warning("校准状态隔离失败 %s: %s", path, exc)
        return None


def _price_block(q_log: np.ndarray, ref_close: float) -> dict:
    q = np.asarray(q_log, dtype=np.float64)
    ref = float(ref_close)
    return {
        "p10": float(ref * math.exp(float(q[0]))),
        "p25": float(ref * math.exp(float(q[1]))),
        "median": float(ref * math.exp(float(q[2]))),
        "p75": float(ref * math.exp(float(q[3]))),
        "p90": float(ref * math.exp(float(q[4]))),
        "width50_bps": round(float((q[3] - q[1]) * BPS), 4),
        "width80_bps": round(float((q[4] - q[0]) * BPS), 4),
    }


@dataclass
class StateKey:
    symbol: str
    horizon: str
    model_hash: str
    calib_version: str = CALIB_VERSION

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "horizon": self.horizon,
                "model_hash": self.model_hash, "calib_version": self.calib_version}


@dataclass
class LiveState:
    """One interval's persisted CQR state plus its exactly-once bookkeeping."""

    key: StateKey
    path: Path
    engine: calib_v6.CalibrationEngine
    eps: float
    source: str = "live"
    bootstrap: dict | None = None
    outputs: dict = field(default_factory=dict)
    live_rows: int = 0
    created_ms: int = 0
    updated_ms: int = 0
    alert: str | None = None
    last_reason: str = "cold"
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------- snapshot
    def document(self) -> dict:
        return {
            "schema": LIVE_SCHEMA,
            "state_version": LIVE_STATE_VERSION,
            "key": self.key.as_dict(),
            "engine_config": {
                "candidate": self.engine.cfg.candidate,
                "window_days": float(self.engine.window_days_eff),
                "gamma": float(self.engine.cfg.gamma),
                "delay_ms": int(self.engine.cfg.delay_ms),
                "min_samples": int(self.engine.cfg.min_samples),
                "horizon_ms": int(self.engine.cfg.horizon_ms),
            },
            "eps": float(self.eps),
            "engine": self.engine.snapshot(),
            "source": self.source,
            "bootstrap": self.bootstrap,
            "outputs": self.outputs,
            "live_rows": int(self.live_rows),
            "created_ms": int(self.created_ms),
            "updated_ms": int(self.updated_ms),
            "alert": self.alert,
            "last_reason": self.last_reason,
        }

    def save(self, now_ms: int | None = None) -> None:
        self.updated_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        calib_v6.write_snapshot(self.path, _json_safe(self.document()))

    # -------------------------------------------------------------- labels
    def advance(self, now_ms: int, labels: SqliteLabelSource | None) -> bool:
        """Consume matured labels in decision order; return whether state moved.

        A missing label row blocks the queue (ordered late-label consumption)
        until ``MISSING_DEADLINE_MS``, after which it is recorded as missing and
        skipped.  The engine clock is only advanced past items that are resolved
        or declared missing, so a late database row is not silently dropped.
        """
        changed = False
        eng = self.engine
        while eng.pending:
            avail, pid = int(eng.pending[0][0]), str(eng.pending[0][1])
            if avail > now_ms:
                break
            decision = _pid_decision(pid)
            y = labels.label(self.key.horizon, decision, now_ms) if (
                labels is not None and decision is not None
                and decision + INTERVAL_MS[self.key.horizon] + eng.cfg.delay_ms <= now_ms
            ) else None
            if y is not None:
                eng.set_label(pid, y)
                eng.advance(avail)
                changed = True
                continue
            if now_ms - avail >= MISSING_DEADLINE_MS:
                if pid not in eng.missing_labels:
                    eng.missing_labels.append(pid)
                eng.advance(avail)
                changed = True
                continue
            break
        if not eng.pending or int(eng.pending[0][0]) > now_ms:
            eng.advance(now_ms)
        self._prune(now_ms)
        return changed

    def _prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - RETAIN_MS
        eng = self.engine
        for attr in ("seen_predict", "seen_observe"):
            values = getattr(eng, attr)
            kept = {pid for pid in values
                    if (lambda d: d is None or d >= cutoff)(_pid_decision(pid))}
            if len(kept) != len(values):
                setattr(eng, attr, kept)
        if len(eng.missing_labels) > MAX_MISSING_RECORDS:
            eng.missing_labels = eng.missing_labels[-MAX_MISSING_RECORDS:]
        if len(self.outputs) > MAX_OUTPUT_CACHE:
            def _key(item):
                return int(item[1].get("decision_ms") or 0)
            ordered = sorted(self.outputs.items(), key=_key)[-MAX_OUTPUT_CACHE:]
            self.outputs = dict(ordered)

    # ------------------------------------------------------------- predict
    def predict(self, decision_ms: int, raw_q, sigma: float, ref_close: float,
                now_ms: int, labels: SqliteLabelSource | None) -> tuple[np.ndarray, dict, bool]:
        """Return (calibrated log quantiles, diagnostics, state_changed)."""
        with self._lock:
            changed = self.advance(now_ms, labels)
            pid = prediction_id(self.key.horizon, self.key.model_hash, decision_ms)
            cached = self.outputs.get(pid)
            if cached is not None:
                if changed:
                    self.save(now_ms)
                q = np.asarray(cached["q_log"], dtype=np.float64)
                diag = dict(cached["diag"])
                diag["reason"] = "duplicate"
                diag["duplicate"] = True
                self.last_reason = "duplicate"
                return q, diag, changed
            if pid in self.engine.seen_predict:
                # Seen after a restart but no cached output survived: never
                # learn twice; serve raw with an explicit reason.
                if changed:
                    self.save(now_ms)
                q = np.sort(np.asarray(raw_q, dtype=np.float64).copy())
                diag = {"active": False, "reason": "duplicate_no_cache",
                        "n_calib": int(self.engine.band50.count), "duplicate": True}
                self.last_reason = "duplicate_no_cache"
                return q, diag, changed
            try:
                out, diag = self.engine.predict(
                    int(decision_ms), np.asarray(raw_q, dtype=np.float64),
                    float(sigma), pid, int(decision_ms) + INTERVAL_MS[self.key.horizon])
            except ValueError as exc:
                # Out-of-order decision (e.g. a rewound bar) must not crash the
                # endpoint or poison the state.
                log.warning("校准拒绝乱序预测 %s %s: %s", self.key.horizon, pid, exc)
                if changed:
                    self.save(now_ms)
                q = np.sort(np.asarray(raw_q, dtype=np.float64).copy())
                diag = {"active": False, "reason": "out_of_order",
                        "n_calib": int(self.engine.band50.count)}
                self.last_reason = "out_of_order"
                return q, diag, changed
            self.outputs[pid] = {
                "decision_ms": int(decision_ms),
                "q_log": [float(x) for x in out],
                "diag": dict(diag),
                "ref_close": float(ref_close),
            }
            self.live_rows += 1
            self.source = "bootstrap+live" if self.bootstrap else "live"
            self.last_reason = str(diag.get("reason") or "ok")
            self.save(now_ms)
            return out, diag, True

    def status(self) -> dict:
        with self._lock:
            eng = self.engine
            return {
                "interval": self.key.horizon,
                "symbol": self.key.symbol,
                "model_hash": self.key.model_hash,
                "calib_version": self.key.calib_version,
                "candidate": eng.cfg.candidate,
                "window_days": float(eng.window_days_eff),
                "n_mature": int(eng.band50.count),
                "n_mature_80": int(eng.band80.count),
                "pending": len(eng.pending),
                "live_rows": int(self.live_rows),
                "source": self.source,
                "bootstrap_source": (self.bootstrap or {}).get("bootstrap_source"),
                "bootstrap_rows": (self.bootstrap or {}).get("rows_used"),
                "in_sample_overlap": (self.bootstrap or {}).get("in_sample_overlap"),
                "alert": self.alert,
                "last_reason": self.last_reason,
                "state_file": str(self.path),
                "updated_ms": int(self.updated_ms),
            }


def _json_safe(value):
    """JSON with plain floats; NaN/inf become None so strict parsers accept it."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _restore_engine(snap: dict, *, candidate: str, window_days: float,
                    min_samples: int, delay_ms: int, capacity: int,
                    eps: float) -> calib_v6.CalibrationEngine:
    """Restore the inner engine snapshot (``doc["engine"]``), not the wrapper."""
    cfg = calib_v6.EngineConfig(candidate=candidate, window_days=window_days,
                                min_samples=min_samples, delay_ms=delay_ms)
    return calib_v6.CalibrationEngine.restore(cfg, capacity, eps, snap)


def open_state(data_root: Path, interval: str, art, *, clock=time.time,
               delay_ms: int = DEFAULT_LABEL_DELAY_MS,
               state_root: Path | None = None,
               labels: SqliteLabelSource | None = None) -> LiveState:
    """Load or cold-start one interval's state without ever mixing artifacts."""
    now_ms = int(clock() * 1000)
    meta = getattr(art, "meta", {}) or {}
    fp = _FINGERPRINTS.get(art)
    symbol = str(meta.get("config", {}).get("symbol") or "ETHUSDT")
    window = float(v6proto.CALIB_WINDOW_DAYS[interval])
    key = StateKey(symbol=symbol, horizon=interval,
                   model_hash=fp["model_hash"], calib_version=CALIB_VERSION)
    root = Path(state_root) if state_root is not None else state_root_for(data_root)
    path = state_path(root, symbol, interval, fp["model_hash"], CALIB_VERSION)
    capacity = int(math.ceil(window * 86_400_000 / INTERVAL_MS[interval])) + 64

    base = {
        "candidate": "cqr", "window_days": window,
        "min_samples": v6proto.MIN_CALIB_SAMPLES, "delay_ms": int(delay_ms),
        "capacity": capacity,
    }
    alert = None
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("schema") != LIVE_SCHEMA:
                raise ValueError("schema mismatch")
            if doc.get("state_version") != LIVE_STATE_VERSION:
                raise ValueError("state version mismatch")
            if doc.get("key") != key.as_dict():
                raise ValueError("state key mismatch")
            persisted_cfg = doc.get("engine_config") or {}
            expected_cfg = {"candidate": base["candidate"],
                            "window_days": base["window_days"],
                            "delay_ms": base["delay_ms"],
                            "min_samples": base["min_samples"]}
            for name, value in expected_cfg.items():
                if persisted_cfg.get(name) != value:
                    raise ValueError(f"engine config mismatch: {name}")
            if not isinstance(doc.get("engine"), dict):
                raise ValueError("missing engine snapshot")
            eps = float(doc["eps"])
            eng = _restore_engine(doc.get("engine") or {}, candidate=base["candidate"],
                                  window_days=base["window_days"],
                                  min_samples=base["min_samples"],
                                  delay_ms=base["delay_ms"],
                                  capacity=capacity, eps=eps)
            state = LiveState(
                key=key, path=path, engine=eng, eps=eps,
                source=str(doc.get("source") or "live"),
                bootstrap=doc.get("bootstrap"),
                outputs={str(k): v for k, v in (doc.get("outputs") or {}).items()},
                live_rows=int(doc.get("live_rows") or 0),
                created_ms=int(doc.get("created_ms") or now_ms),
                updated_ms=int(doc.get("updated_ms") or now_ms),
                alert=doc.get("alert"),
                last_reason=str(doc.get("last_reason") or "cold"),
            )
            if state.advance(now_ms, labels):
                state.save(now_ms)
            return state
        except Exception as exc:  # noqa: BLE001 - damaged state must degrade
            quarantined = _quarantine(path, "corrupt", now_ms)
            alert = f"snapshot_quarantined:{type(exc).__name__}: {exc}"
            message = (f"[calib_v6] {interval} 状态不可用，已隔离到 "
                       f"{quarantined.name if quarantined else 'n/a'}"
                       f"（{type(exc).__name__}: {exc}）；本周期回退 raw")
            log.warning(message)
            print(message, file=sys.stderr, flush=True)

    # Cold start.  Never bootstrap here: the CLI does that before deployment.
    eps = 1e-9
    eng = calib_v6.CalibrationEngine(
        calib_v6.EngineConfig(candidate=base["candidate"], window_days=base["window_days"],
                              min_samples=base["min_samples"], delay_ms=base["delay_ms"]),
        capacity, eps)
    state = LiveState(key=key, path=path, engine=eng, eps=eps, source="live",
                      created_ms=now_ms, updated_ms=now_ms, alert=alert,
                      last_reason="cold")
    return state


# --------------------------------------------------------------- calibrator

CALIB_NONE_BLOCK = {
    "calib": "none",
    "policy": "none",
    "calib_version": CALIB_VERSION,
    "candidate": "none",
    "window_days": None,
    "min_samples": v6proto.MIN_CALIB_SAMPLES,
    "n_mature": 0,
    "active": False,
    "reason": "interval_policy",
    "model_hash": None,
    "model_id": None,
    "feature_hash": None,
    "source": "policy",
    "bootstrap": None,
    "alert": None,
    "delay_ms": DEFAULT_LABEL_DELAY_MS,
    "interval_policy": "none",
}


class LiveCalibrator:
    """Attach CQR to a served forecast, map failures to raw + alert."""

    def __init__(self, data_root: Path, *, clock=time.time,
                 state_root: Path | None = None,
                 label_source: SqliteLabelSource | None = None,
                 delay_ms: int = DEFAULT_LABEL_DELAY_MS) -> None:
        self.data_root = Path(data_root)
        self._clock = clock
        self.state_root = Path(state_root) if state_root is not None else state_root_for(self.data_root)
        self.delay_ms = int(delay_ms)
        self.labels = label_source or SqliteLabelSource(self.data_root / "superE.sqlite",
                                                        delay_ms=self.delay_ms)
        self._states: dict[str, LiveState] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- state
    def _state(self, interval: str, art) -> LiveState:
        with self._lock:
            state = self._states.get(interval)
            if state is None:
                state = open_state(self.data_root, interval, art, clock=self._clock,
                                   delay_ms=self.delay_ms, state_root=self.state_root,
                                   labels=self.labels)
                self._states[interval] = state
            return state

    def status(self) -> dict:
        with self._lock:
            states = list(self._states.values())
        return {
            "enabled": True,
            "calib_version": CALIB_VERSION,
            "intervals": {s.key.horizon: s.status() for s in states},
            "policy": {"cqr": list(CALIB_INTERVALS), "none": list(NONE_INTERVALS)},
            "state_root": str(self.state_root),
        }

    # ----------------------------------------------------------- decorate
    def decorate(self, interval: str, art, pf: dict, now_ms: int) -> dict:
        """Add raw/calibrated/final blocks; return the modified forecast dict."""
        if not isinstance(pf, dict):
            return pf
        ref = pf.get("ref_close")
        try:
            ref = float(ref)
        except (TypeError, ValueError):
            ref = float("nan")
        keys = {k: pf.get(k) for k in QUANTILE_KEYS}
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in keys.values()) \
                or not math.isfinite(ref) or ref <= 0:
            pf["calibration"] = dict(CALIB_NONE_BLOCK, reason="bad_raw_forecast")
            return pf
        raw_q = np.log(np.asarray(
            [float(keys["p10"]) / ref, float(keys["p25"]) / ref,
             float(keys["median"]) / ref, float(keys["p75"]) / ref,
             float(keys["p90"]) / ref], dtype=np.float64))
        raw_block = {
            "p10": float(keys["p10"]), "p25": float(keys["p25"]),
            "median": float(keys["median"]), "p75": float(keys["p75"]),
            "p90": float(keys["p90"]),
            "width50_bps": round(float((raw_q[3] - raw_q[1]) * BPS), 4),
            "width80_bps": round(float((raw_q[4] - raw_q[0]) * BPS), 4),
        }
        pf["raw_price_forecast"] = raw_block

        if interval not in CALIB_INTERVALS:
            pf["calibrated_price_forecast"] = dict(raw_block)
            pf["final_price_forecast"] = dict(raw_block)
            pf["calibration"] = dict(CALIB_NONE_BLOCK)
            return pf

        sigma = pf.get("sigma_hat")
        decision_ms = pf.get("decision_ms")
        try:
            sigma = float(sigma)
            decision_ms = int(decision_ms)
        except (TypeError, ValueError):
            sigma, decision_ms = float("nan"), None
        if not math.isfinite(sigma) or sigma <= 0 or decision_ms is None:
            pf["calibrated_price_forecast"] = dict(raw_block)
            pf["final_price_forecast"] = dict(raw_block)
            pf["calibration"] = dict(CALIB_NONE_BLOCK, reason="missing_sigma_or_decision")
            return pf

        info = {
            "calib": "none", "policy": "cqr", "calib_version": CALIB_VERSION,
            "candidate": "cqr", "window_days": float(v6proto.CALIB_WINDOW_DAYS[interval]),
            "min_samples": v6proto.MIN_CALIB_SAMPLES, "n_mature": 0, "active": False,
            "reason": "error", "model_hash": None, "model_id": None,
            "feature_hash": None, "source": "live", "bootstrap": None, "alert": None,
            "delay_ms": int(self.delay_ms), "interval_policy": "cqr",
        }
        try:
            state = self._state(interval, art)
            out, diag, _changed = state.predict(decision_ms, raw_q, sigma, ref,
                                                now_ms, self.labels)
            fp = _FINGERPRINTS.get(art)
            active = bool(diag.get("active"))
            info.update({
                "calib": "cqr" if active else "none",
                "n_mature": int(diag.get("n_calib") or 0),
                "active": active,
                "reason": str(diag.get("reason") or ("ok" if active else "cold")),
                "model_hash": fp["model_hash"],
                "model_id": fp.get("model_id"),
                "feature_hash": fp.get("feature_hash"),
                "source": state.source,
                "bootstrap": state.bootstrap,
                "alert": state.alert,
                "state_file": str(state.path),
            })
            cal_block = _price_block(out, ref)
            # q50 is invariant; reuse the exact served median to avoid a last-bit
            # difference from the log/exp round trip.
            cal_block["median"] = float(keys["median"])
            if np.array_equal(out, raw_q):
                cal_block = dict(raw_block)
        except Exception as exc:  # noqa: BLE001 - calibration never breaks a forecast
            log.exception("v6 校准失败 %s", interval)
            info["reason"] = "error"
            info["alert"] = f"{type(exc).__name__}: {exc}"
            cal_block = dict(raw_block)
        pf["calibrated_price_forecast"] = cal_block
        pf["final_price_forecast"] = dict(cal_block)
        pf["calibration"] = info
        # Existing display fields point at the calibrated interval; the median is
        # untouched by construction.
        for name in QUANTILE_KEYS:
            pf[name] = cal_block[name]
        pf["width50_bps"] = cal_block["width50_bps"]
        pf["width80_bps"] = cal_block["width80_bps"]
        return pf
