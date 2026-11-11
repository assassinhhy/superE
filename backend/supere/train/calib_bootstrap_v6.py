"""Offline bootstrap of the live v6 CQR state (deployment model replay).

The live state machine needs 200 matured scores before CQR may replace ``none``.
Waiting for manual clicks to produce that many labels would take weeks, so this
CLI replays the **already deployed v5 artifact** over the real, already closed
K-lines of the last 28/90 days and feeds the matured results into the same
``CalibrationEngine`` used online.

Honesty constraints (plan 计划-第六版上线 item 3):

* no weights are retrained, no model files are written;
* only real closed bars and their realised next-bar return are used -- no
  timestamp is rewritten and no synthetic score is created;
* the replay window overlaps the model's training input/holdout period, so the
  output is marked ``bootstrap_source=deployment_model_historical_replay``,
  ``in_sample_overlap=true``, ``fresh_oof=false`` and
  ``strictly_out_of_sample=false``.  It is a practical warm start, not a fresh
  out-of-sample validation, and coverage is not guaranteed.

The output is written only to ``<data_dir>/calib_v6``.  Re-running without
``--force`` is a no-op when a live state for the same artifact already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import calib_live as live
from supere.train import calib_v6
from supere.train import quantile as q
from supere.train import quantile_v3 as v3
from supere.train import v6proto

WARMUP_BARS = 3_000
ONE_MIN_LOOKBACK = 20_160  # two 7-day RV windows before the replay start


def _fmt(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ms) / 1000 + 8 * 3600))


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def _sigma_all(meta: dict, S: v3.SamplesV3, cfg: v3.V3Config) -> np.ndarray:
    """Per-row sigma exactly as ``quantile_v3.predict_latest`` computes it."""
    sig_meta = meta["sigma"]
    kind = str(sig_meta["kind"])
    if kind == "ewma":
        return np.asarray(S.sig_base if S.sig_base.shape[0] == S.m else S.sig_base_full,
                          dtype=np.float64)
    if kind == "garch":
        return np.asarray(S.sig_garch if S.sig_garch.shape[0] == S.m else S.sig_garch_full,
                          dtype=np.float64)
    names, Z = v3._sigma_design(S, cfg, kind)
    index = {name: i for i, name in enumerate(names)}
    x = np.asarray(Z[:, [index[name] for name in sig_meta["feature_names"]]],
                   dtype=np.float64)
    pred = float(sig_meta["intercept"]) + x @ np.asarray(sig_meta["coef"], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        sigma = np.sqrt(np.exp(np.clip(pred, -30.0, 5.0)))
    return np.clip(sigma, float(sig_meta["clip_lo"]), float(sig_meta["clip_hi"]))


def _quantiles_all(meta: dict, art: v3.V3Artifact, S: v3.SamplesV3,
                   cfg: v3.V3Config) -> tuple[np.ndarray, np.ndarray]:
    """Sorted log quantiles for every row, mirroring ``predict_latest``."""
    sigma = _sigma_all(meta, S, cfg)
    shape_meta = meta["shape"]
    shape = v3.ShapeFit(
        kind=str(shape_meta["kind"]),
        boosters=art.boosters,
        zq=np.asarray(shape_meta["zq"], dtype=np.float64) if shape_meta.get("zq") else None,
        df=shape_meta.get("df"),
        scale=shape_meta.get("scale"),
        blend_w=shape_meta.get("blend_w"),
    )
    pos = np.arange(S.m, dtype=np.int64)
    Q = v3.predict_shape(S, shape, sigma, pos, cfg)
    conf_meta = meta["conformal"]
    conf = v3.ConformalFit(
        mode=str(conf_meta["mode"]),
        lo50=conf_meta.get("lo50", conf_meta.get("c50", 0.0)),
        hi50=conf_meta.get("hi50", conf_meta.get("c50", 0.0)),
        lo80=conf_meta.get("lo80", conf_meta.get("c80", 0.0)),
        hi80=conf_meta.get("hi80", conf_meta.get("c80", 0.0)),
        half_life_days=conf_meta.get("half_life_days", 30.0),
    )
    Q = v3.apply_conformal(Q, np.asarray(sigma), conf)
    if meta.get("median_fallback"):
        Q = v3.center_quantiles(Q)
    return np.asarray(Q, dtype=np.float64), np.asarray(sigma, dtype=np.float64)


def _eps_from_history(sigma: np.ndarray, decision_ms: np.ndarray,
                      window_start_ms: int) -> tuple[float, str]:
    old = sigma[decision_ms < window_start_ms]
    old = old[np.isfinite(old) & (old > 0)]
    if old.size >= 1_000:
        return max(1e-12, float(np.quantile(old, 0.001))), "history_sigma_p0.1"
    finite = sigma[np.isfinite(sigma) & (sigma > 0)]
    if finite.size == 0:
        return 1e-9, "fallback_1e-9"
    return max(1e-12, float(np.quantile(finite, 0.001))), "loaded_sigma_p0.1"


def build_interval(data_root: Path, interval: str, *, now_ms: int,
                   state_root: Path, force: bool = False,
                   delay_ms: int = live.DEFAULT_LABEL_DELAY_MS,
                   log_fn=print) -> dict:
    art = v3.load_v3(data_root, interval)
    if art is None:
        raise RuntimeError(f"{interval} 没有可用的 v3 部署模型")
    meta = art.meta or {}
    fp = live.artifact_fingerprint(Path(art.directory), model_id=art.model_id, meta=meta)
    symbol = str((meta.get("config") or {}).get("symbol") or "ETHUSDT")
    key = live.StateKey(symbol=symbol, horizon=interval,
                        model_hash=fp["model_hash"], calib_version=live.CALIB_VERSION)
    path = live.state_path(state_root, symbol, interval, fp["model_hash"], live.CALIB_VERSION)
    if path.exists() and not force:
        return {"interval": interval, "status": "skipped_existing_state",
                "state_file": str(path), "model_hash": fp["model_hash"]}

    cfg = v3.default_config(interval, meta["config"])
    window_days = float(v6proto.CALIB_WINDOW_DAYS[interval])
    window_ms = int(round(window_days * 86_400_000))
    window_start = int(now_ms) - window_ms
    step = INTERVAL_MS[interval]

    bar_limit = int(math.ceil(window_ms / step)) + WARMUP_BARS
    one_min_limit = int(math.ceil(window_ms / 60_000)) + ONE_MIN_LOOKBACK
    t0 = time.time()
    S, last = v3._load_inference_samples(data_root, interval, meta["config"], now_ms,
                                         bar_limit=bar_limit,
                                         one_min_limit=one_min_limit)
    Q, sigma = _quantiles_all(meta, art, S, cfg)
    closes = np.asarray(S.closes, dtype=np.float64)
    decision = np.asarray(S.decision_ms, dtype=np.int64)
    n = decision.shape[0]
    y = np.full(n, np.nan)
    contiguous = np.zeros(n, dtype=bool)
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            y[:-1] = np.log(closes[1:] / closes[:-1])
        contiguous[:-1] = (decision[1:] - decision[:-1]) == step

    core = v3.core_indices(S.names, cfg)
    core_ok = np.all(np.isfinite(S.X[:, core]), axis=1)
    maturity = decision + step + int(delay_ms) <= int(now_ms)
    in_window = decision >= window_start
    usable = (in_window & maturity & contiguous & core_ok
              & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
              & np.all(np.isfinite(Q), axis=1))
    rows = np.nonzero(usable)[0]
    if rows.size == 0:
        raise RuntimeError(f"{interval} bootstrap 没有可用样本（窗口 {window_days} 天）")
    if rows.size < v6proto.MIN_CALIB_SAMPLES:
        log_fn(f"[{interval}] 警告：可用样本 {rows.size} < {v6proto.MIN_CALIB_SAMPLES}，"
               "上线后仍会回退 none")

    eps, eps_source = _eps_from_history(sigma, decision, window_start)
    capacity = 2 * int(math.ceil(window_ms / step)) + 128
    eng = calib_v6.CalibrationEngine(
        calib_v6.EngineConfig(candidate="cqr", window_days=window_days,
                              min_samples=v6proto.MIN_CALIB_SAMPLES,
                              delay_ms=int(delay_ms), horizon_ms=step,
                              symbol=symbol, model_hash=fp["model_hash"],
                              calib_version=live.CALIB_VERSION),
        capacity, eps)

    active = 0
    for i in rows:
        d = int(decision[i])
        pid = live.prediction_id(interval, fp["model_hash"], d)
        eng.set_label(pid, float(y[i]))
        out, diag = eng.predict(d, Q[i], float(sigma[i]), pid, d + step)
        active += int(bool(diag.get("active")))
    eng.advance(int(now_ms))
    pending_after = len(eng.pending)

    # Verify the batch replication against the real live inference path on the
    # newest bar (the newest bar itself is not fed into the state).
    verify = {"checked": False}
    try:
        res = v3.predict_latest(data_root, interval, art, now_ms)
        pf = res.get("price_forecast")
        if pf:
            ref = float(pf["ref_close"])
            q_live = np.log(np.asarray([pf["p10"], pf["p25"], pf["median"],
                                        pf["p75"], pf["p90"]], dtype=np.float64) / ref)
            latest = int(np.nonzero(decision == int(pf["decision_ms"]))[0][0])
            diff = float(np.max(np.abs(q_live - Q[latest])))
            verify = {"checked": True, "decision_ms": int(pf["decision_ms"]),
                      "max_abs_diff_logret": diff, "match": bool(diff <= 1e-8)}
            if diff > 1e-6:
                raise RuntimeError(f"{interval} 批量重算与实时推理不一致：max|diff|={diff}")
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        verify = {"checked": False, "error": f"{type(exc).__name__}: {exc}"}

    snapshot_ms = (meta.get("config") or {}).get("snapshot_ms")
    try:
        snapshot_ms = int(snapshot_ms) if snapshot_ms is not None else None
    except (TypeError, ValueError):
        snapshot_ms = None
    overlap = bool(snapshot_ms is not None and window_start < snapshot_ms)
    bootstrap = {
        "bootstrap_source": "deployment_model_historical_replay",
        "generated_at_utc8": _fmt(int(now_ms)),
        "generated_at_ms": int(now_ms),
        "calib_version": live.CALIB_VERSION,
        "model_hash": fp["model_hash"],
        "model_id": fp.get("model_id"),
        "feature_hash": fp.get("feature_hash"),
        "config_sha256": _sha_json(meta.get("config") or {}),
        "window_days": window_days,
        "delay_ms": int(delay_ms),
        "rows_used": int(rows.size),
        "rows_scanned": int(n),
        "span_ms": [int(decision[rows[0]]), int(decision[rows[-1]])],
        "span_utc8": [_fmt(int(decision[rows[0]])), _fmt(int(decision[rows[-1]]))],
        "window_start_ms": int(window_start),
        "window_start_utc8": _fmt(int(window_start)),
        "scores_matured": int(eng.band50.count),
        "active_rows": int(active),
        "pending_after_bootstrap": int(pending_after),
        "eps": float(eps),
        "eps_source": eps_source,
        "in_sample_overlap": overlap,
        "training_snapshot_ms": snapshot_ms,
        "model_trained_at_utc8": meta.get("trained_at_utc8"),
        "fresh_oof": False,
        "strictly_out_of_sample": False,
        "note": ("用当前部署 v5 模型在真实已收盘 K 线上的历史回放；"
                 "与模型训练/留出期重叠，不是新鲜 OOF，校准覆盖不作保证，仅作热启动。"),
        "fallbacks": dict(eng.fallbacks),
        "live_inference_verify": verify,
        "elapsed_s": round(time.time() - t0, 2),
    }

    if path.exists() and force:
        live._quarantine(path, "forced", int(now_ms))
    state = live.LiveState(key=key, path=path, engine=eng, eps=eps,
                           source="bootstrap", bootstrap=bootstrap,
                           created_ms=int(now_ms), updated_ms=int(now_ms),
                           last_reason="ok" if eng.band50.count >= v6proto.MIN_CALIB_SAMPLES else "cold")
    state.save(int(now_ms))
    report = {"interval": interval, "status": "written", "state_file": str(path),
              "model_hash": fp["model_hash"], "model_id": fp.get("model_id"),
              "n_mature": int(eng.band50.count), "rows_used": int(rows.size),
              "active_rows": int(active), "bootstrap": bootstrap}
    report_dir = Path(state_root) / "bootstrap_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(int(now_ms) / 1000))
    (report_dir / f"{interval}-{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="bootstrap live v6 CQR state from the deployed model")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--intervals", nargs="+", default=list(live.CALIB_INTERVALS))
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--now-ms", type=int, default=None)
    ap.add_argument("--delay-ms", type=int, default=live.DEFAULT_LABEL_DELAY_MS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    data_root = Path(args.data_dir).resolve()
    state_root = Path(args.state_dir) if args.state_dir else live.state_root_for(data_root)
    now_ms = int(args.now_ms if args.now_ms is not None else time.time() * 1000)
    out = {}
    rc = 0
    for interval in args.intervals:
        if interval not in live.CALIB_INTERVALS:
            print(f"[{interval}] 策略为 none，跳过 bootstrap", flush=True)
            continue
        try:
            out[interval] = build_interval(data_root, interval, now_ms=now_ms,
                                           state_root=state_root, force=args.force,
                                           delay_ms=args.delay_ms)
            print(f"[{interval}] {out[interval]['status']} "
                  f"n_mature={out[interval].get('n_mature')} "
                  f"rows={out[interval].get('rows_used')}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report all intervals
            rc = 1
            out[interval] = {"interval": interval, "status": "error",
                             "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{interval}] ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float), flush=True)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
