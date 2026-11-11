"""Row-level out-of-fold (OOF) export for the frozen v5 configuration (stage 0).

The shared data directory contains only *aggregate* CV JSON for v5: no
row-level predictions, so the v6 calibration replay cannot reuse them.  This
module performs the minimal isolated rebuild allowed by the plan:

* it reads the deployed ``meta.json`` config verbatim (never re-tunes it);
* it reuses the existing purged walk-forward blocks
  (:func:`supere.train.quantile.cv_blocks`), the existing fold fit and the
  existing prediction code from ``quantile_v3``;
* it writes per-row predictions, labels, decision times and the frozen HAR
  comparison to a parquet file in the explicit v6 output tree.

Nothing is written to the shared database or to ``models_quantile``.  The v5
holdout is never predicted here.  The export is a *rebuild*, not a new model:
capacity, features, seed and folds are the deployed ones.

CLI::

    python -m supere.train.oof_v6 --intervals 5m 1h --out <run_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import quantile as q
from supere.train import quantile_v3 as v3
from supere.train import v6proto

COLUMNS = [
    "interval", "fold", "sample_pos", "bar_idx", "decision_ms", "horizon_end_ms",
    "y_logret", "ref_close", "close_end", "sigma_model", "sigma_base", "eps_train",
    "q10", "q25", "q50", "q75", "q90",
    "har_q10", "har_q25", "har_q50", "har_q75", "har_q90",
    "rv_past", "rv_future", "model_hash", "feature_hash", "config_sha256",
]


def deployed_config(interval: str, models_dir: Path) -> tuple[v3.V3Config, dict]:
    """Load the deployed config exactly as stored in ``meta.json``."""
    meta = json.loads((Path(models_dir) / interval / "meta.json").read_text(encoding="utf-8"))
    stored = dict(meta["config"])
    overrides = {k: v for k, v in stored.items() if k != "interval"}
    cfg = v3.default_config(interval, overrides)
    if asdict(cfg) != stored:
        diff = {k: (stored.get(k), asdict(cfg).get(k))
                for k in set(stored) | set(asdict(cfg)) if stored.get(k) != asdict(cfg).get(k)}
        raise ValueError(f"{interval}: deployed config does not round-trip: {diff}")
    cfg_sha = v6proto.sha256_json(stored)
    expected_id = f"q3-{interval}-{cfg_sha[:10]}"
    if meta.get("model_id") != expected_id:
        raise ValueError(f"{interval}: model_id {meta.get('model_id')} != {expected_id}")
    return cfg, meta


def _fold_model_hash(cfg: v3.V3Config, fit: v3.V3Fit, blk: q.Block) -> str:
    h = hashlib.sha256()
    h.update(v6proto.sha256_json(asdict(cfg)).encode("ascii"))
    h.update(str(blk.number).encode("ascii"))
    h.update(str(int(blk.train[0])).encode("ascii"))
    h.update(str(int(blk.train[-1])).encode("ascii"))
    h.update(str(int(blk.test[0])).encode("ascii"))
    h.update(str(int(blk.test[-1])).encode("ascii"))
    if fit.shape.boosters:
        for booster in fit.shape.boosters:
            h.update(booster.model_to_string().encode("utf-8"))
    if fit.sigma.coef is not None:
        h.update(np.asarray(fit.sigma.coef, dtype=np.float64).tobytes())
        h.update(np.asarray([fit.sigma.intercept], dtype=np.float64).tobytes())
        h.update(str(fit.sigma.kind).encode("ascii"))
    return h.hexdigest()


def _har_quantiles(S: v3.SamplesV3, blk: q.Block, cfg: v3.V3Config,
                   spec: dict) -> np.ndarray:
    fit = v3.fit_sigma(S, blk.train, cfg, kind=spec["sigma"])
    sigma = v3.predict_sigma(S, fit, cfg)
    return v3._baseline_from_sigma(S, sigma, blk.train, blk.test, spec["shape"], cfg)


def build_interval(interval: str, models_dir: Path, data_root: Path, db_path: Path,
                   log_fn=print) -> tuple[object, dict]:
    """Return a per-row DataFrame plus a manifest for one interval."""
    import pandas as pd

    t0 = time.time()
    cfg, meta = deployed_config(interval, models_dir)
    cfg_sha = v6proto.sha256_json(meta["config"])
    log_fn(f"[{interval}] loading samples (snapshot={cfg.snapshot_ms})")
    S = v3.load_samples(cfg, Path(db_path), Path(data_root))
    feature_hash = v6proto.sha256_list(S.names)
    expected_feat = v6proto.sha256_list(json.loads(
        (Path(models_dir) / interval / "features.json").read_text(encoding="utf-8")))
    feat_matches = feature_hash == expected_feat
    if not feat_matches:
        raise ValueError(f"{interval}: reconstructed feature schema hash mismatch")
    log_fn(f"[{interval}] samples={S.m} bars={S.bars_n} features={len(S.names)} "
           f"span={q._fmt(int(S.decision_ms[0]))}..{q._fmt(int(S.decision_ms[-1]))}")

    qc = v3._qcfg(cfg)
    folds, hold = q.cv_blocks(S, qc)
    spec = v6proto.HAR_BASELINE[interval]
    step = INTERVAL_MS[interval]
    frames = []
    fold_meta = []
    pooled_Q, pooled_pos, pooled_base = [], [], []
    for blk in folds:
        tf = time.time()
        fit = v3.fit_block(S, blk.train, cfg)
        sigma_all = v3.predict_sigma(S, fit.sigma, cfg)
        Q = v3.predict_block(S, fit, blk.test, cfg)
        Qb = _har_quantiles(S, blk, cfg, spec)
        rows = S.idx[blk.test]
        if np.any(rows + 1 >= S.closes.shape[0]):
            raise ValueError(f"{interval}: label bar out of range in fold {blk.number}")
        y = S.y[blk.test]
        close_end = S.closes[rows + 1]
        # The stored label must be exactly the close-to-close log return.
        implied = np.log(close_end / S.ref_close[blk.test])
        if not np.allclose(implied, y, rtol=0, atol=1e-12):
            raise ValueError(f"{interval}: label does not match close ratio in fold {blk.number}")
        model_hash = _fold_model_hash(cfg, fit, blk)
        # eps for the normalized conformal score is fixed by *training* sigma,
        # never by test rows.  Store it per fold so the replay cannot peek.
        eps_train = max(1e-12, float(np.quantile(sigma_all[S.idx[blk.train]], 0.001)))
        df = pd.DataFrame({
            "interval": interval,
            "fold": blk.number,
            "sample_pos": blk.test.astype(np.int64),
            "bar_idx": rows.astype(np.int64),
            "decision_ms": S.decision_ms[blk.test].astype(np.int64),
            "horizon_end_ms": (S.decision_ms[blk.test] + step).astype(np.int64),
            "y_logret": y,
            "ref_close": S.ref_close[blk.test],
            "close_end": close_end,
            "sigma_model": sigma_all[rows],
            "sigma_base": S.sig_base[blk.test],
            "eps_train": np.full(blk.test.shape[0], eps_train),
            "q10": Q[:, 0], "q25": Q[:, 1], "q50": Q[:, 2], "q75": Q[:, 3], "q90": Q[:, 4],
            "har_q10": Qb[:, 0], "har_q25": Qb[:, 1], "har_q50": Qb[:, 2],
            "har_q75": Qb[:, 3], "har_q90": Qb[:, 4],
            "rv_past": S.rvf.rv_past[rows],
            "rv_future": S.rvf.rv_future[rows],
            "model_hash": model_hash,
            "feature_hash": feature_hash,
            "config_sha256": cfg_sha,
        })
        frames.append(df)
        pooled_Q.append(Q)
        pooled_pos.append(blk.test)
        pooled_base.append(Qb)
        fold_meta.append({
            "fold": blk.number,
            "n_train": int(fit.n_train),
            "n_calib_internal": int(fit.n_calib),
            "n_test": int(blk.test.shape[0]),
            "train_span_utc8": [q._fmt(int(S.decision_ms[blk.train[0]])),
                                q._fmt(int(S.decision_ms[blk.train[-1]]))],
            "test_span_utc8": [q._fmt(int(df.decision_ms.iloc[0])),
                               q._fmt(int(df.decision_ms.iloc[-1]))],
            "sigma_kind": fit.sigma.kind,
            "sigma_qlike": fit.sigma.qlike,
            "eps_train": eps_train,
            "shape_kind": fit.shape.kind,
            "model_hash": model_hash,
            "elapsed_s": round(time.time() - tf, 1),
        })
        log_fn(f"[{interval}] fold {blk.number}: train={fit.n_train} test={blk.test.shape[0]} "
               f"qlike={fit.sigma.qlike} {fold_meta[-1]['elapsed_s']}s")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("decision_ms", kind="stable").reset_index(drop=True)

    pos = np.concatenate(pooled_pos)
    Q = np.vstack(pooled_Q)
    Qb = np.vstack(pooled_base)
    y = S.y[pos]
    raw = v3.score_v3(y, Q, S.sig_base[pos])
    centered = v3.score_v3(y, v3.center_quantiles(Q), S.sig_base[pos])
    base = v3.score_v3(y, Qb, S.sig_base[pos])
    meta_cv = meta.get("cv", {}).get("model", {})
    # The v5 CV number includes median fallback when the deployed flag is True.
    v5_candidate = centered if meta.get("median_fallback") else raw
    match = (abs(v5_candidate["pinball_combined_bps"] - meta_cv.get("pinball_combined_bps", np.nan)) < 5e-4
             and abs(v5_candidate["cov50"] - meta_cv.get("cov50", np.nan)) < 5e-4
             and abs(v5_candidate["cov80"] - meta_cv.get("cov80", np.nan)) < 5e-4)
    manifest = {
        "interval": interval,
        "n_rows": int(df.shape[0]),
        "n_folds": len(folds),
        "samples_total": int(S.m),
        "samples_total_from_meta": meta.get("samples"),
        "span_utc8": [q._fmt(int(df.decision_ms.iloc[0])), q._fmt(int(df.decision_ms.iloc[-1]))],
        "config": meta["config"],
        "config_sha256": cfg_sha,
        "feature_hash": feature_hash,
        "feature_hash_matches_deployed": feat_matches,
        "model_files_sha256": {p.name: v6proto.sha256_file(p)
                               for p in sorted((Path(models_dir) / interval).glob("*")) if p.is_file()},
        "har_baseline": spec,
        "folds": fold_meta,
        "metrics_raw_no_fallback": raw,
        "metrics_centered_v5_fallback": centered,
        "metrics_frozen_har": base,
        "meta_cv_model": meta_cv,
        "reproduces_v5_cv": bool(match),
        "v5_cv_reproduced_with": "centered" if meta.get("median_fallback") else "raw",
        "v5_deployed_median_fallback": bool(meta.get("median_fallback")),
        "holdout_read": False,
        "elapsed_s": round(time.time() - t0, 1),
        "rows_sha256": hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest(),
    }
    return df, manifest


def export(intervals: list[str], models_dir: Path, data_root: Path, db_path: Path,
           out_dir: Path, log_fn=print) -> Path:
    out = v6proto.assert_experiment_output(Path(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    all_frames = []
    for interval in intervals:
        df, manifest = build_interval(interval, models_dir, data_root, db_path, log_fn=log_fn)
        path = out / f"oof_{interval}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        all_frames.append(df)
        manifests[interval] = manifest
        log_fn(f"[{interval}] wrote {path} rows={len(df)} reproduces_v5_cv={manifest['reproduces_v5_cv']}")
    manifest_path = out / "oof_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old.update(manifests)
        manifests = old
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_oof(run_dir: Path, interval: str = None):
    import pandas as pd
    run_dir = Path(run_dir)
    if interval is not None:
        return pd.read_parquet(run_dir / f"oof_{interval}.parquet")
    frames = [pd.read_parquet(p) for p in sorted(run_dir.glob("oof_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="export row-level v5 OOF predictions")
    ap.add_argument("--intervals", nargs="+", default=list(v6proto.PRODUCT_INTERVALS))
    ap.add_argument("--data-root", default=r"D:\Tool\cg\superE\data")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data_root = Path(args.data_root)
    models_dir = Path(args.models_dir) if args.models_dir else data_root / "models_quantile"
    db_path = Path(args.db) if args.db else data_root / "superE.sqlite"
    path = export(args.intervals, models_dir, data_root, db_path, Path(args.out))
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
