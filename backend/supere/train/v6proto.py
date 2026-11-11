"""v6 first-batch research protocol (stage 0).

Freezes the evaluation protocol *before* any v6 number is produced:

* the deployed v5 model/config/feature hashes per interval (read-only input);
* the fixed HAR comparison per interval (no outer-test selection);
* the median-fallback decision (pre-registered ``False`` for v6, with a
  bridge to the v5 deployment flag);
* purge/embargo semantics of the existing purged walk-forward folds;
* the evaluation schema, primary/secondary metrics and candidate budget;
* the fact that the v5 holdout has already been inspected and is *not* an
  unseen confirmation sample.

Nothing here trains, writes to the shared data directory, or touches a
running service.  All outputs go to an explicit experiment directory outside
the repository and outside ``D:/Tool/cg/superE/data``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from supere.intervals import INTERVAL_MS
from supere.train import quantile as q

PROTOCOL_ID = "v6a-2026-11-11"
EVAL_PROTOCOL = "v6"

PRODUCT_INTERVALS = q.PRODUCT_INTERVALS

# Frozen from the v5 ledger (计划-第六版优化方案 3.2 item 2).  These names are
# fixed a priori; the replay never picks a different baseline on test rows.
HAR_BASELINE = {
    "5m": {"name": "har_cj_t", "sigma": "har_cj", "shape": "t"},
    "15m": {"name": "har_cj_t", "sigma": "har_cj", "shape": "t"},
    "30m": {"name": "har_rs_emp", "sigma": "har_rs", "shape": "emp"},
    "1h": {"name": "har_rs_t", "sigma": "har_rs", "shape": "t"},
    "4h": {"name": "har_rs_t", "sigma": "har_rs", "shape": "t"},
}

# Rolling calibration window (plan 3.2/阶段1 item 2) and the minimum number of
# matured calibration scores before a candidate may deviate from ``none``.
CALIB_WINDOW_DAYS = {"5m": 28.0, "15m": 28.0, "30m": 28.0, "1h": 28.0, "4h": 90.0}
MIN_CALIB_SAMPLES = 200
# The single inner-only longer-window sensitivity (never selected on test).
LONGER_WINDOW_FACTOR = 2.0

# Tail miss targets.  50% interval -> 25% per tail; 80% -> 10% per tail.
TAIL_TARGET = {50: 0.25, 80: 0.10}

GAMMA_CANDIDATES = (0.001, 0.005)
GAMMA_SELECT_FRACTION = 0.40  # earliest 40% of OOF time is the inner selection set

PRIMARY_INTERVAL = "1h"
SECONDARY_INTERVAL = "4h"

# v6 pre-registers median fallback = False for every interval.  The v5
# deployment flag is reported separately and is never used to select anything.
V6_MEDIAN_FALLBACK = False

# Label availability: a decision at the close of bar i knows y at the close of
# bar i+1.  The primary replay uses 0 ms of extra publication delay (the close
# is the publication event); a 60 s delay is a required sensitivity.
LABEL_DELAY_MS = 0
LABEL_DELAY_SENSITIVITY_MS = (0, 60_000)

EVAL_SCHEMA = {
    "return_units": "y = 1e4 * log(close_end / ref_close)  [bps]",
    "pinball": "tau in [0.1,0.25,0.5,0.75,0.9]; (tau - I(y<q)) * (y - q)",
    "pinball_combined_v5": "mean of the five pinball losses (this is what v5 code used)",
    "pinball_equal_mean_v6": "mean of the five pinball losses (identical to v5 combined by construction)",
    "coverage": "cov50 = P(p25 <= y <= p75); cov80 likewise",
    "tails": "left/right miss rates, ideal 25%/25% (50%) and 10%/10% (80%)",
    "width": "w50/w80 in bps and in price units, reported separately",
    "interval_score": "IS_alpha = (U-L) + 2/alpha*(L-y)I(y<L) + 2/alpha*(y-U)I(y>U); alpha=.5/.2",
    "q50": "MAE of the median, reported on its own",
    "qlike": "only where a traceable scale exists; calibration does not change sigma",
    "relative_skill": "per-day loss divided by the decision-time sigma (no future sigma)",
}

PRIMARY_METRICS = [
    "pooled cov50 in [0.48, 0.52]",
    "pooled cov80 in [0.78, 0.82]",
    "day-block 95% CI for coverage",
    "rolling coverage error (28d, or 90d for 4h)",
    "worst valid conditional-cell deviation",
]
SECONDARY_METRICS = [
    "pinball per tau and equal-weight mean",
    "q50 MAE",
    "left/right tail miss",
    "w50/w80",
    "IS50/IS80",
    "QLIKE (scale unchanged; reported once)",
]

CALIB_ACCEPTANCE = {
    "coverage": "cov50 in [0.48,0.52] and cov80 in [0.78,0.82] with day-block 95% CI; sample-poor >= provisional",
    "improvement": ("relative to none on identical raw predictions: rolling coverage error OR "
                    "worst valid conditional-cell deviation improves by >= 20%"),
    "cost": ("pinball worsens by <= 0.3% relative, IS50 by <= 0.5%, mean w50 grows by <= 3%, "
             "coverage still passes"),
    "tie_rule": ("widths are compared only when test coverage differs by <= 1pp; otherwise report "
                 "coverage-width/IS side by side"),
    "none_wins": "if none is already good or no >=20% improvement appears, keep none",
    "significance": ("2000 paired day-block bootstrap + 7-day block sensitivity; DM/HAC auxiliary; "
                     "Holm correction if all five intervals are claimed"),
}

CONDITIONAL_CELLS = {
    "rv_regime": "ex-ante rv_past terciles; thresholds from the prior 90 days only, refreshed daily",
    "weekday": "UTC+8 weekday vs weekend",
    "session": "UTC+8 00-08, 08-16, 16-24",
    "macro": "calendar event falls inside [decision, horizon_end] vs not",
    "diagnose_rule": ">=200 rows and >=30 distinct days before a <=5pp deviation is diagnosed; otherwise evidence insufficient",
}

CANDIDATES = [
    {"id": "none", "desc": "frozen v5 raw quantiles, no calibration"},
    {"id": "cqr", "desc": "rolling two-sided normalized CQR, fixed 75%/90% past score quantiles"},
    {"id": "cqr_aci", "desc": "rolling two-sided normalized CQR + per-tail ACI (gamma inner-selected)"},
]

PEEK_LEDGER = {
    "v5_holdout": ("The newest 15% of v5 samples was evaluated in the v5 round (and in earlier "
                   "development rounds). It is development history only and is NOT treated as an "
                   "unseen confirmation sample in v6."),
    "v6_holdout_peeks": 0,
    "v6_candidate_variants_beyond_registered": 0,
    "v6_extra_intervals_after_primary": 3,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_list(items) -> str:
    return sha256_json(list(items))


def _sha_dir(directory: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(directory.glob("*")) if p.is_file()}


def model_manifest(models_dir: Path) -> dict:
    """Read-only manifest of the deployed v5 artifacts (no secrets involved)."""
    out: dict = {}
    for interval in PRODUCT_INTERVALS:
        d = Path(models_dir) / interval
        meta_path = d / "meta.json"
        feat_path = d / "features.json"
        if not meta_path.exists() or not feat_path.exists():
            out[interval] = {"present": False}
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = json.loads(feat_path.read_text(encoding="utf-8"))
        cfg = meta.get("config", {})
        cfg_sha = sha256_json(cfg)
        # The v5 model_id is a documented function of the sorted config.
        expected_id = f"q3-{interval}-{cfg_sha[:10]}"
        cv = meta.get("cv", {}).get("model", {})
        hold = meta.get("holdout", {}).get("model", {})
        out[interval] = {
            "present": True,
            "model_id": meta.get("model_id"),
            "model_id_recomputed": expected_id,
            "model_id_matches_config_sha": expected_id == meta.get("model_id"),
            "schema_version": meta.get("schema_version"),
            "trained_at_utc8": meta.get("trained_at_utc8"),
            "samples": meta.get("samples"),
            "sample_span_utc8": meta.get("sample_span_utc8"),
            "n_features": meta.get("n_features"),
            "config": cfg,
            "config_sha256": cfg_sha,
            "features_sha256": sha256_file(feat_path),
            "feature_list_sha256": sha256_list(names),
            "file_sha256": _sha_dir(d),
            "median_fallback_v5": bool(meta.get("median_fallback")),
            "median_source_v5": meta.get("median_source"),
            "cv_model": {
                "n": cv.get("n"),
                "pinball_combined_bps": cv.get("pinball_combined_bps"),
                "cov50": cv.get("cov50"),
                "cov80": cv.get("cov80"),
                "w50_bps": cv.get("w50_bps"),
                "mae_bps": cv.get("mae_bps"),
                "mae_zero_bps": cv.get("mae_zero_bps"),
            },
            "holdout_model": {
                "n": hold.get("n"),
                "pinball_combined_bps": hold.get("pinball_combined_bps"),
                "cov50": hold.get("cov50"),
                "cov80": hold.get("cov80"),
            },
        }
    return out


def _db_snapshot(db_path: Path) -> dict:
    p = Path(db_path)
    stat = p.stat()
    return {
        "path": str(p),
        "size_bytes": int(stat.st_size),
        "mtime_utc8": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(stat.st_mtime + 8 * 3600)),
        "note": ("live SQLite; opened read-only (mode=ro). A byte hash of a concurrently written "
                 "database is not stable, so the snapshot is defined by snapshot_ms plus the "
                 "realised first/last decision timestamps recorded in the OOF manifest."),
    }


def build_protocol(models_dir: Path, db_path: Path, data_root: Path,
                   output_root: Path) -> dict:
    manifest = model_manifest(models_dir)
    emb = {}
    for iv in PRODUCT_INTERVALS:
        step = INTERVAL_MS[iv]
        emb[iv] = {
            "v5_embargo_bars": 3,
            "embargo_ms": 3 * step,
            "horizon_ms": step,
            "embargo_covers_horizon": 3 * step >= step,
            "purge_rule": ("a training sample is kept only when its label-end bar index is < "
                           "test-start bar index - embargo; exit_bar = sample bar index + 1"),
        }
    return {
        "eval_protocol": EVAL_PROTOCOL,
        "protocol_id": PROTOCOL_ID,
        "created_utc8": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600)),
        "scope": "stage 0 (freeze) + stage 1 (conditional audit + offline calibration replay) + stage 2A (Deribit availability probe)",
        "not_in_scope": [
            "stage 2B IV feature engineering / download",
            "stage 3 regime/event models",
            "stage 4 RV sub-model",
            "any training of a larger model or a new model family",
            "any deployment, order API, pyautogui, or service restart",
        ],
        "inputs": {
            "shared_data_root_readonly": str(data_root),
            "db": _db_snapshot(db_path),
            "snapshot_ms": q.SNAPSHOT_MS,
            "snapshot_utc8": "2026-11-05 11:00:00",
            "models_quantile_dir": str(models_dir),
            "models_quantile_manifest": manifest,
        },
        "frozen_baselines": {
            "v5_upgrade_control": "the deployed v5 artifacts above (per-interval)",
            "har": HAR_BASELINE,
            "rule": "fixed a priori; no best-of-baseline selection on outer test",
        },
        "median_fallback": {
            "v6_pre_registered": V6_MEDIAN_FALLBACK,
            "v5_deployed_per_interval": {iv: manifest[iv].get("median_fallback_v5") for iv in PRODUCT_INTERVALS},
            "rule": ("v6 fixes fallback=False for all intervals before looking at any v6 row. "
                     "The v5 deployment flag (True for 1h/4h, chosen on the v5 CV test block) is "
                     "reported only as a bridge and is never used to select a v6 candidate."),
        },
        "splits": {
            "order": "train -> inner validation (inside each training block) -> independent calibration (rolling, matured labels only) -> test",
            "cv": "5-fold purged expanding-window walk-forward on the oldest 85% (existing v5 cv_blocks)",
            "holdout": "newest 15%, not read by the v6 OOF export (v5_holdout_peeks=0)",
            "purge_embargo": emb,
            "overlap": "overlapping CV test rows are scored once; day-block bootstrap clusters them",
        },
        "oof_policy": {
            "reuse_if_credible": "no row-level OOF prediction file exists in the shared data directory (verified 2026-11-11); only aggregate CV JSON exists",
            "rebuild": ("minimal isolated refit with the frozen v5 config, original capacity, in the v6 worktree; "
                        "predictions are exported per fold with decision_ms/horizon_end_ms/label_available_ms"),
            "cost_estimate": "v5 bench log: one 5-fold config ~90 s (n_jobs=2, 150 trees); deployed caps are 300/150/120 trees -> estimate < 30 min for five intervals, far below the 6 h stop rule",
            "stop_rule": "if a single rebuild is projected > 6 h, record the gap and stop",
        },
        "evaluation_schema": EVAL_SCHEMA,
        "primary_metrics": PRIMARY_METRICS,
        "secondary_metrics": SECONDARY_METRICS,
        "primary_interval": PRIMARY_INTERVAL,
        "secondary_interval": SECONDARY_INTERVAL,
        "conditional_cells": CONDITIONAL_CELLS,
        "calibration_candidates": CANDIDATES,
        "calibration_windows_days": CALIB_WINDOW_DAYS,
        "min_calib_samples": MIN_CALIB_SAMPLES,
        "tail_targets": TAIL_TARGET,
        "gamma_candidates": list(GAMMA_CANDIDATES),
        "gamma_inner_selection": f"earliest {GAMMA_SELECT_FRACTION:.0%} of OOF time only",
        "longer_window_sensitivity": f"{LONGER_WINDOW_FACTOR}x window, inner only",
        "label_delay_ms": {"primary": LABEL_DELAY_MS, "sensitivity": list(LABEL_DELAY_SENSITIVITY_MS)},
        "acceptance": CALIB_ACCEPTANCE,
        "candidate_budget": {
            "candidates": 3,
            "intervals": 5,
            "gammas_inner_only": 2,
            "longer_windows_inner_only": 1,
            "protocol_amendments": 0,
            "note": "no additional variants will be tried after seeing test rows",
        },
        "peek_ledger": PEEK_LEDGER,
        "old_holdout_fact": ("v5 holdout metrics are quoted in the v5 model meta and DONE; they have "
                             "been seen. v6 makes no claim of a fresh holdout and does not read it."),
        "prospective_confirmation": ("the real confirmation is a future pre-locked 28-day shadow run "
                                     "(90 days for 4h). This batch does not wait for or claim it."),
        "outputs": {
            "root": str(output_root),
            "rule": "all v6 outputs live here; shared data and models_quantile stay read-only",
        },
        "state_keys": ["symbol", "horizon", "model_hash", "calib_version"],
        "calib_version": "v6-calib-1",
    }


def write_protocol(doc: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"protocol already frozen: {path}")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def assert_experiment_output(path: Path) -> Path:
    """Refuse to write anywhere except the dedicated v6 output tree."""
    p = Path(path).resolve()
    root = Path(os.environ.get("SUPERE_V6_OUTPUT_ROOT",
                               r"D:\Tool\cg\superE-v6-output")).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"output must stay under {root}: {p}")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="freeze the v6a research protocol")
    ap.add_argument("--data-root", default=r"D:\Tool\cg\superE\data")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data_root = Path(args.data_root)
    models_dir = Path(args.models_dir) if args.models_dir else data_root / "models_quantile"
    db_path = Path(args.db) if args.db else data_root / "superE.sqlite"
    out = assert_experiment_output(Path(args.out))
    doc = build_protocol(models_dir, db_path, data_root, out.parent)
    path = write_protocol(doc, out)
    print(f"protocol written: {path}")
    print(json.dumps({iv: doc["inputs"]["models_quantile_manifest"][iv].get("model_id_matches_config_sha")
                      for iv in PRODUCT_INTERVALS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
