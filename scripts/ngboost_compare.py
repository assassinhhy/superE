"""NGBoost distribution-GBM comparison (stage 4).

LightGBMLSS needs torch and was not installed; NGBoost is installed but its
Student-T distribution overflowed during a smoke test, so this script records
that failure and runs the stable Normal distribution as the distribution-GBM
comparison.  It trains on a subsample, predicts a full Normal distribution for
each test row, and scores the same quantiles as the main pipeline.

    python scripts/ngboost_compare.py --data-dir ... --intervals 1h,4h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except ImportError:  # optional comparison dependency
    print("NGBoost is not installed; this comparison is optional and was skipped.")
    raise SystemExit(2)
from sklearn.tree import DecisionTreeRegressor
from scipy import stats

from supere.train import quantile as q
from supere.train import quantile_v3 as v3


def run_interval(data_root: Path, interval: str, *, max_train: int, n_estimators: int) -> dict:
    overrides = {"features": "rv", "sigma_model": "har", "shape": "quantile",
                 "normalize": True, "conformal": "none", "n_estimators": 50, "n_jobs": 2}
    cfg = v3.default_config(interval, overrides)
    S = v3.load_samples(cfg, data_root / "superE.sqlite", data_root)
    folds, _ = q.cv_blocks(S, v3._qcfg(cfg))
    pooled_y, pooled_Q = [], []
    fold_out = []
    student_t_error = ("NGBoost 0.5.11 StudentT overflowed (exp(df) inf) and produced "
                       "NaN gradients on a smoke test; Normal used instead")
    for blk in folds:
        sigma_fit = v3.fit_sigma(S, blk.train, cfg)
        sigma = v3.predict_sigma(S, sigma_fit, cfg)
        z = S.y[blk.train] / sigma[S.idx[blk.train]]
        ok = np.isfinite(z) & (np.abs(z) < 50)
        train = blk.train[ok]
        if train.shape[0] > max_train:
            pick = np.random.default_rng(42 + blk.number).choice(train.shape[0], max_train, replace=False)
            train = train[np.sort(pick)]
        X = np.nan_to_num(S.X[train], nan=0.0, posinf=0.0, neginf=0.0)
        y = S.y[train] / sigma[S.idx[train]]
        X_test = np.nan_to_num(S.X[blk.test], nan=0.0, posinf=0.0, neginf=0.0)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
        model = NGBRegressor(
            Dist=Normal, Base=DecisionTreeRegressor(max_depth=3),
            n_estimators=n_estimators, learning_rate=0.05,
            minibatch_frac=0.5, col_sample=0.5, verbose=False,
            random_state=42,
        )
        model.fit(X, y)
        dist = model.pred_dist(X_test)
        alpha = np.asarray(v3.ALPHAS)
        Qz = np.column_stack([dist.ppf(a) for a in alpha])
        Q = q.sort_rows(Qz * sigma[S.idx[blk.test]][:, None])
        sc = v3.score_v3(S.y[blk.test], Q, S.sig_base[blk.test])
        sc["fold"] = blk.number
        fold_out.append(sc)
        pooled_y.append(S.y[blk.test])
        pooled_Q.append(Q)
    y = np.concatenate(pooled_y)
    Q = np.vstack(pooled_Q)
    model_score = v3.score_v3(y, Q, S.sig_base[np.concatenate([b.test for b in folds])])
    return {"interval": interval, "ngboost_dist": "Normal", "folds": fold_out,
            "model": model_score, "student_t_error": student_t_error,
            "n_train_cap": max_train, "n_estimators": n_estimators}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ngboost_compare")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--intervals", default="1h,4h")
    parser.add_argument("--max-train", type=int, default=15000)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args(argv)
    data_root = Path(args.data_dir).resolve()
    out_dir = data_root / "cv_quantile_v3" / "ngboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for interval in [x.strip() for x in args.intervals.split(",") if x.strip()]:
        try:
            out[interval] = run_interval(data_root, interval, max_train=args.max_train,
                                         n_estimators=args.n_estimators)
            print(f"{interval}: pin={out[interval]['model']['pinball_combined_bps']} "
                  f"cov50={out[interval]['model']['cov50']} w50={out[interval]['model']['w50_bps']}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            out[interval] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{interval}: ERROR {type(exc).__name__}: {exc}", flush=True)
    (out_dir / "ngboost.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
