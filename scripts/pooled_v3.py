"""Pooled five-horizon model ablation (stage 4).

One quantile model is trained on the union of 5m/15m/30m/1h/4h samples, with
``log(h)`` as a feature.  Folds are purged by each sample's own horizon:
training rows must have finished (``decision + horizon``) before the test block
starts.  Metrics are still reported per horizon.

This is a comparison, not the production model.  It uses a common feature
subset so the design matrix is identical across horizons.

    python scripts/pooled_v3.py --data-dir ... --round r4 --max-per-interval 30000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from supere.intervals import INTERVAL_MS
from supere.train import quantile as q
from supere.train import quantile_v3 as v3

FIXED_RV = []
for w in ("60m", "360m", "1440m", "10080m"):
    FIXED_RV += [f"rv_{w}", f"logrv_{w}", f"bpv_{w}", f"jump_{w}", f"rsp_{w}", f"rsn_{w}",
                 f"park_{w}", f"gk_{w}"]
FIXED_RV += ["rv_z_1d", "seas_hour_med_28d", "seas_dow_med_4w", "seas_ratio_hour",
             "seas_ratio_dow", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
for w in ("60m", "360m", "1440m"):
    FIXED_RV += [f"micro_taker_imb_{w}", f"micro_quote_imb_{w}", f"micro_log_trades_{w}",
                 f"micro_log_avg_trade_{w}", f"micro_trades_z_{w}", f"micro_volume_z_{w}"]


def _samples_for(data_root: Path, interval: str, max_per_interval: int):
    cfg = v3.default_config(interval, {"features": "all", "sigma_model": "ewma",
                                       "shape": "quantile", "conformal": "none",
                                       "n_estimators": 100, "n_jobs": 2})
    S = v3.load_samples(cfg, data_root / "superE.sqlite", data_root)
    names = list(S.base_names)
    idx = {n: i for i, n in enumerate(S.names)}
    for n in FIXED_RV:
        if n in idx and n not in names:
            names.append(n)
    X = np.column_stack([S.X[:, idx[n]] for n in names])
    keep = np.arange(S.m)
    if max_per_interval and keep.shape[0] > max_per_interval:
        keep = np.linspace(0, keep.shape[0] - 1, max_per_interval).astype(np.int64)
    return {
        "interval": interval, "names": names, "X": X[keep], "y": S.y[keep],
        "sig": S.sig_base[keep], "decision": S.decision_ms[keep],
        "exit": S.decision_ms[keep] + INTERVAL_MS[interval],
        "pos": keep, "S": S, "cfg": cfg,
        "base": q.baseline_quantiles(S, v3._qcfg(cfg)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pooled_v3")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--round", default="r4")
    parser.add_argument("--max-per-interval", type=int, default=30000)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args(argv)
    root = Path(args.data_dir).resolve()
    per = [_samples_for(root, i, args.max_per_interval) for i in v3.PRODUCT_INTERVALS]
    common = set(per[0]["names"])
    for p in per[1:]:
        common &= set(p["names"])
    common_names = [n for n in per[0]["names"] if n in common]
    print(f"common features {len(common_names)}", flush=True)

    Xs, ys, sigs, decs, exits, groups = [], [], [], [], [], []
    offsets = []
    start = 0
    for k, p in enumerate(per):
        idx = [p["names"].index(n) for n in common_names]
        X = p["X"][:, idx]
        logh = np.full((X.shape[0], 1), np.log(INTERVAL_MS[p["interval"]] / 60_000.0))
        Xs.append(np.column_stack([X, logh]))
        ys.append(p["y"])
        sigs.append(p["sig"])
        decs.append(p["decision"])
        exits.append(p["exit"])
        groups.append(np.full(X.shape[0], k, dtype=np.int64))
        offsets.append((start, start + X.shape[0]))
        start += X.shape[0]
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    sig = np.concatenate(sigs)
    dec = np.concatenate(decs)
    exit_ms = np.concatenate(exits)
    group = np.concatenate(groups)
    order = np.argsort(dec, kind="stable")
    X, y, sig, dec, exit_ms, group = X[order], y[order], sig[order], dec[order], exit_ms[order], group[order]
    print(f"pooled samples {X.shape[0]}", flush=True)

    cut = int(round(X.shape[0] * 0.85))
    edges = np.quantile(dec[:cut], np.linspace(0.2, 1.0, 6))
    cfg = v3.default_config("5m", {"n_estimators": args.n_estimators, "n_jobs": 4})
    fold_scores = {p["interval"]: [] for p in per}
    pooled = {p["interval"]: ([], []) for p in per}
    for k in range(len(edges) - 1):
        t0, t1 = edges[k], edges[k + 1]
        train = (dec < t0) & (exit_ms <= t0)
        test = (dec >= t0) & (dec < t1)
        if train.sum() < 1000 or test.sum() < 50:
            continue
        boosters = q._fit_quantiles(X[train], y[train] / sig[train], cfg)
        Qz = q.predict_raw(boosters, X[test])
        Q = q.sort_rows(Qz * sig[test][:, None])
        for gi, p in enumerate(per):
            mask = test & (group == gi)
            if mask.sum() < 20:
                continue
            sc = v3.score_v3(y[mask], Q[mask[test]], sig[mask])
            sc["fold"] = k
            fold_scores[p["interval"]].append(sc)
            pooled[p["interval"]][0].append(y[mask])
            pooled[p["interval"]][1].append(Q[mask[test]])
        print(f"fold {k}: train={int(train.sum())} test={int(test.sum())}", flush=True)

    out = {"common_features": len(common_names), "samples": int(X.shape[0]),
           "intervals": {}, "note": "pooled model with log(h); per-horizon metrics on the same pooled test rows"}
    for p in per:
        if not pooled[p["interval"]][0]:
            continue
        yy = np.concatenate(pooled[p["interval"]][0])
        QQ = np.vstack(pooled[p["interval"]][1])
        model = v3.score_v3(yy, QQ, np.ones_like(yy))
        out["intervals"][p["interval"]] = {
            "model": model, "folds": fold_scores[p["interval"]],
        }
    out_dir = root / "cv_quantile_v3" / args.round
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pooled.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
