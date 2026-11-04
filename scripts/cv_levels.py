"""Cross-check the debug trade levels against out-of-sample probabilities. Read-only.

For each interval, the model is fitted the same way as ``supere.train.runner`` (walk-forward
folds, plus two holdout splits for sensitivity). For every out-of-sample decision bar whose
probability clears the threshold, two replays are run on the next H bars:

* ``levels``: entry = decision close, stop = entry -/+ k*ATR14, target = entry +/- k*ATR14
  (the rule shown in the UI, ``predict.trade_levels``), and target 2 = 2k*ATR14 (reference).
* ``labelled``: entry = next open (the training label's entry), same barrier widths.

Stop and target are checked bar by bar; a bar that touches both is counted as a stop (the
conservative tie). A trade with no touch within H bars is closed at the H-th close. Results
are in R units (1R = k*ATR at the decision bar). Also reports the gap between the decision
close and the next open in ATR units, and the always-long / always-short baselines.

Nothing is written to the database and nothing is traded.

Usage:
  python scripts/cv_levels.py --data-dir D:/Tool/cg/superE/data --intervals 5m,15m --out cv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from supere.train import config as tcfg
from supere.train import data, features, labels, metrics, model, splits
from supere.train.predict import direction
from supere.train.runner import _window_has_gap

THRESHOLDS = (0.40, 0.45, 0.50)
# (label, mode, valid_frac or None, n_folds): walk-forward is the main check, holdouts are sensitivity.
SPLITS = (
    ("walk_forward_4", "walk_forward", None, 4),
    ("holdout_0.25", "holdout", 0.25, 4),
    ("holdout_0.20", "holdout", 0.20, 4),
    ("holdout_0.30", "holdout", 0.30, 4),
)


def replay(bars: data.Bars, d: int, sign: float, width: float, entry: float, target_mult: float,
           horizon: int) -> tuple[str, float]:
    """First touch among stop and target within ``horizon`` bars after decision bar ``d``."""
    stop = entry - sign * width
    tgt = entry + sign * target_mult * width
    last = min(d + horizon, bars.n - 1)
    for f in range(d + 1, last + 1):
        hi, lo = bars.h[f], bars.l[f]
        if sign > 0:
            hit_stop, hit_tgt = lo <= stop, hi >= tgt
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= tgt
        if hit_stop:  # same-bar touch of both is a stop: conservative
            return "stop", -1.0
        if hit_tgt:
            return "target", target_mult
    end = bars.c[last]
    return "timeout", float(sign * (end - entry) / width)


def summarize(outcomes: list[tuple[str, float]]) -> dict:
    n = len(outcomes)
    if n == 0:
        return {"n": 0}
    kinds = [k for k, _ in outcomes]
    rs = np.array([r for _, r in outcomes], dtype=np.float64)
    return {
        "n": n,
        "target_rate": kinds.count("target") / n,
        "stop_rate": kinds.count("stop") / n,
        "timeout_rate": kinds.count("timeout") / n,
        "mean_R": float(rs.mean()),
        "sum_R": float(rs.sum()),
    }


def level_stats(bars, lab_idx, atr_all, pos_proba, horizon, k, thresholds):
    """Replays for every threshold. ``pos_proba`` rows align with ``lab_idx`` entries."""
    out = {}
    for thr in thresholds:
        levels_out, labelled_out, tgt2_out = [], [], []
        base_long, base_short = [], []
        for j, d in enumerate(lab_idx):
            p = pos_proba[j]
            probs = {"down": p[0], "flat": p[1], "up": p[2]}
            dirn = direction(probs, thr)
            width = k * atr_all[d]
            if not np.isfinite(width) or width <= 0:
                continue
            ref = bars.c[d]
            nxt = d + 1
            if nxt >= bars.n:
                continue
            nopen = bars.o[nxt]
            # Baselines: always long / always short on the same bars, with the level rule.
            base_long.append(replay(bars, d, 1.0, width, ref, 1.0, horizon))
            base_short.append(replay(bars, d, -1.0, width, ref, 1.0, horizon))
            if dirn == "flat":
                continue
            sign = 1.0 if dirn == "up" else -1.0
            levels_out.append(replay(bars, d, sign, width, ref, 1.0, horizon))
            tgt2_out.append(replay(bars, d, sign, width, ref, 2.0, horizon))
            labelled_out.append(replay(bars, d, sign, width, nopen, 1.0, horizon))
        out[f"{thr:.2f}"] = {
            "levels_t1": summarize(levels_out),
            "levels_t2_reference": summarize(tgt2_out),
            "labelled_entry_t1": summarize(labelled_out),
            "baseline_always_long": summarize(base_long),
            "baseline_always_short": summarize(base_short),
        }
    return out


def gap_stats(bars, idx, atr_all, k):
    gaps = []
    for d in idx:
        nxt = d + 1
        w = k * atr_all[d]
        if nxt < bars.n and np.isfinite(w) and w > 0:
            gaps.append(abs(bars.o[nxt] - bars.c[d]) / w)
    g = np.array(gaps, dtype=np.float64)
    if g.size == 0:
        return {"n": 0}
    return {"n": int(g.size), "mean_abs_gap_in_atr": float(g.mean()),
            "p95_abs_gap_in_atr": float(np.percentile(g, 95)),
            "share_gap_over_0_1_atr": float(np.mean(g > 0.1))}


def run_interval(conn, interval: str, symbol: str, market: str, horizon: int, k: float,
                 model_cfg) -> dict:
    cfg = tcfg.build(interval=interval, file_values={}, overrides={})
    bars = data.load_bars(conn, market, symbol, interval, None)
    funding = data.load_funding(conn, symbol)
    metrics_in = data.load_metrics(conn, symbol)
    names, X_all = features.build(bars, funding, metrics_in)
    lab = labels.build(bars, horizon, k, cfg.label.atr_period)
    gaps = data.gap_after(bars)
    core = [names.index(c) for c in features.CORE_FEATURES]
    x_rows = X_all[lab.idx]
    keep = np.all(np.isfinite(x_rows[:, core]), axis=1) & ~_window_has_gap(gaps, lab.idx, horizon)
    sample_bar = lab.idx[keep]
    X, y = x_rows[keep], lab.y[keep]
    exit_bar = lab.exit_bar[keep]
    atr_all = features.atr(bars, cfg.label.atr_period)
    embargo = horizon

    result = {"interval": interval, "horizon": horizon, "k_atr": k, "bars": bars.n,
              "samples": int(sample_bar.shape[0]), "splits": {}}
    for label, mode, valid_frac, n_folds in SPLITS:
        idx_rows, proba_rows = [], []
        fold_acc = []
        for fold in splits.iter_folds(sample_bar, exit_bar, mode=mode,
                                      valid_frac=valid_frac or 0.25, n_folds=n_folds,
                                      embargo_bars=embargo):
            if fold.train.shape[0] < 50:
                continue
            proba, _ = model.fit_predict(X[fold.train], y[fold.train], X[fold.test], model_cfg)
            proba = np.asarray(proba)
            idx_rows.append(fold.test)
            proba_rows.append(proba)
            fold_acc.append(float(np.mean(np.array([-1, 0, 1])[np.argmax(proba, axis=1)] == y[fold.test])))
        if not idx_rows:
            result["splits"][label] = {"error": "没有任何一折能训练"}
            continue
        pos = np.concatenate(idx_rows)
        proba_o = np.vstack(proba_rows)
        cls = metrics.classification(y[pos], proba_o)
        entry = {
            "oos_samples": int(pos.shape[0]),
            "oos_accuracy": cls["accuracy"],
            "oos_log_loss": cls["log_loss"],
            "oos_auc_ovr_macro": cls["auc_ovr_macro"],
            "fold_accuracy": fold_acc,
            "levels": level_stats(bars, sample_bar[pos], atr_all, proba_o, horizon, k, THRESHOLDS),
            "gap": gap_stats(bars, sample_bar[pos], atr_all, k),
        }
        result["splits"][label] = entry
        print(f"{interval} {label}: oos n={pos.shape[0]} acc={cls['accuracy']:.3f}", file=sys.stderr, flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="levels cross-check (read-only)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--intervals", default="5m,15m,30m,1h,4h")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    db = Path(args.data_dir) / "superE.sqlite"
    conn = data.open_readonly(db)
    out = {"symbol": "ETHUSDT", "market": "um", "thresholds": list(THRESHOLDS), "results": []}
    try:
        for interval in [s.strip() for s in args.intervals.split(",") if s.strip()]:
            horizon, k = tcfg.DEFAULT_LABEL[interval]
            model_cfg = tcfg.ModelConfig()
            out["results"].append(run_interval(conn, interval, "ETHUSDT", "um", horizon, k, model_cfg))
            Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
