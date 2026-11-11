"""Final v3 CV -> holdout -> fit for one interval, using one selected config.

The holdout is evaluated once.  Selection of the config was made on the CV
rounds; this script does not tune on holdout.  The final artifact is saved to
``data/models_quantile/<interval>`` with schema_version=3.

    python scripts/finalize_v3.py --data-dir ... --interval 1h \
        --round r6 --config '{"label":"final","set":{...}}'
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from supere.train import quantile as q
from supere.train import quantile_v3 as v3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finalize_v3")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--round", default="r6")
    parser.add_argument("--config", required=True, help="JSON {label,set}")
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--reuse-cv", action="store_true", default=True)
    parser.add_argument("--no-reuse-cv", dest="reuse_cv", action="store_false")
    args = parser.parse_args(argv)
    root = Path(args.data_dir).resolve()
    db = root / "superE.sqlite"
    out = root / "cv_quantile_v3" / args.round
    out.mkdir(parents=True, exist_ok=True)
    item = json.loads(args.config)
    overrides = dict(item.get("set") or {})
    overrides["n_jobs"] = args.n_jobs
    cfg = v3.default_config(args.interval, overrides)

    def log_fn(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(out / f"{args.interval}.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    S = v3.load_samples(cfg, db, root)
    log_fn(f"final {args.interval} samples={S.m} features={len(S.names)} cfg={json.dumps(v3.asdict(cfg), ensure_ascii=False)}")
    base = q.baseline_quantiles(S, v3._qcfg(cfg))
    cv_path = out / f"{args.interval}.json"
    cv = None
    if args.reuse_cv and cv_path.exists():
        try:
            old = json.loads(cv_path.read_text(encoding="utf-8"))
            old_cfg = dict(old.get("config") or {})
            new_cfg = dict(v3.asdict(cfg))
            old_cfg.pop("n_jobs", None)
            new_cfg.pop("n_jobs", None)
            if old_cfg == new_cfg:
                cv = old
                log_fn("reusing matching CV result")
        except Exception:
            cv = None
    if cv is None:
        cv = v3.run_cv(S, cfg, base, log_fn)
        cv_path.write_text(json.dumps(cv, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    log_fn(f"cv pin={cv['model']['pinball_combined_bps']} cov50={cv['model']['cov50']} "
           f"w50={cv['model']['w50_bps']} gates={cv['gates']}")
    hold = v3.run_holdout(S, cfg, base, bool(cv.get("median_fallback")))
    (out / f"{args.interval}.holdout.json").write_text(
        json.dumps(hold, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    log_fn(f"holdout pin={hold['model']['pinball_combined_bps']} "
           f"cov50={hold['model']['cov50']} cov80={hold['model']['cov80']} "
           f"w50={hold['model']['w50_bps']} best_base={hold['best_baseline']}")
    fit = v3.fit_block(S, np.arange(S.m, dtype=np.int64), cfg)
    extra = v3._round_verdict(out / f"{args.interval}.json", out / f"{args.interval}.holdout.json")
    directory = v3.save_artifacts(root, S, cfg, fit, extra,
                                  bool(cv.get("median_fallback")))
    log_fn(f"saved {directory} status={extra.get('status')} sigma={fit.sigma.kind} "
           f"shape={fit.shape.kind} conformal={fit.conformal.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
