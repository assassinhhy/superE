"""Run a list of v3 configurations for one interval (round driver).

The per-interval process lets us run 5m/15m/30m/1h/4h in parallel while keeping
the same purged folds.  Each config is evaluated independently; failures are
logged and do not stop the rest of the round.

    python scripts/run_v3_configs.py --data-dir ... --interval 5m \
        --round r1 --configs D:/.../r1.json --n-jobs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from supere.train import quantile as q
from supere.train import quantile_v3 as v3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_v3_configs")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--configs", required=True, help="JSON list of {label, set:{...}}")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--mode", choices=["cv", "holdout"], default="cv")
    args = parser.parse_args(argv)
    data_root = Path(args.data_dir).resolve()
    db_path = data_root / "superE.sqlite"
    configs = json.loads(Path(args.configs).read_text(encoding="utf-8"))
    out_dir = data_root / "cv_quantile_v3" / args.round
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.interval}.log"
    summary = {}
    for item in configs:
        label = str(item["label"])
        overrides = dict(item.get("set") or {})
        overrides["n_jobs"] = args.n_jobs
        try:
            cfg = v3.default_config(args.interval, overrides)
            started = datetime.now().strftime("%H:%M:%S")
            line = f"[{started}] {args.interval} {label} start cfg={json.dumps(v3.asdict(cfg), ensure_ascii=False)}"
            print(line, flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            S = v3.load_samples(cfg, db_path, data_root)
            base = q.baseline_quantiles(S, v3._qcfg(cfg))

            def log_fn(msg: str) -> None:
                print(msg, flush=True)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")

            if args.mode == "cv":
                res = v3.run_cv(S, cfg, base, log_fn)
            else:
                cv_path = out_dir / f"{args.interval}__best.json"
                mf = False
                if cv_path.exists():
                    mf = bool(json.loads(cv_path.read_text(encoding="utf-8")).get("median_fallback"))
                res = v3.run_holdout(S, cfg, base, mf)
            res["label"] = label
            (out_dir / f"{args.interval}__{label}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            summary[label] = {"model": res["model"], "gates": res.get("gates") or res.get("gates_single_block"),
                              "best_baseline": res.get("best_baseline")}
            done = (f"[{datetime.now().strftime('%H:%M:%S')}] {args.interval} {label} "
                    f"pin={res['model']['pinball_combined_bps']} cov50={res['model']['cov50']} "
                    f"w50={res['model']['w50_bps']}")
            print(done, flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(done + "\n")
        except Exception as exc:  # noqa: BLE001 - one config must not kill the round
            msg = f"[{datetime.now().strftime('%H:%M:%S')}] {args.interval} {label} ERROR {type(exc).__name__}: {exc}"
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
            summary[label] = {"error": f"{type(exc).__name__}: {exc}"}
    (out_dir / f"{args.interval}__summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
