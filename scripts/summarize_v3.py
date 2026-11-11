"""Summarise v3 CV JSON files into a compact table (for reports and selection)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTERVALS = ("5m", "15m", "30m", "1h", "4h")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="summarize_v3")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.data_dir) / "cv_quantile_v3" / args.round
    rows = []
    for path in sorted(root.glob("*__*.json")):
        if path.name.endswith("__summary.json") or path.name.endswith(".holdout.json"):
            continue
        interval, label = path.stem.split("__", 1)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "model" not in d:
            continue
        m = d["model"]
        g = d.get("gates") or d.get("gates_single_block") or {}
        b = d.get("baselines", {}).get(d.get("best_baseline"), {})
        rows.append({
            "interval": interval, "label": label,
            "pin": m.get("pinball_combined_bps"), "cov50": m.get("cov50"),
            "cov80": m.get("cov80"), "w50": m.get("w50_bps"), "w80": m.get("w80_bps"),
            "mae": m.get("mae_bps"), "mae_zero": m.get("mae_zero_bps"),
            "best_base": d.get("best_baseline"), "base_pin": b.get("pinball_combined_bps"),
            "improve": g.get("pinball_improve_vs_best_base"),
            "main": g.get("main_pass"), "secondary": g.get("secondary_pass"),
            "median_fallback": d.get("median_fallback"),
        })
    if args.markdown:
        print("| 周期 | 配置 | pinball | cov50 | cov80 | w50 | w80 | 最佳基线 | 提升 | 主/次 | median回退 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['interval']} | {r['label']} | {r['pin']} | {r['cov50']} | {r['cov80']} | "
                  f"{r['w50']} | {r['w80']} | {r['best_base']} {r['base_pin']} | {r['improve']} | "
                  f"{r['main']}/{r['secondary']} | {r['median_fallback']} |")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
