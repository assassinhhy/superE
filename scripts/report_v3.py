"""Generate the Markdown metric tables used by the v3 DONE report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = {"5m": 0, "15m": 1, "30m": 2, "1h": 3, "4h": 4}


def load_rows(root: Path, round_name: str) -> list[dict]:
    out = []
    for path in sorted((root / "cv_quantile_v3" / round_name).glob("*__*.json")):
        if path.name.endswith("__summary.json") or ".holdout." in path.name:
            continue
        interval, label = path.stem.split("__", 1)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "model" not in d:
            continue
        m = d["model"]
        g = d.get("gates") or {}
        b = d.get("baselines", {}).get(d.get("best_baseline"), {})
        out.append({"interval": interval, "label": label, "m": m, "b": b,
                    "best": d.get("best_baseline"), "g": g,
                    "fallback": d.get("median_fallback")})
    return sorted(out, key=lambda r: (ORDER.get(r["interval"], 9), r["label"]))


def abl_table(root: Path, round_name: str, title: str) -> str:
    rows = load_rows(root, round_name)
    if not rows:
        return f"### {title}\n\n（无结果）\n"
    lines = [f"### {title}", "",
             "| 周期 | 配置 | pinball(bps) | cov50 | cov80 | w50(bps) | w80(bps) | 最佳基线 | 基线pinball | 提升 | median回退 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, b, g = r["m"], r["b"], r["g"]
        lines.append(
            f"| {r['interval']} | {r['label']} | {m.get('pinball_combined_bps')} | "
            f"{m.get('cov50')} | {m.get('cov80')} | {m.get('w50_bps')} | {m.get('w80_bps')} | "
            f"{r['best']} | {b.get('pinball_combined_bps')} | "
            f"{g.get('pinball_improve_vs_best_base')} | {r['fallback']} |")
    return "\n".join(lines) + "\n"


def final_tables(root: Path) -> str:
    out = ["### 最终 CV（pooled，五折 purged walk-forward）", "",
           "| 周期 | n | pinball | cov50 | cov80 | w50 | w80 | 最佳基线 | 基线pinball | 提升 | MAE(bps) | 当前价MAE | median回退 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ho = ["### Holdout（最后 15%，只跑一次）", "",
          "| 周期 | n | pinball | cov50 | cov80 | w50 | 基线pinball | 基线w50 | 提升 | MAE(bps) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for interval in ("5m", "15m", "30m", "1h", "4h"):
        cv = json.loads((root / "cv_quantile_v3" / "r6" / f"{interval}.json").read_text(encoding="utf-8"))
        h = json.loads((root / "cv_quantile_v3" / "r6" / f"{interval}.holdout.json").read_text(encoding="utf-8"))
        m = cv["model"]
        b = cv["baselines"][cv["best_baseline"]]
        hm = h["model"]
        hb = h["baselines"][h["best_baseline"]]
        out.append(f"| {interval} | {m['n']} | {m['pinball_combined_bps']} | {m['cov50']} | "
                   f"{m['cov80']} | {m['w50_bps']} | {m['w80_bps']} | {cv['best_baseline']} | "
                   f"{b['pinball_combined_bps']} | {cv['gates']['pinball_improve_vs_best_base']} | "
                   f"{m['mae_bps']} | {m['mae_zero_bps']} | {cv.get('median_fallback')} |")
        imp = 1.0 - hm["pinball_combined_bps"] / hb["pinball_combined_bps"]
        ho.append(f"| {interval} | {hm['n']} | {hm['pinball_combined_bps']} | {hm['cov50']} | "
                  f"{hm['cov80']} | {hm['w50_bps']} | {hb['pinball_combined_bps']} | "
                  f"{hb['w50_bps']} | {round(imp, 4)} | {hm['mae_bps']} |")
    return "\n".join(out) + "\n\n" + "\n".join(ho) + "\n"


def news_table(root: Path) -> str:
    rows = {}
    for rn in ("r5", "r5b"):
        for r in load_rows(root, rn):
            rows.setdefault((r["interval"], r["label"]), r)
    lines = ["### 新闻消融（共同切片：新闻覆盖起点 + 7 天）", "",
             "| 周期 | 配置 | pinball | cov50 | cov80 | w50 | 提升(相对无新闻) |",
             "|---|---|---|---|---|---|---|"]
    for interval in ("5m", "15m", "30m", "1h", "4h"):
        base = rows.get((interval, "no_news"))
        news = rows.get((interval, "news"))
        if not base or not news:
            continue
        bp = base["m"]["pinball_combined_bps"]
        npin = news["m"]["pinball_combined_bps"]
        delta = 1.0 - npin / bp if bp else None
        for label, r in (("no_news", base), ("+news", news)):
            lines.append(f"| {interval} | {label} | {r['m']['pinball_combined_bps']} | "
                         f"{r['m']['cov50']} | {r['m']['cov80']} | {r['m']['w50_bps']} | "
                         f"{'' if label == 'no_news' else round(delta, 4)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report_v3")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.data_dir).resolve()
    text = []
    text.append(abl_table(root, "r1", "阶段 0–1：基础 / RV / 季节性 / 微观结构消融（EWMA 尺度，分位 GBM）"))
    text.append(abl_table(root, "r2", "阶段 2：直接尺度预测（分位 GBM，features=各周期最优）"))
    text.append(abl_table(root, "r3", "阶段 3：形状与共形校准（30m/1h/4h，sigma=HAR/HAR-RS/HAR-CJ）"))
    text.append(abl_table(root, "r3b", "阶段 3b：分位 GBM + Student-t 集成（blend）"))
    text.append(news_table(root))
    text.append(final_tables(root))
    Path(args.out).write_text("\n\n".join(text), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
