"""``python -m supere.train``: offline training. Read-only, never trades.

Examples (run from the repository root):

    python -m supere.train                                  # 12h, holdout
    python -m supere.train --interval 1h --mode walk_forward --folds 4
    python -m supere.train --data-dir D:\\Tool\\cg\\superE\\data --limit-bars 2000

Exit codes: 0 finished, 1 run failed (see the log), 2 bad configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supere.config import data_dir, load_config
from supere.train import config as tcfg
from supere.train.runner import close_logger, make_logger, run, write_result


def _overrides(args: argparse.Namespace) -> dict:
    train: dict = {}
    if args.limit_bars is not None:
        train["limit_bars"] = args.limit_bars
    if args.threshold is not None:
        train["threshold"] = args.threshold
    if args.no_funding:
        train["use_funding"] = False
    if args.no_metrics:
        train["use_metrics"] = False
    out: dict = {"train": train}
    label = {}
    if args.horizon is not None:
        label["horizon"] = args.horizon
    if args.k_atr is not None:
        label["k_atr"] = args.k_atr
    if label:
        out["label"] = label
    split = {}
    if args.mode is not None:
        split["mode"] = args.mode
    if args.n_folds is not None:
        split["n_folds"] = args.n_folds
    if args.valid_frac is not None:
        split["valid_frac"] = args.valid_frac
    if split:
        out["split"] = split
    model = {}
    if args.n_estimators is not None:
        model["n_estimators"] = args.n_estimators
    if args.n_jobs is not None:
        model["n_jobs"] = args.n_jobs
    if model:
        out["model"] = model
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m supere.train",
        description="离线训练骨架：只读 superE.sqlite，不下单。",
    )
    parser.add_argument("--config", help="config.toml 路径（用于定位数据目录）")
    parser.add_argument("--data-dir", help="数据目录（含 superE.sqlite）；默认按 config.toml")
    parser.add_argument("--train-config", help="train.toml 路径；默认用 config.toml 同目录的 train.toml")
    # No argparse choices here: config.validate() gives the explicit refusal message.
    parser.add_argument("--interval", help="12h（默认）、1h 或 15m")
    parser.add_argument("--horizon", type=int, help="标签期限（根数）")
    parser.add_argument("--k", dest="k_atr", type=float, help="障碍宽度（倍 ATR）")
    parser.add_argument("--mode", choices=("holdout", "walk_forward"))
    parser.add_argument("--folds", dest="n_folds", type=int, help="walk_forward 的折数")
    parser.add_argument("--valid-frac", type=float, help="holdout 的测试段比例")
    parser.add_argument("--threshold", type=float, help="开仓所需概率")
    parser.add_argument("--limit-bars", type=int, help="只用最近 N 根（加速冒烟用）")
    parser.add_argument("--no-funding", action="store_true", help="不用资金费率特征和回测成本")
    parser.add_argument("--no-metrics", action="store_true", help="不用 metrics 特征")
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--print-config", action="store_true", help="只打印合并后的配置")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    try:
        app_cfg, cfg_path = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - report and stop, no traceback noise
        print(f"读取 config.toml 失败：{exc}", file=sys.stderr)
        return 2

    root = Path(args.data_dir).resolve() if args.data_dir else data_dir(app_cfg, cfg_path)
    train_path = Path(args.train_config) if args.train_config else cfg_path.parent / "train.toml"
    file_values = tcfg.load_file(train_path)
    # The symbol comes from config.toml unless train.toml names one.
    file_values.setdefault("train", {}).setdefault("symbol", app_cfg.symbols.um)

    try:
        cfg = tcfg.build(
            interval=args.interval,
            file_values=file_values,
            overrides=_overrides(args),
        )
    except (tcfg.ConfigError, TypeError) as exc:
        print(f"训练配置错误：{exc}", file=sys.stderr)
        return 2

    if args.print_config:
        from dataclasses import asdict

        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        return 0

    db = root / "superE.sqlite"
    log, log_path = make_logger(root / "logs")
    try:
        log.info("数据目录 %s；train.toml %s", root,
                 train_path if train_path.exists() else "（未找到，用默认值）")
        try:
            result = run(cfg, db, log)
        except (FileNotFoundError, ValueError) as exc:
            log.error("训练失败：%s", exc)
            return 1
        except Exception:  # noqa: BLE001 - full traceback goes to the log file
            log.exception("训练异常中止")
            return 1

        json_path = write_result(result, log_path)
        log.info("结果 JSON：%s", json_path)
        log.info("日志：%s", log_path)
        return 0
    finally:
        close_logger(log)


if __name__ == "__main__":
    raise SystemExit(main())
