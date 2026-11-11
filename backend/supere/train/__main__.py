"""``python -m supere.train``: offline training. Read-only, never trades.

Examples (run from the repository root):

    python -m supere.train --interval 1h --mode walk_forward --folds 4
    python -m supere.train --interval 5m,15m,30m,1h,4h --data-dir D:\\Tool\\cg\\superE\\data
    python -m supere.train --interval product               # the five UI intervals
    python -m supere.train --interval 15m --no-save         # evaluate only, keep no model

Each successful interval also saves a debug model under ``<data>/models/<interval>/``
(see ``artifacts.py``). Exit codes: 0 all finished, 1 a run failed (see the log),
2 bad configuration (nothing is run).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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


def _parse_intervals(text: str | None) -> list[str | None]:
    """``None`` keeps the train.toml / default interval. ``product`` is the five UI intervals."""
    if text is None:
        return [None]
    if text.strip().lower() == "product":
        return list(tcfg.PRODUCT_INTERVALS)
    out: list[str] = []
    for part in text.split(","):
        item = part.strip()
        if item and item not in out:
            out.append(item)
    if not out:
        raise tcfg.ConfigError("--interval 不能为空")
    return out


def _train_one(cfg: tcfg.TrainConfig, db: Path, log_dir: Path, save_root: Path | None,
               train_path: Path) -> bool:
    log, log_path = make_logger(log_dir, tag=cfg.interval)
    try:
        log.info("数据目录 %s；train.toml %s；周期 %s", db.parent,
                 train_path if train_path.exists() else "（未找到，用默认值）", cfg.interval)
        try:
            result = run(cfg, db, log, save_root=save_root)
        except (FileNotFoundError, ValueError) as exc:
            log.error("训练失败：%s", exc)
            return False
        except Exception:  # noqa: BLE001 - full traceback goes to the log file
            log.exception("训练异常中止")
            return False

        json_path = write_result(result, log_path)
        log.info("结果 JSON：%s", json_path)
        log.info("日志：%s", log_path)
        return True
    finally:
        close_logger(log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m supere.train",
        description="离线训练骨架：只读 superE.sqlite，不下单。",
    )
    parser.add_argument("--config", help="config.toml 路径（用于定位数据目录）")
    parser.add_argument("--data-dir", help="数据目录（含 superE.sqlite）；默认按 config.toml")
    parser.add_argument("--train-config", help="train.toml 路径；默认用 config.toml 同目录的 train.toml")
    # No argparse choices here: config.validate() gives the explicit refusal message.
    parser.add_argument(
        "--interval",
        help="周期，可用逗号分隔多个，如 5m,15m；product 表示界面的五个周期；不写则用 train.toml（默认 12h）",
    )
    parser.add_argument("--horizon", type=int, help="标签期限（根数）")
    parser.add_argument("--k", dest="k_atr", type=float, help="障碍宽度（倍 ATR）")
    parser.add_argument("--mode", choices=("holdout", "walk_forward"))
    parser.add_argument("--folds", dest="n_folds", type=int, help="walk_forward 的折数")
    parser.add_argument("--valid-frac", type=float, help="holdout 的测试段比例")
    parser.add_argument("--threshold", type=float, help="开仓所需概率")
    parser.add_argument("--limit-bars", type=int, help="只用最近 N 根（加速用）")
    parser.add_argument("--no-funding", action="store_true", help="不用资金费率特征和回测成本")
    parser.add_argument("--no-metrics", action="store_true", help="不用 metrics 特征")
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--print-config", action="store_true", help="只打印合并后的配置")
    parser.add_argument("--no-save", action="store_true", help="只评估，不保存调试模型")
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
    try:
        file_values = tcfg.load_file(train_path, required=bool(args.train_config))
        for section in ("train", "label", "split", "cost", "model"):
            if not isinstance(file_values.get(section, {}), dict):
                raise tcfg.ConfigError(f"train.toml 里的 [{section}] 必须是表")
        # The symbol comes from config.toml unless train.toml names one.
        file_values.setdefault("train", {}).setdefault("symbol", app_cfg.symbols.um)
        intervals = _parse_intervals(args.interval)
        cfgs = [
            tcfg.build(interval=interval, file_values=file_values, overrides=_overrides(args))
            for interval in intervals
        ]
    except (tcfg.ConfigError, TypeError, AttributeError) as exc:
        print(f"训练配置错误：{exc}", file=sys.stderr)
        return 2

    if args.print_config:
        payload = [asdict(c) for c in cfgs]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, ensure_ascii=False, indent=2))
        return 0

    db = root / "superE.sqlite"
    save_root = None if args.no_save else root
    failed: list[str] = []
    for cfg in cfgs:
        if not _train_one(cfg, db, root / "logs", save_root, train_path):
            failed.append(cfg.interval)
    if failed:
        print(f"以下周期失败：{', '.join(failed)}（见数据目录 logs 下的日志）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
