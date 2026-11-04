"""End-to-end offline training run.

Steps: read-only load -> causal features -> triple-barrier labels -> drop rows
with missing core features or a K-line gap in their window -> purged split ->
LightGBM per fold -> out-of-sample metrics and baselines -> backtest.

Every run writes a text log and a JSON result under ``data/logs/``. The JSON
has the config, the hashes, the metrics and the trade summaries, so a run can
be compared later without re-reading the log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from supere.timeutil import UTC8
from supere.train import backtest, data, features, labels, metrics, model, splits
from supere.train.config import TrainConfig
from supere.train.model import CLASSES

LOGGER_NAME = "supere.train"
WINDOW_LOOKBACK = 60  # bars of history a row needs; no gap may fall inside it
MIN_EFFECTIVE_SAMPLES = 300  # plan 7.6: promotion threshold, reported here only


def make_logger(log_dir: Path, *, console: bool = True) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC8).strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"train-{stamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt.converter = lambda ts: datetime.fromtimestamp(ts, tz=UTC8).timetuple()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    return logger, path


def _window_has_gap(gaps: np.ndarray, sample_bar: np.ndarray, horizon: int) -> np.ndarray:
    """True where a gap sits between ``i - LOOKBACK`` and ``i + horizon``."""
    prefix = np.concatenate([[0], np.cumsum(gaps.astype(np.int64))])
    lo = np.maximum(sample_bar - WINDOW_LOOKBACK, 0)
    hi = sample_bar + horizon
    return (prefix[hi] - prefix[lo]) > 0


def _dataset_hash(sample_bar: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(sample_bar.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()[:16]


def close_logger(log: logging.Logger) -> None:
    """Release the file handle so the log can be read or moved on Windows."""
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


def run(cfg: TrainConfig, db_path: Path, log: logging.Logger) -> dict:
    log.info("开始离线训练 周期=%s 标签 H=%d k=%.2f 切分=%s",
             cfg.interval, cfg.label.horizon, cfg.label.k_atr, cfg.split.mode)
    log.info("只读打开 %s", Path(db_path).resolve())

    conn = data.open_readonly(db_path)
    try:
        bars = data.load_bars(conn, cfg.market, cfg.symbol, cfg.interval, cfg.limit_bars)
        funding = data.load_funding(conn, cfg.symbol) if cfg.use_funding else data.Funding.empty()
        metrics_in = data.load_metrics(conn, cfg.symbol) if cfg.use_metrics else None
    finally:
        conn.close()

    log.info("K 线 %d 根，区间 %s ~ %s（UTC+8）",
             bars.n, _fmt(bars.open_time[0]), _fmt(bars.open_time[-1]))
    log.info("资金费率 %d 条；metrics %s",
             funding.time.shape[0], "未启用" if metrics_in is None else f"{metrics_in.ts.shape[0]} 条")

    names, X_all = features.build(bars, funding if cfg.use_funding else None, metrics_in)
    lab = labels.build(bars, cfg.label.horizon, cfg.label.k_atr, cfg.label.atr_period)
    if lab.size == 0:
        raise ValueError("没有可打标签的样本")

    gaps = data.gap_after(bars)
    core_cols = [names.index(c) for c in features.CORE_FEATURES]
    x_rows = X_all[lab.idx]
    keep = np.all(np.isfinite(x_rows[:, core_cols]), axis=1)
    gap_drop = _window_has_gap(gaps, lab.idx, cfg.label.horizon)
    keep &= ~gap_drop
    log.info("样本 %d 个；核心特征缺失或窗口跨缺口剔除 %d 个（缺口 %d 处）",
             lab.size, int(np.sum(~keep)), int(np.sum(gaps)))

    sample_bar = lab.idx[keep]
    X = x_rows[keep]
    y = lab.y[keep]
    exit_bar = lab.exit_bar[keep]
    entry = lab.entry[keep]
    exit_px = lab.exit_price[keep]
    amb = lab.ambiguous[keep]
    m = int(sample_bar.shape[0])
    if m < 50:
        raise ValueError(f"可用样本只有 {m} 个，不足以训练")

    shares = labels.class_shares(y)
    amb_share = float(np.mean(amb)) if m else 0.0
    log.info("标签分布 涨 %.1f%%  观望 %.1f%%  跌 %.1f%%；ambiguous %.1f%%",
             100 * shares["up"], 100 * shares["flat"], 100 * shares["down"], 100 * amb_share)
    if not 0.30 <= shares["flat"] <= 0.50:
        log.warning("观望类占比 %.1f%% 不在 30%%–50%% 的建议区间，可调整 k 或 H",
                    100 * shares["flat"])

    embargo = cfg.split.embargo_bars if cfg.split.embargo_bars is not None else cfg.label.horizon
    oof = np.full((m, 3), np.nan)
    in_oos = np.zeros(m, dtype=bool)
    last_model = None
    fold_rows: list[dict] = []

    for fold in splits.iter_folds(
        sample_bar,
        exit_bar,
        mode=cfg.split.mode,
        valid_frac=cfg.split.valid_frac,
        n_folds=cfg.split.n_folds,
        embargo_bars=embargo,
    ):
        if fold.train.shape[0] < 50:
            log.warning("第 %d 折训练样本只有 %d 个，跳过", fold.number, fold.train.shape[0])
            continue
        try:
            proba, last_model = model.fit_predict(X[fold.train], y[fold.train], X[fold.test], cfg.model)
        except ValueError as exc:
            log.warning("第 %d 折跳过：%s", fold.number, exc)
            continue
        oof[fold.test] = proba
        in_oos[fold.test] = True
        fold_acc = float(np.mean(CLASSES[np.argmax(proba, axis=1)] == y[fold.test]))
        fold_rows.append(
            {
                "fold": fold.number,
                "train": int(fold.train.shape[0]),
                "test": int(fold.test.shape[0]),
                "test_start_utc8": _fmt(bars.open_time[sample_bar[fold.test[0]]]),
                "accuracy": fold_acc,
            }
        )
        log.info("第 %d 折 训练 %d / 测试 %d，测试准确率 %.3f",
                 fold.number, fold.train.shape[0], fold.test.shape[0], fold_acc)

    if not np.any(in_oos):
        raise ValueError("没有任何一折能完成训练")

    pos = np.nonzero(in_oos)[0]
    proba_o = oof[pos]
    y_o = y[pos]
    first_pos = int(pos[0])
    cls = metrics.classification(y_o, proba_o)
    base = metrics.baselines(y_o, y[:first_pos] if first_pos > 0 else y_o, X[pos, names.index("ret_1")])

    eff = len(pos) / max(cfg.label.horizon, 1)
    log.info("样本外 %d 个（约等于 %.0f 个有效独立样本，H=%d）", len(pos), eff, cfg.label.horizon)
    if eff < MIN_EFFECTIVE_SAMPLES:
        log.warning("有效独立样本约 %.0f，低于 %d 的升级门槛；结果只能作流程检查", eff, MIN_EFFECTIVE_SAMPLES)
    log.info("样本外准确率 %.3f；log loss %.4f；Brier %.4f；macro AUC %s",
             cls["accuracy"], cls["log_loss"], cls["brier"],
             "n/a" if np.isnan(cls["auc_ovr_macro"]) else f"{cls['auc_ovr_macro']:.3f}")
    for key, value in base.items():
        log.info("基线 %-9s 准确率 %.3f%s", key, value["accuracy"],
                 f"；log loss {value['log_loss']:.4f}" if "log_loss" in value else "")

    bt = _backtest_block(cfg, bars, funding, sample_bar[pos], exit_bar[pos], entry[pos],
                         exit_px[pos], amb[pos], proba_o, X[pos, names.index("ret_1")], log)

    importance = model.importance_table(last_model, names) if last_model is not None else []
    if importance:
        log.info("特征重要性（最后一折，gain 占比）：")
        for name, share in importance:
            log.info("  %-22s %.3f", name, share)

    result = {
        "finished_utc8": datetime.now(UTC8).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "feature_hash": features.feature_hash(names, {"interval": cfg.interval,
                                                       "horizon": cfg.label.horizon,
                                                       "k": cfg.label.k_atr,
                                                       "funding": cfg.use_funding,
                                                       "metrics": cfg.use_metrics}),
        "dataset_hash": _dataset_hash(sample_bar, y),
        "bars": int(bars.n),
        "bar_range_utc8": [_fmt(bars.open_time[0]), _fmt(bars.open_time[-1])],
        "gaps": int(np.sum(gaps)),
        "samples": m,
        "label_shares": shares,
        "ambiguous_share": amb_share,
        "oos_samples": int(len(pos)),
        "effective_samples": eff,
        "folds": fold_rows,
        "classification": cls,
        "baselines": base,
        "backtest": bt,
        "importance": importance,
        "features": names,
    }
    return result


def _backtest_block(cfg, bars, funding, sample_bar, exit_bar, entry, exit_px, amb, proba, ret1, log):
    interval_ms = bars.interval_ms
    out: dict = {"threshold": cfg.threshold, "scenarios": {}}
    n = int(sample_bar.shape[0])
    dirs_model = backtest.signal_dirs(proba, cfg.threshold)
    dirs_long = np.ones(n, dtype=np.int64)
    dirs_mom = np.where(ret1 > 0, 1, np.where(ret1 < 0, -1, 0)).astype(np.int64)

    for scen in backtest.scenarios(cfg.cost):
        block = {}
        for label, dirs in (("model", dirs_model), ("always_long", dirs_long), ("momentum", dirs_mom)):
            trades = backtest.simulate(
                sample_bar=sample_bar,
                exit_bar=exit_bar,
                entry=entry,
                exit_price=exit_px,
                ambiguous=amb,
                dirs=dirs,
                open_time=bars.open_time,
                interval_ms=interval_ms,
                funding=funding,
                scenario=scen,
                slippage_per_side=cfg.cost.slippage_per_side,
            )
            block[label] = backtest.summarize(trades, n)
        out["scenarios"][scen.name] = block
        mdl = block["model"]
        log.info(
            "回测 %-11s 模型：交易 %d（覆盖 %.1f%%）胜率 %.1f%% 期望 %.3f%% 盈亏比 %.2f 复利 %.2f%% 最大回撤 %.2f%% 最大连亏 %d",
            scen.name, mdl["trades"], 100 * mdl["coverage"], 100 * _nz(mdl["hit_rate"]),
            100 * _nz(mdl["expectancy"]), _nz(mdl["payoff_ratio"]), 100 * mdl["compounded"],
            100 * mdl["max_drawdown"], mdl["max_consecutive_losses"],
        )
        for name in ("always_long", "momentum"):
            b = block[name]
            log.info("  基线 %-11s 交易 %d 期望 %.3f%%", name, b["trades"], 100 * _nz(b["expectancy"]))
    return out


def _nz(value) -> float:
    return 0.0 if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


def _fmt(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")


def write_result(result: dict, log_path: Path) -> Path:
    """JSON next to the text log, same stem (``train-<stamp>.json``)."""
    path = log_path.with_suffix(".json")
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")
