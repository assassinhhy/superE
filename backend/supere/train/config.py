"""Training configuration.

Defaults follow plan section 8.4: 12h is the first experiment, 15m and 1h are
the switchable alternatives. 1d, 1w and 1M never get a directional model
(owner decision, 2026-11-04 batch 1 and batch 3).

Settings can come from ``train.toml`` (optional, may be committed) and from
command-line flags. Nothing here touches keys or trading endpoints.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

# Labelled horizon (bars) and barrier width (x ATR) per interval.
DEFAULT_LABEL: dict[str, tuple[int, float]] = {
    "15m": (4, 1.0),
    "1h": (4, 1.0),
    "12h": (3, 1.0),
}
TRAINABLE_INTERVALS: tuple[str, ...] = tuple(DEFAULT_LABEL)


class ConfigError(ValueError):
    """Raised for settings that would silently produce a wrong experiment."""


@dataclass(frozen=True)
class LabelConfig:
    horizon: int
    k_atr: float
    atr_period: int = 14


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "holdout"  # holdout | walk_forward
    valid_frac: float = 0.25
    n_folds: int = 4
    embargo_bars: int | None = None  # None -> horizon (plan 8.3: at least H bars)


@dataclass(frozen=True)
class CostConfig:
    # Plan section 7.4 / owner batch 2: maker 0.02%, taker 0.05% (VIP0 public
    # rate; the owner's real account rate takes precedence).
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_per_side: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    seed: int = 42
    n_jobs: int = 2  # keep CPU free for the live collector on the same box


@dataclass(frozen=True)
class TrainConfig:
    interval: str = "12h"
    symbol: str = "ETHUSDT"
    market: str = "um"
    label: LabelConfig = field(default_factory=lambda: LabelConfig(3, 1.0))
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: float = 0.40  # probability needed to open a trade
    limit_bars: int | None = None  # keep only the newest N decision bars
    use_funding: bool = True
    use_metrics: bool = True


def validate(cfg: TrainConfig) -> TrainConfig:
    if cfg.interval not in TRAINABLE_INTERVALS:
        raise ConfigError(
            f"周期 {cfg.interval} 不训练方向模型；可选 {', '.join(TRAINABLE_INTERVALS)}"
        )
    if cfg.label.horizon < 1:
        raise ConfigError("标签期限 horizon 至少为 1 根 K 线")
    if cfg.label.k_atr <= 0:
        raise ConfigError("障碍宽度 k_atr 必须为正")
    if cfg.split.mode not in ("holdout", "walk_forward"):
        raise ConfigError("split.mode 只能是 holdout 或 walk_forward")
    if not 0 < cfg.split.valid_frac < 1:
        raise ConfigError("valid_frac 必须在 (0, 1) 之间")
    if cfg.split.mode == "walk_forward" and cfg.split.n_folds < 1:
        raise ConfigError("walk_forward 至少 1 折")
    if not 0 < cfg.threshold < 1:
        raise ConfigError("threshold 必须在 (0, 1) 之间")
    if cfg.limit_bars is not None and cfg.limit_bars < 50:
        raise ConfigError("limit_bars 太小，至少 50 根")
    return cfg


def default_for(interval: str) -> TrainConfig:
    horizon, k = DEFAULT_LABEL.get(interval, (3, 1.0))
    return TrainConfig(interval=interval, label=LabelConfig(horizon=horizon, k_atr=k))


def load_file(path: Path | None, *, required: bool = False) -> dict:
    """Read train.toml. A missing file is fine unless ``required``."""
    if path is None or not Path(path).exists():
        if required:
            raise ConfigError(f"找不到训练配置 {path}")
        return {}
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"train.toml 解析失败：{exc}") from exc


def build(
    interval: str | None = None,
    file_values: dict | None = None,
    overrides: dict | None = None,
) -> TrainConfig:
    """Merge defaults <- train.toml <- explicit overrides, then validate.

    ``file_values`` and ``overrides`` use the same nested keys as the TOML
    sections: ``{"label": {"horizon": 2}, "split": {...}, ...}``.
    """
    file_values = file_values or {}
    overrides = overrides or {}
    top = {**file_values.get("train", {}), **overrides.get("train", {})}
    chosen = interval or top.get("interval") or "12h"
    base = default_for(chosen)

    def section(name: str, current):
        # Partial sections keep the defaults for every key they do not name.
        given = {**file_values.get(name, {}), **overrides.get(name, {})}
        return replace(current, **given) if given else current

    cfg = replace(
        base,
        interval=chosen,
        symbol=top.get("symbol", base.symbol),
        market=top.get("market", base.market),
        label=section("label", base.label),
        split=section("split", base.split),
        cost=section("cost", base.cost),
        model=section("model", base.model),
        threshold=top.get("threshold", base.threshold),
        limit_bars=top.get("limit_bars", base.limit_bars),
        use_funding=top.get("use_funding", base.use_funding),
        use_metrics=top.get("use_metrics", base.use_metrics),
    )
    return validate(cfg)
