"""Saved direction models: one folder per interval, read by the debug predictor.

Layout (``<data>/models/<interval>/``):

* ``model.txt``      LightGBM booster, text format. Classes are stored as 0, 1, 2
                     for -1 (down), 0 (flat), +1 (up), the same as in training.
* ``features.json``  feature names in the order the booster expects.
* ``meta.json``      settings, trained time, out-of-sample summary, the debug banner,
                     and the SHA-256 of the other two files. It is written last, so it
                     is the commit point: a model folder whose hashes do not match
                     ``meta.json`` (a save that was interrupted) is refused on load.

The booster is refit on every labelled sample after the holdout evaluation, so
the saved model sees the newest bars. Its out-of-sample numbers still come from
the evaluation run, which is recorded in ``meta.json`` as such.

Nothing here touches keys, trading endpoints or the SQLite file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from supere.train.config import TrainConfig

SCHEMA_VERSION = 1
DEBUG_NOTE = "调试预测，样本外弱，勿实盘"
MODEL_FILE = "model.txt"
FEATURES_FILE = "features.json"
META_FILE = "meta.json"


class ArtifactError(RuntimeError):
    """A saved model folder exists but cannot be used as it is."""


@dataclass
class Artifact:
    interval: str
    directory: Path
    booster: lgb.Booster
    names: list[str]
    meta: dict

    @property
    def model_id(self) -> str:
        return str(self.meta["model_id"])


def strict(obj):
    """NaN and inf become None so the JSON is strict. numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {k: strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def model_dir(root: Path, interval: str) -> Path:
    return Path(root) / "models" / interval


def save(
    root: Path,
    cfg: TrainConfig,
    result: dict,
    booster: lgb.Booster,
    names: list[str],
    trained_at_utc8: str,
) -> Path:
    """Write the three files for ``cfg.interval`` and return the folder.

    Each file goes to a ``.tmp`` name first and is then swapped in, so a crash
    never leaves a half-written model that the predictor would load.
    """
    directory = model_dir(root, cfg.interval)
    directory.mkdir(parents=True, exist_ok=True)
    model_id = f"{cfg.interval}-{result['feature_hash']}-{result['dataset_hash']}"

    features_text = json.dumps(names, ensure_ascii=False, indent=2)
    _replace_text(directory / FEATURES_FILE, features_text)
    booster.save_model(str(directory / (MODEL_FILE + ".tmp")))
    os.replace(directory / (MODEL_FILE + ".tmp"), directory / MODEL_FILE)

    meta = strict(
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": model_id,
            "debug": True,
            "debug_note": DEBUG_NOTE,
            "interval": cfg.interval,
            "symbol": cfg.symbol,
            "market": cfg.market,
            "horizon": cfg.label.horizon,
            "k_atr": cfg.label.k_atr,
            "atr_period": cfg.label.atr_period,
            "threshold": cfg.threshold,
            "use_funding": cfg.use_funding,
            "use_metrics": cfg.use_metrics,
            "limit_bars": cfg.limit_bars,
            "trained_at_utc8": trained_at_utc8,
            "bars": result["bars"],
            "bar_range_utc8": result["bar_range_utc8"],
            "labelled_samples": result["samples"],
            "label_shares": result["label_shares"],
            "fit": "全量样本重拟合（在样本外评估之后）",
            "oos": {
                "split": result["config"]["split"]["mode"],
                "samples": result["oos_samples"],
                "effective_samples": result["effective_samples"],
                "classification": result["classification"],
                "baselines": result["baselines"],
            },
            "backtest": {
                scen: block["model"] for scen, block in result["backtest"]["scenarios"].items()
            },
            "feature_hash": result["feature_hash"],
            "dataset_hash": result["dataset_hash"],
            "n_features": len(names),
            "model_sha256": _sha256(directory / MODEL_FILE),
            "features_sha256": _sha256(directory / FEATURES_FILE),
        }
    )
    _replace_text(directory / META_FILE, json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False))
    return directory


def load(root: Path, interval: str) -> Artifact | None:
    """None when the folder has no model yet. Anything wrong with the folder raises ``ArtifactError``."""
    directory = model_dir(root, interval)
    if not directory.exists() or not any(directory.iterdir()):
        return None
    missing = [f for f in (MODEL_FILE, FEATURES_FILE, META_FILE) if not (directory / f).is_file()]
    if missing:
        raise ArtifactError(f"{interval} 的模型目录不完整，缺少 {', '.join(missing)}")
    try:
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        names = json.loads((directory / FEATURES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"{interval} 的元数据读不出来：{exc}") from exc

    if not isinstance(meta, dict):
        raise ArtifactError(f"{interval} 的 meta.json 不是对象")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"{interval} 的模型版本 {meta.get('schema_version')} 不识别")
    if meta.get("interval") != interval:
        raise ArtifactError(f"目录 {interval} 里的模型写的是 {meta.get('interval')}")
    for key in REQUIRED_META:
        if key not in meta:
            raise ArtifactError(f"{interval} 的 meta.json 缺少 {key}")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ArtifactError(f"{interval} 的 features.json 不是特征名列表")
    if len(names) != meta.get("n_features"):
        raise ArtifactError(f"{interval} 的特征数对不上")
    hashes_ok = (
        _sha256(directory / MODEL_FILE) == meta.get("model_sha256")
        and _sha256(directory / FEATURES_FILE) == meta.get("features_sha256")
    )
    if not hashes_ok:
        raise ArtifactError(f"{interval} 的模型文件和 meta.json 对不上，像是保存到一半被中断，请重新训练")
    try:
        booster = lgb.Booster(model_file=str(directory / MODEL_FILE))
    except lgb.basic.LightGBMError as exc:
        raise ArtifactError(f"{interval} 的模型文件读不出来：{exc}") from exc
    if booster.num_feature() != len(names):
        raise ArtifactError(f"{interval} 的模型特征数和 features.json 对不上")
    return Artifact(interval=interval, directory=directory, booster=booster, names=names, meta=meta)


REQUIRED_META = (
    "model_id", "use_funding", "use_metrics", "threshold", "symbol", "market",
    "horizon", "trained_at_utc8", "oos", "model_sha256", "features_sha256",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
