"""LightGBM three-class model.

The number of trees is fixed in the config. There is no early stopping on the
test block, because that would let the test set choose the model (plan 8.3).
Classes are stored as -1, 0, +1 in the labels. They are mapped to 0, 1, 2 for
LightGBM and mapped back to the label order for every output.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from supere.train.config import ModelConfig

CLASSES = np.array([-1, 0, 1], dtype=np.int64)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    cfg: ModelConfig,
) -> tuple[np.ndarray, lgb.LGBMClassifier]:
    present = np.unique(y_train)
    if present.shape[0] < 2:
        raise ValueError(f"训练段只有 {present.shape[0]} 个类别，无法拟合")

    model = _make(cfg)
    model.fit(x_train, y_train + 1)  # -> 0, 1, 2

    raw = model.predict_proba(x_test)
    proba = np.zeros((x_test.shape[0], 3), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        proba[:, int(cls)] = raw[:, col]
    return proba, model


def fit_all(x: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> lgb.LGBMClassifier:
    """Final fit on every labelled sample, for the saved debug model.

    Needs all three classes: a booster missing one would map columns wrongly.
    """
    present = np.unique(y)
    if present.shape[0] < 3:
        raise ValueError(f"全量拟合需要三类标签，当前只有 {present.shape[0]} 类")
    model = _make(cfg)
    model.fit(x, y + 1)
    return model


def _make(cfg: ModelConfig) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )


def importance_table(model: lgb.LGBMClassifier, names: list[str], top: int = 10) -> list[tuple[str, float]]:
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1][:top]
    total = float(gains.sum()) or 1.0
    return [(names[i], float(gains[i]) / total) for i in order]
