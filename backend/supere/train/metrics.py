"""Out-of-sample classification metrics and simple baselines (plan 7.4).

Column order of ``proba`` is the class order ``[-1, 0, +1]``. Baselines are
scored with the same metrics so that a model which does not beat them is
visible in the output.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from supere.train.model import CLASSES


def classification(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = CLASSES[np.argmax(proba, axis=1)]
    out: dict = {"n": int(y.shape[0]), "accuracy": float(np.mean(pred == y))}

    per_class: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        tp = float(np.sum((pred == cls) & (y == cls)))
        pred_pos = float(np.sum(pred == cls))
        act_pos = float(np.sum(y == cls))
        per_class[str(int(cls))] = {
            "precision": tp / pred_pos if pred_pos else float("nan"),
            "recall": tp / act_pos if act_pos else float("nan"),
            "support": act_pos,
        }
    out["per_class"] = per_class

    present = set(np.unique(y).tolist())
    if len(present) == 3:
        out["auc_ovr_macro"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=CLASSES))
    else:
        out["auc_ovr_macro"] = float("nan")  # AUC needs every class in the test set

    clipped = np.clip(proba, 1e-9, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    out["log_loss"] = float(log_loss(y, clipped, labels=CLASSES))

    onehot = np.stack([(y == c).astype(np.float64) for c in CLASSES], axis=1)
    out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    return out


def baselines(
    y: np.ndarray,
    train_y: np.ndarray,
    ret_1: np.ndarray,
    seed: int = 42,
) -> dict[str, dict]:
    """Accuracy and log loss for simple predictors on the same test rows.

    * ``prior``: always the training-set class frequencies (the log-loss floor).
    * ``always_up``: always predicts +1.
    * ``momentum``: predicts the sign of the previous bar's return.
    * ``random``: uniform over the three classes, seeded.
    """
    n = y.shape[0]
    prior = np.array([np.mean(train_y == c) for c in CLASSES], dtype=np.float64)
    prior = np.clip(prior, 1e-9, None)
    prior = prior / prior.sum()
    prior_proba = np.tile(prior, (n, 1))

    rng = np.random.default_rng(seed)
    rand_pred = rng.choice(CLASSES, size=n)
    mom_pred = np.where(ret_1 > 0, 1, np.where(ret_1 < 0, -1, 0))

    out: dict[str, dict] = {}
    out["prior"] = {
        "accuracy": float(np.mean(CLASSES[np.argmax(prior_proba, axis=1)] == y)),
        "log_loss": float(log_loss(y, prior_proba, labels=CLASSES)),
    }
    out["always_up"] = {"accuracy": float(np.mean(y == 1))}
    out["momentum"] = {"accuracy": float(np.mean(mom_pred == y))}
    out["random"] = {"accuracy": float(np.mean(rand_pred == y))}
    return out
