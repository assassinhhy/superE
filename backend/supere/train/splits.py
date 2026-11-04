"""Time-ordered splits with purge and embargo (plan 7.3 and 8.3).

Samples are sorted by decision bar. A training sample is kept only when its
label has already finished, ``exit_bar < first_test_bar - embargo``. That is
the purge (labels that overlap the test window are dropped) plus the embargo
(a gap of ``embargo`` bars before the test window). Shuffling is never used.

Two modes:

* ``holdout``: one split. The newest ``valid_frac`` of samples is the test set.
* ``walk_forward``: expanding window. The first block only trains. Each later
  block is tested once, using everything before it that survives purging.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    number: int
    train: np.ndarray  # positions into the sample arrays
    test: np.ndarray


def iter_folds(
    idx: np.ndarray,
    exit_bar: np.ndarray,
    *,
    mode: str,
    valid_frac: float,
    n_folds: int,
    embargo_bars: int,
) -> Iterator[Fold]:
    m = int(idx.shape[0])
    if m < 10:
        raise ValueError(f"样本只有 {m} 个，不足以切分")

    if mode == "holdout":
        cut = int(round(m * (1.0 - valid_frac)))
        blocks = [(0, cut), (cut, m)]
        bounds = [blocks[1]]
    elif mode == "walk_forward":
        edges = np.linspace(0, m, n_folds + 2).astype(np.int64)
        bounds = [(int(edges[k]), int(edges[k + 1])) for k in range(1, n_folds + 1)]
    else:
        raise ValueError(f"未知 split 模式 {mode}")

    for number, (start, stop) in enumerate(bounds, start=1):
        if stop <= start or start == 0:
            continue
        test_first_bar = int(idx[start])
        keep = exit_bar[:start] < (test_first_bar - embargo_bars)
        train = np.nonzero(keep)[0]
        test = np.arange(start, stop, dtype=np.int64)
        yield Fold(number=number, train=train.astype(np.int64), test=test)
