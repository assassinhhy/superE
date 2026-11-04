"""Triple-barrier labels and purged splits."""

from __future__ import annotations

import numpy as np
import pytest

from supere.train import labels, splits
from supere.train.data import Bars

INTERVAL = "12h"
STEP = 43_200_000
T0 = 1_577_836_800_000


def _flat_bars(n: int, price: float = 100.0) -> Bars:
    """Every bar is o=c=100 with a 2-point range, so ATR is exactly 2."""
    ot = T0 + np.arange(n, dtype=np.int64) * STEP
    return Bars(
        interval=INTERVAL,
        open_time=ot,
        o=np.full(n, price),
        h=np.full(n, price + 1.0),
        l=np.full(n, price - 1.0),
        c=np.full(n, price),
        volume=np.full(n, 10.0),
        quote_volume=np.full(n, 1000.0),
        trades=np.full(n, 100.0),
        taker_buy_base=np.full(n, 5.0),
    )


def _label_at(lab: labels.Labels, i: int) -> int:
    pos = int(np.nonzero(lab.idx == i)[0][0])
    return pos


def test_upper_barrier_hit_is_plus_one_and_exit_is_that_bar():
    bars = _flat_bars(40)
    i = 20  # ATR(14) needs bars 0..13 first
    bars.h[i + 2] = 103.0  # entry 100, k*ATR = 2 -> upper barrier 102
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    p = _label_at(lab, i)
    assert lab.y[p] == 1
    assert lab.exit_bar[p] == i + 2
    assert lab.exit_price[p] == pytest.approx(102.0)
    assert not lab.ambiguous[p]


def test_lower_barrier_hit_is_minus_one():
    bars = _flat_bars(40)
    i = 20  # ATR(14) needs bars 0..13 first
    bars.l[i + 1] = 97.0  # lower barrier 98
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    p = _label_at(lab, i)
    assert lab.y[p] == -1
    assert lab.exit_bar[p] == i + 1
    assert lab.exit_price[p] == pytest.approx(98.0)


def test_no_touch_is_flat_and_exits_at_horizon_close():
    bars = _flat_bars(40)
    i = 20  # ATR(14) needs bars 0..13 first
    bars.c[i + 3] = 100.5
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    p = _label_at(lab, i)
    assert lab.y[p] == 0
    assert lab.exit_bar[p] == i + 3
    assert lab.exit_price[p] == pytest.approx(100.5)


def test_both_barriers_in_one_bar_is_ambiguous_and_conservative():
    bars = _flat_bars(40)
    i = 20  # ATR(14) needs bars 0..13 first
    bars.h[i + 1] = 103.0
    bars.l[i + 1] = 97.0
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    p = _label_at(lab, i)
    assert lab.ambiguous[p]
    assert lab.y[p] == -1  # conservative: the unfavourable barrier
    assert lab.exit_price[p] == pytest.approx(98.0)


def test_entry_is_next_bar_open_not_signal_close():
    bars = _flat_bars(40)
    i = 20  # ATR(14) needs bars 0..13 first
    bars.o[i + 1] = 110.0  # the entry moves; ATR is unchanged
    bars.h[i + 1] = 111.0
    bars.l[i + 1] = 109.0
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    p = _label_at(lab, i)
    assert lab.entry[p] == pytest.approx(110.0)  # open of i+1, not c[i] = 100


def test_samples_without_a_full_horizon_are_dropped():
    bars = _flat_bars(20)
    lab = labels.build(bars, horizon=3, k_atr=1.0)
    assert lab.idx.max() == bars.n - 1 - 3
    assert np.all(lab.exit_bar <= bars.n - 1)


def test_label_uses_only_atr_known_at_decision():
    """Changing bars after the decision bar must not change its barriers."""
    base = _flat_bars(40)
    changed = _flat_bars(40)
    changed.h[30:] = 500.0
    changed.l[30:] = 1.0
    i = 20
    a = labels.build(base, horizon=2, k_atr=1.0)
    b = labels.build(changed, horizon=2, k_atr=1.0)
    pa = _label_at(a, i)
    pb = _label_at(b, i)
    assert a.entry[pa] == b.entry[pb]
    assert a.exit_price[pa] == pytest.approx(100.0) and b.exit_price[pb] == pytest.approx(100.0)


def test_class_shares_sum_to_one():
    shares = labels.class_shares(np.array([1, 0, 0, -1]))
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["flat"] == pytest.approx(0.5)


# ---- splits -----------------------------------------------------------------

def _samples(m: int, horizon: int):
    idx = np.arange(m, dtype=np.int64)
    exit_bar = idx + horizon
    return idx, exit_bar


def test_holdout_is_the_newest_block_with_purge_and_embargo():
    idx, exit_bar = _samples(100, horizon=3)
    folds = list(
        splits.iter_folds(idx, exit_bar, mode="holdout", valid_frac=0.25, n_folds=4, embargo_bars=3)
    )
    assert len(folds) == 1
    f = folds[0]
    assert f.test[0] == 75 and f.test[-1] == 99
    first_test_bar = idx[f.test[0]]
    assert np.all(exit_bar[f.train] < first_test_bar - 3)  # purge + embargo
    assert f.train.max() < f.test.min()  # never trains on the future


def test_walk_forward_is_expanding_and_never_shuffles():
    idx, exit_bar = _samples(200, horizon=2)
    folds = list(
        splits.iter_folds(idx, exit_bar, mode="walk_forward", valid_frac=0.25, n_folds=4, embargo_bars=2)
    )
    assert [f.number for f in folds] == [1, 2, 3, 4]
    prev_train_size = -1
    for f in folds:
        assert f.train.shape[0] > prev_train_size  # expanding
        prev_train_size = f.train.shape[0]
        assert np.all(np.diff(f.train) > 0)  # chronological order kept
        assert np.all(exit_bar[f.train] < idx[f.test[0]] - 2)
        assert f.train.max() < f.test.min()


def test_purge_removes_samples_whose_label_overlaps_the_test_block():
    idx, exit_bar = _samples(60, horizon=5)
    # Pretend the last training samples have long labels that reach into the test block.
    exit_bar = exit_bar.copy()
    exit_bar[40:50] = idx[40:50] + 30
    folds = list(
        splits.iter_folds(idx, exit_bar, mode="holdout", valid_frac=0.25, n_folds=1, embargo_bars=0)
    )
    f = folds[0]
    assert f.test[0] == 45
    assert not np.any((f.train >= 40) & (f.train < 45))  # those labels end after the cut


def test_too_few_samples_is_an_error():
    idx, exit_bar = _samples(5, horizon=1)
    with pytest.raises(ValueError):
        list(splits.iter_folds(idx, exit_bar, mode="holdout", valid_frac=0.25, n_folds=1, embargo_bars=1))
