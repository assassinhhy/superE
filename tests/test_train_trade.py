"""Trade-outcome labels (2026-11-05 iteration) and their feature set: hand-checked paths and no-lookahead."""

from __future__ import annotations

import numpy as np
import pytest

from supere.train import trade_features, trade_labels
from supere.train.data import Bars

T0 = 1_577_836_800_000  # 2020-01-01 00:00 UTC, on the 1h and 4h grids
H1 = 3_600_000
H4 = 4 * H1


def _bars(o, h, l, c, interval="1h", step=H1):  # noqa: E741
    n = len(c)
    return Bars(
        interval=interval,
        open_time=T0 + np.arange(n, dtype=np.int64) * step,
        o=np.asarray(o, dtype=np.float64),
        h=np.asarray(h, dtype=np.float64),
        l=np.asarray(l, dtype=np.float64),
        c=np.asarray(c, dtype=np.float64),
        volume=np.full(n, 10.0),
        quote_volume=np.full(n, 1000.0),
        trades=np.full(n, 100.0),
        taker_buy_base=np.full(n, 5.0),
    )


def _flat_with(overrides: dict[int, tuple[float, float, float, float]], n: int = 40) -> Bars:
    """All bars o=c=100, h=101, l=99 (ATR 2) except ``overrides[i] = (o, h, l, c)``."""
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    c = np.full(n, 100.0)
    for i, (oo, hh, ll, cc) in overrides.items():
        o[i], h[i], lo[i], c[i] = oo, hh, ll, cc
    return _bars(o, h, lo, c)


DEC = 20  # decision bar; entry = open of bar 21 = 100, R = ATR 2


def test_long_target_hit_on_window_bar_gives_plus_1_5R_minus_fees():
    # bar 22 (second bar of the window, H=3) reaches 103 = entry + 1.5R
    bars = _flat_with({22: (100.0, 103.5, 99.5, 102.0)})
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    so = lab.sides["long"]
    assert so.outcome[p] == 1
    assert so.exit_bar[p] == 22
    assert so.gross[p] == pytest.approx(1.5)
    # fees: taker both sides 0.1% of 100 = 0.1 price = 0.05R
    assert so.net_taker[p] == pytest.approx(1.5 - 0.05)
    assert so.net_maker[p] == pytest.approx(1.5 - (0.0007 * 100 / 2))
    assert so.y[p] == 1


def test_long_stop_fill_is_gap_open_when_worse():
    # entry 100 (open of bar 21); bar 22 gaps down through the stop (98): open 96 -> fill 96, gross = -2R
    bars = _flat_with({22: (96.0, 96.5, 95.0, 96.0)})
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    so = lab.sides["long"]
    assert so.outcome[p] == -1
    assert so.exit_bar[p] == 22
    assert so.gross[p] == pytest.approx(-2.0)


def test_long_stop_exact_when_open_above_stop():
    bars = _flat_with({22: (100.0, 100.5, 97.5, 98.0)})
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    so = lab.sides["long"]
    assert so.outcome[p] == -1
    assert so.gross[p] == pytest.approx(-1.0)


def test_same_bar_touch_of_both_is_a_stop():
    bars = _flat_with({22: (100.0, 104.0, 97.0, 100.0)})  # reaches 103 and 98 together
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    assert lab.sides["long"].outcome[p] == -1
    assert lab.sides["long"].gross[p] == pytest.approx(-1.0)


def test_timeout_exits_at_last_close_and_can_be_a_loss_after_fees():
    # nothing touched; close at bar 23 (= i+H) is 100.4 -> gross +0.2R, net = 0.2 - 0.05
    bars = _flat_with({23: (100.0, 101.0, 99.0, 100.4)})
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    so = lab.sides["long"]
    assert so.outcome[p] == 0
    assert so.exit_bar[p] == 23
    assert so.gross[p] == pytest.approx(0.2)
    assert so.net_taker[p] == pytest.approx(0.2 - 0.05)
    assert so.y[p] == 1
    # a tiny positive timeout that does not pay fees is not a win
    bars2 = _flat_with({23: (100.0, 101.0, 99.0, 100.06)})
    lab2 = trade_labels.build(bars2, horizon=3, tp_r=1.5)
    p2 = int(np.nonzero(lab2.idx == DEC)[0][0])
    assert lab2.sides["long"].net_taker[p2] < 0
    assert lab2.sides["long"].y[p2] == 0


def test_short_mirrors_long_and_pays_funding_when_rate_positive():
    from supere.train.data import Funding

    # short target: low reaches 100 - 3 = 97
    bars = _flat_with({22: (100.0, 100.5, 96.5, 97.0)})
    # funding 0.01% settled inside (entry, exit]; a short receives it
    fund = Funding(time=np.array([T0 + 22 * H1 - 1000], dtype=np.int64), rate=np.array([0.0001]))
    lab = trade_labels.build(bars, horizon=3, tp_r=1.5, funding=fund)
    p = int(np.nonzero(lab.idx == DEC)[0][0])
    so = lab.sides["short"]
    assert so.outcome[p] == 1
    # funding R = d * rate * entry / R = -1 * 0.0001 * 100 / 2 = -0.005 -> net gets +0.005
    assert so.net_taker[p] == pytest.approx(1.5 - 0.05 + 0.005)


def test_labels_do_not_depend_on_bars_after_the_window():
    rng = np.random.default_rng(0)
    n = 120
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    h = base + rng.uniform(0.2, 1.5, n)
    lo = base - rng.uniform(0.2, 1.5, n)
    bars = _bars(base, h, lo, base + rng.normal(0, 0.3, n))
    H = 3
    full = trade_labels.build(bars, H, 1.5)
    cut = 60
    part = trade_labels.build(bars.slice_to(cut), H, 1.5)
    # decisions whose window ends before the cut must match exactly
    m = part.idx.shape[0]
    assert m > 0
    for name in ("long", "short"):
        np.testing.assert_array_equal(full.sides[name].outcome[:m], part.sides[name].outcome)
        np.testing.assert_allclose(full.sides[name].net_taker[:m], part.sides[name].net_taker)


def _random_bars(n: int, step: int, interval: str, seed: int) -> Bars:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    h = base + rng.uniform(0.2, 1.5, n)
    lo = base - rng.uniform(0.2, 1.5, n)
    return _bars(base, h, lo, base + rng.normal(0, 0.3, n), interval=interval, step=step)


def test_features_truncation_gives_identical_rows_before_the_cut():
    base = _random_bars(400, H1, "1h", 1)
    higher = _random_bars(100, H4, "4h", 2)
    names, full = trade_features.build(base, None, None, higher)
    cut = 250
    names_t, part = trade_features.build(base.slice_to(cut), None, None, higher)
    assert names == names_t
    np.testing.assert_allclose(full[:cut], part, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_higher_timeframe_only_uses_bars_already_closed():
    base = _random_bars(200, H1, "1h", 3)
    higher = _random_bars(60, H4, "4h", 4)
    names, X = trade_features.build(base, None, None, higher)
    cols = [names.index(n) for n in trade_features.MTF_NAMES]
    # changing the higher bar that is still forming at bar i must not change row i
    i = 100
    decision = base.close_time[i]
    j_open_now = int(np.searchsorted(higher.open_time, decision, side="right") - 1)  # forming or just closed
    h2 = higher.slice_to(len(higher.c))
    h2.c = h2.c.copy()
    h2.c[j_open_now] = h2.c[j_open_now] * 1.5  # its close is after the decision when it is still forming
    h2.h = h2.h.copy()
    h2.h[j_open_now] = h2.h[j_open_now] * 1.5
    _, X2 = trade_features.build(base, None, None, h2)
    if higher.close_time[j_open_now] > decision:
        np.testing.assert_allclose(X[i, cols], X2[i, cols], equal_nan=True)


def test_breakout_pullback_candidate_fires_only_on_a_held_retest():
    from supere.train import setups

    # 30 flat bars (high 101), break at bar 30 (close 103), retest at 31 touches 101 and closes 101.5
    ov = {30: (100.0, 104.0, 102.0, 103.0), 31: (103.0, 103.5, 100.5, 101.5)}
    bars = _flat_with(ov, n=40)
    lm, sm, names, block = setups.masks_and_features(bars, lookback=20, recent=10, touch_atr=0.5)
    assert lm[31] and not sm[31]
    # no touch: the low stays above the level plus 0.5 ATR, so no candidate at 31
    ov2 = {30: (100.0, 104.0, 102.0, 103.0), 31: (103.0, 104.0, 102.5, 103.5)}
    lm2, _, _, _ = setups.masks_and_features(_flat_with(ov2, n=40), lookback=20, recent=10, touch_atr=0.5)
    assert not lm2[31]
    # the break is lost: the close at 31 falls back under the level, no candidate
    ov3 = {30: (100.0, 104.0, 102.0, 103.0), 31: (103.0, 103.5, 100.5, 100.8)}
    lm3, _, _, _ = setups.masks_and_features(_flat_with(ov3, n=40), lookback=20, recent=10, touch_atr=0.5)
    assert not lm3[31]
    assert block.shape == (40, 4) and names[0] == "bp_long_dist_atr"
