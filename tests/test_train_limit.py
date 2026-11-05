"""Limit-entry pullback labels (batch 2): hand-checked fill rule, partial exit, breakeven, gaps, fees."""

from __future__ import annotations

import numpy as np
import pytest

from supere.train import trade_features, trade_limit
from supere.train.data import Bars, Funding
from supere.train.trade_labels import FEE_MAKER, FEE_TAKER

T0 = 1_577_836_800_000
H1 = 3_600_000
DEC = 20  # decision bar: close 100, ATR 2 -> R = 2, pb 0.5 -> long limit 99, short limit 101


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


def _flat_with(overrides, n=40) -> Bars:
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    c = np.full(n, 100.0)
    for i, (oo, hh, ll, cc) in overrides.items():
        o[i], h[i], lo[i], c[i] = oo, hh, ll, cc
    return _bars(o, h, lo, c)


def _run(bars, tp_r=2.0, horizon=3, wait=2, pb=0.5, funding=None):
    return trade_limit.build(bars, horizon, tp_r, k_sl=1.0, pb_atr=pb, wait=wait, funding=funding)


def _row(lab):
    return int(np.nonzero(lab.idx == DEC)[0][0])


def test_touch_of_the_limit_does_not_fill():
    # flat bars: low = 99 = limit exactly -> "touch" only, no fill
    lab = _run(_flat_with({}))
    p = _row(lab)
    so = lab.sides["long"]
    assert not so.filled[p]
    assert so.net_taker[p] == 0.0 and so.gross[p] == 0.0
    assert so.outcome[p] == trade_limit.OUT_NOFILL
    assert so.y[p] == 0


def test_strict_pass_through_fills_at_limit_then_partial_then_breakeven():
    # 21: low 98.5 < 99 -> fill at 99 (open 100 is above the limit, so the limit price)
    # 22: high 101 = fill + 1R -> first target 50% at 101; stop moves to breakeven 99
    # 23: low 99 touches breakeven -> rest exits at 99
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (100.0, 101.0, 99.5, 100.5),
                       23: (100.0, 100.5, 99.0, 100.4)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.filled[p] and so.fill_bar[p] == 21
    assert so.outcome[p] == trade_limit.OUT_BE
    assert so.exit_bar[p] == 23
    # weighted exit notional = 0.5*101 + 0.5*99 = 100 -> gross = (100-99)/2 = +0.5R
    assert so.gross[p] == pytest.approx(0.5)
    # taker both sides: entry 0.0005*99/2, exit 0.0005*100/2
    fee_in = FEE_TAKER * 99.0 / 2.0
    fee_out = FEE_TAKER * 100.0 / 2.0
    assert so.net_taker[p] == pytest.approx(0.5 - fee_in - fee_out)
    assert so.net_maker[p] == pytest.approx(0.5 - FEE_MAKER * 99.0 / 2.0 - fee_out)


def test_gap_through_the_limit_fills_at_the_open():
    # 21 opens at 98, below the limit 99: fill at 98 (better), not at 99
    bars = _flat_with({21: (98.0, 98.5, 97.5, 98.2)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.filled[p]
    # fill 98 (the open), stop 96, first target 100 (bar 22 high 101 reaches it), breakeven 98.
    # bar 23: low 99 stays above 98 and high 101 stays below the final target 102 -> timeout at close 100.
    # notional = 0.5*100 + 0.5*100 = 100 -> gross = (100-98)/2 = +1R
    assert so.fill_bar[p] == 21
    assert so.outcome[p] == trade_limit.OUT_TIME
    assert so.gross[p] == pytest.approx(1.0)


def test_stop_before_first_target_is_full_loss_and_gap_aware():
    # fill at 99 on bar 21; bar 22 gaps to 95 with open 97 -> stop at 97 (gap below the stop: open 97 is the stop)
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (96.0, 100.0, 95.0, 96.5)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.outcome[p] == trade_limit.OUT_STOP
    assert so.exit_bar[p] == 22
    # stop at 97 (open 96 is below it -> gap fill at the open 96): gross = (96-99)/2 = -1.5R
    assert so.gross[p] == pytest.approx(-1.5)


def test_final_target_after_first_target_gives_blended_R():
    # fill 99 on 21; 22 tags 101 (first target); 23 tags 103 = 99 + 2R (final target)
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (100.0, 101.0, 99.5, 100.5),
                       23: (100.0, 103.0, 100.0, 102.0)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.outcome[p] == trade_limit.OUT_TP2
    # 0.5*101 + 0.5*103 = 102 -> gross (102-99)/2 = 1.5R
    assert so.gross[p] == pytest.approx(1.5)


def test_timeout_after_first_target_exits_the_rest_at_the_close():
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (100.0, 101.0, 99.5, 100.5),
                       23: (100.0, 100.5, 99.5, 100.4)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.outcome[p] == trade_limit.OUT_TIME
    assert so.exit_bar[p] == 23
    # 0.5*101 + 0.5*100.4 = 100.7 -> gross (100.7-99)/2 = 0.85R
    assert so.gross[p] == pytest.approx(0.85)


def test_stop_checked_before_target_in_the_same_bar():
    # bar 22 reaches both the stop (97) and the first target (101): the stop is taken (conservative)
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (100.0, 101.0, 96.0, 100.0)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.outcome[p] == trade_limit.OUT_STOP
    assert so.gross[p] == pytest.approx(-1.0)


def test_first_target_is_not_taken_in_the_fill_bar():
    # bar 21 fills at 99 and its high (103) would be a target, but the 1R target is only counted
    # from the next bar; bar 21 has low 98.5 -> no stop, so the trade continues to timeout
    bars = _flat_with({21: (100.0, 103.0, 98.5, 102.0)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["long"]
    assert so.filled[p]
    assert so.outcome[p] != trade_limit.OUT_TP2
    assert so.exit_bar[p] == 23


def test_short_mirror_fills_on_pass_through_above_the_limit():
    # short limit = 100 + 1 = 101; bar 21 high 101.5 > 101 -> fill at max(open 100, 101) = 101
    # first target 99 (1R, bar 22 low 98.5); final 97 (2R, bar 23 low 97.5); blended exit 98 -> gross +1.5R
    bars = _flat_with({21: (100.0, 101.5, 99.5, 100.5), 22: (100.0, 100.5, 98.5, 99.5),
                       23: (100.0, 100.5, 96.5, 97.5)})
    lab = _run(bars)
    p = _row(lab)
    so = lab.sides["short"]
    assert so.filled[p] and so.fill_bar[p] == 21
    assert so.outcome[p] == trade_limit.OUT_TP2
    assert so.gross[p] == pytest.approx(1.5)


def test_funding_is_charged_on_the_fill_to_exit_window_with_sign():
    # long pays positive funding: 0.01% inside (fill, exit] -> -d * rate * F / R = -0.0001*99/2
    bars = _flat_with({21: (100.0, 100.5, 98.5, 99.5), 22: (100.0, 101.0, 99.5, 100.5),
                       23: (100.0, 100.5, 99.0, 100.4)})
    fund = Funding(time=np.array([T0 + 22 * H1 + 1000], dtype=np.int64), rate=np.array([0.0001]))
    p = _row(_run(bars, funding=fund))
    base = _row(_run(bars))
    lab_f = _run(bars, funding=fund)
    lab_0 = _run(bars)
    assert lab_f.sides["long"].net_taker[p] == pytest.approx(lab_0.sides["long"].net_taker[base] - 0.0001 * 99.0 / 2.0)


def test_exits_stay_inside_the_window_for_late_fills():
    # Regression: a late fill (bar i+2) once read bar 0 data after the window (exit_bar became 0).
    rng = np.random.default_rng(7)
    n = 300
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    bars = _bars(base, base + rng.uniform(0.2, 2.5, n), base - rng.uniform(0.2, 2.5, n),
                 base + rng.normal(0, 0.4, n))
    H = 4
    lab = _run(bars, tp_r=2.0, horizon=H, wait=3)
    for name, d in (("long", 1), ("short", -1)):
        so = lab.sides[name]
        f = so.filled
        assert f.sum() > 50
        assert np.all(so.exit_bar[f] >= so.fill_bar[f])
        assert np.all(so.exit_bar[f] <= lab.idx[f] + H)
        assert np.all(so.exit_bar[~f] == lab.idx[~f] + H)
        assert np.all(np.isfinite(so.net_taker))


def test_labels_do_not_depend_on_bars_after_the_window():
    rng = np.random.default_rng(0)
    n = 160
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    h = base + rng.uniform(0.2, 1.5, n)
    lo = base - rng.uniform(0.2, 1.5, n)
    bars = _bars(base, h, lo, base + rng.normal(0, 0.3, n))
    H = 4
    full = _run(bars, tp_r=2.0, horizon=H)
    cut = 90
    part = _run(bars.slice_to(cut), tp_r=2.0, horizon=H)
    m = part.idx.shape[0]
    assert m > 0
    for name in ("long", "short"):
        for field in ("net_taker", "outcome", "filled", "fill_bar"):
            np.testing.assert_array_equal(getattr(full.sides[name], field)[:m],
                                          getattr(part.sides[name], field))


def test_regime_features_are_causal_and_adx_is_bounded():
    rng = np.random.default_rng(3)
    n = 500
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    h4 = _bars(base, base + 1.0, base - 1.0, base + rng.normal(0, 0.2, n), interval="4h", step=4 * H1)
    x = trade_features.adx(h4)
    valid = x[np.isfinite(x)]
    assert valid.size > 0 and np.all((valid >= 0) & (valid <= 100))
    # truncation: bars before the cut are unchanged
    base1 = _random_bars(600, H1, "1h", 5)
    names, full = trade_features.build(base1, None, None, h4, use_regime=True)
    _, part = trade_features.build(base1.slice_to(450), None, None, h4, use_regime=True)
    cols = [names.index(nm) for nm in trade_features.REGIME_NAMES]
    np.testing.assert_allclose(full[:450][:, cols], part[:, cols], equal_nan=True, rtol=1e-12, atol=1e-12)


def _random_bars(n, step, interval, seed) -> Bars:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    h = base + rng.uniform(0.2, 1.5, n)
    lo = base - rng.uniform(0.2, 1.5, n)
    return _bars(base, h, lo, base + rng.normal(0, 0.3, n), interval=interval, step=step)
