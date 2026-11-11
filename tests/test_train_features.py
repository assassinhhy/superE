"""No-lookahead tests for the training features (plan 8.3).

Truncation test: the row at bar T must be identical whether the code sees the
whole history or only bars up to T, with funding and metrics filtered to what
was public at T's close. Future-change test: rewriting anything after T must
leave row T unchanged.
"""

from __future__ import annotations

import numpy as np
from synthetic import START_MS, make_bar_arrays

from supere.intervals import INTERVAL_MS
from supere.train import features
from supere.train.data import Bars, Funding, Metrics


def _bars(n: int, seed: int = 11, interval: str = "12h") -> Bars:
    a = make_bar_arrays(n, seed=seed, interval=interval)
    return Bars(
        interval=interval,
        open_time=a["open_time"],
        o=a["o"],
        h=a["h"],
        l=a["l"],
        c=a["c"],
        volume=a["volume"],
        quote_volume=a["quote_volume"],
        trades=a["trades"],
        taker_buy_base=a["taker_buy_base"],
    )


def _funding(bars: Bars, seed: int = 3) -> Funding:
    rng = np.random.default_rng(seed)
    span = bars.open_time[-1] + bars.interval_ms
    times = np.arange(START_MS, span, 8 * 3_600_000, dtype=np.int64)
    return Funding(times, rng.normal(0.0001, 0.0002, times.shape[0]))


def _metrics(bars: Bars, seed: int = 5) -> Metrics:
    rng = np.random.default_rng(seed)
    span = bars.open_time[-1] + bars.interval_ms
    ts = np.arange(START_MS, span, 300_000, dtype=np.int64)
    m = ts.shape[0]
    base = 1.0 + 0.1 * rng.standard_normal((m, 4))
    oi = 1e5 * np.exp(np.cumsum(rng.normal(0, 0.002, m)))
    # Sprinkle NULLs (NaN) to exercise the optional-field path.
    mask = rng.random((m, 4)) < 0.02
    base[mask] = np.nan
    return Metrics(ts, oi, base[:, 0], base[:, 1], base[:, 2], base[:, 3])


def _slice_funding(f: Funding, cutoff: int) -> Funding:
    keep = f.time < cutoff  # settlements strictly before the decision moment
    return Funding(f.time[keep], f.rate[keep])


def _slice_metrics(m: Metrics, cutoff: int, interval_ms: int) -> Metrics:
    known = m.ts + features.METRICS_DELAY_MS + interval_ms
    keep = known <= cutoff
    return Metrics(m.ts[keep], m.oi[keep], m.top_pos_lsr[keep], m.top_acc_lsr[keep],
                   m.global_lsr[keep], m.taker_lsr[keep])


def test_truncated_history_gives_identical_row_at_T():
    bars = _bars(400)
    fund = _funding(bars)
    met = _metrics(bars)
    names, full = features.build(bars, fund, met)

    for T in (80, 150, 233, 399):
        cutoff = int(bars.close_time[T])
        names_t, part = features.build(
            bars.slice_to(T + 1),
            _slice_funding(fund, cutoff),
            _slice_metrics(met, cutoff, bars.interval_ms),
        )
        assert names_t == names
        np.testing.assert_array_equal(part[T], full[T], err_msg=f"T={T}")


def test_rewriting_the_future_does_not_change_row_T():
    bars = _bars(300)
    fund = _funding(bars)
    met = _metrics(bars)
    names, base = features.build(bars, fund, met)

    T = 120
    cutoff = int(bars.close_time[T])
    future = bars.slice_to(bars.n)
    for field in ("o", "h", "l", "c", "volume", "quote_volume", "trades", "taker_buy_base"):
        arr = getattr(future, field).copy()  # slices are views; never touch the originals
        arr[T + 1 :] = arr[T + 1 :] * 3.7 + 11.0
        setattr(future, field, arr)
    future_fund = Funding(fund.time.copy(), fund.rate.copy())
    late = future_fund.time >= cutoff
    future_fund.rate[late] = 0.5  # huge rates after T must not leak back
    future_met = Metrics(met.ts, met.oi.copy(), met.top_pos_lsr.copy(), met.top_acc_lsr.copy(),
                         met.global_lsr.copy(), met.taker_lsr.copy())
    late_m = met.ts + features.METRICS_DELAY_MS + bars.interval_ms > cutoff
    future_met.oi[late_m] *= 10.0
    future_met.global_lsr[late_m] = 99.0

    _, changed = features.build(future, future_fund, future_met)
    np.testing.assert_array_equal(changed[T], base[T])


def test_funding_at_the_decision_moment_is_not_used():
    bars = _bars(60)
    decision = int(bars.close_time[30])
    fund = Funding(np.array([decision], dtype=np.int64), np.array([0.01]))
    names, X = features.build(bars, fund, None)
    col = names.index("funding_last")
    assert np.isnan(X[30, col])  # settlement exactly at the close: not yet known
    assert np.isnan(X[29, col])
    assert X[31, col] == 0.01  # from the next bar onwards it is known


def test_metrics_wait_for_five_minutes_plus_one_interval():
    bars = _bars(60)
    ts0 = int(bars.open_time[20])  # a metrics bar that starts at bar 20's open
    met = Metrics(
        np.array([ts0], dtype=np.int64),
        np.array([100.0]),
        np.array([1.1]),
        np.array([1.2]),
        np.array([1.3]),
        np.array([1.4]),
    )
    names, X = features.build(bars, None, met)
    col = names.index("um_global_lsr")
    usable = ts0 + features.METRICS_DELAY_MS + bars.interval_ms
    for i in range(bars.n):
        decision = int(bars.close_time[i])
        if decision >= usable:
            assert X[i, col] == 1.3, f"bar {i} should see the metrics row"
        else:
            assert np.isnan(X[i, col]), f"bar {i} must not see it yet"


def test_null_metrics_become_nan_not_errors():
    bars = _bars(120)
    met = _metrics(bars)
    met.global_lsr[:] = np.nan
    names, X = features.build(bars, _funding(bars), met)
    assert np.all(np.isnan(X[:, names.index("um_global_lsr")]))
    assert np.isfinite(X[:, names.index("ret_1")][1:]).all()


def test_core_features_have_warmup_then_are_finite():
    bars = _bars(200)
    names, X = features.build(bars, None, None)
    core = [names.index(c) for c in features.CORE_FEATURES]
    first_ok = int(np.argmax(np.all(np.isfinite(X[:, core]), axis=1)))
    assert 20 <= first_ok <= 60  # rolling windows need some history, not too much
    assert np.all(np.isfinite(X[first_ok:, core]))


def test_time_features_use_utc8_decision_moment():
    bars = _bars(4)
    names, X = features.build(bars, None, None)
    hour = X[:, names.index("hour_utc8")]
    # 12h bars close at 12:00 and 00:00 UTC, i.e. 20:00 and 08:00 UTC+8.
    assert set(hour.tolist()) <= {8.0, 20.0}


def test_feature_hash_changes_with_settings():
    names = ["a", "b"]
    assert features.feature_hash(names, {"h": 3}) != features.feature_hash(names, {"h": 4})
    assert features.feature_hash(names) == features.feature_hash(names)


def test_interval_ms_sanity():
    # Guards the tests above: they assume 12h bars.
    assert INTERVAL_MS["12h"] == 43_200_000
