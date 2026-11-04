"""Backtest accounting, configuration rules, and read-only database access."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from synthetic import build_synthetic_db

from supere.train import backtest, config
from supere.train.config import ConfigError
from supere.train.data import Funding, open_readonly

INTERVAL_MS = 43_200_000
T0 = 1_577_836_800_000
NO_FUNDING = Funding.empty()
TAKER = backtest.FeeScenario("taker_both", 0.0005, 0.0005)
MAKER = backtest.FeeScenario("maker_entry", 0.0002, 0.0005)


def _one_trade(**kw):
    args = dict(
        sample_bar=np.array([5]),
        exit_bar=np.array([7]),
        entry=np.array([100.0]),
        exit_price=np.array([102.0]),
        upper=np.array([102.0]),
        lower=np.array([98.0]),
        ambiguous=np.array([False]),
        dirs=np.array([1]),
        open_time=T0 + np.arange(20, dtype=np.int64) * INTERVAL_MS,
        interval_ms=INTERVAL_MS,
        funding=NO_FUNDING,
        scenario=TAKER,
        slippage_per_side=0.0,
    )
    args.update(kw)
    return backtest.simulate(**args)


# ---- backtest ----------------------------------------------------------------

def test_long_trade_net_is_gross_minus_both_taker_fees():
    (t,) = _one_trade()
    assert t["gross"] == pytest.approx(0.02)
    assert t["net"] == pytest.approx(0.02 - 0.001)


def test_maker_entry_scenario_charges_the_lower_entry_fee():
    (t,) = _one_trade(scenario=MAKER)
    assert t["fees"] == pytest.approx(0.0002 + 0.0005)


def test_short_trade_loses_when_price_rises():
    (t,) = _one_trade(dirs=np.array([-1]), exit_price=np.array([102.0]))
    assert t["gross"] == pytest.approx(-0.02)


def test_funding_is_paid_by_longs_and_received_by_shorts():
    # entry time = open of bar 6, exit time = close of bar 7; the settlement sits inside.
    entry_time = T0 + 6 * INTERVAL_MS
    fund = Funding(np.array([entry_time + 1000], dtype=np.int64), np.array([0.001]))
    (long_t,) = _one_trade(funding=fund)
    (short_t,) = _one_trade(funding=fund, dirs=np.array([-1]))
    assert long_t["funding"] == pytest.approx(0.001)
    assert short_t["funding"] == pytest.approx(-0.001)
    assert long_t["net"] == pytest.approx(0.019 - 0.001)


def test_settlement_exactly_at_entry_is_not_charged():
    entry_time = T0 + 6 * INTERVAL_MS
    fund = Funding(np.array([entry_time], dtype=np.int64), np.array([0.5]))
    (t,) = _one_trade(funding=fund)
    assert t["funding"] == 0.0


def test_only_one_position_at_a_time():
    trades = backtest.simulate(
        sample_bar=np.array([5, 6, 9]),
        exit_bar=np.array([9, 8, 11]),
        entry=np.array([100.0, 100.0, 100.0]),
        exit_price=np.array([101.0, 101.0, 101.0]),
        upper=np.array([101.0, 101.0, 101.0]),
        lower=np.array([99.0, 99.0, 99.0]),
        ambiguous=np.zeros(3, dtype=bool),
        dirs=np.array([1, 1, 1]),
        open_time=T0 + np.arange(20, dtype=np.int64) * INTERVAL_MS,
        interval_ms=INTERVAL_MS,
        funding=NO_FUNDING,
        scenario=TAKER,
        slippage_per_side=0.0,
    )
    # Sample 5 opens bar 6 and holds to bar 9. Sample 6 wants bar 7: blocked.
    # Sample 9 wants bar 10: free again.
    assert [t["entry_bar"] for t in trades] == [6, 10]


def test_slippage_is_charged_on_both_sides():
    (t,) = _one_trade(slippage_per_side=0.0005)
    assert t["slippage"] == pytest.approx(0.001)
    assert t["net"] == pytest.approx(0.02 - 0.001 - 0.001)


def test_signal_dirs_respects_threshold_and_tie_rule():
    proba = np.array(
        [
            [0.10, 0.20, 0.70],  # long
            [0.70, 0.20, 0.10],  # short
            [0.45, 0.10, 0.45],  # tie, neither side wins
            [0.30, 0.40, 0.30],  # below threshold
        ]
    )
    assert backtest.signal_dirs(proba, 0.40).tolist() == [1, -1, 0, 0]


def test_summary_drawdown_and_losing_streak():
    trades = [{"net": r, "gross": r, "funding": 0.0, "ambiguous": False} for r in (0.10, -0.05, -0.05, 0.02)]
    s = backtest.summarize(trades, n_samples=8)
    assert s["trades"] == 4
    assert s["coverage"] == pytest.approx(0.5)
    assert s["max_consecutive_losses"] == 2
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["max_drawdown"] < 0


def test_empty_trade_list_is_nan_not_zero_division():
    s = backtest.summarize([], n_samples=10)
    assert s["trades"] == 0
    assert np.isnan(s["hit_rate"])


# ---- configuration -----------------------------------------------------------

def test_default_is_12h_with_horizon_3():
    cfg = config.build()
    assert cfg.interval == "12h"
    assert cfg.label.horizon == 3


@pytest.mark.parametrize("interval, horizon", [("1h", 4), ("15m", 4)])
def test_alternative_intervals_get_their_own_defaults(interval, horizon):
    assert config.build(interval=interval).label.horizon == horizon


@pytest.mark.parametrize("interval", ["1d", "1w", "1M", "3d"])
def test_non_trainable_intervals_are_refused(interval):
    with pytest.raises(ConfigError):
        config.build(interval=interval)


def test_partial_label_override_keeps_interval_default():
    cfg = config.build(interval="1h", overrides={"label": {"k_atr": 1.5}})
    assert cfg.label.k_atr == 1.5
    assert cfg.label.horizon == 4  # not reset to the 12h value


def test_bad_split_mode_is_refused():
    with pytest.raises(ConfigError):
        config.build(overrides={"split": {"mode": "shuffle"}})


def test_train_toml_values_apply_and_flags_win():
    file_values = {"train": {"interval": "1h", "threshold": 0.5}}
    cfg = config.build(file_values=file_values, overrides={"train": {"threshold": 0.45}})
    assert cfg.interval == "1h"
    assert cfg.threshold == 0.45


def test_repo_train_toml_loads_and_validates():
    repo = Path(__file__).resolve().parents[1]
    cfg = config.build(file_values=config.load_file(repo / "train.toml"))
    assert cfg.interval == "12h"
    assert cfg.label.horizon == 3


# ---- read-only access --------------------------------------------------------

def test_training_connection_cannot_write(tmp_path: Path):
    db = build_synthetic_db(tmp_path / "superE.sqlite", n=60, with_metrics=False)
    conn = open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM klines")
    finally:
        conn.close()


def test_missing_database_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        open_readonly(tmp_path / "nope.sqlite")


def test_ambiguous_bar_takes_the_adverse_fill_for_each_side():
    # Entry 100, barriers 102 / 98, both touched in one bar.
    # A long exits at the lower barrier (-2%); a short exits at the upper one (-2%).
    amb = dict(ambiguous=np.array([True]), exit_price=np.array([98.0]),
               upper=np.array([102.0]), lower=np.array([98.0]))
    (long_t,) = _one_trade(dirs=np.array([1]), **amb)
    (short_t,) = _one_trade(dirs=np.array([-1]), **amb)
    assert long_t["gross"] == pytest.approx(-0.02)
    assert short_t["gross"] == pytest.approx(-0.02)
