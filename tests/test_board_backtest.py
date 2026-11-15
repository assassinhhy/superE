"""Trade accounting for the board backtest. The 3-month run is a script, not this file."""

from supere.board.backtest import notify_gate, simulate_trade


def _bar(high, low, close=None):
    return {"high": high, "low": low, "close": close if close is not None else (high + low) / 2, "open_time": 0, "close_time": 0}


def test_half_at_t1_then_t2_deducts_round_trip_fee():
    bars = [
        _bar(101, 99),
        _bar(99, 96),
        _bar(95, 90),
    ]
    result = simulate_trade(
        bars, 0, side="short", entry=100, stop=102, targets=[96, 90], slippage_ticks=0,
    )
    assert result["outcome"] == "hit"
    assert result["hit_t1"] is True
    assert result["hit_t2"] is True
    assert result["exit"] == "t2"
    assert abs(result["net"] - (0.5 * 4 + 0.5 * 10 - 0.1)) < 1e-9
    assert abs(result["rr"] - result["net"] / 2) < 1e-9


def test_slippage_makes_the_same_path_worse_by_one_tick_each_side():
    bars = [_bar(101, 99), _bar(99, 96), _bar(95, 90)]
    plain = simulate_trade(bars, 0, side="short", entry=100, stop=102, targets=[96, 90], slippage_ticks=0)
    slipped = simulate_trade(bars, 0, side="short", entry=100, stop=102, targets=[96, 90], slippage_ticks=1)
    assert slipped["net"] < plain["net"]
    # Entry is 0.01 worse and each of the two exits is 0.01 worse.
    # One tick on the entry and one tick on each of the two half-exits.
    assert abs((plain["net"] - slipped["net"]) - 0.02) < 1e-9


def test_stop_before_target_is_a_loss_and_a_shared_bar_counts_as_the_stop():
    stopped = simulate_trade(
        [_bar(101, 99), _bar(103, 99)], 0, side="short", entry=100, stop=102, targets=[96], slippage_ticks=0,
    )
    assert stopped["outcome"] == "fail"
    assert abs(stopped["net"] - (-2 - 0.1)) < 1e-9
    both = simulate_trade(
        [_bar(101, 99), _bar(103, 90)], 0, side="short", entry=100, stop=102, targets=[96], slippage_ticks=0,
    )
    assert both["outcome"] == "fail"


def test_neither_target_nor_stop_inside_24h_is_invalid_and_later_bars_are_ignored():
    quiet = [_bar(100.2, 99.8, 100)] * 4
    quiet.append(_bar(110, 80, 100))
    result = simulate_trade(
        quiet, 0, side="short", entry=100, stop=102, targets=[96], horizon=3, slippage_ticks=0,
    )
    assert result["outcome"] == "invalid"
    assert notify_gate({"counted": 49, "hit_rate": 0.5, "avg_rr": 2}) is False
    assert notify_gate({"counted": 50, "hit_rate": 0.45, "avg_rr": 1.3}) is True
