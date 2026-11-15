"""Fixed entry: nearer end, breakout offset, counter midpoint, tick, and R."""

from datetime import datetime

from supere.board.plans import (
    build_plans,
    fixed_entry_price,
    round_to_tick,
    stamp_plan_times,
)
from supere.board.settings import load_analysis
from supere.timeutil import UTC8

ROOT_ANALYSIS = "analysis.toml"


def _cfg():
    from pathlib import Path

    cfg, _path = load_analysis(str(Path(__file__).resolve().parents[1] / ROOT_ANALYSIS))
    return cfg


def test_tick_rounding_uses_half_up():
    assert round_to_tick(100.004, 0.01) == 100.0
    assert round_to_tick(100.006, 0.01) == 100.01
    assert round_to_tick(100.015, 0.01) == 100.02


def test_trend_entry_is_the_end_nearer_the_key_and_the_tie_is_the_stop_side():
    # Key sits next to the low end.
    assert fixed_entry_price(
        kind="顺势", side="short", entry_low=100, entry_high=110, key=101, atr=10, tick=0.01,
    ) == 100
    # Key sits next to the high end.
    assert fixed_entry_price(
        kind="顺势", side="long", entry_low=100, entry_high=110, key=109, atr=10, tick=0.01,
    ) == 110
    # Live zone is key ± band, so the ends tie. Short sells the high end, long buys the low end.
    assert fixed_entry_price(
        kind="顺势", side="short", entry_low=107, entry_high=113, key=110, atr=10, tick=0.01,
    ) == 113
    assert fixed_entry_price(
        kind="顺势", side="long", entry_low=87, entry_high=93, key=90, atr=10, tick=0.01,
    ) == 87


def test_breakout_steps_through_the_key_and_counter_takes_the_midpoint():
    assert fixed_entry_price(
        kind="破位", side="long", entry_low=98, entry_high=102, key=100, atr=10, tick=0.01,
    ) == 101
    assert fixed_entry_price(
        kind="破位", side="short", entry_low=98, entry_high=102, key=100, atr=10, tick=0.01, offset_atr=0.1,
    ) == 99
    assert fixed_entry_price(
        kind="逆势", side="long", entry_low=87, entry_high=93, key=90, atr=10, tick=0.01,
    ) == 90
    # 100.004 - 0.1 * 1 = 99.904 → 99.90
    assert fixed_entry_price(
        kind="破位", side="short", entry_low=99, entry_high=101, key=100.004, atr=1, tick=0.01,
    ) == 99.90


def _bars(price, count=20, step=900_000):
    rows = []
    for index in range(count):
        open_time = index * step
        rows.append(
            {
                "open_time": open_time,
                "close_time": open_time + step - 1,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 10.0,
                "taker_buy": 4.0,
            }
        )
    return rows


def test_built_plans_recompute_r_from_the_fixed_entry_and_carry_the_bar_clock():
    cfg = _cfg()
    packs = {
        "15m": {"bars": _bars(105), "atr": [10.0], "rsi": [50.0], "ema": {50: []}},
        "1h": {"bars": _bars(105, step=3_600_000), "atr": [20.0], "rsi": [50.0], "ema": {20: []}},
    }
    levels = {
        "resistance": [{"price": 110.0, "sources": ["前日高"], "touches": 2}],
        "support": [{"price": 90.0, "sources": ["前日低"], "touches": 2}],
    }
    rng = {"high": 120.0, "low": 100.0, "blocked": False}
    plans, meta = build_plans(
        bias="只做空", price=105.0, levels=levels, rng=rng, packs=packs, analysis=cfg,
    )
    trend = next(plan for plan in plans if plan["id"] == "trend")
    # Zone is the key ± 0.3×ATR15. The ends tie, so a short takes the high end.
    assert trend["entry"] == round_to_tick(trend["entry_high"], cfg.plans.tick)
    assert trend["entry"] > (trend["entry_low"] + trend["entry_high"]) / 2
    assert trend["entry_rule"] == "trend_near_key"
    assert abs(trend["r_distance"] - abs(trend["entry"] - trend["stop"])) < 1e-9
    assert abs(trend["t1_r"] - (trend["entry"] - 90) / trend["r_distance"]) < 1e-9
    assert trend["calc"]["entry"] == trend["entry"]

    broke = next(plan for plan in plans if plan["id"] == "breakout")
    assert broke["entry"] == 99  # range low 100, short, 0.1 × ATR15
    assert broke["entry_rule"] == "breakout_atr"

    counter = next(plan for plan in plans if plan["id"] == "counter")
    assert counter["entry"] == 90
    assert counter["entry_rule"] == "counter_mid"

    boundary = int(datetime(2026, 11, 15, 21, 45, tzinfo=UTC8).timestamp() * 1000)
    stamp_plan_times(plans, meta, boundary_ms=boundary, valid_bars=4)
    assert trend["validity"] == "21:45 生成 · 有效至 22:45"
    assert meta["validity"] == "21:45 生成 · 有效至 22:45"
    assert meta["expires_ms"] - meta["generated_ms"] == 4 * 15 * 60 * 1000
    assert all(plan["generated_utc8"] == "21:45" for plan in plans)
