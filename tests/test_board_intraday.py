"""v2 intraday rules, the trade simulator, and the two 2026-11-14 replays."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from supere.board.backtest import judge_intraday, simulate_v2
from supere.board.intraday import build_plans_v2, entries_open, session_key
from supere.board.replay import replay_at
from supere.board.settings import AnalysisConfig, load_analysis

UTC8 = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "board"
AT_1109 = int(datetime(2026, 11, 14, 11, 9, tzinfo=UTC8).timestamp() * 1000)
AT_1351 = int(datetime(2026, 11, 14, 13, 51, tzinfo=UTC8).timestamp() * 1000)


def _cfg() -> AnalysisConfig:
    cfg, _path = load_analysis(str(ROOT / "analysis.toml"))
    cfg.board.profile = "intraday_v2"
    cfg.intraday.session_enabled.asia = True
    cfg.intraday.session_enabled.europe = True
    cfg.intraday.session_enabled.us = True
    return cfg


def _ms(day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 11, day, hour, minute, tzinfo=UTC8).timestamp() * 1000)


def _bars(count: int, price: float, *, start: int, step: int = 900_000, volume: float = 10.0, high=None, low=None):
    bars = []
    for index in range(count):
        open_time = start + index * step
        bars.append(
            {
                "open_time": open_time,
                "close_time": open_time + step - 1,
                "open": price,
                "high": price + 0.4 if high is None else high,
                "low": price - 0.4 if low is None else low,
                "close": price,
                "volume": volume,
                "taker_buy": volume * 0.5,
            }
        )
    return bars


def _pack(bars, atr: float = 10.0, ema: float | None = None):
    value = bars[-1]["close"] if ema is None else ema
    length = len(bars)
    return {
        "bars": bars,
        "closes": [bar["close"] for bar in bars],
        "ema": {20: [value] * length, 50: [value] * length, 200: [value] * length},
        "rsi": [50.0] * length,
        "hist": [0.0] * length,
        "atr": [atr] * length,
    }


def _plans(**kwargs):
    cfg = kwargs.pop("analysis", _cfg())
    packs = kwargs.pop("packs")
    return build_plans_v2(
        bias=kwargs.pop("bias", "只做空"),
        price=kwargs.pop("price", packs["15m"]["bars"][-1]["close"]),
        levels=kwargs.pop("levels", {"resistance": [], "support": [], "all": []}),
        rng=kwargs.pop("rng", None),
        packs=packs,
        analysis=cfg,
        frames=kwargs.pop("frames", {"1h": "空", "4h": "空"}),
        now_ms=kwargs.pop("now_ms", packs["15m"]["bars"][-1]["close_time"]),
        metrics=kwargs.pop("metrics", {"status": "missing", "rows": []}),
        book=kwargs.pop("book", {}),
        daily_rsi=kwargs.pop("daily_rsi", None),
        weekend=kwargs.pop("weekend", False),
        event_block=kwargs.pop("event_block", False),
    )


def test_sessions_are_left_closed_and_new_plans_stop_at_1am():
    cfg = _cfg().intraday
    assert session_key(_ms(12, 8, 0), cfg) == "asia"
    assert session_key(_ms(12, 14, 59), cfg) == "asia"
    assert session_key(_ms(12, 15, 0), cfg) == "europe"
    assert session_key(_ms(12, 20, 29), cfg) == "europe"
    assert session_key(_ms(12, 20, 30), cfg) == "us"
    assert session_key(_ms(13, 1, 59), cfg) == "us"
    assert session_key(_ms(13, 2, 0), cfg) is None
    assert entries_open(_ms(12, 0, 59), cfg) is True
    assert entries_open(_ms(12, 1, 0), cfg) is False
    assert entries_open(_ms(12, 7, 59), cfg) is False
    assert entries_open(_ms(12, 8, 0), cfg) is True


def test_opposite_frames_make_no_plan():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start)
    packs = {"15m": _pack(bars), "1h": _pack(bars)}
    plans, meta = _plans(packs=packs, frames={"1h": "多", "4h": "空"})
    assert plans == []
    assert meta["pause_reason"] == "方向打架，不做"


def test_one_hour_range_only_allows_range_plans():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start)
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=10)}
    wide, _meta = _plans(
        packs=packs,
        frames={"1h": "震荡", "4h": "空"},
        rng={"high": 140, "low": 100, "width": 40},
        price=100,
    )
    assert wide
    assert {plan["kind"] for plan in wide} <= {"区间", "翻转"}
    assert any(plan["kind"] == "区间" for plan in wide)
    narrow, _meta = _plans(
        packs=packs,
        frames={"1h": "震荡", "4h": "震荡"},
        rng={"high": 110, "low": 100, "width": 10},
        price=100,
    )
    assert all(plan["kind"] != "区间" for plan in narrow)


def test_range_target_inside_the_minimum_is_marked_no_room():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start)
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=10)}
    cfg = _cfg()
    cfg.intraday.t1_min_r = 5
    plans, _meta = _plans(
        packs=packs,
        analysis=cfg,
        frames={"1h": "震荡", "4h": "震荡"},
        rng={"high": 130, "low": 100, "width": 30},
        price=100,
    )
    ranged = [plan for plan in plans if plan["kind"] == "区间"]
    assert ranged
    assert all(plan["status"] == "空间不够" for plan in ranged)
    assert all(plan["confirm"]["ok"] is False for plan in ranged)


def test_trend_target_is_at_least_the_threshold():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start)
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=20)}
    levels = {
        "resistance": [{"price": 110, "sources": ["摆动高"], "touches": 1}],
        "support": [{"price": 99, "sources": ["摆动低"], "touches": 1}],
        "all": [],
    }
    plans, _meta = _plans(packs=packs, levels=levels, price=100)
    trend = next(plan for plan in plans if plan["kind"] == "顺势")
    assert trend["t1_r"] >= 1.2 - 1e-9
    assert trend["t2_r"] >= 2 - 1e-9
    assert trend["hold_until_utc8"]
    assert trend["r"] > 0


def test_direct_breakout_needs_volume_and_rising_oi():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start, volume=10)
    bars[-1] = {**bars[-1], "open": 100, "high": 101, "low": 96, "close": 97, "volume": 20}
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=10)}
    rng = {"high": 120, "low": 100, "width": 20}
    levels = {
        "resistance": [{"price": 120, "sources": ["摆动高"], "touches": 1}],
        "support": [{"price": 90, "sources": ["摆动低"], "touches": 1}],
        "all": [],
    }
    waiting, _meta = _plans(packs=packs, levels=levels, rng=rng, metrics={"status": "missing", "rows": []})
    breakout = next(plan for plan in waiting if plan["kind"] == "破位")
    assert breakout["status"] == "等待信号"
    rows = [
        {"ts": bars[-2]["close_time"] - 300_000, "known_at": bars[-2]["close_time"], "oi": 100.0},
        {"ts": bars[-1]["close_time"] - 300_000, "known_at": bars[-1]["close_time"], "oi": 110.0},
    ]
    fired, _meta = _plans(
        packs=packs, levels=levels, rng=rng, metrics={"status": "ok", "rows": rows}
    )
    live = next(plan for plan in fired if plan["kind"] == "破位")
    assert live["status"] == "可进场"
    assert live["confirm"]["names"] == ["直接跟进"]
    assert live["stop"] > rng["low"]


def test_retest_within_four_bars_can_enter():
    start = _ms(12, 10)
    bars = _bars(30, 110, start=start)
    bars[-3] = {**bars[-3], "open": 101, "high": 102, "low": 90, "close": 95, "volume": 10}
    bars[-1] = {**bars[-1], "open": 106, "high": 108, "low": 105, "close": 107, "volume": 10}
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=10)}
    rng = {"high": 120, "low": 100, "width": 20}
    levels = {
        "resistance": [{"price": 130, "sources": ["摆动高"], "touches": 1}],
        "support": [{"price": 80, "sources": ["摆动低"], "touches": 1}],
        "all": [],
    }
    plans, _meta = _plans(packs=packs, levels=levels, rng=rng, metrics={"status": "missing", "rows": []})
    breakout = next(plan for plan in plans if plan["kind"] == "破位")
    assert breakout["status"] == "可进场"
    assert "回抽" in "".join(breakout["confirm"]["names"])


def test_counter_needs_every_condition_and_only_one_per_day():
    start = _ms(12, 10)
    bars = _bars(30, 100, start=start, volume=10, high=101, low=99)
    bars[-1] = {**bars[-1], "open": 100, "high": 102, "low": 90, "close": 101, "volume": 30}
    packs = {"15m": _pack(bars, atr=10), "1h": _pack(bars, atr=10)}
    level = {"price": 95, "sources": ["插针扫过"], "touches": 3}
    levels = {
        "resistance": [{"price": 120, "sources": ["摆动高"], "touches": 1}],
        "support": [level],
        "all": [level, {"price": 120, "sources": ["摆动高"], "touches": 1}],
    }
    quiet = [
        {"ts": 1, "known_at": bars[-2]["close_time"], "oi": 100.0},
        {"ts": 2, "known_at": bars[-1]["close_time"], "oi": 90.0},
    ]
    blocked, _meta = _plans(packs=packs, levels=levels, metrics={"status": "ok", "rows": quiet}, daily_rsi=40)
    assert all(plan["kind"] != "逆势" for plan in blocked)
    plans, _meta = _plans(packs=packs, levels=levels, metrics={"status": "ok", "rows": quiet}, daily_rsi=20)
    counter = next(plan for plan in plans if plan["kind"] == "逆势")
    assert counter["status"] == "可进场"
    assert counter["full_at_t1"] is True
    assert counter["size_ratio"] == 0.5
    assert len(counter["targets"]) == 1
    again, meta = _plans(
        packs=packs,
        levels=levels,
        metrics={"status": "ok", "rows": quiet},
        daily_rsi=20,
        book={"counter": 1, "day_r": 0, "loss_streak": 0, "open": 0},
    )
    assert all(plan["kind"] != "逆势" for plan in again)
    stopped, meta = _plans(
        packs=packs,
        levels=levels,
        metrics={"status": "ok", "rows": quiet},
        daily_rsi=20,
        book={"counter": 0, "day_r": -3, "loss_streak": 0, "open": 0},
    )
    assert meta["pause_reason"]
    assert all(plan["status"] != "可进场" for plan in stopped)


def test_closed_hours_and_a_disabled_session_block_new_plans():
    late = _bars(20, 100, start=_ms(12, 1))
    packs = {"15m": _pack(late), "1h": _pack(late)}
    _plans_late, meta = _plans(packs=packs)
    assert "不开新单" in meta["pause_reason"]
    evening = _bars(8, 100, start=_ms(12, 20, 30))
    packs = {"15m": _pack(evening), "1h": _pack(evening)}
    cfg = _cfg()
    cfg.intraday.session_enabled.us = False
    _none, meta = _plans(packs=packs, analysis=cfg)
    assert "美国" in meta["pause_reason"]
    assert "已关闭" in meta["pause_reason"]


def test_time_stop_flat_and_the_extra_stop_tick():
    start = _ms(12, 10)
    quiet = []
    for index in range(20):
        open_time = start + index * 300_000
        quiet.append(
            {
                "open_time": open_time,
                "close_time": open_time + 300_000 - 1,
                "open": 100,
                "high": 100.4,
                "low": 99.8,
                "close": 100.1,
                "volume": 1,
                "taker_buy": 1,
            }
        )
    stopped = simulate_v2(
        quiet,
        entry_ms=start,
        side="long",
        entry=100,
        stop=98,
        targets=[110, 120],
        time_stop_ms=start + 6 * 15 * 60_000,
        max_hold_ms=start + 24 * 60 * 60_000,
        flat_ms=start + 24 * 60 * 60_000,
    )
    assert stopped["exit_reason"] == "time_stop"
    assert stopped["outcome"] == "fail"

    spike = [
        {
            "open_time": start,
            "close_time": start + 300_000 - 1,
            "open": 100,
            "high": 103,
            "low": 99,
            "close": 100,
            "volume": 1,
            "taker_buy": 1,
        }
    ]
    plain = simulate_v2(
        spike, entry_ms=start, side="short", entry=100, stop=102, targets=[90],
        time_stop_ms=start + 86_400_000, max_hold_ms=start + 86_400_000, flat_ms=start + 86_400_000,
        stop_extra_ticks=0,
    )
    worse = simulate_v2(
        spike, entry_ms=start, side="short", entry=100, stop=102, targets=[90],
        time_stop_ms=start + 86_400_000, max_hold_ms=start + 86_400_000, flat_ms=start + 86_400_000,
        stop_extra_ticks=1,
    )
    assert plain["exit_reason"] == "stop"
    assert worse["own_r"] < plain["own_r"]
    assert abs((plain["net"] - worse["net"]) - 0.01) < 1e-9

    flat_at = _ms(12, 2)
    entry_ms = _ms(12, 1, 30)
    overnight = []
    for index in range(8):
        open_time = entry_ms + index * 300_000
        overnight.append(
            {
                "open_time": open_time,
                "close_time": open_time + 300_000 - 1,
                "open": 100,
                "high": 100.2,
                "low": 99.8,
                "close": 100,
                "volume": 1,
                "taker_buy": 1,
            }
        )
    flat = simulate_v2(
        overnight,
        entry_ms=entry_ms,
        side="long",
        entry=100,
        stop=90,
        targets=[130, 140],
        time_stop_ms=entry_ms + 6 * 15 * 60_000,
        max_hold_ms=entry_ms + 4 * 3_600_000,
        flat_ms=flat_at,
    )
    assert flat["exit_reason"] == "flat"
    assert flat["exit_ms"] == flat_at


def test_half_then_trail_uses_the_5m_ema():
    start = _ms(12, 10)
    bars = []
    for index in range(20):
        open_time = start - (20 - index) * 300_000
        bars.append(
            {
                "open_time": open_time,
                "close_time": open_time + 300_000 - 1,
                "open": 110,
                "high": 110.2,
                "low": 109.8,
                "close": 110,
                "volume": 1,
                "taker_buy": 1,
            }
        )
    bars.append(
        {
            "open_time": start,
            "close_time": start + 300_000 - 1,
            "open": 100,
            "high": 112,
            "low": 100.5,
            "close": 111,
            "volume": 1,
            "taker_buy": 1,
        }
    )
    bars.append(
        {
            "open_time": start + 300_000,
            "close_time": start + 600_000 - 1,
            "open": 108,
            "high": 108,
            "low": 104,
            "close": 105,
            "volume": 1,
            "taker_buy": 1,
        }
    )
    trailed = simulate_v2(
        bars,
        entry_ms=start,
        side="long",
        entry=100,
        stop=90,
        targets=[111, 130],
        time_stop_ms=start + 86_400_000,
        max_hold_ms=start + 86_400_000,
        flat_ms=start + 86_400_000,
    )
    assert trailed["exit_reason"] == "trail"
    assert trailed["outcome"] == "hit"


def test_gate_lists_every_miss():
    empty = {
        "expectancy": 0.0,
        "counted": 0,
        "hit_rate": 0.0,
        "payoff": None,
        "max_loss_streak": 0,
        "max_drawdown_r": 0.0,
    }
    failed = judge_intraday(empty, empty, empty)
    assert failed["passed"] is False
    assert len(failed["failed"]) >= 5
    good = {
        "expectancy": 0.2,
        "counted": 120,
        "hit_rate": 0.5,
        "payoff": 1.4,
        "max_loss_streak": 4,
        "max_drawdown_r": 6,
    }
    assert judge_intraday(good, good, good)["passed"] is True
    soft = dict(good)
    soft["expectancy"] = 0.05
    missed = judge_intraday(good, good, soft)
    assert missed["passed"] is False
    assert any("0.1R" in item for item in missed["failed"])


def test_local_toml_overrides_the_new_thresholds(tmp_path: Path):
    base = tmp_path / "analysis.toml"
    base.write_text('[board]\nprofile = "intraday_v2"\n[intraday]\nt1_min_r = 1.2\n', encoding="utf-8")
    (tmp_path / "analysis.local.toml").write_text(
        "[intraday]\nt1_min_r = 2\ntime_stop_bars = 8\n[notify]\nsystem = false\n",
        encoding="utf-8",
    )
    cfg, _path = load_analysis(str(base))
    assert cfg.intraday.t1_min_r == 2
    assert cfg.intraday.time_stop_bars == 8
    assert cfg.notify.system is False
    assert cfg.board.profile == "intraday_v2"


def test_v1_profile_still_builds_the_original_plans():
    cfg = _cfg()
    cfg.board.profile = "v1"
    result = replay_at(FIXTURE, AT_1109, cfg)
    assert result["profile"] == "v1"
    assert any(plan["kind"] == "顺势" for plan in result["plans"])
    assert all(not plan.get("backtest_note") for plan in result["plans"])
    assert result["notify"]["system"] is False


def test_replay_20261114_under_v2():
    cfg = _cfg()
    cfg.intraday.session_enabled.asia = True
    cfg.intraday.session_enabled.europe = True
    cfg.intraday.session_enabled.us = True
    morning = replay_at(FIXTURE, AT_1109, cfg)
    assert morning["profile"] == "intraday_v2"
    assert morning["trend"]["frames"] == {"4h": "空", "1h": "空"}
    assert morning["trend"]["direction"] == "空头"
    assert "方向打架" not in morning["conclusion"]["text"]
    assert "不在插针里追空" in morning["conclusion"]["text"]
    assert morning["sentiment"]["label"] == "多单在爆仓/平仓"
    levels = morning["levels"]["support"] + morning["levels"]["resistance"]
    assert any(abs(level["price"] - 1893.39) < 1 and "插针扫过" in level["sources"] for level in levels)
    assert morning["notify"]["system"] is False
    assert morning["notify"]["eligible"] is morning["backtest_gate"]["passed"]
    if not morning["backtest_gate"]["passed"] and morning["plans"]:
        assert any(plan.get("backtest_note") == "未通过回测，仅供参考" for plan in morning["plans"])
    assert morning["backtest_gate"]["passed"] or morning["backtest_gate"]["label"]

    midday = replay_at(FIXTURE, AT_1351, cfg)
    assert midday["range"]["shrink"] is True
    assert "横盘堆仓" in midday["sentiment"]["label"]
    assert midday["position"]["zone"] == "middle"
    assert "位置不好，不开单" in midday["conclusion"]["text"]
    assert all(plan["status"] != "可进场" for plan in midday["plans"])
    # Keep the sentences used in the report next to the assertions above.
    assert morning["conclusion"]["text"]
    assert midday["conclusion"]["text"]


def test_passing_gate_does_not_turn_system_notices_on():
    from supere.board import gate_v2

    cfg = _cfg()
    cfg.intraday.session_enabled.asia = True
    original = dict(gate_v2.GATE)
    gate_v2.GATE.clear()
    gate_v2.GATE.update({**original, "passed": True, "eligible": True, "failed": [], "by_kind": {"顺势": {"expectancy": 0.2}}})
    try:
        result = replay_at(FIXTURE, AT_1109, cfg)
        assert result["notify"]["eligible"] is True
        assert result["notify"]["system"] is False
        assert all(not plan.get("backtest_note") for plan in result["plans"])
        trend = next(plan for plan in result["plans"] if plan["kind"] == "顺势")
        assert trend["expectancy_text"] == "按回测这类单的期望 +0.20R"
    finally:
        gate_v2.GATE.clear()
        gate_v2.GATE.update(original)
