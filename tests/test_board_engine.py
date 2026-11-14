"""Rule rows, indicator parity, and the two 2026-11-14 replays."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from supere.board.confirm import engulf_confirms, rsi_turn, wick_confirms
from supere.board.engine import DISCLAIMER, analyze
from supere.board.indicators import atr_series, ema_series, macd_series, rsi_series
from supere.board.levels import _merge, _touches
from supere.board.plans import _closes_beyond, protective_stop
from supere.board.range_ import build_range
from supere.board.replay import load_fixture, replay_at, snapshot_at
from supere.board.risk import assess, load_calendar
from supere.board.sentiment import crowding, funding_heat, quadrant, volume_spike
from supere.board.settings import AnalysisConfig, load_analysis
from supere.board.trend import classify_frame, daily_rsi_tag, macd_fading, synthesize

UTC8 = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "board"
AT_1109 = int(datetime(2026, 11, 14, 11, 9, tzinfo=UTC8).timestamp() * 1000)
AT_1351 = int(datetime(2026, 11, 14, 13, 51, tzinfo=UTC8).timestamp() * 1000)


def _cfg() -> AnalysisConfig:
    cfg, _path = load_analysis(str(ROOT / "analysis.toml"))
    return cfg


def _bar(index, price, *, step=900_000, volume=10.0, high=None, low=None, open_=None, taker=None):
    open_time = index * step
    return {
        "open_time": open_time,
        "close_time": open_time + step - 1,
        "open": price if open_ is None else open_,
        "high": price + 0.4 if high is None else high,
        "low": price - 0.4 if low is None else low,
        "close": price,
        "volume": volume,
        "taker_buy": volume * 0.5 if taker is None else taker,
    }


def test_ema_hand_values_and_frontend_parity():
    assert ema_series([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]
    closes = [100 + (i % 7) - 3 + i * 0.15 for i in range(80)]
    bars = [{"high": close + 1.2, "low": close - 0.8, "close": close, "open": close - 0.2} for close in closes]
    script = """
import { rsiSeries, macdSeries } from './src/signals.js';
import { atr } from './src/indicators.js';
const payload = JSON.parse(process.argv[1]);
const macd = macdSeries(payload.closes);
const hist = macd.dif.map((value, index) => (
  value == null || macd.dea[index] == null ? null : value - macd.dea[index]
));
console.log(JSON.stringify({
  rsi: rsiSeries(payload.closes, 14),
  dif: macd.dif,
  dea: macd.dea,
  hist,
  atr: atr(payload.bars, 14),
}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, json.dumps({"closes": closes, "bars": bars})],
        cwd=ROOT / "frontend",
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads(completed.stdout)
    dif, dea, hist = macd_series(closes)
    for name, got in (
        ("rsi", rsi_series(closes, 14)),
        ("dif", dif),
        ("dea", dea),
        ("hist", hist),
        ("atr", atr_series(bars, 14)),
    ):
        for index, (left, right) in enumerate(zip(got, expected[name])):
            if left is None or right is None:
                assert left is None and right is None, (name, index, left, right)
            else:
                assert abs(left - right) < 1e-6, (name, index, left, right)


def test_trend_table_and_boundaries():
    assert classify_frame(98, 99, 99.1, 0.15) == "震荡"
    assert classify_frame(98, 99, 99.2, 0.15) == "空"
    assert classify_frame(102, 101, 100.5, 0.15) == "多"
    assert classify_frame(100, 100, 100, 0.15) == "震荡"
    cases = {
        ("空", "空"): ("只做空", "空头"),
        ("多", "多"): ("只做多", "多头"),
        ("空", "震荡"): ("偏空", "空头"),
        ("空", "多"): ("偏空", "空头"),
        ("多", "震荡"): ("偏多", "多头"),
        ("多", "空"): ("偏多", "多头"),
        ("震荡", "空"): ("震荡，只做区间两头或不做", "震荡"),
        ("震荡", "多"): ("震荡，只做区间两头或不做", "震荡"),
        ("震荡", "震荡"): ("震荡，只做区间两头或不做", "震荡"),
    }
    for frames, expected in cases.items():
        assert synthesize(*frames) == expected
    assert daily_rsi_tag(30, overbought=70, oversold=30) is None
    assert daily_rsi_tag(29.9, overbought=70, oversold=30) == "大级别超卖，注意反抽"
    assert daily_rsi_tag(70, overbought=70, oversold=30) is None
    assert daily_rsi_tag(70.1, overbought=70, oversold=30) == "大级别超买，注意回调"
    assert macd_fading([None, -5, -4, -3], bullish=False, bars=2)
    assert not macd_fading([None, -5, -4, -4], bullish=False, bars=2)
    assert not macd_fading([None, 5, 4, 3], bullish=False, bars=2)
    assert macd_fading([5, 4, 3], bullish=True, bars=2)


def test_sentiment_table_boundaries_and_side_notes():
    assert quadrant(-0.3, -0.3, 0.3, 0.3) == "多单在爆仓/平仓"
    assert quadrant(-0.3, 0.3, 0.3, 0.3) == "新空单进场，空头主动"
    assert quadrant(0.3, -0.3, 0.3, 0.3) == "空头回补"
    assert quadrant(0.3, 0.3, 0.3, 0.3) == "新多进场，上涨较健康"
    assert quadrant(0.0, 0.3, 0.3, 0.3) == "横盘堆仓，后续突破波动会放大"
    assert quadrant(0.0, -0.3, 0.3, 0.3) == "横盘去杠杆，中性"
    assert quadrant(-0.299, -0.5, 0.3, 0.3) == "横盘去杠杆，中性"
    assert quadrant(-0.3, -0.299, 0.3, 0.3) == "仓位无明显变化"
    assert quadrant(0.5, 0.0, 0.3, 0.3) == "仓位无明显变化"
    assert crowding(3.0, 3.0, 0.8) == "散户多头拥挤，向下破位易连环爆多"
    assert crowding(0.8, 3.0, 0.8) == "散户空头拥挤，向上挤压易连环爆空"
    assert crowding(2.99, 3.0, 0.8) is None
    assert funding_heat(0.0003, 0.0003, 0.00005) == "费率偏热"
    assert funding_heat(0.00005, 0.0003, 0.00005) == "中性"
    assert funding_heat(0.000051, 0.0003, 0.00005) is None
    assert volume_spike([1] * 20 + [3], 3, 20)
    assert not volume_spike([1] * 20 + [2.99], 3, 20)


def test_levels_merge_priority_and_touch_runs():
    bars = []
    for index in range(6):
        bars.append(_bar(index, 1900, high=1901, low=1899))
    bars.append(_bar(6, 1950, high=1951, low=1949))
    bars.append(_bar(7, 1900, high=1901, low=1899))
    assert _touches(bars, 1900, 2) == 2
    merged = _merge(
        [
            {"price": 1900, "kind": "插针扫过", "dynamic": False},
            {"price": 1901, "kind": "整数", "dynamic": False},
            {"price": 1899.5, "kind": "前日低", "dynamic": False},
        ],
        atr=10,
        merge_atr=0.5,
        bars=bars,
        touch_atr=0.2,
    )
    assert len(merged) == 1
    assert merged[0]["price"] == 1900
    assert merged[0]["sources"][0] == "插针扫过"
    assert "前日低" in merged[0]["sources"]
    assert "整数" in merged[0]["sources"]


def test_range_position_and_shrink_boundaries():
    params = _cfg().range
    bars = [_bar(index, 100, high=110, low=100, volume=10) for index in range(20)]
    for index in range(16, 20):
        bars[index] = _bar(index, 100, high=110, low=100, volume=6)
    # recent/base = 6/10 = 0.6, which is not below the ratio.
    same = build_range(bars, 102.5, params)
    assert same["shrink"] is False
    assert same["position"] == "lower"
    for index in range(16, 20):
        bars[index] = _bar(index, 107.5, high=110, low=100, volume=5.99)
    shrunk = build_range(bars, 107.5, params)
    assert shrunk["shrink"] is True
    assert "缩量横盘" in shrunk["shrink_text"]
    assert shrunk["position"] == "upper"
    middle = build_range(bars, 105, params)
    assert middle["position"] == "middle"
    assert middle["blocked"] is True
    edge = build_range(bars, 102.5, params)
    assert edge["position"] == "lower"
    upper_edge = build_range(bars, 107.5, params)
    assert upper_edge["position"] == "upper"


def test_stop_matches_the_confirmed_example():
    # 进场区上沿 1912，ATR15m 10.5，ATR1h 21.7，a=0.5，b=0.35 → 1919.595
    assert protective_stop(1912, 10.5, 21.7, a=0.5, b=0.35, side="short") == 1912 + max(5.25, 0.35 * 21.7)
    assert protective_stop(1912, 10.5, 21.7, a=0.5, b=0.35, side="long") == 1912 - max(5.25, 0.35 * 21.7)


def test_confirm_and_flip_boundaries():
    zone = (100, 102)
    pin = {"open": 100, "close": 101, "high": 104, "low": 100}
    # upper wick 3, body 1, so 3 >= 1.5×body and 3 >= 0.5×ATR when ATR is 6.
    assert wick_confirms(pin, side="short", atr=6, body_mult=1.5, atr_mult=0.5, zone=zone)
    assert not wick_confirms(pin, side="short", atr=7, body_mult=1.5, atr_mult=0.5, zone=zone)
    prev = {"open": 100.6, "close": 100.3, "high": 100.7, "low": 100.2, "volume": 1}
    bull = {"open": 100.2, "close": 100.7, "high": 100.8, "low": 100.1, "volume": 1}
    assert engulf_confirms(prev, bull, side="long", zone=(100, 101))
    assert not engulf_confirms(prev, bull, side="short", zone=(100, 101))
    assert rsi_turn([70, 68, 66, 64, 62, 61, 55], side="short", lookback=6, short_from=60, long_from=40)
    assert not rsi_turn([70, 50, 51, 52, 53, 54, 55], side="short", lookback=6, short_from=60, long_from=40)
    above = [_bar(index, 101) for index in range(2)]
    assert _closes_beyond(above, 100, "short", 2)
    assert not _closes_beyond(above[:1], 100, "short", 2)
    assert _closes_beyond(above[:1], 100, "short", 1)


def test_weekend_and_calendar_window(tmp_path: Path):
    params = _cfg().risk
    saturday = int(datetime(2026, 11, 14, 11, 9, tzinfo=UTC8).timestamp() * 1000)
    friday = int(datetime(2026, 11, 13, 11, 9, tzinfo=UTC8).timestamp() * 1000)
    assert assess(saturday, [], [], params)["weekend"] is True
    assert assess(friday, [], [], params)["weekend"] is False
    event_at = saturday + 10 * 60_000
    inside = assess(saturday, [{"ts_event": event_at, "kind": "PPI", "title": "美国 PPI"}], [], params)
    assert inside["events"][0]["blocking"] is True
    assert "不开新单" in inside["notes"]
    far = assess(saturday, [{"ts_event": saturday + 3 * 3600_000, "kind": "CPI", "title": "CPI"}], [], params)
    assert far["events"][0]["blocking"] is False
    path = tmp_path / "calendar.csv"
    path.write_text("time_utc8,kind,title\n2026-11-14 20:30,PPI,美国 PPI\n", encoding="utf-8")
    loaded = load_calendar(path)
    assert loaded[0]["kind"] == "PPI"
    assert loaded[0]["title"] == "美国 PPI"


def test_stale_metrics_are_not_used_in_the_conclusion():
    cfg = _cfg()
    bars = [_bar(index, 100 - index * 0.2, step=900_000) for index in range(80)]
    snapshot = {
        "symbol": "ETHUSDT",
        "now_ms": bars[-1]["close_time"],
        "now_utc8": "2026-11-14 11:09:00",
        "price": {"value": bars[-1]["close"], "status": "ok", "source": "K 线", "utc8": "", "label": ""},
        "ticker": {"status": "missing"},
        "klines": {
            frame: {"status": "ok", "bars": bars, "source": "fixture"}
            for frame in ("5m", "15m", "1h", "4h", "1d")
        },
        "metrics": {
            "status": "stale",
            "data_ts": bars[0]["open_time"],
            "rows": [{"ts": 1, "known_at": 2, "oi": 1, "global_lsr": 9, "top_pos_lsr": 9, "taker_lsr": 9}],
            "label": "上次数据 10:00:00（已过期）",
        },
        "funding": {"status": "missing", "value": None},
        "crosscheck": {"pause_plans": False, "message": "交叉核对未获取"},
        "limit": {"active": False, "message": ""},
        "events": [],
        "calendar": [],
    }
    result = analyze(snapshot, cfg)
    assert result["sentiment"]["status"] == "missing"
    assert result["sentiment"]["text"] == "未获取"
    assert "爆仓" not in result["conclusion"]["text"]
    assert "堆仓" not in result["conclusion"]["text"]
    assert result["disclaimer"] == DISCLAIMER


def test_replay_20261114_1109_matches_the_call():
    result = replay_at(FIXTURE, AT_1109, _cfg())
    assert result["trend"]["direction"] == "空头"
    assert result["trend"]["frames"]["4h"] == "空"
    assert result["trend"]["frames"]["1h"] == "空"
    levels = result["levels"]["support"] + result["levels"]["resistance"]
    pin = [level for level in levels if abs(level["price"] - 1893.39) < 1]
    assert pin, levels
    assert "插针扫过" in pin[0]["sources"]
    assert result["sentiment"]["label"] == "多单在爆仓/平仓"
    assert "不在插针里追空" in result["conclusion"]["text"]
    assert result["disclaimer"] == DISCLAIMER


def test_replay_20261114_1351_is_a_middle_range():
    result = replay_at(FIXTURE, AT_1351, _cfg())
    assert result["range"]["shrink"] is True
    assert "缩量横盘" in result["range"]["shrink_text"]
    assert "横盘堆仓" in result["sentiment"]["label"]
    assert result["position"]["zone"] == "middle"
    assert "位置不好，不开单" in result["conclusion"]["text"]


def test_replay_does_not_read_the_future():
    cfg = _cfg()
    fixture = load_fixture(FIXTURE)
    before = analyze(snapshot_at(fixture, AT_1109, cfg), cfg)
    fixture["frames"]["15m"].append(
        {
            "open_time": AT_1109 + 60_000,
            "close_time": AT_1109 + 900_000,
            "open": 1,
            "high": 5000,
            "low": 1,
            "close": 1,
            "volume": 1_000_000,
            "taker_buy": 1,
        }
    )
    fixture["metrics"].append(
        {
            "ts": AT_1109,
            "known_at": AT_1109 + 1,
            "oi": 1e12,
            "oi_value": 1e12,
            "global_lsr": 9,
            "top_pos_lsr": 9,
            "taker_lsr": 9,
        }
    )
    after = analyze(snapshot_at(fixture, AT_1109, cfg), cfg)
    assert after["conclusion"] == before["conclusion"]
    assert after["sentiment"]["label"] == before["sentiment"]["label"]
    assert [level["price"] for level in after["levels"]["support"]] == [
        level["price"] for level in before["levels"]["support"]
    ]
