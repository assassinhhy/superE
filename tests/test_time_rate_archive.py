from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from supere.binance.archive import parse_checksum, verify_bytes, zip_text
from supere.binance.parse import (
    parse_funding_csv,
    parse_kline_csv,
    parse_metrics_csv,
    parse_ws_message,
)
from supere.clock import TimeSample, choose_offset
from supere.errors import ChecksumError
from supere.intervals import align_open, close_time_of, next_open
from supere.ingest.gaps import find_missing_ranges
from supere.ratelimit import WeightLimiter, kline_request_weight
from supere.timeutil import format_utc8, to_millis


def test_utc8_display_keeps_binance_utc_buckets():
    # 2020-01-01 00:00 UTC is 08:00 in UTC+8. The daily candle still opens there.
    open_ms = 1_577_836_800_000
    assert format_utc8(open_ms) == "2020-01-01 08:00:00"
    assert format_utc8(next_open("1d", open_ms)) == "2020-01-02 08:00:00"
    assert align_open("1d", open_ms + 3_600_000) == open_ms
    assert close_time_of("1d", open_ms) == next_open("1d", open_ms) - 1


def test_week_month_and_3d_alignment_matches_archive():
    monday = 1_578_268_800_000  # 2020-01-06, from the 1w archive
    assert align_open("1w", monday) == monday
    assert align_open("1w", monday - 1) == monday - 7 * 86_400_000
    # Current 3d phase (day index % 3 == 1), checked against 2024-01 archives.
    jan1 = 1_704_067_200_000  # 2024-01-01
    jan4 = jan1 + 3 * 86_400_000  # 2024-01-04
    assert align_open("3d", jan1) == jan1
    assert align_open("3d", jan1 + 86_400_000) == jan1
    assert align_open("3d", jan4 - 1) == jan1
    assert align_open("3d", jan4) == jan4
    assert align_open("1M", 1_577_836_800_000) == 1_577_836_800_000


def test_timestamp_units():
    assert to_millis(1_748_736_000_000_000) == 1_748_736_000_000
    assert to_millis(1_748_736_000_000) == 1_748_736_000_000
    assert to_millis(1_700_000_000) == 1_700_000_000_000


def _utc_ms(year: int, month: int, day: int) -> int:
    from datetime import UTC, datetime

    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)


def test_3d_phase_change_is_not_a_multi_year_gap():
    # Old phase through 2023-06-27, then the post-2023-08 phase through the
    # latest bar. Coverage must not paint 2023-08-17..2026-10-30 as one hole
    # just because those dates sit on the retired grid.
    step = 3 * 86_400_000
    old_start = _utc_ms(2022, 9, 27)
    old_last = _utc_ms(2023, 6, 27)
    new_start = _utc_ms(2023, 8, 16)
    new_last = _utc_ms(2026, 10, 29)
    opens = list(range(old_start, old_last + 1, step))
    opens.extend(range(new_start, new_last + 1, step))
    false_start = _utc_ms(2023, 8, 17)
    false_end = _utc_ms(2026, 10, 30)
    gaps = find_missing_ranges("3d", opens, opens[0], new_last)
    assert not any(gap[0] == false_start and gap[1] >= false_end for gap in gaps)
    # A real hole on one phase is still reported. 1w stays on Mondays.
    week = _utc_ms(2024, 1, 1)
    assert align_open("1w", week) == week
    missing_week = find_missing_ranges("1w", [week, week + 14 * 86_400_000], week, week + 14 * 86_400_000)
    assert missing_week == [(week + 7 * 86_400_000, week + 7 * 86_400_000)]
    jan = _utc_ms(2024, 1, 1)
    mar = _utc_ms(2024, 3, 1)
    missing_month = find_missing_ranges("1M", [jan, mar], jan, mar)
    assert missing_month == [(_utc_ms(2024, 2, 1), _utc_ms(2024, 2, 1))]


def test_kline_weight_tiers():
    assert kline_request_weight(50) == 1
    assert kline_request_weight(100) == 2
    assert kline_request_weight(499) == 2
    assert kline_request_weight(500) == 5
    assert kline_request_weight(1000) == 5
    assert kline_request_weight(1500) == 10


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, delay: float) -> None:
        self.t += delay


async def test_limiter_cap_header_and_halt():
    clock = _Clock()
    limiter = WeightLimiter(
        "futures",
        2400,
        budget_ratio=0.5,
        header_slowdown_ratio=0.6,
        clock=clock,
        sleeper=clock.sleep,
    )
    await limiter.acquire(1176)
    assert limiter.peak == 1176
    assert limiter.peak < 2400 * 0.5
    waiter = limiter.acquire(1)
    # The window is full, so this must move the clock forward.
    await waiter
    assert clock.t >= 60
    limiter.observe_header(1441)
    assert limiter.slowdown is True
    limiter.observe_header(1440)
    assert limiter.slowdown is False
    assert limiter.note_429() == 1
    assert limiter.note_429() == 2
    limiter.note_success()
    assert limiter.note_429() == 1
    limiter.halt()
    with pytest.raises(Exception):
        await limiter.acquire(1)


def test_clock_picks_shortest_rtt():
    samples = [
        TimeSample(1_000, 1_500, 10_000),
        TimeSample(2_000, 2_200, 8_000),
    ]
    chosen = choose_offset(samples)
    assert chosen is samples[1]
    assert chosen.offset_ms == 8_000 - (2_000 + 2_200) // 2


def test_archive_csv_header_microseconds_and_checksum():
    plain = "1577836800000,1,2,0.5,1.5,10,1577836859999,15,4,3,4,0\n"
    headed = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n" + plain
    rows = parse_kline_csv(headed, "um", "ethusdt", "1m", "archive", 1, 1)
    assert rows[0].symbol == "ETHUSDT"
    assert rows[0].o == "1"
    micro = "1748736000000000,1,2,0.5,1.5,10,1748736059999999,15,4,3,4,0\n"
    micro_rows = parse_kline_csv(micro, "spot", "ETHUSDT", "1m", "archive", 1, 1)
    assert micro_rows[0].open_time == 1_748_736_000_000
    funding = parse_funding_csv(
        "calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.00027858\n",
        "ETHUSDT",
    )
    assert funding[0].rate.startswith("0.00027858")
    metrics = parse_metrics_csv(
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2024-01-02 00:05:00,ETHUSDT,2,3,1.1,1.2,1.3,1.4\n"
        "2024-01-02 00:00:00,ETHUSDT,1,2,1.1,1.2,1.3,1.4\n"
    )
    assert [row.ts for row in metrics] == sorted(row.ts for row in metrics)
    assert metrics[0].oi == "1"
    blob = b"abc"
    digest = hashlib.sha256(blob).hexdigest()
    verify_bytes(blob, f"{digest}  file.zip\n")
    with pytest.raises(ChecksumError):
        parse_checksum("nope")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ETHUSDT-1m.csv", plain)
    assert "1577836800000" in zip_text(buffer.getvalue())


def test_metrics_quoted_empty_ratios_are_null(caplog):
    import logging
    from pathlib import Path

    text = (Path(__file__).resolve().parent / "fixtures" / "ethusdt_metrics_empty_quotes.csv").read_text(
        encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="supere.parse"):
        rows = parse_metrics_csv(text, "archive")
    assert len(rows) == 4
    good = rows[0]
    assert good.oi == "472000.00000000"
    assert good.oi_value == "1750000000.00000000"
    assert good.top_acc_lsr == "1.2345"
    assert good.top_pos_lsr == "1.1111"
    assert good.global_lsr == "0.9"
    assert good.taker_lsr == "1.0"
    empty = rows[1]
    assert empty.oi == "472186.06100000"
    assert empty.oi_value == "1759026965.49760320"
    assert empty.top_acc_lsr is None
    assert empty.top_pos_lsr is None
    assert empty.global_lsr is None
    assert empty.taker_lsr is None
    blank = rows[3]
    assert blank.top_acc_lsr is None
    assert blank.top_pos_lsr is None
    assert blank.global_lsr is None
    assert blank.taker_lsr is None
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1


def test_ws_messages_parse():
    kline = parse_ws_message(
        '{"e":"kline","E":10,"s":"ETHUSDT","k":{"t":0,"T":59999,"s":"ETHUSDT","i":"1m","o":"1","c":"2","h":"3","l":"0.5","v":"4","n":5,"x":false,"q":"6","V":"7","Q":"8","B":"0"}}'
    )
    assert kline["kind"] == "kline" and kline["closed"] is False
    assert parse_ws_message('{"result":null,"id":1}') is None
    depth = parse_ws_message(
        '{"e":"depthUpdate","E":11,"T":10,"s":"ETHUSDT","u":99,"b":[["1","2"]],"a":[["3","4"]]}'
    )
    assert depth["update_id"] == 99 and depth["bids_px"] == ["1"]
    trade = parse_ws_message(
        '{"e":"aggTrade","E":12,"s":"ETHUSDT","a":7,"p":"10","q":"0.2","f":1,"l":2,"T":12,"m":false}'
    )
    assert trade["dedupe_key"] == ("aggTrade", 7)
    mark = parse_ws_message(
        '{"e":"markPriceUpdate","E":13,"s":"ETHUSDT","p":"100","i":"99","r":"0.0001","T":20}'
    )
    assert mark["rate"] == "0.0001"
    force = parse_ws_message(
        '{"e":"forceOrder","E":14,"o":{"s":"ETHUSDT","S":"SELL","o":"LIMIT","f":"IOC","q":"1","p":"2000","ap":"2000","X":"FILLED","l":"1","z":"1","T":14}}'
    )
    assert force["kind"] == "forceOrder"
    assert force["side"] == "SELL"
