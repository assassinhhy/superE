"""Offline tests for the Deribit probe helpers (no network is used)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from supere.train import deribit_probe as p


def test_parse_instrument_name():
    out = p.parse_instrument_name("ETH-27DEC24-3000-C")
    assert out["currency"] == "ETH"
    assert out["strike"] == 3000.0
    assert out["option_type"] == "C"
    assert out["expiry_utc"].startswith("2024-12-27")
    assert p.parse_instrument_name("bad-name") == {}


def test_iv_null_rate():
    rows = [{"iv": 55.0}, {"iv": None}, {"iv": 0}, {"iv": 60.0}, {"other": 1}]
    out = p.iv_null_rate(rows)
    assert out["n"] == 4
    assert out["non_null"] == 2
    assert out["null_rate"] == 0.5
    assert p.iv_null_rate([])["null_rate"] is None


def test_row_count_and_has_more():
    assert p._row_count([1, 2, 3]) == 3
    assert p._row_count({"data": [[1], [2]]}) == 2
    assert p._row_count({"price": [1, 2, 3]}) == 3
    assert p._row_count(123) == 0
    assert p._has_more({"has_more": True}) is True
    assert p._has_more({"continuation": "abc"}) is True
    assert p._has_more({"continuation": None}) is False
    assert p._has_more([]) is None


def test_budget_stops_at_limit():
    b = p.Budget(max_requests=2)
    b.take()
    b.take()
    with pytest.raises(RuntimeError):
        b.take()


def test_select_current_atm_and_tenors():
    now = int(datetime(2026, 11, 11, tzinfo=timezone.utc).timestamp() * 1000)
    chain = []
    for days in (1, 30, 180):
        exp = now + days * 86_400_000
        for strike in (2800.0, 3000.0, 3200.0):
            for typ in ("call", "put"):
                chain.append({"instrument_name": f"ETH-X-{int(strike)}-{typ[0].upper()}",
                              "expiration_timestamp": exp, "strike": strike,
                              "option_type": typ})
    out = p.select_current(chain, underlying=3010.0, now_ms=now)
    assert set(out) == {"1d", "30d", "180d"}
    assert out["1d"]["strike"] == 3000.0
    assert out["1d"]["call"].endswith("-C")
    assert out["30d"]["call"].startswith("ETH-X-3000")
    assert out["180d"]["put"].endswith("-P")


def test_series_summary_handles_list_and_empty_data():
    out = p._series_summary([[1, 2.0], [3, 4.0]])
    assert out["rows"] == 2 and out["first_ts"] == 1 and out["last_ts"] == 3
    out2 = p._series_summary({"data": [], "timestamp": [10, 20], "price": [1.0, 2.0]})
    assert out2["rows"] == 2 and out2["first_ts"] == 10


def test_availability_classification():
    base = {
        "instruments": {"current": {"n": 10}, "expired": {"n": 2}},
        "book_summary": {"n": 10},
        "dvol": {"rows": 100},
        "mark_price_history": {"rows": 10},
        "old_metadata": {"n": 1},
        "historical_trades": {"n": 50, "rows": 50},
    }
    av = p.infer_availability(dict(base))
    assert av["overall"] == "B"
    empty = dict(base, historical_trades={"n": 0, "rows": 0}, old_metadata={"n": 0})
    assert p.infer_availability(empty)["overall"] == "C"
    # No endpoint can be class A for historical quoted IV in this probe.
    items = {i["item"]: i for i in av["items"]}
    assert items["historical quoted IV snapshot"]["class"] == "C"
