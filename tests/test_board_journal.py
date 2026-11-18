"""Journal dedupe, schema migration, and the after-the-fact resolver."""

import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from supere.board.journal import (
    PlanJournal,
    assess_plan,
    journal_path,
    parse_bound,
    rules_fingerprint,
)
from supere.board.serve import create_app
from supere.board.settings import load_analysis
from supere.timeutil import UTC8

ROOT = Path(__file__).resolve().parents[1]
GENERATED = int(datetime(2026, 11, 15, 21, 45, tzinfo=UTC8).timestamp() * 1000)
EXPIRES = GENERATED + 60 * 60 * 1000


def _cfg():
    cfg, _path = load_analysis(str(ROOT / "analysis.toml"))
    return cfg


def _plan(**overrides):
    plan = {
        "id": "trend",
        "kind": "顺势",
        "title": "顺势反抽空（首选）",
        "side": "short",
        "status": "等待信号",
        "entry": 113.0,
        "entry_rule": "trend_near_key",
        "key_level": 110.0,
        "entry_low": 107.0,
        "entry_high": 113.0,
        "stop": 120.0,
        "targets": [90.0, 80.0],
        "r_distance": 7.0,
        "t1_r": 23 / 7,
        "t2_r": 33 / 7,
        "size_ratio": 1.0,
        "confirm": {"ok": False, "names": []},
        "generated_ms": GENERATED,
        "generated_utc8": "21:45",
        "basis_close_ms": GENERATED,
        "expires_ms": EXPIRES,
        "expires_utc8": "22:45",
        "validity": "21:45 生成 · 有效至 22:45",
    }
    plan.update(overrides)
    return plan


def _board(*plans, price=105.0):
    return {
        "symbol": "ETHUSDT",
        "as_of": GENERATED + 30_000,
        "price": {"value": price, "status": "ok"},
        "funding": {"rate": 0.0001, "status": "ok"},
        "oi": {"value": 12345.0, "status": "ok"},
        "atr": {"15m": 10.0},
        "plans": list(plans),
    }


def _minute(offset_min, high, low, close=None):
    open_time = GENERATED + offset_min * 60_000
    return {
        "open_time": open_time,
        "close_time": open_time + 60_000 - 1,
        "open": close if close is not None else (high + low) / 2,
        "high": high,
        "low": low,
        "close": close if close is not None else (high + low) / 2,
    }


def test_polls_inside_one_bar_share_a_row_and_a_resolved_row_stays_put(tmp_path: Path):
    journal = PlanJournal(journal_path(tmp_path))
    cfg = _cfg()
    first, second = rules_fingerprint(cfg)
    assert first == "fixed-entry-1"
    assert rules_fingerprint(cfg) == (first, second)
    journal.record_board(_board(_plan()), cfg)
    journal.record_board(_board(_plan(status="可进场", confirm={"ok": True, "names": ["长影线"]})), cfg)
    rows = journal.list_plans()
    assert len(rows) == 1
    assert rows[0]["plan_id"] == f"ETHUSDT:{GENERATED}:trend"
    assert rows[0]["status"] == "可进场"
    assert rows[0]["entry"] == 113
    assert rows[0]["funding_rate"] == 0.0001
    assert rows[0]["oi"] == 12345
    assert rows[0]["atr"] == 10
    assert rows[0]["rules_version"] == "fixed-entry-1"
    assert rows[0]["config_hash"] == second
    assert rows[0]["triggers"]["entry_rule"] == "trend_near_key"
    assert rows[0]["triggers"]["confirm"] == ["长影线"]
    assert rows[0]["validity"] == "21:45 生成 · 有效至 22:45"
    mode = journal.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"

    later = GENERATED + 15 * 60 * 1000
    journal.record_board(_board(_plan(generated_ms=later, basis_close_ms=later, expires_ms=later + 3_600_000)), cfg)
    assert len(journal.list_plans()) == 2

    def bars(_interval, _start, _end):
        return [_minute(10, 121, 112, 118)]

    changed = journal.resolve_pending(bars, GENERATED + 20 * 60_000)
    assert changed == 1
    resolved = journal.list_plans(start_ms=GENERATED, end_ms=GENERATED)[0]
    assert resolved["outcome"] == "resolved"
    assert resolved["filled"] == 1
    assert resolved["first_hit"] == "stop"
    journal.record_board(_board(_plan(status="已作废")), cfg)
    again = journal.list_plans(start_ms=GENERATED, end_ms=GENERATED)[0]
    assert again["status"] == "可进场"
    assert again["r_taker"] == resolved["r_taker"]
    journal.close()


def test_resolver_expired_t1_t2_and_restart(tmp_path: Path):
    quiet = [_minute(index, 112, 110, 111) for index in range(70)]
    expired = assess_plan(_plan(), quiet, EXPIRES + 60_000, interval_ms=60_000)
    assert expired["outcome"] == "expired"
    assert expired["expired_unfilled"] == 1
    assert expired["filled"] == 0

    # Touch 113, then trade down through 90 and 80 without the stop at 120.
    path = [_minute(5, 114, 112, 113), _minute(6, 113, 79, 80)]
    hit = assess_plan(_plan(), path, GENERATED + 10 * 60_000, interval_ms=60_000)
    assert hit["filled"] == 1
    assert hit["first_hit"] == "t1"
    assert hit["exit_reason"] == "t2"
    assert hit["t1_ms"] is not None and hit["t2_ms"] == hit["t1_ms"]
    assert hit["mfe"] >= 113 - 79
    assert hit["mae"] >= 114 - 113
    assert hit["r_taker"] < hit["r_maker"]
    assert hit["r_taker"] > 0

    journal = PlanJournal(journal_path(tmp_path))
    cfg = _cfg()
    journal.record_board(_board(_plan()), cfg)
    journal.close()
    restored = PlanJournal(journal_path(tmp_path))
    assert restored.list_plans()[0]["outcome"] == "pending"

    def bars(interval, _start, _end):
        if interval == "1m":
            return [_minute(3, 121, 100, 110)]
        raise AssertionError("1m was available")

    restored.resolve_pending(bars, GENERATED + 10 * 60_000)
    row = restored.list_plans()[0]
    assert row["filled"] == 1
    assert row["first_hit"] == "stop"
    assert row["bar_interval"] == "1m"
    assert row["fill_utc8"].endswith("21:48")
    assert row["r_maker"] > row["r_taker"]
    restored.close()


def test_fill_bar_cannot_also_hit_a_target():
    # Entry 113, T1 90 and T2 80 all print inside the fill bar. Only the stop
    # may resolve on that bar, so no win is booked from the same bar.
    same_bar = [_minute(5, 114, 79, 80)]
    out = assess_plan(_plan(), same_bar, GENERATED + 10 * 60_000, interval_ms=60_000)
    assert out["filled"] == 1
    assert out["t1_ms"] is None
    assert out["t2_ms"] is None
    assert out["first_hit"] is None
    assert out["exit_reason"] is None
    assert out["outcome"] == "open"

    # The same bar touching the stop is still the conservative outcome.
    stopped = [_minute(5, 121, 79, 80)]
    out = assess_plan(_plan(), stopped, GENERATED + 10 * 60_000, interval_ms=60_000)
    assert out["exit_reason"] == "stop"
    assert out["first_hit"] == "stop"

    # Targets are checked from the bar after the fill.
    later = [_minute(5, 114, 79, 80), _minute(6, 112, 89, 89)]
    out = assess_plan(_plan(), later, GENERATED + 10 * 60_000, interval_ms=60_000)
    assert out["first_hit"] == "t1"
    assert out["exit_reason"] == "t1"
    assert out["t1_ms"] == later[1]["close_time"]
    assert out["t2_ms"] is None


def test_migration_keeps_rows_already_in_the_data_dir(tmp_path: Path):
    path = journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    main = tmp_path / "superE.sqlite"
    main.write_bytes(b"main-db-not-touched")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE plan_journal (plan_id TEXT PRIMARY KEY, symbol TEXT, generated_ms INTEGER)"
    )
    conn.execute("INSERT INTO plan_journal (plan_id, symbol, generated_ms) VALUES ('old', 'ETHUSDT', 1)")
    conn.commit()
    conn.close()
    journal = PlanJournal(path)
    rows = journal.list_plans()
    assert rows[0]["plan_id"] == "old"
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["entry"] is None
    assert "r_taker" in rows[0]
    assert journal.conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert journal.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    journal.close()
    assert main.read_bytes() == b"main-db-not-touched"


def test_plans_api_filters_by_utc8_and_exports_csv(tmp_path: Path):
    cfg = _cfg()
    app = create_app(tmp_path, cfg, schedule=False, frozen={"ok": True})
    with TestClient(app) as client:
        journal = app.state.journal
        journal.record_board(_board(_plan()), cfg)
        later = GENERATED + 2 * 60 * 60 * 1000
        journal.record_board(
            _board(_plan(generated_ms=later, basis_close_ms=later, expires_ms=later + 3_600_000)),
            cfg,
        )
        listed = client.get("/api/board/plans", params={"start": "2026-11-15 21:00", "end": "2026-11-15 22:30"})
        assert listed.status_code == 200
        body = listed.json()
        assert body["count"] == 1
        assert body["plans"][0]["plan_id"].endswith(":trend")
        assert body["plans"][0]["generated_utc8"] == "2026-11-15 21:45"
        exported = client.get("/api/board/plans.csv", params={"start": "2026-11-15 21:45", "end": "2026-11-15 21:45"})
        assert exported.status_code == 200
        assert "text/csv" in exported.headers["content-type"]
        text = exported.content.decode("utf-8-sig")
        assert text.splitlines()[0].startswith("plan_id,")
        assert "顺势反抽空" in text
        assert "trend_near_key" in text
        bad = client.get("/api/board/plans", params={"start": "tomorrow"})
        assert bad.status_code == 400
    assert parse_bound("2026-11-15", end=True) > parse_bound("2026-11-15")
