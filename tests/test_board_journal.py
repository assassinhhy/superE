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


def _set_outcome(journal, plan_id, *, outcome, filled=0, expired_unfilled=0, r_taker=None, r_maker=None):
    journal.conn.execute(
        """
        UPDATE plan_journal
        SET outcome=?, filled=?, expired_unfilled=?, r_taker=?, r_maker=?
        WHERE plan_id=?
        """,
        (outcome, filled, expired_unfilled, r_taker, r_maker, plan_id),
    )
    journal.conn.commit()


def _history(journal, cfg) -> dict[str, str]:
    """Five plans covering every outcome bucket, kind and status."""
    entries = (
        ("trend", "顺势", "可进场", GENERATED),
        ("break", "破位", "等待信号", GENERATED + 900_000),
        ("counter", "逆势", "可进场", GENERATED + 1_800_000),
        ("trend2", "顺势", "可进场", GENERATED + 2_700_000),
        ("flip", "翻转", "等待信号", GENERATED + 3_600_000),
    )
    ids = {}
    for rule_id, kind, status, generated in entries:
        journal.record_board(
            _board(
                _plan(
                    id=rule_id,
                    kind=kind,
                    status=status,
                    generated_ms=generated,
                    basis_close_ms=generated,
                    expires_ms=generated + 3_600_000,
                )
            ),
            cfg,
        )
        ids[rule_id] = f"ETHUSDT:{generated}:{rule_id}"
    _set_outcome(journal, ids["trend"], outcome="resolved", filled=1, r_taker=2.0, r_maker=2.1)
    _set_outcome(journal, ids["break"], outcome="timeout", filled=1, r_taker=-1.0, r_maker=-0.95)
    _set_outcome(journal, ids["counter"], outcome="expired", expired_unfilled=1)
    _set_outcome(journal, ids["trend2"], outcome="open", filled=1)
    return ids


def test_summarize_plans_counts_groups_and_filters(tmp_path: Path):
    cfg = _cfg()
    journal = PlanJournal(journal_path(tmp_path))
    ids = _history(journal, cfg)

    stats = journal.summarize_plans()
    assert stats["total"] == 5
    assert stats["filled"] == 3
    assert stats["expired"] == 1
    assert stats["open"] == 1
    assert stats["pending"] == 0
    assert stats["resolved"] == 2
    assert stats["skipped"] == 1
    assert stats["fill_rate"] == 0.75  # 3 filled / 4 tradable (翻转 has no entry)
    assert stats["win_rate"] == 0.5  # one of the two settled trades made R
    assert stats["total_r_taker"] == 1.0
    assert stats["avg_r_taker"] == 0.5
    assert stats["total_r_maker"] == 1.15
    assert stats["avg_r_maker"] == 0.575

    assert set(stats["by_kind"]) == {"顺势", "破位", "逆势", "翻转"}
    trend = stats["by_kind"]["顺势"]
    assert (trend["total"], trend["filled"], trend["resolved"]) == (2, 2, 1)
    assert trend["win_rate"] == 1.0
    assert trend["total_r_taker"] == 2.0
    breakout = stats["by_kind"]["破位"]
    assert breakout["win_rate"] == 0.0
    assert breakout["total_r_taker"] == -1.0
    assert stats["by_kind"]["逆势"]["expired"] == 1
    assert stats["by_kind"]["翻转"]["skipped"] == 1
    assert stats["by_kind"]["翻转"]["fill_rate"] == 0.0

    assert stats["by_status"]["可进场"]["total"] == 3
    assert stats["by_status"]["等待信号"]["total"] == 2

    only_trend = journal.summarize_plans(kind="顺势")
    assert only_trend["total"] == 2
    assert set(only_trend["by_kind"]) == {"顺势"}
    waiting = journal.summarize_plans(status="等待信号")
    assert waiting["total"] == 2
    assert set(waiting["by_status"]) == {"等待信号"}
    first_bar = journal.summarize_plans(start_ms=GENERATED, end_ms=GENERATED)
    assert first_bar["total"] == 1
    assert first_bar["by_kind"]["顺势"]["total"] == 1

    journal.update_note(ids["trend"], followed=True)
    followed = journal.summarize_plans(followed=True)
    assert followed["total"] == 1
    assert followed["total_r_taker"] == 2.0
    assert journal.summarize_plans(followed=False)["total"] == 4
    journal.update_note(ids["trend"], followed=False)
    journal.close()


def test_list_plans_filters_by_status_kind_and_followed(tmp_path: Path):
    cfg = _cfg()
    journal = PlanJournal(journal_path(tmp_path))
    ids = _history(journal, cfg)

    assert [row["rule_id"] for row in journal.list_plans(status="可进场")] == [
        "trend2",
        "counter",
        "trend",
    ]
    assert [row["rule_id"] for row in journal.list_plans(kind="破位")] == ["break"]
    assert journal.list_plans(followed=True) == []
    journal.update_note(ids["counter"], followed=True)
    followed = journal.list_plans(followed=True)
    assert [row["rule_id"] for row in followed] == ["counter"]
    assert followed[0]["followed"] == 1
    assert followed[0]["user_note"] is None
    combined = journal.list_plans(status="可进场", kind="顺势", followed=True)
    assert combined == []
    assert len(journal.list_plans(start_ms=GENERATED, end_ms=GENERATED + 900_000)) == 2
    journal.close()


def test_update_note_writes_flag_and_review_text(tmp_path: Path):
    cfg = _cfg()
    journal = PlanJournal(journal_path(tmp_path))
    ids = _history(journal, cfg)
    plan_id = ids["trend"]

    row = journal.update_note(plan_id, followed=True, note="  按计划执行，止损不挪  ")
    assert row is not None
    assert row["plan_id"] == plan_id
    assert row["followed"] == 1
    assert row["user_note"] == "按计划执行，止损不挪"

    again = journal.update_note(plan_id, note="   ")
    assert again["followed"] == 1
    assert again["user_note"] is None

    unflagged = journal.update_note(plan_id, followed=False)
    assert unflagged["followed"] == 0
    assert unflagged["user_note"] is None

    untouched = journal.list_plans(kind="破位")[0]
    assert untouched["followed"] == 0
    assert untouched["user_note"] is None
    assert journal.update_note("no-such-plan", followed=True) is None
    journal.close()


def test_note_columns_are_added_to_an_existing_table(tmp_path: Path):
    path = journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE plan_journal (plan_id TEXT PRIMARY KEY, symbol TEXT, generated_ms INTEGER)"
    )
    conn.execute("INSERT INTO plan_journal (plan_id, symbol, generated_ms) VALUES ('old', 'ETHUSDT', 1)")
    conn.commit()
    conn.close()

    journal = PlanJournal(path)
    row = journal.list_plans()[0]
    assert row["followed"] == 0
    assert row["user_note"] is None
    updated = journal.update_note("old", followed=True, note="补记")
    assert updated["followed"] == 1
    assert updated["user_note"] == "补记"
    assert "followed" in journal.to_csv([updated]).splitlines()[0]
    assert journal.to_csv([updated]).splitlines()[0].endswith("followed,user_note")
    journal.close()


def test_stats_and_note_api_with_filters(tmp_path: Path):
    cfg = _cfg()
    app = create_app(tmp_path, cfg, schedule=False, frozen={"ok": True})
    with TestClient(app) as client:
        journal = app.state.journal
        ids = _history(journal, cfg)

        stats = client.get("/api/board/stats")
        assert stats.status_code == 200
        body = stats.json()
        assert body["total"] == 5
        assert body["win_rate"] == 0.5
        assert body["by_kind"]["破位"]["total"] == 1
        filtered = client.get("/api/board/stats", params={"status": "可进场"})
        assert filtered.status_code == 200
        assert filtered.json()["total"] == 3
        by_kind = client.get("/api/board/stats", params={"kind": "顺势"})
        assert by_kind.json()["total"] == 2
        bad = client.get("/api/board/stats", params={"end": "not-a-date"})
        assert bad.status_code == 400

        listed = client.get("/api/board/plans", params={"status": "可进场", "kind": "顺势"})
        assert listed.status_code == 200
        assert listed.json()["count"] == 2
        assert client.get("/api/board/plans", params={"followed": "true"}).json()["count"] == 0

        marked = client.post(
            f"/api/board/plans/{ids['trend']}/note",
            json={"followed": True, "note": "复盘：按计划"},
        )
        assert marked.status_code == 200
        saved = marked.json()["plan"]
        assert saved["plan_id"] == ids["trend"]
        assert saved["followed"] == 1
        assert saved["user_note"] == "复盘：按计划"
        assert client.get("/api/board/plans", params={"followed": "true"}).json()["count"] == 1
        assert client.get("/api/board/stats", params={"followed": "true"}).json()["total"] == 1

        cleared = client.post(f"/api/board/plans/{ids['trend']}/note", json={"note": ""})
        assert cleared.status_code == 200
        assert cleared.json()["plan"]["user_note"] is None
        assert cleared.json()["plan"]["followed"] == 1

        missing = client.post("/api/board/plans/nope/note", json={"followed": True})
        assert missing.status_code == 404

        exported = client.get("/api/board/plans.csv", params={"followed": "true"})
        assert exported.status_code == 200
        text = exported.content.decode("utf-8-sig")
        header = text.splitlines()[0]
        assert header.endswith("followed,user_note")
        assert ids["trend"] in text

        preflight = client.request(
            "OPTIONS",
            f"/api/board/plans/{ids['trend']}/note",
            headers={
                "Origin": "http://127.0.0.1:8766",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert "POST" in preflight.headers["access-control-allow-methods"]


def test_board_poll_resolves_once_a_minute_while_plan_endpoints_force(tmp_path: Path):
    cfg = _cfg()
    clock = {"ms": GENERATED}
    app = create_app(tmp_path, cfg, schedule=False, clock=lambda: clock["ms"])
    journal = app.state.journal
    calls: list[int] = []
    journal.resolve_pending = lambda load_bars, now_ms: (calls.append(int(now_ms)), 0)[1]
    with TestClient(app) as client:
        assert calls == [GENERATED]  # startup forces one resolve
        assert client.get("/api/board").status_code == 200
        assert client.get("/api/board").status_code == 200
        assert calls == [GENERATED]  # the 10s panel poll is throttled
        assert client.get("/api/board/plans").status_code == 200
        assert calls == [GENERATED, GENERATED]  # plans force a resolve
        clock["ms"] = GENERATED + 61_000
        assert client.get("/api/board").status_code == 200
        assert calls == [GENERATED, GENERATED, GENERATED + 61_000]
        assert client.get("/api/board/stats").status_code == 200
        assert calls[-1] == GENERATED + 61_000
        assert len(calls) == 4  # stats force a resolve even inside the window
