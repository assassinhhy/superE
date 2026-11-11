from __future__ import annotations

import datetime as dt

from supere.hist.backfill import build_items, monthly_gap_items
from supere.hist.schema import dataset_name


class _FakeClient:
    def __init__(self, mapping: dict[str, dict[str, int]]) -> None:
        self.mapping = mapping

    def list_archives(self, prefix: str) -> dict[str, int]:
        return self.mapping.get(prefix, {})


def _daily(prefix: str, start: dt.date, end: dt.date) -> dict[str, int]:
    names = {}
    cursor = start
    while cursor <= end:
        names[f"ETHUSDT-{prefix}-{cursor.isoformat()}.zip"] = 1000
        cursor += dt.timedelta(days=1)
    return names


def _monthly(start: dt.date, end: dt.date) -> dict[str, int]:
    names = {}
    cursor = start.replace(day=1)
    while cursor <= end:
        names[f"ETHUSDT-aggTrades-{cursor.strftime('%Y-%m')}.zip"] = 1000
        cursor = (cursor.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return names


def _client() -> _FakeClient:
    return _FakeClient(
        {
            "data/futures/um/daily/aggTrades/ETHUSDT/": _daily(
                "aggTrades", dt.date(2019, 12, 31), dt.date(2026, 11, 4)
            ),
            "data/futures/um/monthly/aggTrades/ETHUSDT/": _monthly(
                dt.date(2020, 1, 1), dt.date(2026, 10, 31)
            ),
            "data/futures/um/daily/bookDepth/ETHUSDT/": _daily(
                "bookDepth", dt.date(2023, 1, 1), dt.date(2026, 11, 4)
            ),
            "data/futures/um/monthly/bookDepth/ETHUSDT/": {},
            "data/futures/um/daily/metrics/ETHUSDT/": _daily(
                "metrics", dt.date(2021, 12, 1), dt.date(2026, 11, 4)
            ),
            "data/futures/um/monthly/metrics/ETHUSDT/": {},
        }
    )


def _covered_days(items) -> set[str]:
    days: set[str] = set()
    for item in items:
        if item.period == "daily":
            days.add(item.stamp)
        else:
            year, month = map(int, item.stamp.split("-"))
            cursor = dt.date(year, month, 1)
            while cursor.month == month:
                days.add(cursor.isoformat())
                cursor += dt.timedelta(days=1)
    return days


def test_all_agg_trades_stages_cover_the_range_without_overlap():
    client = _client()
    recent, missing_recent = build_items("aggtrades-recent", client=client)
    old, missing_old = build_items("aggtrades-old", client=client)
    assert not missing_recent and not missing_old
    days = _covered_days(recent) | _covered_days(old)
    assert min(days) == "2019-12-31"
    assert max(days) == "2026-11-04"
    assert len(days) == (dt.date(2026, 11, 4) - dt.date(2019, 12, 31)).days + 1
    assert not (_covered_days(recent) & _covered_days(old))
    assert min(_covered_days(recent)) == "2024-11-05"
    assert max(_covered_days(old)) == "2024-11-04"
    assert sum(1 for item in recent if item.period == "monthly") == 23
    assert sum(1 for item in old if item.period == "monthly") == 58
    assert sum(1 for item in recent if item.period == "daily") == 30
    assert sum(1 for item in old if item.period == "daily") == 5


def test_daily_only_stages_and_missing_slots():
    client = _client()
    books, missing_books = build_items("bookdepth", client=client)
    assert len(books) == 1404  # 2023-01-01 .. 2026-11-04 inclusive in the fake listing
    assert books[0].stamp == "2023-01-01"
    assert books[-1].stamp == "2026-11-04"
    assert not missing_books

    client.mapping["data/futures/um/daily/bookDepth/ETHUSDT/"].pop(
        "ETHUSDT-bookDepth-2023-02-08.zip"
    )
    _books, missing = build_items("bookdepth", client=client)
    assert "2023-02-08" in missing


def test_recent_stage_start_does_not_slide_with_end():
    client = _client()
    recent, _ = build_items("aggtrades-recent", client=client, end=dt.date(2027, 1, 1))
    assert min(_covered_days(recent)) == "2024-11-05"


def test_monthly_gapfill_uses_daily_archives(tmp_path):
    import json

    raw = tmp_path / "raw"
    (raw / "_hist").mkdir(parents=True)
    record = {
        "kind": "aggTrades",
        "period": "monthly",
        "stamp": "2022-08",
        "filename": "ETHUSDT-aggTrades-2022-08.zip",
        "url": "u",
        "status": "converted",
        "rows": 10,
        "date_counts": {"2022-08-01": 1, "2022-08-02": 1},
    }
    (raw / "_hist" / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    client = _FakeClient(
        {
            "data/futures/um/daily/aggTrades/ETHUSDT/": {
                "ETHUSDT-aggTrades-2022-08-28.zip": 1
            }
        }
    )
    items, missing = monthly_gap_items(raw, client)
    assert [item.stamp for item in items] == ["2022-08-28"]
    assert "2022-08-03" in missing  # fake daily listing has no such file


def test_dataset_names_are_hist_suffixed():
    assert dataset_name("aggTrades") == "aggTrades_hist"
    assert dataset_name("bookDepth") == "bookDepth_hist"
    assert dataset_name("metrics") == "metrics_hist"
