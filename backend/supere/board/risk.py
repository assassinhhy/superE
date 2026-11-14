"""Weekend note and the macro window. Loss streak itself is counted in the page."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from supere.timeutil import UTC8, format_utc8

_WEEKDAY = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def load_calendar(path: Path) -> list[dict]:
    file = Path(path)
    if not file.is_file():
        return []
    rows = []
    with file.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            text = (item.get("time_utc8") or "").strip()
            kind = (item.get("kind") or "").strip()
            if not text or not kind:
                continue
            try:
                moment = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC8)
            except ValueError:
                continue
            rows.append(
                {
                    "ts_event": int(moment.timestamp() * 1000),
                    "kind": kind,
                    "title": (item.get("title") or kind).strip(),
                }
            )
    return rows


def assess(now_ms: int, events: list[dict], calendar: list[dict], params) -> dict:
    weekday = _WEEKDAY[datetime.fromtimestamp(now_ms / 1000, tz=UTC8).weekday()]
    weekend = weekday in set(params.weekend_days)
    window = params.event_window_minutes * 60_000
    kinds = {kind.upper() for kind in params.event_kinds}
    merged = []
    for item in list(events) + list(calendar):
        kind = str(item.get("kind") or "").upper()
        if kind not in kinds:
            continue
        ts = int(item["ts_event"])
        if abs(ts - now_ms) > 18 * 3600_000:
            continue
        blocking = abs(ts - now_ms) <= window
        start = format_utc8(ts - window, "%H:%M")
        end = format_utc8(ts + window, "%H:%M")
        merged.append(
            {
                "ts_event": ts,
                "utc8": format_utc8(ts, "%H:%M"),
                "kind": kind,
                "title": item.get("title") or kind,
                "blocking": blocking,
                "window": f"{start}–{end}",
            }
        )
    merged.sort(key=lambda item: item["ts_event"])
    notes = []
    if weekend:
        notes.append("周末流动性薄，仓位减半、止损放宽，别贴整数位")
    if any(item["blocking"] for item in merged):
        notes.append("不开新单")
    if not notes:
        notes.append("连亏计数在页面里手动加减，到点停手")
    return {
        "weekend": weekend,
        "weekday": weekday,
        "loss_streak_stop": params.loss_streak_stop,
        "events": merged,
        "notes": notes,
    }
