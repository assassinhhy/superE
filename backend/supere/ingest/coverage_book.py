"""In-memory counters flushed into ``raw_coverage``."""

from __future__ import annotations

from dataclasses import dataclass, field

from supere.storage.db import Database


@dataclass
class _Bucket:
    first_ts: int | None = None
    last_ts: int | None = None
    msg_count: int = 0
    gaps: list[dict] = field(default_factory=list)


class CoverageBook:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def add(self, stream: str, date: str, ts: int | None) -> None:
        bucket = self._buckets.setdefault((stream, date), _Bucket())
        bucket.msg_count += 1
        if ts is None:
            return
        bucket.first_ts = ts if bucket.first_ts is None else min(bucket.first_ts, ts)
        bucket.last_ts = ts if bucket.last_ts is None else max(bucket.last_ts, ts)

    def gap(self, stream: str, date: str, start_ms: int, end_ms: int, reason: str) -> None:
        bucket = self._buckets.setdefault((stream, date), _Bucket())
        bucket.gaps.append({"from": start_ms, "to": end_ms, "reason": reason})

    def flush(self, db: Database) -> None:
        items = list(self._buckets.items())
        self._buckets.clear()
        for (stream, date), bucket in items:
            if bucket.msg_count == 0 and not bucket.gaps:
                continue
            db.apply_raw_coverage(
                stream,
                date,
                bucket.first_ts,
                bucket.last_ts,
                bucket.msg_count,
                bucket.gaps,
            )
