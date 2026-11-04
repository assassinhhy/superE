"""Missing kline open times, and aggTrade id holes."""

from __future__ import annotations

from collections.abc import Iterable

from supere.intervals import align_open, next_open


def find_missing_ranges(
    interval: str,
    opens_sorted: Iterable[int],
    start_open: int,
    end_open: int,
) -> list[tuple[int, int]]:
    """Inclusive ranges of missing open times between ``start_open`` and ``end_open``."""
    if end_open < start_open:
        return []
    expected = align_open(interval, start_open)
    if expected < start_open:
        expected = next_open(interval, expected)
    gaps: list[tuple[int, int]] = []
    iterator = iter(opens_sorted)
    current = next(iterator, None)
    gap_start: int | None = None
    last_missing: int | None = None
    while expected <= end_open:
        while current is not None and current < expected:
            current = next(iterator, None)
        if current == expected:
            if gap_start is not None and last_missing is not None:
                gaps.append((gap_start, last_missing))
                gap_start = None
                last_missing = None
            current = next(iterator, None)
        else:
            if gap_start is None:
                gap_start = expected
            last_missing = expected
        nxt = next_open(interval, expected)
        if nxt <= expected:
            break
        expected = nxt
    if gap_start is not None and last_missing is not None:
        gaps.append((gap_start, last_missing))
    return gaps


def id_gaps(ids: Iterable[int]) -> list[tuple[int, int]]:
    """Inclusive missing id ranges. Duplicate ids are ignored."""
    ordered = sorted(set(ids))
    holes: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if right > left + 1:
            holes.append((left + 1, right - 1))
    return holes
