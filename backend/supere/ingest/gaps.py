"""Missing kline open times, and aggTrade id holes."""

from __future__ import annotations

from collections.abc import Iterable

from supere.intervals import align_open, next_open, previous_open


def find_missing_ranges(
    interval: str,
    opens_sorted: Iterable[int],
    start_open: int,
    end_open: int,
) -> list[tuple[int, int]]:
    """Inclusive ranges of missing open times between ``start_open`` and ``end_open``.

    Holes are measured from the bars actually stored, stepping with
    ``next_open`` of the previous bar. That keeps a 3-day series on whatever
    phase Binance used for that bar. A single synthetic grid would mark every
    candle after the 2023-08 phase change as missing.
    """
    if end_open < start_open:
        return []
    opens = sorted({int(item) for item in opens_sorted if start_open <= int(item) <= end_open})
    gaps: list[tuple[int, int]] = []
    if not opens:
        aligned = align_open(interval, start_open)
        if aligned < start_open:
            aligned = next_open(interval, aligned)
        tail_end = end_open if align_open(interval, end_open) == end_open else align_open(interval, end_open)
        if tail_end < aligned:
            return []
        if tail_end > end_open:
            tail_end = previous_open(interval, tail_end)
        if tail_end >= aligned:
            gaps.append((aligned, tail_end))
        return gaps
    if opens[0] > start_open:
        lead = _bounded_gap(interval, start_open, opens[0])
        if lead is not None:
            gaps.append(lead)
    for left, right in zip(opens, opens[1:]):
        nxt = next_open(interval, left)
        if right <= nxt:
            continue
        hole_end = previous_open(interval, right)
        if hole_end >= nxt:
            gaps.append((nxt, hole_end))
    if end_open > opens[-1]:
        nxt = next_open(interval, opens[-1])
        tail_end = end_open if align_open(interval, end_open) == end_open else align_open(interval, end_open)
        if tail_end > end_open:
            tail_end = previous_open(interval, tail_end)
        if tail_end >= nxt:
            gaps.append((nxt, tail_end))
    return gaps


def _bounded_gap(interval: str, start_open: int, next_present: int) -> tuple[int, int] | None:
    aligned = align_open(interval, start_open)
    if aligned < start_open:
        aligned = next_open(interval, aligned)
    hole_end = previous_open(interval, next_present)
    if hole_end >= aligned:
        return aligned, hole_end
    return None


def id_gaps(ids: Iterable[int]) -> list[tuple[int, int]]:
    """Inclusive missing id ranges. Duplicate ids are ignored."""
    ordered = sorted(set(ids))
    holes: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if right > left + 1:
            holes.append((left + 1, right - 1))
    return holes
