"""Download + convert stages for the order-book archive backfill.

Stage order is fixed by the plan::

    bookDepth 2023-01-01 -> now
    aggTrades  recent 2 years
    metrics    2021-12-01 -> now
    aggTrades  down to 2019-12-31

Downloads use at most four worker threads.  Conversion stays in the calling
thread, one archive at a time, so the training job keeps its CPU.
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from supere.hist.archives import (
    ArchiveItem,
    ArchiveMissing,
    VisionClient,
    archive_prefix,
    archive_url,
)
from supere.hist.convert import convert_archive
from supere.hist.manifest import ArchiveRecord, Manifest
from supere.hist.schema import KIND_FIRST_DAY, partition_dir

log = logging.getLogger("supere.hist.backfill")

STAGES = ("bookdepth", "aggtrades-recent", "metrics", "aggtrades-old", "aggtrades-gapfill")

# Fixed boundary between the "recent two years" stage and the older history.
# Keeping it constant prevents the two windows from sliding apart as the
# calendar advances.
AGGT_OLD_END = dt.date(2024, 11, 4)

STAGE_KIND = {
    "bookdepth": "bookDepth",
    "aggtrades-recent": "aggTrades",
    "aggtrades-old": "aggTrades",
    "aggtrades-gapfill": "aggTrades",
    "metrics": "metrics",
}


@dataclass
class StageReport:
    stage: str
    planned: int
    converted: int
    skipped: int
    missing: int
    failed: int
    rows: int
    bytes: int
    missing_stamps: list[str]


def _parse_day(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def _month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


def _next_month(day: dt.date) -> dt.date:
    return (day.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def _month_last(day: dt.date) -> dt.date:
    return _next_month(day) - dt.timedelta(days=1)


def _days(start: dt.date, end: dt.date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += dt.timedelta(days=1)


def _available(client: VisionClient, kind: str, period: str, symbol: str) -> dict[str, int]:
    """List the bucket; a failure must stop the stage, not look like "0 files"."""
    prefix = archive_prefix(kind, period, symbol)
    return client.list_archives(prefix)


def _stamp_from_filename(kind: str, filename: str) -> str | None:
    prefix = f"-{kind}-"
    if prefix not in filename:
        return None
    tail = filename.split(prefix, 1)[1]
    if tail.endswith(".zip"):
        tail = tail[:-4]
    return tail


def _latest_available_daily(client: VisionClient, symbol: str) -> dt.date:
    latest: dt.date | None = None
    for kind in ("aggTrades", "bookDepth", "metrics"):
        names = _available(client, kind, "daily", symbol)
        for name in names:
            stamp = _stamp_from_filename(kind, name)
            if stamp and len(stamp) == 10:
                day = _parse_day(stamp)
                latest = day if latest is None or day > latest else latest
    if latest is None:
        latest = dt.date.today() - dt.timedelta(days=1)
    return latest


def build_items(
    stage: str,
    *,
    symbol: str = "ETHUSDT",
    client: VisionClient,
    end: dt.date | None = None,
) -> tuple[list[ArchiveItem], list[str]]:
    """Return ``(items, missing_stamps)`` for a stage.

    ``missing_stamps`` lists calendar slots that should exist but were absent
    from the bucket listing.  They are reported, not silently skipped.
    """
    kind = STAGE_KIND[stage]
    symbol = symbol.upper()
    daily = _available(client, kind, "daily", symbol)
    monthly = _available(client, kind, "monthly", symbol) if kind == "aggTrades" else {}
    latest = end or _latest_available_daily(client, symbol)
    first = _parse_day(KIND_FIRST_DAY[kind])

    if stage == "aggtrades-recent":
        start = AGGT_OLD_END + dt.timedelta(days=1)
    elif stage == "aggtrades-old":
        start = first
        latest = min(end or AGGT_OLD_END, AGGT_OLD_END)
    else:
        start = first

    if start < first:
        start = first

    items: list[ArchiveItem] = []
    missing: list[str] = []

    if kind == "aggTrades":
        cursor = _month_start(start)
        while cursor <= latest:
            m_first = cursor
            m_last = _month_last(cursor)
            slot_start = max(m_first, start)
            slot_end = min(m_last, latest)
            month_stamp = cursor.strftime("%Y-%m")
            full = slot_start == m_first and slot_end == m_last
            if full and f"{symbol}-aggTrades-{month_stamp}.zip" in monthly:
                items.append(
                    ArchiveItem(
                        kind="aggTrades",
                        period="monthly",
                        stamp=month_stamp,
                        filename=f"{symbol}-aggTrades-{month_stamp}.zip",
                        url=archive_url("aggTrades", "monthly", month_stamp, symbol),
                    )
                )
            else:
                for day in _days(slot_start, slot_end):
                    stamp = day.isoformat()
                    name = f"{symbol}-aggTrades-{stamp}.zip"
                    if name in daily:
                        items.append(
                            ArchiveItem(
                                kind="aggTrades",
                                period="daily",
                                stamp=stamp,
                                filename=name,
                                url=archive_url("aggTrades", "daily", stamp, symbol),
                            )
                        )
                    else:
                        missing.append(stamp)
            cursor = _next_month(cursor)
    else:
        for day in _days(start, latest):
            stamp = day.isoformat()
            name = f"{symbol}-{kind}-{stamp}.zip"
            if name in daily:
                items.append(
                    ArchiveItem(
                        kind=kind,
                        period="daily",
                        stamp=stamp,
                        filename=name,
                        url=archive_url(kind, "daily", stamp, symbol),
                    )
                )
            else:
                missing.append(stamp)
    return items, missing


def monthly_gap_items(
    raw_root: Path,
    client: VisionClient,
    symbol: str = "ETHUSDT",
) -> tuple[list[ArchiveItem], list[str]]:
    """Find days missing from converted monthly aggTrades archives.

    Binance monthly archives are occasionally missing a day or three even
    though the daily archive exists.  Return one daily archive item per such
    day so the gap-fill stage can overwrite the month with complete data.
    """
    symbol = symbol.upper()
    manifest = Manifest(Path(raw_root) / "_hist" / "manifest.jsonl")
    daily = _available(client, "aggTrades", "daily", symbol)
    items: list[ArchiveItem] = []
    missing: list[str] = []
    for record in manifest.converted().values():
        if record.get("kind") != "aggTrades" or record.get("period") != "monthly":
            continue
        stamp = record.get("stamp")
        if not stamp or len(stamp) != 7:
            continue
        year, month = map(int, stamp.split("-"))
        have = set((record.get("date_counts") or {}).keys())
        for day in _days(dt.date(year, month, 1), _month_last(dt.date(year, month, 1))):
            date = day.isoformat()
            if date in have:
                continue
            name = f"{symbol}-aggTrades-{date}.zip"
            if name in daily:
                items.append(
                    ArchiveItem(
                        kind="aggTrades",
                        period="daily",
                        stamp=date,
                        filename=name,
                        url=archive_url("aggTrades", "daily", date, symbol),
                    )
                )
            else:
                missing.append(date)
    return items, missing


def _already_converted(record: dict | None, raw_root: Path, symbol: str) -> bool:
    if not record:
        return False
    counts = record.get("date_counts") or {}
    if not counts:
        return False
    for date, expected in counts.items():
        path = partition_dir(raw_root, record["kind"], date, symbol) / "part-000.parquet"
        if not path.exists():
            return False
        try:
            if pq.read_metadata(path).num_rows != int(expected):
                return False
        except Exception:  # noqa: BLE001 - unreadable partition => rebuild
            return False
    return True


def run_stage(
    stage: str,
    *,
    client: VisionClient,
    raw_root: Path,
    symbol: str = "ETHUSDT",
    workers: int = 4,
    end: dt.date | None = None,
    on_progress=None,
) -> StageReport:
    raw_root = Path(raw_root)
    manifest = Manifest(raw_root / "_hist" / "manifest.jsonl")
    known = manifest.converted()

    items, missing_stamps = (
        monthly_gap_items(raw_root, client, symbol)
        if stage == "aggtrades-gapfill"
        else build_items(stage, symbol=symbol, client=client, end=end)
    )
    pending: list[ArchiveItem] = []
    skipped = 0
    for item in items:
        if _already_converted(known.get(item.filename), raw_root, symbol):
            skipped += 1
        else:
            pending.append(item)

    log.info("阶段 %s: 计划 %d，已完成 %d，待处理 %d，缺失槽位 %d",
             stage, len(items), skipped, len(pending), len(missing_stamps))
    if on_progress:
        on_progress(
            f"阶段 {stage} 计划 {len(items)} 个存档，已完成 {skipped}，待处理 {len(pending)}"
        )

    converted = failed = 0
    rows_total = bytes_total = 0
    missing = 0
    tmp_root = raw_root / "_hist_tmp"
    zip_dir = raw_root / "_hist_zips"
    done_count = 0

    def record_missing(item: ArchiveItem, exc: Exception) -> None:
        nonlocal missing
        missing += 1
        manifest.append(
            ArchiveRecord(
                kind=item.kind,
                period=item.period,
                stamp=item.stamp,
                filename=item.filename,
                url=item.url,
                status="missing",
                error=str(exc)[:300],
            )
        )

    def process(item: ArchiveItem, result) -> None:
        nonlocal converted, failed, rows_total, bytes_total
        if result is None:
            return
        try:
            conv = convert_archive(
                result.path,
                item.kind,
                raw_root,
                symbol=symbol,
                period=item.period,
                tmp_root=tmp_root,
            )
        except Exception as exc:  # noqa: BLE001 - keep the zip for a retry
            failed += 1
            log.exception("转换失败 %s: %s", item.filename, exc)
            manifest.append(
                ArchiveRecord(
                    kind=item.kind,
                    period=item.period,
                    stamp=item.stamp,
                    filename=item.filename,
                    url=item.url,
                    sha256=result.sha256,
                    bytes=result.bytes,
                    status="failed",
                    error=str(exc)[:500],
                )
            )
            return
        manifest.append(
            ArchiveRecord(
                kind=item.kind,
                period=item.period,
                stamp=item.stamp,
                filename=item.filename,
                url=item.url,
                sha256=result.sha256,
                bytes=result.bytes,
                rows=conv.rows,
                date_counts=conv.date_counts,
                parquet=conv.parquet,
                status="converted",
            )
        )
        result.path.unlink(missing_ok=True)
        side = result.path.with_name(result.path.name + ".sha256")
        side.unlink(missing_ok=True)
        converted += 1
        rows_total += conv.rows
        bytes_total += result.bytes

    def report(force: bool = False) -> None:
        if not force and done_count % 10 and done_count != len(pending):
            return
        msg = (
            f"进度：{stage} {done_count}/{len(pending)} 完成 {converted} "
            f"失败 {failed} 缺失 {missing} 行 {rows_total:,} "
            f"下载 {bytes_total / 1e9:.2f} GB"
        )
        log.info(msg)
        if on_progress:
            on_progress(msg)

    executor = ThreadPoolExecutor(max_workers=max(1, min(4, workers)))
    iterator = iter(pending)
    inflight: dict = {}
    try:
        while True:
            while len(inflight) < max(1, min(4, workers)):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                inflight[executor.submit(client.download, item, zip_dir)] = item
            if not inflight:
                break
            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                item = inflight.pop(future)
                done_count += 1
                try:
                    result = future.result()
                except ArchiveMissing as exc:
                    record_missing(item, exc)
                except Exception as exc:  # noqa: BLE001 - one bad archive must not stop the stage
                    failed += 1
                    log.exception("下载失败 %s: %s", item.filename, exc)
                    manifest.append(
                        ArchiveRecord(
                            kind=item.kind,
                            period=item.period,
                            stamp=item.stamp,
                            filename=item.filename,
                            url=item.url,
                            status="failed",
                            error=str(exc)[:500],
                        )
                    )
                else:
                    process(item, result)
                report()
    finally:
        executor.shutdown(wait=True)

    report(force=True)
    return StageReport(
        stage=stage,
        planned=len(items),
        converted=converted,
        skipped=skipped,
        missing=missing,
        failed=failed,
        rows=rows_total,
        bytes=bytes_total,
        missing_stamps=missing_stamps,
    )
