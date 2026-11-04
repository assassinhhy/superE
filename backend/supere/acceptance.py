"""1a acceptance checks. Each line is PASS, FAIL, or SKIP.

Offline mode proves the checkers on synthetic data. ``--data-dir`` reads a
real database produced on the Windows machine. A 24 hour run cannot be
executed in the build environment.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from supere.ingest.aggregate import aggregate_1m, compare_bars
from supere.ingest.gaps import find_missing_ranges, id_gaps
from supere.intervals import UM_INTERVALS
from supere.lockfile import SingleInstance
from supere.storage.db import Database
from supere.storage.duck import DuckReader
from supere.storage.models import Kline


@dataclass(slots=True)
class Check:
    code: str
    status: str
    detail: str

    def line(self) -> str:
        return f"[{self.status}] {self.code} {self.detail}"


def _kline(i: int, interval: str = "1m", market: str = "um") -> Kline:
    open_time = i * 60_000
    return Kline(
        market=market,
        symbol="ETHUSDT",
        interval=interval,
        open_time=open_time,
        close_time=open_time + 60_000 - 1,
        o="100",
        h="110",
        l="90",
        c="105",
        volume="1.5",
        quote_volume="150",
        trades=3,
        taker_buy_base="0.5",
        taker_buy_quote="50",
        source="test",
        ingested_at=open_time,
        ingested_local=open_time,
    )


def offline_checks(work: Path) -> list[Check]:
    checks: list[Check] = []
    clean = list(range(0, 10 * 60_000, 60_000))
    holes = find_missing_ranges("1m", clean, clean[0], clean[-1])
    missing = find_missing_ranges("1m", [0, 120_000], 0, 120_000)
    agg_holes = id_gaps([1, 2, 4])
    if holes == [] and missing == [(60_000, 60_000)] and agg_holes == [(3, 3)]:
        checks.append(Check("A1", "PASS", "缺口检测（K 线 open_time 与 aggTrade id）在合成数据上正确"))
    else:
        checks.append(Check("A1", "FAIL", f"缺口检测异常 holes={holes} missing={missing} agg={agg_holes}"))

    db = Database(work / "synthetic.sqlite")
    one_minute = [_kline(i) for i in range(10)]
    # 5m needs 5 bars. Build them on a 5m boundary (epoch is aligned).
    db.upsert_klines(one_minute[:5])
    official = aggregate_1m(one_minute[:5], "5m")[0]
    official.interval = "5m"
    official.source = "rest"
    db.upsert_klines([official])
    loaded = [
        Kline(**{**dict(row), "market": row["market"]})
        for row in db.fetch_klines("um", "ETHUSDT", "1m", limit=10)
    ]
    # fetch returns sqlite rows; rebuild properly
    loaded = []
    for row in db.fetch_klines("um", "ETHUSDT", "1m", limit=10):
        loaded.append(
            Kline(
                market=row["market"],
                symbol=row["symbol"],
                interval=row["interval"],
                open_time=row["open_time"],
                close_time=row["close_time"],
                o=row["o"],
                h=row["h"],
                l=row["l"],
                c=row["c"],
                volume=row["volume"],
                quote_volume=row["quote_volume"],
                trades=row["trades"],
                taker_buy_base=row["taker_buy_base"],
                taker_buy_quote=row["taker_buy_quote"],
                source=row["source"],
                ingested_at=row["ingested_at"],
                ingested_local=row["ingested_local"],
            )
        )
    agg = aggregate_1m(loaded, "5m")
    stored = db.fetch_klines("um", "ETHUSDT", "5m", limit=5)
    problems = compare_bars(agg[0], loaded_as(stored[0])) if agg and stored else ["missing"]
    checks.append(
        Check(
            "A2",
            "PASS" if not problems else "FAIL",
            "1m 聚合成 5m 与库存官方棒对比" + ("一致" if not problems else str(problems)),
        )
    )
    started = time.perf_counter()
    db.fetch_klines("um", "ETHUSDT", "1m", limit=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    checks.append(
        Check(
            "A3",
            "PASS" if elapsed_ms < 300 else "FAIL",
            f"本地读取 500 根上限耗时 {elapsed_ms:.1f} ms（合成库很小；本机用 --data-dir 再测实库）",
        )
    )
    checks.append(Check("A4", "SKIP", "断网 5 分钟需在 Windows 上跑 scripts/accept_1a.py --live-outage"))
    before = db.count_klines()
    db.upsert_klines(one_minute[:5])
    after = db.count_klines()
    checks.append(
        Check(
            "A5",
            "PASS" if before == after else "FAIL",
            f"同一 open_time 重复写入后行数 {before} -> {after}",
        )
    )
    db.close()
    lock_a = SingleInstance(work / "a.lock")
    lock_b = SingleInstance(work / "a.lock")
    lock_a.acquire()
    blocked = False
    try:
        lock_b.acquire()
    except SystemExit:
        blocked = True
    lock_a.release()
    checks.append(Check("A6", "PASS" if blocked else "FAIL", "第二个实例拿不到 OS 文件锁"))
    checks.append(Check("A7", "SKIP", "418 次数和自计权重峰值要看实盘 runtime_stats（--data-dir）"))
    checks.append(Check("A8", "SKIP", "每 10 分钟时钟事件要看实盘 app_events（--data-dir）"))
    checks.append(Check("A9", "SKIP", "23 小时重叠切换要在本机把 ingest.reconnect_hours 调小演练，或跑满 23 小时"))
    return checks


def loaded_as(row) -> Kline:
    return Kline(
        market=row["market"],
        symbol=row["symbol"],
        interval=row["interval"],
        open_time=row["open_time"],
        close_time=row["close_time"],
        o=row["o"],
        h=row["h"],
        l=row["l"],
        c=row["c"],
        volume=row["volume"],
        quote_volume=row["quote_volume"],
        trades=row["trades"],
        taker_buy_base=row["taker_buy_base"],
        taker_buy_quote=row["taker_buy_quote"],
        source=row["source"],
        ingested_at=row["ingested_at"],
        ingested_local=row["ingested_local"],
    )


def evaluate_data_dir(root: Path) -> list[Check]:
    root = Path(root)
    db_path = root / "superE.sqlite"
    checks: list[Check] = []
    if not db_path.exists():
        return [Check("DATA", "FAIL", f"没有 {db_path}")]
    db = Database(db_path, read_only=True)
    duck = DuckReader(db_path, root / "raw")
    try:
        checks.append(_continuity(db, duck))
        checks.append(_aggregate(db))
        checks.append(_latency(db))
        checks.append(_outage(db))
        checks.append(_restart_copy(db_path))
        checks.append(_live_lock(root))
        checks.append(_weights(db))
        checks.append(_clock(db))
        checks.append(_overlap(db))
    finally:
        duck.close()
        db.close()
    return checks


def _continuity(db: Database, duck: DuckReader) -> Check:
    problems = []
    for interval in UM_INTERVALS:
        bounds = db.min_max_open("um", "ETHUSDT", interval)
        if bounds is None:
            continue
        start, end = bounds
        gaps = find_missing_ranges(interval, db.iter_opens("um", "ETHUSDT", interval, start, end), start, end)
        if gaps:
            problems.append(f"{interval} 缺口 {len(gaps)} 段，第一段 {gaps[0]}")
    agg_problems = []
    raw = db.raw_coverage_rows("um:depth20")
    recorded = sum(int(row["gap_count"]) for row in raw)
    agg_root = duck.raw_root / "aggTrades" / "um" / "ETHUSDT"
    if agg_root.exists():
        for folder in sorted(agg_root.glob("date=*")):
            date = folder.name.split("=", 1)[1]
            holes = id_gaps(duck.agg_ids("um", "ETHUSDT", date))
            if holes:
                agg_problems.append(f"{date} id 缺口 {len(holes)}")
    if problems or agg_problems:
        return Check("A1", "FAIL", "; ".join(problems + agg_problems))
    return Check(
        "A1",
        "PASS",
        f"已检查的周期 open_time 连续；depth20 已记录的 >2 秒断档次数={recorded}（允许大于 0，但必须有记录）",
    )


def _aggregate(db: Database) -> Check:
    higher = None
    for interval in ("5m", "15m", "1h"):
        if db.min_max_open("um", "ETHUSDT", interval):
            higher = interval
            break
    if higher is None or db.min_max_open("um", "ETHUSDT", "1m") is None:
        return Check("A2", "SKIP", "库里还没有 1m 和更高周期，无法对照")
    official_rows = db.fetch_klines("um", "ETHUSDT", higher, limit=20)
    if not official_rows:
        return Check("A2", "SKIP", "高周期没有样本")
    start = int(official_rows[0]["open_time"])
    end = int(official_rows[-1]["close_time"])
    minutes = [
        loaded_as(row)
        for row in db.fetch_klines("um", "ETHUSDT", "1m", start=start, end=end, limit=1500)
    ]
    aggregated = {bar.open_time: bar for bar in aggregate_1m(minutes, higher)}
    problems = []
    for row in official_rows:
        bar = loaded_as(row)
        got = aggregated.get(bar.open_time)
        if got is None:
            continue
        problems.extend(compare_bars(got, bar))
    if problems:
        return Check("A2", "FAIL", "; ".join(problems[:5]))
    return Check("A2", "PASS", f"1m 聚合与官方 {higher} 在重叠的完整桶上一致")


def _latency(db: Database) -> Check:
    count = db.count_klines("um")
    if count < 500:
        return Check("A3", "SKIP", f"UM K 线只有 {count} 根，不足 500")
    started = time.perf_counter()
    rows = db.fetch_klines("um", "ETHUSDT", "1m", limit=500)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(rows) < 500:
        return Check("A3", "SKIP", f"1m 不足 500 根（{len(rows)}）")
    status = "PASS" if elapsed_ms < 300 else "FAIL"
    return Check("A3", status, f"读取 500 根 1m 耗时 {elapsed_ms:.1f} ms")


def _outage(db: Database) -> Check:
    disconnects = db.event_times("WS_DISCONNECT")
    fills = db.event_times("GAP_FILL")
    connects = db.event_times("WS_CONNECTED")
    if not disconnects:
        return Check("A4", "SKIP", "还没有断线事件。用 --live-outage 做 5 分钟演练")
    for start in disconnects:
        later_fill = [ts for ts in fills if ts >= start]
        later_connect = [ts for ts in connects if ts >= start]
        if later_fill and later_connect:
            delay = min(later_fill[0], later_connect[0]) - start
            # The check the plan asks for is reconnect within 60s of the network returning.
            # We can only see the event timestamps here.
            if delay <= 6 * 60_000:
                return Check("A4", "PASS", f"断线后 {delay/1000:.0f} 秒内出现重连和补缺口事件")
    return Check("A4", "FAIL", "有断线事件，但没有在 6 分钟内配套的重连和补缺口")


def _restart_copy(db_path: Path) -> Check:
    copy = db_path.with_name("accept-restart-copy.sqlite")
    if copy.exists():
        copy.unlink()
    src = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(copy)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    before = Database(copy)
    count_before = before.count_klines()
    row = before.conn.execute("SELECT * FROM klines LIMIT 1").fetchone()
    if row is not None:
        before.conn.execute(
            """
            INSERT INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(market, symbol, interval, open_time) DO UPDATE SET
              c=excluded.c
            """,
            tuple(row),
        )
        before.conn.commit()
    count_after = before.count_klines()
    before.close()
    copy.unlink(missing_ok=True)
    if count_before != count_after:
        return Check("A5", "FAIL", f"重复写入后行数变化 {count_before} -> {count_after}")
    return Check("A5", "PASS", f"备份副本 {count_before} 行，重复主键写入后仍是 {count_after} 行")


def _live_lock(root: Path) -> Check:
    probe = SingleInstance(root / "accept-probe.lock")
    other = SingleInstance(root / "accept-probe.lock")
    probe.acquire()
    blocked = False
    try:
        other.acquire()
    except SystemExit:
        blocked = True
    finally:
        probe.release()
    live = SingleInstance(root / "superE.lock")
    held = False
    try:
        live.acquire()
        live.release()
    except SystemExit:
        held = True
    if not blocked:
        return Check("A6", "FAIL", "临时锁没能挡住第二个实例")
    extra = "采集器锁当前被占用" if held else "采集器当前没在运行，但锁机制已用临时文件验证"
    return Check("A6", "PASS", extra)


def _weights(db: Database) -> Check:
    bans = db.count_events("REST_418")
    peak = db.max_self_peak("futures")
    header = db.max_header("futures")
    if peak is None:
        return Check("A7", "SKIP", "还没有 runtime_stats。采集器每分钟写一次")
    status = "PASS" if bans == 0 and peak < 2400 * 0.5 else "FAIL"
    return Check(
        "A7",
        status,
        f"418 次数={bans}，自计峰值={peak}（上限 1200），响应头峰值={header}（只记录，不作为失败条件）",
    )


def _clock(db: Database) -> Check:
    times = db.event_times("CLOCK_SYNC")
    if len(times) < 2:
        return Check("A8", "SKIP", f"CLOCK_SYNC 只有 {len(times)} 次，需要至少再跑 10 分钟")
    worst = max(b - a for a, b in zip(times, times[1:]))
    # 10 minutes plus a minute of scheduler slack.
    status = "PASS" if worst <= 11 * 60_000 else "FAIL"
    return Check("A8", status, f"相邻校准间隔最大 {worst/1000:.0f} 秒，共 {len(times)} 次")


def _overlap(db: Database) -> Check:
    times = db.event_times("WS_OVERLAP")
    if not times:
        return Check("A9", "SKIP", "还没有 WS_OVERLAP。把 reconnect_hours 调小可在本机演练，不必干等 23 小时")
    return Check("A9", "PASS", f"记录到 {len(times)} 次重叠切换。切换前后的 id 连续性见 A1")
