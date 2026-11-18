"""Plan journal. The board process writes this file only, never the main database.

One row per plan per closed 15m bar. Ten-second polls update the live status of
that row and do not insert another. A resolver later reads 1m klines (15m if
1m is missing) and fills the outcome. Nothing here places an order.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from supere.board.plans import RULES_VERSION
from supere.timeutil import UTC8, format_utc8

log = logging.getLogger("supere.board.journal")

TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
HORIZON_MS = 24 * 60 * 60 * 1000
SCHEMA_VERSION = 1
JOURNAL_NAME = "plan_journal.sqlite"

# Columns added on top of a table an older build may already have created.
_ADDED: dict[str, str] = {
    "symbol": "TEXT",
    "rule_id": "TEXT",
    "kind": "TEXT",
    "title": "TEXT",
    "direction": "TEXT",
    "entry_rule": "TEXT",
    "generated_ms": "INTEGER",
    "generated_utc8": "TEXT",
    "basis_close_ms": "INTEGER",
    "expires_ms": "INTEGER",
    "expires_utc8": "TEXT",
    "validity": "TEXT",
    "recorded_at": "INTEGER",
    "entry": "REAL",
    "entry_low": "REAL",
    "entry_high": "REAL",
    "stop": "REAL",
    "t1": "REAL",
    "t2": "REAL",
    "r_distance": "REAL",
    "t1_r": "REAL",
    "t2_r": "REAL",
    "key_level": "REAL",
    "price": "REAL",
    "funding_rate": "REAL",
    "oi": "REAL",
    "atr": "REAL",
    "status": "TEXT",
    "triggers": "TEXT",
    "rules_version": "TEXT",
    "config_hash": "TEXT",
    "outcome": "TEXT NOT NULL DEFAULT 'pending'",
    "filled": "INTEGER NOT NULL DEFAULT 0",
    "fill_ms": "INTEGER",
    "fill_utc8": "TEXT",
    "first_hit": "TEXT",
    "exit_reason": "TEXT",
    "t1_ms": "INTEGER",
    "t2_ms": "INTEGER",
    "stop_ms": "INTEGER",
    "expired_unfilled": "INTEGER NOT NULL DEFAULT 0",
    "mfe": "REAL",
    "mae": "REAL",
    "r_taker": "REAL",
    "r_maker": "REAL",
    "bar_interval": "TEXT",
    "resolved_at": "INTEGER",
    "detail_json": "TEXT",
    # Review annotations, written through ``update_note`` only.
    "followed": "INTEGER DEFAULT 0",
    "user_note": "TEXT",
}

_CREATE = """
CREATE TABLE IF NOT EXISTS plan_journal (
  plan_id TEXT PRIMARY KEY,
  symbol TEXT,
  rule_id TEXT,
  kind TEXT,
  title TEXT,
  direction TEXT,
  entry_rule TEXT,
  generated_ms INTEGER,
  generated_utc8 TEXT,
  basis_close_ms INTEGER,
  expires_ms INTEGER,
  expires_utc8 TEXT,
  validity TEXT,
  recorded_at INTEGER,
  entry REAL,
  entry_low REAL,
  entry_high REAL,
  stop REAL,
  t1 REAL,
  t2 REAL,
  r_distance REAL,
  t1_r REAL,
  t2_r REAL,
  key_level REAL,
  price REAL,
  funding_rate REAL,
  oi REAL,
  atr REAL,
  status TEXT,
  triggers TEXT,
  rules_version TEXT,
  config_hash TEXT,
  outcome TEXT NOT NULL DEFAULT 'pending',
  filled INTEGER NOT NULL DEFAULT 0,
  fill_ms INTEGER,
  fill_utc8 TEXT,
  first_hit TEXT,
  exit_reason TEXT,
  t1_ms INTEGER,
  t2_ms INTEGER,
  stop_ms INTEGER,
  expired_unfilled INTEGER NOT NULL DEFAULT 0,
  mfe REAL,
  mae REAL,
  r_taker REAL,
  r_maker REAL,
  bar_interval TEXT,
  resolved_at INTEGER,
  detail_json TEXT,
  followed INTEGER DEFAULT 0,
  user_note TEXT
);
"""

CSV_FIELDS = [
    "plan_id",
    "symbol",
    "rule_id",
    "kind",
    "title",
    "direction",
    "entry_rule",
    "generated_ms",
    "generated_utc8",
    "basis_close_ms",
    "expires_ms",
    "expires_utc8",
    "validity",
    "recorded_at",
    "entry",
    "entry_low",
    "entry_high",
    "stop",
    "t1",
    "t2",
    "r_distance",
    "t1_r",
    "t2_r",
    "key_level",
    "price",
    "funding_rate",
    "oi",
    "atr",
    "status",
    "triggers",
    "rules_version",
    "config_hash",
    "outcome",
    "filled",
    "fill_ms",
    "fill_utc8",
    "first_hit",
    "exit_reason",
    "t1_ms",
    "t2_ms",
    "stop_ms",
    "expired_unfilled",
    "mfe",
    "mae",
    "r_taker",
    "r_maker",
    "bar_interval",
    "resolved_at",
    "followed",
    "user_note",
]

# Everything the JSON API exposes. ``detail_json`` stays out of the CSV export.
_PUBLIC_FIELDS = [*CSV_FIELDS, "detail_json"]

_OUTCOME_FIELDS = (
    "outcome",
    "filled",
    "fill_ms",
    "fill_utc8",
    "first_hit",
    "exit_reason",
    "t1_ms",
    "t2_ms",
    "stop_ms",
    "expired_unfilled",
    "mfe",
    "mae",
    "r_taker",
    "r_maker",
    "bar_interval",
    "resolved_at",
    "detail_json",
)

_LIVE_UPDATE = ("status", "price", "funding_rate", "oi", "triggers")


def journal_path(data_root: Path) -> Path:
    return Path(data_root) / "board" / JOURNAL_NAME


def rules_fingerprint(analysis) -> tuple[str, str]:
    """Rules version plus a short hash of the plan and confirm config."""
    payload = {
        "rules": RULES_VERSION,
        "plans": analysis.plans.model_dump(),
        "confirm": analysis.confirm.model_dump(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return RULES_VERSION, hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_bound(value: str | None, *, end: bool = False) -> int | None:
    """Epoch ms, or ``YYYY-MM-DD[ HH:MM[:SS]]`` read as UTC+8.

    A date with no clock is the start of that UTC+8 day. As an end bound it
    is the last millisecond of that day.
    """
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            moment = datetime.strptime(text, fmt).replace(tzinfo=UTC8)
        except ValueError:
            continue
        ms = int(moment.timestamp() * 1000)
        if end and fmt == "%Y-%m-%d":
            ms = int((moment + timedelta(days=1)).timestamp() * 1000) - 1
        return ms
    raise ValueError(text)


def _leg(side: str, entry: float, exit_px: float) -> float:
    if side == "short":
        return entry - exit_px
    return exit_px - entry


def _fee_r(entry: float, stop: float, gross: float) -> tuple[float | None, float | None]:
    risk = abs(entry - stop)
    if risk <= 0:
        return None, None
    taker = (gross - entry * TAKER_FEE * 2) / risk
    maker = (gross - entry * MAKER_FEE * 2) / risk
    return taker, maker


def _covers(bars: list[dict], start_ms: int, end_ms: int, interval_ms: int) -> bool:
    if int(end_ms) <= int(start_ms):
        return True
    if not bars:
        return False
    first = min(int(bar["open_time"]) for bar in bars)
    last = max(int(bar["close_time"]) for bar in bars)
    return first <= int(start_ms) + int(interval_ms) and last >= int(end_ms) - int(interval_ms)


def assess_plan(plan: dict, bars: list[dict], now_ms: int, *, interval_ms: int) -> dict | None:
    """Outcome for one stored plan. ``None`` means the klines do not say yet.

    A fill is a later bar whose high-low touches the fixed entry before expiry.
    The same bar hitting the stop and a target counts as the stop. The fill bar
    itself may only resolve the stop: a target touched by that same bar could
    have printed before the fill, so T1/T2 are checked from the next bar on.
    Final R is the backtest's half-at-T1 path, minus taker 0.05%×2 and maker
    0.02%×2.
    """
    entry = plan.get("entry")
    stop = plan.get("stop")
    if plan.get("kind") == "翻转" or entry is None or stop is None:
        return {
            "outcome": "skipped",
            "filled": 0,
            "expired_unfilled": 0,
            "resolved_at": int(now_ms),
        }
    entry = float(entry)
    stop = float(stop)
    generated = int(plan["generated_ms"])
    expires = int(plan["expires_ms"])
    side = plan.get("direction") or plan.get("side") or "long"
    targets = [float(level) for level in (plan.get("targets") or [])]
    t1 = plan.get("t1") if plan.get("t1") is not None else (targets[0] if targets else None)
    t2 = plan.get("t2") if plan.get("t2") is not None else (targets[1] if len(targets) > 1 else None)
    t1 = None if t1 is None else float(t1)
    t2 = None if t2 is None else float(t2)
    closed = [
        bar
        for bar in bars
        if int(bar["close_time"]) <= int(now_ms) and int(bar["open_time"]) >= generated
    ]
    closed.sort(key=lambda bar: int(bar["open_time"]))
    window = [bar for bar in closed if int(bar["open_time"]) < expires]
    fill = next((bar for bar in window if float(bar["low"]) <= entry <= float(bar["high"])), None)
    entry_covered = _covers(closed, generated, min(int(now_ms), expires), interval_ms)
    if fill is None:
        if int(now_ms) >= expires and entry_covered:
            return {
                "outcome": "expired",
                "filled": 0,
                "expired_unfilled": 1,
                "first_hit": None,
                "exit_reason": "unfilled",
                "fill_ms": None,
                "fill_utc8": None,
                "t1_ms": None,
                "t2_ms": None,
                "stop_ms": None,
                "mfe": None,
                "mae": None,
                "r_taker": None,
                "r_maker": None,
                "resolved_at": int(now_ms),
                "detail_json": None,
            }
        return None

    fill_ms = int(fill["close_time"])
    mfe = 0.0
    mae = 0.0
    t1_ms = t2_ms = stop_ms = None
    exit_reason = None
    gross = None
    half = False
    finished = False
    last_close = float(fill["close"])
    horizon_end = int(fill["open_time"]) + HORIZON_MS
    path = [bar for bar in closed if int(bar["open_time"]) >= int(fill["open_time"])]
    for bar in path:
        if int(bar["open_time"]) >= horizon_end:
            break
        last_close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        if side == "short":
            mfe = max(mfe, entry - low)
            mae = max(mae, high - entry)
            hit_stop = high >= stop
            hit_t1 = t1 is not None and low <= t1
            hit_t2 = t2 is not None and low <= t2
        else:
            mfe = max(mfe, high - entry)
            mae = max(mae, entry - low)
            hit_stop = low <= stop
            hit_t1 = t1 is not None and high >= t1
            hit_t2 = t2 is not None and high >= t2
        stamp = int(bar["close_time"])
        if int(bar["open_time"]) == int(fill["open_time"]):
            # Entry is only known to have traded somewhere inside this bar.
            # A target touched by the fill bar may have printed before the
            # fill, so only the stop is allowed to resolve here.
            hit_t1 = False
            hit_t2 = False
        if not half:
            if hit_stop:
                stop_ms = stamp
                exit_reason = "stop"
                gross = _leg(side, entry, stop)
                finished = True
                break
            if hit_t1:
                t1_ms = stamp
                half = True
                exit_reason = "t1"
                gross = 0.5 * _leg(side, entry, t1)
                if t2 is None:
                    finished = True
                    break
                if hit_t2:
                    t2_ms = stamp
                    exit_reason = "t2"
                    gross += 0.5 * _leg(side, entry, t2)
                    finished = True
                    break
            continue
        if hit_stop:
            stop_ms = stamp
            exit_reason = "t1_stop"
            gross = (gross or 0.0) + 0.5 * _leg(side, entry, stop)
            finished = True
            break
        if hit_t2:
            t2_ms = stamp
            exit_reason = "t2"
            gross = (gross or 0.0) + 0.5 * _leg(side, entry, t2)
            finished = True
            break

    if not finished:
        horizon_known = int(now_ms) >= horizon_end and _covers(path or closed, int(fill["open_time"]), horizon_end, interval_ms)
        if horizon_known:
            if half:
                exit_reason = "t1_timeout"
                gross = (gross or 0.0) + 0.5 * _leg(side, entry, last_close)
                outcome = "resolved"
            else:
                exit_reason = "timeout"
                gross = _leg(side, entry, last_close)
                outcome = "timeout"
            finished = True
        else:
            outcome = "open"
            gross = None
    elif exit_reason == "timeout":
        outcome = "timeout"
    else:
        outcome = "resolved"

    if exit_reason == "stop" or (stop_ms is not None and t1_ms is None):
        first_hit = "stop"
    elif t1_ms is not None:
        first_hit = "t1"
    elif t2_ms is not None:
        first_hit = "t2"
    else:
        first_hit = None
    r_taker = r_maker = None
    if gross is not None:
        r_taker, r_maker = _fee_r(entry, stop, gross)
    detail = None
    if gross is not None:
        detail = json.dumps({"gross": gross}, separators=(",", ":"))
    return {
        "outcome": outcome,
        "filled": 1,
        "expired_unfilled": 0,
        "fill_ms": fill_ms,
        "fill_utc8": format_utc8(fill_ms, "%Y-%m-%d %H:%M"),
        "first_hit": first_hit,
        "exit_reason": exit_reason,
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "stop_ms": stop_ms,
        "mfe": mfe,
        "mae": mae,
        "r_taker": r_taker,
        "r_maker": r_maker,
        "resolved_at": int(now_ms) if finished else None,
        "detail_json": detail,
    }


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(plan_journal)")}
    for name, decl in _ADDED.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE plan_journal ADD COLUMN {name} {decl}")


def _public_row(row: sqlite3.Row) -> dict:
    available = set(row.keys())
    item = {key: row[key] for key in _PUBLIC_FIELDS if key in available}
    raw = item.get("triggers")
    if isinstance(raw, str) and raw:
        try:
            item["triggers"] = json.loads(raw)
        except json.JSONDecodeError:
            pass
    for key in ("filled", "expired_unfilled", "followed"):
        if item.get(key) is not None:
            item[key] = int(item[key])
    return item


def _filter_clauses(
    start_ms: int | None,
    end_ms: int | None,
    status: str | None,
    kind: str | None,
    followed: bool | None,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if start_ms is not None:
        clauses.append("generated_ms>=?")
        params.append(int(start_ms))
    if end_ms is not None:
        clauses.append("generated_ms<=?")
        params.append(int(end_ms))
    if status is not None:
        clauses.append("status=?")
        params.append(str(status))
    if kind is not None:
        clauses.append("kind=?")
        params.append(str(kind))
    if followed is not None:
        clauses.append("followed=?")
        params.append(1 if followed else 0)
    return clauses, params


def _ratio(numerator: float, denominator: int) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / int(denominator), 4) + 0.0


def _rounded(value: float) -> float:
    # ``+ 0.0`` normalises a negative zero so JSON never renders ``-0.0``.
    return round(float(value), 4) + 0.0


def _stats(rows: list[dict]) -> dict:
    """Counts and R multiples for one set of plan rows.

    ``fill_rate`` excludes skipped plans (翻转 has no entry to fill), and
    ``win_rate`` divides positive ``r_taker`` by settled filled plans
    (``resolved``/``timeout`` with an R value).
    """
    total = len(rows)
    filled = sum(1 for row in rows if int(row.get("filled") or 0))
    expired = sum(
        1
        for row in rows
        if row.get("outcome") == "expired" or int(row.get("expired_unfilled") or 0)
    )
    open_count = sum(1 for row in rows if row.get("outcome") == "open")
    pending = sum(1 for row in rows if row.get("outcome") == "pending")
    skipped = sum(1 for row in rows if row.get("outcome") == "skipped")
    settled = [row for row in rows if row.get("outcome") in ("resolved", "timeout")]
    taker = [float(row["r_taker"]) for row in settled if row.get("r_taker") is not None]
    maker = [float(row["r_maker"]) for row in settled if row.get("r_maker") is not None]
    wins = sum(1 for value in taker if value > 0)
    return {
        "total": total,
        "filled": filled,
        "expired": expired,
        "open": open_count,
        "pending": pending,
        "resolved": len(settled),
        "skipped": skipped,
        "fill_rate": _ratio(filled, total - skipped),
        "win_rate": _ratio(wins, len(taker)),
        "total_r_taker": _rounded(sum(taker)),
        "avg_r_taker": _ratio(sum(taker), len(taker)),
        "total_r_maker": _rounded(sum(maker)),
        "avg_r_maker": _ratio(sum(maker), len(maker)),
    }


def _grouped(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        name = row.get(key)
        if not name:
            continue
        groups.setdefault(str(name), []).append(row)
    return {name: _stats(group) for name, group in groups.items()}


class PlanJournal:
    """WAL sqlite under ``data/board/plan_journal.sqlite``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_CREATE)
        _ensure_columns(self.conn)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_journal_generated ON plan_journal(generated_ms)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_journal_outcome ON plan_journal(outcome)")
        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def record_board(self, board: dict, analysis, *, recorded_at: int | None = None) -> int:
        """Insert plans for this 15m bar. A repeat poll updates status only."""
        plans = board.get("plans") or []
        if not plans:
            return 0
        symbol = board.get("symbol") or analysis.board.symbol
        version, digest = rules_fingerprint(analysis)
        price = (board.get("price") or {}).get("value")
        funding = (board.get("funding") or {}).get("rate")
        oi = (board.get("oi") or {}).get("value")
        atr = (board.get("atr") or {}).get("15m")
        moment = int(recorded_at if recorded_at is not None else board.get("as_of") or 0)
        inserted = 0
        with self._lock:
            for plan in plans:
                if plan.get("generated_ms") is None:
                    continue
                generated = int(plan["generated_ms"])
                plan_id = f"{symbol}:{generated}:{plan.get('id')}"
                targets = [float(level) for level in (plan.get("targets") or [])]
                triggers = {
                    "rule_id": plan.get("id"),
                    "entry_rule": plan.get("entry_rule"),
                    "confirm": (plan.get("confirm") or {}).get("names") or [],
                }
                tradable = plan.get("kind") != "翻转" and plan.get("stop") is not None and plan.get("entry") is not None
                row = {
                    "plan_id": plan_id,
                    "symbol": symbol,
                    "rule_id": plan.get("id"),
                    "kind": plan.get("kind"),
                    "title": plan.get("title"),
                    "direction": plan.get("side"),
                    "entry_rule": plan.get("entry_rule"),
                    "generated_ms": generated,
                    "generated_utc8": format_utc8(generated, "%Y-%m-%d %H:%M"),
                    "basis_close_ms": int(plan.get("basis_close_ms") or generated),
                    "expires_ms": int(plan["expires_ms"]),
                    "expires_utc8": format_utc8(int(plan["expires_ms"]), "%Y-%m-%d %H:%M"),
                    "validity": plan.get("validity"),
                    "recorded_at": moment,
                    "entry": plan.get("entry"),
                    "entry_low": plan.get("entry_low"),
                    "entry_high": plan.get("entry_high"),
                    "stop": plan.get("stop"),
                    "t1": targets[0] if targets else None,
                    "t2": targets[1] if len(targets) > 1 else None,
                    "r_distance": plan.get("r_distance"),
                    "t1_r": plan.get("t1_r"),
                    "t2_r": plan.get("t2_r"),
                    "key_level": plan.get("key_level"),
                    "price": price,
                    "funding_rate": funding,
                    "oi": oi,
                    "atr": atr,
                    "status": plan.get("status"),
                    "triggers": json.dumps(triggers, ensure_ascii=False, separators=(",", ":")),
                    "rules_version": version,
                    "config_hash": digest,
                    "outcome": "pending" if tradable else "skipped",
                    "filled": 0,
                    "expired_unfilled": 0,
                }
                columns = list(row)
                placeholders = ",".join("?" for _ in columns)
                updates = ", ".join(f"{name}=excluded.{name}" for name in _LIVE_UPDATE)
                before = self.conn.total_changes
                self.conn.execute(
                    f"""
                    INSERT INTO plan_journal ({",".join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT(plan_id) DO UPDATE SET {updates}
                    WHERE plan_journal.outcome IN ('pending', 'open')
                    """,
                    [row[name] for name in columns],
                )
                if self.conn.total_changes > before:
                    inserted += 1
            self.conn.commit()
        return inserted

    def resolve_pending(self, load_bars, now_ms: int) -> int:
        """Fill outcomes for rows that are still pending or open. Safe to repeat."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM plan_journal WHERE outcome IN ('pending', 'open')"
            ).fetchall()
            changed = 0
            for raw in rows:
                plan = dict(raw)
                try:
                    result, interval = _resolve_with_bars(plan, load_bars, int(now_ms))
                except Exception:
                    log.exception("计划 %s 结算失败", plan.get("plan_id"))
                    continue
                if result is None:
                    continue
                if interval:
                    result["bar_interval"] = interval
                self._apply(plan["plan_id"], result)
                changed += 1
            self.conn.commit()
            return changed

    def _apply(self, plan_id: str, fields: dict) -> None:
        usable = {key: value for key, value in fields.items() if key in _OUTCOME_FIELDS}
        if not usable:
            return
        assignments = ", ".join(f"{name}=?" for name in usable)
        self.conn.execute(
            f"UPDATE plan_journal SET {assignments} WHERE plan_id=?",
            (*usable.values(), plan_id),
        )

    def list_plans(
        self,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 200,
        status: str | None = None,
        kind: str | None = None,
        followed: bool | None = None,
    ) -> list[dict]:
        clauses, params = _filter_clauses(start_ms, end_ms, status, kind, followed)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit), 5000)))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM plan_journal{where} ORDER BY generated_ms DESC, plan_id ASC LIMIT ?",
                params,
            ).fetchall()
        return [_public_row(row) for row in rows]

    def summarize_plans(
        self,
        start_ms: int | None = None,
        end_ms: int | None = None,
        status: str | None = None,
        kind: str | None = None,
        followed: bool | None = None,
    ) -> dict:
        """Performance summary over every matching row (no limit)."""
        clauses, params = _filter_clauses(start_ms, end_ms, status, kind, followed)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = [
                dict(row)
                for row in self.conn.execute(
                    """
                    SELECT kind, status, outcome, filled, expired_unfilled, r_taker, r_maker
                    FROM plan_journal
                    """ + where,
                    params,
                ).fetchall()
            ]
        summary = _stats(rows)
        summary["by_kind"] = _grouped(rows, "kind")
        summary["by_status"] = _grouped(rows, "status")
        return summary

    def update_note(
        self,
        plan_id: str,
        *,
        followed: bool | None = None,
        note: str | None = None,
    ) -> dict | None:
        """Set the follow flag and review note on one plan. ``None`` = no such row.

        A whitespace-only note clears the column back to NULL.
        """
        assignments: list[str] = []
        params: list[object] = []
        if followed is not None:
            assignments.append("followed=?")
            params.append(1 if followed else 0)
        if note is not None:
            text = str(note).strip()
            assignments.append("user_note=?")
            params.append(text or None)
        with self._lock:
            exists = self.conn.execute(
                "SELECT 1 FROM plan_journal WHERE plan_id=?", (plan_id,)
            ).fetchone()
            if exists is None:
                return None
            if assignments:
                self.conn.execute(
                    f"UPDATE plan_journal SET {', '.join(assignments)} WHERE plan_id=?",
                    (*params, plan_id),
                )
                self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM plan_journal WHERE plan_id=?", (plan_id,)
            ).fetchone()
        return _public_row(row)

    def to_csv(self, rows: list[dict]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            triggers = flat.get("triggers")
            if not isinstance(triggers, str):
                flat["triggers"] = json.dumps(triggers or {}, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(flat)
        return buffer.getvalue()


def _resolve_with_bars(plan: dict, load_bars, now_ms: int) -> tuple[dict | None, str | None]:
    if plan.get("kind") == "翻转" or plan.get("entry") is None or plan.get("stop") is None:
        return (
            {"outcome": "skipped", "filled": 0, "expired_unfilled": 0, "resolved_at": now_ms},
            None,
        )
    generated = int(plan["generated_ms"])
    span_end = int(plan["expires_ms"]) + HORIZON_MS
    chosen: list[dict] | None = None
    interval = None
    step = 60_000
    for name, width in (("1m", 60_000), ("15m", 900_000)):
        try:
            loaded = load_bars(name, generated, span_end) or []
        except Exception:
            loaded = []
        bars = [bar for bar in loaded if bar.get("open_time") is not None and bar.get("close_time") is not None]
        if not bars:
            continue
        chosen = bars
        interval = name
        step = width
        if name == "1m":
            break
    if chosen is None:
        return None, None
    return assess_plan(plan, chosen, now_ms, interval_ms=step), interval
