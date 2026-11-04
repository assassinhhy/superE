"""SQLite access. The ingestor is the only writer; the API opens ``mode=ro``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from supere.storage.models import FundingRow, Kline, MetricRow
from supere.storage.schema import init_schema


def _ro_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


class Database:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            self.conn = sqlite3.connect(
                _ro_uri(self.path), uri=True, check_same_thread=False
            )
            self.conn.execute("PRAGMA query_only=ON")
            self.conn.execute("PRAGMA busy_timeout=5000")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=5000")
            init_schema(self.conn)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        conn.close()
        self.conn = None

    def checkpoint(self) -> None:
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def upsert_klines(self, rows: list[Kline]) -> int:
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO klines (
              market, symbol, interval, open_time, close_time,
              o, h, l, c, volume, quote_volume, trades,
              taker_buy_base, taker_buy_quote, source, ingested_at, ingested_local
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(market, symbol, interval, open_time) DO UPDATE SET
              close_time=excluded.close_time,
              o=excluded.o, h=excluded.h, l=excluded.l, c=excluded.c,
              volume=excluded.volume, quote_volume=excluded.quote_volume,
              trades=excluded.trades,
              taker_buy_base=excluded.taker_buy_base,
              taker_buy_quote=excluded.taker_buy_quote,
              source=excluded.source,
              ingested_at=excluded.ingested_at,
              ingested_local=excluded.ingested_local
            """,
            [
                (
                    row.market,
                    row.symbol,
                    row.interval,
                    row.open_time,
                    row.close_time,
                    row.o,
                    row.h,
                    row.l,
                    row.c,
                    row.volume,
                    row.quote_volume,
                    row.trades,
                    row.taker_buy_base,
                    row.taker_buy_quote,
                    row.source,
                    row.ingested_at,
                    row.ingested_local,
                )
                for row in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def fetch_klines(
        self,
        market: str,
        symbol: str,
        interval: str,
        *,
        start: int | None = None,
        end: int | None = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        clauses = ["market=?", "symbol=?", "interval=?"]
        params: list[object] = [market, symbol, interval]
        if start is not None:
            clauses.append("open_time>=?")
            params.append(start)
        if end is not None:
            clauses.append("open_time<=?")
            params.append(end)
        params.append(limit)
        where = " AND ".join(clauses)
        # Newest page, returned in chronological order.
        sql = f"""
            SELECT * FROM (
              SELECT * FROM klines WHERE {where}
              ORDER BY open_time DESC LIMIT ?
            ) ORDER BY open_time ASC
        """
        return list(self.conn.execute(sql, params))

    def iter_opens(
        self, market: str, symbol: str, interval: str, start: int, end: int
    ):
        cur = self.conn.execute(
            """
            SELECT open_time FROM klines
            WHERE market=? AND symbol=? AND interval=?
              AND open_time>=? AND open_time<=?
            ORDER BY open_time ASC
            """,
            (market, symbol, interval, start, end),
        )
        for row in cur:
            yield int(row[0])

    def min_max_open(self, market: str, symbol: str, interval: str) -> tuple[int, int] | None:
        row = self.conn.execute(
            """
            SELECT MIN(open_time), MAX(open_time), COUNT(*)
            FROM klines WHERE market=? AND symbol=? AND interval=?
            """,
            (market, symbol, interval),
        ).fetchone()
        if row is None or row[2] == 0:
            return None
        return int(row[0]), int(row[1])

    def count_klines(self, market: str | None = None) -> int:
        if market is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0])
        return int(
            self.conn.execute("SELECT COUNT(*) FROM klines WHERE market=?", (market,)).fetchone()[0]
        )

    def upsert_metrics(self, rows: list[MetricRow]) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO um_metrics (
              symbol, ts, oi, oi_value, top_pos_lsr, top_acc_lsr, global_lsr, taker_lsr, source
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
              oi=COALESCE(excluded.oi, um_metrics.oi),
              oi_value=COALESCE(excluded.oi_value, um_metrics.oi_value),
              top_pos_lsr=COALESCE(excluded.top_pos_lsr, um_metrics.top_pos_lsr),
              top_acc_lsr=COALESCE(excluded.top_acc_lsr, um_metrics.top_acc_lsr),
              global_lsr=COALESCE(excluded.global_lsr, um_metrics.global_lsr),
              taker_lsr=COALESCE(excluded.taker_lsr, um_metrics.taker_lsr),
              source=excluded.source
            """,
            [
                (
                    row.symbol,
                    row.ts,
                    row.oi,
                    row.oi_value,
                    row.top_pos_lsr,
                    row.top_acc_lsr,
                    row.global_lsr,
                    row.taker_lsr,
                    row.source,
                )
                for row in rows
            ],
        )
        self.conn.commit()

    def recent_metrics(self, symbol: str, limit: int = 10) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT * FROM um_metrics WHERE symbol=?
                ORDER BY ts DESC LIMIT ?
                """,
                (symbol, limit),
            )
        )

    def metrics_between(
        self,
        symbol: str,
        start: int | None,
        end: int | None,
        limit: int = 2000,
    ) -> list[sqlite3.Row]:
        clauses = ["symbol=?"]
        params: list[object] = [symbol]
        if start is not None:
            clauses.append("ts>=?")
            params.append(start)
        if end is not None:
            clauses.append("ts<=?")
            params.append(end)
        params.append(limit)
        where = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM (
              SELECT * FROM um_metrics WHERE {where}
              ORDER BY ts DESC LIMIT ?
            ) ORDER BY ts ASC
        """
        return list(self.conn.execute(sql, params))

    def funding_between(
        self,
        symbol: str,
        start: int | None,
        end: int | None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        clauses = ["symbol=?"]
        params: list[object] = [symbol]
        if start is not None:
            clauses.append("funding_time>=?")
            params.append(start)
        if end is not None:
            clauses.append("funding_time<=?")
            params.append(end)
        params.append(limit)
        where = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM (
              SELECT * FROM funding WHERE {where}
              ORDER BY funding_time DESC LIMIT ?
            ) ORDER BY funding_time ASC
        """
        return list(self.conn.execute(sql, params))

    def latest_runtime(self, venue: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runtime_stats WHERE venue=? ORDER BY ts DESC LIMIT 1",
            (venue,),
        ).fetchone()

    def list_signals(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        )

    def list_jobs(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM jobs ORDER BY COALESCE(started_at, 0) DESC LIMIT ?",
                (limit,),
            )
        )

    def list_models(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        )

    def upsert_funding(self, rows: list[FundingRow]) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO funding (symbol, funding_time, rate, mark_price, source)
            VALUES (?,?,?,?,?)
            ON CONFLICT(symbol, funding_time) DO UPDATE SET
              rate=excluded.rate,
              mark_price=COALESCE(excluded.mark_price, funding.mark_price),
              source=excluded.source
            """,
            [
                (row.symbol, row.funding_time, row.rate, row.mark_price, row.source)
                for row in rows
            ],
        )
        self.conn.commit()

    def upsert_funding_live(
        self,
        symbol: str,
        rate: str | None,
        next_funding_time: int | None,
        mark_price: str | None,
        index_price: str | None,
        updated_at: int,
        updated_local: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO funding_live (
              symbol, rate, next_funding_time, mark_price, index_price, updated_at, updated_local
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(symbol) DO UPDATE SET
              rate=excluded.rate,
              next_funding_time=excluded.next_funding_time,
              mark_price=excluded.mark_price,
              index_price=excluded.index_price,
              updated_at=excluded.updated_at,
              updated_local=excluded.updated_local
            """,
            (symbol, rate, next_funding_time, mark_price, index_price, updated_at, updated_local),
        )
        self.conn.commit()

    def upsert_coverage(
        self,
        market: str,
        symbol: str,
        interval: str,
        range_from: int | None,
        range_to: int | None,
        gaps: list[tuple[int, int]],
        last_check: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO coverage (market, symbol, interval, range_from, range_to, gaps_json, last_check)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(market, symbol, interval) DO UPDATE SET
              range_from=excluded.range_from,
              range_to=excluded.range_to,
              gaps_json=excluded.gaps_json,
              last_check=excluded.last_check
            """,
            (
                market,
                symbol,
                interval,
                range_from,
                range_to,
                json.dumps([{"from": a, "to": b} for a, b in gaps], separators=(",", ":")),
                last_check,
            ),
        )
        self.conn.commit()

    def list_coverage(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM coverage ORDER BY market, interval"))

    def add_event(
        self,
        ts: int,
        level: str,
        component: str,
        code: str | None,
        msg: str,
        detail: dict | None = None,
        ts_local: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO app_events (ts, ts_local, level, component, code, msg, detail_json)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                ts,
                ts_local,
                level,
                component,
                code,
                msg,
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")) if detail else None,
            ),
        )
        self.conn.commit()

    def count_events(self, code: str, since: int | None = None) -> int:
        if since is None:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM app_events WHERE code=?", (code,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM app_events WHERE code=? AND ts>=?", (code, since)
            ).fetchone()
        return int(row[0])

    def list_events(self, limit: int = 100, code: str | None = None) -> list[sqlite3.Row]:
        if code is None:
            return list(
                self.conn.execute(
                    "SELECT * FROM app_events ORDER BY id DESC LIMIT ?", (limit,)
                )
            )
        return list(
            self.conn.execute(
                "SELECT * FROM app_events WHERE code=? ORDER BY id DESC LIMIT ?",
                (code, limit),
            )
        )

    def event_times(self, code: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT ts FROM app_events WHERE code=? ORDER BY ts ASC", (code,)
        )
        return [int(row[0]) for row in rows]

    def apply_raw_coverage(
        self,
        stream: str,
        date: str,
        first_ts: int | None,
        last_ts: int | None,
        msg_count: int,
        new_gaps: list[dict],
    ) -> None:
        row = self.conn.execute(
            "SELECT first_ts, last_ts, msg_count, gap_count, gaps_json FROM raw_coverage WHERE stream=? AND date=?",
            (stream, date),
        ).fetchone()
        if row is None:
            gaps = new_gaps[-200:]
            self.conn.execute(
                """
                INSERT INTO raw_coverage (stream, date, first_ts, last_ts, msg_count, gap_count, gaps_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    stream,
                    date,
                    first_ts,
                    last_ts,
                    msg_count,
                    len(new_gaps),
                    json.dumps(gaps, separators=(",", ":")),
                ),
            )
        else:
            old_gaps = json.loads(row["gaps_json"] or "[]")
            merged = (old_gaps + new_gaps)[-200:]
            first = row["first_ts"]
            if first_ts is not None:
                first = first_ts if first is None else min(int(first), first_ts)
            last = row["last_ts"]
            if last_ts is not None:
                last = last_ts if last is None else max(int(last), last_ts)
            self.conn.execute(
                """
                UPDATE raw_coverage
                SET first_ts=?, last_ts=?, msg_count=?, gap_count=?, gaps_json=?
                WHERE stream=? AND date=?
                """,
                (
                    first,
                    last,
                    int(row["msg_count"]) + msg_count,
                    int(row["gap_count"]) + len(new_gaps),
                    json.dumps(merged, separators=(",", ":")),
                    stream,
                    date,
                ),
            )
        self.conn.commit()

    def raw_coverage_rows(self, stream: str | None = None) -> list[sqlite3.Row]:
        if stream is None:
            return list(self.conn.execute("SELECT * FROM raw_coverage ORDER BY date, stream"))
        return list(
            self.conn.execute(
                "SELECT * FROM raw_coverage WHERE stream=? ORDER BY date", (stream,)
            )
        )

    def set_stream_state(self, stream: str, last_id: int | None, last_ts: int | None, updated_at: int) -> None:
        self.conn.execute(
            """
            INSERT INTO stream_state (stream, last_id, last_ts, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(stream) DO UPDATE SET
              last_id=COALESCE(excluded.last_id, stream_state.last_id),
              last_ts=COALESCE(excluded.last_ts, stream_state.last_ts),
              updated_at=excluded.updated_at
            """,
            (stream, last_id, last_ts, updated_at),
        )
        self.conn.commit()

    def get_stream_state(self, stream: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM stream_state WHERE stream=?", (stream,)
        ).fetchone()

    def add_runtime_stat(
        self,
        ts: int,
        venue: str,
        minute: int,
        peak: int,
        header: int | None,
        header_limit: int,
        halted: bool,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runtime_stats (
              ts, venue, self_weight_minute, self_weight_peak, header_weight, header_limit, halted
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(ts, venue) DO UPDATE SET
              self_weight_minute=excluded.self_weight_minute,
              self_weight_peak=excluded.self_weight_peak,
              header_weight=excluded.header_weight,
              header_limit=excluded.header_limit,
              halted=excluded.halted
            """,
            (ts, venue, minute, peak, header, header_limit, 1 if halted else 0),
        )
        self.conn.commit()

    def max_self_peak(self, venue: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(self_weight_peak) FROM runtime_stats WHERE venue=?", (venue,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def max_header(self, venue: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(header_weight) FROM runtime_stats WHERE venue=?", (venue,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def insert_anomaly(
        self,
        ts: int,
        kind: str,
        value: float,
        threshold: float,
        detail: dict,
        linked_news_ids: list[int],
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO market_anomaly (ts, kind, value, threshold, detail_json, linked_news_ids)
            VALUES (?,?,?,?,?,?)
            """,
            (
                ts,
                kind,
                value,
                threshold,
                json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                json.dumps(linked_news_ids),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_anomalies(self, limit: int = 100) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM market_anomaly ORDER BY ts DESC LIMIT ?", (limit,)
            )
        )

    def insert_news(self, fields: dict) -> int:
        columns = [
            "ts_published",
            "ts_ingested",
            "ts_local",
            "source",
            "source_tier",
            "lang",
            "title",
            "body",
            "url",
            "url_hash",
            "text_simhash",
            "cluster_id",
            "category",
            "direction",
            "impact",
            "eth_relevance",
            "credibility",
            "entities",
            "scored_by",
            "score_json",
            "user_label",
            "status",
        ]
        values = [fields.get(name) for name in columns]
        placeholders = ",".join("?" for _ in columns)
        cur = self.conn.execute(
            f"INSERT INTO news ({','.join(columns)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def news_by_url_hash(self, url_hash: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM news WHERE url_hash=? ORDER BY id DESC LIMIT 1", (url_hash,)
        ).fetchone()

    def news_ids_between(self, start: int, end: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM news WHERE ts_ingested BETWEEN ? AND ? ORDER BY id",
            (start, end),
        )
        return [int(row[0]) for row in rows]

    def upsert_fear_greed(
        self,
        ts: int,
        value: int,
        classification: str | None,
        ts_ingested: int,
        ts_local: int,
        known_at: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO fear_greed (ts, value, classification, ts_ingested, ts_local, known_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(ts) DO UPDATE SET
              value=excluded.value,
              classification=excluded.classification,
              ts_ingested=excluded.ts_ingested,
              ts_local=excluded.ts_local,
              known_at=excluded.known_at
            """,
            (ts, value, classification, ts_ingested, ts_local, known_at),
        )
        self.conn.commit()

    def list_fear_greed(self, limit: int = 30) -> list[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM fear_greed ORDER BY ts DESC LIMIT ?", (limit,))
        )
