"""Read-only analytical queries. DuckDB is not the system of record."""

from __future__ import annotations

from pathlib import Path

import duckdb


class DuckReader:
    def __init__(self, sqlite_path: Path, raw_root: Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.raw_root = Path(raw_root)
        self.con = duckdb.connect(database=":memory:")
        if self.sqlite_path.exists():
            uri = self.sqlite_path.resolve().as_posix().replace("'", "''")
            self.con.execute(f"ATTACH '{uri}' AS market (TYPE SQLITE, READ_ONLY)")

    def close(self) -> None:
        con = getattr(self, "con", None)
        if con is None:
            return
        con.close()
        self.con = None

    def query(self, sql: str, params: list | None = None):
        stripped = sql.lstrip().lower()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("DuckDB 入口只接受 SELECT")
        banned = ("insert", "update", "delete", "attach", "copy", "create", "drop", "pragma")
        if any(token in stripped for token in banned):
            raise ValueError("DuckDB 入口只接受只读查询")
        return self.con.execute(sql, params or []).fetchall()

    def kline_count(self, market: str, symbol: str, interval: str) -> int:
        rows = self.query(
            """
            SELECT COUNT(*) FROM market.klines
            WHERE market=? AND symbol=? AND interval=?
            """,
            [market, symbol, interval],
        )
        return int(rows[0][0])

    def parquet_summary(self, dataset: str, market: str, symbol: str, date: str) -> dict:
        folder = self.raw_root / dataset / market / symbol / f"date={date}"
        files = sorted(folder.glob("part-*.parquet")) if folder.exists() else []
        if not files:
            return {"rows": 0, "files": 0}
        glob = (folder / "part-*.parquet").as_posix()
        if dataset == "aggTrades":
            sql = """
                SELECT COUNT(*), MIN(agg_id), MAX(agg_id), MIN(ts), MAX(ts)
                FROM read_parquet(?)
            """
            count, min_id, max_id, min_ts, max_ts = self.con.execute(sql, [glob]).fetchone()
            return {
                "rows": int(count),
                "files": len(files),
                "min_id": min_id,
                "max_id": max_id,
                "min_ts": min_ts,
                "max_ts": max_ts,
            }
        if dataset == "depth":
            sql = "SELECT COUNT(*), MIN(recv_ms), MAX(recv_ms) FROM read_parquet(?)"
        else:
            sql = "SELECT COUNT(*), MIN(event_ms), MAX(event_ms) FROM read_parquet(?)"
        count, min_ts, max_ts = self.con.execute(sql, [glob]).fetchone()
        return {"rows": int(count), "files": len(files), "min_ts": min_ts, "max_ts": max_ts}

    def agg_ids(self, market: str, symbol: str, date: str) -> list[int]:
        folder = self.raw_root / "aggTrades" / market / symbol / f"date={date}"
        if not folder.exists() or not any(folder.glob("part-*.parquet")):
            return []
        glob = (folder / "part-*.parquet").as_posix()
        rows = self.con.execute(
            "SELECT DISTINCT agg_id FROM read_parquet(?) ORDER BY agg_id", [glob]
        ).fetchall()
        return [int(row[0]) for row in rows]
