"""Online SQLite backup and a monthly restore drill."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from supere.timeutil import UTC8


def backup_sqlite(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(src)
    target = sqlite3.connect(dest)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def copy_new_parquet(raw_root: Path, dest_root: Path, manifest_path: Path) -> int:
    """Copy Parquet files whose size or mtime is not in the manifest."""
    import json

    seen: dict[str, list] = {}
    if manifest_path.is_file():
        seen = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied = 0
    if raw_root.exists():
        for path in raw_root.rglob("*.parquet"):
            rel = path.relative_to(raw_root).as_posix()
            marker = [path.stat().st_size, path.stat().st_mtime_ns]
            if seen.get(rel) == marker:
                continue
            target = dest_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            seen[rel] = marker
            copied += 1
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(seen), encoding="utf-8")
    return copied


def latest_sqlite_backup(backup_dir: Path) -> Path | None:
    files = sorted(backup_dir.glob("superE-*.sqlite"))
    return files[-1] if files else None


def prune_old_backups(backup_dir: Path, keep_days: int, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC8)
    removed = 0
    for path in backup_dir.glob("superE-*.sqlite"):
        # name superE-YYYYMMDD.sqlite
        stamp = path.stem.split("-", 1)[-1]
        try:
            day = datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=UTC8)
        except ValueError:
            continue
        if now - day > timedelta(days=keep_days):
            path.unlink()
            removed += 1
    return removed


def restore_drill(backup: Path, work_dir: Path) -> dict:
    """Copy a backup aside and run integrity_check. Does not replace live data."""
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / "restore-drill.sqlite"
    if dest.exists():
        dest.unlink()
    shutil.copy2(backup, dest)
    conn = sqlite3.connect(dest)
    try:
        check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        klines = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM app_events").fetchone()[0]
    finally:
        conn.close()
    return {"integrity": check, "klines": int(klines), "events": int(events), "path": str(dest)}
