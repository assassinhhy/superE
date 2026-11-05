"""Append-only JSONL manifest for archive downloads and conversions.

One line per event.  A crash between a parquet write and the manifest append is
safe: the next run converts the still-present zip again and overwrites the
partition atomically.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveRecord:
    kind: str
    period: str
    stamp: str
    filename: str
    url: str
    sha256: str | None = None
    bytes: int | None = None
    rows: int | None = None
    date_counts: dict[str, int] | None = None
    parquet: list[str] | None = None
    status: str = "downloaded"  # downloaded | converted | missing | failed
    error: str | None = None
    at: str | None = None

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ArchiveRecord) -> None:
        line = record.to_json()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn last line is possible after a hard kill.
                    continue
        return out

    def converted(self) -> dict[str, dict[str, Any]]:
        """Last *successful* record per ``filename``.

        A later failed retry must not hide an earlier conversion; the Parquet
        partitions from that successful run are still on disk.
        """
        out: dict[str, dict[str, Any]] = {}
        for record in self.records():
            name = record.get("filename")
            if name and record.get("status") == "converted":
                out[name] = record
        return out

    def summaries(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.records():
            name = record.get("filename")
            if name:
                latest[name] = record
        return latest
