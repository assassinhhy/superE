from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from supere.hist.archives import DownloadResult
from supere.hist.backfill import run_stage
from supere.hist.schema import partition_dir


class _FakeClient:
    def __init__(self, source: Path, prefix: str, filename: str) -> None:
        self.source = source
        self.prefix = prefix
        self.filename = filename

    def list_archives(self, prefix: str) -> dict[str, int]:
        if prefix == self.prefix:
            return {self.filename: self.source.stat().st_size}
        return {}

    def download(self, item, dest_dir: Path):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item.filename
        shutil.copyfile(self.source, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        return DownloadResult(item, dest, digest, dest.stat().st_size)


def _book_zip(tmp_path: Path) -> Path:
    text = ["timestamp,percentage,depth,notional"]
    for pct in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5):
        text.append(f"2023-01-01 00:00:05,{pct},{100 + pct}.5,{200 + pct}.25")
    csv = tmp_path / "ETHUSDT-bookDepth-2023-01-01.csv"
    csv.write_text("\n".join(text) + "\n", encoding="utf-8")
    archive = tmp_path / "ETHUSDT-bookDepth-2023-01-01.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(csv, arcname=csv.name)
    return archive


def test_run_stage_converts_then_resumes_without_redownload(tmp_path: Path):
    source = _book_zip(tmp_path)
    client = _FakeClient(
        source,
        "data/futures/um/daily/bookDepth/ETHUSDT/",
        "ETHUSDT-bookDepth-2023-01-01.zip",
    )
    raw = tmp_path / "raw"
    first = run_stage(
        "bookdepth", client=client, raw_root=raw, symbol="ETHUSDT", workers=1, end=None
    )
    assert first.planned == 1
    assert first.converted == 1
    assert first.failed == 0
    part = partition_dir(raw, "bookDepth", "2023-01-01") / "part-000.parquet"
    assert part.exists()

    second = run_stage(
        "bookdepth", client=client, raw_root=raw, symbol="ETHUSDT", workers=1, end=None
    )
    assert second.converted == 0
    assert second.skipped == 1
