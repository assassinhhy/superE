"""Free-space check on the volume that holds the data directory."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DiskStatus:
    free_gb: float
    total_gb: float
    level: str  # ok, warn, critical


def check_disk(path: Path, warn_free_gb: float, critical_free_gb: float) -> DiskStatus:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    if free_gb < critical_free_gb:
        level = "critical"
    elif free_gb < warn_free_gb:
        level = "warn"
    else:
        level = "ok"
    return DiskStatus(free_gb=free_gb, total_gb=total_gb, level=level)
