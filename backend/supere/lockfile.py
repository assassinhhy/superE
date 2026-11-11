"""One ingestor at a time, using an OS file lock (msvcrt on Windows, fcntl elsewhere)."""

from __future__ import annotations

import os
from pathlib import Path

import portalocker


class SingleInstance:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException as exc:
            handle.close()
            raise SystemExit(
                "另一个 superE 采集器已在运行（文件锁被占用），本实例退出。"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            portalocker.unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None
