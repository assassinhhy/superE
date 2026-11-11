"""Daily log files named in UTC+8, with credential-like text masked."""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from supere.timeutil import UTC8

_SECRET = re.compile(
    r"(?i)(api[_-]?key\s*[=:]\s*\S+"
    r"|secret\s*[=:]\s*\S+"
    r"|authorization\s*[=:]\s*\S+"
    r"|bearer\s+\S+"
    r"|sk-[A-Za-z0-9_\-]{8,})"
)


def mask_secrets(text: str) -> str:
    return _SECRET.sub("[redacted]", text)


class _MaskFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_secrets(record.msg)
        if record.args:
            record.args = tuple(
                mask_secrets(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        return True


class _Utc8Formatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC8)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class _DailyUtc8Handler(logging.Handler):
    """One UTF-8 file per process per UTC+8 date.

    The ingestor and the API used to append to the same file. Two processes
    flushing a text buffer can split a 3-byte character, which shows up as
    ``已连�`` or ``请检�``. Each line is one ``os.write`` of UTF-8 bytes, and
    each process has its own file: ``supere-<component>-YYYY-MM-DD.log``.
    """

    def __init__(self, directory: Path, component: str) -> None:
        super().__init__()
        self.directory = directory
        self.component = component
        self.directory.mkdir(parents=True, exist_ok=True)
        self._date: str | None = None
        self._fd: int | None = None

    def _open_for(self, date: str) -> int:
        if self._fd is not None and self._date == date:
            return self._fd
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        path = self.directory / f"supere-{self.component}-{date}.log"
        self._fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        self._date = date
        return self._fd

    def emit(self, record: logging.LogRecord) -> None:
        try:
            date = datetime.fromtimestamp(record.created, tz=UTC8).strftime("%Y-%m-%d")
            fd = self._open_for(date)
            os.write(fd, (self.format(record) + "\n").encode("utf-8"))
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        if self._fd is None:
            return
        try:
            os.fsync(self._fd)
        except OSError:
            pass

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        super().close()


def configure_stdio() -> None:
    """Force UTF-8 on stdout and stderr.

    A Windows pipe or console often uses GBK. Chinese log lines then lose the
    last byte of a character (``已连接`` becomes ``已连�``) when something
    decodes those bytes again. The log file is already opened as UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def flush_logs() -> None:
    logger = logging.getLogger("supere")
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            continue


def setup_logging(log_dir: Path, component: str, level: int = logging.INFO) -> logging.Logger:
    configure_stdio()
    logger = logging.getLogger("supere")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = _Utc8Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    mask = _MaskFilter()

    file_handler = _DailyUtc8Handler(log_dir, component)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(mask)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(mask)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
