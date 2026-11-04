"""Daily log files named in UTC+8, with credential-like text masked."""

from __future__ import annotations

import logging
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
    """One file per UTC+8 date: ``supere-YYYY-MM-DD.log``."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._date: str | None = None
        self._stream = None

    def _open_for(self, date: str):
        if self._stream is not None and self._date == date:
            return self._stream
        if self._stream is not None:
            self._stream.close()
        path = self.directory / f"supere-{date}.log"
        self._stream = path.open("a", encoding="utf-8")
        self._date = date
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            date = datetime.fromtimestamp(record.created, tz=UTC8).strftime("%Y-%m-%d")
            stream = self._open_for(date)
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
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


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    configure_stdio()
    logger = logging.getLogger("supere")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = _Utc8Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    mask = _MaskFilter()

    file_handler = _DailyUtc8Handler(log_dir)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(mask)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(mask)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
