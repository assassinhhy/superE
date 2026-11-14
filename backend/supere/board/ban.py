"""Binance 429 / -1003 / 418 backoff. The ban survives a process restart.

During the wait this source is not requested. The clock time is written to
``data/run/binance_ban.json`` so a restart does not hit the wall again.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

_BANNED_UNTIL = re.compile(r"banned until\s+(\d+)", re.IGNORECASE)


@dataclass
class Penalty:
    kind: str
    until_ms: int
    wait_seconds: float
    reason: str
    source_url: str
    strikes: int

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "until_ms": self.until_ms,
            "wait_seconds": self.wait_seconds,
            "reason": self.reason,
            "source_url": self.source_url,
            "strikes": self.strikes,
        }


def _code_and_msg(body) -> tuple[object, str]:
    if isinstance(body, dict):
        return body.get("code"), str(body.get("msg") or "")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None, body
        if isinstance(parsed, dict):
            return parsed.get("code"), str(parsed.get("msg") or "")
        return None, body
    return None, ""


def _is_1003(code) -> bool:
    try:
        return int(code) == -1003
    except (TypeError, ValueError):
        return False


def _retry_after_seconds(headers, now_ms: int) -> float | None:
    if headers is None:
        return None
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("Retry-After")
        if raw is None:
            raw = headers.get("retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, moment.timestamp() * 1000 - now_ms) / 1000


def _banned_until_ms(msg: str) -> int | None:
    match = _BANNED_UNTIL.search(msg or "")
    if not match:
        return None
    value = int(match.group(1))
    if value < 10_000_000_000:
        value *= 1000
    return value


def parse_penalty(
    *,
    status: int,
    headers,
    body,
    now_ms: int,
    strikes: int,
    source_url: str,
    first_seconds: int = 60,
    max_seconds: int = 900,
    ban_extra_seconds: int = 30,
    ban_parse_fail_seconds: int = 600,
) -> Penalty | None:
    """Return a penalty, or None when the response is not a rate limit."""
    code, msg = _code_and_msg(body)
    banned_ms = _banned_until_ms(msg)
    limited = status in {418, 429} or _is_1003(code) or banned_ms is not None
    if not limited:
        return None

    if status == 418 or banned_ms is not None:
        if banned_ms is not None:
            until = banned_ms + ban_extra_seconds * 1000
            reason = f"banned until {banned_ms}"
        else:
            until = now_ms + ban_parse_fail_seconds * 1000
            reason = "HTTP 418，未能解析 banned until"
        return Penalty(
            kind="ban",
            until_ms=int(until),
            wait_seconds=max(0.0, (until - now_ms) / 1000),
            reason=reason,
            source_url=source_url,
            strikes=strikes,
        )

    retry = _retry_after_seconds(headers, now_ms)
    if retry is not None:
        wait = retry
        reason = f"HTTP {status} Retry-After {wait:.0f}s"
        next_strikes = strikes
    else:
        wait = float(min(first_seconds * (2 ** strikes), max_seconds))
        reason = f"HTTP {status}" + (" code -1003" if _is_1003(code) else "") + f"，退避 {wait:.0f}s"
        next_strikes = strikes + 1
    return Penalty(
        kind="backoff",
        until_ms=int(now_ms + wait * 1000),
        wait_seconds=wait,
        reason=reason,
        source_url=source_url,
        strikes=next_strikes,
    )


class BanStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.kind = ""
        self.until_ms = 0
        self.reason = ""
        self.source_url = ""
        self.strikes = 0
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.kind = str(data.get("kind") or "")
        self.until_ms = int(data.get("until_ms") or 0)
        self.reason = str(data.get("reason") or "")
        self.source_url = str(data.get("source_url") or "")
        self.strikes = int(data.get("strikes") or 0)

    def active(self, now_ms: int) -> bool:
        return self.until_ms > int(now_ms)

    def record(self, penalty: Penalty) -> None:
        self.kind = penalty.kind
        self.until_ms = penalty.until_ms
        self.reason = penalty.reason
        self.source_url = penalty.source_url
        self.strikes = penalty.strikes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(penalty.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def note_success(self, now_ms: int) -> None:
        self.strikes = 0
        if self.until_ms <= int(now_ms):
            self.until_ms = 0
            self.kind = ""
            self.reason = ""
        if self.path.is_file() and not self.active(now_ms):
            try:
                self.path.unlink()
            except OSError:
                pass
