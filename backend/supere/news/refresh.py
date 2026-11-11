"""Hourly news refresh, manual refresh, and the one-hour prediction cache.

The schedule follows Asia/Singapore (UTC+8, no daylight saving). A refresh
runs once when the service starts, then at each hour boundary. Manual refresh
waits 60 seconds between accepted calls and never overlaps another refresh.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from supere.news import live
from supere.timeutil import format_utc8

log = logging.getLogger("supere.news.refresh")

try:
    SGT = ZoneInfo("Asia/Singapore")
except Exception:  # pragma: no cover - tzdata is present on the runtime image
    SGT = ZoneInfo("Etc/GMT-8")

STALE_MS = 3_600_000
COOLDOWN_MS = 60_000


def hour_floor_ms(now_ms: int) -> int:
    dt = datetime.fromtimestamp(int(now_ms) / 1000, tz=SGT)
    floored = dt.replace(minute=0, second=0, microsecond=0)
    return int(floored.timestamp() * 1000)


def next_hour_ms(now_ms: int) -> int:
    dt = datetime.fromtimestamp(int(now_ms) / 1000, tz=SGT)
    floored = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return int(floored.timestamp() * 1000)


def seconds_until_next_hour(now_ms: int) -> float:
    return max(0.0, (next_hour_ms(now_ms) - int(now_ms)) / 1000.0)


class HourlyPlan:
    """Decides when a refresh is due. Startup always fetches; later ticks fire once per hour."""

    def __init__(self) -> None:
        self.last_slot: int | None = None

    def on_start(self, now_ms: int) -> bool:
        self.last_slot = hour_floor_ms(now_ms)
        return True

    def on_tick(self, now_ms: int) -> bool:
        slot = hour_floor_ms(now_ms)
        if self.last_slot is None or slot > self.last_slot:
            self.last_slot = slot
            return True
        return False


def run_schedule(stop_event, now_ms_fn: Callable[[], int], plan: HourlyPlan,
                 refresh_fn: Callable[[int], None], wait: Callable[[float], bool]) -> None:
    """Fetch immediately, then once each time the Singapore hour changes."""
    now = int(now_ms_fn())
    plan.on_start(now)
    refresh_fn(now)
    while not stop_event.is_set():
        delay = seconds_until_next_hour(int(now_ms_fn())) + 0.05
        if wait(delay):
            return
        now = int(now_ms_fn())
        if plan.on_tick(now):
            refresh_fn(now)


class NewsRefresher:
    def __init__(self, db_path: Path, *, fetcher: Callable[[int], dict],
                 scorer: Callable[[dict], dict], now_ms: Callable[[], int],
                 digest_fn: Callable[[list[dict], int], dict | None] | None = None,
                 cooldown_ms: int = COOLDOWN_MS) -> None:
        self._db_path = Path(db_path)
        self._fetcher = fetcher
        self._scorer = scorer
        self._now_ms = now_ms
        self._digest_fn = digest_fn
        self._cooldown_ms = int(cooldown_ms)
        self._lock = threading.Lock()
        self._refreshing = False
        self._last_refresh_at: int | None = None
        self._manual_at: int | None = None
        self._items: list[dict] = []
        self._sources: dict = {}
        self._status = "empty"
        self._score_counts: dict = {}
        self._digest: dict | None = None
        self._plan = HourlyPlan()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def sources(self) -> dict:
        return dict(self._sources)

    @property
    def digest(self) -> dict | None:
        return dict(self._digest) if self._digest else None

    @property
    def last_refresh_at(self) -> int | None:
        return self._last_refresh_at

    def _cooldown_remaining(self, now_ms: int) -> int:
        if self._manual_at is None:
            return 0
        left = self._cooldown_ms - (int(now_ms) - int(self._manual_at))
        if left <= 0:
            return 0
        return int(math.ceil(left / 1000))

    def _is_fresh(self, now_ms: int) -> bool:
        if self._last_refresh_at is None:
            return False
        return int(now_ms) - int(self._last_refresh_at) <= STALE_MS

    def _cached(self, now_ms: int) -> dict:
        return {
            "items": [dict(item) for item in self._items],
            "sources": dict(self._sources),
            "status": self._status,
            "fetched_at": self._last_refresh_at or int(now_ms),
            "scored": True,
            "from_cache": True,
            "score_counts": dict(self._score_counts),
        }

    def public_state(self, now_ms: int) -> dict:
        nxt = next_hour_ms(now_ms)
        last = self._last_refresh_at
        return {
            "last_refresh_at": last,
            "last_refresh_at_utc8": format_utc8(last) if last else None,
            "next_refresh_at": nxt,
            "next_refresh_at_utc8": format_utc8(nxt),
            "refreshing": bool(self._refreshing),
            "cooldown_remaining_s": self._cooldown_remaining(now_ms),
            "sources": dict(self._sources),
            "score_counts": dict(self._score_counts),
            "digest": self.digest,
        }

    def _busy(self, now_ms: int) -> dict:
        return {
            "accepted": False,
            "refreshing": True,
            "cooling": False,
            "cooldown_remaining_s": self._cooldown_remaining(now_ms),
            "snapshot": self._cached(now_ms),
        }

    def _cooling(self, now_ms: int, remain: int) -> dict:
        return {
            "accepted": False,
            "refreshing": False,
            "cooling": True,
            "cooldown_remaining_s": int(remain),
            "snapshot": self._cached(now_ms),
        }

    def _run(self, now_ms: int) -> dict:
        self._refreshing = True
        try:
            try:
                snap = self._fetcher(int(now_ms)) or {}
            except Exception as exc:
                log.warning("news fetch failed error=%s", type(exc).__name__)
                self._status = "error"
                self._sources = {}
                failed = {
                    "items": [],
                    "sources": {},
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "fetched_at": int(now_ms),
                    "scored": True,
                    "score_counts": {},
                }
                return {
                    "accepted": True,
                    "refreshing": False,
                    "cooling": False,
                    "cooldown_remaining_s": self._cooldown_remaining(now_ms),
                    "snapshot": failed,
                }
            items = list(snap.get("items") or [])
            snap["items"] = items
            try:
                live.persist(items, self._db_path, int(now_ms))
            except Exception:
                log.warning("news persist failed")
            try:
                scored = self._scorer(snap) or snap
            except Exception:
                log.warning("news score failed")
                scored = snap
            self._items = list(scored.get("items") or [])
            self._sources = dict(scored.get("sources") or {})
            self._status = scored.get("status") or ("ok" if self._items else "empty")
            self._score_counts = dict(scored.get("score_counts") or {})
            self._last_refresh_at = int(scored.get("fetched_at") or now_ms)
            scored["scored"] = True
            scored["fetched_at"] = self._last_refresh_at
            scored["from_cache"] = False
            if self._digest_fn and self._items:
                try:
                    digest = self._digest_fn(self._items, int(now_ms))
                    if digest:
                        self._digest = digest
                except Exception:
                    log.warning("news digest failed")
            log.info("news refresh status=%s items=%s", self._status, len(self._items))
            return {
                "accepted": True,
                "refreshing": False,
                "cooling": False,
                "cooldown_remaining_s": self._cooldown_remaining(now_ms),
                "snapshot": scored,
            }
        finally:
            self._refreshing = False

    def refresh(self, now_ms: int, *, manual: bool) -> dict:
        if manual:
            if not self._lock.acquire(blocking=False):
                return self._busy(now_ms)
            try:
                remain = self._cooldown_remaining(now_ms)
                if remain:
                    return self._cooling(now_ms, remain)
                self._manual_at = int(now_ms)
                return self._run(now_ms)
            finally:
                self._lock.release()
        self._lock.acquire()
        try:
            return self._run(now_ms)
        finally:
            self._lock.release()

    def for_predict(self, now_ms: int) -> dict:
        """Use the cache when it is at most one hour old; otherwise fetch once."""
        if self._is_fresh(now_ms):
            return self._cached(now_ms)
        self._lock.acquire()
        try:
            if self._is_fresh(now_ms):
                return self._cached(now_ms)
            result = self._run(now_ms)
            return result.get("snapshot") or self._cached(now_ms)
        finally:
            self._lock.release()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def _refresh(now: int) -> None:
            self.refresh(now, manual=False)

        self._thread = threading.Thread(
            target=run_schedule,
            args=(self._stop, self._now_ms, self._plan, _refresh, self._stop.wait),
            name="supere-news-refresh",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
