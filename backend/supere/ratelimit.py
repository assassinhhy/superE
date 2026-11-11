"""Per-process request weights.

The exchange limit is shared with whoever else uses the same proxy exit, so
the header ``X-MBX-USED-WEIGHT-1M`` is a brake, not a private counter. This
limiter spends only its own budget and slows further when the header crosses
60 percent. The hard cap is 49 percent of the exchange limit so a full minute
stays strictly under the acceptance threshold of 50 percent.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from supere.errors import RestHalted


def kline_request_weight(limit: int) -> int:
    """Weight of ``/fapi/v1/klines`` by the ``limit`` argument.

    Tiers from the 2021-01-04 futures API announcement: <100 = 1, <500 = 2,
    <=1000 = 5, >1000 = 10. ``limit=499`` is the cheap page size.
    """
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class WeightLimiter:
    def __init__(
        self,
        name: str,
        limit_per_window: int,
        *,
        budget_ratio: float = 0.5,
        header_slowdown_ratio: float = 0.6,
        window_s: float = 60.0,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self.limit = int(limit_per_window)
        self.budget_ratio = budget_ratio
        self.header_slowdown_ratio = header_slowdown_ratio
        self.window_s = window_s
        self._sleep = sleeper or asyncio.sleep
        self._now = clock or time.monotonic
        self._events: deque[tuple[float, int]] = deque()
        self._window_total = 0
        self.peak = 0
        self.header_used: int | None = None
        self.slowdown = False
        self.halted = False
        self._429_attempt = 0
        self._lock = asyncio.Lock()

    def current_cap(self) -> int:
        ratio = min(self.budget_ratio, 0.49)
        cap = int(self.limit * ratio)
        if self.slowdown:
            cap = max(1, cap // 2)
        return max(1, cap)

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window_s:
            _, weight = self._events.popleft()
            self._window_total -= weight

    async def acquire(self, weight: int) -> None:
        if weight <= 0:
            return
        async with self._lock:
            while True:
                if self.halted:
                    raise RestHalted(f"{self.name} REST 已因 HTTP 418 停止")
                now = self._now()
                self._prune(now)
                cap = self.current_cap()
                absolute_cap = int(self.limit * min(self.budget_ratio, 0.49))
                if weight > max(1, absolute_cap):
                    raise ValueError(
                        f"{self.name} 单次权重 {weight} 超过本进程预算上限 {absolute_cap}"
                    )
                if weight <= cap and self._window_total + weight <= cap:
                    self._events.append((now, weight))
                    self._window_total += weight
                    self.peak = max(self.peak, self._window_total)
                    return
                if self._events:
                    wait = self.window_s - (now - self._events[0][0]) + 0.01
                else:
                    wait = 0.2
                if self.slowdown:
                    wait = max(wait, 0.5)
                await self._sleep(max(0.01, min(wait, self.window_s)))

    def observe_header(self, used: int | None) -> None:
        if used is None:
            return
        self.header_used = int(used)
        self.slowdown = self.header_used > self.limit * self.header_slowdown_ratio

    def note_429(self) -> float:
        delay = min(60.0, float(2**self._429_attempt))
        self._429_attempt += 1
        return delay

    def note_success(self) -> None:
        self._429_attempt = 0

    def halt(self) -> None:
        self.halted = True

    def snapshot(self) -> tuple[int, int, int | None, bool]:
        self._prune(self._now())
        return self._window_total, self.peak, self.header_used, self.halted

    @property
    def window_total(self) -> int:
        return self._window_total
