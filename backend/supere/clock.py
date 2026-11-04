"""Offset against Binance ``GET /fapi/v1/time``.

``offset = serverTime - (t_send + t_recv) / 2`` using the sample with the
shortest round trip. Later local timestamps that mean "when we received it"
are stored as local time plus this offset, and the raw local time is kept
beside them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeSample:
    t_send_ms: int
    t_recv_ms: int
    server_ms: int

    @property
    def rtt_ms(self) -> int:
        return self.t_recv_ms - self.t_send_ms

    @property
    def offset_ms(self) -> int:
        midpoint = (self.t_send_ms + self.t_recv_ms) // 2
        return self.server_ms - midpoint


def choose_offset(samples: list[TimeSample]) -> TimeSample:
    if not samples:
        raise ValueError("时钟校准需要至少一次采样")
    return min(samples, key=lambda sample: (sample.rtt_ms, abs(sample.offset_ms)))


@dataclass(slots=True)
class ClockState:
    offset_ms: int = 0
    rtt_ms: int | None = None
    calibrated: bool = False
    warn: bool = False

    def server_ms(self, local_ms: int) -> int:
        return local_ms + self.offset_ms

    def apply(self, sample: TimeSample, *, warn_abs_ms: int, change_alert_ms: int) -> str:
        """Apply a sample. Returns ``ok``, ``changed``, or ``skew``."""
        previous = self.offset_ms if self.calibrated else None
        self.offset_ms = sample.offset_ms
        self.rtt_ms = sample.rtt_ms
        self.calibrated = True
        self.warn = abs(self.offset_ms) > warn_abs_ms
        if previous is not None and abs(self.offset_ms - previous) > change_alert_ms:
            return "changed"
        if self.warn:
            return "skew"
        return "ok"
