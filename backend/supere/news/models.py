"""Normalized item produced by every source adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RawItem:
    source: str
    source_tier: int
    lang: str
    title: str
    body: str
    url: str
    ts_published_ms: int | None = None
    external_id: str | None = None


@dataclass(slots=True)
class FearGreedPoint:
    ts_ms: int
    value: int
    classification: str
    known_at_ms: int
