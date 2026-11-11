"""Overlap two websocket connections so a planned reconnect does not open a hole.

Messages are deduped by depth ``u``, aggTrade ``a``, and kline ``(interval, open_time, E)``.
The new connection is promoted after it has delivered ``confirm`` data frames,
including frames that were duplicates of the old connection.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict


class OverlapGate:
    def __init__(self, confirm: int, maxlen: int = 100_000) -> None:
        self.confirm = confirm
        self.maxlen = maxlen
        self.pending: str | None = None
        self.pending_hits = 0
        self.promoted = False
        self.active = "old"
        self._seen: dict[str, OrderedDict] = defaultdict(OrderedDict)

    def start(self, conn_id: str) -> None:
        self.pending = conn_id
        self.pending_hits = 0
        self.promoted = False

    def observe(self, conn_id: str, kind: str, key: tuple) -> bool:
        """Return True when the caller should apply this message."""
        if self.pending is not None and conn_id == self.pending:
            self.pending_hits += 1
            if self.pending_hits >= self.confirm:
                self.promoted = True
                self.active = conn_id
                self.pending = None
        store = self._seen[kind]
        if key in store:
            return False
        store[key] = None
        while len(store) > self.maxlen:
            store.popitem(last=False)
        return True
