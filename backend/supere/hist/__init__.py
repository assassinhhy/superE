"""Historical market-data archives from data.binance.vision.

The live collector writes ``data/raw/<dataset>/um/ETHUSDT/date=.../``.  The
backfill writes the same partition shape under ``<dataset>_hist`` so the two
never overwrite each other.  Values that the live schema keeps as decimal
strings (price, quantity, depth, notional, metrics) stay strings here too.
"""

from __future__ import annotations

__all__ = ["schema", "archives", "convert", "manifest", "backfill"]
