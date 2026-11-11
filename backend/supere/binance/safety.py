"""Refuse anything that could place or manage an order.

superE reads public market data only. There is no Binance API key.
"""

from __future__ import annotations

from supere.errors import TradingDisabled

# Matched against the lowercased URL path (and query).
FORBIDDEN_PATH_MARKERS: tuple[str, ...] = (
    "/fapi/v1/order",
    "/fapi/v1/batchorders",
    "/fapi/v1/allopenorders",
    "/fapi/v2/account",
    "/fapi/v1/leverage",
    "/fapi/v1/margintype",
    "/fapi/v1/positionmargin",
    "/fapi/v1/listenkey",
    "/fapi/v1/allforceorders",
    "/api/v3/order",
    "/api/v3/openorders",
    "/api/v3/account",
    "/sapi/",
    "api_key=",
    "signature=",
)


def guard_url(url: str) -> None:
    lowered = url.lower()
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker in lowered:
            raise TradingDisabled(f"拒绝交易或账户接口: {url}")
