"""Turn REST rows, archive CSV, and websocket payloads into storage rows."""

from __future__ import annotations

import json
from decimal import Decimal

from supere.storage.models import FundingRow, Kline, MetricRow
from supere.timeutil import parse_utc_datetime, to_millis


def dec_str(value: object) -> str:
    if value is None or value == "":
        return "0"
    return format(Decimal(str(value)), "f")


def _as_int(value: object) -> int:
    return int(value)


def kline_from_list(
    market: str,
    symbol: str,
    interval: str,
    row: list,
    source: str,
    ingested_at: int,
    ingested_local: int,
) -> Kline:
    return Kline(
        market=market,
        symbol=symbol.upper(),
        interval=interval,
        open_time=to_millis(_as_int(row[0])),
        close_time=to_millis(_as_int(row[6])),
        o=dec_str(row[1]),
        h=dec_str(row[2]),
        l=dec_str(row[3]),
        c=dec_str(row[4]),
        volume=dec_str(row[5]),
        quote_volume=dec_str(row[7]),
        trades=_as_int(row[8]),
        taker_buy_base=dec_str(row[9]),
        taker_buy_quote=dec_str(row[10]),
        source=source,
        ingested_at=ingested_at,
        ingested_local=ingested_local,
    )


def parse_kline_csv(
    text: str,
    market: str,
    symbol: str,
    interval: str,
    source: str,
    ingested_at: int,
    ingested_local: int,
) -> list[Kline]:
    rows: list[Kline] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if not parts[0].strip().lstrip("-").isdigit():
            continue
        rows.append(
            kline_from_list(
                market, symbol, interval, parts, source, ingested_at, ingested_local
            )
        )
    return rows


def parse_funding_csv(text: str, symbol: str, source: str = "archive") -> list[FundingRow]:
    rows: list[FundingRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if not parts[0].strip().isdigit():
            continue
        rate = parts[2] if len(parts) > 2 else parts[1]
        rows.append(
            FundingRow(
                symbol=symbol.upper(),
                funding_time=to_millis(int(parts[0])),
                rate=dec_str(rate),
                mark_price=None,
                source=source,
            )
        )
    return rows


def parse_metrics_csv(text: str, source: str = "archive") -> list[MetricRow]:
    rows: list[MetricRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("create_time"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        ts_raw = parts[0].strip()
        ts = parse_utc_datetime(ts_raw) if not ts_raw.isdigit() else to_millis(int(ts_raw))
        rows.append(
            MetricRow(
                symbol=parts[1].strip().upper(),
                ts=ts,
                oi=dec_str(parts[2]),
                oi_value=dec_str(parts[3]),
                top_acc_lsr=dec_str(parts[4]),
                top_pos_lsr=dec_str(parts[5]),
                global_lsr=dec_str(parts[6]),
                taker_lsr=dec_str(parts[7]),
                source=source,
            )
        )
    rows.sort(key=lambda row: row.ts)
    return rows


def funding_from_rest(item: dict, source: str = "rest") -> FundingRow:
    return FundingRow(
        symbol=str(item.get("symbol", "")).upper(),
        funding_time=to_millis(int(item["fundingTime"])),
        rate=dec_str(item.get("fundingRate")),
        mark_price=dec_str(item["markPrice"]) if item.get("markPrice") is not None else None,
        source=source,
    )


def metric_ts_from_rest(timestamp_ms: int, period: str = "5m") -> int:
    """REST ``timestamp`` is the bar end. Archive ``create_time`` is the bar start.

    CC review matched openInterestHist to the metrics file with a 5 minute shift.
    The same shift is applied to the other 5-minute ratio endpoints.
    """
    shift = {
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
    }.get(period, 5 * 60_000)
    return int(timestamp_ms) - shift


def _ratio(item: dict) -> str | None:
    if "longShortRatio" in item:
        return dec_str(item["longShortRatio"])
    if "buySellRatio" in item:
        return dec_str(item["buySellRatio"])
    return None


def metrics_from_oi_hist(items: list[dict], period: str = "5m") -> list[MetricRow]:
    rows = []
    for item in items:
        rows.append(
            MetricRow(
                symbol=str(item["symbol"]).upper(),
                ts=metric_ts_from_rest(int(item["timestamp"]), period),
                oi=dec_str(item.get("sumOpenInterest")),
                oi_value=dec_str(item.get("sumOpenInterestValue")),
                source="rest",
            )
        )
    return rows


def metrics_from_ratio(items: list[dict], field: str, period: str = "5m") -> list[MetricRow]:
    rows = []
    for item in items:
        value = _ratio(item)
        kwargs = {
            "symbol": str(item.get("symbol", "ETHUSDT")).upper(),
            "ts": metric_ts_from_rest(int(item["timestamp"]), period),
            "source": "rest",
            field: value,
        }
        rows.append(MetricRow(**kwargs))
    return rows


def unwrap_ws(payload: dict) -> dict:
    if "data" in payload and isinstance(payload["data"], dict):
        return payload["data"]
    return payload


def parse_ws_message(raw: str | bytes) -> dict | None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    if "result" in payload and "id" in payload and "e" not in payload and "data" not in payload:
        return None
    if "error" in payload and "e" not in payload:
        return {"kind": "error", "error": payload.get("error"), "raw": payload}
    data = unwrap_ws(payload)
    event = data.get("e")
    if event == "kline":
        k = data["k"]
        return {
            "kind": "kline",
            "symbol": str(data.get("s") or k.get("s") or "").upper(),
            "interval": k["i"],
            "event_ms": int(data.get("E") or 0),
            "closed": bool(k.get("x")),
            "row": [
                k["t"],
                k["o"],
                k["h"],
                k["l"],
                k["c"],
                k["v"],
                k["T"],
                k["q"],
                k["n"],
                k["V"],
                k["Q"],
                k.get("B", "0"),
            ],
            "dedupe_key": ("kline", k["i"], int(k["t"]), int(data.get("E") or 0)),
        }
    if event == "aggTrade":
        return {
            "kind": "aggTrade",
            "symbol": str(data["s"]).upper(),
            "agg_id": int(data["a"]),
            "price": dec_str(data["p"]),
            "qty": dec_str(data["q"]),
            "first_id": int(data["f"]),
            "last_id": int(data["l"]),
            "ts": int(data["T"]),
            "event_ms": int(data.get("E") or data["T"]),
            "is_buyer_maker": bool(data.get("m")),
            "dedupe_key": ("aggTrade", int(data["a"])),
        }
    if event == "depthUpdate" or "bids" in data or "b" in data and "a" in data and "e" not in data:
        bids = data.get("b") or data.get("bids") or []
        asks = data.get("a") or data.get("asks") or []
        update_id = int(data.get("u") or data.get("lastUpdateId") or 0)
        return {
            "kind": "depth",
            "symbol": str(data.get("s") or "").upper(),
            "event_ms": int(data.get("E") or 0),
            "transact_ms": int(data.get("T") or 0),
            "update_id": update_id,
            "bids_px": [dec_str(level[0]) for level in bids],
            "bids_qty": [dec_str(level[1]) for level in bids],
            "asks_px": [dec_str(level[0]) for level in asks],
            "asks_qty": [dec_str(level[1]) for level in asks],
            "dedupe_key": ("depth", update_id),
        }
    if event == "forceOrder":
        order = data.get("o") or {}
        return {
            "kind": "forceOrder",
            "symbol": str(order.get("s") or data.get("s") or "").upper(),
            "event_ms": int(data.get("E") or 0),
            "side": order.get("S"),
            "order_type": order.get("o"),
            "tif": order.get("f"),
            "qty": dec_str(order.get("q")),
            "price": dec_str(order.get("p")),
            "avg_price": dec_str(order.get("ap")),
            "last_qty": dec_str(order.get("l")),
            "accum_qty": dec_str(order.get("z")),
            "status": order.get("X"),
            "trade_time": int(order.get("T") or 0),
            "dedupe_key": (
                "forceOrder",
                int(data.get("E") or 0),
                order.get("S"),
                order.get("p"),
                order.get("q"),
                int(order.get("T") or 0),
            ),
        }
    if event == "markPriceUpdate":
        return {
            "kind": "markPrice",
            "symbol": str(data["s"]).upper(),
            "event_ms": int(data.get("E") or 0),
            "mark_price": dec_str(data.get("p")),
            "index_price": dec_str(data.get("i")),
            "rate": dec_str(data.get("r")) if data.get("r") is not None else None,
            "next_funding_time": int(data["T"]) if data.get("T") is not None else None,
            "dedupe_key": ("markPrice", str(data["s"]).upper(), int(data.get("E") or 0)),
        }
    if event in {"24hrMiniTicker", "24hrTicker"}:
        return {
            "kind": "miniTicker",
            "symbol": str(data["s"]).upper(),
            "event_ms": int(data.get("E") or 0),
            "open": dec_str(data.get("o")),
            "high": dec_str(data.get("h")),
            "low": dec_str(data.get("l")),
            "close": dec_str(data.get("c")),
            "volume": dec_str(data.get("v")),
            "quote_volume": dec_str(data.get("q")),
            "dedupe_key": ("miniTicker", str(data["s"]).upper(), int(data.get("E") or 0)),
        }
    return {"kind": "ignore", "event": event, "dedupe_key": ("ignore", event)}


def depth_gap_key(update_id: int) -> tuple:
    return ("depth", update_id)
