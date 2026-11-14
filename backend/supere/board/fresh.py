"""Freshness stamps. A stale value keeps its original timestamp and is never marked ok."""

from __future__ import annotations

from supere.timeutil import format_utc8


def judge(value, data_ts: int | None, fetched_at: int | None, source: str, now_ms: int, max_age_ms: int) -> dict:
    if value is None or data_ts is None:
        return {
            "value": None,
            "data_ts": data_ts,
            "fetched_at": fetched_at,
            "source": source,
            "status": "missing",
            "utc8": None,
            "label": "未获取",
        }
    age = int(now_ms) - int(data_ts)
    status = "ok" if age <= int(max_age_ms) else "stale"
    clock = format_utc8(int(data_ts), "%H:%M:%S")
    if status == "stale":
        label = f"上次数据 {clock}（已过期）"
    else:
        label = clock
    return {
        "value": value,
        "data_ts": int(data_ts),
        "fetched_at": None if fetched_at is None else int(fetched_at),
        "source": source,
        "status": status,
        "utc8": format_utc8(int(data_ts)),
        "label": label,
    }


def kline_status(latest_close_time: int | None, now_ms: int, interval_ms: int, grace_ms: int) -> str:
    if latest_close_time is None:
        return "missing"
    if int(now_ms) - int(latest_close_time) > int(interval_ms) + int(grace_ms):
        return "stale"
    return "ok"


def crosscheck(binance_price, okx_price, max_diff_pct: float, *, binance_ok: bool, okx_ok: bool) -> dict:
    """Compare two fresh prices. A missing OKX price does not block the panel."""
    if not okx_ok or okx_price is None or not binance_ok or binance_price is None or float(binance_price) <= 0:
        return {
            "status": "missing",
            "binance": binance_price,
            "okx": okx_price,
            "diff_pct": None,
            "message": "交叉核对未获取",
            "pause_plans": False,
        }
    diff = abs(float(binance_price) - float(okx_price)) / float(binance_price) * 100
    if diff > float(max_diff_pct):
        return {
            "status": "anomaly",
            "binance": float(binance_price),
            "okx": float(okx_price),
            "diff_pct": diff,
            "message": f"数据异常：币安 {binance_price} / OKX {okx_price}",
            "pause_plans": True,
        }
    return {
        "status": "ok",
        "binance": float(binance_price),
        "okx": float(okx_price),
        "diff_pct": diff,
        "message": "",
        "pause_plans": False,
    }
