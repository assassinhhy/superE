"""Assemble a freshness-tagged snapshot. Stale rows keep their exchange timestamp."""

from __future__ import annotations

from supere.board.cache import BoardCache
from supere.board.fresh import crosscheck, judge, kline_status
from supere.board.reader import BOARD_INTERVALS, METRIC_MS, MainReader
from supere.board.settings import AnalysisConfig
from supere.intervals import INTERVAL_MS
from supere.timeutil import format_utc8

DAY_MS = 86_400_000


def _metric_pack(rows: list[dict], source: str, fetched_at: int | None, now_ms: int, max_age_ms: int):
    if not rows:
        return None
    known = int(rows[-1]["known_at"])
    status = "ok" if now_ms - known <= max_age_ms else "stale"
    return {"rows": rows, "source": source, "known_at": known, "fetched_at": fetched_at, "status": status}


def choose_metrics(main_rows: list[dict], cache_rows: list[dict], now_ms: int, max_age_ms: int, cache_fetched_at: int | None):
    """Main database wins while it is fresh. A stale series is never relabeled ok."""
    main = _metric_pack(main_rows, "主库 um_metrics", None, now_ms, max_age_ms)
    cached = _metric_pack(cache_rows, "币安 /futures/data", cache_fetched_at, now_ms, max_age_ms)
    if main and main["status"] == "ok":
        return main
    if cached and cached["status"] == "ok":
        return cached
    options = [item for item in (main, cached) if item]
    if not options:
        return {"rows": [], "source": "", "known_at": None, "fetched_at": None, "status": "missing"}
    options.sort(key=lambda item: item["known_at"], reverse=True)
    return options[0]


def ticker_from_bars(bars: list[dict], now_ms: int) -> dict | None:
    window = [bar for bar in bars if bar["close_time"] >= int(now_ms) - DAY_MS]
    use = window or bars[-96:]
    if not use:
        return None
    open_ = use[0]["open"]
    last = use[-1]["close"]
    change = None if not open_ else (last - open_) / open_ * 100
    return {
        "open": open_,
        "high": max(bar["high"] for bar in use),
        "low": min(bar["low"] for bar in use),
        "last": last,
        "change_pct": change,
        "data_ts": use[-1]["close_time"],
    }


def _closed(bars: list[dict], now_ms: int) -> list[dict]:
    return [bar for bar in bars if bar["close_time"] <= int(now_ms)]


def _merge_bars(primary: list[dict], extra: list[dict], limit: int) -> list[dict]:
    by_open = {bar["open_time"]: bar for bar in extra}
    for bar in primary:
        by_open[bar["open_time"]] = bar
    ordered = [by_open[key] for key in sorted(by_open)]
    return ordered[-limit:]


def _limit_block(ban, now_ms: int) -> dict:
    if ban is None or not ban.active(now_ms):
        return {"active": False, "until_ms": None, "until_utc8": None, "reason": "", "message": ""}
    until = int(ban.until_ms)
    clock = format_utc8(until, "%H:%M:%S")
    return {
        "active": True,
        "until_ms": until,
        "until_utc8": format_utc8(until),
        "reason": ban.reason,
        "message": f"限流中，{clock} 后恢复",
    }


def _cache_metrics(cache: BoardCache, symbol: str) -> tuple[list[dict], int | None]:
    points = cache.points("metric", symbol, limit=500)
    rows = []
    fetched = None
    for point in points:
        payload = dict(point["payload"])
        payload["ts"] = int(payload.get("ts") or point["ts"])
        payload["known_at"] = int(payload.get("known_at") or payload["ts"] + METRIC_MS)
        rows.append(payload)
        fetched = point["fetched_at"] if fetched is None else max(fetched, point["fetched_at"])
    rows.sort(key=lambda item: item["ts"])
    deduped = []
    seen = set()
    for row in rows:
        if row["ts"] in seen:
            continue
        seen.add(row["ts"])
        deduped.append(row)
    return deduped, fetched


def _cache_bars(cache: BoardCache, symbol: str, interval: str) -> list[dict]:
    points = cache.points(f"kline:{interval}", symbol, limit=120)
    bars = []
    for point in points:
        payload = dict(point["payload"])
        payload["open_time"] = int(payload["open_time"])
        payload["close_time"] = int(payload["close_time"])
        bars.append(payload)
    bars.sort(key=lambda bar: bar["open_time"])
    return bars


def assemble(
    reader: MainReader,
    cache: BoardCache,
    ban,
    analysis: AnalysisConfig,
    now_ms: int,
) -> dict:
    symbol = analysis.board.symbol
    limit = analysis.board.warmup_bars
    stale = analysis.board.stale
    grace = stale.kline_grace_seconds * 1000
    klines = {}
    for interval in BOARD_INTERVALS:
        primary = _closed(reader.klines(symbol, interval, limit=limit, as_of_ms=now_ms), now_ms)
        extra = _closed(_cache_bars(cache, symbol, interval), now_ms)
        merged = _merge_bars(primary, extra, limit)
        latest = merged[-1]["close_time"] if merged else None
        status = kline_status(latest, now_ms, INTERVAL_MS[interval], grace)
        source = "主库 klines"
        if merged and merged[-1]["open_time"] not in {bar["open_time"] for bar in primary}:
            source = "币安 /fapi/v1/klines"
        klines[interval] = {
            "status": status,
            "source": source if merged else "",
            "data_ts": latest,
            "utc8": None if latest is None else format_utc8(latest),
            "label": "未获取" if status == "missing" else (
                f"上次数据 {format_utc8(latest, '%H:%M:%S')}（已过期）" if status == "stale" else format_utc8(latest, "%H:%M:%S")
            ),
            "bars": merged,
        }

    main_metrics = reader.metrics(symbol, limit=limit, as_of_ms=now_ms)
    cache_metrics, cache_fetched = _cache_metrics(cache, symbol)
    if now_ms:
        cache_metrics = [row for row in cache_metrics if row["known_at"] <= int(now_ms)]
    chosen = choose_metrics(
        main_metrics,
        cache_metrics,
        now_ms,
        stale.metrics_minutes * 60_000,
        cache_fetched,
    )
    metrics_stamp_value = chosen["rows"][-1]["oi"] if chosen["rows"] else None
    metrics_view = judge(
        metrics_stamp_value if chosen["status"] != "missing" else None,
        chosen["known_at"],
        chosen["fetched_at"],
        chosen["source"],
        now_ms,
        stale.metrics_minutes * 60_000,
    )
    # judge() would also compute status. Keep choose_metrics' decision so an empty value
    # inside a present row still reports the series status.
    if chosen["status"] == "missing":
        metrics_view["status"] = "missing"
        metrics_view["label"] = "未获取"
    else:
        metrics_view["status"] = chosen["status"]
        metrics_view["data_ts"] = chosen["known_at"]
        clock = format_utc8(int(chosen["known_at"]), "%H:%M:%S")
        metrics_view["label"] = clock if chosen["status"] == "ok" else f"上次数据 {clock}（已过期）"
        metrics_view["value"] = metrics_stamp_value

    oi_point = cache.latest("oi", symbol)
    if oi_point and int(oi_point["ts"]) <= int(now_ms):
        oi_view = judge(
            oi_point["payload"].get("open_interest"),
            int(oi_point["ts"]),
            oi_point["fetched_at"],
            oi_point["source"],
            now_ms,
            stale.oi_seconds * 1000,
        )
    else:
        oi_view = judge(None, None, None, "币安 /fapi/v1/openInterest", now_ms, stale.oi_seconds * 1000)

    live = reader.funding_live(symbol)
    funding_cache = cache.latest("funding", symbol)
    funding_candidates = []
    if live and live.get("updated_at") is not None and (now_ms is None or live["updated_at"] <= now_ms):
        funding_candidates.append(
            judge(
                live,
                int(live["updated_at"]),
                int(live["updated_at"]),
                "主库 funding_live",
                now_ms,
                stale.funding_seconds * 1000,
            )
        )
    if funding_cache and int(funding_cache["ts"]) <= int(now_ms):
        funding_candidates.append(
            judge(
                funding_cache["payload"],
                int(funding_cache["ts"]),
                funding_cache["fetched_at"],
                funding_cache["source"],
                now_ms,
                stale.funding_seconds * 1000,
            )
        )
    fresh_funding = [item for item in funding_candidates if item["status"] == "ok"]
    if fresh_funding:
        funding_view = fresh_funding[0]
    elif funding_candidates:
        funding_candidates.sort(key=lambda item: item["data_ts"] or 0, reverse=True)
        funding_view = funding_candidates[0]
    else:
        funding_view = judge(None, None, None, "币安 /fapi/v1/premiumIndex", now_ms, stale.funding_seconds * 1000)

    okx_point = cache.latest("okx", symbol)
    if okx_point and int(okx_point["ts"]) <= int(now_ms):
        okx_view = judge(
            okx_point["payload"].get("price"),
            int(okx_point["ts"]),
            okx_point["fetched_at"],
            "OKX",
            now_ms,
            stale.okx_seconds * 1000,
        )
    else:
        okx_view = judge(None, None, None, "OKX", now_ms, stale.okx_seconds * 1000)

    price_view = _price_view(klines, funding_view, now_ms, grace)
    binance_price = price_view["value"] if price_view["status"] == "ok" else None
    check = crosscheck(
        binance_price,
        okx_view["value"] if okx_view["status"] == "ok" else None,
        analysis.board.crosscheck.max_diff_pct,
        binance_ok=price_view["status"] == "ok",
        okx_ok=okx_view["status"] == "ok",
    )
    base_bars = klines["5m"]["bars"] or klines["15m"]["bars"]
    ticker = ticker_from_bars(base_bars, now_ms) if base_bars else None
    ticker_status = klines["5m"]["status"] if klines["5m"]["bars"] else klines["15m"]["status"]
    return {
        "symbol": symbol,
        "now_ms": int(now_ms),
        "now_utc8": format_utc8(int(now_ms)),
        "price": price_view,
        "ticker": {
            "value": ticker,
            "status": "missing" if ticker is None else ticker_status,
            "source": "K 线" if ticker else "",
            "data_ts": None if ticker is None else ticker["data_ts"],
            "utc8": None if ticker is None else format_utc8(ticker["data_ts"], "%H:%M:%S"),
        },
        "klines": klines,
        "metrics": {
            **metrics_view,
            "rows": chosen["rows"],
        },
        "oi": oi_view,
        "funding": funding_view,
        "okx": okx_view,
        "crosscheck": check,
        "limit": _limit_block(ban, now_ms),
        "events": reader.events(),
    }


def _price_view(klines: dict, funding_view: dict, now_ms: int, grace_ms: int) -> dict:
    if funding_view["status"] == "ok" and isinstance(funding_view["value"], dict) and funding_view["value"].get("mark"):
        return {
            **funding_view,
            "value": float(funding_view["value"]["mark"]),
            "source": funding_view["source"],
        }
    for interval in ("5m", "15m", "1h"):
        series = klines[interval]
        if series["bars"] and series["status"] == "ok":
            last = series["bars"][-1]
            return judge(last["close"], last["close_time"], last["close_time"], series["source"], now_ms, INTERVAL_MS[interval] + grace_ms)
    for interval in ("5m", "15m", "1h"):
        series = klines[interval]
        if series["bars"]:
            last = series["bars"][-1]
            view = judge(last["close"], last["close_time"], last["close_time"], series["source"], now_ms, INTERVAL_MS[interval] + grace_ms)
            view["status"] = series["status"]
            if series["status"] == "stale":
                view["label"] = f"上次数据 {format_utc8(last['close_time'], '%H:%M:%S')}（已过期）"
            return view
    return judge(None, None, None, "", now_ms, grace_ms)
