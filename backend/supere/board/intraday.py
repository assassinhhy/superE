"""Short-term intraday plans. Analysis only: nothing here places an order.

v2 changes from the 2026-11-14 rules. Anything this module does not touch
stays on the v1 path in ``plans.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from supere.board.confirm import confirm_bar
from supere.board.indicators import last_value
from supere.board.plans import _calc, _closes_beyond, _fmt, _nearer_ma, protective_stop
from supere.board.sentiment import aligned_oi
from supere.timeutil import UTC8, format_utc8

SESSION_LABEL = {"asia": "亚洲", "europe": "欧洲", "us": "美国"}


def _hhmm(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def clock_minutes(ms: int) -> int:
    moment = datetime.fromtimestamp(ms / 1000, tz=UTC8)
    return moment.hour * 60 + moment.minute


def sgt_day(ms: int) -> str:
    return format_utc8(ms, "%Y-%m-%d")


def session_key(ms: int, cfg) -> str | None:
    """Left-closed, right-open. 15:00 is Europe, 20:30 is the US, 02:00 is neither."""
    minutes = clock_minutes(ms)
    hours = cfg.session
    for key in ("asia", "europe"):
        start, end = (_hhmm(hours.__getattribute__(key)[0]), _hhmm(hours.__getattribute__(key)[1]))
        if start <= minutes < end:
            return key
    start, end = _hhmm(hours.us[0]), _hhmm(hours.us[1])
    if minutes >= start or minutes < end:
        return "us"
    return None


def session_label(ms: int, cfg) -> str:
    key = session_key(ms, cfg)
    return SESSION_LABEL.get(key or "", "其他")


def entries_open(ms: int, cfg) -> bool:
    """No new plans from ``no_new_after`` until the Asia session opens."""
    minutes = clock_minutes(ms)
    stop = _hhmm(cfg.no_new_after)
    reopen = _hhmm(cfg.session.asia[0])
    if stop <= minutes < reopen:
        return False
    return True


def next_clock_ms(ms: int, hhmm: str) -> int:
    moment = datetime.fromtimestamp(ms / 1000, tz=UTC8)
    hour, minute = (int(part) for part in hhmm.split(":"))
    target = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= moment:
        target += timedelta(days=1)
    return int(target.timestamp() * 1000)


def hold_until_text(signal_ms: int, cfg) -> str:
    entry_ms = int(signal_ms) + 1
    max_ms = entry_ms + int(cfg.max_hold_hours * 3_600_000)
    flat_ms = next_clock_ms(entry_ms, cfg.flat_by)
    return format_utc8(min(max_ms, flat_ms), "%H:%M")


def _vol_ratio(bars: list[dict], index: int, window: int) -> float | None:
    if index < window:
        return None
    base = bars[index - window:index]
    avg = sum(bar["volume"] for bar in base) / window
    if avg <= 0:
        return None
    return bars[index]["volume"] / avg


def _oi_delta(rows: list[dict], previous_ms: int, at_ms: int, status: str) -> float | None:
    if status != "ok" or not rows:
        return None
    end = aligned_oi(rows, at_ms)
    start = aligned_oi(rows, previous_ms)
    if not end or not start:
        return None
    if end.get("oi") is None or start.get("oi") is None:
        return None
    if int(end["known_at"]) == int(start["known_at"]):
        return None
    return float(end["oi"]) - float(start["oi"])


def _strong(level: dict) -> bool:
    sources = " ".join(level.get("sources") or [])
    if "插针扫过" in sources or "前日低" in sources or "前日高" in sources:
        return True
    return int(level.get("touches") or 0) >= 2


def _place_targets(entry: float, side: str, prices: list[float], risk: float, t1_min: float, t2_min: float) -> tuple[float, float]:
    """T1 is the further of the nearest level and ``t1_min`` R. T2 is the next level, at least ``t2_min`` R."""
    if side == "long":
        ahead = [price for price in prices if price > entry]
        floor1 = entry + t1_min * risk
        t1 = max(ahead[0], floor1) if ahead else floor1
        further = [price for price in ahead if price > t1 + 1e-9]
        floor2 = max(entry + t2_min * risk, t1)
        t2 = further[0] if further and further[0] >= floor2 - 1e-9 else floor2
        return t1, max(t2, t1)
    ahead = [price for price in prices if price < entry]
    floor1 = entry - t1_min * risk
    t1 = min(ahead[0], floor1) if ahead else floor1
    further = [price for price in ahead if price < t1 - 1e-9]
    floor2 = min(entry - t2_min * risk, t1)
    t2 = further[0] if further and further[0] <= floor2 + 1e-9 else floor2
    return t1, min(t2, t1)


def _reward_ok(entry: float, target: float, risk: float, side: str, minimum: float) -> bool:
    if risk <= 0:
        return False
    move = (target - entry) if side == "long" else (entry - target)
    return move + 1e-9 >= minimum * risk


def _confirmed(side: str, zone: tuple[float, float], packs: dict, analysis, atr: float) -> list[str]:
    names: list[str] = []
    for frame in analysis.confirm.frames:
        pack = packs.get(frame)
        if not pack:
            continue
        names.extend(
            confirm_bar(
                pack["bars"],
                pack["rsi"],
                side=side,
                atr=atr,
                zone=zone,
                params=analysis.confirm,
            )
        )
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def _annotate(plan: dict, entry: float, signal_ms: int, cfg) -> dict:
    stop = plan.get("stop")
    if stop is None or entry is None:
        return plan
    risk = abs(entry - stop)
    plan["r"] = risk
    plan["entry_ref"] = entry
    targets = plan.get("targets") or []
    plan["t1_r"] = (abs(targets[0] - entry) / risk) if risk and targets else None
    plan["t2_r"] = (abs(targets[1] - entry) / risk) if risk and len(targets) > 1 else None
    plan["hold_until_utc8"] = hold_until_text(signal_ms, cfg)
    plan["flat_by_utc8"] = cfg.flat_by
    return plan


def _edge_before(bars: list[dict], index: int, width: int, side: str) -> float | None:
    """Range edge of the ``width`` bars before ``index``.

    The signal bar is not part of the range. A close cannot be beyond the
    extreme of a window that already contains that bar.
    """
    if index < width:
        return None
    window = bars[index - width:index]
    if side == "short":
        return min(bar["low"] for bar in window)
    return max(bar["high"] for bar in window)


def _direct_break(bars: list[dict], edge: float, side: str, atr: float, cfg, rows, status: str) -> bool:
    last = bars[-1]
    reach = cfg.breakout_direct_atr * atr
    if side == "short":
        far = edge - last["close"] >= reach - 1e-9
    else:
        far = last["close"] - edge >= reach - 1e-9
    if not far:
        return False
    ratio = _vol_ratio(bars, len(bars) - 1, cfg.breakout_vol_window)
    if ratio is None or ratio < cfg.breakout_vol_mult:
        return False
    previous = bars[-2]["close_time"] if len(bars) > 1 else last["open_time"]
    delta = _oi_delta(rows, previous, last["close_time"], status)
    # New positions pushing the break: OI has to rise. Missing OI fails the condition.
    return delta is not None and delta > 0


def _retest_break(bars: list[dict], side: str, band: float, limit: int, width: int) -> float | None:
    """Return the broken edge when the latest bar retests it within ``limit`` bars."""
    if len(bars) < width + 2:
        return None
    last_index = len(bars) - 1
    last = bars[last_index]
    start = max(width, last_index - limit)
    for index in range(start, last_index):
        edge = _edge_before(bars, index, width, side)
        if edge is None:
            continue
        if side == "short":
            broke = bars[index]["close"] < edge
            touched = last["high"] >= edge - band
            held = last["close"] < edge
        else:
            broke = bars[index]["close"] > edge
            touched = last["low"] <= edge + band
            held = last["close"] > edge
        if broke and touched and held:
            return edge
    return None


def _sweep(bars: list[dict], level: float, side: str) -> dict | None:
    """Latest bar that traded through ``level`` while the newest close is back inside."""
    last = bars[-1]
    if side == "long" and last["close"] <= level:
        return None
    if side == "short" and last["close"] >= level:
        return None
    found = None
    for bar in bars[-12:]:
        if side == "long" and bar["low"] < level:
            found = bar
        if side == "short" and bar["high"] > level:
            found = bar
    return found


def _block_reason(book: dict, analysis) -> str:
    if not book:
        return ""
    cfg = analysis.intraday
    if int(book.get("open") or 0) >= cfg.max_open:
        return "已有持仓，不加仓"
    if float(book.get("day_r") or 0) <= -cfg.day_loss_r:
        return f"当天亏损到 {cfg.day_loss_r:g}R，停止新计划"
    if int(book.get("loss_streak") or 0) >= analysis.risk.loss_streak_stop:
        return "连亏已到上限，停止新计划"
    return ""


def build_plans_v2(
    *,
    bias: str,
    price: float | None,
    levels: dict,
    rng: dict | None,
    packs: dict,
    analysis,
    frames: dict | None = None,
    now_ms: int = 0,
    metrics: dict | None = None,
    book: dict | None = None,
    daily_rsi: float | None = None,
    weekend: bool = False,
    event_block: bool = False,
) -> tuple[list[dict], dict]:
    del bias
    cfg = analysis.intraday
    frames = frames or {}
    book = book or {}
    metrics = metrics or {}
    meta = {
        "paused": False,
        "pause_reason": "",
        "session_note": "",
        "invalidated": False,
        "valid_bars": analysis.plans.valid_bars_15m,
        "conflict": False,
        "profile": "intraday_v2",
    }
    if price is None or not packs.get("15m") or not packs.get("1h"):
        meta["paused"] = True
        meta["pause_reason"] = "未获取"
        return [], meta
    atr15 = last_value(packs["15m"]["atr"])
    atr1h = last_value(packs["1h"]["atr"])
    if atr15 is None or atr1h is None or atr15 <= 0:
        meta["paused"] = True
        meta["pause_reason"] = "未获取"
        return [], meta
    h1 = frames.get("1h")
    h4 = frames.get("4h")
    if h1 not in {"多", "空", "震荡"} or h4 not in {"多", "空", "震荡"}:
        meta["paused"] = True
        meta["pause_reason"] = "未获取"
        return [], meta
    signal_ms = int(packs["15m"]["bars"][-1]["close_time"] or now_ms)
    if (h1 == "多" and h4 == "空") or (h1 == "空" and h4 == "多"):
        meta["conflict"] = True
        meta["paused"] = True
        meta["pause_reason"] = "方向打架，不做"
        return [], meta
    if not entries_open(signal_ms, cfg):
        meta["paused"] = True
        meta["session_note"] = "接近收盘时间，只管理已有仓位，不开新单"
        meta["pause_reason"] = meta["session_note"]
        return [], meta
    key = session_key(signal_ms, cfg)
    enabled = cfg.session_enabled
    if key and not bool(getattr(enabled, key)):
        meta["paused"] = True
        meta["pause_reason"] = f"{SESSION_LABEL[key]}时段期望为负，已关闭"
        return [], meta

    trend_side = None
    if h1 in {"多", "空"} and (h4 == h1 or h4 == "震荡"):
        trend_side = "long" if h1 == "多" else "short"
    range_only = h1 == "震荡"
    plans_cfg = analysis.plans
    rows = metrics.get("rows") or []
    status = metrics.get("status") or "missing"
    plans: list[dict] = []

    def stop_beyond(zone_edge: float, plan_side: str) -> float:
        return protective_stop(
            zone_edge,
            atr15,
            atr1h,
            a=plans_cfg.stop_atr15,
            b=plans_cfg.stop_atr1h_factor,
            side=plan_side,
        )

    def sized(ratio: float) -> float:
        return ratio * (0.5 if weekend else 1.0)

    def add(plan: dict, entry: float) -> None:
        _annotate(plan, entry, signal_ms, cfg)
        if len(plans) < plans_cfg.max_plans:
            plans.append(plan)

    resistance = levels.get("resistance") or []
    support = levels.get("support") or []
    catalog = levels.get("all") or (support + resistance)

    if trend_side and not range_only:
        anchor = resistance if trend_side == "short" else support
        other = support if trend_side == "short" else resistance
        if anchor and other:
            center = anchor[0]["price"]
            band = plans_cfg.entry_band_atr * atr15
            entry_low, entry_high = center - band, center + band
            entry = (entry_low + entry_high) / 2
            if trend_side == "short":
                stop = stop_beyond(entry_high, "short")
                title = "顺势反抽空（首选）"
            else:
                stop = stop_beyond(entry_low, "long")
                title = "顺势回踩多（首选）"
            risk = abs(entry - stop)
            prices = [level["price"] for level in other]
            t1, t2 = _place_targets(entry, trend_side, prices, risk, cfg.t1_min_r, cfg.t2_min_r)
            names = _confirmed(trend_side, (entry_low, entry_high), packs, analysis, atr15)
            room = _reward_ok(entry, t1, risk, trend_side, cfg.t1_min_r)
            add(
                {
                    "id": "trend",
                    "kind": "顺势",
                    "title": title,
                    "side": trend_side,
                    "status": "可进场" if names and room else ("空间不够" if not room else "等待信号"),
                    "entry_low": entry_low,
                    "entry_high": entry_high,
                    "stop": stop,
                    "targets": [t1, t2],
                    "size_ratio": sized(1.0),
                    "full_at_t1": False,
                    "note": "" if room else "空间不够",
                    "confirm": {"ok": bool(names and room), "names": names if room else []},
                    "calc": _calc(entry, stop, trend_side, [t1, t2], atr15, analysis, sized(1.0)),
                },
                entry,
            )

        if rng:
            band = plans_cfg.breakout_retest_atr * atr15
            width = analysis.range.bars
            bars15 = packs["15m"]["bars"]
            edge = _edge_before(bars15, len(bars15) - 1, width, trend_side)
            retest_edge = _retest_break(bars15, trend_side, band, plans_cfg.valid_bars_15m, width)
            direct = edge is not None and _direct_break(bars15, edge, trend_side, atr15, cfg, rows, status)
            retest = retest_edge is not None and not direct
            if retest:
                edge = retest_edge
            if edge is not None and direct:
                entry = packs["15m"]["bars"][-1]["close"]
                if trend_side == "short":
                    stop = edge + cfg.breakout_direct_stop_atr * atr15
                else:
                    stop = edge - cfg.breakout_direct_stop_atr * atr15
                note = (
                    f"15m 收线超出区间沿 {_fmt(cfg.breakout_direct_atr)}×ATR，放量且 OI 增加，下一根开盘跟进。"
                    "止损在区间沿内侧"
                )
                mode = "direct"
            elif edge is not None:
                entry = edge
                if trend_side == "short":
                    stop = edge + plans_cfg.breakout_stop_atr * atr15
                else:
                    stop = edge - plans_cfg.breakout_stop_atr * atr15
                note = (
                    f"15m 收破 {_fmt(edge)} 后 {_fmt(plans_cfg.valid_bars_15m)} 根内回抽到 "
                    f"{_fmt(edge - band)}–{_fmt(edge + band)} 不收回再进，不在插针里追"
                )
                mode = "retest" if retest else "wait"
            if edge is not None:
                ahead = support if trend_side == "short" else resistance
                prices = [level["price"] for level in ahead]
                risk = abs(entry - stop)
                t1, t2 = _place_targets(entry, trend_side, prices, risk, cfg.t1_min_r, cfg.t2_min_r) if risk > 0 else (entry, entry)
                room = risk > 0 and _reward_ok(entry, t1, risk, trend_side, cfg.t1_min_r)
                ready = (direct or retest) and room
                add(
                    {
                        "id": "breakout",
                        "kind": "破位",
                        "title": "破位空" if trend_side == "short" else "破位多",
                        "side": trend_side,
                        "status": "可进场" if ready else ("空间不够" if (direct or retest) and not room else "等待信号"),
                        "entry_low": edge - band,
                        "entry_high": edge + band,
                        "stop": stop,
                        "targets": [t1, t2],
                        "size_ratio": sized(1.0),
                        "full_at_t1": False,
                        "entry_mode": mode,
                        "note": note if room or not (direct or retest) else "空间不够",
                        "confirm": {"ok": ready, "names": ["直接跟进" if direct else "回抽不收回"] if ready else []},
                        "calc": _calc(entry, stop, trend_side, [t1, t2], atr15, analysis, sized(1.0)),
                    },
                    entry,
                )

        _add_counter(
            add,
            trend_side=trend_side,
            price=price,
            packs=packs,
            analysis=analysis,
            atr15=atr15,
            atr1h=atr1h,
            support=support,
            resistance=resistance,
            catalog=catalog,
            rows=rows,
            status=status,
            daily_rsi=daily_rsi,
            h4=h4,
            book=book,
            sized=sized,
            stop_beyond=stop_beyond,
        )

    if range_only and rng:
        _add_range(
            add,
            rng=rng,
            atr15=atr15,
            packs=packs,
            analysis=analysis,
            price=price,
            sized=sized,
        )

    side_for_flip = trend_side
    if side_for_flip is None and rng:
        side_for_flip = "short" if price >= (rng["high"] + rng["low"]) / 2 else "long"
    if side_for_flip:
        ema15 = last_value(packs["15m"]["ema"].get(50, []))
        ema1h = last_value(packs["1h"]["ema"].get(20, []))
        line = _nearer_ma(price, ema15, ema1h)
        if line is not None:
            flipped = (
                _closes_beyond(packs["15m"]["bars"], line, side_for_flip, plans_cfg.flip_closes_15m)
                or _closes_beyond(packs.get("1h", {}).get("bars") or [], line, side_for_flip, plans_cfg.flip_closes_1h)
                or _closes_beyond(packs.get("4h", {}).get("bars") or [], line, side_for_flip, plans_cfg.flip_closes_4h)
            )
            add(
                {
                    "id": "flip",
                    "kind": "翻转",
                    "title": "翻转条件",
                    "side": "long" if side_for_flip == "short" else "short",
                    "status": "已触发" if flipped else "等待信号",
                    "entry_low": line,
                    "entry_high": line,
                    "stop": None,
                    "targets": [],
                    "size_ratio": 0,
                    "note": (
                        f"15m 连续 {plans_cfg.flip_closes_15m} 根，或 1h/4h 各 1 根，"
                        f"收在 {_fmt(line)} {'上' if side_for_flip == 'short' else '下'} → 当前方向作废"
                    ),
                    "confirm": {"ok": flipped, "names": ["收稳"] if flipped else []},
                    "calc": None,
                    "line": line,
                },
                line,
            )
            if flipped:
                meta["invalidated"] = True
                for plan in plans:
                    if plan["id"] != "flip" and plan["status"] == "可进场":
                        plan["status"] = "已作废"
                        plan["confirm"]["ok"] = False

    reason = _block_reason(book, analysis)
    if event_block:
        reason = reason or "重要数据窗口，不开新单"
    if reason:
        meta["pause_reason"] = reason
        for plan in plans:
            if plan["status"] == "可进场":
                plan["status"] = "等待信号"
                plan["confirm"] = {"ok": False, "names": []}
    return plans, meta


def _add_counter(
    add,
    *,
    trend_side,
    price,
    packs,
    analysis,
    atr15,
    atr1h,
    support,
    resistance,
    catalog,
    rows,
    status,
    daily_rsi,
    h4,
    book,
    sized,
    stop_beyond,
) -> None:
    del atr1h, price
    cfg = analysis.intraday
    if int(book.get("counter") or 0) >= cfg.counter_per_day:
        return
    counter = "long" if trend_side == "short" else "short"
    rsi_ok = h4 == "震荡"
    if daily_rsi is not None:
        if counter == "long" and daily_rsi < cfg.counter_rsi_long:
            rsi_ok = True
        if counter == "short" and daily_rsi > cfg.counter_rsi_short:
            rsi_ok = True
    if not rsi_ok:
        return
    pool = []
    for level in catalog:
        if not _strong(level):
            continue
        if counter == "long" and level["price"] < packs["15m"]["bars"][-1]["close"]:
            pool.append(level)
        if counter == "short" and level["price"] > packs["15m"]["bars"][-1]["close"]:
            pool.append(level)
    if not pool:
        return
    level = min(pool, key=lambda item: abs(item["price"] - packs["15m"]["bars"][-1]["close"]))
    swept = _sweep(packs["15m"]["bars"], level["price"], counter)
    if swept is None:
        return
    try:
        index = next(i for i, bar in enumerate(packs["15m"]["bars"]) if bar is swept)
    except StopIteration:
        return
    ratio = _vol_ratio(packs["15m"]["bars"], index, cfg.counter_vol_window)
    previous_ms = packs["15m"]["bars"][index - 1]["close_time"] if index else swept["open_time"]
    delta = _oi_delta(rows, previous_ms, swept["close_time"], status)
    if ratio is None or ratio < cfg.counter_vol_mult or delta is None or delta >= 0:
        return
    band = analysis.plans.entry_band_atr * atr15
    zone = (level["price"] - band, level["price"] + band)
    entry = level["price"]
    if counter == "long":
        stop = stop_beyond(zone[0], "long")
        opposite = [item["price"] for item in resistance if item["price"] > entry]
        title = "逆势轻仓多"
        note = f"扫 {_fmt(level['price'])} 后收回，放量且 OI 下降。半仓，到 T1 全平"
    else:
        stop = stop_beyond(zone[1], "short")
        opposite = [item["price"] for item in support if item["price"] < entry]
        title = "逆势轻仓空"
        note = f"扫 {_fmt(level['price'])} 后收回，放量且 OI 下降。半仓，到 T1 全平"
    if not opposite:
        add(
            {
                "id": "counter",
                "kind": "逆势",
                "title": title,
                "side": counter,
                "status": "空间不够",
                "entry_low": zone[0],
                "entry_high": zone[1],
                "stop": stop,
                "targets": [],
                "size_ratio": sized(analysis.plans.counter_size_ratio),
                "full_at_t1": True,
                "note": "空间不够",
                "confirm": {"ok": False, "names": []},
                "calc": _calc(entry, stop, counter, [], atr15, analysis, sized(analysis.plans.counter_size_ratio)),
            },
            entry,
        )
        return
    t1 = opposite[0]
    risk = abs(entry - stop)
    ratio_size = sized(analysis.plans.counter_size_ratio)
    room = _reward_ok(entry, t1, risk, counter, cfg.t1_min_r)
    add(
        {
            "id": "counter",
            "kind": "逆势",
            "title": title,
            "side": counter,
            "status": "可进场" if room else "空间不够",
            "entry_low": zone[0],
            "entry_high": zone[1],
            "stop": stop,
            "targets": [t1],
            "size_ratio": ratio_size,
            "full_at_t1": True,
            "note": note if room else "空间不够",
            "confirm": {"ok": room, "names": ["扫位收回"] if room else []},
            "calc": _calc(entry, stop, counter, [t1], atr15, analysis, ratio_size),
        },
        entry,
    )


def _add_range(add, *, rng, atr15, packs, analysis, price, sized) -> None:
    cfg = analysis.intraday
    width = rng["width"]
    if width < cfg.range_width_min_atr * atr15 or width > cfg.range_width_max_atr * atr15:
        return
    mid = (rng["high"] + rng["low"]) / 2
    band = analysis.plans.entry_band_atr * atr15
    specs = (
        ("long", rng["low"], rng["low"] - cfg.range_stop_atr * atr15, "区间下沿多", rng["high"] - band),
        ("short", rng["high"], rng["high"] + cfg.range_stop_atr * atr15, "区间上沿空", rng["low"] + band),
    )
    for side, edge, stop, title, t2 in specs:
        zone = (edge - band, edge + band)
        entry = edge
        risk = abs(entry - stop)
        t1 = mid
        room = _reward_ok(entry, t1, risk, side, cfg.t1_min_r)
        names = _confirmed(side, zone, packs, analysis, atr15) if room else []
        # Only the edge the price is actually testing should be able to confirm.
        if side == "long" and price > zone[1]:
            names = []
        if side == "short" and price < zone[0]:
            names = []
        ratio = sized(1.0)
        add(
            {
                "id": f"range-{side}",
                "kind": "区间",
                "title": title,
                "side": side,
                "status": "可进场" if names and room else ("空间不够" if not room else "等待信号"),
                "entry_low": zone[0],
                "entry_high": zone[1],
                "stop": stop,
                "targets": [t1, t2],
                "size_ratio": ratio,
                "full_at_t1": False,
                "note": "止损在区间外，T1 看中线，T2 看到对侧内沿" if room else "空间不够",
                "confirm": {"ok": bool(names and room), "names": names},
                "calc": _calc(entry, stop, side, [t1, t2], atr15, analysis, ratio),
            },
            entry,
        )


def v2_conclusion(text: str, h1: str | None, h4: str | None, extra: list[str]) -> str:
    """Keep v1 tags such as the chase note, and put the v2 regime in front."""
    lead = None
    if h1 in {"多", "空"} and h4 in {"多", "空"} and h1 != h4:
        lead = "方向打架，不做"
    elif h1 == "震荡":
        lead = "1h 震荡：只做区间两头"
    elif h4 == "震荡" and h1 in {"多", "空"}:
        lead = "4h 震荡、1h 有方向：可以做顺势"
    drop = {
        "空头：只做空",
        "多头：只做多",
        "偏空：反抽找空",
        "偏多：回踩找多",
        "震荡，只做区间两头或不做",
        "方向未获取",
    }
    parts = [part for part in text.split("；") if part]
    if lead:
        parts = [part for part in parts if part not in drop]
        parts.insert(0, lead)
    for item in extra:
        if item and item not in parts:
            parts.append(item)
    return "；".join(parts)
