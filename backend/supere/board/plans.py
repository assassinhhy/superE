"""At most four conditional plans. Nothing here places an order."""

from __future__ import annotations

from supere.board.calc import estimate
from supere.board.confirm import confirm_bar
from supere.board.indicators import last_value


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def protective_stop(zone_edge: float, atr15: float, atr1h: float, *, a: float, b: float, side: str) -> float:
    """Stop beyond the entry zone: edge + max(a×ATR15m, b×ATR1h). Longs mirror it."""
    buffer = max(a * atr15, b * atr1h)
    if side == "short":
        return zone_edge + buffer
    return zone_edge - buffer


def _nearer_ma(price: float, ema15_50: float | None, ema1h_20: float | None) -> float | None:
    options = [value for value in (ema15_50, ema1h_20) if value is not None]
    if not options:
        return None
    return min(options, key=lambda value: abs(value - price))


def _closes_beyond(bars: list[dict], line: float, side: str, count: int) -> bool:
    if len(bars) < count:
        return False
    recent = bars[-count:]
    if side == "short":
        return all(bar["close"] > line for bar in recent)
    return all(bar["close"] < line for bar in recent)


def _strong(levels: list[dict], side: str, price: float) -> dict | None:
    pool = []
    for level in levels:
        sources = " ".join(level.get("sources") or [])
        strong = "插针扫过" in sources or "前日低" in sources or "前日高" in sources or level.get("touches", 0) >= 2
        if not strong:
            continue
        if side == "short" and level["price"] < price:
            pool.append(level)
        if side == "long" and level["price"] > price:
            pool.append(level)
    if not pool:
        return None
    return min(pool, key=lambda level: abs(level["price"] - price))


def _swept(bars: list[dict], level: float, side: str) -> bool:
    if side == "short":
        went_through = any(bar["low"] < level for bar in bars[-12:])
        back = bars[-1]["close"] > level
        return went_through and back
    went_through = any(bar["high"] > level for bar in bars[-12:])
    back = bars[-1]["close"] < level
    return went_through and back


def _calc(entry: float, stop: float, side: str, targets: list[float], atr: float, analysis, size_ratio: float) -> dict:
    calc = analysis.calc
    return estimate(
        entry=entry,
        stop=stop,
        side=side,
        targets=targets,
        atr=atr,
        capital=calc.capital_u,
        max_loss_u=calc.max_loss_u,
        max_loss_pct=calc.max_loss_pct,
        leverage=calc.leverage,
        leverage_max=calc.leverage_max,
        mmr=calc.mmr,
        taker_fee=calc.taker_fee,
        liq_min_atr=calc.liq_min_atr,
        fee_min_mult=calc.fee_min_mult,
        rr_min=calc.rr_min,
        size_ratio=size_ratio,
    )


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
    # Keep order, drop duplicates.
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def build_plans(
    *,
    bias: str,
    price: float | None,
    levels: dict,
    rng: dict | None,
    packs: dict,
    analysis,
) -> tuple[list[dict], dict]:
    meta = {"paused": False, "pause_reason": "", "invalidated": False, "valid_bars": analysis.plans.valid_bars_15m}
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
    side = "short" if bias in {"只做空", "偏空"} else "long" if bias in {"只做多", "偏多"} else None
    if side is None:
        # Range regime: still describe the two edges, without a trend chase.
        side = "short" if rng and price >= (rng["high"] + rng["low"]) / 2 else "long"
        directional = False
    else:
        directional = True
    plans_cfg = analysis.plans
    def stop_beyond(zone_edge: float, plan_side: str) -> float:
        return protective_stop(
            zone_edge,
            atr15,
            atr1h,
            a=plans_cfg.stop_atr15,
            b=plans_cfg.stop_atr1h_factor,
            side=plan_side,
        )
    resistance = levels.get("resistance") or []
    support = levels.get("support") or []
    plans: list[dict] = []

    def add(plan: dict) -> None:
        if len(plans) < plans_cfg.max_plans:
            plans.append(plan)

    anchor = (resistance if side == "short" else support)
    other = support if side == "short" else resistance
    if directional and anchor and other:
        center = anchor[0]["price"]
        band = plans_cfg.entry_band_atr * atr15
        entry_low = center - band
        entry_high = center + band
        if side == "short":
            stop = stop_beyond(entry_high, "short")
            title = "顺势反抽空（首选）"
        else:
            stop = stop_beyond(entry_low, "long")
            title = "顺势回踩多（首选）"
        targets = [level["price"] for level in other[:2]]
        zone = (entry_low, entry_high)
        names = _confirmed(side, zone, packs, analysis, atr15)
        entry = (entry_low + entry_high) / 2
        add(
            {
                "id": "trend",
                "kind": "顺势",
                "title": title,
                "side": side,
                "status": "可进场" if names else "等待信号",
                "entry_low": entry_low,
                "entry_high": entry_high,
                "stop": stop,
                "targets": targets,
                "size_ratio": 1.0,
                "note": "",
                "confirm": {"ok": bool(names), "names": names},
                "calc": _calc(entry, stop, side, targets, atr15, analysis, 1.0),
            }
        )

    if rng:
        edge = rng["low"] if side == "short" else rng["high"]
        band = plans_cfg.breakout_retest_atr * atr15
        if side == "short":
            stop = edge + plans_cfg.breakout_stop_atr * atr15
            targets = [level["price"] for level in support if level["price"] < edge][:1] or [rng["low"] - atr15]
            note = f"15m 收破 {_fmt(edge)} 后回抽到 {_fmt(edge - band)}–{_fmt(edge + band)} 不收回再进，不在插针里追"
        else:
            stop = edge - plans_cfg.breakout_stop_atr * atr15
            targets = [level["price"] for level in resistance if level["price"] > edge][:1] or [rng["high"] + atr15]
            note = f"15m 收破 {_fmt(edge)} 后回抽到 {_fmt(edge - band)}–{_fmt(edge + band)} 不收回再进，不在插针里追"
        zone = (edge - band, edge + band)
        names = _confirmed(side, zone, packs, analysis, atr15)
        broken = any(
            (bar["close"] < rng["low"] if side == "short" else bar["close"] > rng["high"])
            for bar in packs["15m"]["bars"][-plans_cfg.valid_bars_15m:]
        )
        add(
            {
                "id": "breakout",
                "kind": "破位",
                "title": "破位空" if side == "short" else "破位多",
                "side": side,
                "status": "可进场" if broken and names else "等待信号",
                "entry_low": zone[0],
                "entry_high": zone[1],
                "stop": stop,
                "targets": targets,
                "size_ratio": 1.0,
                "note": note,
                "confirm": {"ok": bool(broken and names), "names": names if broken else []},
                "calc": _calc((zone[0] + zone[1]) / 2, stop, side, targets, atr15, analysis, 1.0),
            }
        )

    counter_side = "long" if side == "short" else "short"
    strong = _strong(support + resistance, side, price)
    if strong:
        level = strong["price"]
        band = plans_cfg.entry_band_atr * atr15
        zone = (level - band, level + band)
        swept = _swept(packs["15m"]["bars"], level, side)
        names = _confirmed(counter_side, zone, packs, analysis, atr15) if swept else []
        if counter_side == "long":
            stop = stop_beyond(zone[0], "long")
            targets = [resistance[0]["price"]] if resistance else [level + atr15 * 2]
            title = "逆势轻仓多"
            note = f"扫 {_fmt(level)} 后 15m 收回其上才看，半仓，不在插针里追"
        else:
            stop = stop_beyond(zone[1], "short")
            targets = [support[0]["price"]] if support else [level - atr15 * 2]
            title = "逆势轻仓空"
            note = f"扫 {_fmt(level)} 后 15m 收回其下才看，半仓，不在插针里追"
        add(
            {
                "id": "counter",
                "kind": "逆势",
                "title": title,
                "side": counter_side,
                "status": "可进场" if names else "等待信号",
                "entry_low": zone[0],
                "entry_high": zone[1],
                "stop": stop,
                "targets": targets,
                "size_ratio": plans_cfg.counter_size_ratio,
                "note": note,
                "confirm": {"ok": bool(names), "names": names},
                "calc": _calc((zone[0] + zone[1]) / 2, stop, counter_side, targets, atr15, analysis, plans_cfg.counter_size_ratio),
            }
        )

    ema15 = last_value(packs["15m"]["ema"].get(50, [])) if packs.get("15m") else None
    ema1h = last_value(packs["1h"]["ema"].get(20, [])) if packs.get("1h") else None
    line = _nearer_ma(price, ema15, ema1h)
    if line is not None:
        flipped = (
            _closes_beyond(packs["15m"]["bars"], line, side, plans_cfg.flip_closes_15m)
            or _closes_beyond(packs.get("1h", {}).get("bars", []), line, side, plans_cfg.flip_closes_1h)
            or _closes_beyond(packs.get("4h", {}).get("bars", []), line, side, plans_cfg.flip_closes_4h)
        )
        add(
            {
                "id": "flip",
                "kind": "翻转",
                "title": "翻转条件",
                "side": "long" if side == "short" else "short",
                "status": "已触发" if flipped else "等待信号",
                "entry_low": line,
                "entry_high": line,
                "stop": None,
                "targets": [],
                "size_ratio": 0,
                "note": (
                    f"15m 连续 {plans_cfg.flip_closes_15m} 根，或 1h/4h 各 {plans_cfg.flip_closes_1h} 根，"
                    f"收在 {_fmt(line)} {'上' if side == 'short' else '下'} → 当前方向作废"
                ),
                "confirm": {"ok": flipped, "names": ["收稳"] if flipped else []},
                "calc": None,
                "line": line,
            }
        )
        if flipped:
            meta["invalidated"] = True
            for plan in plans:
                if plan["id"] != "flip":
                    plan["status"] = "已作废"
    return plans, meta
