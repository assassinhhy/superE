"""At most four conditional plans. Nothing here places an order."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from supere.board.calc import estimate
from supere.board.confirm import confirm_bar
from supere.board.indicators import last_value
from supere.intervals import INTERVAL_MS
from supere.timeutil import format_utc8

# Bump when the entry, stop, or target rules change. Stored on every journal row.
RULES_VERSION = "fixed-entry-1"
FIFTEEN_MS = INTERVAL_MS["15m"]


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


def round_to_tick(price: float, tick: float) -> float:
    """Nearest tick, half away from zero via ``ROUND_HALF_UP`` on the tick grid."""
    if tick <= 0:
        return float(price)
    step = Decimal(str(tick))
    units = (Decimal(str(price)) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(units * step)


def _nearer_end(low: float, high: float, key: float, side: str) -> float:
    """Range end closer to ``key``. A tie uses the end the stop is measured from.

    The live trend zone is ``key ± band``, so both ends are the same distance
    from the level. The tie goes to the high end of a short (the pullback has
    to reach resistance) and the low end of a long (the pullback has to reach
    support).
    """
    if abs(low - key) + 1e-9 < abs(high - key):
        return low
    if abs(high - key) + 1e-9 < abs(low - key):
        return high
    return high if side == "short" else low


def fixed_entry_price(
    *,
    kind: str,
    side: str,
    entry_low: float,
    entry_high: float,
    key: float,
    atr: float,
    tick: float,
    offset_atr: float = 0.1,
) -> float:
    """One price inside the published entry range, rounded to the tick.

    * 顺势 pullback: the range end nearer the key level (tie: stop-side end).
    * 破位 breakout/breakdown: key ± ``offset_atr`` × ATR, in the trade direction.
    * 逆势 stop-sweep: the midpoint of the range.
    """
    if kind == "破位":
        raw = key + offset_atr * atr if side == "long" else key - offset_atr * atr
    elif kind == "逆势":
        raw = (entry_low + entry_high) / 2.0
    elif kind == "顺势":
        raw = _nearer_end(entry_low, entry_high, key, side)
    else:
        raw = (entry_low + entry_high) / 2.0
    return round_to_tick(raw, tick)


def r_multiples(entry: float, stop: float | None, side: str, targets: list[float]) -> tuple[float | None, float | None, float | None]:
    """Stop distance and T1/T2 R from the fixed entry. Reward is price, before fees."""
    if stop is None:
        return None, None, None
    risk = abs(entry - stop)
    if risk <= 0:
        return risk, None, None

    def reward(target: float | None) -> float | None:
        if target is None:
            return None
        move = (target - entry) if side == "long" else (entry - target)
        return move / risk

    t1 = targets[0] if targets else None
    t2 = targets[1] if len(targets) > 1 else None
    return risk, reward(t1), reward(t2)


def _priced(
    *,
    rule_id: str,
    kind: str,
    title: str,
    side: str,
    status: str,
    entry_low: float,
    entry_high: float,
    stop: float | None,
    targets: list[float],
    size_ratio: float,
    note: str,
    names: list[str],
    analysis,
    atr: float,
    key: float,
    entry_rule: str,
    line: float | None = None,
) -> dict:
    cfg = analysis.plans
    entry = fixed_entry_price(
        kind=kind,
        side=side,
        entry_low=entry_low,
        entry_high=entry_high,
        key=key,
        atr=atr,
        tick=cfg.tick,
        offset_atr=cfg.breakout_entry_atr,
    )
    risk, t1_r, t2_r = r_multiples(entry, stop, side, targets)
    calc = None
    if stop is not None and size_ratio:
        calc = _calc(entry, stop, side, targets, atr, analysis, size_ratio)
    plan = {
        "id": rule_id,
        "kind": kind,
        "title": title,
        "side": side,
        "status": status,
        "entry": entry,
        "entry_rule": entry_rule,
        "key_level": key,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "targets": targets,
        "r_distance": risk,
        "t1_r": t1_r,
        "t2_r": t2_r,
        "size_ratio": size_ratio,
        "note": note,
        "confirm": {"ok": bool(names), "names": names},
        "calc": calc,
    }
    if line is not None:
        plan["line"] = line
    return plan


def stamp_plan_times(plans: list[dict], meta: dict, *, boundary_ms: int | None, valid_bars: int, interval_ms: int = FIFTEEN_MS) -> None:
    """Stamp every plan with the closed 15m bar it used and the hour it expires.

    ``boundary_ms`` is the candle's quoted close (HH:MM UTC+8), not the poll time.
    Expiry is ``valid_bars`` steps later. Four 15m bars is one hour: 21:45 → 22:45.
    """
    if boundary_ms is None:
        return
    boundary = int(boundary_ms)
    expires = boundary + int(valid_bars) * int(interval_ms)
    clock = format_utc8(boundary, "%H:%M")
    until = format_utc8(expires, "%H:%M")
    text = f"{clock} 生成 · 有效至 {until}"
    meta["generated_utc8"] = clock
    meta["generated_ms"] = boundary
    meta["basis_close_ms"] = boundary
    meta["basis_close_utc8"] = clock
    meta["expires_ms"] = expires
    meta["expires_utc8"] = until
    meta["validity"] = text
    for plan in plans:
        plan["generated_ms"] = boundary
        plan["generated_utc8"] = clock
        plan["basis_close_ms"] = boundary
        plan["expires_ms"] = expires
        plan["expires_utc8"] = until
        plan["validity"] = text


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
        add(
            _priced(
                rule_id="trend",
                kind="顺势",
                title=title,
                side=side,
                status="可进场" if names else "等待信号",
                entry_low=entry_low,
                entry_high=entry_high,
                stop=stop,
                targets=targets,
                size_ratio=1.0,
                note="",
                names=names,
                analysis=analysis,
                atr=atr15,
                key=center,
                entry_rule="trend_near_key",
            )
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
        shown = names if broken else []
        add(
            _priced(
                rule_id="breakout",
                kind="破位",
                title="破位空" if side == "short" else "破位多",
                side=side,
                status="可进场" if broken and names else "等待信号",
                entry_low=zone[0],
                entry_high=zone[1],
                stop=stop,
                targets=targets,
                size_ratio=1.0,
                note=note,
                names=shown,
                analysis=analysis,
                atr=atr15,
                key=edge,
                entry_rule="breakout_atr",
            )
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
            _priced(
                rule_id="counter",
                kind="逆势",
                title=title,
                side=counter_side,
                status="可进场" if names else "等待信号",
                entry_low=zone[0],
                entry_high=zone[1],
                stop=stop,
                targets=targets,
                size_ratio=plans_cfg.counter_size_ratio,
                note=note,
                names=names,
                analysis=analysis,
                atr=atr15,
                key=level,
                entry_rule="counter_mid",
            )
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
        flip_side = "long" if side == "short" else "short"
        add(
            _priced(
                rule_id="flip",
                kind="翻转",
                title="翻转条件",
                side=flip_side,
                status="已触发" if flipped else "等待信号",
                entry_low=line,
                entry_high=line,
                stop=None,
                targets=[],
                size_ratio=0,
                note=(
                    f"15m 连续 {plans_cfg.flip_closes_15m} 根，或 1h/4h 各 {plans_cfg.flip_closes_1h} 根，"
                    f"收在 {_fmt(line)} {'上' if side == 'short' else '下'} → 当前方向作废"
                ),
                names=["收稳"] if flipped else [],
                analysis=analysis,
                atr=atr15,
                key=line,
                entry_rule="flip_line",
                line=line,
            )
        )
        if flipped:
            meta["invalidated"] = True
            for plan in plans:
                if plan["id"] != "flip":
                    plan["status"] = "已作废"
    return plans, meta
