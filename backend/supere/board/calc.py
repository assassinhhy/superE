"""Position size from a stop. Isolated liquidation is an estimate and ignores fees."""

from __future__ import annotations


def estimate(
    *,
    entry: float,
    stop: float,
    side: str,
    targets: list[float],
    atr: float | None,
    capital: float,
    max_loss_u: float,
    max_loss_pct: float,
    leverage: float,
    leverage_max: float,
    mmr: float,
    taker_fee: float,
    liq_min_atr: float,
    fee_min_mult: float,
    rr_min: float,
    size_ratio: float = 1.0,
) -> dict:
    warnings: list[str] = []
    notes: list[str] = []
    if leverage > leverage_max:
        warnings.append("杠杆超过上限")
        leverage = leverage_max
    elif leverage > 20:
        notes.append("杠杆高于 20x，强平更靠近止损")
    risk = capital * max_loss_pct if max_loss_pct > 0 else max_loss_u
    risk *= size_ratio
    stop_pct = abs(entry - stop) / entry if entry else 0
    notional = None if stop_pct <= 0 else risk / stop_pct
    margin = None if notional is None or leverage <= 0 else notional / leverage
    if side == "long":
        liq = entry * (1 - 1 / leverage + mmr)
        outside = liq < stop
        distance = stop - liq
    else:
        liq = entry * (1 + 1 / leverage - mmr)
        outside = liq > stop
        distance = liq - stop
    if atr is not None and (not outside or distance < atr * liq_min_atr):
        warnings.append("杠杆过高/保证金不足")
    legs = []
    for target in targets:
        if notional is None or risk <= 0:
            break
        move = (target - entry) if side == "long" else (entry - target)
        pnl = move / entry * notional
        rr = pnl / risk
        legs.append({"target": target, "pnl_u": pnl, "rr": rr})
        if rr < rr_min:
            warnings.append("不划算")
            break
    fee = None if notional is None else notional * taker_fee * 2
    if fee is not None and legs and legs[0]["pnl_u"] < fee * fee_min_mult:
        warnings.append("波动太小，手续费吃利润")
    return {
        "entry": entry,
        "risk_u": risk,
        "stop_pct": stop_pct,
        "notional": notional,
        "margin": margin,
        "liq": liq,
        "liq_note": "估算，忽略手续费和维持保证金阶梯",
        "fee_u": fee,
        "legs": legs,
        "warnings": warnings,
        "notes": notes,
        "size_ratio": size_ratio,
    }
