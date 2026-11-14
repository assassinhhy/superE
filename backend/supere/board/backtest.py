"""Replay the same analyzer over closed 15m bars. No orders, no lookahead.

Hit: target 1 is traded before the stop. Fail: the stop is first. Invalid: neither
within 24 hours. A bar that touches both is counted as a stop. Round-trip taker
fee is ``0.05% × 2`` of the entry price, subtracted in price units. The slippage
pass fills one tick worse on the way in and on the way out.
"""

from __future__ import annotations

from datetime import datetime

from supere.board.engine import analyze
from supere.board.replay import snapshot_at
from supere.board.settings import AnalysisConfig
from supere.timeutil import UTC8

FEE_RATE = 0.0005
TICK = 0.01
HORIZON_BARS = 96  # 24 hours of 15m


def _slip(price: float, side: str, ticks: int, *, exiting: bool, tick: float = TICK) -> float:
    """Ticks against the fill. Shorts sell lower and buy back higher."""
    if not ticks:
        return price
    worse = ticks * tick
    if side == "short":
        return price - worse if not exiting else price + worse
    return price + worse if not exiting else price - worse


def _touch(bar: dict, side: str, stop: float, target: float | None) -> str | None:
    if side == "short":
        hit_stop = bar["high"] >= stop
        hit_target = target is not None and bar["low"] <= target
    else:
        hit_stop = bar["low"] <= stop
        hit_target = target is not None and bar["high"] >= target
    if hit_stop and hit_target:
        return "stop"
    if hit_stop:
        return "stop"
    if hit_target:
        return "target"
    return None


def simulate_trade(
    bars: list[dict],
    entry_index: int,
    *,
    side: str,
    entry: float,
    stop: float,
    targets: list[float],
    horizon: int = HORIZON_BARS,
    fee_rate: float = FEE_RATE,
    slippage_ticks: int = 0,
    tick: float = TICK,
) -> dict:
    """Walk bars after ``entry_index``. The entry bar itself is not a fill path."""
    del tick
    entry_fill = _slip(entry, side, slippage_ticks, exiting=False)
    planned_risk = abs(entry - stop)
    if planned_risk <= 0 or not targets:
        return {"outcome": "invalid", "rr": 0.0, "bars_used": 0, "net": 0.0}
    t1 = targets[0]
    t2 = targets[1] if len(targets) > 1 else None
    half = False
    end = min(len(bars), entry_index + 1 + horizon)
    outcome = "invalid"
    exit_pnl = 0.0
    used = 0
    for index in range(entry_index + 1, end):
        used = index - entry_index
        bar = bars[index]
        if not half:
            step = _touch(bar, side, stop, t1)
            if step == "stop":
                outcome = "fail"
                exit_pnl = _leg(side, entry_fill, _slip(stop, side, slippage_ticks, exiting=True))
                break
            if step == "target":
                half = True
                outcome = "hit"
                exit_pnl = 0.5 * _leg(side, entry_fill, _slip(t1, side, slippage_ticks, exiting=True))
                if t2 is None:
                    break
                rest = _touch(bar, side, stop, t2)
                if rest == "stop":
                    exit_pnl += 0.5 * _leg(side, entry_fill, _slip(stop, side, slippage_ticks, exiting=True))
                    break
                if rest == "target":
                    exit_pnl += 0.5 * _leg(side, entry_fill, _slip(t2, side, slippage_ticks, exiting=True))
                    break
            continue
        rest = _touch(bar, side, stop, t2)
        if rest == "stop":
            exit_pnl += 0.5 * _leg(side, entry_fill, _slip(stop, side, slippage_ticks, exiting=True))
            break
        if rest == "target":
            exit_pnl += 0.5 * _leg(side, entry_fill, _slip(t2, side, slippage_ticks, exiting=True))
            break
    else:
        if half:
            last = bars[end - 1]["close"] if end > entry_index + 1 else entry
            exit_pnl += 0.5 * _leg(side, entry_fill, _slip(last, side, slippage_ticks, exiting=True))
    fee = entry * fee_rate * 2
    net = exit_pnl - fee
    rr = net / planned_risk if planned_risk else 0.0
    return {"outcome": outcome, "rr": rr, "bars_used": used, "net": net}


def _leg(side: str, entry_fill: float, exit_fill: float) -> float:
    if side == "short":
        return entry_fill - exit_fill
    return exit_fill - entry_fill


def _summarize(trades: list[dict]) -> dict:
    counted = [trade for trade in trades if trade["outcome"] in {"hit", "fail"}]
    hits = [trade for trade in counted if trade["outcome"] == "hit"]
    fails = [trade for trade in counted if trade["outcome"] == "fail"]
    streak = 0
    worst = 0
    for trade in trades:
        if trade["outcome"] == "fail":
            streak += 1
            worst = max(worst, streak)
        elif trade["outcome"] == "hit":
            streak = 0
    avg = sum(trade["rr"] for trade in counted) / len(counted) if counted else 0.0
    return {
        "signals": len(trades),
        "counted": len(counted),
        "hits": len(hits),
        "fails": len(fails),
        "invalid": sum(1 for trade in trades if trade["outcome"] == "invalid"),
        "hit_rate": (len(hits) / len(counted)) if counted else 0.0,
        "avg_rr": avg,
        "max_loss_streak": worst,
    }


def _group(trades: list[dict], key: str) -> dict:
    buckets: dict[str, list] = {}
    for trade in trades:
        buckets.setdefault(str(trade.get(key)), []).append(trade)
    return {name: _summarize(rows) for name, rows in sorted(buckets.items())}


def walk(
    fixture: dict,
    analysis: AnalysisConfig,
    *,
    factor: float,
    slippage_ticks: int,
    fee_rate: float = FEE_RATE,
    horizon: int = HORIZON_BARS,
    min_bars: int = 80,
) -> dict:
    cfg = analysis.model_copy(deep=True)
    cfg.plans.stop_atr1h_factor = factor
    bars = fixture["frames"].get("15m") or []
    trades: list[dict] = []
    busy_until: dict[str, int] = {}
    for index, bar in enumerate(bars):
        if index < min_bars:
            continue
        if busy_until and all(until > index for until in busy_until.values()) and len(busy_until) >= 3:
            continue
        result = analyze(snapshot_at(fixture, bar["close_time"], cfg), cfg)
        if result.get("position") and result["position"].get("blocked"):
            continue
        if result.get("plans_meta", {}).get("paused"):
            continue
        if any(item.get("blocking") for item in result.get("risk", {}).get("events") or []):
            continue
        weekend = bool(result.get("risk", {}).get("weekend"))
        for plan in result.get("plans") or []:
            if plan.get("kind") not in {"顺势", "破位", "逆势"}:
                continue
            if plan.get("status") != "可进场" or plan.get("stop") is None or not plan.get("targets"):
                continue
            if busy_until.get(plan["id"], -1) > index:
                continue
            close = bar["close"]
            entry = close if plan["entry_low"] <= close <= plan["entry_high"] else (plan["entry_low"] + plan["entry_high"]) / 2
            sim = simulate_trade(
                bars,
                index,
                side=plan["side"],
                entry=entry,
                stop=plan["stop"],
                targets=plan["targets"],
                horizon=horizon,
                fee_rate=fee_rate,
                slippage_ticks=slippage_ticks,
            )
            trades.append(
                {
                    "kind": plan["kind"],
                    "side": plan["side"],
                    "weekend": weekend,
                    "entry_ms": bar["close_time"],
                    **sim,
                }
            )
            busy_until[plan["id"]] = index + 1 + max(sim["bars_used"], 1)
    summary = _summarize(trades)
    summary.update(
        {
            "factor": factor,
            "slippage_ticks": slippage_ticks,
            "fee_rate": fee_rate,
            "by_kind": _group(trades, "kind"),
            "by_side": _group(trades, "side"),
            "by_weekend": _group(trades, "weekend"),
        }
    )
    return summary


def notify_gate(summary: dict, *, min_sample: int = 50, min_hit: float = 0.45, min_rr: float = 1.3) -> bool:
    return summary["counted"] >= min_sample and summary["hit_rate"] >= min_hit and summary["avg_rr"] >= min_rr


def render_report(rows: list[dict], *, start: str, end: str, bars: int) -> str:
    lines = [
        "# 盘面情景计划回测",
        "",
        f"- 数据：15m 收盘逐根调用同一套 `analyze`，区间 {start} 至 {end}，15m 共 {bars} 根。",
        "- 进场：计划状态变成「可进场」，且价格不在区间中间 50%（那里的结论是不开单），也不在数据日历窗口里。",
        "- 命中：先到 T1 算命中，先到止损算失败，24 小时都没到算无效、不进分母。同一根 K 同时碰到止损和目标，按止损。",
        "- 盈亏：到 T1 平一半，剩下的一半看到 T2、止损或 24 小时收盘。盈亏比用扣费后的价格盈亏除以计划止损距离。",
        "- 费用：每一笔扣往返 taker 手续费 0.05%×2。滑点版在进场和出场各再坏 1 个 tick（ETH 0.01）。",
        "- 止损距离是进场区外侧加 `max(0.5×ATR15m, b×ATR1h)`，b 分别取 0.25、0.35、0.5。",
        "- 没有前视：指标只用当时已收盘的 K 线，合约数据只用 `known_at` 已经到点的行。",
        "",
        "| b | 滑点 | 触发 | 计入 | 命中 | 失败 | 无效 | 命中率 | 平均盈亏比 | 最大连亏 | 通知门槛 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        gate = "达到" if notify_gate(row) else "未达到"
        lines.append(
            f"| {row['factor']:.2f} | {row['slippage_ticks']} | {row['signals']} | {row['counted']} | "
            f"{row['hits']} | {row['fails']} | {row['invalid']} | {row['hit_rate']:.1%} | {row['avg_rr']:.2f} | "
            f"{row['max_loss_streak']} | {gate} |"
        )
    lines.append("")
    lines.append("通知门槛是样本不少于 50、命中率不低于 45%、平均盈亏比不低于 1.3。这一轮没有改 `notify.system`，默认仍然关闭。")
    lines.append("")
    if rows:
        lines.append("## 默认档分组（b=0.35，只扣手续费）")
        lines.append("")
        chosen = next((row for row in rows if row["factor"] == 0.35 and row["slippage_ticks"] == 0), rows[0])
        for title, key in (("计划类型", "by_kind"), ("方向", "by_side"), ("周末", "by_weekend")):
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| 组 | 计入 | 命中率 | 平均盈亏比 | 最大连亏 |")
            lines.append("|---|---:|---:|---:|---:|")
            for name, stats in chosen[key].items():
                label = {"True": "周末", "False": "非周末"}.get(name, name)
                lines.append(
                    f"| {label} | {stats['counted']} | {stats['hit_rate']:.1%} | {stats['avg_rr']:.2f} | {stats['max_loss_streak']} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def simulate_v2(
    bars5: list[dict],
    *,
    entry_ms: int,
    side: str,
    entry: float,
    stop: float,
    targets: list[float],
    full_at_t1: bool = False,
    time_stop_ms: int,
    max_hold_ms: int,
    flat_ms: int,
    fee_rate: float = FEE_RATE,
    slip_ticks: int = 1,
    stop_extra_ticks: int = 1,
    tick: float = TICK,
    time_stop_min_r: float = 0.5,
    ema_period: int = 20,
    warmup: int = 600,
) -> dict:
    """Fill at ``entry`` (the next open) and walk 5m bars. No lookahead past each close."""
    from supere.board.indicators import ema_series

    risk = abs(entry - stop)
    t1 = targets[0] if targets else None
    t2 = targets[1] if len(targets) > 1 else None
    if risk <= 0 or t1 is None:
        return {"outcome": "skip", "own_r": 0.0, "exit_ms": entry_ms, "exit_reason": "skip", "hold_min": 0.0}
    if side == "long" and (entry <= stop or entry >= t1):
        return {"outcome": "skip", "own_r": 0.0, "exit_ms": entry_ms, "exit_reason": "skip", "hold_min": 0.0}
    if side == "short" and (entry >= stop or entry <= t1):
        return {"outcome": "skip", "own_r": 0.0, "exit_ms": entry_ms, "exit_reason": "skip", "hold_min": 0.0}
    entry_fill = _slip(entry, side, slip_ticks, exiting=False, tick=tick)
    pad = entry * fee_rate * 2
    breakeven = entry + pad if side == "long" else entry - pad
    active_stop = stop
    half = False
    pnl = 0.0
    mfe = 0.0
    reason = ""
    exit_ms = entry_ms
    started = False
    closes = [bar["close"] for bar in bars5]

    def leg(fill: float) -> float:
        return _leg(side, entry_fill, fill)

    def slip_out(price: float, extra: int = 0) -> float:
        return _slip(price, side, slip_ticks + extra, exiting=True, tick=tick)

    for index, bar in enumerate(bars5):
        if bar["open_time"] < entry_ms:
            continue
        started = True
        hit_flat = bar["open_time"] >= flat_ms
        hit_hold = bar["open_time"] >= max_hold_ms
        if hit_flat or hit_hold:
            if hit_flat and flat_ms <= max_hold_ms:
                reason = "flat"
            else:
                reason = "max_hold"
            portion = 0.5 if half else 1.0
            pnl += portion * leg(slip_out(bar["open"]))
            exit_ms = bar["open_time"]
            break
        if side == "long":
            hit_stop = bar["low"] <= active_stop
            hit_t1 = (not half) and bar["high"] >= t1
            hit_t2 = t2 is not None and bar["high"] >= t2 and (half or hit_t1)
            favorable = bar["high"] - entry
        else:
            hit_stop = bar["high"] >= active_stop
            hit_t1 = (not half) and bar["low"] <= t1
            hit_t2 = t2 is not None and bar["low"] <= t2 and (half or hit_t1)
            favorable = entry - bar["low"]
        mfe = max(mfe, favorable)
        if hit_stop and (hit_t1 or (hit_t2 and not half)):
            hit_t1 = False
            hit_t2 = False
        if hit_stop:
            portion = 0.5 if half else 1.0
            pnl += portion * leg(slip_out(active_stop, stop_extra_ticks))
            reason = "stop"
            exit_ms = bar["close_time"]
            break
        if hit_t1:
            if full_at_t1 or t2 is None or abs(t1 - (t2 if t2 is not None else t1)) <= 1e-9:
                pnl += leg(slip_out(t1))
                reason = "t1"
                exit_ms = bar["close_time"]
                break
            pnl += 0.5 * leg(slip_out(t1))
            half = True
            if hit_t2:
                pnl += 0.5 * leg(slip_out(t2))
                reason = "t2"
                exit_ms = bar["close_time"]
                break
            active_stop = breakeven
            continue
        if half and hit_t2:
            pnl += 0.5 * leg(slip_out(t2))
            reason = "t2"
            exit_ms = bar["close_time"]
            break
        if not half and bar["close_time"] >= time_stop_ms - 1 and mfe < time_stop_min_r * risk:
            pnl += leg(slip_out(bar["close"]))
            reason = "time_stop"
            exit_ms = bar["close_time"]
            break
        if half:
            window = closes[max(0, index + 1 - warmup): index + 1]
            ema = ema_series(window, ema_period)[-1] if window else None
            crossed = ema is not None and ((side == "long" and bar["close"] < ema) or (side == "short" and bar["close"] > ema))
            if crossed:
                pnl += 0.5 * leg(slip_out(bar["close"]))
                reason = "trail"
                exit_ms = bar["close_time"]
                break
    if not started or not reason:
        return {"outcome": "invalid", "own_r": 0.0, "exit_ms": exit_ms, "exit_reason": "invalid", "hold_min": 0.0, "net": 0.0}
    net = pnl - entry * fee_rate * 2
    own_r = net / risk
    return {
        "outcome": "hit" if own_r > 0 else "fail",
        "own_r": own_r,
        "net": net,
        "exit_ms": exit_ms,
        "exit_reason": reason,
        "hold_min": max(0.0, (exit_ms - entry_ms) / 60_000),
    }


def summarize_v2(trades: list[dict]) -> dict:
    counted = [trade for trade in trades if trade["outcome"] in {"hit", "fail"}]
    wins = [trade for trade in counted if trade["own_r"] > 0]
    losses = [trade for trade in counted if trade["own_r"] <= 0]
    streak = 0
    worst = 0
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for trade in counted:
        equity += trade["account_r"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        if trade["own_r"] <= 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    expectancy = sum(trade["own_r"] for trade in counted) / len(counted) if counted else 0.0
    avg_win = sum(trade["own_r"] for trade in wins) / len(wins) if wins else 0.0
    avg_loss = sum(trade["own_r"] for trade in losses) / len(losses) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if wins and losses and avg_loss < 0 else None
    holds = [trade["hold_min"] for trade in counted]
    timed = [trade for trade in counted if trade.get("exit_reason") == "time_stop"]
    return {
        "signals": len(trades),
        "counted": len(counted),
        "hits": len(wins),
        "fails": len(losses),
        "invalid": sum(1 for trade in trades if trade["outcome"] == "invalid"),
        "hit_rate": (len(wins) / len(counted)) if counted else 0.0,
        "expectancy": expectancy,
        "payoff": payoff,
        "max_loss_streak": worst,
        "max_drawdown_r": drawdown,
        "avg_hold_min": (sum(holds) / len(holds)) if holds else 0.0,
        "time_stop_pct": (len(timed) / len(counted)) if counted else 0.0,
    }


def _group_v2(trades: list[dict], key: str) -> dict:
    buckets: dict[str, list] = {}
    for trade in trades:
        buckets.setdefault(str(trade.get(key)), []).append(trade)
    return {name: summarize_v2(rows) for name, rows in sorted(buckets.items())}


def _fresh_book(day: str) -> dict:
    return {"day": day, "day_r": 0.0, "loss_streak": 0, "counter": 0, "open": 0}


def _apply_book(book: dict, trade: dict) -> None:
    book["day_r"] = float(book.get("day_r") or 0) + float(trade["account_r"])
    if trade["own_r"] <= 0:
        book["loss_streak"] = int(book.get("loss_streak") or 0) + 1
    else:
        book["loss_streak"] = 0
    if trade["kind"] == "逆势":
        book["counter"] = int(book.get("counter") or 0) + 1


_KIND_ORDER = {"顺势": 0, "破位": 1, "区间": 2, "逆势": 3}


def walk_v2(
    fixture: dict,
    analysis: AnalysisConfig,
    *,
    start_ms: int,
    end_ms: int,
    min_bars: int = 80,
) -> list[dict]:
    """One position at a time. The entry is the next 15m open after the signal."""
    from supere.board.intraday import entries_open, next_clock_ms, session_key, session_label, sgt_day

    cfg = analysis
    bars = fixture["frames"].get("15m") or []
    bars5 = fixture["frames"].get("5m") or []
    rules = cfg.intraday
    book = _fresh_book("")
    trades: list[dict] = []
    pending: dict | None = None
    for index, bar in enumerate(bars):
        day = sgt_day(bar["close_time"])
        if book["day"] != day:
            book = _fresh_book(day)
        if pending is not None and bar["close_time"] >= pending["exit_ms"]:
            if sgt_day(pending["exit_ms"]) == day:
                _apply_book(book, pending)
            pending = None
            continue
        if pending is not None:
            continue
        if index < min_bars or bar["close_time"] < start_ms or bar["close_time"] >= end_ms:
            continue
        if not entries_open(bar["close_time"], rules):
            continue
        session = session_key(bar["close_time"], rules)
        if session is None or not bool(getattr(rules.session_enabled, session)):
            continue
        if index + 1 >= len(bars):
            break
        entry_bar = bars[index + 1]
        if entry_bar["open_time"] >= end_ms:
            break
        snap = snapshot_at(fixture, bar["close_time"], cfg)
        snap["book"] = book
        result = analyze(snap, cfg)
        if result.get("plans_meta", {}).get("paused") and not any(
            plan.get("status") == "可进场" for plan in result.get("plans") or []
        ):
            continue
        ready = [
            plan
            for plan in result.get("plans") or []
            if plan.get("status") == "可进场" and plan.get("stop") is not None and plan.get("targets")
        ]
        if not ready:
            continue
        ready.sort(key=lambda plan: _KIND_ORDER.get(plan["kind"], 9))
        plan = ready[0]
        entry = float(entry_bar["open"])
        risk = abs(entry - plan["stop"])
        t1 = plan["targets"][0]
        side = plan["side"]
        if risk <= 0:
            continue
        move = (t1 - entry) if side == "long" else (entry - t1)
        if move < rules.t1_min_r * risk - 1e-9:
            continue
        entry_ms = int(entry_bar["open_time"])
        sim = simulate_v2(
            bars5,
            entry_ms=entry_ms,
            side=side,
            entry=entry,
            stop=float(plan["stop"]),
            targets=[float(price) for price in plan["targets"]],
            full_at_t1=bool(plan.get("full_at_t1")),
            time_stop_ms=entry_ms + int(rules.time_stop_bars) * 15 * 60_000,
            max_hold_ms=entry_ms + int(rules.max_hold_hours * 3_600_000),
            flat_ms=next_clock_ms(entry_ms, rules.flat_by),
            fee_rate=rules.fee_rate,
            slip_ticks=rules.slippage_ticks,
            stop_extra_ticks=rules.stop_extra_ticks,
            tick=rules.tick,
            time_stop_min_r=rules.time_stop_min_r,
            ema_period=rules.trail_ema,
            warmup=cfg.board.warmup_bars,
        )
        if sim["outcome"] not in {"hit", "fail"}:
            continue
        size = float(plan.get("size_ratio") or 1.0)
        trade = {
            "kind": plan["kind"],
            "side": side,
            "session": session_label(entry_ms, rules),
            "entry_ms": entry_ms,
            "size_ratio": size,
            "account_r": sim["own_r"] * size,
            "weekend": bool(result.get("risk", {}).get("weekend")),
            **sim,
        }
        trades.append(trade)
        pending = trade
    return trades


def judge_intraday(in_sample: dict, out_sample: dict, overall: dict) -> dict:
    """Section I. Hit rate, payoff, streak and drawdown use the combined sample."""
    failed = []
    if not (in_sample["expectancy"] > 0 and out_sample["expectancy"] > 0):
        failed.append("样本内或样本外期望不大于 0")
    if overall["expectancy"] < 0.1:
        failed.append(f"总体期望 {overall['expectancy']:.3f}R < 0.1R")
    if overall["counted"] < 100:
        failed.append(f"总样本 {overall['counted']} < 100")
    if overall["hit_rate"] < 0.45:
        failed.append(f"命中率 {overall['hit_rate']:.1%} < 45%")
    payoff = overall.get("payoff")
    if payoff is None or payoff < 1.3:
        shown = "无" if payoff is None else f"{payoff:.2f}"
        failed.append(f"平均盈亏比 {shown} < 1.3")
    if overall["max_loss_streak"] > 8:
        failed.append(f"最大连亏 {overall['max_loss_streak']} > 8")
    if overall["max_drawdown_r"] > 15:
        failed.append(f"最大回撤 {overall['max_drawdown_r']:.2f}R > 15R")
    return {"passed": not failed, "eligible": not failed, "failed": failed}


def window_label(bars: list[dict]) -> tuple[str, str]:
    if not bars:
        return "—", "—"
    start = datetime.fromtimestamp(bars[0]["open_time"] / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")
    end = datetime.fromtimestamp(bars[-1]["close_time"] / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")
    return start, end
