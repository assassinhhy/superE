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


def _slip(price: float, side: str, ticks: int, *, exiting: bool) -> float:
    """One tick against the fill. Shorts sell lower and buy back higher."""
    if not ticks:
        return price
    worse = ticks * TICK
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


def window_label(bars: list[dict]) -> tuple[str, str]:
    if not bars:
        return "—", "—"
    start = datetime.fromtimestamp(bars[0]["open_time"] / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")
    end = datetime.fromtimestamp(bars[-1]["close_time"] / 1000, tz=UTC8).strftime("%Y-%m-%d %H:%M")
    return start, end
