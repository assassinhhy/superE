"""Intraday v2 backtest. Reads public klines and does not trade.

The first 60 UTC days are the only sample used to pick parameters. The next 30
are reported as out of sample. All 18 groups are written out.

    python scripts/board_backtest_v2.py --fixture data/board/backtest-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from supere.board.backtest import _group_v2, judge_intraday, summarize_v2, walk_v2  # noqa: E402
from supere.board.replay import load_fixture, replay_at  # noqa: E402
from supere.board.settings import load_analysis  # noqa: E402

UTC = timezone.utc
START = datetime(2026, 8, 16, tzinfo=UTC)
IN_END = START + timedelta(days=60)
OUT_END = START + timedelta(days=90)
GRID = [
    (b, t1, bars)
    for b in (0.35, 0.5)
    for t1 in (1.2, 1.5, 2.0)
    for bars in (4, 6, 8)
]
KINDS = ("顺势", "区间", "破位", "逆势")
SESSIONS = ("亚洲", "欧洲", "美国")


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _enable_all(cfg):
    cfg = cfg.model_copy(deep=True)
    cfg.board.profile = "intraday_v2"
    cfg.intraday.session_enabled.asia = True
    cfg.intraday.session_enabled.europe = True
    cfg.intraday.session_enabled.us = True
    return cfg


def _with_params(base, b: float, t1: float, bars: int):
    cfg = _enable_all(base)
    cfg.plans.stop_atr1h_factor = b
    cfg.intraday.t1_min_r = t1
    cfg.intraday.time_stop_bars = bars
    return cfg


def _bundle(trades: list[dict]) -> dict:
    return {
        "overall": summarize_v2(trades),
        "by_kind": _group_v2(trades, "kind"),
        "by_session": _group_v2(trades, "session"),
    }


def _pick(rows: list[dict]) -> dict:
    def key(row: dict):
        stats = row["bundle"]["overall"]
        distance = abs(row["b"] - 0.35) + abs(row["t1"] - 1.2) + abs(row["bars"] - 6) / 10
        return (stats["expectancy"], stats["counted"], -stats["max_drawdown_r"], -distance)

    return max(rows, key=key)


def _cell(stats: dict | None, field: str, digits: int = 3) -> str:
    if not stats:
        return "—"
    value = stats.get(field)
    if value is None:
        return "—"
    if field == "hit_rate" or field == "time_stop_pct":
        return f"{value:.1%}"
    if field == "payoff":
        return f"{value:.2f}"
    if field in {"counted", "max_loss_streak"}:
        return str(int(value))
    return f"{value:.{digits}f}"


def _csv_rows(group: str, split: str, bundle: dict) -> list[dict]:
    rows = []
    blocks = [("总体", "总体", bundle["overall"])]
    for name in KINDS:
        blocks.append(("类型", name, bundle["by_kind"].get(name)))
    for name in SESSIONS:
        blocks.append(("时段", name, bundle["by_session"].get(name)))
    for bucket_type, bucket, stats in blocks:
        stats = stats or summarize_v2([])
        rows.append(
            {
                "group": group,
                "split": split,
                "bucket_type": bucket_type,
                "bucket": bucket,
                "counted": stats["counted"],
                "hits": stats["hits"],
                "hit_rate": f"{stats['hit_rate']:.6f}",
                "expectancy_r": f"{stats['expectancy']:.6f}",
                "payoff": "" if stats["payoff"] is None else f"{stats['payoff']:.6f}",
                "max_loss_streak": stats["max_loss_streak"],
                "max_drawdown_r": f"{stats['max_drawdown_r']:.6f}",
                "avg_hold_min": f"{stats['avg_hold_min']:.4f}",
                "time_stop_pct": f"{stats['time_stop_pct']:.6f}",
            }
        )
    return rows


def _summary_line(label: str, stats: dict) -> str:
    return (
        f"| {label} | {stats['counted']} | {_cell(stats, 'hit_rate')} | {_cell(stats, 'expectancy')} | "
        f"{_cell(stats, 'payoff')} | {stats['max_loss_streak']} | {_cell(stats, 'max_drawdown_r', 2)} | "
        f"{_cell(stats, 'avg_hold_min', 1)} | {_cell(stats, 'time_stop_pct')} |"
    )


def _breakdown(bundle: dict) -> list[str]:
    lines = [
        "| 分组 | 笔数 | 命中率 | 期望 R | 盈亏比 | 最大连亏 | 最大回撤 R | 平均持仓 分钟 | 时间止损占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _summary_line("总体", bundle["overall"]),
    ]
    for name in KINDS:
        lines.append(_summary_line(name, bundle["by_kind"].get(name) or summarize_v2([])))
    for name in SESSIONS:
        lines.append(_summary_line(name, bundle["by_session"].get(name) or summarize_v2([])))
    return lines


def _replay_section(analysis) -> list[str]:
    from datetime import timezone as tz
    from datetime import timedelta as delta

    utc8 = tz(delta(hours=8))
    lines = [
        "## 2026-11-14 回放",
        "",
        "下面按规则把三个时段都打开来看当时的判断。某个时段如果期望为负，实盘面板默认不再出该时段的新计划。",
        "",
    ]
    cfg = _enable_all(analysis)
    for label, stamp in (("11:09", datetime(2026, 11, 14, 11, 9, tzinfo=utc8)), ("13:51", datetime(2026, 11, 14, 13, 51, tzinfo=utc8))):
        result = replay_at(ROOT / "tests" / "fixtures" / "board", int(stamp.timestamp() * 1000), cfg)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- 结论：{result['conclusion']['text']}")
        lines.append(
            f"- 方向：4h {result['trend']['frames'].get('4h')}，1h {result['trend']['frames'].get('1h')}，"
            f"{result['trend']['direction']}"
        )
        position = result.get("position") or {}
        lines.append(f"- 位置：{position.get('text') or '未获取'}；情绪：{result['sentiment'].get('label') or '未获取'}")
        if not result["plans"]:
            lines.append(f"- 计划：没有。{result['plans_meta'].get('pause_reason') or ''}")
        for plan in result["plans"]:
            if plan["kind"] == "翻转":
                lines.append(f"- 翻转：{plan['status']}。{plan.get('note') or ''}")
                continue
            t1 = plan.get("t1_r")
            t2 = plan.get("t2_r")
            risk = plan.get("r")
            lines.append(
                f"- {plan['kind']} {plan['title']}：{plan['status']}，"
                f"R {'—' if risk is None else f'{float(risk):.2f}'}，"
                f"T1 {'—' if t1 is None else f'{t1:.2f}R'}，"
                f"T2 {'—' if t2 is None else f'{t2:.2f}R'}，"
                f"最长持仓到 {plan.get('hold_until_utc8') or '—'}"
            )
        lines.append("")
    return lines


def render(rows: list[dict], chosen: dict, full: dict, gate: dict, analysis) -> str:
    lines = [
        "# 盘面分析 v2 回测（短线日内）",
        "",
        "- 数据：币安 U 本位 ETHUSDT 永续，data.binance.vision 日包。15m / 5m / 1h / 4h 从 2026-06-16 预热，正式样本是 2026-08-16 00:00 UTC 到 2026-11-14 00:00 UTC（90 天）。1d 另有约 250 天预热。11 月 14 日的日包当天还没有，窗口在 11 月 14 日 00:00 UTC 截止，不缺正式样本。",
        "- 成交：信号那根 15m 收盘确认后，下一根 15m 开盘进场。手续费 taker 0.05%×2，按整笔进场价扣一轮。滑点每边 1 tick（0.01），止损单再加 1 tick。",
        "- 记账：每笔期望用自身 R（扣费和滑点后的价格盈亏 ÷ 进场到止损的距离）。半仓和周末减半记在账户 R 上，日亏损上限和最大回撤用账户 R。",
        "- 调参只看前 60 天（到 2026-10-15 00:00 UTC）。后 30 天不参与挑选。18 组全部列出。",
        "- 选组规则：前 60 天期望最高；并列时笔数更多、回撤更小、更接近默认 b=0.35、T1=1.2R、时间止损 6 根。",
        "- 命中：扣费和滑点后自身 R > 0。盈亏比：盈利笔平均 R ÷ 亏损笔平均 R 的绝对值。同一根 K 同时碰到止损和目标，按止损。",
        "- 缺 OI 时，直接跟进和逆势单的 OI 条件视为不满足。同一时间只拿 1 笔。平仓那根 15m 不再开新仓。",
        "- 破位用的区间沿取信号 K 线之前的 12 根 15m，不把破位那根算进区间。算进去的话，收盘价不可能超出自己所在窗口的最低价，条件会永远不成立。",
        "",
        f"选定组：b={chosen['b']:.2f}，T1 门槛 {chosen['t1']:.1f}R，时间止损 {chosen['bars']} 根 15m。",
        "",
        "## 上线判定",
        "",
    ]
    if gate["passed"]:
        lines.append("达标。`notify.eligible = true`，系统通知仍然关闭，要手动打开。")
    else:
        lines.append("不达标。面板上的情景计划标注「未通过回测，仅供参考」，系统通知保持关闭。")
    lines.append("")
    for item in gate["failed"]:
        lines.append(f"- 没过：{item}")
    if not gate["failed"]:
        lines.append("- 六项都过了。")
    overall = full["all"]["overall"]
    lines.append(
        f"- 最大连亏 {overall['max_loss_streak']} 笔（上限 8），"
        f"{'过了' if overall['max_loss_streak'] <= 8 else '没过'}。"
    )
    lines.append(
        f"- 最大回撤 {overall['max_drawdown_r']:.2f}R（上限 15），"
        f"{'过了' if overall['max_drawdown_r'] <= 15 else '没过'}。"
    )
    lines.append("")
    lines.append("## 选定组：前 60 天 / 后 30 天 / 合计")
    lines.append("")
    lines.append("### 前 60 天")
    lines.append("")
    lines.extend(_breakdown(full["in"]))
    lines.append("")
    lines.append("### 后 30 天（样本外）")
    lines.append("")
    lines.extend(_breakdown(full["out"]))
    lines.append("")
    lines.append("### 90 天合计")
    lines.append("")
    lines.extend(_breakdown(full["all"]))
    lines.append("")
    lines.append("## 18 组参数（只在前 60 天）")
    lines.append("")
    lines.append("| 组 | b | T1 门槛 | 时间止损 | 笔数 | 命中率 | 期望 R | 盈亏比 | 最大连亏 | 最大回撤 R | 平均持仓 分钟 | 时间止损占比 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(rows, start=1):
        stats = row["bundle"]["overall"]
        mark = " ← 选定" if row is chosen else ""
        lines.append(
            f"| {index}{mark} | {row['b']:.2f} | {row['t1']:.1f}R | {row['bars']} | "
            f"{stats['counted']} | {_cell(stats, 'hit_rate')} | {_cell(stats, 'expectancy')} | {_cell(stats, 'payoff')} | "
            f"{stats['max_loss_streak']} | {_cell(stats, 'max_drawdown_r', 2)} | {_cell(stats, 'avg_hold_min', 1)} | "
            f"{_cell(stats, 'time_stop_pct')} |"
        )
    lines.append("")
    lines.append("## 18 组按单子类型（前 60 天）")
    lines.append("")
    lines.append("| 组 | b | T1 | 时间止损 | 类型 | 笔数 | 命中率 | 期望 R | 盈亏比 | 最大连亏 | 最大回撤 R | 平均持仓 分钟 | 时间止损占比 |")
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(rows, start=1):
        for name in KINDS:
            stats = row["bundle"]["by_kind"].get(name) or summarize_v2([])
            lines.append(
                f"| {index} | {row['b']:.2f} | {row['t1']:.1f} | {row['bars']} | {name} | "
                f"{stats['counted']} | {_cell(stats, 'hit_rate')} | {_cell(stats, 'expectancy')} | {_cell(stats, 'payoff')} | "
                f"{stats['max_loss_streak']} | {_cell(stats, 'max_drawdown_r', 2)} | {_cell(stats, 'avg_hold_min', 1)} | "
                f"{_cell(stats, 'time_stop_pct')} |"
            )
    lines.append("")
    lines.append("## 18 组按时段（前 60 天）")
    lines.append("")
    lines.append("| 组 | b | T1 | 时间止损 | 时段 | 笔数 | 命中率 | 期望 R | 盈亏比 | 最大连亏 | 最大回撤 R | 平均持仓 分钟 | 时间止损占比 |")
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(rows, start=1):
        for name in SESSIONS:
            stats = row["bundle"]["by_session"].get(name) or summarize_v2([])
            lines.append(
                f"| {index} | {row['b']:.2f} | {row['t1']:.1f} | {row['bars']} | {name} | "
                f"{stats['counted']} | {_cell(stats, 'hit_rate')} | {_cell(stats, 'expectancy')} | {_cell(stats, 'payoff')} | "
                f"{stats['max_loss_streak']} | {_cell(stats, 'max_drawdown_r', 2)} | {_cell(stats, 'avg_hold_min', 1)} | "
                f"{_cell(stats, 'time_stop_pct')} |"
            )
    lines.append("")
    negative = gate.get("disabled_sessions") or []
    lines.append("## 时段开关")
    lines.append("")
    if negative:
        lines.append("期望为负的时段已在 `analysis.toml` 里默认关掉：" + "、".join(negative) + "。`analysis.local.toml` 可以重新打开。")
    else:
        lines.append("选定组的 90 天合计里，没有期望为负且有样本的时段。三个时段都保持打开。")
    lines.append("")
    lines.extend(_replay_section(analysis))
    lines.append("只做分析和提醒，没有下单，没有连接交易账户。")
    lines.append("")
    return "\n".join(lines)


def _public_stats(stats: dict) -> dict:
    return {
        "counted": stats["counted"],
        "hits": stats["hits"],
        "hit_rate": stats["hit_rate"],
        "expectancy": stats["expectancy"],
        "payoff": stats["payoff"],
        "max_loss_streak": stats["max_loss_streak"],
        "max_drawdown_r": stats["max_drawdown_r"],
        "avg_hold_min": stats["avg_hold_min"],
        "time_stop_pct": stats["time_stop_pct"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盘面 v2 回测。不下单。")
    parser.add_argument("--fixture", default=str(ROOT / "data/board/backtest-cache"))
    parser.add_argument("--analysis", default=str(ROOT / "analysis.toml"))
    parser.add_argument("--out", default=str(ROOT / "docs/board-backtest-v2.md"))
    args = parser.parse_args(argv)
    fixture = load_fixture(Path(args.fixture))
    bars = fixture["frames"].get("15m") or []
    if len(bars) < 1000:
        print(f"15m 只有 {len(bars)} 根。", file=sys.stderr)
        return 2
    base, _path = load_analysis(args.analysis)
    start_ms, in_ms, out_ms = _ms(START), _ms(IN_END), _ms(OUT_END)
    rows = []
    for b, t1, bars_n in GRID:
        print(f"in-sample b={b} t1={t1} time={bars_n}", flush=True)
        cfg = _with_params(base, b, t1, bars_n)
        trades = walk_v2(fixture, cfg, start_ms=start_ms, end_ms=in_ms)
        rows.append({"b": b, "t1": t1, "bars": bars_n, "bundle": _bundle(trades), "trades": trades})
    chosen = _pick(rows)
    print(f"selected b={chosen['b']} t1={chosen['t1']} bars={chosen['bars']}", flush=True)
    full_cfg = _with_params(base, chosen["b"], chosen["t1"], chosen["bars"])
    full_trades = walk_v2(fixture, full_cfg, start_ms=start_ms, end_ms=out_ms)
    inside = [trade for trade in full_trades if trade["entry_ms"] < in_ms]
    outside = [trade for trade in full_trades if trade["entry_ms"] >= in_ms]
    full = {"in": _bundle(inside), "out": _bundle(outside), "all": _bundle(full_trades)}
    gate = judge_intraday(full["in"]["overall"], full["out"]["overall"], full["all"]["overall"])
    disabled = []
    names = {"亚洲": "asia", "欧洲": "europe", "美国": "us"}
    for label, key in names.items():
        stats = full["all"]["by_session"].get(label) or summarize_v2([])
        if stats["counted"] and stats["expectancy"] < 0:
            disabled.append(label)
            setattr(base.intraday.session_enabled, key, False)
    by_kind = {}
    for name in KINDS:
        stats = full["all"]["by_kind"].get(name)
        if stats and stats["counted"]:
            by_kind[name] = {"expectancy": stats["expectancy"], "counted": stats["counted"]}
    by_session = {}
    for label in SESSIONS:
        stats = full["all"]["by_session"].get(label) or summarize_v2([])
        by_session[label] = {"expectancy": stats["expectancy"], "counted": stats["counted"]}
    payload = {
        "passed": gate["passed"],
        "eligible": gate["passed"],
        "failed": gate["failed"],
        "label": "" if gate["passed"] else "未通过回测，仅供参考",
        "selected": {"b": chosen["b"], "t1_min_r": chosen["t1"], "time_stop_bars": chosen["bars"]},
        "by_kind": by_kind,
        "by_session": by_session,
        "disabled_sessions": disabled,
        "in_sample": _public_stats(full["in"]["overall"]),
        "out_of_sample": _public_stats(full["out"]["overall"]),
        "overall": _public_stats(full["all"]["overall"]),
    }
    gate_path = ROOT / "backend/supere/board/gate_v2.py"
    literal = json.dumps(payload, ensure_ascii=False, indent=2)
    literal = literal.replace("true", "True").replace("false", "False").replace("null", "None")
    gate_path.write_text('"""v2 backtest gate. Rewritten by scripts/board_backtest_v2.py."""\n\nGATE = ' + literal + "\n", encoding="utf-8")
    text = Path(args.analysis).read_text(encoding="utf-8")
    for key, value in (
        ("asia", base.intraday.session_enabled.asia),
        ("europe", base.intraday.session_enabled.europe),
        ("us", base.intraday.session_enabled.us),
    ):
        text = re.sub(
            rf"(?m)^({key} = )(?:true|false)$",
            rf"\g<1>{'true' if value else 'false'}",
            text,
            count=1,
        )
    # Live defaults follow the group chosen on the first 60 days.
    for key, value in (
        ("stop_atr1h_factor", f"{chosen['b']:.2f}"),
        ("t1_min_r", f"{chosen['t1']:.1f}"),
        ("time_stop_bars", str(chosen["bars"])),
    ):
        text = re.sub(rf"(?m)^({key} = ).*$", rf"\g<1>{value}", text, count=1)
    Path(args.analysis).write_text(text, encoding="utf-8")
    out = Path(args.out)
    out.write_text(render(rows, chosen, full, payload, base), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    csv_rows = []
    for index, row in enumerate(rows, start=1):
        group = f"{index}:b={row['b']:.2f},t1={row['t1']:.1f},time={row['bars']}"
        csv_rows.extend(_csv_rows(group, "前60天", row["bundle"]))
    csv_rows.extend(_csv_rows("选定", "前60天", full["in"]))
    csv_rows.extend(_csv_rows("选定", "后30天", full["out"]))
    csv_rows.extend(_csv_rows("选定", "90天合计", full["all"]))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    trade_path = out.with_name("board-backtest-v2-trades.csv")
    with trade_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["entry_ms", "kind", "side", "session", "outcome", "exit_reason", "own_r", "account_r", "hold_min", "size_ratio", "weekend"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trade in full_trades:
            writer.writerow({key: trade.get(key) for key in fields})
    print(out.read_text(encoding="utf-8"))
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
