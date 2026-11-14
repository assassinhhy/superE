"""One pure ``analyze(snapshot, params)`` used by the live panel, replay and backtest."""

from __future__ import annotations

from supere.board.indicators import last_value, pack_bars
from supere.board.levels import build_levels
from supere.board.plans import build_plans
from supere.board.range_ import build_range
from supere.board.risk import assess
from supere.board.sentiment import aligned_oi, crowding, funding_heat, quadrant, taker_pressure, volume_spike
from supere.board.settings import AnalysisConfig
from supere.board.trend import daily_rsi_tag, frame_view, macd_fading, synthesize
from supere.timeutil import format_utc8

DISCLAIMER = "概率判断，仅供参考，不构成投资建议。"


def _usable(series: dict | None) -> bool:
    return bool(series) and series.get("status") == "ok" and series.get("bars")


def _pack_map(snapshot: dict, analysis: AnalysisConfig) -> dict:
    ind = analysis.indicators
    packs = {}
    for interval, series in snapshot.get("klines", {}).items():
        if not _usable(series):
            continue
        if len(series["bars"]) < ind.rsi + 2:
            continue
        packs[interval] = pack_bars(
            series["bars"],
            ema_periods=ind.ema,
            rsi_period=ind.rsi,
            macd=ind.macd,
            atr_period=ind.atr,
        )
    return packs


def _hist_state(hist: list[float | None]) -> str:
    values = [value for value in hist if value is not None]
    if len(values) < 2:
        return ""
    prev, last = values[-2], values[-1]
    if prev <= 0 < last:
        return "翻正"
    if prev >= 0 > last:
        return "翻负"
    if abs(last) > abs(prev):
        return "扩大"
    if abs(last) < abs(prev):
        return "收窄"
    return "持平"


def _vs_ema(close: float, pack: dict) -> str:
    parts = []
    for period in (20, 50, 200):
        value = last_value(pack["ema"].get(period, []))
        if value is None:
            parts.append("—")
        else:
            parts.append("上" if close > value else "下")
    return "/".join(parts)


def _technical(packs: dict, analysis: AnalysisConfig, fade: bool) -> dict:
    rows = []
    for frame in ("5m", "15m", "1h", "4h", "1d"):
        pack = packs.get(frame)
        if not pack:
            rows.append({"frame": frame, "status": "missing", "text": "未获取"})
            continue
        rsi = last_value(pack["rsi"])
        hist = [value for value in pack["hist"] if value is not None]
        rows.append(
            {
                "frame": frame,
                "status": "ok",
                "rsi": rsi,
                "hist": hist[-1] if hist else None,
                "hist_prev": hist[-2] if len(hist) > 1 else None,
                "hist_state": _hist_state(pack["hist"]),
                "vs_ema": _vs_ema(pack["closes"][-1], pack),
                "warn": "大级别动能衰减" if frame == "4h" and fade else "",
            }
        )
    return {"rows": rows}


def _sentiment(snapshot: dict, packs: dict, analysis: AnalysisConfig) -> dict:
    params = analysis.sentiment
    missing = {
        "status": "missing",
        "text": "未获取",
        "label": "",
        "dp_pct": None,
        "doi_pct": None,
        "lines": ["未获取"],
        "funding": {"status": "missing", "text": "未获取"},
        "global_lsr": None,
        "top_pos_lsr": None,
        "taker": None,
    }
    metrics = snapshot.get("metrics") or {}
    if metrics.get("status") != "ok":
        missing["funding"] = _funding_line(snapshot, analysis)
        return missing
    pack = packs.get(params.frame)
    if not pack:
        return missing
    bars = pack["bars"]
    look = params.lookback_bars
    if len(bars) <= look:
        return missing
    close_now = bars[-1]["close"]
    close_then = bars[-1 - look]["close"]
    if not close_then:
        return missing
    dp = (close_now - close_then) / close_then * 100
    at = bars[-1]["close_time"]
    span = bars[-1]["close_time"] - bars[-1 - look]["close_time"]
    now_oi = aligned_oi(metrics.get("rows") or [], at)
    then_oi = aligned_oi(metrics.get("rows") or [], at - span)
    if not now_oi or not then_oi or not then_oi.get("oi"):
        missing["text"] = "未获取"
        return missing
    doi = (now_oi["oi"] - then_oi["oi"]) / then_oi["oi"] * 100
    label = quadrant(dp, doi, params.dp_pct, params.doi_pct)
    latest = metrics["rows"][-1]
    crowd = crowding(latest.get("global_lsr"), params.global_lsr_crowded_long, params.global_lsr_crowded_short)
    lines = [f"OI {doi:+.1f}%，价格 {dp:+.1f}% → {label}"]
    if crowd:
        lines.append(crowd)
    taker = _taker_line(bars[-1], params)
    spikes = _spikes(packs, params)
    return {
        "status": "ok",
        "text": lines[0],
        "label": label,
        "dp_pct": dp,
        "doi_pct": doi,
        "lines": lines + spikes,
        "funding": _funding_line(snapshot, analysis),
        "global_lsr": latest.get("global_lsr"),
        "top_pos_lsr": latest.get("top_pos_lsr"),
        "taker": taker,
        "crowd": crowd,
    }


def _funding_line(snapshot: dict, analysis: AnalysisConfig) -> dict:
    funding = snapshot.get("funding") or {}
    if funding.get("status") != "ok" or not isinstance(funding.get("value"), dict):
        return {"status": funding.get("status") or "missing", "text": "未获取", "rate": None}
    rate = funding["value"].get("rate")
    heat = funding_heat(rate, analysis.sentiment.funding_hot_abs, analysis.sentiment.funding_neutral_abs)
    text = "未获取" if rate is None else f"费率 {rate * 100:.4f}%"
    if heat:
        text += f" {heat}"
    return {
        "status": "ok",
        "text": text,
        "rate": rate,
        "heat": heat,
        "next_funding_time": funding["value"].get("next_funding_time"),
        "utc8": funding.get("utc8"),
        "source": funding.get("source"),
    }


def _taker_ratio(bar: dict) -> float | None:
    sell = bar["volume"] - bar.get("taker_buy", 0)
    if sell <= 0:
        return None
    return bar.get("taker_buy", 0) / sell


def _taker_line(bar: dict, params) -> dict | None:
    ratio = _taker_ratio(bar)
    if ratio is None:
        return None
    return {
        "ratio": ratio,
        "pressure": taker_pressure(ratio, params.taker_sell, params.taker_buy),
        "utc8": format_utc8(bar["close_time"], "%H:%M:%S"),
        "source": "15m K 线",
    }


def _spikes(packs: dict, params) -> list[str]:
    notes = []
    for frame in ("5m", "15m"):
        pack = packs.get(frame)
        if not pack:
            continue
        bars = pack["bars"]
        volumes = [bar["volume"] for bar in bars]
        # Mark every spike in the last 16 closed bars, compared with the 20 before it.
        start = max(params.vol_spike_window, len(bars) - 16)
        for index in range(start, len(bars)):
            if not volume_spike(volumes[: index + 1], params.vol_spike_mult, params.vol_spike_window):
                continue
            bar = bars[index]
            ratio = _taker_ratio(bar)
            pressure = taker_pressure(ratio, params.taker_sell, params.taker_buy)
            mult = bar["volume"] / (sum(volumes[index - params.vol_spike_window:index]) / params.vol_spike_window)
            text = f"{format_utc8(bar['open_time'], '%H:%M')} 放量{'阴线' if bar['close'] < bar['open'] else '阳线'}（量 {mult:.1f}×"
            if ratio is not None:
                text += f"，主动买卖比 {ratio:.2f}"
            text += "）"
            if pressure:
                text += pressure
            text += f"  {frame} K 线"
            notes.append(text)
    return notes[-4:]


def _chase(side: str | None, levels: dict, bars: list[dict]) -> bool:
    if side not in {"short", "long"} or not bars:
        return False
    pins = [level for level in levels.get("all") or [] if "插针扫过" in level.get("sources", [])]
    recent = bars[-8:]
    last = bars[-1]["close"]
    for level in pins:
        price = level["price"]
        if side == "short" and any(bar["low"] < price for bar in recent) and last > price:
            return True
        if side == "long" and any(bar["high"] > price for bar in recent) and last < price:
            return True
    return False


def _conclusion(trend: dict, rng: dict | None, chase: bool, side: str | None) -> dict:
    parts = []
    if trend["direction"] == "未获取":
        parts.append("方向未获取")
    elif rng and rng.get("blocked"):
        parts.append("位置不好，不开单")
    elif trend["bias"] == "只做空":
        parts.append("空头：只做空")
    elif trend["bias"] == "偏空":
        parts.append("偏空：反抽找空")
    elif trend["bias"] == "只做多":
        parts.append("多头：只做多")
    elif trend["bias"] == "偏多":
        parts.append("偏多：回踩找多")
    else:
        parts.append(trend["bias"])
    if chase and side == "short":
        parts.append("不在插针里追空")
    elif chase and side == "long":
        parts.append("不在插针里追多")
    if trend.get("daily_rsi") and trend["daily_rsi"].get("tag"):
        parts.append(trend["daily_rsi"]["tag"])
    if trend.get("macd_fade"):
        parts.append("大级别动能衰减")
    return {"text": "；".join(parts), "do_not_chase": chase}


def _bullets(sentiment: dict, rng: dict | None, packs: dict) -> list[dict]:
    bullets = []
    for line in sentiment.get("lines") or []:
        if line == "未获取":
            continue
        if line.startswith("OI") or "拥挤" in line:
            continue
        bullets.append({"text": line, "source": "K 线"})
    if rng:
        for item in rng.get("structure") or []:
            bullets.append({"text": item, "source": "15m K 线"})
        if rng.get("shrink_text"):
            bullets.append({"text": rng["shrink_text"], "source": "15m K 线"})
    return bullets


def analyze(snapshot: dict, analysis: AnalysisConfig) -> dict:
    packs = _pack_map(snapshot, analysis)
    h4 = frame_view(packs.get("4h"), analysis.trend.tangle_pct) if _usable(snapshot.get("klines", {}).get("4h")) else None
    h1 = frame_view(packs.get("1h"), analysis.trend.tangle_pct) if _usable(snapshot.get("klines", {}).get("1h")) else None
    if h4 is None or h1 is None:
        bias, direction = "未获取", "未获取"
        fade = False
    else:
        bias, direction = synthesize(h4, h1)
        fade = False
        if packs.get("4h"):
            fade = macd_fading(
                packs["4h"]["hist"],
                bullish=(h4 == "多"),
                bars=analysis.trend.macd_fade_bars,
            )
    daily = None
    if packs.get("1d"):
        tag = daily_rsi_tag(
            last_value(packs["1d"]["rsi"]),
            overbought=analysis.trend.daily_rsi_ob,
            oversold=analysis.trend.daily_rsi_os,
        )
        daily = {"value": last_value(packs["1d"]["rsi"]), "tag": tag}
    trend = {
        "frames": {"4h": h4, "1h": h1},
        "bias": bias,
        "direction": direction,
        "daily_rsi": daily,
        "macd_fade": fade,
        "text": "未获取" if direction == "未获取" else f"4h {h4} · 1h {h1}",
    }
    price_block = snapshot.get("price") or {}
    price = price_block.get("value") if price_block.get("status") == "ok" else None
    bars15 = packs["15m"]["bars"] if packs.get("15m") else []
    rng = build_range(bars15, price, analysis.range) if bars15 else None
    sentiment = _sentiment(snapshot, packs, analysis)
    atr15 = last_value(packs["15m"]["atr"]) if packs.get("15m") else None
    levels = build_levels(
        bars_15=bars15,
        bars_5=packs["5m"]["bars"] if packs.get("5m") else [],
        packs=packs,
        price=price,
        now_ms=int(snapshot.get("now_ms") or 0),
        atr=atr15,
        params=analysis.levels,
    )
    plans, meta = build_plans(
        bias=bias,
        price=price,
        levels=levels,
        rng=rng,
        packs=packs,
        analysis=analysis,
    )
    if rng and rng.get("blocked"):
        meta["position_blocked"] = True
        for plan in plans:
            if plan["status"] == "可进场":
                plan["status"] = "等待信号"
    cross = snapshot.get("crosscheck") or {}
    if cross.get("pause_plans"):
        meta["paused"] = True
        meta["pause_reason"] = cross.get("message") or "数据异常"
        for plan in plans:
            if plan["status"] == "可进场":
                plan["status"] = "暂停"
    side = "short" if direction == "空头" else "long" if direction == "多头" else None
    chase = _chase(side, levels, bars15)
    conclusion = _conclusion(trend, rng, chase, side)
    if sentiment["status"] != "ok" and any(word in conclusion["text"] for word in ("爆仓", "堆仓", "新空", "新多", "回补")):
        conclusion["text"] = conclusion["text"]
    risk = assess(
        int(snapshot.get("now_ms") or 0),
        snapshot.get("events") or [],
        snapshot.get("calendar") or [],
        analysis.risk,
    )
    meta["generated_utc8"] = format_utc8(int(snapshot.get("now_ms") or 0), "%H:%M")
    return {
        "symbol": snapshot.get("symbol") or analysis.board.symbol,
        "as_of": snapshot.get("now_ms"),
        "as_of_utc8": snapshot.get("now_utc8") or format_utc8(int(snapshot.get("now_ms") or 0)),
        "price": price_block,
        "ticker": snapshot.get("ticker"),
        "crosscheck": cross,
        "limit": snapshot.get("limit") or {"active": False, "message": ""},
        "trend": trend,
        "position": None if rng is None else {
            "zone": rng["position"],
            "text": rng["position_text"],
            "blocked": rng["blocked"],
        },
        "range": rng,
        "sentiment": sentiment,
        "technical": _technical(packs, analysis, fade),
        "levels": {"resistance": levels["resistance"], "support": levels["support"]},
        "level_book": levels["all"],
        "plans": plans,
        "plans_meta": meta,
        "calc_defaults": analysis.calc.model_dump(),
        "risk": risk,
        "conclusion": conclusion,
        "bullets": _bullets(sentiment, rng, packs),
        "atr": {
            "15m": atr15,
            "1h": last_value(packs["1h"]["atr"]) if packs.get("1h") else None,
        },
        "disclaimer": DISCLAIMER,
    }
