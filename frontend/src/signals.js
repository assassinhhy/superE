// Closed-bar signals for ETHUSDT perpetual. No orders, no forming-bar decisions.
import { atr } from "./indicators.js";

export const SIGNAL_INTERVALS = ["5m", "15m", "1h", "4h"];
export const SIGNAL_GROUPS = ["全部", "RSI", "MACD", "均线", "布林", "ATR", "背离"];

const META = {
  rsi_enter_ob: { group: "RSI", label: "RSI 进入超买", tone: "down" },
  rsi_exit_ob: { group: "RSI", label: "RSI 离开超买", tone: "up" },
  rsi_enter_os: { group: "RSI", label: "RSI 进入超卖", tone: "down" },
  rsi_exit_os: { group: "RSI", label: "RSI 离开超卖", tone: "up" },
  macd_golden: { group: "MACD", label: "MACD 金叉", tone: "up" },
  macd_death: { group: "MACD", label: "MACD 死叉", tone: "down" },
  px_up_ma20: { group: "均线", label: "价格上穿 MA20", tone: "up" },
  px_dn_ma20: { group: "均线", label: "价格下穿 MA20", tone: "down" },
  px_up_ma50: { group: "均线", label: "价格上穿 MA50", tone: "up" },
  px_dn_ma50: { group: "均线", label: "价格下穿 MA50", tone: "down" },
  ma_golden: { group: "均线", label: "MA20 上穿 MA50", tone: "up" },
  ma_death: { group: "均线", label: "MA20 下穿 MA50", tone: "down" },
  boll_up: { group: "布林", label: "突破布林上轨", tone: "up" },
  boll_dn: { group: "布林", label: "跌破布林下轨", tone: "down" },
  atr_expand: { group: "ATR", label: "ATR 突然放大", tone: "warn" },
  rsi_div_bull: { group: "背离", label: "RSI 底背离", tone: "up" },
  rsi_div_bear: { group: "背离", label: "RSI 顶背离", tone: "down" },
};

const COOLDOWN_MS = {
  "5m": 10 * 60 * 1000,
  "15m": 30 * 60 * 1000,
  "1h": 2 * 60 * 60 * 1000,
  "4h": 8 * 60 * 60 * 1000,
};

const ATR_JUMP = 1.8;
const PIVOT_WING = 3;

export function cooldownMs(interval) {
  return COOLDOWN_MS[interval] || 15 * 60 * 1000;
}

function finite(value) {
  return Number.isFinite(value);
}

function smaSeries(values, period) {
  const out = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (!finite(value)) {
      sum = 0;
      continue;
    }
    sum += value;
    if (i >= period - 1 && values.slice(i - period + 1, i + 1).every(finite)) {
      out[i] = sum / period;
      sum -= values[i - (period - 1)];
    }
  }
  return out;
}

/** Same RSI(14) seed and Wilder smoothing as KLineChart. */
export function rsiSeries(closes, period = 14) {
  const out = new Array(closes.length).fill(null);
  let gainSum = 0;
  let lossSum = 0;
  let avgGain;
  let avgLoss;
  for (let i = 0; i < closes.length; i += 1) {
    const change = i === 0 ? 0 : closes[i] - closes[i - 1];
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    gainSum += gain;
    lossSum += loss;
    if (i < period) continue;
    if (avgGain === undefined) {
      avgGain = gainSum / period;
      avgLoss = lossSum / period;
    } else {
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    if (avgLoss === 0) out[i] = 100;
    else if (avgGain === 0) out[i] = 0;
    else out[i] = 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

/** MACD 12/26/9, same first-SMA seed as KLineChart. */
export function macdSeries(closes, short = 12, long = 26, signal = 9) {
  const dif = new Array(closes.length).fill(null);
  const dea = new Array(closes.length).fill(null);
  let closeSum = 0;
  let emaShort = 0;
  let emaLong = 0;
  let difSum = 0;
  let deaValue = 0;
  const maxPeriod = Math.max(short, long);
  for (let i = 0; i < closes.length; i += 1) {
    const close = closes[i];
    closeSum += close;
    if (i >= short - 1) {
      emaShort = i > short - 1 ? (2 * close + (short - 1) * emaShort) / (short + 1) : closeSum / short;
    }
    if (i >= long - 1) {
      emaLong = i > long - 1 ? (2 * close + (long - 1) * emaLong) / (long + 1) : closeSum / long;
    }
    if (i >= maxPeriod - 1) {
      const difValue = emaShort - emaLong;
      dif[i] = difValue;
      difSum += difValue;
      if (i >= maxPeriod + signal - 2) {
        deaValue = i > maxPeriod + signal - 2
          ? (difValue * 2 + deaValue * (signal - 1)) / (signal + 1)
          : difSum / signal;
        dea[i] = deaValue;
      }
    }
  }
  return { dif, dea };
}

function bollAt(closes, index, period = 20, mult = 2) {
  if (index < period - 1) return null;
  let sum = 0;
  for (let i = index - period + 1; i <= index; i += 1) sum += closes[i];
  const mid = sum / period;
  let squares = 0;
  for (let i = index - period + 1; i <= index; i += 1) {
    const delta = closes[i] - mid;
    squares += delta * delta;
  }
  const width = mult * Math.sqrt(squares / period);
  return { mid, up: mid + width, dn: mid - width };
}

function smaNullable(values, period) {
  const out = new Array(values.length).fill(null);
  for (let i = period - 1; i < values.length; i += 1) {
    let sum = 0;
    let ok = true;
    for (let j = i - period + 1; j <= i; j += 1) {
      if (!finite(values[j])) {
        ok = false;
        break;
      }
      sum += values[j];
    }
    if (ok) out[i] = sum / period;
  }
  return out;
}

export function packBars(bars) {
  const closes = bars.map((bar) => bar.close);
  const rsi = rsiSeries(closes, 14);
  const macd = macdSeries(closes);
  const ma20 = smaSeries(closes, 20);
  const ma50 = smaSeries(closes, 50);
  const boll = closes.map((_, index) => bollAt(closes, index));
  const atrValues = atr(bars, 14);
  const atrSma = smaNullable(atrValues, 20);
  const atrRatio = atrValues.map((value, index) => (
    finite(value) && finite(atrSma[index]) && atrSma[index] > 0 ? value / atrSma[index] : null
  ));
  return { rsi, macd, ma20, ma50, boll, atrRatio };
}

function crossedUp(prevPrice, price, prevLevel, level) {
  return [prevPrice, price, prevLevel, level].every(finite) && prevPrice <= prevLevel && price > level;
}

function crossedDown(prevPrice, price, prevLevel, level) {
  return [prevPrice, price, prevLevel, level].every(finite) && prevPrice >= prevLevel && price < level;
}

function isPivot(bars, index, field) {
  const value = bars[index]?.[field];
  if (!finite(value)) return false;
  for (let j = index - PIVOT_WING; j <= index + PIVOT_WING; j += 1) {
    if (j === index) continue;
    if (j < 0 || j >= bars.length || !finite(bars[j][field])) return false;
    if (field === "low" && bars[j][field] < value) return false;
    if (field === "high" && bars[j][field] > value) return false;
  }
  return true;
}

function previousPivot(bars, before, field) {
  const start = Math.max(PIVOT_WING, before - 40);
  for (let index = before - 1; index >= start; index -= 1) {
    if (isPivot(bars, index, field)) return index;
  }
  return -1;
}

function divergenceAt(bars, rsi, index) {
  const pivot = index - PIVOT_WING;
  if (pivot < PIVOT_WING) return "";
  if (isPivot(bars, pivot, "low")) {
    const prev = previousPivot(bars, pivot, "low");
    if (prev >= 0 && finite(rsi[pivot]) && finite(rsi[prev])
      && bars[pivot].low < bars[prev].low && rsi[pivot] > rsi[prev]) {
      return "rsi_div_bull";
    }
  }
  if (isPivot(bars, pivot, "high")) {
    const prev = previousPivot(bars, pivot, "high");
    if (prev >= 0 && finite(rsi[pivot]) && finite(rsi[prev])
      && bars[pivot].high > bars[prev].high && rsi[pivot] < rsi[prev]) {
      return "rsi_div_bear";
    }
  }
  return "";
}

export function signalsAt(bars, index, pack, { divergence = false } = {}) {
  if (index < 1) return [];
  const bar = bars[index];
  const prev = bars[index - 1];
  const found = [];
  const push = (type) => {
    const meta = META[type];
    found.push({
      type,
      group: meta.group,
      label: meta.label,
      tone: meta.tone,
      timeMs: bar.closeTime,
      openTime: bar.timestamp,
      price: bar.close,
    });
  };
  const rsi = pack.rsi;
  if (finite(rsi[index]) && finite(rsi[index - 1])) {
    const before = rsi[index - 1];
    const now = rsi[index];
    if (before < 70 && now >= 70) push("rsi_enter_ob");
    if (before >= 70 && now < 70) push("rsi_exit_ob");
    if (before > 30 && now <= 30) push("rsi_enter_os");
    if (before <= 30 && now > 30) push("rsi_exit_os");
  }
  const { dif, dea } = pack.macd;
  if ([dif[index], dif[index - 1], dea[index], dea[index - 1]].every(finite)) {
    if (dif[index - 1] <= dea[index - 1] && dif[index] > dea[index]) push("macd_golden");
    if (dif[index - 1] >= dea[index - 1] && dif[index] < dea[index]) push("macd_death");
  }
  if (crossedUp(prev.close, bar.close, pack.ma20[index - 1], pack.ma20[index])) push("px_up_ma20");
  if (crossedDown(prev.close, bar.close, pack.ma20[index - 1], pack.ma20[index])) push("px_dn_ma20");
  if (crossedUp(prev.close, bar.close, pack.ma50[index - 1], pack.ma50[index])) push("px_up_ma50");
  if (crossedDown(prev.close, bar.close, pack.ma50[index - 1], pack.ma50[index])) push("px_dn_ma50");
  if (crossedUp(pack.ma20[index - 1], pack.ma20[index], pack.ma50[index - 1], pack.ma50[index])) push("ma_golden");
  if (crossedDown(pack.ma20[index - 1], pack.ma20[index], pack.ma50[index - 1], pack.ma50[index])) push("ma_death");
  const upper = pack.boll[index];
  const upperPrev = pack.boll[index - 1];
  if (upper && upperPrev) {
    if (crossedUp(prev.close, bar.close, upperPrev.up, upper.up)) push("boll_up");
    if (crossedDown(prev.close, bar.close, upperPrev.dn, upper.dn)) push("boll_dn");
  }
  const ratio = pack.atrRatio;
  if (finite(ratio[index]) && finite(ratio[index - 1]) && ratio[index - 1] < ATR_JUMP && ratio[index] >= ATR_JUMP) {
    push("atr_expand");
  }
  if (divergence) {
    const kind = divergenceAt(bars, rsi, index);
    if (kind) push(kind);
  }
  return found;
}

export function emptySignalState() {
  return { items: [], announced: [], lastFire: {}, seenOpen: {} };
}

export function signalId(interval, sig) {
  return `${interval}:${sig.type}:${sig.openTime}`;
}

function known(state, id) {
  return state.items.some((item) => item.id === id);
}

export function ingestInterval(state, interval, bars, nowMs, { divergence = false } = {}) {
  const closed = (bars || [])
    .filter((bar) => finite(bar.closeTime) && bar.closeTime <= nowMs && finite(bar.timestamp))
    .slice()
    .sort((a, b) => a.timestamp - b.timestamp);
  const next = {
    items: state.items.slice(),
    announced: state.announced.slice(),
    lastFire: { ...state.lastFire },
    seenOpen: { ...state.seenOpen },
  };
  if (closed.length < 2) return { state: next, announce: [] };
  const lastTs = closed[closed.length - 1].timestamp;
  const prevTs = state.seenOpen[interval];
  const first = prevTs == null;
  let from = 1;
  if (!first) {
    const idx = closed.findIndex((bar) => bar.timestamp === prevTs);
    from = idx < 0 ? closed.length - 1 : idx + 1;
    if (from >= closed.length) return { state: next, announce: [] };
  } else {
    from = 1;
  }
  const pack = packBars(closed);
  const accepted = [];
  for (let i = from; i < closed.length; i += 1) {
    for (const sig of signalsAt(closed, i, pack, { divergence })) {
      const id = signalId(interval, sig);
      const key = `${interval}:${sig.type}`;
      const fired = next.lastFire[key] || 0;
      if (known(next, id)) continue;
      if (fired && sig.timeMs - fired < cooldownMs(interval)) continue;
      const item = { id, interval, ...sig };
      next.lastFire[key] = sig.timeMs;
      next.items.unshift(item);
      accepted.push(item);
    }
  }
  next.items = next.items.slice(0, 200);
  next.seenOpen[interval] = lastTs;
  const announce = [];
  if (first) {
    if (accepted.length) announce.push(accepted[accepted.length - 1]);
    for (const item of accepted) {
      if (!next.announced.includes(item.id)) next.announced.push(item.id);
    }
  } else {
    for (const item of accepted) {
      if (!next.announced.includes(item.id)) {
        announce.push(item);
        next.announced.push(item.id);
      }
    }
  }
  next.announced = next.announced.slice(-500);
  return { state: next, announce };
}

export function filterSignals(items, interval, group) {
  return (items || []).filter((item) => (
    (interval === "全部" || item.interval === interval)
    && (group === "全部" || item.group === group)
  ));
}

const STORAGE_KEY = "supere.signals.v1";

export function loadSignalState(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem?.(STORAGE_KEY);
    if (!raw) return emptySignalState();
    const parsed = JSON.parse(raw);
    const base = emptySignalState();
    base.items = Array.isArray(parsed.items) ? parsed.items : [];
    base.announced = Array.isArray(parsed.announced) ? parsed.announced : [];
    base.lastFire = parsed.lastFire && typeof parsed.lastFire === "object" ? parsed.lastFire : {};
    base.seenOpen = parsed.seenOpen && typeof parsed.seenOpen === "object" ? parsed.seenOpen : {};
    return base;
  } catch {
    return emptySignalState();
  }
}

export function saveSignalState(state, storage = globalThis.localStorage) {
  try {
    storage?.setItem?.(STORAGE_KEY, JSON.stringify({
      items: state.items,
      announced: state.announced,
      lastFire: state.lastFire,
      seenOpen: state.seenOpen,
    }));
  } catch {
    /* private mode */
  }
}
