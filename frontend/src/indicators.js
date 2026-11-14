/** Chart-only formulas. Model features stay on the backend in phase 2. */

export const INTERVALS = [
  { id: "1m", span: 1, type: "minute" },
  { id: "3m", span: 3, type: "minute" },
  { id: "5m", span: 5, type: "minute" },
  { id: "15m", span: 15, type: "minute" },
  { id: "30m", span: 30, type: "minute" },
  { id: "1h", span: 1, type: "hour" },
  { id: "2h", span: 2, type: "hour" },
  { id: "4h", span: 4, type: "hour" },
  { id: "6h", span: 6, type: "hour" },
  { id: "8h", span: 8, type: "hour" },
  { id: "12h", span: 12, type: "hour" },
  { id: "1d", span: 1, type: "day" },
  { id: "3d", span: 3, type: "day" },
  { id: "1w", span: 1, type: "week" },
  { id: "1M", span: 1, type: "month" },
];

/** Prediction stops at 4h (2026-11-05). Longer K-line intervals stay for charting only. */
export const PREDICT_INTERVALS = ["5m", "15m", "30m", "1h", "4h"];

/** Each period is its own main-pane instance. KLineChart's built-in MA would draw them as one indicator. */
export const MA_LINES = [
  { id: "MA20", period: 20, color: "#d7a15f" },
  { id: "MA50", period: 50, color: "#7eb6d9" },
];

/** Sub panes keep this height. The chart body grows and the pane scrolls instead of crushing them. */
export const SUB_PANE_PX = 156;
/** Floor for the candle pane. Above this, the candle fills whatever the first screen has left. */
export const MAIN_PANE_PX = 340;
export const AXIS_PX = 36;
/** The only sub pane that sits on the first screen. Every other sub pane is reached by scrolling. */
export const PRIMARY_SUB = "MACD";

export function chartBodyHeight(subCount, { separator = 1, axis = AXIS_PX, mainPx = MAIN_PANE_PX } = {}) {
  const subs = Math.max(0, subCount);
  return mainPx + subs * SUB_PANE_PX + subs * separator + axis;
}

/**
 * Candle pane height that makes candle + MACD exactly fill the visible viewport.
 * Without MACD the candle takes the whole viewport. Never below MAIN_PANE_PX.
 */
export function mainPaneFit(viewport, { hasPrimarySub = true, separator = 1, axis = AXIS_PX } = {}) {
  const reserved = axis + (hasPrimarySub ? SUB_PANE_PX + separator : 0);
  return Math.max(MAIN_PANE_PX, Math.floor((Number(viewport) || 0) - reserved));
}

export function maCreate(line) {
  return {
    name: "MA",
    id: line.id,
    paneId: "candle_pane",
    calcParams: [line.period],
    styles: {
      lines: [{ style: "solid", smooth: false, size: 1, dashedValue: [2, 2], color: line.color }],
    },
  };
}

export function toggleMember(enabled, id) {
  const next = new Set(enabled);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

/** Live 1a API rejects limit above 1000. Pages walk backward with `end` when the server honors it. */
export const SERIES_PAGE = 1000;

export function appendOlderPage(collected, page, timeKey) {
  const times = [];
  for (const row of page || []) {
    const stamp = Number(row?.[timeKey]);
    if (Number.isFinite(stamp)) times.push(stamp);
  }
  if (!times.length) return { rows: collected, cursor: null, grew: false };
  const min = Math.min(...times);
  let prevMin = Infinity;
  for (const row of collected) {
    const stamp = Number(row?.[timeKey]);
    if (Number.isFinite(stamp)) prevMin = Math.min(prevMin, stamp);
  }
  if (min >= prevMin) return { rows: collected, cursor: null, grew: false };
  const seen = new Set(collected.map((row) => Number(row?.[timeKey])));
  const rows = collected.slice();
  for (const row of page) {
    const stamp = Number(row?.[timeKey]);
    if (!Number.isFinite(stamp) || seen.has(stamp)) continue;
    seen.add(stamp);
    rows.push(row);
  }
  return { rows, cursor: min - 1, grew: true };
}

// 5 分钟指标：窗口里有多少 5m bar，就需要多少页（加 2 页余量）。
// 不能用写死的页数，否则更早的窗口会被悄悄截掉。
export const METRIC_STEP_MS = 300_000;

export function metricPagesFor(spanMs, stepMs = METRIC_STEP_MS) {
  if (!Number.isFinite(spanMs) || spanMs <= 0) return 1;
  return Math.ceil(spanMs / stepMs / SERIES_PAGE) + 2;
}

const ACTIVE_FILL = new Set(["GAP_FILL", "BACKFILL"]);

export function formatUtc8(ms) {
  const shifted = new Date(Number(ms) + 8 * 60 * 60 * 1000);
  const pad = (value) => String(value).padStart(2, "0");
  return (
    `${shifted.getUTCFullYear()}-${pad(shifted.getUTCMonth() + 1)}-${pad(shifted.getUTCDate())}` +
    ` ${pad(shifted.getUTCHours())}:${pad(shifted.getUTCMinutes())}:${pad(shifted.getUTCSeconds())}`
  );
}

export function formatUtc8Clock(ms) {
  return formatUtc8(ms).slice(11, 19);
}

export function isBackfilling(events, nowMs) {
  return (events || []).some(
    (event) => ACTIVE_FILL.has(event.code) && nowMs - Number(event.ts) < 120_000,
  );
}

export function paintDelay(now, lastPaint, interval = 250) {
  return Math.max(0, interval - (now - lastPaint));
}

/** Wilder ATR, the same smoothing TA-Lib uses. First value is the SMA of `period` true ranges. */
export function atr(bars, period = 14) {
  const out = new Array(bars.length).fill(null);
  if (period < 1) return out;
  let prevClose = null;
  let seed = 0;
  let value = null;
  for (let i = 0; i < bars.length; i += 1) {
    const bar = bars[i];
    let tr = bar.high - bar.low;
    if (prevClose != null) {
      tr = Math.max(tr, Math.abs(bar.high - prevClose), Math.abs(bar.low - prevClose));
    }
    prevClose = bar.close;
    if (i < period) seed += tr;
    if (i === period - 1) {
      value = seed / period;
      out[i] = value;
    } else if (i >= period) {
      value = (value * (period - 1) + tr) / period;
      out[i] = value;
    }
  }
  return out;
}

/** Session VWAP. The session is the Binance UTC day, not a UTC+8 civil day. */
export function vwap(bars) {
  let day = null;
  let cumPv = 0;
  let cumV = 0;
  return bars.map((bar) => {
    const current = Math.floor(bar.timestamp / 86_400_000);
    if (current !== day) {
      day = current;
      cumPv = 0;
      cumV = 0;
    }
    const volume = bar.volume || 0;
    cumPv += ((bar.high + bar.low + bar.close) / 3) * volume;
    cumV += volume;
    return cumV === 0 ? null : cumPv / cumV;
  });
}

/** Cumulative taker-buy minus taker-sell, in base units. */
export function cvd(bars) {
  let total = 0;
  return bars.map((bar) => {
    const volume = bar.volume || 0;
    const buy = bar.takerBuy || 0;
    total += buy - (volume - buy);
    return total;
  });
}

export function asofSeries(bars, points, delayMs = 0) {
  let index = 0;
  let current = null;
  return bars.map((bar) => {
    const limit = bar.closeTime - delayMs;
    while (index < points.length && points[index].ts <= limit) {
      current = points[index].value;
      index += 1;
    }
    return current;
  });
}

export function basisSeries(bars, spot) {
  let index = 0;
  return bars.map((bar) => {
    while (index < spot.length && spot[index].timestamp < bar.timestamp) index += 1;
    let cursor = index;
    let spotClose = null;
    while (cursor < spot.length && spot[cursor].timestamp <= bar.closeTime) {
      spotClose = spot[cursor].close;
      cursor += 1;
    }
    index = cursor;
    return spotClose == null ? null : bar.close - spotClose;
  });
}

export function liquidationQuote(bars, events) {
  let index = 0;
  return bars.map((bar) => {
    while (index < events.length && events[index].ts < bar.timestamp) index += 1;
    let cursor = index;
    let sum = 0;
    let any = false;
    while (cursor < events.length && events[cursor].ts <= bar.closeTime) {
      sum += Number(events[cursor].price) * Number(events[cursor].qty);
      any = true;
      cursor += 1;
    }
    index = cursor;
    return any ? sum : null;
  });
}

function num(value) {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function toBar(row) {
  const volume = num(row.volume) ?? 0;
  const takerBuy = num(row.taker_buy_base) ?? 0;
  return {
    timestamp: Number(row.open_time),
    closeTime: Number(row.close_time),
    open: num(row.o),
    high: num(row.h),
    low: num(row.l),
    close: num(row.c),
    volume,
    turnover: num(row.quote_volume) ?? 0,
    takerBuy,
  };
}

export function joinBarExtras(bars, sources) {
  const metrics = (sources.metrics || [])
    .map((row) => ({
      ts: Number(row.ts),
      oi: num(row.oi),
      lsr: num(row.global_lsr),
    }))
    .sort((a, b) => a.ts - b.ts);
  const funding = (sources.funding || [])
    .map((row) => ({ ts: Number(row.funding_time), value: num(row.rate) }))
    .sort((a, b) => a.ts - b.ts);
  const spot = (sources.spot || [])
    .map((row) => ({ timestamp: Number(row.timestamp ?? row.open_time), close: num(row.close ?? row.c) }))
    .sort((a, b) => a.timestamp - b.timestamp);
  const premium = new Map(
    (sources.premium || []).map((row) => [Number(row.timestamp ?? row.open_time), num(row.close ?? row.c)]),
  );
  const liquidations = (sources.liquidations || [])
    .map((row) => ({ ts: Number(row.ts), price: row.price, qty: row.qty }))
    .sort((a, b) => a.ts - b.ts);
  const oi = asofSeries(bars, metrics.map((row) => ({ ts: row.ts, value: row.oi })), 300_000);
  const lsr = asofSeries(bars, metrics.map((row) => ({ ts: row.ts, value: row.lsr })), 300_000);
  const fundingLine = asofSeries(bars, funding, 0);
  const basis = basisSeries(bars, spot);
  const liq = liquidationQuote(bars, liquidations);
  return bars.map((bar, i) => ({
    ...bar,
    oi: oi[i],
    lsr: lsr[i],
    funding: fundingLine[i],
    basis: basis[i],
    premium: premium.has(bar.timestamp) ? premium.get(bar.timestamp) : null,
    liq: liq[i],
  }));
}
