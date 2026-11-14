// Polls closed ETHUSDT perpetual bars and announces new signals. Runs on every tab.
import { reactive } from "vue";
import { api } from "./api.js";
import { formatUtc8, toBar } from "./indicators.js";
import { pushNotice } from "./notify.js";
import {
  SIGNAL_INTERVALS,
  ingestInterval,
  loadSignalState,
  saveSignalState,
} from "./signals.js";

export const desk = reactive({
  items: [],
  error: "",
});

let state = loadSignalState();
let started = false;
let timer = 0;
desk.items = state.items;

function noticeOf(item) {
  return {
    id: item.id,
    title: `${item.interval} ${item.label}`,
    body: `${formatUtc8(item.timeMs)} · ${Number(item.price).toFixed(2)}`,
    tone: item.tone,
  };
}

export async function pollSignals(divergence = false) {
  const now = Date.now();
  const before = { ...state.seenOpen };
  const batch = [];
  let failed = 0;
  for (const interval of SIGNAL_INTERVALS) {
    try {
      const body = await api.klines({ market: "um", symbol: "ETHUSDT", interval, limit: 200 });
      const bars = (body.bars || [])
        .map(toBar)
        .filter((bar) => Number.isFinite(bar.timestamp))
        .sort((a, b) => a.timestamp - b.timestamp);
      const result = ingestInterval(state, interval, bars, now, { divergence });
      state = result.state;
      batch.push(...result.announce);
    } catch {
      failed += 1;
    }
  }
  desk.error = failed === SIGNAL_INTERVALS.length ? "信号没有读到 K 线" : "";
  desk.items = state.items.slice();
  saveSignalState(state);
  const anyKnown = SIGNAL_INTERVALS.some((interval) => before[interval] != null);
  if (!anyKnown) {
    const newest = batch.slice().sort((a, b) => b.timeMs - a.timeMs)[0];
    if (newest) pushNotice(noticeOf(newest));
    return;
  }
  for (const item of batch) pushNotice(noticeOf(item));
}

export function startSignalDesk(readDivergence) {
  if (started) return;
  started = true;
  const tick = () => pollSignals(Boolean(readDivergence?.()));
  tick();
  timer = setInterval(tick, 8000);
}

export function stopSignalDesk() {
  if (timer) clearInterval(timer);
  timer = 0;
  started = false;
}
