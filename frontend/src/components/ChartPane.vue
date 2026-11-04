<script setup>
import { dispose, init, registerIndicator } from "klinecharts";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api, loadOlderSeries } from "../api.js";
import { state } from "../live.js";
import {
  INTERVALS,
  MAIN_PANE_PX,
  MA_LINES,
  SERIES_PAGE,
  SUB_PANE_PX,
  atr,
  chartBodyHeight,
  joinBarExtras,
  maCreate,
  paintDelay,
  toBar,
  toggleMember,
} from "../indicators.js";

const props = defineProps({
  market: { type: String, required: true },
  interval: { type: String, required: true },
});

const scrollEl = ref(null);
const host = ref(null);
const message = ref("");
const enabled = ref(new Set(["BOLL", "MACD", "ATR", "OI", "FUNDING", "LSR", "TAKER"]));
let chart = null;
let pushBar = null;
let pending = [];
let timer = null;
let lastPaint = 0;
let seenClosed = new Set();
let resizeObserver = null;

const MAIN = [
  ...MA_LINES.map((line) => ({ id: line.id, label: line.id, ma: line })),
  { id: "BOLL", label: "BOLL" },
];
const SUB = [
  { id: "MACD", label: "MACD" },
  { id: "ATR", label: "ATR" },
  { id: "OI", label: "OI" },
  { id: "FUNDING", label: "资金费率" },
  { id: "LSR", label: "多空比" },
  { id: "TAKER", label: "主动买卖比" },
];
const TOOLS = [
  { id: "segment", label: "线段" },
  { id: "rayLine", label: "射线" },
  { id: "horizontalStraightLine", label: "水平线" },
  { id: "fibonacciLine", label: "斐波那契" },
  { id: "simpleAnnotation", label: "标注" },
];

const NAMES = {
  ATR: "SUPERE_ATR",
  OI: "SUPERE_OI",
  FUNDING: "SUPERE_FUNDING",
  LSR: "SUPERE_LSR",
  TAKER: "SUPERE_TAKER",
};

let registered = false;
function registerCustom() {
  if (registered) return;
  registered = true;
  const line = (name, shortName, key, title, series, precision, calc) => {
    registerIndicator({
      name,
      shortName,
      series,
      precision,
      calcParams: name === "SUPERE_ATR" ? [14] : [],
      figures: [{ key, title, type: name === "SUPERE_LIQ" ? "bar" : "line" }],
      calc,
    });
  };
  line("SUPERE_ATR", "ATR", "atr", "ATR: ", "normal", 2, (data, indicator) => {
    const period = Number(indicator.calcParams?.[0] || 14);
    return atr(data, period).map((value) => ({ atr: value }));
  });
  line("SUPERE_OI", "OI", "oi", "OI: ", "normal", 0, (data) => data.map((bar) => ({ oi: bar.oi ?? null })));
  line("SUPERE_FUNDING", "资金费率", "funding", "费率: ", "normal", 6, (data) =>
    data.map((bar) => ({ funding: bar.funding ?? null })),
  );
  line("SUPERE_LSR", "多空比", "lsr", "多空比: ", "normal", 3, (data) =>
    data.map((bar) => ({ lsr: bar.lsr ?? null })),
  );
  line("SUPERE_TAKER", "主动买占比", "taker", "主动买: ", "normal", 3, (data) =>
    data.map((bar) => ({
      taker: bar.volume ? bar.takerBuy / bar.volume : null,
    })),
  );
}

function indicatorName(id) {
  return NAMES[id] || id;
}

function periodOf(id) {
  return INTERVALS.find((item) => item.id === id) || INTERVALS[3];
}

async function decorate(rawBars) {
  const bars = rawBars.map(toBar).filter((bar) => Number.isFinite(bar.timestamp));
  if (!bars.length) return [];
  const start = bars[0].timestamp - 86_400_000;
  const end = bars[bars.length - 1].closeTime;
  const liqStart = Math.max(start, end - 14 * 86_400_000);
  const [metricRows, fundingRows, spot, premium, liq] = await Promise.all([
    loadOlderSeries(
      (cursor) =>
        api
          .metrics({ symbol: "ETHUSDT", limit: SERIES_PAGE, end: cursor })
          .then((body) => body.rows || []),
      "ts",
      start,
    ).catch(() => []),
    loadOlderSeries(
      (cursor) =>
        api
          .funding({ symbol: "ETHUSDT", limit: SERIES_PAGE, end: cursor })
          .then((body) => body.settled || []),
      "funding_time",
      start,
    ).catch(() => []),
    props.market === "um"
      ? api.klines({ market: "spot", symbol: "ETHUSDT", interval: "1m", start, end, limit: 1500 }).catch(() => ({ bars: [] }))
      : Promise.resolve({ bars: [] }),
    api
      .klines({ market: "um_premium", symbol: "ETHUSDT", interval: props.interval, start, end, limit: 1500 })
      .catch(() => ({ bars: [] })),
    api.liquidations({ symbol: "ETHUSDT", start: liqStart, end }).catch(() => ({ rows: [] })),
  ]);
  return joinBarExtras(bars, {
    metrics: metricRows,
    funding: fundingRows,
    spot: spot.bars,
    premium: premium.bars,
    liquidations: liq.rows,
  });
}

function schedule(bar) {
  pending.push(bar);
  const delay = paintDelay(performance.now(), lastPaint);
  if (timer) return;
  timer = setTimeout(() => {
    timer = null;
    lastPaint = performance.now();
    const batch = pending;
    pending = [];
    if (!pushBar) return;
    for (const item of batch) pushBar(item);
  }, delay);
}

function carry(raw) {
  const bar = toBar(raw);
  const list = chart?.getDataList?.() || [];
  const prev = list.length ? list[list.length - 1] : null;
  const same = prev && prev.timestamp === bar.timestamp ? prev : null;
  return {
    ...bar,
    oi: same?.oi ?? null,
    lsr: same?.lsr ?? null,
    funding: same?.funding ?? null,
    basis: same?.basis ?? null,
    premium: same?.premium ?? null,
    liq: same?.liq ?? null,
  };
}

function loader() {
  return {
    getBars: async ({ type, timestamp, callback }) => {
      if (type === "backward" || type === "update") {
        callback([], { forward: false, backward: false });
        return;
      }
      try {
        const params = {
          market: props.market,
          symbol: "ETHUSDT",
          interval: props.interval,
          limit: 500,
        };
        if (type === "forward" && timestamp) params.end = timestamp - 1;
        if (type === "init") params.include_forming = true;
        const body = await api.klines(params);
        const rows = await decorate(body.bars || []);
        message.value = rows.length ? "" : "这个周期本地还没有 K 线";
        callback(rows, { forward: rows.length >= 500, backward: false });
      } catch {
        message.value = "K 线没有读出来";
        callback([], { forward: false, backward: false });
      }
    },
    subscribeBar: ({ callback }) => {
      pushBar = callback;
    },
    unsubscribeBar: () => {
      pushBar = null;
    },
  };
}

function mainItem(id) {
  return MAIN.find((item) => item.id === id);
}

function showIndicator(id) {
  const item = mainItem(id);
  if (item?.ma) {
    chart.createIndicator(maCreate(item.ma), true);
    return;
  }
  const name = indicatorName(id);
  if (item) {
    chart.createIndicator({ name, paneId: "candle_pane" }, true);
    return;
  }
  chart.createIndicator(name);
}

function hideIndicator(id) {
  const item = mainItem(id);
  if (item?.ma) {
    chart.removeIndicator({ id: item.ma.id });
    return;
  }
  chart.removeIndicator({ name: indicatorName(id) });
}

function mountIndicators() {
  for (const id of enabled.value) showIndicator(id);
}

function subCount() {
  return [...enabled.value].filter((id) => SUB.some((item) => item.id === id)).length;
}

function reflow() {
  if (!chart || !host.value || !scrollEl.value) return;
  const sep = Number(chart.getStyles()?.separator?.size) || 1;
  const needed = chartBodyHeight(subCount(), { separator: sep, axis: 36 });
  const viewport = scrollEl.value.clientHeight || 0;
  const next = Math.max(needed, viewport);
  if (Math.abs(host.value.getBoundingClientRect().height - next) < 2) return;
  host.value.style.height = `${next}px`;
  chart.resize();
}

function toggle(id) {
  const next = toggleMember(enabled.value, id);
  if (next.has(id)) showIndicator(id);
  else hideIndicator(id);
  enabled.value = next;
  reflow();
}

function draw(name) {
  chart?.createOverlay(name);
}

function clearDrawings() {
  chart?.removeOverlay({});
}

function onLive() {
  if (!chart || !pushBar) return;
  for (const raw of state.closed) {
    if (raw.market !== props.market || raw.interval !== props.interval) continue;
    const key = `${raw.interval}:${raw.open_time}`;
    if (seenClosed.has(key)) continue;
    seenClosed.add(key);
    schedule(carry(raw));
  }
  const forming = state.forming.find(
    (bar) => bar.market === props.market && bar.interval === props.interval && bar.closed === false,
  );
  if (forming) schedule(carry(forming));
}

onMounted(() => {
  registerCustom();
  chart = init(host.value, {
    locale: "zh-CN",
    timezone: "Asia/Shanghai",
    layout: {
      pane: { dragEnabled: true, minHeight: SUB_PANE_PX, height: SUB_PANE_PX },
    },
    styles: {
      grid: {
        horizontal: { color: "#2a3124" },
        vertical: { color: "#2a3124" },
      },
      candle: {
        bar: {
          upColor: "#7dba6a",
          downColor: "#d4654f",
          upBorderColor: "#7dba6a",
          downBorderColor: "#d4654f",
          upWickColor: "#7dba6a",
          downWickColor: "#d4654f",
        },
        priceMark: {
          last: { upColor: "#7dba6a", downColor: "#d4654f" },
          high: { color: "#9a947f" },
          low: { color: "#9a947f" },
        },
        tooltip: { text: { color: "#e6e2d6" } },
      },
      indicator: { tooltip: { text: { color: "#d7d2c4" } } },
      xAxis: { axisLine: { color: "#2a3124" }, tickText: { color: "#9a947f" } },
      yAxis: { axisLine: { color: "#2a3124" }, tickText: { color: "#9a947f" } },
      separator: { color: "#3a4234" },
      crosshair: {
        horizontal: {
          line: { color: "#d7a15f" },
          text: { backgroundColor: "#d7a15f", color: "#17190f" },
        },
        vertical: {
          line: { color: "#d7a15f" },
          text: { backgroundColor: "#d7a15f", color: "#17190f" },
        },
      },
    },
  });
  const period = periodOf(props.interval);
  chart.setSymbol({
    ticker: props.market === "um" ? "ETHUSDT 永续" : "ETHUSDT 现货",
    pricePrecision: 2,
    volumePrecision: 3,
  });
  chart.setPeriod({ span: period.span, type: period.type });
  chart.setPaneOptions({ id: "candle_pane", minHeight: MAIN_PANE_PX });
  mountIndicators();
  chart.setDataLoader(loader());
  reflow();
  resizeObserver = new ResizeObserver(() => reflow());
  resizeObserver.observe(scrollEl.value);
});

watch(() => props.interval, (id) => {
  seenClosed = new Set();
  const period = periodOf(id);
  chart?.setPeriod({ span: period.span, type: period.type });
});

watch(() => props.market, (market) => {
  seenClosed = new Set();
  chart?.setSymbol({
    ticker: market === "um" ? "ETHUSDT 永续" : "ETHUSDT 现货",
    pricePrecision: 2,
    volumePrecision: 3,
  });
});

watch(() => state.seq, onLive);

onBeforeUnmount(() => {
  if (timer) clearTimeout(timer);
  resizeObserver?.disconnect();
  if (host.value) dispose(host.value);
  chart = null;
});
</script>

<template>
  <div class="chart-pane">
    <div class="toolrow">
      <span>主图</span>
      <button
        v-for="item in MAIN"
        :key="item.id"
        type="button"
        :class="{ on: enabled.has(item.id) }"
        @click="toggle(item.id)"
      >
        {{ item.label }}
      </button>
      <span class="split">副图</span>
      <button
        v-for="item in SUB"
        :key="item.id"
        type="button"
        :class="{ on: enabled.has(item.id) }"
        @click="toggle(item.id)"
      >
        {{ item.label }}
      </button>
      <span class="split">画线</span>
      <button v-for="tool in TOOLS" :key="tool.id" type="button" @click="draw(tool.id)">
        {{ tool.label }}
      </button>
      <button type="button" @click="clearDrawings">清除</button>
    </div>
    <p v-if="message" class="chart-note">{{ message }}</p>
    <div ref="scrollEl" class="chart-scroll">
      <div ref="host" class="chart-host"></div>
    </div>
  </div>
</template>
