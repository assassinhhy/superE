<script setup>
import { dispose, init, registerIndicator } from "klinecharts";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { api } from "../api.js";
import { BOARD_INTERVALS, registerBoardOverlays } from "../boardOverlays.js";
import {
  BOARD_GROUP,
  DRAW_TOOLS,
  USER_GROUP,
  drawingKey,
  isUserOverlay,
  loadBook,
  saveBook,
} from "../drawings.js";
import { state } from "../live.js";
import {
  AXIS_PX,
  INTERVALS,
  MAIN_PANE_PX,
  MA_LINES,
  PRIMARY_SUB,
  SUB_PANE_PX,
  atr,
  chartBodyHeight,
  maCreate,
  mainPaneFit,
  paintDelay,
  toBar,
  toggleMember,
} from "../indicators.js";

const props = defineProps({
  market: { type: String, required: true },
  interval: { type: String, required: true },
  board: { type: Object, default: null },
  focusPlan: { type: String, default: "" },
});

const STORE = "supere.drawings.v1";
const BOARD_TOGGLE = "supere.board.overlay";
const boardOn = ref(true);
try {
  boardOn.value = localStorage.getItem(BOARD_TOGGLE) !== "0";
} catch {
  boardOn.value = true;
}
const scrollEl = ref(null);
const host = ref(null);
const message = ref("");
const selectedId = ref("");
const enabled = ref(new Set(["MA20", "MA50", "BOLL", "MACD", "RSI", "ATR"]));
let chart = null;
let mainHeight = 0;
let pushBar = null;
let pending = [];
let timer = null;
let lastPaint = 0;
let seenClosed = new Set();
let resizeObserver = null;
let bookKey = "";
let restoring = false;

const MAIN = [
  ...MA_LINES.map((line) => ({ id: line.id, label: line.id, ma: line })),
  { id: "BOLL", label: "BOLL" },
];
const SUB = [
  { id: "RSI", label: "RSI" },
  { id: PRIMARY_SUB, label: "MACD" },
  { id: "ATR", label: "ATR" },
];

function readStore() {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "{}");
  } catch {
    return {};
  }
}

function writeStore(value) {
  localStorage.setItem(STORE, JSON.stringify(value));
}

const NAMES = { ATR: "SUPERE_ATR" };

let registered = false;
function registerCustom() {
  if (registered) return;
  registered = true;
  registerIndicator({
    name: "SUPERE_ATR",
    shortName: "ATR",
    series: "normal",
    precision: 2,
    calcParams: [14],
    figures: [{ key: "atr", title: "ATR: ", type: "line" }],
    calc: (data, indicator) => {
      const period = Number(indicator.calcParams?.[0] || 14);
      return atr(data, period).map((value) => ({ atr: value }));
    },
  });
}

function indicatorName(id) {
  return NAMES[id] || id;
}

function periodOf(id) {
  return INTERVALS.find((item) => item.id === id) || INTERVALS[3];
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
        const rows = (body.bars || []).map(toBar).filter((bar) => Number.isFinite(bar.timestamp));
        message.value = rows.length ? "" : "这个周期本地还没有 K 线";
        callback(rows, { forward: rows.length >= 500, backward: false });
        paintBoard();
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
  if (item) {
    chart.createIndicator({ name: indicatorName(id), paneId: "candle_pane" }, true);
    return;
  }
  if (id === "RSI") {
    chart.createIndicator({ name: "RSI", calcParams: [14] });
    return;
  }
  chart.createIndicator(indicatorName(id));
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
  for (const item of MAIN) {
    if (enabled.value.has(item.id)) showIndicator(item.id);
  }
  for (const item of SUB) {
    if (enabled.value.has(item.id)) showIndicator(item.id);
  }
}

function subCount() {
  return SUB.filter((item) => enabled.value.has(item.id)).length;
}

function reflow() {
  if (!chart || !host.value || !scrollEl.value) return;
  const sep = Number(chart.getStyles()?.separator?.size) || 1;
  const viewport = scrollEl.value.clientHeight || 0;
  const main = mainPaneFit(viewport, { hasPrimarySub: enabled.value.has(PRIMARY_SUB), separator: sep, axis: AXIS_PX });
  if (main !== mainHeight) {
    mainHeight = main;
    chart.setPaneOptions({ id: "candle_pane", height: main, minHeight: MAIN_PANE_PX });
  }
  const needed = chartBodyHeight(subCount(), { separator: sep, axis: AXIS_PX, mainPx: main });
  const next = Math.max(needed, viewport);
  if (Math.abs(host.value.getBoundingClientRect().height - next) < 2) return;
  host.value.style.height = `${next}px`;
  chart.resize();
}

function restackSubs(before) {
  for (const item of SUB) {
    if (before.has(item.id)) chart.removeIndicator({ name: indicatorName(item.id) });
  }
  for (const item of SUB) {
    if (enabled.value.has(item.id)) showIndicator(item.id);
  }
}

function toggle(id) {
  const before = enabled.value;
  const next = toggleMember(before, id);
  enabled.value = next;
  if (SUB.some((item) => item.id === id) && next.has(id)) restackSubs(before);
  else if (next.has(id)) showIndicator(id);
  else hideIndicator(id);
  reflow();
}

function overlayHooks() {
  return {
    onDrawEnd: () => persistSoon(),
    onPressedMoveEnd: () => persistSoon(),
    onRemoved: () => persistSoon(),
    onSelected: (event) => {
      selectedId.value = event?.overlay?.id || "";
    },
    onDeselected: (event) => {
      if (selectedId.value === event?.overlay?.id) selectedId.value = "";
    },
  };
}

function persist() {
  if (!chart || !bookKey || restoring) return;
  const overlays = (chart.getOverlays?.() || []).filter(isUserOverlay);
  writeStore(saveBook(readStore(), bookKey, overlays));
}

function persistSoon() {
  if (restoring) return;
  setTimeout(persist, 0);
}

function applyBook(key) {
  if (!chart) return;
  restoring = true;
  bookKey = key;
  chart.removeOverlay({ groupId: USER_GROUP });
  for (const row of loadBook(readStore(), key)) {
    chart.createOverlay({
      name: row.name,
      id: row.id || undefined,
      groupId: USER_GROUP,
      points: row.points,
      ...overlayHooks(),
    });
  }
  restoring = false;
  selectedId.value = "";
}

function draw(name) {
  chart?.createOverlay({ name, groupId: USER_GROUP, ...overlayHooks() });
}

function deleteSelected() {
  if (!chart || !selectedId.value) return;
  chart.removeOverlay({ id: selectedId.value });
  selectedId.value = "";
  persistSoon();
}

function clearDrawings() {
  if (!chart) return;
  restoring = true;
  chart.removeOverlay({ groupId: USER_GROUP });
  restoring = false;
  selectedId.value = "";
  persist();
}

function toggleBoard() {
  boardOn.value = !boardOn.value;
  try {
    localStorage.setItem(BOARD_TOGGLE, boardOn.value ? "1" : "0");
  } catch {
    /* ignore */
  }
  paintBoard();
}

function paintBoard() {
  if (!chart) return;
  chart.removeOverlay({ groupId: BOARD_GROUP });
  const view = props.board;
  if (!boardOn.value || !BOARD_INTERVALS.has(props.interval) || !view) return;
  const stamp = Date.now();
  const levels = [
    ...(view.levels?.resistance || []).map((level) => ({ ...level, side: "resistance" })),
    ...(view.levels?.support || []).map((level) => ({ ...level, side: "support" })),
  ];
  for (const level of levels) {
    chart.createOverlay({
      name: "boardLevel",
      groupId: BOARD_GROUP,
      lock: true,
      points: [{ timestamp: stamp, value: level.price }],
      extendData: level,
    });
  }
  for (const plan of view.plans || []) {
    if (plan.id === "flip" || plan.entry_low == null || plan.entry_high == null) continue;
    chart.createOverlay({
      name: "boardZone",
      groupId: BOARD_GROUP,
      lock: true,
      points: [
        { timestamp: stamp, value: plan.entry_high },
        { timestamp: stamp, value: plan.entry_low },
      ],
      extendData: { ...plan, highlight: props.focusPlan === plan.id },
    });
  }
  if (props.focusPlan) chart.scrollToRealTime?.();
}

function onKey(event) {
  if (event.key !== "Delete" && event.key !== "Backspace") return;
  const tag = event.target?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || event.target?.isContentEditable) return;
  if (!selectedId.value) return;
  event.preventDefault();
  deleteSelected();
}

function onLive() {
  if (!chart || !pushBar) return;
  for (const raw of state.closed) {
    if (raw.market !== props.market || raw.interval !== props.interval) continue;
    const key = `${raw.interval}:${raw.open_time}`;
    if (seenClosed.has(key)) continue;
    seenClosed.add(key);
    schedule(toBar(raw));
  }
  const formingCandidates = state.forming.filter(
    (bar) => bar.market === props.market && bar.interval === props.interval && bar.closed === false,
  );
  const forming = formingCandidates.reduce((best, bar) => {
    if (!best) return bar;
    return (bar.open_time || 0) >= (best.open_time || 0) ? bar : best;
  }, null);
  if (forming) schedule(toBar(forming));
}

onMounted(() => {
  registerCustom();
  registerBoardOverlays();
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
  chart.setStyles({
    overlay: {
      line: { color: "#d7a15f" },
      point: { color: "#d7a15f", activeColor: "#7dba6a" },
      text: { color: "#17190f", backgroundColor: "#d7a15f", borderColor: "#d7a15f" },
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
  applyBook(drawingKey(props.market, props.interval));
  paintBoard();
  reflow();
  resizeObserver = new ResizeObserver(() => reflow());
  resizeObserver.observe(scrollEl.value);
  window.addEventListener("keydown", onKey);
});

watch(() => props.interval, (id) => {
  seenClosed = new Set();
  persist();
  const period = periodOf(id);
  chart?.setPeriod({ span: period.span, type: period.type });
  applyBook(drawingKey(props.market, id));
  paintBoard();
});

watch(() => props.market, (market) => {
  seenClosed = new Set();
  persist();
  chart?.setSymbol({
    ticker: market === "um" ? "ETHUSDT 永续" : "ETHUSDT 现货",
    pricePrecision: 2,
    volumePrecision: 3,
  });
  applyBook(drawingKey(market, props.interval));
  paintBoard();
});

watch(() => [props.board, props.focusPlan, boardOn.value], () => paintBoard());

watch(() => state.seq, onLive);

onBeforeUnmount(() => {
  persist();
  window.removeEventListener("keydown", onKey);
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
      <button v-for="tool in DRAW_TOOLS" :key="tool.id" type="button" @click="draw(tool.id)">
        {{ tool.label }}
      </button>
      <button type="button" :disabled="!selectedId" @click="deleteSelected">删除</button>
      <button type="button" @click="clearDrawings">清除</button>
      <span class="split">盘面</span>
      <button type="button" :class="{ on: boardOn }" @click="toggleBoard">盘面标注</button>
    </div>
    <p v-if="message" class="chart-note">{{ message }}</p>
    <div ref="scrollEl" class="chart-scroll">
      <div ref="host" class="chart-host"></div>
    </div>
  </div>
</template>
