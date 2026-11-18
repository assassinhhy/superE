import { reactive } from "vue";
import { api } from "./api.js";

export const state = reactive({
  ws: false,
  preview: false,
  version: "",
  status: {},
  depth: null,
  ticker: null,
  mark: null,
  trades: [],
  liquidations: [],
  forming: [],
  closed: [],
  anomalies: [],
  events: [],
  error: "",
  seq: 0,
});

let socket = null;
let started = false;
let timers = [];

function applyFrame(frame) {
  state.depth = frame.depth || null;
  state.ticker = frame.ticker || null;
  state.mark = frame.mark || null;
  state.trades = frame.trades || [];
  state.liquidations = frame.liquidations || [];
  state.forming = frame.forming || [];
  state.closed = frame.closed || [];
  if (frame.status && Object.keys(frame.status).length) state.status = frame.status;
  state.seq += 1;
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws/live`);
  socket.onopen = () => {
    state.ws = true;
    state.error = "";
  };
  socket.onclose = () => {
    state.ws = false;
  };
  socket.onerror = () => {
    state.ws = false;
  };
  socket.onmessage = (event) => {
    try {
      applyFrame(JSON.parse(event.data));
    } catch {
      state.error = "实时数据解析失败";
    }
  };
}

async function refresh() {
  try {
    const [health, anomalies, events] = await Promise.all([
      api.health(),
      api.anomalies(),
      api.events(),
    ]);
    state.preview = Boolean(health.preview);
    state.version = health.version || "";
    state.anomalies = anomalies.rows || [];
    state.events = events.rows || [];
  } catch {
    state.error = "连不上本机 API";
  }
}

export function startLive() {
  if (started) return stopLive;
  started = true;
  connect();
  refresh();
  timers = [
    setInterval(refresh, 5000),
    setInterval(() => {
      if (!socket || socket.readyState === WebSocket.CLOSED) connect();
    }, 2000),
  ];
  return stopLive;
}

export function stopLive() {
  started = false;
  for (const id of timers) clearInterval(id);
  timers = [];
  if (socket) {
    socket.onopen = null;
    socket.onclose = null;
    socket.onerror = null;
    socket.onmessage = null;
    try {
      socket.close();
    } catch {
      /* already closed */
    }
    socket = null;
  }
  state.ws = false;
}
