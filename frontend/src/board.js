// Poll the board service and decide which in-page notices are due.
import { BOARD_ORIGIN } from "./api.js";
import { formatUtc8 } from "./indicators.js";
import { pushNotice } from "./notify.js";

export const BOARD_POLL_MS = 10_000;
const LOSS_KEY = "supere.board.risk";
const CALC_KEY = "supere.board.calc";

export function boardNotice(event, now, cooldownMs = 30 * 60 * 1000) {
  const bucket = Math.floor(now / cooldownMs);
  return {
    id: `board:${event.planId}:${event.kind}:${bucket}`,
    title: event.title,
    body: event.body,
    tone: event.tone || "warn",
    system: false,
  };
}

export function collectBoardEvents(previous, next) {
  if (!next) return [];
  const events = [];
  const prevPlans = previous?.plans || [];
  if (next.crosscheck?.status === "anomaly" && previous?.crosscheck?.status !== "anomaly") {
    events.push({
      kind: "anomaly",
      planId: "data",
      title: "数据异常",
      body: next.crosscheck.message || "币安和 OKX 价差过大",
      tone: "warn",
    });
  }
  for (const plan of next.plans || []) {
    const prev = prevPlans.find((item) => item.id === plan.id);
    if (plan.status === "可进场" && prev?.status !== "可进场") {
      events.push({
        kind: "confirm",
        planId: plan.id,
        title: plan.title,
        body: `${(plan.confirm?.names || []).join("、") || "确认信号"}，看进场区`,
        tone: plan.side === "short" ? "down" : "up",
      });
    }
    if (plan.id === "flip" && plan.status === "已触发" && prev?.status !== "已触发") {
      events.push({
        kind: "flip",
        planId: "flip",
        title: "翻转条件触发",
        body: plan.note || "当前方向作废",
        tone: "warn",
      });
    }
  }
  const prevEvents = previous?.risk?.events || [];
  for (const item of next.risk?.events || []) {
    const before = prevEvents.find((row) => row.ts_event === item.ts_event);
    if (item.blocking && !before?.blocking) {
      events.push({
        kind: "calendar",
        planId: item.kind,
        title: item.title || item.kind,
        body: `${item.window} 不开新单`,
        tone: "warn",
      });
    }
  }
  return events;
}

export function lossState(stored, now) {
  const day = formatUtc8(now).slice(0, 10);
  if (!stored || stored.day !== day) return { day, losses: 0 };
  return { day, losses: Math.max(0, Number(stored.losses) || 0) };
}

export function bumpLoss(stored, delta, now) {
  const current = lossState(stored, now);
  return { day: current.day, losses: Math.max(0, current.losses + delta) };
}

export function loadCalc(storage = globalThis.localStorage) {
  const defaults = { capital: 100, maxLossU: 4, maxLossPct: 0, leverage: 20 };
  try {
    const raw = storage?.getItem?.(CALC_KEY);
    return { ...defaults, ...(raw ? JSON.parse(raw) : {}) };
  } catch {
    return defaults;
  }
}

export function saveCalc(value, storage = globalThis.localStorage) {
  storage?.setItem?.(CALC_KEY, JSON.stringify(value));
}

export function loadLoss(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem?.(LOSS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveLoss(value, storage = globalThis.localStorage) {
  storage?.setItem?.(LOSS_KEY, JSON.stringify(value));
}

let timer = null;
let last = null;

export function startBoardDesk(state, { fetchImpl = globalThis.fetch, now = () => Date.now() } = {}) {
  if (timer) return;
  async function tick() {
    try {
      const response = await fetchImpl(`${BOARD_ORIGIN}/api/board`);
      if (!response.ok) throw new Error(String(response.status));
      const next = await response.json();
      const moment = now();
      for (const event of collectBoardEvents(last, next)) {
        pushNotice(boardNotice(event, moment));
      }
      last = next;
      state.view = next;
      state.offline = false;
      state.error = "";
    } catch {
      state.offline = true;
      state.error = "盘面服务未启动";
    }
  }
  tick();
  timer = setInterval(tick, BOARD_POLL_MS);
}

export function stopBoardDesk() {
  if (timer) clearInterval(timer);
  timer = null;
  last = null;
}
