// Poll the board service and decide which in-page notices are due.
import { BOARD_ORIGIN } from "./api.js";
import { formatUtc8, formatUtc8Clock } from "./indicators.js";
import { pushNotice } from "./notify.js";

export const BOARD_POLL_MS = 10_000;
export const BOARD_STALE_MS = 30_000;
export const FIFTEEN_MS = 15 * 60 * 1000;
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

export function boardClock(view, fetchedAt) {
  if (Number.isFinite(fetchedAt)) return formatUtc8Clock(fetchedAt);
  const text = view?.as_of_utc8;
  if (typeof text === "string" && text.length >= 19) return text.slice(11, 19);
  return "—";
}

export function basisStamp(view) {
  const basis = view?.basis;
  if (basis?.close_utc8) {
    return { close: basis.close_utc8, next: basis.next_close_utc8 || "" };
  }
  const asOf = Number(view?.as_of);
  if (!Number.isFinite(asOf)) return { close: "", next: "" };
  const closeBoundary = asOf - (asOf % FIFTEEN_MS);
  const next = closeBoundary + FIFTEEN_MS;
  return {
    close: formatUtc8Clock(closeBoundary),
    next: formatUtc8(next).slice(11, 16),
  };
}

export function boardFreshLine(view, fetchedAt) {
  const stamp = basisStamp(view);
  const close = stamp.close || "—";
  const next = stamp.next || "—";
  return `更新于 ${boardClock(view, fetchedAt)} · 依据 15m 收线 ${close} · 下次重算 ${next}`;
}

/** Seconds since the last successful pull once the gap is past `limit`. Null while still fresh. */
export function boardStaleSeconds(anchor, now, limit = BOARD_STALE_MS) {
  if (!Number.isFinite(anchor) || !Number.isFinite(now)) return null;
  const age = now - anchor;
  if (age <= limit) return null;
  return Math.floor(age / 1000);
}

export function applyBoardPoll(state, previous, outcome, now) {
  const since = Number.isFinite(state?.since) ? state.since : now;
  if (!outcome?.ok) {
    return {
      view: state?.view ?? null,
      previous: previous ?? null,
      offline: true,
      error: outcome?.error || "盘面服务未启动",
      fetchedAt: Number.isFinite(state?.fetchedAt) ? state.fetchedAt : null,
      since,
      events: [],
    };
  }
  return {
    view: outcome.view,
    previous: outcome.view,
    offline: false,
    error: "",
    fetchedAt: now,
    since,
    events: collectBoardEvents(previous, outcome.view),
  };
}

let timer = null;
let last = null;

export function startBoardDesk(state, { fetchImpl = globalThis.fetch, now = () => Date.now() } = {}) {
  if (timer) return;
  async function tick() {
    const moment = now();
    let outcome;
    try {
      const response = await fetchImpl(`${BOARD_ORIGIN}/api/board`);
      if (!response.ok) throw new Error(String(response.status));
      outcome = { ok: true, view: await response.json() };
    } catch {
      outcome = { ok: false, error: "盘面服务未启动" };
    }
    const applied = applyBoardPoll(
      { view: state.view, fetchedAt: state.fetchedAt, since: state.since },
      last,
      outcome,
      moment,
    );
    for (const event of applied.events) pushNotice(boardNotice(event, moment));
    last = applied.previous;
    state.view = applied.view;
    state.offline = applied.offline;
    state.error = applied.error;
    state.fetchedAt = applied.fetchedAt;
    state.since = applied.since;
  }
  tick();
  timer = setInterval(tick, BOARD_POLL_MS);
}

export function stopBoardDesk() {
  if (timer) clearInterval(timer);
  timer = null;
  last = null;
}
