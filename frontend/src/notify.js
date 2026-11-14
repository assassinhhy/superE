// In-page notices, an optional beep, and a Windows balloon when the desktop bridge exists.

const KEY = "supere.notify.v1";

export function rememberNotice(seen, id) {
  const list = Array.isArray(seen) ? seen.slice() : [];
  if (!id || list.includes(id)) return { seen: list, fresh: false };
  list.push(id);
  return { seen: list.slice(-400), fresh: true };
}

export function loadNotify(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem?.(KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return {
      sound: parsed.sound !== false,
      seen: Array.isArray(parsed.seen) ? parsed.seen : [],
    };
  } catch {
    return { sound: true, seen: [] };
  }
}

let state = { sound: true, seen: [] };
let booted = false;
const listeners = new Set();

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* private mode */
  }
}

export function bootNotify() {
  if (booted) return;
  booted = true;
  state = loadNotify();
}

export function soundEnabled() {
  bootNotify();
  return state.sound;
}

export function setSoundEnabled(on) {
  bootNotify();
  state.sound = Boolean(on);
  persist();
}

export function subscribeNotices(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function systemBridgeReady() {
  return typeof globalThis.pywebview?.api?.notify === "function";
}

/** Existing signal notices still ask for a balloon. Board plans pass system: false. */
export function wantsSystemNotice(notice) {
  return notice?.system !== false;
}

function beep(tone) {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) return;
  try {
    const ctx = beep.ctx || new Ctx();
    beep.ctx = ctx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = tone === "down" ? 320 : tone === "up" ? 640 : 470;
    gain.gain.setValueAtTime(0.05, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.14);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
  } catch {
    /* autoplay blocked */
  }
}

async function systemNotify(notice) {
  const bridge = globalThis.pywebview?.api?.notify;
  if (typeof bridge === "function") {
    try {
      await bridge(String(notice.title || "superE"), String(notice.body || ""));
      return;
    } catch {
      /* the page toast still shows */
    }
  }
  if (typeof Notification !== "undefined" && Notification.permission === "granted") {
    try {
      new Notification(notice.title || "superE", { body: notice.body || "" });
    } catch {
      /* WebView2 often has no Notification */
    }
  }
}

export function pushNotice(notice) {
  bootNotify();
  const result = rememberNotice(state.seen, notice?.id);
  state.seen = result.seen;
  persist();
  if (!result.fresh) return false;
  for (const fn of listeners) fn(notice);
  if (state.sound) beep(notice?.tone);
  if (wantsSystemNotice(notice)) systemNotify(notice);
  return true;
}
