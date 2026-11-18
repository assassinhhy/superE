import { SERIES_PAGE, appendOlderPage } from "./indicators.js";

const DEFAULT_TIMEOUT_MS = 15000;
const MAX_ERROR_BODY = 500;

// Fetch JSON with an optional caller-supplied AbortSignal and a default timeout.
// Pass timeoutMs: 0 to disable the timeout. Non-2xx responses throw an Error
// whose message keeps the original "<status> <path>" prefix and appends any
// response body text (truncated) for easier debugging.
async function getJSON(path, options) {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS, ...init } = options || {};
  const controller = new AbortController();
  let timedOut = false;
  const timer =
    timeoutMs > 0
      ? setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeoutMs)
      : null;
  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", onAbort, { once: true });
  }

  let response;
  try {
    response = await fetch(path, { ...init, signal: controller.signal });
  } catch (error) {
    if (timedOut) {
      const timeoutError = new Error(`timeout after ${timeoutMs}ms ${path}`);
      timeoutError.name = "TimeoutError";
      timeoutError.path = path;
      timeoutError.cause = error;
      throw timeoutError;
    }
    throw error;
  } finally {
    if (timer != null) clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onAbort);
  }

  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.text()).trim();
    } catch (error) {
      detail = "";
    }
    if (detail.length > MAX_ERROR_BODY) detail = `${detail.slice(0, MAX_ERROR_BODY)}…`;
    const error = new Error(detail ? `${response.status} ${path}: ${detail}` : `${response.status} ${path}`);
    error.status = response.status;
    error.path = path;
    throw error;
  }
  return response.json();
}

function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") search.set(key, String(value));
  }
  return search.toString();
}

// fetchPage(end) must return the newest page at or before `end` (undefined = newest overall).
// The first call should pass the chart batch's own end, so older bars get their own window.
export async function loadOlderSeries(fetchPage, timeKey, until, maxPages = 8) {
  let rows = [];
  let end;
  for (let page = 0; page < maxPages; page += 1) {
    const batch = await fetchPage(end);
    const next = appendOlderPage(rows, batch, timeKey);
    rows = next.rows;
    if (!next.grew || next.cursor == null) break;
    if (batch.length < SERIES_PAGE) break;
    if (until != null && next.cursor <= until) break;
    end = next.cursor;
  }
  return rows;
}

// News is its own process. The page on 8766 calls it directly so 8766 stays up.
export const NEWS_ORIGIN = "http://127.0.0.1:8769";
export const BOARD_ORIGIN = "http://127.0.0.1:8770";

function newsURL(path, params) {
  const q = params ? `?${query(params)}` : "";
  return `${NEWS_ORIGIN}${path}${q}`;
}

export const api = {
  health: () => getJSON("/api/health"),
  status: () => getJSON("/api/status"),
  klines: (params) => getJSON(`/api/klines?${query(params)}`),
  metrics: (params) => getJSON(`/api/metrics?${query(params)}`),
  funding: (params) => getJSON(`/api/funding?${query(params)}`),
  liquidations: (params) => getJSON(`/api/liquidations?${query(params)}`),
  anomalies: () => getJSON("/api/anomalies?limit=50"),
  events: () => getJSON("/api/events?limit=30"),
  catalog: () => getJSON("/api/catalog"),
  settings: () => getJSON("/api/settings"),
  signals: () => getJSON("/api/signals"),
  // Forecasts are only fetched by the explicit 预测 button.
  predictOnClick: () => getJSON("/api/predict", { method: "POST" }),
  // Cached news. GET never fetches and never runs a prediction.
  news: (params) => getJSON(newsURL("/api/news", params || {})),
  refreshNews: () => getJSON(newsURL("/api/news/refresh"), { method: "POST" }),
  aggregate: (interval, bars) =>
    getJSON(`/api/checks/aggregate?${query({ interval, bars })}`),
};
