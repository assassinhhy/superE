import { SERIES_PAGE, appendOlderPage } from "./indicators.js";

async function getJSON(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${path}`);
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
  news: (params) => getJSON(`/api/news?${query(params || {})}`),
  refreshNews: () => getJSON("/api/news/refresh", { method: "POST" }),
  aggregate: (interval, bars) =>
    getJSON(`/api/checks/aggregate?${query({ interval, bars })}`),
};
