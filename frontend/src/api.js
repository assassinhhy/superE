import { SERIES_PAGE, appendOlderPage } from "./indicators.js";

async function getJSON(path) {
  const response = await fetch(path);
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
  // Debug predictions come from the train service on 8768 (vite proxies /api/predict there).
  predict: (interval) => getJSON(interval ? `/api/predict?${query({ interval })}` : "/api/predict"),
  aggregate: (interval, bars) =>
    getJSON(`/api/checks/aggregate?${query({ interval, bars })}`),
};
