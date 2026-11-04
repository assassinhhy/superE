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
  aggregate: (interval, bars) =>
    getJSON(`/api/checks/aggregate?${query({ interval, bars })}`),
};
