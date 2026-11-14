// Drawings saved per market and interval. Points are timestamps, so a reload can restore them.

export const DRAW_TOOLS = [
  { id: "horizontalStraightLine", label: "水平线" },
  { id: "segment", label: "趋势线" },
  { id: "rayLine", label: "射线" },
  { id: "priceChannelLine", label: "平行通道" },
  { id: "fibonacciLine", label: "斐波那契" },
  { id: "priceLine", label: "价格标注" },
];

const MIN_POINTS = {
  horizontalStraightLine: 1,
  segment: 2,
  rayLine: 2,
  priceChannelLine: 3,
  fibonacciLine: 2,
  priceLine: 1,
};

export function drawingKey(market, interval) {
  return `${market}|ETHUSDT|${interval}`;
}

export function serializeOverlay(overlay) {
  const name = overlay?.name;
  const need = MIN_POINTS[name];
  if (!need) return null;
  const points = (overlay.points || [])
    .map((point) => ({ timestamp: Number(point?.timestamp), value: Number(point?.value) }))
    .filter((point) => Number.isFinite(point.timestamp) && Number.isFinite(point.value));
  if (points.length < need) return null;
  return { id: String(overlay.id || ""), name, points: points.slice(0, need) };
}

export function loadBook(raw, key) {
  const books = raw && typeof raw === "object" ? raw.books : null;
  const rows = books && Array.isArray(books[key]) ? books[key] : [];
  return rows.map(serializeOverlay).filter(Boolean);
}

export function saveBook(raw, key, overlays) {
  const books = { ...(raw?.books || {}) };
  books[key] = (overlays || []).map(serializeOverlay).filter(Boolean);
  return { version: 1, books };
}
