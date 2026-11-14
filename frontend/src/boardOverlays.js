// Chart figures for board levels. They live in group "board", never in the user drawing book.
import { registerOverlay } from "klinecharts";

let registered = false;

function levelColor(data) {
  if (data?.side === "support") return "#7dba6a";
  if (data?.side === "resistance") return "#d4654f";
  return "#d7a15f";
}

export function levelFigures({ coordinates, bounding, overlay }) {
  const y = coordinates?.[0]?.y;
  if (!Number.isFinite(y)) return [];
  const data = overlay?.extendData || {};
  const color = levelColor(data);
  const name = data.label || (data.sources || [])[0] || "";
  const text = `${Number(data.price).toFixed(2)} ${name}${data.touches ? ` · 测${data.touches}` : ""}`;
  return [
    {
      type: "line",
      ignoreEvent: true,
      attrs: { coordinates: [{ x: 0, y }, { x: bounding.width, y }] },
      styles: { color, size: 1, style: data.dynamic ? "dashed" : "solid" },
    },
    {
      type: "text",
      ignoreEvent: true,
      attrs: { x: 8, y: y - 2, text, align: "left", baseline: "bottom" },
      styles: { color, size: 11, backgroundColor: "transparent" },
    },
  ];
}

export function zoneFigures({ coordinates, bounding, overlay, yAxis }) {
  const y1 = coordinates?.[0]?.y;
  const y2 = coordinates?.[1]?.y;
  if (!Number.isFinite(y1) || !Number.isFinite(y2)) return [];
  const data = overlay?.extendData || {};
  const top = Math.min(y1, y2);
  const height = Math.max(1, Math.abs(y2 - y1));
  const fill = data.highlight
    ? "rgba(224,177,90,0.28)"
    : data.side === "long"
      ? "rgba(125,186,106,0.16)"
      : "rgba(212,101,79,0.16)";
  const figures = [
    {
      type: "rect",
      ignoreEvent: true,
      attrs: { x: 0, y: top, width: bounding.width, height },
      styles: { style: "fill", color: fill },
    },
  ];
  const dash = (price, color) => {
    const y = yAxis?.convertToPixel?.(price);
    if (!Number.isFinite(y)) return;
    figures.push({
      type: "line",
      ignoreEvent: true,
      attrs: { coordinates: [{ x: 0, y }, { x: bounding.width, y }] },
      styles: { color, size: 1, style: "dashed", dashedValue: [4, 4] },
    });
  };
  if (data.stop != null) dash(data.stop, "#d4654f");
  for (const target of data.targets || []) dash(target, "#7dba6a");
  return figures;
}

export function registerBoardOverlays() {
  if (registered) return;
  registered = true;
  registerOverlay({
    name: "boardLevel",
    totalStep: 2,
    lock: true,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: levelFigures,
  });
  registerOverlay({
    name: "boardZone",
    totalStep: 3,
    lock: true,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: zoneFigures,
  });
}

export const BOARD_INTERVALS = new Set(["15m", "1h"]);
