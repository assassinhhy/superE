// Pure helpers for the news panel. No DOM, no Vue.
// The API returns items already sorted newest-first; these helpers only format
// and classify. Direction is a scored sentiment in [-1, 1], not a trade signal.

function finite(v) {
  return v != null && Number.isFinite(Number(v));
}

export function directionClass(direction) {
  const d = Number(direction);
  if (!Number.isFinite(d)) return "neutral";
  if (d > 0.12) return "up";
  if (d < -0.12) return "down";
  return "neutral";
}

export function directionLabel(direction) {
  const cls = directionClass(direction);
  return cls === "up" ? "偏多" : cls === "down" ? "偏空" : "中性";
}

export function isUrgent(item) {
  return finite(item?.urgency) && Number(item.urgency) >= 0.7;
}

export function formatNewsRow(item) {
  const direction = finite(item?.direction) ? Number(item.direction) : 0;
  return {
    id: item?.id,
    time: item?.available_at_utc8 || item?.ts_ingested_utc8 || "—",
    source: item?.source || "unknown",
    title: item?.title || "(无标题)",
    url: item?.url || "",
    direction,
    directionClass: directionClass(direction),
    directionLabel: directionLabel(direction),
    urgency: finite(item?.urgency) ? Number(item.urgency) : 0,
    urgent: isUrgent(item),
    eventType: item?.event_type || "",
    scoredBy: item?.scored_by || "",
    timingQuality: item?.timing_quality || "",
  };
}

export function formatNewsRows(items) {
  return (items || []).map(formatNewsRow);
}

export function summaryLine(summary) {
  if (!summary) return "";
  if (summary.line) return String(summary.line);
  const count = Number(summary.count || 0);
  const high = Number(summary.high_urgency || 0);
  return `近 ${summary.window_hours || 24}h 新闻 ${count} 条，整体${summary.label || "中性"}`
    + (high ? `，高紧急度 ${high} 条` : "");
}
