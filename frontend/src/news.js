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
  return cls === "up" ? "利多" : cls === "down" ? "利空" : "中性";
}

export function isUrgent(item) {
  return finite(item?.urgency) && Number(item.urgency) >= 0.7;
}

const SOURCE_CN = {
  binance_announcement: "币安公告",
  google_news: "Google 新闻",
  coindesk: "CoinDesk",
  cointelegraph: "Cointelegraph",
  decrypt: "Decrypt",
};

export function sourceLabel(source) {
  const name = String(source || "unknown");
  return SOURCE_CN[name] || name;
}

export function urgencyLabel(urgency) {
  const value = finite(urgency) ? Number(urgency) : 0;
  if (value >= 0.7) return "高";
  if (value >= 0.4) return "中";
  return "低";
}

export function formatNewsRow(item) {
  const direction = finite(item?.direction) ? Number(item.direction) : 0;
  const title = item?.title || "(无标题)";
  const summary = String(item?.summary_zh || item?.summary || "").trim();
  return {
    id: item?.id,
    time: item?.available_at_utc8 || item?.ts_published_utc8 || item?.ts_ingested_utc8 || "—",
    source: sourceLabel(item?.source),
    title,
    url: item?.url || "",
    direction,
    directionClass: directionClass(direction),
    directionLabel: item?.sentiment_label || directionLabel(direction),
    summary: summary || `${title}。当前只有标题信息，具体细节和影响仍不确定。`,
    urgency: finite(item?.urgency) ? Number(item.urgency) : 0,
    urgencyLabel: urgencyLabel(item?.urgency),
    urgent: isUrgent(item),
    eventType: item?.event_type || "",
    scoredBy: item?.scored_by || "",
    scoreVersion: item?.score_version || "",
    timingQuality: item?.timing_quality || "",
    contentHash: item?.content_hash || "",
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
  return `近 ${summary.window_hours || 24}h 共 ${count} 条 ETH 相关消息，整体${summary.label || "中性"}`
    + (high ? `；其中高紧急度 ${high} 条` : "");
}

// The backend builds the alert from observed headlines.  This helper only
// normalizes fields and supplies an honest empty state; it never invents news.
export function formatAlert(alert) {
  if (!alert) {
    return {
      level: "none",
      title: "暂无显著异常",
      what: "暂无足够的异常信息。",
      why: "",
      consistency: "",
      sources: "",
      recency: "",
      urgency: "低",
      watchWindow: "常规关注即可",
      uncertain: false,
      note: "",
      items: [],
    };
  }
  return {
    level: alert.level || "none",
    title: alert.title || (alert.level === "none" ? "暂无显著异常" : "有值得关注的消息"),
    what: alert.what || "",
    why: alert.why || "",
    consistency: alert.consistency || "",
    sources: (alert.sources || "").split("、").filter(Boolean).map(sourceLabel).join("、"),
    recency: alert.recency || "",
    urgency: alert.urgency || "低",
    watchWindow: alert.watch_window || "",
    uncertain: Boolean(alert.uncertain),
    note: alert.note || "",
    items: (alert.items || []).map(formatNewsRow),
  };
}
