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
  theblock: "The Block",
};

export const NEWS_FILTERS = ["全部", "利多", "利空", "中性", "只看重要"];
export const NEWS_POLL_MS = 5 * 60 * 1000;
export const NEWS_PAGE = 15;
export const NEWS_FOLD_MS = 6 * 60 * 60 * 1000;

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

export function relativeTime(ts, nowMs) {
  if (!finite(ts) || !finite(nowMs)) return "时间未知";
  const diff = Math.max(0, Number(nowMs) - Number(ts));
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  return `${Math.floor(hours / 24)} 天前`;
}

export function formatNewsRow(item, nowMs = Date.now()) {
  const direction = finite(item?.direction) ? Number(item.direction) : 0;
  const title = item?.title || "(无标题)";
  const summary = String(item?.summary_zh || item?.summary || "").trim();
  const ts = finite(item?.ts_published) ? Number(item.ts_published)
    : finite(item?.available_at) ? Number(item.available_at) : null;
  const importance = finite(item?.importance) ? Number(item.importance) : (isUrgent(item) ? 3 : 1);
  const urgent = isUrgent(item) || importance >= 3;
  return {
    id: item?.id,
    time: item?.available_at_utc8 || item?.ts_published_utc8 || item?.ts_ingested_utc8 || "—",
    ts,
    relative: relativeTime(ts, nowMs),
    source: sourceLabel(item?.source),
    title,
    url: item?.url || "",
    direction,
    directionClass: directionClass(direction),
    directionLabel: item?.sentiment_label || directionLabel(direction),
    summary: summary || `${title}。当前只有标题信息，具体细节和影响仍不确定。`,
    reason: String(item?.reason_zh || "").trim(),
    impact: String(item?.impact_zh || "").trim(),
    importance,
    urgent,
    important: importance >= 2 || urgent,
    urgency: finite(item?.urgency) ? Number(item.urgency) : 0,
    urgencyLabel: urgencyLabel(item?.urgency),
    eventType: item?.event_type || "",
    scoredBy: item?.scored_by || "",
    scoreVersion: item?.score_version || "",
    timingQuality: item?.timing_quality || "",
    contentHash: item?.content_hash || "",
  };
}

export function formatNewsRows(items, nowMs = Date.now()) {
  return (items || []).map((item) => formatNewsRow(item, nowMs));
}

export function filterNews(rows, mode) {
  if (mode === "利多" || mode === "利空" || mode === "中性") {
    return (rows || []).filter((row) => row.directionLabel === mode);
  }
  if (mode === "只看重要") return (rows || []).filter((row) => row.important);
  return (rows || []).slice();
}

export function importantRows(rows, limit = 5) {
  return (rows || [])
    .filter((row) => row.important)
    .sort((a, b) => (b.importance - a.importance) || ((b.ts || 0) - (a.ts || 0)))
    .slice(0, limit);
}

export function layoutNews(rows, nowMs, { limit = NEWS_PAGE, showOlder = false } = {}) {
  const recent = [];
  const older = [];
  for (const row of rows || []) {
    if (row?.ts && Number(nowMs) - Number(row.ts) > NEWS_FOLD_MS) older.push(row);
    else recent.push(row);
  }
  const shownRecent = recent.slice(0, limit);
  return {
    rows: showOlder ? shownRecent.concat(older) : shownRecent,
    hiddenRecent: Math.max(0, recent.length - shownRecent.length),
    olderCount: older.length,
    olderHidden: showOlder ? 0 : older.length,
  };
}

export function refreshCountdown(nextAtMs, nowMs) {
  if (!finite(nextAtMs) || !finite(nowMs)) return "—";
  const total = Math.floor((Number(nextAtMs) - Number(nowMs)) / 1000);
  if (total <= 0) return "即将刷新";
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `${hours} 小时 ${minutes} 分`;
  if (minutes > 0) return `${minutes} 分 ${seconds} 秒`;
  return `${seconds} 秒`;
}

export function refreshButtonState({ refreshing = false, cooldownUntilMs = 0, cooldownRemainingSec = 0, nowMs = Date.now() } = {}) {
  if (refreshing) return { disabled: true, spinning: true, label: "刷新中" };
  const until = Number(cooldownUntilMs) || 0;
  const remain = until > 0
    ? Math.max(0, Math.ceil((until - Number(nowMs)) / 1000))
    : Math.max(0, Math.ceil(Number(cooldownRemainingSec) || 0));
  if (remain > 0) return { disabled: true, spinning: false, label: `${remain} 秒后可刷新` };
  return { disabled: false, spinning: false, label: "刷新" };
}

export function summaryHeadline(summary) {
  if (!summary) return "最近 1 小时暂无消息";
  if (summary.headline) return String(summary.headline);
  const counts = summary.counts_1h || {};
  const bias = summary.bias_1h || "中性";
  return `最近 1 小时${bias} · 利多 ${counts.positive || 0} / 利空 ${counts.negative || 0} / 中性 ${counts.neutral || 0}`;
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
