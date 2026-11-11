import test from "node:test";
import assert from "node:assert/strict";
import {
  NEWS_POLL_MS,
  directionClass,
  directionLabel,
  filterNews,
  formatAlert,
  formatNewsRows,
  isUrgent,
  layoutNews,
  refreshButtonState,
  refreshCountdown,
  relativeTime,
  sourceLabel,
  summaryHeadline,
  summaryLine,
  urgencyLabel,
} from "./news.js";

test("source names are shown in Chinese where known", () => {
  assert.equal(sourceLabel("google_news"), "Google 新闻");
  assert.equal(sourceLabel("binance_announcement"), "币安公告");
  assert.equal(sourceLabel("cointelegraph"), "Cointelegraph");
});

test("sentiment labels are 利多 / 利空 / 中性", () => {
  assert.equal(directionClass(0.6), "up");
  assert.equal(directionClass(-0.6), "down");
  assert.equal(directionClass(0.01), "neutral");
  assert.equal(directionClass(null), "neutral");
  assert.equal(directionLabel(0.5), "利多");
  assert.equal(directionLabel(-1), "利空");
  assert.equal(directionLabel(0), "中性");
});

test("urgency at or above 0.7 is highlighted", () => {
  assert.equal(isUrgent({ urgency: 0.7 }), true);
  assert.equal(isUrgent({ urgency: 0.69 }), false);
  assert.equal(isUrgent({}), false);
  assert.equal(urgencyLabel(0.8), "高");
  assert.equal(urgencyLabel(0.5), "中");
  assert.equal(urgencyLabel(0.1), "低");
});

test("rows keep API order and expose title, badge and Chinese summary", () => {
  const rows = formatNewsRows([
    { id: 2, title: "ETH ETF inflow", summary_zh: "ETF 资金流入增加，可能提升 ETH 买盘。", direction: 0.5, urgency: 0.8, available_at_utc8: "2026-11-05 12:00", source: "gdelt" },
    { id: 1, title: "SEC delays decision", summary_zh: "SEC 推迟决定，短期不确定性上升。", direction: -0.5, urgency: 0.1, available_at_utc8: "2026-11-05 11:00", source: "binance" },
  ]);
  assert.deepEqual(rows.map((r) => r.id), [2, 1]);
  assert.equal(rows[0].directionClass, "up");
  assert.equal(rows[0].directionLabel, "利多");
  assert.equal(rows[0].summary, "ETF 资金流入增加，可能提升 ETH 买盘。");
  assert.equal(rows[0].urgent, true);
  assert.equal(rows[1].directionLabel, "利空");
});

test("missing summary falls back to an honest title-based sentence", () => {
  const [row] = formatNewsRows([{ title: "Some ETH headline", direction: 0 }]);
  assert.match(row.summary, /Some ETH headline/);
  assert.match(row.summary, /细节和影响仍不确定/);
});

test("summary line uses the API aggregate", () => {
  assert.equal(summaryLine({ line: "近 24h 共 3 条 ETH 相关消息，整体中性" }), "近 24h 共 3 条 ETH 相关消息，整体中性");
  assert.match(summaryLine({ count: 4, high_urgency: 1, label: "利多", window_hours: 24 }),
    /近 24h 共 4 条 ETH 相关消息，整体利多/);
});

test("alert normalizes the backend Chinese alert", () => {
  const alert = formatAlert({
    level: "high",
    title: "出现高紧急 ETH 相关消息",
    what: "「ETH ETF inflow」（利多）",
    why: "资金流向和买盘可能变化",
    urgency: "高",
    watch_window: "未来 1–4 小时",
    uncertain: true,
    note: "部分条目只有标题。",
    items: [{ title: "ETH ETF inflow", sentiment_label: "利多", urgency: 0.9 }],
  });
  assert.equal(alert.level, "high");
  assert.equal(alert.watchWindow, "未来 1–4 小时");
  assert.equal(alert.items[0].directionLabel, "利多");
});

test("filter keeps one sentiment or only important rows", () => {
  const now = 1_700_000_000_000;
  const rows = formatNewsRows([
    { id: 1, title: "a", summary_zh: "甲", sentiment_label: "利多", importance: 3, ts_published: now },
    { id: 2, title: "b", summary_zh: "乙", sentiment_label: "利空", importance: 1, ts_published: now },
    { id: 3, title: "c", summary_zh: "丙", sentiment_label: "中性", importance: 1, ts_published: now },
  ], now);
  assert.deepEqual(filterNews(rows, "全部").map((row) => row.id), [1, 2, 3]);
  assert.deepEqual(filterNews(rows, "利空").map((row) => row.id), [2]);
  assert.deepEqual(filterNews(rows, "只看重要").map((row) => row.id), [1]);
});

test("countdown and relative time use Chinese units", () => {
  const now = 1_700_000_000_000;
  assert.equal(refreshCountdown(now + (23 * 60 + 10) * 1000, now), "23 分 10 秒");
  assert.equal(refreshCountdown(now - 1000, now), "即将刷新");
  assert.equal(refreshCountdown(now + 90 * 60 * 1000, now), "1 小时 30 分");
  assert.equal(relativeTime(now - 23 * 60 * 1000, now), "23 分钟前");
  assert.equal(NEWS_POLL_MS, 300000);
});

test("refresh button shows spinning or remaining seconds", () => {
  const now = 1_700_000_000_000;
  assert.deepEqual(refreshButtonState({ refreshing: true, nowMs: now }), {
    disabled: true, spinning: true, label: "刷新中",
  });
  assert.deepEqual(refreshButtonState({ cooldownUntilMs: now + 42_000, nowMs: now }), {
    disabled: true, spinning: false, label: "42 秒后可刷新",
  });
  assert.deepEqual(refreshButtonState({ cooldownRemainingSec: 0, nowMs: now }), {
    disabled: false, spinning: false, label: "刷新",
  });
});

test("default list is 15 and news older than 6 hours stays folded", () => {
  const now = 1_700_000_000_000;
  const rows = [];
  for (let i = 0; i < 16; i += 1) {
    rows.push({ id: i, ts: now - i * 60_000, directionLabel: "中性", important: false });
  }
  rows.push({ id: 99, ts: now - 7 * 60 * 60 * 1000, directionLabel: "利空", important: false });
  const folded = layoutNews(rows, now);
  assert.equal(folded.rows.length, 15);
  assert.equal(folded.hiddenRecent, 1);
  assert.equal(folded.olderHidden, 1);
  const open = layoutNews(rows, now, { limit: 20, showOlder: true });
  assert.equal(open.rows.length, 17);
  assert.equal(open.olderHidden, 0);
});

test("headline prefers the API line", () => {
  assert.equal(
    summaryHeadline({ headline: "最近 1 小时偏空 · 利多 2 / 利空 5 / 中性 3" }),
    "最近 1 小时偏空 · 利多 2 / 利空 5 / 中性 3",
  );
});

test("missing alert is an explicit no-anomaly state, not invented news", () => {
  const alert = formatAlert(null);
  assert.equal(alert.level, "none");
  assert.equal(alert.title, "暂无显著异常");
  assert.equal(alert.what, "暂无足够的异常信息。");
  assert.equal(alert.items.length, 0);
});
