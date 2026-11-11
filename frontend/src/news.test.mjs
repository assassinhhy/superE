import test from "node:test";
import assert from "node:assert/strict";
import {
  directionClass,
  directionLabel,
  formatAlert,
  formatNewsRows,
  isUrgent,
  sourceLabel,
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

test("missing alert is an explicit no-anomaly state, not invented news", () => {
  const alert = formatAlert(null);
  assert.equal(alert.level, "none");
  assert.equal(alert.title, "暂无显著异常");
  assert.equal(alert.what, "暂无足够的异常信息。");
  assert.equal(alert.items.length, 0);
});
