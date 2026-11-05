import test from "node:test";
import assert from "node:assert/strict";
import {
  directionClass,
  directionLabel,
  formatNewsRows,
  isUrgent,
  summaryLine,
} from "./news.js";

test("direction colors follow the scored sentiment sign", () => {
  assert.equal(directionClass(0.6), "up");
  assert.equal(directionClass(-0.6), "down");
  assert.equal(directionClass(0.01), "neutral");
  assert.equal(directionClass(null), "neutral");
  assert.equal(directionLabel(-1), "偏空");
});

test("urgency at or above 0.7 is highlighted", () => {
  assert.equal(isUrgent({ urgency: 0.7 }), true);
  assert.equal(isUrgent({ urgency: 0.69 }), false);
  assert.equal(isUrgent({}), false);
});

test("rows keep API order and expose display fields", () => {
  const rows = formatNewsRows([
    { id: 2, title: "b", direction: 0.5, urgency: 0.8, available_at_utc8: "2026-11-05 12:00", source: "gdelt" },
    { id: 1, title: "a", direction: -0.5, urgency: 0.1, available_at_utc8: "2026-11-05 11:00", source: "binance" },
  ]);
  assert.deepEqual(rows.map((r) => r.id), [2, 1]);
  assert.equal(rows[0].directionClass, "up");
  assert.equal(rows[0].urgent, true);
  assert.equal(rows[1].directionLabel, "偏空");
});

test("summary line uses the API aggregate", () => {
  assert.equal(summaryLine({ line: "近 24h 新闻 3 条，整体中性" }), "近 24h 新闻 3 条，整体中性");
  assert.match(summaryLine({ count: 4, high_urgency: 1, label: "偏多", window_hours: 24 }),
    /近 24h 新闻 4 条，整体偏多，高紧急度 1 条/);
});
