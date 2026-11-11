import { test } from "node:test";
import assert from "node:assert/strict";
import { forecastRow, forecastRows } from "./forecastRows.js";

const INTERVALS = ["5m", "15m", "30m", "1h", "4h"];

const okItem = {
  interval: "1h",
  status: "ok",
  message: "",
  price_forecast: {
    interval: "1h",
    asof_utc8: "2026-11-05 10:00",
    horizon_end_utc8: "2026-11-05 11:00",
    ref_close: 2340.12,
    median: 2352.3,
    p10: 2330.4,
    p25: 2345.2,
    p75: 2360.4,
    p90: 2372.9,
    width50_bps: 12.3,
    width80_bps: 30.1,
    coverage_ok: true,
    quality_pass: false,
    explain: "中位数略高于现价，区间偏窄。",
    top_features: [],
  },
  news_overlay: {
    applied: true,
    u: 0.8,
    d: 0.6,
    r: 0.9,
    gate: { threshold: 0.15, u_r: 0.72, triggered: true },
    width_multiplier: 1.252,
    median_delta: 4.2,
    median_delta_cap: 9.5,
    width50_bps: 15.4,
    width80_bps: 37.7,
    adjusted: { median: 2356.5, p10: 2329.1, p25: 2343.0, p75: 2370.0, p90: 2383.9 },
    counts: { count: 4, high_urgency_1h: 1, high_urgency_4h: 2, high_urgency_24h: 3 },
    top_news: [
      { id: 1, title: "ETH ETF inflow", source: "coindesk", time_utc8: "2026-11-05 09:40", sentiment_label: "利多", direction: 0.8, urgency: 0.9, relevance: 0.9, contribution: 0.7, url: "https://example.com/1" },
    ],
  },
};

test("raw model row keeps median and p25-p75", () => {
  const row = forecastRow("1h", okItem);
  assert.equal(row.hasForecast, true);
  assert.equal(row.raw.headline, "1h后 ≈ 2352（2345–2360）");
  assert.equal(row.raw.interval80, "80%区间：2330–2373");
  assert.equal(row.raw.width50, "50%区间宽度：12.3 bps");
  assert.equal(row.raw.width80, "80%区间宽度：30.1 bps");
  assert.equal(row.explain, "中位数略高于现价，区间偏窄。");
});

test("adjusted row is shown separately from the model row", () => {
  const row = forecastRow("1h", okItem);
  assert.ok(row.adjusted);
  assert.equal(row.adjusted.headline, "1h后 ≈ 2357（2343–2370）");
  assert.equal(row.adjusted.applied, true);
  assert.equal(row.adjusted.widthMultiplier, 1.252);
  assert.equal(row.overlay.u, 0.8);
  assert.equal(row.topNews.length, 1);
  assert.equal(row.topNews[0].label, "利多");
});

test("no news overlay keeps the raw row and reports no adjusted row", () => {
  const item = { ...okItem, news_overlay: null };
  const row = forecastRow("1h", item);
  assert.equal(row.adjusted, null);
  assert.equal(row.overlay, null);
  assert.equal(row.topNews.length, 0);
});

test("gate below threshold shows an adjusted copy with the same numbers", () => {
  const item = {
    ...okItem,
    news_overlay: {
      ...okItem.news_overlay,
      applied: false,
      u: 0.1,
      r: 0.2,
      width_multiplier: 1,
      median_delta: 0,
      adjusted: { median: 2352.3, p10: 2330.4, p25: 2345.2, p75: 2360.4, p90: 2372.9 },
    },
  };
  const row = forecastRow("1h", item);
  assert.equal(row.adjusted.applied, false);
  assert.equal(row.adjusted.median, 2352.3);
});

test("non-ok status shows no-forecast row with item message", () => {
  const row = forecastRow("4h", { interval: "4h", status: "stale", message: "数据过期，等待下一根 K 线" });
  assert.equal(row.hasForecast, false);
  assert.equal(row.headline, "4h：暂无预测");
  assert.equal(row.message, "数据过期，等待下一根 K 线");
});

test("null price_forecast shows no-forecast row", () => {
  const row = forecastRow("5m", { interval: "5m", status: "ok", message: "", price_forecast: null });
  assert.equal(row.hasForecast, false);
  assert.equal(row.headline, "5m：暂无预测");
});

test("missing item shows no-forecast row, never NaN", () => {
  const row = forecastRow("15m", undefined);
  assert.equal(row.headline, "15m：暂无预测");
  assert.ok(!row.headline.includes("NaN"));
});

test("unreasonable (non-finite) quantiles are treated as no forecast", () => {
  const bad = { ...okItem, price_forecast: { ...okItem.price_forecast, median: null } };
  assert.equal(forecastRow("1h", bad).hasForecast, false);
});

test("no user-facing quality wording is passed through", () => {
  const row = forecastRow("1h", okItem);
  assert.ok(!("status" in row));
  assert.ok(!JSON.stringify(row).includes("仅参考"));
  assert.ok(!JSON.stringify(row).includes("供参考"));
});

test("forecastRows keeps the given interval order and fills gaps", () => {
  const rows = forecastRows([okItem, { interval: "5m", status: "no_model", message: "未接入" }], INTERVALS);
  assert.deepEqual(rows.map((r) => r.interval), INTERVALS);
  assert.equal(rows[0].headline, "5m：暂无预测");
  assert.equal(rows[3].headline, "1h后 ≈ 2352（2345–2360）");
  assert.equal(rows[4].headline, "4h：暂无预测");
});
