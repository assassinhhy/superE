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
    status: "达标",
    explain: "中位数略高于现价，区间偏窄。",
    top_features: [],
  },
};

test("main row uses rounded median and p25-p75", () => {
  const row = forecastRow("1h", okItem);
  assert.equal(row.hasForecast, true);
  assert.equal(row.headline, "1h后 ≈ 2352（2345–2360）");
  assert.equal(row.status, "达标");
  assert.equal(row.explain, "中位数略高于现价，区间偏窄。");
  assert.equal(row.detail.interval80, "80%区间：2330–2373");
  assert.equal(row.detail.width50, "50%区间宽度：12.3 bps");
  assert.equal(row.detail.width80, "80%区间宽度：30.1 bps");
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

test("仅参考 status is passed through unchanged", () => {
  const item = { ...okItem, price_forecast: { ...okItem.price_forecast, status: "仅参考" } };
  assert.equal(forecastRow("1h", item).status, "仅参考");
});

test("forecastRows keeps the given interval order and fills gaps", () => {
  const rows = forecastRows([okItem, { interval: "5m", status: "no_model", message: "未接入" }], INTERVALS);
  assert.deepEqual(rows.map((r) => r.interval), INTERVALS);
  assert.equal(rows[0].headline, "5m：暂无预测");
  assert.equal(rows[3].headline, "1h后 ≈ 2352（2345–2360）");
  assert.equal(rows[4].headline, "4h：暂无预测");
});
