import assert from "node:assert/strict";
import test from "node:test";

import {
  INTERVALS,
  PREDICT_INTERVALS,
  asofSeries,
  atr,
  basisSeries,
  cvd,
  formatUtc8,
  isBackfilling,
  joinBarExtras,
  liquidationQuote,
  paintDelay,
  vwap,
} from "./indicators.js";

test("the chart lists every UM interval, including the nine prediction periods", () => {
  const ids = INTERVALS.map((item) => item.id);
  assert.deepEqual(ids, [
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M",
  ]);
  for (const id of PREDICT_INTERVALS) assert.ok(ids.includes(id));
});

test("UTC+8 formatting does not follow the machine zone", () => {
  assert.equal(formatUtc8(0), "1970-01-01 08:00:00");
  assert.equal(formatUtc8(60_000), "1970-01-01 08:01:00");
});

test("Wilder ATR matches a hand calculation", () => {
  const bars = [
    { high: 10, low: 8, close: 9 },
    { high: 11, low: 9, close: 10 },
    { high: 12, low: 9, close: 11 },
    { high: 11, low: 10, close: 10 },
  ];
  const values = atr(bars, 3);
  assert.equal(values[0], null);
  assert.equal(values[1], null);
  assert.ok(Math.abs(values[2] - 7 / 3) < 1e-12);
  assert.ok(Math.abs(values[3] - 17 / 9) < 1e-12);
});

test("VWAP resets on the UTC day boundary", () => {
  const bars = [
    { timestamp: 0, high: 3, low: 1, close: 2, volume: 2 },
    { timestamp: 60_000, high: 3, low: 1, close: 2, volume: 2 },
    { timestamp: 86_400_000, high: 6, low: 0, close: 3, volume: 1 },
  ];
  assert.deepEqual(vwap(bars), [2, 2, 3]);
});

test("CVD is taker buy minus taker sell", () => {
  assert.deepEqual(
    cvd([
      { volume: 10, takerBuy: 6 },
      { volume: 10, takerBuy: 4 },
    ]),
    [2, 0],
  );
});

test("metric values stay hidden until five minutes after the bar start", () => {
  const bars = [
    { closeTime: 299_999 },
    { closeTime: 300_000 },
    { closeTime: 600_000 },
  ];
  const points = [
    { ts: 0, value: 5 },
    { ts: 300_000, value: 9 },
  ];
  assert.deepEqual(asofSeries(bars, points, 300_000), [null, 5, 9]);
});

test("basis ignores a spot bar that opens after the contract bar closes", () => {
  const bars = [{ timestamp: 0, closeTime: 59_999, close: 10 }];
  const spot = [
    { timestamp: 0, close: 8 },
    { timestamp: 60_000, close: 1 },
  ];
  assert.deepEqual(basisSeries(bars, spot), [2]);
});

test("liquidation quote stays inside the bar", () => {
  const bars = [{ timestamp: 0, closeTime: 59_999 }];
  const events = [
    { ts: 10, price: "100", qty: "2" },
    { ts: 60_000, price: "100", qty: "9" },
  ];
  assert.deepEqual(liquidationQuote(bars, events), [200]);
});

test("joined extras do not use a future metric or a future funding print", () => {
  const bars = [{ timestamp: 0, closeTime: 60_000, open: 1, high: 2, low: 1, close: 10, volume: 1, takerBuy: 1 }];
  const [bar] = joinBarExtras(bars, {
    metrics: [{ ts: 0, oi: "5", global_lsr: "1.1" }],
    funding: [
      { funding_time: 0, rate: "0.0001" },
      { funding_time: 120_000, rate: "0.9" },
    ],
    spot: [{ open_time: 0, c: "8" }],
    premium: [{ open_time: 0, c: "0.0002" }],
    liquidations: [],
  });
  assert.equal(bar.oi, null);
  assert.equal(bar.funding, 0.0001);
  assert.equal(bar.basis, 2);
  assert.equal(bar.premium, 0.0002);
});

test("backfill banner ignores the standing BACKFILL_REQUIRED notice", () => {
  const now = 1_000_000;
  assert.equal(isBackfilling([{ code: "BACKFILL_REQUIRED", ts: now - 1000 }], now), false);
  assert.equal(isBackfilling([{ code: "GAP_FILL", ts: now - 1000 }], now), true);
  assert.equal(isBackfilling([{ code: "GAP_FILL", ts: now - 130_000 }], now), false);
});

test("redraw throttle waits out the remainder of 250ms", () => {
  assert.equal(paintDelay(1000, 800), 50);
  assert.equal(paintDelay(1100, 800), 0);
});
