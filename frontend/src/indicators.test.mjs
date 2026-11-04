import assert from "node:assert/strict";
import test from "node:test";

import { loadOlderSeries } from "./api.js";
import {
  AXIS_PX,
  INTERVALS,
  MAIN_PANE_PX,
  MA_LINES,
  PREDICT_INTERVALS,
  SERIES_PAGE,
  SUB_PANE_PX,
  appendOlderPage,
  chartBodyHeight,
  asofSeries,
  atr,
  basisSeries,
  cvd,
  formatUtc8,
  isBackfilling,
  joinBarExtras,
  liquidationQuote,
  maCreate,
  mainPaneFit,
  metricPagesFor,
  paintDelay,
  toggleMember,
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

test("MA15 and MA25 are separate overlays, and the old periods are gone", () => {
  assert.deepEqual(MA_LINES.map((line) => line.period), [15, 25]);
  const ma15 = maCreate(MA_LINES[0]);
  const ma25 = maCreate(MA_LINES[1]);
  assert.deepEqual(ma15.calcParams, [15]);
  assert.deepEqual(ma25.calcParams, [25]);
  assert.notEqual(ma15.id, ma25.id);
  let on = new Set(["MA15", "MA25"]);
  on = toggleMember(on, "MA15");
  assert.equal(on.has("MA15"), false);
  assert.equal(on.has("MA25"), true);
  on = toggleMember(on, "MA25");
  assert.equal(on.has("MA15") || on.has("MA25"), false);
});

test("metrics paging stops when the server ignores end and repeats the latest page", () => {
  const latest = [{ ts: 300 }, { ts: 200 }, { ts: 100 }];
  const first = appendOlderPage([], latest, "ts");
  assert.equal(first.grew, true);
  assert.equal(first.cursor, 99);
  const repeated = appendOlderPage(first.rows, latest, "ts");
  assert.equal(repeated.grew, false);
  assert.equal(repeated.cursor, null);
  assert.equal(repeated.rows.length, 3);
  const older = appendOlderPage(first.rows, [{ ts: 50 }, { ts: 40 }], "ts");
  assert.equal(older.grew, true);
  assert.equal(older.cursor, 39);
  assert.equal(older.rows.length, 5);
});

// A fake /api/metrics that honours `end` the way the 1b backend does:
// newest `limit` rows with ts <= end, ascending.
function fakeMetrics(step, count) {
  const all = Array.from({ length: count }, (_, i) => ({ ts: i * step }));
  return (end) => {
    const rows = all.filter((row) => end == null || row.ts <= end);
    return rows.slice(-SERIES_PAGE);
  };
}

test("metrics window follows the chart batch instead of the newest page", async () => {
  const step = 300_000;
  const serve = fakeMetrics(step, 20_000);
  const windowEnd = 4_000 * step;
  const start = 3_000 * step;
  const rows = await loadOlderSeries(
    (cursor) => Promise.resolve(serve(cursor ?? windowEnd)),
    "ts",
    start,
    metricPagesFor(windowEnd - start),
  );
  assert.ok(rows.some((row) => row.ts === start), "the batch start must be covered");
  assert.ok(rows.every((row) => row.ts <= windowEnd), "no row after the batch end");
  // Old behaviour: no end, so the first page is the newest 1000 rows and this window is missed.
  const stale = await loadOlderSeries((cursor) => Promise.resolve(serve(cursor)), "ts", start, 8);
  assert.equal(stale.some((row) => row.ts === start), false);
});

test("metric page budget grows with the window and never drops to zero", () => {
  assert.equal(metricPagesFor(0), 1);
  assert.equal(metricPagesFor(-5), 1);
  // 5.2 days of 5m bars is about 1500 rows: two pages, plus the margin.
  assert.equal(metricPagesFor(5.2 * 86_400_000), 4);
  // 90 days of 5m bars is 25920 rows: 26 pages, plus the margin.
  assert.equal(metricPagesFor(90 * 86_400_000), 28);
});

test("six sub panes keep a full main pane instead of splitting the viewport", () => {
  const viewport = 720;
  const body = chartBodyHeight(6);
  assert.ok(body > viewport);
  assert.ok(body - 6 * 156 >= 340);
});

test("candle plus MACD fill the first screen and the other subs start below it", () => {
  const viewport = 900;
  const sep = 1;
  const main = mainPaneFit(viewport, { hasPrimarySub: true, separator: sep, axis: AXIS_PX });
  // The candle is clearly bigger than the old fixed 340px.
  assert.ok(main >= 600, `candle ${main}px should dominate the first screen`);
  // Candle, separator, MACD and the axis reserve add up to the viewport: nothing else is visible above the fold.
  assert.equal(main + sep + SUB_PANE_PX + AXIS_PX, viewport);
  // With seven more subs (RSI, ATR, OI, funding, LSR, taker, plus MACD) the body grows past the viewport and scrolls.
  const body = chartBodyHeight(8, { separator: sep, axis: AXIS_PX, mainPx: main });
  assert.ok(body > viewport);
  // The MACD bottom edge sits at the fold: main + separator + MACD height = viewport - axis.
  assert.equal(main + sep + SUB_PANE_PX, viewport - AXIS_PX);
});

test("candle keeps its floor on a short viewport and fills the screen without MACD", () => {
  assert.equal(mainPaneFit(400, { hasPrimarySub: true }), MAIN_PANE_PX);
  assert.equal(mainPaneFit(0, { hasPrimarySub: true }), MAIN_PANE_PX);
  assert.equal(mainPaneFit(900, { hasPrimarySub: false, axis: AXIS_PX }), 900 - AXIS_PX);
});
