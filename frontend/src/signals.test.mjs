import test from "node:test";
import assert from "node:assert/strict";
import {
  cooldownMs,
  filterSignals,
  ingestInterval,
  packBars,
  signalsAt,
  emptySignalState,
} from "./signals.js";

function bar(index, close, extra = {}) {
  const timestamp = index * 300_000;
  return {
    timestamp,
    closeTime: timestamp + 299_999,
    open: close,
    high: close + 1,
    low: close - 1,
    close,
    ...extra,
  };
}

test("a close through MA20 is a cross, and the next bar of the same type waits out the cooldown", () => {
  const bars = [];
  for (let i = 0; i < 30; i += 1) bars.push(bar(i, 100));
  bars.push(bar(30, 130));
  const pack = packBars(bars);
  const found = signalsAt(bars, 30, pack).map((item) => item.type);
  assert.ok(found.includes("px_up_ma20"));
  const first = ingestInterval(emptySignalState(), "5m", bars, bars[30].closeTime + 1);
  assert.equal(first.state.items.some((item) => item.type === "px_up_ma20"), true);
  const again = bars.concat([bar(31, 140)]);
  const second = ingestInterval(first.state, "5m", again, again[31].closeTime + 1);
  assert.equal(second.announce.some((item) => item.type === "px_up_ma20"), false);
  assert.equal(cooldownMs("5m"), 10 * 60 * 1000);
});

test("MACD golden cross and ATR expansion fire only on the close that crosses", () => {
  const bars = [];
  for (let i = 0; i < 40; i += 1) bars.push(bar(i, 100));
  for (let i = 40; i < 55; i += 1) bars.push(bar(i, 100 + (i - 39) * 2));
  const pack = packBars(bars);
  let golden = 0;
  for (let i = 1; i < bars.length; i += 1) {
    golden += signalsAt(bars, i, pack).filter((item) => item.type === "macd_golden").length;
  }
  assert.equal(golden, 1);

  const calm = [];
  for (let i = 0; i < 40; i += 1) calm.push(bar(i, 100, { high: 100.2, low: 99.8 }));
  calm.push(bar(40, 100, { high: 120, low: 80 }));
  const expanded = signalsAt(calm, 40, packBars(calm)).map((item) => item.type);
  assert.ok(expanded.includes("atr_expand"));
  assert.equal(signalsAt(calm, 39, packBars(calm)).some((item) => item.type === "atr_expand"), false);
});

test("RSI divergence is optional and a forming bar is ignored", () => {
  const bars = [];
  for (let i = 0; i < 30; i += 1) bars.push(bar(i, 100));
  const now = bars[19].closeTime + 1;
  const forming = bars.slice(0, 20).concat([{ ...bar(20, 80), closeTime: now + 300_000 }]);
  const silent = ingestInterval(emptySignalState(), "15m", forming, now, { divergence: true });
  assert.equal(silent.state.seenOpen["15m"], bars[19].timestamp);

  const shaped = bars.map((item, index) => {
    if (index === 14) return bar(index, 80, { low: 70, high: 82 });
    if (index === 18) return bar(index, 120, { low: 60, high: 122 });
    if (index > 14 && index < 18) return bar(index, 90 + (index - 14) * 8);
    return item;
  });
  const at = 21;
  const withDiv = signalsAt(shaped, at, packBars(shaped), { divergence: true }).map((item) => item.type);
  const without = signalsAt(shaped, at, packBars(shaped), { divergence: false }).map((item) => item.type);
  assert.equal(without.includes("rsi_div_bull"), false);
  assert.ok(withDiv.includes("rsi_div_bull"));
});

test("filters keep interval and group", () => {
  const items = [
    { id: "a", interval: "5m", group: "RSI" },
    { id: "b", interval: "1h", group: "MACD" },
  ];
  assert.deepEqual(filterSignals(items, "1h", "全部").map((item) => item.id), ["b"]);
  assert.deepEqual(filterSignals(items, "全部", "RSI").map((item) => item.id), ["a"]);
});
