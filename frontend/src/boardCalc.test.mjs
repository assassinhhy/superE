import test from "node:test";
import assert from "node:assert/strict";
import { estimatePosition } from "./boardCalc.js";

test("notional, margin and both liquidation prices", () => {
  const shared = { entry: 100, leverage: 20, mmr: 0.005, maxLossU: 4, capital: 100, targets: [] };
  const long = estimatePosition({ ...shared, side: "long", stop: 98, atr: 1 });
  const short = estimatePosition({ ...shared, side: "short", stop: 102, atr: 1 });
  assert.equal(long.stopPct, 0.02);
  assert.equal(long.notional, 200);
  assert.equal(long.margin, 10);
  assert.ok(Math.abs(long.liq - 95.5) < 1e-9);
  assert.ok(Math.abs(short.liq - 104.5) < 1e-9);
});

test("liquidation must sit outside the stop by at least one ATR", () => {
  const base = { entry: 100, side: "short", leverage: 20, mmr: 0.005, maxLossU: 4, targets: [96] };
  const ok = estimatePosition({ ...base, stop: 103, atr: 1.5 });
  assert.equal(ok.warnings.includes("杠杆过高/保证金不足"), false);
  const tight = estimatePosition({ ...base, stop: 103, atr: 1.6 });
  assert.equal(tight.warnings.includes("杠杆过高/保证金不足"), true);
  const inside = estimatePosition({ ...base, side: "long", stop: 95, atr: 1, targets: [110] });
  assert.equal(inside.warnings.includes("杠杆过高/保证金不足"), true);
});

test("reward below 1 is called out, and a fee that eats the target is called out", () => {
  const poor = estimatePosition({
    entry: 100, stop: 102, side: "short", leverage: 20, mmr: 0.005,
    maxLossU: 4, atr: 0.1, targets: [99], takerFee: 0.0005,
  });
  assert.equal(poor.legs[0].rr, 0.5);
  assert.equal(poor.warnings.includes("不划算"), true);
  const feeBound = estimatePosition({
    entry: 100, stop: 101, side: "short", leverage: 20, mmr: 0.005,
    maxLossU: 4, atr: 0.01, targets: [99.9], takerFee: 0.0005, rrMin: 0,
  });
  assert.equal(feeBound.warnings.includes("不划算"), false);
  assert.equal(feeBound.warnings.includes("波动太小，手续费吃利润"), true);
  const exact = estimatePosition({
    entry: 100, stop: 102, side: "short", leverage: 20, mmr: 0.005,
    maxLossU: 4, atr: 0.1, targets: [98], rrMin: 1,
  });
  assert.equal(exact.legs[0].rr, 1);
  assert.equal(exact.warnings.includes("不划算"), false);
});

test("half size halves the risk budget, and leverage above 20 is marked", () => {
  const half = estimatePosition({
    entry: 100, stop: 102, side: "short", maxLossU: 4, sizeRatio: 0.5, targets: [],
  });
  assert.equal(half.riskU, 2);
  assert.equal(half.notional, 100);
  const hot = estimatePosition({
    entry: 100, stop: 102, side: "short", leverage: 50, leverageMax: 50, targets: [], atr: null,
  });
  assert.equal(hot.notes.includes("杠杆高于 20x，强平更靠近止损"), true);
  assert.equal(hot.warnings.includes("杠杆超过上限"), false);
});
