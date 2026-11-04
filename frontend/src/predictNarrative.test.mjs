import assert from "node:assert/strict";
import test from "node:test";

import { buildNarrative } from "./predictNarrative.js";

const LEVELS = {
  side: "long",
  entry: 67000,
  stop: 66200,
  targets: [67800, 68600],
  stop_pct: -0.0119,
  target_pct: [0.0119, 0.0239],
  backtested_targets: [true, false],
};

function row(interval, over = {}) {
  return {
    interval,
    status: "ok",
    direction: "flat",
    probs: { down: 0.2, flat: 0.5, up: 0.3 },
    levels: null,
    levels_reason: "观望：没有方向概率过阈值，不给点位",
    threshold: 0.4,
    oos_accuracy: 0.41,
    message: "调试预测，样本外弱，勿实盘",
    context: {
      ref_close: 67010,
      window_bars: 20,
      high: 67500,
      low: 66400,
      funding: { rate: 0.0001, time_utc8: "2026-11-05 00:00" },
    },
    ...over,
  };
}

function textOf(out, key) {
  return out.lines.find((l) => l.key === key).text;
}

test("all flat: conclusion says wait and no levels are given", () => {
  const items = ["5m", "15m", "30m", "1h", "4h"].map((i) => row(i));
  const out = buildNarrative({ items, price: "67010" });
  assert.equal(out.lead, null);
  assert.match(textOf(out, "conclusion"), /先观望，不给点位/);
  assert.equal(textOf(out, "levels"), "点位：观望，不给进场、止损、止盈。");
  assert.deepEqual(
    out.lines.map((l) => l.key),
    ["conclusion", "price", "intervals", "funding", "levels", "leverage", "debug"],
  );
});

test("one up row leads: conclusion, key levels and the levels line", () => {
  const up = row("1h", {
    direction: "up",
    probs: { down: 0.1, flat: 0.45, up: 0.45 },
    levels: LEVELS,
    levels_reason: null,
  });
  const items = [row("5m"), row("15m"), row("30m"), up, row("4h")];
  const out = buildNarrative({ items, price: "67010" });
  assert.deepEqual(out.lead, { interval: "1h", direction: "up" });
  assert.equal(textOf(out, "conclusion"), "结论：偏多，主看 1h（多 45.0%）。");
  assert.equal(textOf(out, "price"), "现价 67010；关键位（1h 近 20 根）：高 67500 / 低 66400。");
  assert.equal(
    textOf(out, "levels"),
    "点位（1h）：入场 67000 · 止损 66200（-1.19%） · 止盈 67800（+1.19%，已回测口径，毛收益未扣费） / 68600（+2.39%，未回测）",
  );
  assert.match(textOf(out, "intervals"), /1h 多（45%）/);
});

test("opposite directions are called out as a disagreement", () => {
  const items = [
    row("5m", { direction: "down", probs: { down: 0.5, flat: 0.2, up: 0.3 } }),
    row("1h", { direction: "up", probs: { down: 0.2, flat: 0.25, up: 0.55 }, levels: null, levels_reason: "x" }),
  ];
  const out = buildNarrative({ items, price: "1" });
  assert.match(textOf(out, "conclusion"), /分歧，可信度低/);
  assert.equal(out.lead.interval, "1h");
});

test("funding sign tells who pays, and missing funding is said plainly", () => {
  const neg = buildNarrative({
    items: [row("1h", { context: { ...row("1h").context, funding: { rate: -0.0002, time_utc8: "t" } } })],
    price: "1",
  });
  assert.match(textOf(neg, "funding"), /空头付给多头/);
  assert.match(textOf(neg, "funding"), /-0\.0200%/);

  const none = buildNarrative({
    items: [row("1h", { context: { ...row("1h").context, funding: null } })],
    price: "1",
  });
  assert.equal(textOf(none, "funding"), "资金面：没有可用的资金费率数据。");
});

test("no usable rows: conclusion explains why and nothing crashes", () => {
  const stale = { interval: "5m", status: "stale", direction: null, message: "数据已过期" };
  const out = buildNarrative({ items: [stale, undefined], price: "—" });
  assert.match(textOf(out, "conclusion"), /没有可用的预测（数据已过期）/);
  assert.equal(textOf(out, "debug"), "调试勿实盘：模型样本外弱，没有可用的样本外准确率。");
  assert.equal(buildNarrative({ items: undefined, price: "—" }).lines.length, 7);
});

test("every narrative keeps the debug and leverage lines", () => {
  const out = buildNarrative({ items: [row("5m")], price: "1" });
  assert.match(textOf(out, "debug"), /^调试勿实盘/);
  assert.match(textOf(out, "leverage"), /^高杠杆提醒/);
});
