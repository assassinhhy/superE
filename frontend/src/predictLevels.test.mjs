import assert from "node:assert/strict";
import test from "node:test";

import { formatLevels } from "./predictLevels.js";

test("no prediction row means nothing to annotate", () => {
  assert.equal(formatLevels(undefined), null);
});

test("levels present: entry, stop and both targets with signed percents", () => {
  const out = formatLevels({
    levels: {
      side: "long",
      entry: 67000,
      stop: 66200,
      targets: [67800, 68600],
      stop_pct: -0.0119,
      target_pct: [0.0119, 0.0239],
      backtested_targets: [true, false],
    },
  });
  assert.equal(out.muted, false);
  assert.equal(
    out.text,
    "入场 67000 · 止损 66200（-1.19%） · 止盈 67800（+1.19%，已回测口径，毛收益未扣费） / 68600（+2.39%，未回测）",
  );
});

test("levels null: muted line carries the Chinese reason", () => {
  assert.deepEqual(formatLevels({ levels: null, levels_reason: "观望，无方向" }), {
    text: "点位：无（观望，无方向）",
    muted: true,
  });
  assert.deepEqual(formatLevels({ levels: null, levels_reason: null }), {
    text: "点位：无",
    muted: true,
  });
});
