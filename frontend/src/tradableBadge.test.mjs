import assert from "node:assert/strict";
import test from "node:test";

import { TRADABLE_FALLBACK_TEXT, tradableBadge } from "./tradableBadge.js";

test("tradable=false shows the API status text and reason", () => {
  assert.deepEqual(
    tradableBadge({ tradable: false, tradable_status: "未达标/仅参考", tradable_reason: "样本外准确率低于门槛" }),
    { text: "未达标/仅参考", reason: "样本外准确率低于门槛" },
  );
});

test("tradable=false without a status falls back to the fixed text", () => {
  assert.deepEqual(tradableBadge({ tradable: false }), { text: TRADABLE_FALLBACK_TEXT, reason: "" });
  assert.equal(TRADABLE_FALLBACK_TEXT, "未达标/仅参考");
  assert.deepEqual(tradableBadge({ tradable: false, tradable_status: "   " }), {
    text: TRADABLE_FALLBACK_TEXT,
    reason: "",
  });
});

test("tradable=true shows no badge, even with a status string", () => {
  assert.equal(tradableBadge({ tradable: true, tradable_status: "可用（已达标）", tradable_reason: "" }), null);
});

test("missing tradable field keeps the old display (no badge)", () => {
  assert.equal(tradableBadge({ status: "ok", direction: "up" }), null);
  assert.equal(tradableBadge({ tradable_status: "未达标/仅参考", tradable_reason: "x" }), null);
});

test("no prediction row means no badge", () => {
  assert.equal(tradableBadge(undefined), null);
  assert.equal(tradableBadge(null), null);
});

test("non-string status or reason are ignored", () => {
  assert.deepEqual(tradableBadge({ tradable: false, tradable_status: 1, tradable_reason: null }), {
    text: TRADABLE_FALLBACK_TEXT,
    reason: "",
  });
});
