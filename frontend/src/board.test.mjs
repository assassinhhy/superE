import test from "node:test";
import assert from "node:assert/strict";
import { boardNotice, bumpLoss, collectBoardEvents, lossState } from "./board.js";
import { wantsSystemNotice } from "./notify.js";

test("board notices stay on the page and reuse a bucket inside 30 minutes", () => {
  const first = boardNotice({ planId: "trend", kind: "confirm", title: "顺势", body: "确认" }, 0);
  const again = boardNotice({ planId: "trend", kind: "confirm", title: "顺势", body: "确认" }, 10 * 60 * 1000);
  const later = boardNotice({ planId: "trend", kind: "confirm", title: "顺势", body: "确认" }, 31 * 60 * 1000);
  assert.equal(first.system, false);
  assert.equal(wantsSystemNotice(first), false);
  assert.equal(wantsSystemNotice({ title: "信号" }), true);
  assert.equal(first.id, again.id);
  assert.notEqual(first.id, later.id);
});

test("confirm, flip, anomaly and the calendar window each emit once", () => {
  const next = {
    crosscheck: { status: "anomaly", message: "数据异常：币安 1 / OKX 2" },
    plans: [
      { id: "trend", title: "顺势反抽空", side: "short", status: "可进场", confirm: { names: ["长影线"] } },
      { id: "flip", title: "翻转", status: "已触发", note: "作废" },
    ],
    risk: { events: [{ ts_event: 10, kind: "PPI", title: "美国 PPI", blocking: true, window: "20:15–20:45" }] },
  };
  const events = collectBoardEvents(null, next);
  assert.deepEqual(events.map((item) => item.kind).sort(), ["anomaly", "calendar", "confirm", "flip"]);
  const again = collectBoardEvents(next, next);
  assert.equal(again.length, 0);
});

test("the loss count resets at UTC+8 midnight", () => {
  const before = Date.parse("2026-11-14T15:59:00Z");
  const after = Date.parse("2026-11-14T16:01:00Z");
  const stored = bumpLoss(null, 2, before);
  assert.equal(stored.losses, 2);
  assert.equal(stored.day, "2026-11-14");
  assert.equal(lossState(stored, after).losses, 0);
  assert.equal(bumpLoss(stored, 1, after).day, "2026-11-15");
});
