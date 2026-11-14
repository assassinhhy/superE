import test from "node:test";
import assert from "node:assert/strict";
import { loadNotify, rememberNotice } from "./notify.js";

test("the same notice id is not fresh twice", () => {
  const first = rememberNotice([], "5m:macd_golden:10");
  assert.equal(first.fresh, true);
  const second = rememberNotice(first.seen, "5m:macd_golden:10");
  assert.equal(second.fresh, false);
  assert.equal(rememberNotice(second.seen, "news:abc").fresh, true);
});

test("sound defaults on and a stored off switch is kept", () => {
  const storage = {
    raw: JSON.stringify({ sound: false, seen: ["a"] }),
    getItem() { return this.raw; },
  };
  assert.equal(loadNotify(storage).sound, false);
  assert.deepEqual(loadNotify(storage).seen, ["a"]);
  assert.equal(loadNotify({ getItem() { return ""; } }).sound, true);
});
