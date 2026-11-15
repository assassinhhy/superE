import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  applyBoardPoll,
  basisStamp,
  boardClock,
  boardFreshLine,
  boardNotice,
  boardStaleSeconds,
  bumpLoss,
  collectBoardEvents,
  loadPlanHistory,
  lossState,
  planHistoryExportURL,
  planOutcomeLine,
  planValidity,
} from "./board.js";
import { wantsSystemNotice } from "./notify.js";

const SRC = dirname(fileURLToPath(import.meta.url));

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

test("the section clock prefers the last successful fetch, in UTC+8", () => {
  const fetchedAt = Date.parse("2026-11-15T10:00:05Z");
  const view = { as_of_utc8: "2026-11-15 18:14:59", as_of: Date.parse("2026-11-15T10:07:00Z") };
  assert.equal(boardClock(view, fetchedAt), "18:00:05");
  assert.equal(boardClock(view, null), "18:14:59");
  assert.equal(basisStamp(view).close, "18:00:00");
  assert.equal(basisStamp(view).next, "18:15");
  assert.equal(
    basisStamp({ basis: { close_utc8: "18:00:00", next_close_utc8: "18:15" } }).next,
    "18:15",
  );
  assert.match(boardFreshLine(view, fetchedAt), /^更新于 18:00:05 · 依据 15m 收线 18:00:00 · 下次重算 18:15$/);
});

test("an exact 15m boundary is that bar's close, and the next one is 15 minutes later", () => {
  const asOf = Date.parse("2026-11-15T10:15:00Z");
  assert.deepEqual(basisStamp({ as_of: asOf }), { close: "18:15:00", next: "18:30" });
});

test("the stale age stays quiet until the last success is more than 30 seconds old", () => {
  const fetchedAt = 1_000;
  assert.equal(boardStaleSeconds(fetchedAt, fetchedAt + 30_000), null);
  assert.equal(boardStaleSeconds(fetchedAt, fetchedAt + 30_001), 30);
  assert.equal(boardStaleSeconds(fetchedAt, fetchedAt + 45_900), 45);
  assert.equal(boardStaleSeconds(null, fetchedAt + 90_000), null);
});

test("a failed pull keeps the last board and does not move the success clock", () => {
  const previous = { plans: [], as_of_utc8: "2026-11-15 18:00:05" };
  const state = { view: previous, fetchedAt: 5_000, since: 1_000 };
  const failed = applyBoardPoll(state, previous, { ok: false }, 12_000);
  assert.equal(failed.offline, true);
  assert.equal(failed.error, "盘面服务未启动");
  assert.equal(failed.view, previous);
  assert.equal(failed.fetchedAt, 5_000);
  assert.equal(failed.events.length, 0);
  const next = { plans: [], as_of_utc8: "2026-11-15 18:00:15" };
  const again = applyBoardPoll(failed, failed.previous, { ok: true, view: next }, 20_000);
  assert.equal(again.offline, false);
  assert.equal(again.fetchedAt, 20_000);
  assert.equal(again.view, next);
});

test("the board panel prints the update clock and the stale banner", () => {
  const panel = readFileSync(join(SRC, "components", "BoardPanel.vue"), "utf8");
  assert.match(panel, /数据已过期 \{\{ staleSeconds \}\} 秒/);
  assert.match(panel, /走势要点 <span class="board-fresh">\{\{ freshLine \}\}<\/span>/);
  assert.match(panel, /合约情绪 <span class="board-fresh">\{\{ freshLine \}\}<\/span>/);
  assert.match(panel, /情景计划 <span class="board-fresh">\{\{ freshLine \}\}<\/span>/);
  assert.match(panel, /实时 最新价/);
  assert.match(panel, /进场 \{\{ px\(plan\.entry\) \}\}/);
  assert.match(panel, /区间 \{\{ px\(plan\.entry_low\) \}\}–\{\{ px\(plan\.entry_high\) \}\}/);
  assert.match(panel, /planValidity\(plan\)/);
  assert.match(panel, /历史计划/);
  assert.match(panel, /导出 CSV/);
  assert.match(panel, /planOutcomeLine\(row\)/);
});

test("each plan carries the 15m close it was built on and the hour it expires", () => {
  assert.equal(planValidity({ validity: "21:45 生成 · 有效至 22:45" }), "21:45 生成 · 有效至 22:45");
  assert.equal(
    planValidity({ generated_utc8: "2026-11-15 21:45", expires_utc8: "2026-11-15 22:45" }),
    "21:45 生成 · 有效至 22:45",
  );
});

test("history lines name the outcome and the export url stays on the board", () => {
  assert.equal(planOutcomeLine({ outcome: "pending" }), "未结算");
  assert.equal(planOutcomeLine({ outcome: "expired", expired_unfilled: 1 }), "到期未成交");
  assert.equal(
    planOutcomeLine({
      outcome: "resolved",
      filled: 1,
      fill_utc8: "2026-11-15 21:51",
      exit_reason: "stop",
      first_hit: "stop",
      r_taker: -1.05,
      r_maker: -1.02,
    }),
    "已成交 21:51 · 先到止损 · 费后 R taker -1.05 / maker -1.02",
  );
  assert.equal(planHistoryExportURL({ start: "2026-11-13 18:00" }),
    "http://127.0.0.1:8770/api/board/plans.csv?start=2026-11-13+18%3A00");
});

test("the history loader asks the board for one page", async () => {
  const calls = [];
  const payload = await loadPlanHistory({ start: "2026-11-15 00:00" }, {
    fetchImpl: async (url) => {
      calls.push(url);
      return { ok: true, json: async () => ({ plans: [{ plan_id: "a" }], count: 1 }) };
    },
  });
  assert.equal(payload.count, 1);
  assert.match(calls[0], /^http:\/\/127\.0\.0\.1:8770\/api\/board\/plans\?/);
  assert.match(calls[0], /start=2026-11-15/);
  assert.match(calls[0], /limit=100/);
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
