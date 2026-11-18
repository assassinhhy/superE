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
  historyFilters,
  loadPlanHistory,
  loadPlanStats,
  lossState,
  planGuideLevels,
  planHistoryExportURL,
  planHistoryQuery,
  planOutcomeLine,
  planValidity,
  savePlanNote,
  statsSummary,
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

test("a focused plan yields labeled entry, stop, T1 and T2 guides", () => {
  assert.deepEqual(
    planGuideLevels({ id: "trend", entry: 2500, stop: 2560, targets: [2440, 2380] }),
    [
      { price: 2500, label: "进场", tone: "entry" },
      { price: 2560, label: "止损", tone: "stop" },
      { price: 2440, label: "T1", tone: "target" },
      { price: 2380, label: "T2", tone: "target" },
    ],
  );
  // Journal rows carry t1/t2 directly; missing levels are skipped.
  assert.deepEqual(planGuideLevels({ id: "x", entry: null, stop: 2, t1: 3, t2: 4 }), [
    { price: 2, label: "止损", tone: "stop" },
    { price: 3, label: "T1", tone: "target" },
    { price: 4, label: "T2", tone: "target" },
  ]);
  assert.deepEqual(planGuideLevels({ id: "flip", entry: 1, stop: 2 }), []);
  assert.deepEqual(planGuideLevels(null), []);
});

test("scope and kind map onto the board query parameters", () => {
  assert.deepEqual(historyFilters(), {});
  assert.deepEqual(historyFilters("all", ""), {});
  assert.deepEqual(historyFilters("ready", "顺势"), { status: "可进场", kind: "顺势" });
  assert.deepEqual(historyFilters("followed", "逆势"), { followed: true, kind: "逆势" });
  const params = new URLSearchParams(
    planHistoryQuery({ status: "可进场", kind: "破位", followed: false, limit: 20 }).slice(1),
  );
  assert.equal(params.get("status"), "可进场");
  assert.equal(params.get("kind"), "破位");
  assert.equal(params.get("followed"), "false");
  assert.equal(params.get("limit"), "20");
  assert.match(planHistoryExportURL({ status: "可进场", followed: true }), /status=.*followed=true/);
});

test("the stats loader uses the same filters and formats a compact bar", async () => {
  const calls = [];
  const payload = await loadPlanStats({ kind: "逆势", followed: true }, {
    fetchImpl: async (url) => {
      calls.push(url);
      return {
        ok: true,
        json: async () => ({
          total: 5,
          filled: 3,
          resolved: 2,
          fill_rate: 0.75,
          win_rate: 0.5,
          total_r_taker: 1,
          total_r_maker: 1.15,
          avg_r_taker: 0.5,
          avg_r_maker: 0.575,
        }),
      };
    },
  });
  const url = new URL(calls[0]);
  assert.equal(url.pathname, "/api/board/stats");
  assert.equal(url.searchParams.get("kind"), "逆势");
  assert.equal(url.searchParams.get("followed"), "true");
  const line = statsSummary(payload);
  assert.equal(line.fillRate, "75.0%");
  assert.equal(line.winRate, "50.0%");
  assert.equal(line.totalTaker, "+1.00");
  assert.equal(line.totalMaker, "+1.15");
  assert.equal(line.avgTaker, "+0.50");
  assert.equal(line.avgMaker, "+0.57");
  assert.equal(line.filled, 3);
  assert.equal(line.resolved, 2);
  assert.equal(line.total, 5);
  assert.equal(statsSummary(null), null);
});

test("saving a review note POSTs JSON and returns the updated row", async () => {
  const calls = [];
  const saved = await savePlanNote("ETHUSDT:1:trend", { followed: true, note: "按计划执行" }, {
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      return {
        ok: true,
        json: async () => ({
          plan: { plan_id: "ETHUSDT:1:trend", followed: 1, user_note: "按计划执行" },
        }),
      };
    },
  });
  assert.equal(saved.followed, 1);
  assert.equal(saved.user_note, "按计划执行");
  assert.equal(calls[0].url, "http://127.0.0.1:8770/api/board/plans/ETHUSDT%3A1%3Atrend/note");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0].init.body), { followed: true, note: "按计划执行" });

  const cleared = [];
  await savePlanNote("p", { note: "" }, {
    fetchImpl: async (url, init) => {
      cleared.push(JSON.parse(init.body));
      return { ok: true, json: async () => ({ plan: { user_note: null } }) };
    },
  });
  assert.deepEqual(cleared[0], { note: "" });
  await assert.rejects(savePlanNote("   ", {}, { fetchImpl: async () => assert.fail("no fetch") }), /plan id required/);
});

test("the history panel renders the stats bar, filters, follow flags and notes", () => {
  const panel = readFileSync(join(SRC, "components", "BoardPanel.vue"), "utf8");
  assert.match(panel, /loadPlanStats/);
  assert.match(panel, /savePlanNote/);
  assert.match(panel, /historyFilters\(scope\.value, kind\.value\)/);
  assert.match(panel, /成交率 \{\{ statsLine\.fillRate \}\}/);
  assert.match(panel, /胜率 \{\{ statsLine\.winRate \}\}/);
  assert.match(panel, /累计 R \{\{ statsLine\.totalTaker \}\}/);
  assert.match(panel, /平均 R \{\{ statsLine\.avgTaker \}\}/);
  assert.match(panel, /PLAN_SCOPES/);
  assert.match(panel, /已跟单/);
  assert.match(panel, /保存备注/);
  assert.match(panel, /PLAN_KINDS/);
  assert.match(panel, /v-model="kind"/);
  assert.match(panel, /@change="setFollowed\(row, \$event\.target\.checked\)"/);
  assert.match(panel, /@click="saveRowNote\(row\)"/);
  assert.match(panel, /取消图看/);
});

test("the chart toggles labeled guide lines with the focused plan", () => {
  const chart = readFileSync(join(SRC, "components", "ChartPane.vue"), "utf8");
  assert.match(chart, /import \{ planGuideLevels \} from "\.\.\/board\.js"/);
  assert.match(chart, /const focused = props\.focusPlan/);
  assert.match(chart, /for \(const level of planGuideLevels\(focused\)\)/);
  assert.match(chart, /lead: true, guide: true/);
  assert.match(chart, /watch\(\(\) => \[props\.board, props\.focusPlan, boardOn\.value\], \(\) => paintBoard\(\)\)/);
  const overlays = readFileSync(join(SRC, "boardOverlays.js"), "utf8");
  assert.match(overlays, /data\?\.tone === "stop"/);
  assert.match(overlays, /data\.lead \? `\$\{name\} \$\{priceText\}/);
});
