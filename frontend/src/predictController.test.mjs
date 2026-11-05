import test from "node:test";
import assert from "node:assert/strict";
import { createPredictController } from "./predictController.js";

function fakeApi({ failPredict = false } = {}) {
  const calls = { predict: 0, news: 0 };
  return {
    calls,
    news: async () => {
      calls.news += 1;
      return { items: [{ id: 1, title: "cached ETH news", direction: 0.4 }], summary: { line: "缓存" }, alert: null };
    },
    predictOnClick: async () => {
      calls.predict += 1;
      if (failPredict) throw new Error("network down");
      return {
        items: [
          {
            interval: "1h",
            status: "ok",
            price_forecast: { median: 100, p25: 99, p75: 101, p10: 98, p90: 102 },
            news_overlay: { applied: true, adjusted: { median: 100.1, p25: 99, p75: 101, p10: 98, p90: 102 } },
          },
        ],
        news: { items: [{ id: 2, title: "fresh ETH news" }], summary: { line: "fresh" }, alert: { title: "有消息" } },
        click_at_utc8: "2026-11-05 20:30",
      };
    },
  };
}

test("loading cached news never calls the prediction endpoint", async () => {
  const api = fakeApi();
  const controller = createPredictController({ api, intervals: ["1h"] });
  await controller.loadCachedNews();
  assert.equal(api.calls.news, 1);
  assert.equal(api.calls.predict, 0);
  assert.equal(controller.state.predictions["1h"], undefined);
  assert.equal(controller.state.hasRun, false);
});

test("only run() predicts, and it updates rows and news", async () => {
  const api = fakeApi();
  const controller = createPredictController({ api, intervals: ["1h"] });
  const payload = await controller.run();
  assert.equal(api.calls.predict, 1);
  assert.ok(payload);
  assert.equal(controller.state.predictions["1h"].news_overlay.applied, true);
  assert.equal(controller.state.news.items[0].id, 2);
  assert.equal(controller.state.hasRun, true);
  assert.equal(controller.state.clickAt, "2026-11-05 20:30");
});

test("concurrent clicks are guarded against re-entry", async () => {
  const api = fakeApi();
  const controller = createPredictController({ api, intervals: ["1h"] });
  const first = controller.run();
  const second = controller.run();
  assert.equal(await second, null);
  await first;
  assert.equal(api.calls.predict, 1);
  assert.equal(controller.state.predicting, false);
});

test("a failed news fetch during a click keeps the cached news panel", async () => {
  const api = fakeApi();
  const controller = createPredictController({ api, intervals: ["1h"] });
  await controller.loadCachedNews();
  const cached = controller.state.news;
  api.predictOnClick = async () => ({
    items: [],
    news: { items: [], status: "error", message: "新闻超时", alert: null, sources: {} },
    click_at_utc8: "2026-11-05 20:31",
  });
  await controller.run();
  assert.equal(controller.state.news, cached);
  assert.equal(controller.state.news.items[0].id, 1);
  assert.match(controller.state.newsError, /新闻超时/);
});

test("a failed click keeps the previous rows and shows an error", async () => {
  const api = fakeApi();
  const controller = createPredictController({ api, intervals: ["1h"] });
  await controller.run();
  const previous = controller.state.predictions;
  api.predictOnClick = async () => {
    api.calls.predict += 1;
    throw new Error("network down");
  };
  await controller.run();
  assert.equal(controller.state.predictions, previous);
  assert.match(controller.state.error, /network down/);
  assert.equal(controller.state.predicting, false);
});
