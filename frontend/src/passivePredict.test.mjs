import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const SRC = dirname(fileURLToPath(import.meta.url));

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return files(path);
    return /\.(vue|js)$/.test(name) ? [path] : [];
  });
}

test("the desk shows signals and news, and no prediction control", () => {
  const panel = readFileSync(join(SRC, "components", "SidePanel.vue"), "utf8");
  const app = readFileSync(join(SRC, "App.vue"), "utf8");
  const news = readFileSync(join(SRC, "components", "NewsView.vue"), "utf8");
  const visible = panel + app + news;
  assert.match(panel, /信号通知/);
  assert.match(news, /setInterval\(load, NEWS_POLL_MS\)/);
  assert.match(news, /@click="refresh"/);
  assert.ok(!/runPredict|predictOnClick|模型原值|加消息面后/.test(visible));
  assert.ok(!/买卖盘|盘口价/.test(panel));
});

test("the controller never calls prediction from its news path", () => {
  const controller = readFileSync(join(SRC, "predictController.js"), "utf8");
  const loadNews = controller.match(/async function loadCachedNews\(\) \{([\s\S]*?)\n  \}/);
  assert.ok(loadNews);
  assert.match(loadNews[1], /api\.news/);
  assert.ok(!/predictOnClick/.test(loadNews[1]));
  assert.ok(!/setInterval|setTimeout/.test(controller));
});

test("no user-facing source file contains the removed reference wording", () => {
  const forbidden = ["仅参考", "供参考", "仅供参考", "非建议"];
  for (const path of files(SRC)) {
    if (path.endsWith(".test.mjs")) continue;
    const text = readFileSync(path, "utf8");
    for (const term of forbidden) {
      assert.ok(!text.includes(term), `${term} found in ${path}`);
    }
  }
});
