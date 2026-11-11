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

test("SidePanel polls cached news and predicts only from the click handler", () => {
  const panel = readFileSync(join(SRC, "components", "SidePanel.vue"), "utf8");
  assert.ok(!/setTimeout/.test(panel));
  assert.ok(!/WebSocket/.test(panel));
  assert.match(panel, /setInterval\(\(\) => controller\.loadCachedNews\(\), NEWS_POLL_MS\)/);
  assert.ok(!/setInterval\([\s\S]{0,120}predict/.test(panel));
  const mounted = panel.match(/onMounted\(\(\) => \{([\s\S]*?)\}\);/);
  assert.ok(mounted, "onMounted block present");
  assert.match(mounted[1], /loadCachedNews/);
  assert.ok(!/runPredict|predictOnClick|api\.predict/.test(mounted[1]));
  assert.match(panel, /@click="runPredict"/);
  assert.match(panel, /@click="refreshNews"/);
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
