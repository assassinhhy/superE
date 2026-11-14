import test from "node:test";
import assert from "node:assert/strict";
import { drawingKey, loadBook, saveBook, serializeOverlay } from "./drawings.js";

test("a finished segment is kept and an unfinished one is dropped", () => {
  const kept = serializeOverlay({
    id: "a",
    name: "segment",
    points: [{ timestamp: 10, value: 1 }, { timestamp: 20, value: 2 }, { timestamp: 30, value: 3 }],
  });
  assert.equal(kept.points.length, 2);
  assert.equal(serializeOverlay({ id: "b", name: "segment", points: [{ timestamp: 10, value: 1 }] }), null);
  assert.equal(serializeOverlay({ name: "brush", points: [{ timestamp: 1, value: 1 }] }), null);
});

test("books are stored separately for each interval", () => {
  const key = drawingKey("um", "15m");
  const other = drawingKey("um", "1h");
  const saved = saveBook(null, key, [{
    id: "line-1",
    name: "horizontalStraightLine",
    points: [{ timestamp: 100, value: 3500 }],
  }]);
  const both = saveBook(saved, other, [{
    id: "fib-1",
    name: "fibonacciLine",
    points: [{ timestamp: 100, value: 3400 }, { timestamp: 200, value: 3600 }],
  }]);
  assert.equal(loadBook(both, key)[0].name, "horizontalStraightLine");
  assert.equal(loadBook(both, other)[0].points[1].value, 3600);
  assert.deepEqual(loadBook(both, drawingKey("spot", "15m")), []);
});
