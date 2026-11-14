import test from "node:test";
import assert from "node:assert/strict";
import { BOARD_GROUP, USER_GROUP, drawingKey, loadBook, persistableOverlays, saveBook, serializeOverlay, withoutGroup } from "./drawings.js";

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

test("board overlays are not stored and clearing user lines leaves them", () => {
  const overlays = [
    {
      id: "user-1",
      name: "horizontalStraightLine",
      groupId: USER_GROUP,
      points: [{ timestamp: 10, value: 1900 }],
    },
    {
      id: "board-1",
      name: "boardLevel",
      groupId: BOARD_GROUP,
      points: [{ timestamp: 10, value: 1893.39 }],
    },
  ];
  const saved = saveBook(null, drawingKey("um", "15m"), overlays);
  assert.equal(loadBook(saved, drawingKey("um", "15m")).length, 1);
  assert.equal(loadBook(saved, drawingKey("um", "15m"))[0].name, "horizontalStraightLine");
  assert.equal(persistableOverlays(overlays).length, 1);
  const left = withoutGroup(overlays, USER_GROUP);
  assert.equal(left.length, 1);
  assert.equal(left[0].groupId, BOARD_GROUP);
  assert.equal(withoutGroup(overlays, BOARD_GROUP)[0].groupId, USER_GROUP);
});
