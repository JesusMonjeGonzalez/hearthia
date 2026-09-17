import { test } from "node:test";
import assert from "node:assert/strict";
import { LogBuffer } from "../../src/hearthia/web/log-buffer.mjs";

test("filter is literal, case insensitive and works across network chunks", () => {
  const logs = new LogBuffer();
  logs.append("ready\n[ERR");
  logs.append("OR] model failed\nerror recovered\n");
  assert.equal(logs.visible("  ERROR  "), "[ERROR] model failed\nerror recovered");
  assert.equal(logs.visible("["), "[ERROR] model failed");
  assert.equal(logs.visible("missing"), "");
});

test("pause preserves a searchable snapshot while new logs remain bounded", () => {
  const logs = new LogBuffer(10);
  logs.append("old\n");
  logs.setPaused(true);
  logs.append("new data that exceeds the limit");
  assert.equal(logs.visible(), "old\n");
  assert.equal(logs.visible("new"), "");
  assert.equal(logs.text.length, 10);
  logs.setPaused(false);
  assert.equal(logs.visible(), " the limit");
});

test("clear removes both history and the frozen snapshot", () => {
  const logs = new LogBuffer();
  logs.append("secret\n");
  logs.setPaused(true);
  logs.clear();
  logs.append("after clear");
  assert.equal(logs.visible(), "");
  logs.setPaused(false);
  assert.equal(logs.visible(), "after clear");
});
