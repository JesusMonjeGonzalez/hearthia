import test from "node:test";
import assert from "node:assert/strict";
import { planProgress, shouldAutoContinue } from "../../src/hearthia/web/plan-progress.mjs";

test("progress counts only valid, in-range done indices", () => {
  const plan = { steps: ["a", "b", "c"], done: [1, 3, 9, 0, "x", 2.5] };
  const progress = planProgress(plan);
  assert.equal(progress.total, 3);
  assert.equal(progress.done, 2); // 1 and 3; 9/0/"x"/2.5 rejected
  assert.equal(progress.pending, 1);
  assert.equal(progress.complete, false);
});

test("no plan means no progress and no auto-continue", () => {
  assert.equal(planProgress(undefined).total, 0);
  assert.equal(planProgress({ steps: [] }).complete, false);
  assert.equal(shouldAutoContinue({ plan: undefined, strike: 0, limit: 8, enabled: true }), false);
});

test("a complete plan never auto-continues", () => {
  const plan = { steps: ["a", "b"], done: [1, 2] };
  assert.equal(planProgress(plan).complete, true);
  assert.equal(shouldAutoContinue({ plan, strike: 0, limit: 8, enabled: true }), false);
});

test("auto-continue needs the toggle, a pending step and budget left", () => {
  const plan = { steps: ["a", "b"], done: [1] };
  assert.equal(shouldAutoContinue({ plan, strike: 0, limit: 8, enabled: true }), true);
  assert.equal(shouldAutoContinue({ plan, strike: 0, limit: 8, enabled: false }), false);
  assert.equal(shouldAutoContinue({ plan, strike: 8, limit: 8, enabled: true }), false);
  assert.equal(shouldAutoContinue({ plan, strike: 7, limit: 8, enabled: true }), true);
});
