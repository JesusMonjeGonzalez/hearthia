import test from "node:test";
import assert from "node:assert/strict";
import { describeTool } from "../../src/hearthia/web/tool-result.mjs";

test("describes command failures, limits and output honestly", () => {
  const result = describeTool({ content: JSON.stringify({ kind: "command", ok: false,
    argv: ["python", "tests.py"], exit_code: -9, limit: "timeout", duration_seconds: 1,
    output: "test log", cwd: "/project", output_truncated: true }) });
  assert.equal(result.failed, true);
  assert.match(result.title, /Failed.*exit -9/);
  assert.match(result.detail, /timeout.*truncated/);
});

test("edit diff is plain text, including HTML-like source", () => {
  const diff = '-old\n+<img src=x onerror=alert(1)>\n';
  const result = describeTool({ content: JSON.stringify({ kind: "edit", ok: true,
    path: "index.html", diff, changed: true }) });
  assert.equal(result.body, diff);
  assert.equal(result.kind, "edit");
});

test("plain-text errors and malformed structured results have a safe fallback", () => {
  assert.equal(describeTool({ content: "Error: old_text does not match" }).failed, true);
  assert.equal(describeTool({ content: '{"kind":"edit","diff":null}' }).kind, "text");
});

test("a broken edit is flagged as failed with the syntax error in the title", () => {
  const result = describeTool({
    content: JSON.stringify({
      kind: "edit", ok: true, path: "code.py", changed: true,
      diff: "+def f(:\n", syntax_error: "line 1: invalid syntax",
    }),
  });
  assert.equal(result.failed, true);
  assert.match(result.title, /SYNTAX ERROR \(line 1/);
  assert.match(result.detail, /fix it before running anything/);
});

test("a clean edit is not failed", () => {
  const result = describeTool({
    content: JSON.stringify({ kind: "edit", ok: true, path: "code.py", changed: true, diff: "+x = 1\n" }),
  });
  assert.equal(result.failed, false);
});

test("plan results render as a checklist with progress", () => {
  const result = describeTool({ content: JSON.stringify({ kind: "plan", ok: true,
    steps: ["leer", "editar", "probar"], done: [1, 3], pending: 1, warning: "unverified edits" }) });
  assert.equal(result.kind, "plan");
  assert.match(result.title, /2\/3 done.*1 pending.*unverified edits/);
  assert.equal(result.body, "✔ leer\n○ editar\n✔ probar");
});

test("job status renders state, exit code and tail", () => {
  const result = describeTool({ content: JSON.stringify({ kind: "job", ok: true, id: "abc123",
    state: "failed", exit_code: 1, duration_seconds: 12.4, argv: "pytest -q", tail: "2 failed" }) });
  assert.equal(result.failed, true);
  assert.match(result.title, /Job abc123 · failed · exit 1 · 12.4s/);
  assert.match(result.body, /pytest -q[\s\S]*2 failed/);
});

test("job list and refusals are distinct", () => {
  const list = describeTool({ content: JSON.stringify({ kind: "job", ok: true, running: 1,
    jobs: [{ id: "a", state: "running", duration_seconds: 1.2, argv: "sleep 60" }] }) });
  assert.equal(list.failed, false);
  assert.match(list.title, /Jobs · 1 running/);
  assert.match(list.body, /running a {2}1.2s {2}sleep 60/);
  const refused = describeTool({ content: JSON.stringify({ kind: "job", ok: false,
    error: "3 jobs already running" }) });
  assert.equal(refused.failed, true);
  assert.match(refused.body, /already running/);
});

test("subagent reports keep their header as the card title", () => {
  const result = describeTool({ content:
    "Subagent report (complete · 2 rounds · model tiny · tools: read_file):\nla causa está en a.py:12" });
  assert.equal(result.kind, "task");
  assert.match(result.title, /Subagent · complete · 2 rounds · model tiny/);
  assert.equal(result.body, "la causa está en a.py:12");
  assert.match(describeTool({ content: "Error: boom" }).title, /Tool result/);
});

test("mcp results name the server and tool", () => {
  const result = describeTool({ content: JSON.stringify({ kind: "mcp", ok: false,
    server: "fake", tool: "boom", output: "falló a propósito" }) });
  assert.equal(result.kind, "mcp");
  assert.equal(result.failed, true);
  assert.match(result.title, /MCP fake\/boom · error/);
});

test("a status-only plan ack still renders as a progress card", () => {
  const result = describeTool({ content: JSON.stringify({ kind: "plan", ok: true,
    done: [1, 2, 3, 4], pending: 0 }) });
  assert.equal(result.kind, "plan");
  assert.match(result.title, /Plan · 4\/4 done/);
  assert.equal(result.body, "(steps unchanged)");
});

test("usage summary reports turns, totals, projection and last-round cache", async () => {
  const { usageSummary } = await import("../../src/hearthia/web/usage-summary.mjs");
  const lines = usageSummary(
    { turns: 3, last_input_tokens: 2181, allowance_tokens: 59329, peak_input_tokens: 5116,
      total_prompt_tokens: 4383, total_output_tokens: 944 },
    { remaining_tokens: 57148, growth_tokens_per_turn: 640, turns_until_trimming: 89 },
    { prompt_n: 118, cache_n: 4998, predicted_per_second: 14.9 },
  );
  assert.match(lines[0], /Turns: 3/);
  assert.match(lines[1], /Last input: 2,181 of 59,329 tok/);
  assert.match(lines.join("\n"), /growing ~640 tok\/turn/);
  assert.match(lines.join("\n"), /omitted: 89/);
  assert.match(lines.at(-1), /98% cached.*14\.9 tok\/s/);
  assert.equal(usageSummary(null, null, null), null);
  assert.equal(usageSummary({ turns: 0 }, null, null), null);
});

test("auto-continue asks for verification when the last turn left edits unchecked", async () => {
  const { autoContinuePrompt } = await import("../../src/hearthia/web/plan-progress.mjs");
  const plan = { steps: ["a", "b", "c"], done: [1] };
  const plain = autoContinuePrompt(plan, { edited: [], verified: true });
  assert.match(plain, /paso 2/);
  const unverified = autoContinuePrompt(plan, { edited: ["a.py"], verified: false });
  assert.match(unverified, /editó 1 archivo\(s\) sin verificar/);
  assert.match(unverified, /paso 2/);
  const verified = autoContinuePrompt(plan, { edited: ["a.py"], verified: true });
  assert.equal(verified, plain);
  assert.match(autoContinuePrompt(null, null), /paso 1/);
});

test("api error bodies become readable messages", async () => {
  const { errorMessage } = await import("../../src/hearthia/web/http-error.mjs");
  assert.equal(errorMessage('{"detail":"does not fit the budget"}'), "does not fit the budget");
  assert.equal(errorMessage('{"error":"gateway down"}'), "gateway down");
  assert.equal(errorMessage('{"detail":{"nested":1}}'), '{"nested":1}');
  assert.equal(errorMessage("plain text failure"), "plain text failure");
  assert.equal(errorMessage(""), "request failed");
  assert.equal(errorMessage(null, "load failed"), "load failed");
});
