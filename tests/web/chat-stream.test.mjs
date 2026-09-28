import test from "node:test";
import assert from "node:assert/strict";
import { ChatStream } from "../../src/hearthia/web/chat-stream.mjs";

test("preserves byte-split Unicode, structured events and completion", () => {
  const parser = new ChatStream();
  const bytes = new TextEncoder().encode('data: {"choices":[{"delta":{"content":"español 🔥"}}]}\r\n\r\ndata: {"tool_event":{"status":"complete"}}\n\ndata: [DONE]\n\n');
  const events = [];
  for (const b of bytes) events.push(...parser.feed(new Uint8Array([b])));
  assert.equal(events[0].choices[0].delta.content, "español 🔥");
  assert.equal(events[1].tool_event.status, "complete");
  assert.deepEqual(events[2], { done: true });
});

test("malformed events do not lose subsequent valid events", () => {
  const parser = new ChatStream();
  assert.deepEqual(parser.feed(new TextEncoder().encode('data: invalid\n\ndata: {"context":{}}\n\n')), [{ context: {} }]);
});

test("caps unterminated event buffers", () => {
  assert.throws(() => new ChatStream().feed(new Uint8Array(1_048_577)), /stream limit/);
});
