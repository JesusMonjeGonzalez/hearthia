import { test } from "node:test";
import assert from "node:assert/strict";

for (const failure of ["EOF", "read error", "HTTP error", "missing body", "fetch error"]) {
  test(`paused view exposes ${failure} and retains a gap warning after reconnect`, async t => {
    const elements = new Map();
    let active = true;
    const document = {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, {
          textContent: "", value: "", scrollHeight: 0, scrollTop: 0, clientHeight: 0,
          classList: { contains: () => active },
          listeners: {},
          addEventListener(event, listener) { this.listeners[event] = listener; },
          setAttribute() {},
        });
        return elements.get(selector);
      },
    };
    for (const [name, value] of Object.entries({ document, marked: { setOptions() {} } })) {
      const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
      Object.defineProperty(globalThis, name, { configurable: true, value });
      t.after(() => {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      });
    }
    const { startLogs } = await import(`../../src/hearthia/web/logs.js?failure=${failure}`);
    const status = document.querySelector("#logs-status");
    const pause = document.querySelector("#logs-pause");
    const view = document.querySelector("#log-view");
    pause.listeners.click();
    let attempts = 0;
    let releases = 0;
    let retries = 0;
    const encoder = new TextEncoder();
    t.mock.method(globalThis, "fetch", async url => {
      assert.equal(url, "/api/logs/stream");
      assert.match(status.textContent, /View paused.*Connecting/);
      attempts++;
      if (attempts === 1) {
        assert.doesNotMatch(status.textContent, /may be missing/);
        if (failure === "fetch error") throw new Error("synthetic offline");
        if (failure === "HTTP error") return { ok: false };
        if (failure === "missing body") return { ok: true, body: null };
      }
      let reads = 0;
      return { ok: true, body: { getReader: () => ({
        async read() {
          assert.match(status.textContent, /View paused.*Live/);
          if (attempts === 2) {
            assert.match(status.textContent, /Stream interrupted; some logs may be missing/);
            active = false;
          }
          if (attempts === 1 && failure === "read error") throw new Error("synthetic reset");
          return reads++ === 0
            ? { done: false, value: encoder.encode("synthetic log\n") }
            : { done: true };
        },
        releaseLock() { releases++; },
      }) } };
    });
    t.mock.method(globalThis, "setTimeout", (resolve, delay) => {
      assert.equal(delay, 3000);
      assert.match(status.textContent, /View paused.*Disconnected.*reconnecting in 3 s/);
      assert.match(status.textContent, /some logs may be missing/);
      assert.doesNotMatch(status.textContent, /continues buffering/);
      retries++;
      resolve();
    });

    await startLogs();
    assert.equal(attempts, 2);
    assert.equal(retries, 1);
    assert.equal(releases, ["EOF", "read error"].includes(failure) ? 2 : 1);
    assert.equal(view.textContent, "");
    assert.match(status.textContent, /View paused.*Disconnected.*some logs may be missing/);
    pause.listeners.click();
    assert.match(view.textContent, /synthetic log/);
    assert.doesNotMatch(status.textContent, /View paused/);
    assert.match(status.textContent, /Disconnected.*some logs may be missing/);
  });
}
