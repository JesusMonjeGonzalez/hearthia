/* logs.js — live log streaming from llama-swap. */

import { $ } from "./api.js";

let logsStarted = false;

export async function startLogs() {
  if (logsStarted) return;
  logsStarted = true;
  const view = $("#log-view");
  for (;;) {
    try {
      const r = await fetch("/api/logs/stream");
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const atBottom = view.scrollHeight - view.scrollTop - view.clientHeight < 40;
        view.textContent += dec.decode(value, { stream: true });
        if (view.textContent.length > 300000)
          view.textContent = view.textContent.slice(-200000);
        if (atBottom) view.scrollTop = view.scrollHeight;
      }
    } catch {}
    // reconnect only while the user is looking at the Logs tab
    if (!document.querySelector("#tab-logs").classList.contains("active")) break;
    view.textContent += "\n[log stream disconnected — reconnecting in 3 s]\n";
    await new Promise((res) => setTimeout(res, 3000));
  }
  logsStarted = false;
}

$("#logs-clear").addEventListener("click", () => ($("#log-view").textContent = ""));
