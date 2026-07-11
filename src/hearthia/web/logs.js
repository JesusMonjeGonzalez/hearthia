/* logs.js — live log streaming from llama-swap. */

import { $ } from "./api.js";

let logsStarted = false;

export async function startLogs() {
  if (logsStarted) return;
  logsStarted = true;
  const view = $("#log-view");
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
  } catch {
    view.textContent += "\n[log stream disconnected]";
  }
  logsStarted = false;
}

$("#logs-clear").addEventListener("click", () => ($("#log-view").textContent = ""));
