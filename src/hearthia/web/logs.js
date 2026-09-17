/* logs.js — local search, pause and export over the live gateway stream. */
import { $ } from "./api.js";
import { LogBuffer } from "./log-buffer.mjs";

const buffer = new LogBuffer();
const view = $("#log-view");
const search = $("#logs-search");
const pause = $("#logs-pause");
const status = $("#logs-status");
let logsStarted = false;
let connection = "Connecting…";
let interrupted = false;

function render(force = false) {
  if (buffer.snapshot !== null && !force) return;
  const atBottom = view.scrollHeight - view.scrollTop - view.clientHeight < 40;
  view.textContent = buffer.visible(search.value);
  if (atBottom) view.scrollTop = view.scrollHeight;
}

function updateStatus() {
  const paused = buffer.snapshot !== null ? "View paused · " : "";
  const warning = interrupted ? " · Stream interrupted; some logs may be missing" : "";
  status.textContent = paused + connection + warning;
}

export async function startLogs() {
  if (logsStarted) return;
  logsStarted = true;
  try {
    for (;;) {
      let reader;
      try {
        connection = "Connecting…";
        updateStatus();
        const response = await fetch("/api/logs/stream");
        if (!response.ok || !response.body) throw new Error("Log stream unavailable");
        reader = response.body.getReader();
        const decoder = new TextDecoder();
        connection = "Live · latest 200,000 characters retained locally";
        updateStatus();
        for (;;) {
          const { done, value } = await reader.read();
          buffer.append(done ? decoder.decode() : decoder.decode(value, { stream: true }));
          render();
          if (done) break;
        }
      } catch {
        // A disconnected gateway is retried while this tab remains active.
      } finally {
        reader?.releaseLock();
      }
      interrupted = true;
      connection = "Disconnected";
      updateStatus();
      if (!$("#tab-logs").classList.contains("active")) break;
      connection = "Disconnected · reconnecting in 3 s";
      updateStatus();
      await new Promise(resolve => setTimeout(resolve, 3000));
      if (!$("#tab-logs").classList.contains("active")) break;
    }
  } finally {
    logsStarted = false;
  }
}

search.addEventListener("input", () => render(true));
pause.addEventListener("click", () => {
  const paused = buffer.snapshot === null;
  buffer.setPaused(paused);
  pause.textContent = paused ? "Resume" : "Pause view";
  pause.setAttribute("aria-pressed", String(paused));
  updateStatus();
  render(true);
});
$("#logs-clear").addEventListener("click", () => {
  buffer.clear();
  render(true);
});
$("#logs-export").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([buffer.visible(search.value)], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `hearthia-logs-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
