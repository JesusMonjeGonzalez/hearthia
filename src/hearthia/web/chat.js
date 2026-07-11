/* chat.js — conversation management, streaming chat. */

import { $, api, esc, renderMD, highlightIn } from "./api.js";
import { refreshStatus } from "./models.js";

const LS_KEY = "hearthia.convs";
const LS_ACTIVE = "hearthia.active";
let convs = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
let activeId = localStorage.getItem(LS_ACTIVE);
let streaming = false;
let aborter = null;

function saveConvs() {
  localStorage.setItem(LS_KEY, JSON.stringify(convs));
  if (activeId) localStorage.setItem(LS_ACTIVE, activeId);
}

function activeConv() {
  return convs.find((c) => c.id === activeId);
}

function newConv() {
  const c = { id: String(Date.now()), title: "New chat", system: "", model: "", messages: [] };
  convs.unshift(c);
  activeId = c.id;
  saveConvs();
  renderConvList();
  renderConv();
}

function renderConvList() {
  const wrap = $("#conv-items");
  wrap.innerHTML = "";
  for (const c of convs) {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === activeId ? " active" : "");
    item.innerHTML = `<span class="title">${esc(c.title)}</span><button class="del" title="Delete conversation">×</button>`;
    item.addEventListener("click", (e) => {
      if (e.target.classList.contains("del")) return;
      activeId = c.id;
      saveConvs();
      renderConvList();
      renderConv();
    });
    item.querySelector(".del").addEventListener("click", () => {
      convs = convs.filter((x) => x.id !== c.id);
      if (activeId === c.id) activeId = convs[0]?.id || null;
      saveConvs();
      renderConvList();
      renderConv();
    });
    wrap.appendChild(item);
  }
}

function renderConv() {
  const log = $("#chat-log");
  log.innerHTML = "";
  const c = activeConv();
  $("#chat-stats").textContent = "";
  if (!c || !c.messages.length) {
    log.innerHTML = `<div class="empty">No messages yet. The first message loads the model into RAM automatically.</div>`;
  } else {
    for (const m of c.messages) addMsg(m.role, m.content, m.reasoning);
  }
  if (c) {
    $("#chat-system").value = c.system || "";
    if (c.model) $("#chat-model").value = c.model;
  }
}

function addMsg(role, content, reasoning) {
  const log = $("#chat-log");
  const d = document.createElement("div");
  d.className = "msg " + role;
  if (role === "assistant") {
    let html = "";
    if (reasoning) {
      html += `<details class="reasoning"><summary>reasoning</summary><div class="rbody">${esc(reasoning)}</div></details>`;
    }
    d.innerHTML = html + renderMD(content);
    highlightIn(d);
  } else {
    d.textContent = content;
  }
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

$("#conv-new").addEventListener("click", newConv);

$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#chat-stop").addEventListener("click", () => aborter?.abort());

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (streaming) return;
  const text = $("#chat-input").value.trim();
  if (!text) return;
  if (!activeConv()) newConv();
  const c = activeConv();
  c.system = $("#chat-system").value.trim();
  c.model = $("#chat-model").value;
  if (c.messages.length === 0) c.title = text.slice(0, 42);

  $("#chat-input").value = "";
  $("#chat-log").querySelector(".empty")?.remove();
  c.messages.push({ role: "user", content: text });
  addMsg("user", text);
  saveConvs();
  renderConvList();

  const el = addMsg("assistant", "");
  el.classList.add("thinking");
  el.textContent = "…waiting for model (first token loads it into RAM)";

  const messages = [];
  if (c.system) messages.push({ role: "system", content: c.system });
  for (const m of c.messages) messages.push({ role: m.role, content: m.content });

  const body = { model: c.model, messages, stream: true };
  const t = parseFloat($("#s-temp").value);
  const p = parseFloat($("#s-topp").value);
  const mt = parseInt($("#s-maxtok").value, 10);
  if (!Number.isNaN(t)) body.temperature = t;
  if (!Number.isNaN(p)) body.top_p = p;
  if (!Number.isNaN(mt)) body.max_tokens = mt;

  streaming = true;
  aborter = new AbortController();
  $("#chat-send").hidden = true;
  $("#chat-stop").hidden = false;
  let out = "",
    reasoning = "",
    tokens = 0,
    tStart = performance.now(),
    tFirst = null;
  const log = $("#chat-log");

  const redraw = () => {
    let html = "";
    if (reasoning) {
      html += `<details class="reasoning" open><summary>reasoning</summary><div class="rbody">${esc(reasoning)}</div></details>`;
    }
    el.innerHTML = html + renderMD(out);
    highlightIn(el);
    log.scrollTop = log.scrollHeight;
  };

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: aborter.signal,
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "",
      lastDraw = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {
          const j = JSON.parse(payload);
          if (j.error) {
            out += "\n`[error]` " + JSON.stringify(j.error);
            continue;
          }
          const delta = j.choices?.[0]?.delta || {};
          if (delta.reasoning_content) reasoning += delta.reasoning_content;
          if (delta.content) {
            out += delta.content;
            tokens++;
          }
          if (delta.content || delta.reasoning_content) {
            if (tFirst === null) {
              tFirst = performance.now();
              el.classList.remove("thinking");
            }
            const now = performance.now();
            if (now - lastDraw > 80) {
              redraw();
              lastDraw = now;
            }
          }
        } catch {}
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") out += "\n`[request failed]` " + err.message;
  } finally {
    redraw();
    const rdetails = el.querySelector("details.reasoning");
    if (rdetails) rdetails.open = false;
    if (tFirst !== null && tokens > 0) {
      const gen = (performance.now() - tFirst) / 1000;
      $("#chat-stats").textContent =
        `${(tokens / Math.max(gen, 0.001)).toFixed(1)} tok/s · first token ${((tFirst - tStart) / 1000).toFixed(1)}s`;
    }
    c.messages.push({ role: "assistant", content: out, reasoning });
    saveConvs();
    streaming = false;
    aborter = null;
    $("#chat-send").hidden = false;
    $("#chat-stop").hidden = true;
    refreshStatus();
  }
});

renderConvList();
if (!activeId && convs.length) activeId = convs[0].id;
renderConv();
