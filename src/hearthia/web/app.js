/* LLM Stack Control — vanilla JS + vendored marked/DOMPurify/highlight.js (all offline). */
"use strict";

const $ = (s) => document.querySelector(s);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
const GB = 1024 ** 3;
const fmtGB = (b) => (b / GB).toFixed(1) + " GB";
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

marked.setOptions({ gfm: true, breaks: false });
const renderMD = (text) => DOMPurify.sanitize(marked.parse(text || ""));

function highlightIn(el) {
  el.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement?.classList.contains("codewrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "codewrap";
    pre.replaceWith(wrap);
    wrap.appendChild(pre);
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(pre.textContent);
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    });
    wrap.appendChild(btn);
    pre.querySelectorAll("code").forEach((c) => hljs.highlightElement(c));
  });
}

/* ── tabs ── */
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
    if (t.dataset.tab === "logs") startLogs();
    if (t.dataset.tab === "config") loadConfig();
    if (t.dataset.tab === "library") { refreshFiles(); refreshDownloads(); }
    if (t.dataset.tab === "brain") brainStatus();
  })
);

/* ── status + memory map ── */
let lastStatus = null;

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    lastStatus = s;
    $("#swap-dot").classList.toggle("up", s.swap_up);
    const sys = s.system;
    const models = s.running.filter((m) => m.rss);
    const modelBytes = models.reduce((a, m) => a + m.rss, 0);
    const otherUsed = Math.max(0, sys.ram_used - modelBytes);

    $("#v-models").textContent = s.running.length ? s.running.map((m) => m.model).join(", ") : "none";
    $("#v-ram").textContent = `${fmtGB(sys.ram_used)} / ${fmtGB(sys.ram_total)}`;
    const tps = s.running.map((m) => m.tok_s).filter(Boolean);
    $("#v-toks").textContent = tps.length ? tps.map((t) => t.toFixed(0)).join("/") + " tok/s" : "–";
    $("#v-cpu").textContent = sys.cpu_percent.toFixed(0) + " %";
    $("#v-swap").textContent = fmtGB(sys.swap_used);
    $("#v-disk").textContent = fmtGB(sys.disk_free);
    $("#mem-total").textContent = Math.round(sys.ram_total / GB) + " GB";
    $("#mem-mid").textContent = Math.round(sys.ram_total / GB / 2) + " GB";

    if (sys.wired_limit) {
      const pct = (100 * sys.wired_limit) / sys.ram_total;
      $("#memstrip-wired").style.left = pct + "%";
      $("#wired-label").textContent = "GPU limit " + Math.round(sys.wired_limit / GB) + "G";
    }

    const track = $("#memstrip-track");
    track.innerHTML = "";
    const seg = (cls, bytes, label, title) => {
      const d = document.createElement("div");
      d.className = "memseg " + cls;
      d.style.width = (100 * bytes) / sys.ram_total + "%";
      d.title = title || label;
      if (label) d.innerHTML = `<span>${esc(label)}</span>`;
      track.appendChild(d);
    };
    for (const m of s.running) {
      const starting = m.state && m.state !== "ready";
      seg("model" + (starting ? " starting" : ""), m.rss || 0.02 * sys.ram_total,
        `${m.model} · ${m.rss ? fmtGB(m.rss) : "starting…"}`,
        `${m.model} — ${m.state}`);
    }
    seg("system", otherUsed, otherUsed > 4 * GB ? "system · " + fmtGB(otherUsed) : "", "macOS + apps: " + fmtGB(otherUsed));
  } catch {
    $("#swap-dot").classList.remove("up");
  }
}

/* ── model cards ── */
let modelsCache = [];

async function refreshModels() {
  try {
    const { models } = await api("/api/models");
    modelsCache = models;
    const wrap = $("#model-cards");
    wrap.innerHTML = "";
    const sel = $("#chat-model");
    const cur = sel.value;
    sel.innerHTML = "";
    for (const m of models) {
      if (!m.embedding) {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = m.name;
        sel.appendChild(o);
      }

      const card = document.createElement("div");
      card.className = "card" + (m.state !== "stopped" ? " loaded" : "");
      card.dataset.id = m.id;
      const perf = m.tok_s ? `${m.tok_s.toFixed(1)} tok/s gen${m.prompt_tok_s ? " · " + m.prompt_tok_s.toFixed(0) + " tok/s prompt" : ""}` : "–";
      card.innerHTML = `
        <div class="card-head"><h3>${esc(m.name)}</h3>
          <span class="state ${esc(m.state)}">${esc(m.state)}</span></div>
        <p class="desc">${esc(m.description || "")}</p>
        <dl>
          <dt>id</dt><dd>${esc(m.id)}${m.aliases.length ? " · " + m.aliases.map(esc).join(", ") : ""}</dd>
          <dt>context</dt><dd>${m.ctx ? m.ctx.toLocaleString() + " tokens" : "model default"}</dd>
          <dt>last speed</dt><dd class="perf">${perf}</dd>
          <dt>auto-unload</dt><dd class="unload-dd" data-ttl="${m.ttl || 0}" data-last="${m.last_activity || 0}">${m.lifecycle ? "🔗 " + esc(m.lifecycle) : m.ttl ? "after " + m.ttl + " s idle" : "never"}</dd>
          <dt>weights</dt><dd>${m.size ? fmtGB(m.size) : "–"}</dd>
        </dl>
        ${m.file_exists ? "" : `<div class="missing">weights file missing — still downloading?</div>`}
        <div class="card-actions">
          <button class="btn btn-primary act-load" ${m.state !== "stopped" || !m.file_exists ? "disabled" : ""}>Load into RAM</button>
          <button class="btn btn-danger act-unload" ${m.state === "stopped" ? "disabled" : ""}>Unload</button>
          <button class="btn btn-quiet act-settings" title="TTL, context, temperature">Settings</button>
        </div>
        <form class="card-settings" hidden>
          <label>ttl (s)<input name="ttl" type="number" min="0" step="60" value="${m.ttl ?? ""}"></label>
          <label>context<input name="ctx" type="number" min="1024" step="1024" value="${m.ctx ?? ""}" ${m.ctx ? "" : "disabled"}></label>
          <label>temp<input name="temp" type="number" min="0" max="2" step="0.05" value="${m.temp ?? ""}" ${m.temp != null ? "" : "disabled"}></label>
          <button class="btn btn-primary" type="submit">Apply</button>
        </form>`;
      card.querySelector(".act-load").addEventListener("click", async (e) => {
        e.target.textContent = "Loading…";
        e.target.disabled = true;
        try { await api(`/api/models/${m.id}/load`, { method: "POST" }); } catch {}
        refreshAll();
      });
      card.querySelector(".act-unload").addEventListener("click", async () => {
        await api(`/api/models/${m.id}/unload`, { method: "POST" });
        setTimeout(refreshAll, 500);
      });
      const form = card.querySelector(".card-settings");
      card.querySelector(".act-settings").addEventListener("click", () => (form.hidden = !form.hidden));
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fd = new FormData(form);
        const body = {};
        if (fd.get("ttl") !== "") body.ttl = Number(fd.get("ttl"));
        if (fd.get("ctx")) body.ctx = Number(fd.get("ctx"));
        if (fd.get("temp")) body.temp = Number(fd.get("temp"));
        const btn = form.querySelector("button");
        btn.textContent = "Applying…";
        try {
          await api(`/api/models/${m.id}/settings`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          await api("/api/swap/restart", { method: "POST" });
          btn.textContent = "Applied ✓";
          setTimeout(refreshAll, 800);
        } catch (err) {
          btn.textContent = "Apply";
          alert(err.message);
        }
      });
      wrap.appendChild(card);
    }
    if (cur) sel.value = cur;
  } catch {}
}

/* TTL countdown ticker: 1 s resolution, no network */
setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll(".card").forEach((card) => {
    const m = modelsCache.find((x) => x.id === card.dataset.id);
    const dd = card.querySelector(".unload-dd");
    if (!m || !dd) return;
    if (m.lifecycle) return;
    if (m.state !== "stopped" && m.ttl && m.last_activity) {
      const left = m.ttl - (now - m.last_activity);
      dd.innerHTML = left > 0
        ? `<span class="countdown">unloads in ${fmtClock(left)}</span> · ttl ${m.ttl}s`
        : `<span class="countdown">unloading…</span>`;
    } else {
      dd.textContent = m.ttl ? "after " + m.ttl + " s idle" : "never";
    }
  });
}, 1000);

$("#btn-unload-all").addEventListener("click", async () => {
  await api("/api/models/unload-all", { method: "POST" });
  setTimeout(refreshAll, 500);
});

function refreshAll() { refreshStatus(); refreshModels(); }
setInterval(refreshStatus, 2000);
setInterval(refreshModels, 6000);
refreshAll();

/* ── conversations (localStorage) ── */
const LS_KEY = "llmstack.convs";
const LS_ACTIVE = "llmstack.active";
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
  let out = "", reasoning = "", tokens = 0, tStart = performance.now(), tFirst = null;
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
    let buf = "", lastDraw = 0;
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
          if (j.error) { out += "\n`[error]` " + JSON.stringify(j.error); continue; }
          const delta = j.choices?.[0]?.delta || {};
          if (delta.reasoning_content) reasoning += delta.reasoning_content;
          if (delta.content) { out += delta.content; tokens++; }
          if (delta.content || delta.reasoning_content) {
            if (tFirst === null) { tFirst = performance.now(); el.classList.remove("thinking"); }
            const now = performance.now();
            if (now - lastDraw > 80) { redraw(); lastDraw = now; }
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

/* ── library ── */
function fitBadge(size) {
  if (!lastStatus) return "";
  const limit = lastStatus.system.wired_limit || lastStatus.system.ram_total * 0.75;
  if (size + 2 * GB > limit) return `<span class="fit no">won't fit</span>`;
  if (size + 2 * GB > limit * 0.85) return `<span class="fit tight">tight</span>`;
  return `<span class="fit ok">fits</span>`;
}

async function refreshFiles() {
  const { files } = await api("/api/files");
  const wrap = $("#file-list");
  wrap.innerHTML = files.length ? "" : `<div class="empty">No model files yet.</div>`;
  for (const f of files) {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span class="name" title="${esc(f.name)}">${esc(f.name)}</span>
      <span class="size">${fmtGB(f.size)}</span>
      <button class="btn btn-quiet btn-danger">Delete</button>`;
    row.querySelector("button").addEventListener("click", async () => {
      if (!confirm(`Delete ${f.name} from disk?`)) return;
      try { await api(`/api/files/${encodeURIComponent(f.name)}`, { method: "DELETE" }); }
      catch (e) { alert(e.message); }
      refreshFiles();
    });
    wrap.appendChild(row);
  }
}

$("#hf-search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#hf-query").value.trim();
  if (!q) return;
  const wrap = $("#hf-results");
  wrap.innerHTML = `<div class="empty">Searching…</div>`;
  const { results } = await api(`/api/hf/search?q=${encodeURIComponent(q)}`);
  wrap.innerHTML = results.length ? "" : `<div class="empty">No GGUF repos found.</div>`;
  for (const r of results.slice(0, 8)) {
    const row = document.createElement("div");
    row.className = "hf-row";
    row.innerHTML = `<span class="name">${esc(r.id)}</span>
      <span class="size">${r.downloads.toLocaleString()} ↓</span>
      <button class="btn">Files</button>`;
    row.querySelector("button").addEventListener("click", () => {
      $("#hf-repo").value = r.id;
      $("#hf-form").requestSubmit();
    });
    wrap.appendChild(row);
  }
});

$("#hf-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const repo = $("#hf-repo").value.trim();
  if (!repo) return;
  const wrap = $("#hf-files");
  wrap.innerHTML = `<div class="empty">Looking up ${esc(repo)}…</div>`;
  try {
    const { files } = await api(`/api/hf/files?repo=${encodeURIComponent(repo)}`);
    wrap.innerHTML = files.length ? "" : `<div class="empty">No .gguf files in that repo.</div>`;
    for (const f of files) {
      const row = document.createElement("div");
      row.className = "hf-row";
      row.innerHTML = `<span class="name" title="${esc(f.path)}">${esc(f.path)}</span>
        ${fitBadge(f.size)}
        <span class="size">${fmtGB(f.size)}</span>
        <button class="btn">Download</button>`;
      row.querySelector("button").addEventListener("click", async (ev) => {
        ev.target.disabled = true;
        ev.target.textContent = "Queued";
        await api("/api/downloads", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ repo, path: f.path }),
        });
        refreshDownloads();
      });
      wrap.appendChild(row);
    }
  } catch (err) {
    wrap.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
});

async function refreshDownloads() {
  const { downloads } = await api("/api/downloads");
  const wrap = $("#dl-list");
  if (!downloads.length) {
    wrap.innerHTML = `<div class="empty">Nothing downloading.</div>`;
    return;
  }
  wrap.innerHTML = "";
  for (const d of downloads) {
    const pct = d.total ? Math.min(100, (100 * d.bytes) / d.total) : 0;
    const rate = d.bytes / Math.max(d.elapsed, 1) / 1024 ** 2;
    const row = document.createElement("div");
    row.className = "dl-row";
    row.innerHTML = `
      <div class="dl-head"><span class="name">${esc(d.file)}</span>
        <span class="size">${d.state === "downloading" ? rate.toFixed(0) + " MB/s · " : ""}${fmtGB(d.bytes)}${d.total ? " / " + fmtGB(d.total) : ""} · ${esc(d.state)}</span>
        <button class="btn btn-quiet">${d.state === "downloading" ? "Cancel" : "Dismiss"}</button></div>
      <div class="dl-bar"><i style="width:${pct}%"></i></div>`;
    row.querySelector("button").addEventListener("click", async () => {
      await api(`/api/downloads/${encodeURIComponent(d.file)}`, { method: "DELETE" });
      refreshDownloads();
      refreshFiles();
    });
    wrap.appendChild(row);
  }
}
setInterval(() => {
  if ($("#tab-library").classList.contains("active")) refreshDownloads();
}, 2000);

/* ── brain ── */
async function brainStatus() {
  try {
    const s = await api("/api/brain/status");
    $("#brain-status").textContent = `${s.files} notes · ${s.chunks} chunks indexed · ${s.vault}`;
  } catch {}
}

$("#brain-reindex").addEventListener("click", async (e) => {
  e.target.disabled = true;
  e.target.textContent = "Indexing…";
  try {
    const r = await api("/api/brain/reindex", { method: "POST" });
    $("#brain-status").textContent = `reindexed: ${r.indexed} updated, ${r.removed} removed · ${r.files} notes, ${r.chunks} chunks`;
  } catch (err) {
    $("#brain-status").textContent = err.message;
  }
  e.target.disabled = false;
  e.target.textContent = "Reindex vault";
});

$("#brain-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#brain-q").value.trim();
  if (!q) return;
  const wrap = $("#brain-results");
  wrap.innerHTML = `<div class="empty">Searching (loads the embedding model if needed)…</div>`;
  try {
    const { results } = await api(`/api/brain/search?q=${encodeURIComponent(q)}`);
    wrap.innerHTML = results.length ? "" : `<div class="empty">No matches. Capture something with the brain command first.</div>`;
    for (const r of results) {
      const a = document.createElement("a");
      a.className = "brain-hit";
      a.href = `obsidian://open?vault=Brain&file=${encodeURIComponent(r.path.replace(/\.md$/, ""))}`;
      a.innerHTML = `
        <div class="hit-head">
          <span class="hit-title">${esc(r.title)}</span>
          <span><span class="hit-folder">${esc(r.folder)}</span> <span class="hit-score">${r.score}</span></span>
        </div>
        <div class="hit-snippet">${esc(r.snippet)}</div>`;
      wrap.appendChild(a);
    }
    brainStatus();
  } catch (err) {
    wrap.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  }
});

/* ── config ── */
async function loadConfig() {
  const { yaml } = await api("/api/config");
  $("#cfg-editor").value = yaml;
  $("#cfg-msg").textContent = "";
}

async function saveConfig() {
  const msg = $("#cfg-msg");
  try {
    await api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yaml: $("#cfg-editor").value }),
    });
    msg.textContent = "saved";
    msg.className = "cfg-msg ok";
    return true;
  } catch (e) {
    msg.textContent = e.message;
    msg.className = "cfg-msg err";
    return false;
  }
}

$("#cfg-save").addEventListener("click", saveConfig);
$("#cfg-apply").addEventListener("click", async () => {
  if (!(await saveConfig())) return;
  const msg = $("#cfg-msg");
  msg.textContent = "restarting llama-swap…";
  msg.className = "cfg-msg";
  try {
    await api("/api/swap/restart", { method: "POST" });
    msg.textContent = "saved · llama-swap restarted";
    msg.className = "cfg-msg ok";
    refreshAll();
  } catch (e) {
    msg.textContent = e.message;
    msg.className = "cfg-msg err";
  }
});

/* ── logs ── */
let logsStarted = false;
async function startLogs() {
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
      if (view.textContent.length > 300000) view.textContent = view.textContent.slice(-200000);
      if (atBottom) view.scrollTop = view.scrollHeight;
    }
  } catch {
    view.textContent += "\n[log stream disconnected]";
  }
  logsStarted = false;
}
$("#logs-clear").addEventListener("click", () => ($("#log-view").textContent = ""));
