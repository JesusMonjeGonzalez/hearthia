/* Durable sessions; only one history page and one live turn are held in memory. */
import { $, api, esc, renderMD, highlightIn } from "./api.js";
import { refreshStatus } from "./models.js";
import { ChatStream } from "./chat-stream.mjs";
import { describeTool } from "./tool-result.mjs";
import { autoContinuePrompt, planProgress, shouldAutoContinue } from "./plan-progress.mjs";
import { usageSummary } from "./usage-summary.mjs";

const ROOT = "/api/conversations";
let conversations = [], current = null, listOffset = 0;
let streaming = false, aborter = null, attachments = [], opening = 0;
let lastBreakdown = null;
let autoStrike = 0, autoSubmitting = false;
let jobsTimer = null;
const AUTO_LIMIT = 8;

const notice = (text) => { $("#chat-stats").textContent = text; };
const jsonPost = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const guarded = (fn) => async (...args) => { try { await fn(...args); } catch (error) { notice(error.message); } };
function remember(key, value) { try { localStorage.setItem(key, value); } catch { /* optional preference */ } }
function saved(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }

function setBusy(value) {
  streaming = value;
  $("#chat-send").hidden = value;
  $("#chat-stop").hidden = !value;
  for (const selector of ["#conv-new", "#conv-import", "#conv-refresh", "#conv-prev", "#conv-next", "#conv-fork", "#conv-retry", "#chat-older", "#chat-latest", "#chat-workspace", "#chat-system", "#chat-model", "#chat-mode"]) {
    $(selector).disabled = value;
  }
}

async function refreshList() {
  conversations = (await api(`${ROOT}?limit=50&offset=${listOffset}`)).conversations;
  const wrap = $("#conv-items");
  wrap.replaceChildren();
  for (const c of conversations) {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === current?.id ? " active" : "");
    item.innerHTML = `<span class="title">${esc(c.title)}</span><button class="del" title="Delete conversation">×</button>`;
    item.addEventListener("click", guarded(async (event) => {
      if (streaming || event.target.classList.contains("del")) return;
      await openConversation(c.id);
      $(".conv-list").classList.remove("open");
    }));
    item.querySelector(".del").addEventListener("click", guarded(async () => {
      if (streaming) return;
      await api(`${ROOT}/${c.id}`, { method: "DELETE" });
      if (current?.id === c.id) { current = null; renderConversation(); }
      await refreshList();
    }));
    wrap.appendChild(item);
  }
  $("#conv-prev").disabled = streaming || listOffset === 0;
  $("#conv-next").disabled = streaming || conversations.length < 50;
}

async function openConversation(id, before = null) {
  const ticket = ++opening;
  const page = await api(`${ROOT}/${id}?limit=40${before ? `&before=${before}` : ""}`);
  if (ticket !== opening) return;
  current = page;
  remember("hearthia.active", JSON.stringify(id));
  renderConversation();
  await refreshList();
}

async function newConversation() {
  if (streaming) return;
  const c = await api(ROOT, jsonPost({}));
  listOffset = 0;
  await openConversation(c.id);
}

function statsLine(stats) {
  if (!stats) return "";
  const parts = [];
  if (stats.prompt_n != null) {
    const cached = stats.cache_n ? ` · ${stats.cache_n} cached` : "";
    parts.push(`prefill ${stats.prompt_n} tok${cached}`);
  }
  if (stats.predicted_per_second) {
    parts.push(`${stats.predicted_per_second.toFixed(1)} tok/s · ${stats.predicted_n ?? "?"} tokens`);
  }
  return parts.join(" · ");
}

function addMessage(message) {
  const el = document.createElement("div");
  el.className = "msg " + message.role;
  if (message.role === "tool") {
    const tool = describeTool(message);
    el.classList.add(`tool-${tool.kind}`);
    if (tool.failed) el.classList.add("tool-failed");
    const details = document.createElement("details");
    details.open = tool.kind === "edit";
    const summary = document.createElement("summary");
    summary.textContent = tool.title;
    const pre = document.createElement("pre");
    if (tool.kind === "edit") {
      for (const line of tool.body.split("\n")) {
        const span = document.createElement("span");
        span.className = line.startsWith("+") ? "diff-add" : line.startsWith("-") ? "diff-remove" : "";
        span.textContent = line + "\n";
        pre.appendChild(span);
      }
    } else pre.textContent = tool.body;
    const detail = document.createElement("div");
    detail.className = "hint";
    detail.textContent = tool.detail;
    details.append(summary, detail, pre);
    el.appendChild(details);
  } else if (message.role === "assistant") {
    el.innerHTML = (message.reasoning ? `<details class="reasoning"><summary>Reasoning</summary><div class="rbody">${esc(message.reasoning)}</div></details>` : "") + renderMD(message.content || "");
    if (message.tool_calls) {
      const tool = document.createElement("div");
      tool.className = "hint";
      tool.textContent = message.tool_calls.map((t) => t.function.name).join(" · ");
      el.appendChild(tool);
    }
    if (message.model) {
      const meta = document.createElement("div");
      meta.className = "hint";
      meta.textContent =
        message.model +
        (message.stats?.predicted_per_second
          ? ` · ${message.stats.predicted_per_second.toFixed(1)} tok/s`
          : "");
      el.appendChild(meta);
    }
    if (message.error || message.partial) {
      const state = document.createElement("div");
      state.className = "msg-error";
      state.textContent = message.error || "Interrupted reply — saved partial output";
      el.appendChild(state);
    }
    highlightIn(el);
  } else el.textContent = message.content;
  $("#chat-log").appendChild(el);
  return el;
}

const params = new URLSearchParams(location.search);
const preset = {
  workspace: params.get("workspace") || "",
  model: params.get("model") || "",
  mode: params.get("mode") || "",
  fresh: params.get("new") === "1",
};
let presetModelApplied = false;

function applyPreset() {
  const workspace = $("#chat-workspace");
  if (!workspace.value && preset.workspace) workspace.value = preset.workspace;
  const model = $("#chat-model");
  if (!presetModelApplied && preset.model && [...model.options].some((o) => o.value === preset.model)) {
    model.value = preset.model;
    presetModelApplied = true;
  }
  if (preset.mode === "read" || preset.mode === "build") {
    $("#chat-mode").value = preset.mode;
    updateModeHint();
  }
}

function renderConversation() {
  $("#chat-log").replaceChildren();
  $("#chat-activity").textContent = "";
  $("#chat-older").hidden = !current?.next_before;
  $("#chat-latest").hidden = !current || current.messages.at(-1)?.seq === current.revision;
  if (!current?.messages.length) $("#chat-log").innerHTML = '<div class="empty">Choose a project and start a conversation.</div>';
  else for (const message of current.messages) addMessage(message);
  $("#chat-system").value = current?.system || "";
  $("#chat-workspace").value = current?.workspace || "";
  $("#chat-mode").value = current?.mode || "read";
  updateModeHint();
  if (current?.model) $("#chat-model").value = current.model;
  notice(current ? `Saved locally · ${current.status} · ${current.revision} messages` : "");
  renderVerification();
  renderPlan();
  renderUsagePanel();
  renderWindowUsage();
  applyPreset();
  // Opening an older page should land on that page, not be yanked to the
  // newest message; only the latest page scrolls to the bottom.
  const log = $("#chat-log");
  log.scrollTop = current?.next_before ? 0 : log.scrollHeight;
}

async function refreshJobs() {
  const el = $("#chat-jobs");
  try {
    const data = await api("/api/jobs");
    const running = (data.jobs || []).filter((job) => job.state === "running");
    el.replaceChildren();
    if (!running.length) {
      el.hidden = true;
      clearInterval(jobsTimer);
      jobsTimer = null;
      return;
    }
    el.hidden = false;
    el.append(`Background jobs: ${running.length} running — `);
    for (const job of running) {
      const button = document.createElement("button");
      button.className = "btn btn-quiet";
      button.textContent = `stop ${job.id}`;
      button.title = job.argv || "";
      button.addEventListener("click", guarded(async () => {
        await api(`/api/jobs/${job.id}/stop`, { method: "POST" });
        await refreshJobs();
      }));
      el.appendChild(button);
      el.append(" ");
    }
    if (!jobsTimer) jobsTimer = setInterval(guarded(refreshJobs), 10000);
  } catch {
    el.hidden = true;
  }
}

function renderVerification() {
  const health = current?.last_turn;
  const el = $("#chat-verify");
  if (health && health.edited?.length && !health.verified) {
    const shown = health.edited.slice(0, 3).join(", ");
    el.hidden = false;
    el.textContent =
      `Unverified: ${health.edited.length} file(s) edited with no command afterwards ` +
      `(${shown}${health.edited.length > 3 ? ", …" : ""})`;
  } else {
    el.hidden = true;
    el.textContent = "";
  }
}

function renderPlan() {
  const plan = current?.plan;
  const panel = $("#chat-plan");
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  if (!steps.length) {
    panel.hidden = true;
    $("#chat-plan-steps").replaceChildren();
    return;
  }
  panel.hidden = false;
  const age = plan.updated ? Math.max(0, Math.round((Date.now() / 1000 - plan.updated) / 60)) : null;
  const progress = planProgress(plan);
  $("#chat-plan-summary").textContent =
    `Plan · ${progress.done}/${progress.total} done` +
    (age != null ? ` · updated ${age < 1 ? "just now" : `${age} min ago`}` : "");
  const list = $("#chat-plan-steps");
  list.replaceChildren();
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    const finished = progress.doneSet.has(index + 1);
    item.textContent = `${finished ? "✔" : "○"} ${String(step)}`;
    if (finished) item.style.opacity = "0.65";
    list.appendChild(item);
  });
}

function renderUsagePanel() {
  const panel = $("#chat-usage");
  const lastAssistant = [...(current?.messages || [])].reverse()
    .find((m) => m.role === "assistant" && m.stats);
  const lines = usageSummary(current?.usage, current?.projection, lastAssistant?.stats);
  if (!lines) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("#chat-usage-summary").textContent = "Usage (this conversation)";
  const list = $("#chat-usage-lines");
  list.replaceChildren();
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    list.appendChild(item);
  }
}

function renderWindowUsage() {
  const usage = current?.usage;
  const projection = current?.projection;
  const el = $("#chat-context");
  if (!projection || !usage) { el.textContent = ""; return; }
  const parts = [
    `window ${Math.round(100 * projection.used_fraction)}%`,
    `${projection.remaining_tokens.toLocaleString()} tok free`,
  ];
  if (projection.growth_tokens_per_turn) {
    parts.push(`~${Math.round(projection.growth_tokens_per_turn).toLocaleString()} tok/turn`);
  }
  if (projection.turns_until_trimming != null) {
    parts.push(`≈${projection.turns_until_trimming} turns until oldest turns are omitted`);
  }
  el.textContent = parts.join(" · ");
  if (lastBreakdown) {
    const fmt = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));
    el.title =
      "prompt elements (tokens): " +
      Object.entries(lastBreakdown)
        .filter(([, value]) => value > 0)
        .map(([key, value]) => `${key.replaceAll("_", " ")} ${fmt(value)}`)
        .join(" · ");
  }
}

function updateModeHint() {
  $("#chat-mode-hint").textContent = $("#chat-mode").value === "build"
    ? "Develop: edit project files and run local commands with your user permissions."
    : "Consult: read and search files.";
}
$("#chat-mode").addEventListener("change", updateModeHint);

let searchTimer = null;
$("#conv-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(guarded(async () => {
    const query = $("#conv-search").value.trim();
    if (!query) { await refreshList(); return; }
    if (streaming) return;
    const results = (await api(`${ROOT}/search?q=${encodeURIComponent(query)}&limit=30`)).results;
    const wrap = $("#conv-items");
    wrap.replaceChildren();
    if (!results.length) {
      const empty = document.createElement("div");
      empty.className = "hint";
      empty.textContent = "No matches.";
      wrap.appendChild(empty);
      return;
    }
    for (const hit of results) {
      const item = document.createElement("div");
      item.className = "conv-item";
      item.title = hit.workspace || "";
      item.innerHTML =
        `<span class="title">${esc(hit.title)} · ${hit.hits} hit${hit.hits === 1 ? "" : "s"}</span>`;
      const snippet = document.createElement("span");
      snippet.className = "conv-snippet";
      // FTS marks matches with [brackets]; escaped first, then highlighted.
      snippet.innerHTML = esc(hit.snippet).replace(/\[([^\]]+)\]/g, "<mark>$1</mark>");
      item.appendChild(snippet);
      item.addEventListener("click", guarded(async () => {
        if (streaming) return;
        await openConversation(hit.id, hit.seq + 1);
        $(".conv-list").classList.remove("open");
      }));
      wrap.appendChild(item);
    }
  }), 250);
});

$("#conv-new").addEventListener("click", guarded(newConversation));
$("#conv-toggle").addEventListener("click", () => $(".conv-list").classList.toggle("open"));
$("#conv-refresh").addEventListener("click", guarded(async () => {
  if (streaming) return;
  if (current) await openConversation(current.id); else await refreshList();
}));
$("#conv-prev").addEventListener("click", guarded(async () => { listOffset = Math.max(0, listOffset - 50); await refreshList(); }));
$("#conv-next").addEventListener("click", guarded(async () => { listOffset += 50; await refreshList(); }));
$("#chat-older").addEventListener("click", guarded(async () => { if (current && !streaming) await openConversation(current.id, current.next_before); }));
$("#chat-latest").addEventListener("click", guarded(async () => { if (current && !streaming) await openConversation(current.id); }));
$("#conv-fork").addEventListener("click", guarded(async () => {
  if (!current || streaming) return;
  const forked = await api(`${ROOT}/${current.id}/fork`, jsonPost({}));
  listOffset = 0;
  await openConversation(forked.id);
  notice("Forked — this copy diverges from the original.");
}));

$("#conv-retry").addEventListener("click", guarded(async () => {
  if (!current || streaming) return;
  const retry = await api(`${ROOT}/${current.id}/retry`, jsonPost({}));
  listOffset = 0;
  await openConversation(retry.conversation.id);
  $("#chat-input").value = retry.message;
  $("#chat-form").requestSubmit();
}));

$("#conv-export").addEventListener("click", () => {
  if (!current) return;
  const link = document.createElement("a");
  link.href = `${ROOT}/${current.id}/export`;
  link.download = "conversation.md";
  link.click();
});

// Migration is explicit and idempotent; the original browser copy is retained.
$("#conv-import").addEventListener("click", guarded(async () => {
  if (streaming) return;
  const legacy = saved("hearthia.convs", []);
  if (!Array.isArray(legacy)) throw new Error("Legacy chat data is invalid; original data was preserved");
  let count = 0;
  for (const c of legacy) {
    await api(ROOT, jsonPost({ title: c.title || "Imported chat", system: c.system || "", model: c.model || "", messages: c.messages || [] }));
    count++;
  }
  await refreshList();
  notice(`Imported ${count} conversations. Original browser data retained.`);
}));

const sampling = saved("hearthia.sampling", {});
for (const [selector, key] of [["#s-temp", "temp"], ["#s-topp", "top_p"], ["#s-maxtok", "max_tokens"]]) {
  if (sampling[key] != null) $(selector).value = sampling[key];
  $(selector).addEventListener("change", () => {
    sampling[key] = $(selector).value;
    remember("hearthia.sampling", JSON.stringify(sampling));
  });
}

function renderAttachments() {
  $("#chat-attachments").innerHTML = attachments.map((a, i) => `<span class="attach-badge">${esc(a.name)} <button type="button" data-index="${i}">×</button></span>`).join("");
  $("#chat-attachments").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    attachments.splice(Number(button.dataset.index), 1);
    renderAttachments();
  }));
}
async function attach(files) {
  for (const file of files) {
    if (attachments.length >= 8 || file.size > 200_000 || attachments.reduce((n, a) => n + a.bytes, 0) + file.size > 400_000) {
      notice("Attachment limit: 8 files, 200 KB each, 400 KB total. Use workspace tools for large files.");
      break;
    }
    const content = await file.text();
    attachments.push({ name: file.name, content, bytes: file.size });
  }
  renderAttachments();
}
$("#chat-attach").addEventListener("click", () => {
  const input = document.createElement("input");
  input.type = "file"; input.multiple = true;
  input.addEventListener("change", guarded(() => attach(input.files)));
  input.click();
});
for (const target of [$("#chat-log"), $("#chat-form")]) {
  target.addEventListener("dragover", (event) => event.preventDefault());
  target.addEventListener("drop", guarded(async (event) => { event.preventDefault(); await attach(event.dataTransfer.files); }));
}

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("#chat-form").requestSubmit(); }
});
$("#chat-stop").addEventListener("click", () => aborter?.abort());

// Shortcuts: Escape stops a running turn, Cmd/Ctrl+K starts a new conversation.
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && streaming) {
    aborter?.abort();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!streaming) guarded(newConversation)();
  }
});

$("#chat-form").addEventListener("submit", guarded(async (event) => {
  event.preventDefault();
  if (streaming) return;
  if (!autoSubmitting) autoStrike = 0;  // a human turn resets the continuation budget
  autoSubmitting = false;
  const text = $("#chat-input").value.trim();
  if (!text && !attachments.length) return;
  const content = [text, ...attachments.map((a) => `File: ${a.name}\n\`\`\`\n${a.content}\n\`\`\``)].filter(Boolean).join("\n\n");
  const workspace = $("#chat-workspace").value.trim(), system = $("#chat-system").value.trim();
  const model = $("#chat-model").value || "default";
  const mode = $("#chat-mode").value;
  if (mode === "build" && !workspace) { notice("Choose a project directory for Develop mode."); return; }
  setBusy(true);
  try {
    if (!current) current = { ...await api(ROOT, jsonPost({})), messages: [] };
  } catch (error) { setBusy(false); throw error; }
  const id = current.id;
  const body = { session_id: id, revision: current.revision, model, workspace, system, mode, messages: [{ role: "user", content }] };
  for (const [selector, key] of [["#s-temp", "temperature"], ["#s-topp", "top_p"], ["#s-maxtok", "max_tokens"]]) {
    if ($(selector).value !== "") body[key] = Number($(selector).value);
  }
  setBusy(true);
  aborter = new AbortController();
  let reader, output = "", reasoning = "", lastDraw = 0, completed = false, accepted = false;
  let el, stats = null, failure = "", contextStatus = "";
  const started = performance.now();
  const draw = (final = false) => {
    if (!el) return;
    const log = $("#chat-log"), follow = log.scrollHeight - log.scrollTop - log.clientHeight < 100;
    el.innerHTML = (reasoning ? `<details class="reasoning"><summary>Reasoning</summary><div class="rbody">${esc(reasoning.slice(-60_000))}</div></details>` : "") + (final ? renderMD(output) : `<div style="white-space:pre-wrap">${esc(output)}</div>`);
    if (final) highlightIn(el);
    if (follow) log.scrollTop = log.scrollHeight;
  };
  try {
    notice("Preparing project and checking memory…");
    const response = await fetch("/api/chat", { ...jsonPost(body), signal: aborter.signal });
    if (!response.ok) throw new Error(await response.text());
    accepted = true;
    $("#chat-input").value = "";
    attachments = []; renderAttachments();
    $("#chat-log").querySelector(".empty")?.remove();
    addMessage({ role: "user", content });
    el = addMessage({ role: "assistant", content: "Waiting for model…" });
    reader = response.body.getReader();
    const parser = new ChatStream();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const message of parser.feed(value)) {
        if (message.done) completed = true;
        if (message.error) failure = message.error.message || JSON.stringify(message.error);
        if (message.context) {
          const c = message.context;
          lastBreakdown = c.element_tokens
            ? { ...c.element_tokens, tools_schema: c.tools_schema_tokens }
            : null;
          const used = c.measured_input_tokens ?? c.estimated_input_tokens ?? 0;
          const allowance = c.input_allowance_tokens || 0;
          const pct = allowance ? Math.round((100 * used) / allowance) : 0;
          const extra = [];
          if (c.omitted_turns) extra.push(`${c.omitted_turns} older turns omitted`);
          if (c.trimmed_tool_results) extra.push(`${c.trimmed_tool_results} tool results shortened`);
          if (c.trimmed_context_blocks) extra.push("project context shortened");
          if (c.history_limited) extra.push("bounded history window");
          contextStatus =
            `input ${used.toLocaleString()}/${allowance.toLocaleString()} tok (${pct}% of allowance · ` +
            `${Math.round(c.context_window / 1024)}K window) · ${c.max_tokens} output reserve · ` +
            `${c.measured_input_tokens ? "measured" : "estimated"}` +
            (extra.length ? " · " + extra.join(" · ") : "");
          notice(contextStatus);
        }
        if (message.tool_event) {
          const t = message.tool_event;
          $("#chat-activity").textContent = `${t.status} · ${t.label || t.name}${t.preview ? ` — ${t.preview}` : ""}`;
        }
        if (message.timings) stats = message.timings;
        const delta = message.choices?.[0]?.delta;
        if (delta?.content) output = (output + delta.content).slice(-300_000);
        if (delta?.reasoning_content) reasoning = (reasoning + delta.reasoning_content).slice(-60_000);
      }
      if (performance.now() - lastDraw > 150) { draw(); lastDraw = performance.now(); }
    }
    if (!completed) failure ||= "Connection ended before completion; partial output was saved.";
  } catch (error) {
    failure = error.name === "AbortError" ? "Stopped. Partial output saved." : error.message;
  } finally {
    if (reader) { try { await reader.cancel(); } catch {} reader.releaseLock(); }
    draw(true);
    aborter = null;
    try {
      refreshJobs();
    // Give the server a bounded interval to record cancellation before reloading.
      if (accepted) for (let attempt = 0; attempt < 10; attempt++) {
        const page = await api(`${ROOT}/${id}?limit=40`);
        current = page;
        if (page.status !== "running") break;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      if (accepted) renderConversation();
      await refreshList();
    } catch (error) { failure ||= `Unable to reload saved conversation: ${error.message}`; }
    setBusy(false);
    notice(failure || `${statsLine(stats) || `${((performance.now() - started) / 1000).toFixed(1)}s`} · ${contextStatus}`);
    refreshStatus();
    const autoEnabled = $("#chat-autocontinue").checked;
    if (
      !failure &&
      shouldAutoContinue({ plan: current?.plan, strike: autoStrike, limit: AUTO_LIMIT, enabled: autoEnabled })
    ) {
      autoStrike += 1;
      $("#chat-input").value = autoContinuePrompt(current?.plan, current?.last_turn);
      autoSubmitting = true;
      notice(`auto-continue ${autoStrike}/${AUTO_LIMIT} — the plan still has pending steps`);
      setTimeout(() => $("#chat-form").requestSubmit(), 250);
    }
  }
}));

await guarded(async () => {
  await refreshList();
  if (preset.fresh) {
    await newConversation();  // `hearth chat --new`: a clean conversation on boot
    return;
  }
  const active = saved("hearthia.active", null);
  const id = conversations.find((c) => c.id === active)?.id || conversations[0]?.id;
  if (id) await openConversation(id); else renderConversation();
})();
