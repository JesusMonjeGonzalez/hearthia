/* models.js — status, memory map, model cards, TTL countdown rings. */

import { $, api, GB, fmtGB, esc, fmtClock, stateLabel, STATE_WORDS } from "./api.js";

let lastStatus = null;
let modelsCache = [];

export function getStatus() {
  return lastStatus;
}

export async function refreshStatus() {
  try {
    const s = await api("/api/status");
    lastStatus = s;
    $("#swap-dot").classList.toggle("up", s.swap_up);
    $("#demo-badge").hidden = !s.demo;
    document.body.classList.toggle(
      "hearth-warm",
      s.running.some((m) => m.state === "ready"),
    );
    setBanner(
      !s.swap_up
        ? "Gateway is down — start it with `hearth up gateway`, then check the Logs tab."
        : s.health && !s.health.events_connected
          ? "Gateway event stream disconnected — reconnecting. Activity tracking is paused."
          : s.health && s.health.crash_loop
            ? "A model server is crash-looping (3+ exits in 5 min) — check the Logs tab."
            : "",
    );
    const sys = s.system;
    const models = s.running.filter((m) => m.rss);
    const modelBytes = models.reduce((a, m) => a + m.rss, 0);
    const otherUsed = Math.max(0, sys.ram_used - modelBytes);
    const estFor = (id) => (modelsCache.find((x) => x.id === id) || {}).est_resident;

    $("#v-models").textContent = s.running.length
      ? s.running.map((m) => m.model).join(", ")
      : "none";
    $("#v-ram").textContent = `${fmtGB(sys.ram_used)} / ${fmtGB(sys.ram_total)}`;
    const tps = s.running.map((m) => m.tok_s).filter(Boolean);
    $("#v-toks").textContent = tps.length
      ? tps.map((t) => t.toFixed(0)).join("/") + " tok/s"
      : "–";
    $("#v-cpu").textContent = sys.cpu_percent.toFixed(0) + " %";
    $("#v-swap").textContent = fmtGB(sys.swap_used);
    $("#v-disk").textContent = fmtGB(sys.disk_free);
    $("#mem-total").textContent = Math.round(sys.ram_total / GB) + " GB";
    $("#mem-mid").textContent = Math.round(sys.ram_total / GB / 2) + " GB";

    if (sys.wired_limit) {
      const pct = (100 * sys.wired_limit) / sys.ram_total;
      $("#memstrip-wired").style.left = pct + "%";
      $("#wired-label").textContent =
        "GPU limit " + Math.round(sys.wired_limit / GB) + "G";
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
      const est = estFor(m.model);
      const bytes = m.rss || est || 0.02 * sys.ram_total;
      const label = m.rss
        ? `${m.model} · ${fmtGB(m.rss)}`
        : `${m.model} · ${starting ? "kindling…" : ""}est. ${fmtGB(bytes)}`;
      const title = `${m.model} — ${stateLabel(m.state)}${m.rss ? "" : ` (est. resident ${fmtGB(bytes)})`}`;
      seg("model" + (starting ? " starting" : ""), bytes, label, title);
    }
    seg(
      "system",
      otherUsed,
      otherUsed > 4 * GB ? "system · " + fmtGB(otherUsed) : "",
      "macOS + apps: " + fmtGB(otherUsed),
    );
  } catch {
    $("#swap-dot").classList.remove("up");
    setBanner("Hearthia daemon unreachable — is hearthd running? (`hearth up daemon`)");
  }
}

function setBanner(text) {
  const b = $("#banner");
  b.textContent = text;
  b.hidden = !text;
}

function ttlRingSVG(left, total) {
  if (!total || left <= 0) return "";
  const pct = Math.max(0, Math.min(1, left / total));
  const r = 14;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - pct);
  const color = pct > 0.5 ? "#E8A33D" : pct > 0.2 ? "#E8753D" : "#E83D3D";
  return `<svg class="ttl-ring" viewBox="0 0 32 32">
    <circle cx="16" cy="16" r="${r}" fill="none" stroke="#2c261e" stroke-width="3"/>
    <circle class="ttl-arc" cx="16" cy="16" r="${r}" fill="none" stroke="${color}" stroke-width="3"
      stroke-dasharray="${circ}" stroke-dashoffset="${offset}"
      transform="rotate(-90 16 16)"/>
    <text class="ttl-text" x="16" y="20" text-anchor="middle" font-size="9" fill="${color}">${Math.ceil(left)}s</text>
  </svg>`;
}

function updateTtlRing(svg, left, total) {
  const pct = Math.max(0, Math.min(1, left / total));
  const r = 14;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - pct);
  const color = pct > 0.5 ? "#E8A33D" : pct > 0.2 ? "#E8753D" : "#E83D3D";
  const arc = svg.querySelector(".ttl-arc");
  const text = svg.querySelector(".ttl-text");
  if (arc) {
    arc.style.strokeDashoffset = offset;
    arc.style.stroke = color;
  }
  if (text) {
    text.textContent = `${Math.ceil(left)}s`;
    text.style.fill = color;
  }
}

export async function refreshModels() {
  try {
    const { models } = await api("/api/models");
    modelsCache = models;
    const wrap = $("#model-cards");
    // skip the rebuild while the user is mid-interaction: recreating the cards
    // would close an open settings form and steal focus from its inputs
    const focused = document.activeElement;
    const editing =
      wrap.querySelector(".card-settings:not([hidden])") ||
      (wrap.contains(focused) && focused.matches("input, select, textarea"));
    if (editing) return;
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
      const emberState = stateLabel(m.state);
      card.className = "card" + (m.state !== "stopped" ? " loaded" : "");
      card.dataset.id = m.id;
      card.classList.add(`ember-${emberState}`);
      const perf = m.tok_s
        ? `${m.tok_s.toFixed(1)} tok/s gen${m.prompt_tok_s ? " · " + m.prompt_tok_s.toFixed(0) + " tok/s prompt" : ""}`
        : "–";
      const loadoutBadges = (m.loadouts || []).length
        ? `<div class="loadout-badges">${m.loadouts.map((name) => `<span class="loadout-badge">${esc(name)}</span>`).join("")}</div>`
        : "";
      const lifecycleTxt = m.roles && m.roles.length
        ? "🔗 " + m.roles.map(esc).join(", ")
        : "";
      const unloadTxt = lifecycleTxt
        ? lifecycleTxt
        : m.ttl
          ? "after " + m.ttl + " s idle"
          : "never";
      card.innerHTML = `
         <div class="card-head"><h3>${esc(m.name)}</h3>
           <span class="state ember-${esc(emberState)}">${esc(emberState)}</span></div>
         <p class="desc">${esc(m.description || "")}</p>
         ${loadoutBadges}
         <dl>
          <dt>id</dt><dd>${esc(m.id)}${m.aliases.length ? " · " + m.aliases.map(esc).join(", ") : ""}</dd>
          <dt>context</dt><dd>${m.ctx ? m.ctx.toLocaleString() + " tokens" : "model default"}</dd>
          <dt>last speed</dt><dd class="perf">${perf}</dd>
          <dt>auto-unload</dt><dd class="unload-dd" data-ttl="${m.ttl || 0}" data-last="${m.last_activity || 0}" data-lifecycle="${lifecycleTxt ? 1 : 0}"><span class="ttl-wrap"></span>${esc(unloadTxt)}</dd>
          <dt>weights</dt><dd>${m.size ? fmtGB(m.size) : "–"}${m.est_resident ? ` · <span class="est-res" title="From the GGUF header: weights + KV cache at the configured context${m.est_known ? "" : " (file-size guess)"}">est. ${fmtGB(m.est_resident)} resident</span>` : ""}</dd>
        </dl>
        ${m.file_exists ? "" : `<div class="missing">weights file missing — still downloading?</div>`}
        <div class="card-actions">
          <button class="btn btn-primary act-load" ${m.state !== "stopped" || !m.file_exists ? "disabled" : ""}>Warm</button>
          <button class="btn btn-cool act-unload" ${m.state === "stopped" ? "disabled" : ""}>Cool</button>
          <button class="btn btn-quiet act-settings" title="TTL, context, temperature">Settings</button>
        </div>
        <form class="card-settings" hidden>
          <label>ttl (s)<input name="ttl" type="number" min="0" step="60" value="${m.ttl ?? ""}"></label>
          <label>context<input name="ctx" type="number" min="1024" step="1024" value="${m.ctx ?? ""}" ${m.ctx ? "" : "disabled"}></label>
          <label>temp<input name="temp" type="number" min="0" max="2" step="0.05" value="${m.temp ?? ""}" ${m.temp != null ? "" : "disabled"}></label>
          <button class="btn btn-primary" type="submit">Apply</button>
        </form>`;
      card.querySelector(".act-load").addEventListener("click", async (e) => {
        e.target.textContent = "Kindling…";
        e.target.disabled = true;
        e.target.blur();
        try {
          await api(`/api/models/${m.id}/load`, { method: "POST" });
        } catch (err) {
          e.target.textContent = "Failed";
          alert(`Load failed: ${err.message}`);
        }
        refreshAll();
      });
      card.querySelector(".act-unload").addEventListener("click", async (e) => {
        e.target.blur();
        try {
          await api(`/api/models/${m.id}/unload`, { method: "POST" });
        } catch (err) {
          alert(`Unload failed: ${err.message}`);
        }
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
    if (cur) {
      sel.value = cur;
    } else {
      const saved = JSON.parse(localStorage.getItem("hearthia.sampling") || "{}");
      if (saved.model) sel.value = saved.model;
    }
  } catch {}
}

/* TTL countdown ticker: 1 s resolution, no network.
   Updates the SVG ring in place (no flicker) and the countdown text. */
setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll(".card").forEach((card) => {
    const m = modelsCache.find((x) => x.id === card.dataset.id);
    if (!m) return;
    const dd = card.querySelector(".unload-dd");
    const ringWrap = card.querySelector(".ttl-wrap");
    if (!dd || !ringWrap) return;
    const hasLifecycle = dd.dataset.lifecycle === "1";
    if (hasLifecycle) return;
    if (m.state !== "stopped" && m.ttl && m.last_activity) {
      const left = m.ttl - (now - m.last_activity);
      if (left > 0) {
        const existing = ringWrap.querySelector("svg.ttl-ring");
        if (existing) {
          updateTtlRing(existing, left, m.ttl);
        } else {
          ringWrap.innerHTML = ttlRingSVG(left, m.ttl);
        }
        const textNode = dd.lastChild;
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          textNode.textContent = ` · unloads in ${fmtClock(left)}`;
        } else {
          dd.appendChild(document.createTextNode(` · unloads in ${fmtClock(left)}`));
        }
      } else {
        ringWrap.innerHTML = "";
        const textNode = dd.lastChild;
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          textNode.textContent = " · cooling…";
        }
      }
    } else {
      if (ringWrap.firstChild) ringWrap.innerHTML = "";
      const textNode = dd.lastChild;
      if (textNode && textNode.nodeType === Node.TEXT_NODE) {
        textNode.textContent = m.ttl ? "after " + m.ttl + " s idle" : "never";
      }
    }
  });
}, 1000);

$("#btn-unload-all").addEventListener("click", async () => {
  try {
    await api("/api/models/unload-all", { method: "POST" });
  } catch (err) {
    alert(`Cool down failed: ${err.message}`);
  }
  setTimeout(refreshAll, 500);
});

export function refreshAll() {
  refreshStatus();
  refreshModels();
}

setInterval(refreshStatus, 2000);
setInterval(refreshModels, 6000);
