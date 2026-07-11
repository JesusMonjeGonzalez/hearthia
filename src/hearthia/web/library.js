/* library.js — on-disk files, HF search, downloads. */

import { $, api, GB, fmtGB, esc } from "./api.js";
import { getStatus, refreshAll } from "./models.js";

function fitBadge(size) {
  const lastStatus = getStatus();
  if (!lastStatus) return "";
  const limit = lastStatus.system.wired_limit || lastStatus.system.ram_total * 0.75;
  if (size + 2 * GB > limit) return `<span class="fit no">won't fit</span>`;
  if (size + 2 * GB > limit * 0.85) return `<span class="fit tight">tight</span>`;
  return `<span class="fit ok">fits</span>`;
}

export async function refreshFiles() {
  const { files } = await api("/api/files");
  const wrap = $("#file-list");
  wrap.innerHTML = files.length ? "" : `<div class="empty">No model files yet.</div>`;
  for (const f of files) {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span class="name" title="${esc(f.name)}">${esc(f.name)}</span>
      ${f.configured ? `<span class="fit ok">in config</span>` : ""}
      <span class="size">${fmtGB(f.size)}</span>
      ${f.configured ? "" : `<button class="btn btn-primary act-add">Add to config</button>`}
      <button class="btn btn-quiet btn-danger act-del">Delete</button>`;
    row.querySelector(".act-del").addEventListener("click", async () => {
      if (!confirm(`Delete ${f.name} from disk?`)) return;
      try {
        await api(`/api/files/${encodeURIComponent(f.name)}`, { method: "DELETE" });
      } catch (e) {
        alert(e.message);
      }
      refreshFiles();
    });
    wrap.appendChild(row);
    if (!f.configured) {
      const stem = f.name.replace(/\.gguf$/, "");
      const form = document.createElement("form");
      form.className = "file-add-form";
      form.hidden = true;
      form.innerHTML = `
        <label>id<input name="id" value="${esc(stem.toLowerCase().replace(/ /g, "-"))}" required></label>
        <label>name<input name="name" value="${esc(stem)}"></label>
        <label>context<input name="ctx" type="number" min="1024" step="1024" value="32768"></label>
        <label>ttl (s)<input name="ttl" type="number" min="0" step="60" value="600"></label>
        <button class="btn btn-primary" type="submit">Add &amp; restart</button>`;
      row.querySelector(".act-add").addEventListener("click", () => (form.hidden = !form.hidden));
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fd = new FormData(form);
        const btn = form.querySelector("button[type=submit]");
        btn.textContent = "Adding…";
        btn.disabled = true;
        try {
          await api("/api/models/add", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              id: fd.get("id"),
              name: fd.get("name"),
              file: f.name,
              ctx: Number(fd.get("ctx")) || undefined,
              ttl: Number(fd.get("ttl")) || undefined,
            }),
          });
          btn.textContent = "Restarting gateway…";
          await api("/api/swap/restart", { method: "POST" });
          refreshFiles();
          refreshAll();
        } catch (err) {
          btn.textContent = "Add & restart";
          btn.disabled = false;
          alert(err.message);
        }
      });
      wrap.appendChild(form);
    }
  }
}

export async function refreshDownloads() {
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
        <span class="size">${d.state === "downloading" ? rate.toFixed(0) + " MB/s · " : ""}${fmtGB(d.bytes)}${d.total ? " / " + fmtGB(d.total) : ""} · ${esc(d.state)}${d.error ? " — " + esc(d.error) : ""}</span>
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

setInterval(() => {
  if ($("#tab-library").classList.contains("active")) refreshDownloads();
}, 2000);
