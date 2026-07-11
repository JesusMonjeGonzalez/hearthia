/* brain.js — semantic search over the vault. */

import { $, api, esc } from "./api.js";

let vaultName = "";

export async function brainStatus() {
  try {
    const s = await api("/api/brain/status");
    vaultName = (s.vault || "").split("/").filter(Boolean).pop() || "";
    $("#brain-status").textContent = `${s.files} notes · ${s.chunks} chunks indexed · ${s.vault}`;
  } catch {}
}

$("#brain-reindex").addEventListener("click", async (e) => {
  e.target.disabled = true;
  e.target.textContent = "Indexing…";
  try {
    const r = await api("/api/brain/reindex", { method: "POST" });
    $("#brain-status").textContent =
      `reindexed: ${r.indexed} updated, ${r.removed} removed · ${r.files} notes, ${r.chunks} chunks`;
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
  if (!vaultName) brainStatus();
  const wrap = $("#brain-results");
  wrap.innerHTML = `<div class="empty">Searching (loads the embedding model if needed)…</div>`;
  try {
    const { results } = await api(`/api/brain/search?q=${encodeURIComponent(q)}`);
    wrap.innerHTML = results.length
      ? ""
      : `<div class="empty">No matches. Capture something with 'hearth brain capture' first.</div>`;
    for (const r of results) {
      const a = document.createElement("a");
      a.className = "brain-hit";
      a.href = `obsidian://open?vault=${encodeURIComponent(vaultName || "Brain")}&file=${encodeURIComponent(r.path.replace(/\.md$/, ""))}`;
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
