/* treepact.js — read-only TreePact review panel.
 *
 * This module only ever issues GET requests to /api/treepact/*. It cannot
 * start, resume, cancel or clean up a run — those operations exist solely in
 * TreePact's own CLI, under explicit human authorization. Every value shown
 * here comes from TreePact's minimized `review` contract (run identity,
 * state, gates, evidence metadata); it never includes task text, repository
 * or worktree paths, artifact content, prompts, logs or diffs. All text is
 * rendered with textContent, never interpreted as HTML or Markdown, because
 * it originates outside Hearthia's own trust boundary. */

import { $, api } from "./api.js";

let pollTimer = null;
let selectedRunId = null;

function empty(text) {
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = text;
  return div;
}

function fitClass(value) {
  const v = (value || "").toLowerCase();
  if (["accepted", "passed"].includes(v)) return "ok";
  if (["rejected", "failed"].includes(v)) return "no";
  if (["needs_review", "insufficient_evidence", "cancelled"].includes(v)) return "tight";
  return "";
}

function runRow(run) {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "file-row treepact-row";
  row.dataset.runId = run.run_id;
  if (run.run_id === selectedRunId) row.classList.add("selected");

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = run.run_id;
  row.appendChild(name);

  const badge = document.createElement("span");
  badge.className = `fit ${fitClass(run.decision || run.state)}`;
  badge.textContent = run.decision || run.state;
  row.appendChild(badge);

  const meta = document.createElement("span");
  meta.className = "hint";
  meta.textContent = run.updated_at;
  row.appendChild(meta);

  row.addEventListener("click", () => selectRun(run.run_id));
  return row;
}

export async function refreshTreePactRuns() {
  const list = $("#treepact-runs");
  try {
    const doc = await api("/api/treepact/runs?limit=20");
    list.replaceChildren();
    if (!doc.runs.length) {
      list.appendChild(empty("No TreePact runs yet."));
      return;
    }
    doc.runs.forEach((run) => list.appendChild(runRow(run)));
  } catch (err) {
    list.replaceChildren(empty(`TreePact is unavailable: ${err.message}`));
  }
}

function gateRow(gate) {
  const row = document.createElement("div");
  row.className = "treepact-gate";

  const badge = document.createElement("span");
  badge.className = `fit ${fitClass(gate.state)}`;
  badge.textContent = gate.state;
  row.appendChild(badge);

  const label = document.createElement("span");
  label.textContent = `${gate.gate_id} — ${gate.reason_code}`;
  row.appendChild(label);

  return row;
}

function artifactRow(artifact) {
  const row = document.createElement("div");
  row.className = "treepact-artifact";
  row.textContent =
    `${artifact.kind} · ${artifact.media_type} · ${artifact.size_bytes} B · ` +
    `${artifact.sha256.slice(0, 12)}…`;
  return row;
}

function detailField(label, value) {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value ?? "—";
  return [dt, dd];
}

export async function selectRun(runId) {
  selectedRunId = runId;
  document.querySelectorAll(".treepact-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.runId === runId);
  });

  const detail = $("#treepact-detail");
  detail.replaceChildren(empty("Loading…"));
  try {
    const doc = await api(`/api/treepact/runs/${encodeURIComponent(runId)}`);
    const run = doc.run;

    const card = document.createElement("div");
    card.className = "card treepact-detail-card";

    const dl = document.createElement("dl");
    detailField("Project", run.project_id).forEach((el) => dl.appendChild(el));
    detailField("State", run.state).forEach((el) => dl.appendChild(el));
    detailField("Decision", run.decision).forEach((el) => dl.appendChild(el));
    detailField("Reason", run.reason_code).forEach((el) => dl.appendChild(el));
    detailField("Assurance", run.assurance_level).forEach((el) => dl.appendChild(el));
    detailField("Created", run.created_at).forEach((el) => dl.appendChild(el));
    detailField("Updated", run.updated_at).forEach((el) => dl.appendChild(el));
    card.appendChild(dl);

    const gatesHeading = document.createElement("h3");
    gatesHeading.textContent = "Gates";
    card.appendChild(gatesHeading);
    if (run.gates.length) {
      run.gates.forEach((gate) => card.appendChild(gateRow(gate)));
    } else {
      card.appendChild(empty("No gates calculated yet."));
    }

    const evidenceHeading = document.createElement("h3");
    evidenceHeading.textContent = "Evidence";
    card.appendChild(evidenceHeading);
    const bundleLine = document.createElement("div");
    bundleLine.className = "hint";
    bundleLine.textContent = run.evidence.bundle_available
      ? `Decision bundle available. Event chain head: ${run.evidence.event_chain_head?.slice(0, 16)}…`
      : "No decision bundle recorded for this run.";
    card.appendChild(bundleLine);
    if (run.evidence.artifacts.length) {
      run.evidence.artifacts.forEach((artifact) => card.appendChild(artifactRow(artifact)));
    }

    detail.replaceChildren(card);
  } catch (err) {
    detail.replaceChildren(empty(`Could not load this run: ${err.message}`));
  }
}

export function startTreePactPolling() {
  stopTreePactPolling();
  refreshTreePactRuns();
  pollTimer = setInterval(() => {
    if (document.hidden) return;
    refreshTreePactRuns();
  }, 5000);
}

export function stopTreePactPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

$("#treepact-refresh")?.addEventListener("click", refreshTreePactRuns);
