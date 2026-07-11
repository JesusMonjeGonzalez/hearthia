/* config.js — YAML editor, save, restart. */

import { $, api } from "./api.js";
import { refreshAll } from "./models.js";

export async function loadConfig() {
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
    jumpToErrorLine(e.message);
    return false;
  }
}

/* ruamel errors embed "line N, column M" — put the cursor there */
function jumpToErrorLine(message) {
  const m = message.match(/line (\d+)/);
  if (!m) return;
  const line = parseInt(m[1], 10);
  const ed = $("#cfg-editor");
  const lines = ed.value.split("\n");
  const offset = lines.slice(0, line - 1).reduce((a, l) => a + l.length + 1, 0);
  ed.focus();
  ed.setSelectionRange(offset, offset + (lines[line - 1]?.length || 0));
  const lineHeight = parseFloat(getComputedStyle(ed).lineHeight) || 21;
  ed.scrollTop = Math.max(0, (line - 4) * lineHeight);
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
