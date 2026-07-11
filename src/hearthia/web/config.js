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
