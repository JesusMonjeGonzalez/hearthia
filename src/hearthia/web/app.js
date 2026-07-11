/* app.js — Hearthia dashboard entry point. ES module that imports per-tab modules. */

import { refreshAll } from "./models.js";
import { brainStatus } from "./brain.js";
import { refreshFiles, refreshDownloads } from "./library.js";
import { loadConfig } from "./config.js";
import { startLogs } from "./logs.js";

/* ── tabs ── */
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document
      .querySelectorAll(".panel")
      .forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
    if (t.dataset.tab === "logs") startLogs();
    if (t.dataset.tab === "config") loadConfig();
    if (t.dataset.tab === "library") {
      refreshFiles();
      refreshDownloads();
    }
    if (t.dataset.tab === "brain") brainStatus();
  }),
);

/* ── boot ── */
refreshAll();
