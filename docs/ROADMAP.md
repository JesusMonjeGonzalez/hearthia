# Hearthia — Roadmap

What remains beyond the shipped 0.4.0 feature set. Publishing-related work
(CI, releases, Homebrew tap) is tracked separately and excluded here.

Updated: 2026-09-01. **Unreleased (this revision):**

- **Self-calibrating memory model.** `hearth calibration` / `GET
  /api/calibration` expose a persisted, per-model correction factor learned
  from real measured RSS after each warm, folded into the budget gate,
  `hearth est`/`hearth advise`, and the dashboard's resident-size estimate.
  See `calibration.py`.
- **GGUF license & provenance inspector.** `hearth provenance`, `hearth gguf`
  and `GET /api/models/{id}/provenance` read license/lineage metadata
  straight from the GGUF header. See `provenance.py`.
- **Battery/thermal-aware RAM budget.** `hearth power` and a power-aware
  ceiling reduction folded into `plan_warm`/`plan_warm_now`. See `power.py`.
- **Model drift detector.** Re-quantized or replaced GGUFs are caught by a
  `(size, mtime)` fingerprint and reset their stale RAM calibration. See
  `drift.py`.
- **Predictive idle forecast.** `Telemetry.usage_forecast` predicts whether a
  model is likely to see more activity before its TTL idles it out, from the
  trend of its own recent request gaps. Surfaced in `hearth status` and the
  dashboard (🔮).
- **Calibration-aware `hearth advise`.** The KV/context ladder search and
  loadout planning now correct every candidate estimate through the same
  learned `calibration.py` factor the budget gate uses.
- **Shadow-eval health gate.** `hearth warm --verify` / `hearth verify` fire
  one real canary completion after a warm, catching a broken quantization or
  chat template that HTTP health alone cannot see. See `shadow_eval.py`.
- **GGUF weight dedupe across runtimes.** `hearth dedupe [--link]` finds and
  optionally hardlinks byte-identical GGUFs across Hearthia, Ollama and LM
  Studio folders. See `dedupe.py`.
- **Loadout auto-advisor on drift.** `drift.py` now re-checks every declared
  loadout referencing a changed model and surfaces the result at `GET
  /api/drift-warnings` and in `hearth doctor`. See
  `loadouts.loadouts_affected_by_drift`.
- **Loadout session replay.** `hearth sessions list|replay` records stable
  model combinations and replays one through the same whole-set budget check
  a declared loadout uses. See `sessions.py`.
- **Real token usage ledger.** `hearth usage` / `GET /api/usage` persist
  llama.cpp's own `--metrics` token counters per model, surviving restarts.
  See `usage_ledger.py`.
- **Context right-sizing advisor.** `hearth rightsize` / `GET
  /api/rightsizing` suggest a lower `--ctx-size` from llama.cpp's real
  `n_tokens_max` high-water mark. See `budget.rightsizing_advice`.
- **Speculative-decoding acceptance advisor.** `hearth spec-decode` / `GET
  /api/spec-decode` flag draft-model configs with a low real acceptance
  rate. See `spec_decode.py`.
- **Config lint.** `hearth lint` checks `--ctx-size` against a model's
  trained context, alias collisions, missing weights, and loadout/lifecycle
  rules referencing unknown models. See `lint.py`.
- **Warm-time ETA predictor.** `hearth warm` / `GET /api/models` predict how
  long a warm will take from a persisted per-model EWMA of real durations.
  See `load_time.py`.
- **Sleep prevention while warm.** A standard `caffeinate` hold tracks
  whether any model is warm. See `sleep_guard.py`.
- **Storage hygiene advisor.** `hearth storage` / `GET /api/storage` flag
  model weights unused for 30+ days. See `storage.py`.
- **Fleet health rehearsal.** `hearth rehearse` canary-checks every cold
  model, then cools it back down. See `rehearsal.py`.
- **Disk-space preflight for `hearth pull`.** Refuses a download upfront
  when the destination cannot hold it.

Updated: 2026-08-31. **Shipped in 0.4.0:**

- **TreePact facade.** Human-operated `hearth treepact` commands launch and
  inspect governed runs through a version-pinned subprocess. The dashboard and
  API expose only TreePact's bounded, read-only review contract.

**Unreleased:**

- Loadout membership is projected into registry metadata, exposed in the model
  API/dashboard, and shared models are preserved during loadout cooling.
- `hearth est`/`hearth advise` accept `--json` for scriptable machine-readable
  output.
- Fixed `hearth brain search` always falling through to the slow
  pure-Python cosine loop instead of the sqlite-vec KNN index (vec0 rejects
  bound `LIMIT ?`; use `k = ?`), and removed that now-dead fallback path.
- Chat: the conversation list is now reachable under 760px via a `#conv-toggle`
  drawer instead of being permanently `display:none`, and conversations can be
  exported as Markdown (client-side, from the already-loaded `localStorage`
  transcript — no server-side storage exists to export from).
- MCP now exposes point-in-time `hearthia://status`, `hearthia://health`, and
  `hearthia://logs/recent` resources, plus health/status subscriptions with
  `notifications/resources/updated` over the concurrent stdio transport.
- The Playwright smoke script now covers all dashboard tabs, mobile chat drawer
  reachability, and the read-only TreePact surface without console errors.

**Shipped in 0.3.2:**

- **Everything under one roof.** The short-lived `ggufram` standalone
  extraction is folded back into Hearthia (the repo remains public but
  Hearthia no longer depends on it). New: `hearth gguf <file>` prices any
  GGUF on disk from its header alone — geometry, KV cost per 1K tokens,
  resident estimate — with no config and no gateway.

**Shipped in 0.3.1:** (superseded — the arithmetic is internal again)

**Shipped in 0.3.0:**

- **MCP server** (`hearth mcp`): stdio JSON-RPC with the stdlib only —
  agents (OpenCode, Zed, Claude…) get budget-enforced warm/cool, what-if
  planning, loadouts and Brain search (`docs/MCP.md`).
- **`hearth advise`**: when a set does not fit, enumerate uniform
  change-sets that do — KV-cache quantisation, lower context, cooling a
  running model — each with the header-derived arithmetic.
- **Named loadouts** (`hearth loadout list|show|load|cool` + `[loadouts]`
  in config.toml): whole-set budget check, then per-model warms in order.
- **`GET /api/health`**: aggregate probe (gateway up, event watcher
  connected, crash-loop) for monitors and scripts; dashboard banners were
  already fed by `/api/status`.
- **Configurable Brain filing**: `[brain].folders` and `[brain].prompt_path`
  replace the hardcoded vault layout.
- `est` now points at `advise` when the verdict is DOES NOT FIT.

Closed since the last revision (verified in code): line-anchored config
validation (`config.js` jumps to the ruamel error), vault name derived from
`/api/brain/status`, TTL insertion for lifecycle-managed models, brain shim
delegation, `hearth logs -f`, pull progress + `--add`, structured hearthd
logging, dashboard failure banners.

## Remaining P2

No P2 items currently remain.

## Candidate next differentiators (not yet started)

Ideas evaluated for uniqueness against Ollama, LM Studio, llama-swap, Jan and
LocalAI before any code is written. None of these are committed work yet —
listed here, not in the changelog, until they exist as tested code.

- **Adaptive pre-warming.** Learn each model's actual usage cadence (time of
  day, follow-on app) from `telemetry.activity` history and pre-warm inside
  the budget gate slightly before it is next needed, instead of purely
  reactive TTL cooling. The gap-tracking added for the idle forecast
  (`Telemetry._gaps`) is a first building block toward this.
- **Energy accounting.** Deliberately deferred: an honest Wh-per-1K-tokens
  figure needs real SoC power draw, and `powermetrics` requires `sudo` on
  macOS — inconsistent with a loopback-only, single-user daemon that never
  asks for elevated privileges. Revisit only if a non-privileged power
  signal becomes available.
- **Focus/Do Not Disturb awareness.** Deliberately deferred: macOS Focus
  state has no stable public API — every community tool reads an
  undocumented `~/Library/DoNotDisturb` plist that breaks across macOS
  releases. Not worth shipping as "supported" on that foundation.

## Remaining P3 — polish and hardening

No P3 items currently remain.

## Explicitly out (per spec §13 / current decision)

Multi-user/auth · Linux/Windows · v2 single binary · auto-configuring
Continue/OpenCode/Obsidian · MLX backend · anything requiring GitHub
(Actions CI, releases, Homebrew tap).
