# Changelog

## 0.5.0 — 2026-09-01

### Added

- **Warm-time ETA predictor** (`hearth warm` prints a predicted duration,
  `GET /api/models` exposes `eta_seconds`): times every real warm end to
  end and folds it into a persisted per-model EWMA, so the next warm of
  that model says roughly how long it will take instead of leaving a
  spinner with no estimate. No local-model runtime predicts its own warm
  time from real history.
- **Sleep prevention while warm** (`sleep_guard.py`, `sleep_prevented` in
  `GET /api/status` and `hearth status`): holds a standard `caffeinate -s
  -i` process for as long as any model is warm, releasing it the instant
  the last one cools — no local-model runtime works around macOS
  suspending an in-flight generation when the lid closes.
- **Storage hygiene advisor** (`hearth storage`, `GET /api/storage`):
  cross-references each model's real file size on disk with when it was
  last actually warmed, flagging weights unused for 30+ days — real
  numbers, not a guess, and independent of `--metrics` being configured.
- **Fleet health rehearsal** (`hearth rehearse [model_id...]`): warms every
  cold model just long enough to fire one shadow-eval canary
  (`shadow_eval.py`), then cools it back down — an explicit, manual health
  check across the whole roster that never disturbs a model already in
  use. No local-model runtime offers this.
- **Disk-space preflight for `hearth pull`**: refuses a download upfront
  when the destination volume does not have enough free space for it,
  instead of failing partway through a multi-gigabyte transfer.
- **Speculative-decoding acceptance advisor** (`hearth spec-decode`, `GET
  /api/spec-decode`): folds llama.cpp's `spec_decode_num_draft_tokens_total`
  and `spec_decode_num_accepted_tokens_total` counters into a persisted
  per-model acceptance rate, and flags models below 30% acceptance (past
  200 draft tokens) as very likely paying drafting overhead without a real
  speedup. No local-model runtime surfaces this ratio anywhere.
- **Config lint** (`hearth lint`): checks `llama-swap.yaml` against the
  models' real GGUF headers and Hearthia's own settings — `--ctx-size`
  exceeding a model's trained context without a RoPE-scaling flag, alias
  collisions between models, missing weights files, and loadout/lifecycle
  rules pointing at unknown model ids. No local-model runtime lints its
  config against the models it actually manages.
- **Real token usage ledger** (`hearth usage`, `GET /api/usage`): folds
  llama.cpp's own `--metrics` counters (`prompt_tokens_total`,
  `tokens_predicted_total`) into a persisted, per-model lifetime total that
  survives a cool/warm cycle and a daemon restart — measured, not estimated,
  and tolerant of the counter reset a process restart causes. No local-model
  runtime keeps this history.
- **Context right-sizing advisor** (`hearth rightsize`, `GET
  /api/rightsizing`): reads llama.cpp's `n_tokens_max` metric — the real
  high-water mark of context actually used — and suggests a lower
  `--ctx-size` with the GiB of KV cache it would free, only once a model has
  genuinely never needed the ceiling it is configured with.
- **Calibration-aware `hearth advise`**: the KV-cache/context ladder search
  now corrects every candidate estimate through the same learned
  `calibration.py` factor the budget gate uses, so a suggested change-set
  reflects this Mac's measured reality — not just the header arithmetic.
  Loadout planning (`loadouts.py`) picks up the same correction.
- **Shadow-eval health gate** (`hearth warm --verify`, `hearth verify
  <model>`): fires one minimal real completion at a model after it reports
  HTTP-healthy, catching a broken chat template, quantization, or
  `--ctx-size` overflow that a health check alone cannot see. No local-model
  runtime verifies actual inference before calling a warm "healthy".
- **GGUF weight dedupe across runtimes** (`hearth dedupe [--path DIR]
  [--link]`): finds byte-identical GGUFs across Hearthia's own `models/`,
  Ollama and LM Studio folders (same size, then same SHA-256) and, with
  `--link`, reclaims the duplicated disk space with hardlinks — same
  filesystem only, reported rather than silently skipped otherwise.
- **Loadout auto-advisor on drift** (`GET /api/drift-warnings`, surfaced in
  `hearth doctor`): when `drift.py` catches a re-quantized or replaced GGUF,
  every declared loadout referencing that model is immediately re-checked
  against the budget, instead of waiting for the next `hearth loadout load`
  to fail with a stale plan.
- **Loadout session replay** (`hearth sessions list|replay`, `GET/POST
  /api/sessions`): records stable combinations of models that were warm
  together (60s+, not a fleeting debug warm) and replays one with a single
  command through the same whole-set budget check a declared loadout uses.
  No local-model runtime remembers a past resident set at all.

- **Self-calibrating memory model** (`hearth calibration`, `GET
  /api/calibration`): Hearthia now reconciles its GGUF-header RAM estimates
  against the real measured RSS of each model once it has stayed warm long
  enough to settle (~45s), and folds the disagreement into a persisted,
  per-model EWMA correction factor (`~/.hearthia/calibration.json`). Once a
  model has two real measurements, every later estimate — the budget gate,
  `hearth est`/`hearth advise`, the dashboard's resident-size badge (🎯) —
  is corrected by what Hearthia has actually observed on this exact Mac,
  clamped to a safe `[0.6x, 1.8x]` range and never trusted from an
  implausible single reading. No other local-model runtime (Ollama, LM
  Studio, llama-swap) closes this loop between its own estimate and reality.

- **GGUF license & provenance inspector** (`hearth provenance <model>`,
  `hearth gguf` now prints a provenance section, `GET
  /api/models/{id}/provenance`): reads `general.license`,
  `general.base_model.*`, `general.source.*`, `general.quantized_by` and
  `general.tags` straight from the GGUF header — the same metadata a
  quantizer preserved from the source model card. No network access; only
  reports what is actually present in the file on disk. No local-model
  runtime surfaces this today.
- **Battery/thermal-aware RAM budget** (`hearth power`, `power.py`): the
  wired-memory ceiling `budget.py` enforces now flexes down under a
  genuinely constrained power state — battery below 20% (×0.7) or active
  thermal throttling reported by `pmset -g therm` (×0.85), stacked
  multiplicatively and floored at 4 GiB — so `hearth warm`/the dashboard's
  Warm button can refuse a warm they would otherwise allow. AC power with a
  low reported battery, or a nominal state, changes nothing. Best-effort:
  probe failures leave the ceiling untouched rather than guessing.
- **Model drift detector** (`drift.py`, wired into the calibration
  recorder): a `(size, mtime)` fingerprint per model catches a re-quantized
  or replaced GGUF the moment it warms again with a changed file underneath
  the same model id, and drops that model's now-stale RAM calibration
  instead of silently trusting samples that describe a file that no longer
  exists.
- **Predictive idle forecast** (`Telemetry.usage_forecast`, surfaced in
  `hearth status` and `/api/models`/`/api/status`): from the trend of gaps
  between a model's own recent requests, forecasts whether it is likely to
  see more activity before its TTL idles it out (🔮 in the dashboard) —
  no local-model runtime forecasts its own TTL behavior instead of just
  counting down a static timer.

- `hearth loadout sync` projects the authoritative `[loadouts]` configuration
  into readable `metadata.loadout` fields while preserving model roles and YAML
  comments.
- Model API responses and dashboard cards expose current loadout membership.
- `hearth est --json` and `hearth advise --json` emit machine-readable output
  for scripts, alongside the existing agent-facing text and MCP tools.
- Export the active chat conversation as Markdown (`Export .md` button in the
  chat sidebar) — purely client-side, since conversations only ever lived in
  `localStorage`.
- MCP now exposes `resources/list` and `resources/read` for daemon status,
  health, and bounded recent logs. All three resources support
  `resources/subscribe` with `notifications/resources/updated` over stdio.

### Changed

- Cooling a loadout preserves models declared in another loadout and reports the
  shared membership instead of freeing a model another working set still needs.

### Fixed

- `hearth brain search` always silently ran the O(n) pure-Python cosine
  fallback instead of the sqlite-vec KNN index: vec0 rejects a bound `LIMIT ?`
  parameter on `MATCH` queries, so every search raised `OperationalError` and
  fell through. Fixed the query to use vec0's `k = ?` constraint and removed
  the now-unused fallback (it also masked schema drift — a stale/missing
  index now raises a clear error pointing at `hearth brain reindex`).
- Chat: the conversation list was permanently `display:none` under 760px
  with no way to reopen it (`#conv-new` was unreachable too, since it lived
  inside the hidden sidebar). It's now a toggleable drawer (`#conv-toggle`).

## 0.4.0 — 2026-08-31

### Added
- **TreePact facade** (`hearth treepact doctor|validate|run`): a human-operated,
  version-pinned subprocess bridge to TreePact's independent worktree, gate and
  evidence engine. TreePact mutations are intentionally not exposed through
  Hearthia MCP.
- Integrated TreePact runs require a named Hearthia loadout and warm it through
  the GGUF-derived memory gate before the governed run starts.
- Bounded TreePact review commands (`status`, `diff`, `evidence`, `verify`)
  preserve the independent CLI's output and exit codes without opening its
  database or copying artifacts into Hearthia.
- **Read-only TreePact dashboard panel and API** (`GET /api/treepact/runs`,
  `GET /api/treepact/runs/{run_id}`): backed by TreePact's own strict,
  versioned `review` contract via a version-pinned, environment-minimized,
  timeout-bounded subprocess. Excludes task text, paths, artifact content,
  prompts, provider payloads, logs and diffs; renders everything as text.
  Absent from `hearth demo`. Each subprocess runs off the event loop
  (`asyncio.to_thread`) with at most two in flight at once, so it cannot
  stall the rest of the daemon or be used to spawn unbounded processes.

## 0.3.2 — 2026-08-28

### Changed
- **Everything under one roof.** The standalone `ggufram` extraction is
  folded back into Hearthia: the GGUF header reader and the KV-cache /
  resident-RAM arithmetic live in `hearthia.gguf` / `hearthia.library`
  again, with no external dependency.

### Added
- **`hearth gguf <file>`**: header-only cost report for any GGUF on disk —
  architecture geometry, KV cost per 1K tokens, resident estimate at a
  chosen context and cache type — no config, no gateway, no model data
  touched.

## 0.3.1 — 2026-08-28

### Changed
- **Extracted the memory arithmetic into [ggufram](https://github.com/JesusMonjeGonzalez/ggufram)**,
  a standalone, dependency-free package (GGUF header reader, KV-cache
  bytes, resident-RAM estimate, set-fits check). Hearthia now depends on
  it (pinned to `v0.1.0`); `hearthia.gguf` and `hearthia.library`
  re-export the same symbols, so behavior and the public surface are
  unchanged — every other tool can now use the arithmetic the budget
  gate enforces.

## 0.3.0 — 2026-08-28

The agent-interaction release: Hearthia becomes tooling that AI agents can
use themselves, plus smarter memory planning for humans.

### Added
- **MCP server** (`hearth mcp`): speaks the Model Context Protocol over
  stdio with the standard library only — zero new dependencies. Agents
  (OpenCode, Zed, Claude Desktop/Code) get eight budget-enforced tools:
  status, models, warm, cool, est, advise, loadout and Brain search. A
  refused warm returns the arithmetic plus fit options, so an agent adapts
  instead of failing. Setup for every client in `docs/MCP.md`.
- **`hearth advise`**: when a set of models does not fit, enumerate the
  uniform change-sets that make it fit — KV-cache quantisation (keeps
  context), lower context (keeps precision), cooling a running model —
  ranked and printed with the GGUF-header arithmetic. Nothing loads.
- **Named loadouts**: `[loadouts.<name>]` in config.toml defines a set;
  `hearth loadout list|show|load|cool` warms and cools it as one unit.
  Loading runs a whole-set budget check first, then per-model
  budget-checked warms in order; already-warm members are skipped.
- **`GET /api/health`**: aggregate probe (gateway up, event watcher
  connected, no crash loop) for monitors and scripts.
- **Configurable Brain filing**: `[brain].folders` (with the first as the
  fallback inbox) and `[brain].prompt_path` replace the hardcoded vault
  layout; the JSON schema for the filing model is built from the folders.
- `hearth est` now points at `hearth advise` when the verdict is DOES NOT
  FIT.

## 0.2.2 — 2026-08-27

### Added
- New brand mark: hearth arch + self-tending flame, in SVG and PNG
  (social preview) renditions; matching favicon and dashboard masthead.
- Qwen3.8-27B headlines the demo and the README examples with real
  measured numbers from a 36 GB Mac.

## 0.2.1 — 2026-08-27

### Added
- **`hearth est`**: what-if loadout planning — computes each model's
  resident estimate (GGUF-header maths, optional `--ctx` override) and
  delivers a FITS / DOES NOT FIT verdict against the wired ceiling and
  available RAM, loading nothing.
- Dashboard memory map: kindling models render with their GGUF-header
  resident estimate instead of a placeholder segment — the map is truthful
  during load, not only after.
- Brain reindex and code-index build embed in concurrent batches
  (3 in flight, `embed_batches`), keeping the GPU fed on large vaults.
- Animated demo GIF in the README, regenerable with
  `scripts/capture_demo.py` (Playwright + Pillow against the live demo).

## 0.2.0 — 2026-08-27

The memory-protection release: the RAM budget is now enforced, not advisory —
plus a zero-setup demo and one-command adoption of models you already have.

### Added
- **Model adoption** (`hearth adopt-ollama`, `hearth scan`): bring GGUF
  weights already on disk — Ollama blobs (parsed from manifests), LM Studio
  folders or any directory — into `llama-swap.yaml without re-downloading.
  Listing includes each model's real resident-RAM cost from its GGUF header.
- **`hearth status` upgrade**: per-model resident RAM, tok/s and live TTL
  countdown, plus a budget line (committed vs wired ceiling) sourced from
  the daemon.
- **RAM budget gate** (`hearth warm`, dashboard, lifecycle engine): every warm
  is checked against the GPU-wired ceiling before it loads. Estimates come
  from the GGUF header (layers, KV-head geometry, head dimensions, context),
  not file size; co-resident models use measured RSS where available.
  Blocked warms print the full arithmetic; `--force` overrides, and
  `[memory] mode = enforce | warn | off` configures the policy.
- **GGUF header reader** (`gguf.py`): pure-stdlib metadata parser that skips
  arrays with seeks, so planning costs kilobytes regardless of model size.
- **`hearth demo`**: a fully synthetic stack (sparse GGUF shells with real
  headers, canned streaming chat, live-looking logs and telemetry) served by
  the real dashboard — evaluation needs no llama.cpp, models or downloads.
  DEMO badge in the dashboard chrome; `memory.mode = warn` in demo so it
  never refuses its audience.
- **Resident-RAM estimates in the dashboard**: model cards show
  `est. N GiB resident` derived from the header.
- **Benchmark script** (`scripts/benchmark.py`): measured-vs-estimated RSS,
  KV cost per 1K tokens per model, co-resident total vs wired ceiling.
- **Packaging**: Homebrew formula (`packaging/hearthia.rb`) and a
  `curl | sh` installer (`packaging/install.sh`).
- **Release automation**: tagged `v*` releases run the full suite and attach
  built distributions to the GitHub release.
- **Community layer**: CONTRIBUTING guide, bug/feature issue templates, PR
  checklist, and `docs/RECIPES.md` with editor/Obsidian integration recipes
  and runtime-migration paths.
- **Daemon logging**: hearthd writes tagged logs to `hearthd.log` in the logs
  dir, logs startup/shutdown, foreign-origin rejections and swallowed
  poller exceptions.
- README: budget-gate documentation with real incident numbers, memory
  budget chart, demo-first quick start.

### Notes
- `hearth warm` now queries the gateway's `/running` before loading —
  budget math needs the co-resident set.
- `Registry.Model` carries the raw `cmd` so cache quantisation flags feed
  the KV estimate.
- Demo models' GGUF shells are sparse files with valid headers: the demo
  exercises the real RAM planner end to end.

## 0.1.0 — 2026-07-11

First working release: full port of the ad-hoc `~/llm-stack` dashboard into an
installable package, per the approved design spec.

### Added
- `hearth` CLI: status, models, warm/cool, pull (`--add`, resume, SHA-256
  verification), logs `-f`, daemon, install/uninstall/up/down/restart,
  doctor, migrate, brain capture/search/reindex.
- `hearthd` daemon on :9300: models/status/chat/config/logs/brain/library
  API + packaged web dashboard (ES modules, no build step).
- Lifecycle engine: TTL auto-unload, follow rules (`app:` and `role:` with a
  sensible chat fallback), crash-loop detection with macOS notification and
  dashboard banner.
- Model library: HF search, verified resumable downloads, fit check,
  one-click **Add to config** (generated ruamel block, roles metadata).
- Brain: sqlite-vec index, incremental reindex, frontmatter-stripped chunks,
  true-cosine scores; `brain` shim delegates to `hearth brain capture`.
- Dashboard: warm-soot Hearthia identity, ambient hearth glow, temperature-
  semantic actions, TTL countdown rings, failure banners, live logs.
- `hearth migrate` adopts an existing `~/llm-stack` in place.

### Notes
- Regression guards: TTL-poisoning (never poll `/upstream/...`), hermetic
  test settings, SSE reconnect, vec_chunks hygiene.
- E2E smoke: `uvx --from playwright python tests/e2e/smoke.py`.
