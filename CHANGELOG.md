# Changelog

## Unreleased

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
