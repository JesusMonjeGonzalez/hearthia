# Changelog

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
