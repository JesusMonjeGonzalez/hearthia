# Changelog

## Unreleased

### Changed

- **Packing is 33× faster** (120 ms → 3.7 ms on a 481-message conversation):
  message sizes are cached and the total maintained incrementally, the digest
  is built without re-encoding its lines on every drop, and the question
  boundaries are found by scanning the front instead of the whole list. The
  reported byte count stays exact (asserted for plain, trimmed and
  summary paths). Fixed a latent cache bug this exposed: a freshly built
  digest could inherit the size of a freed one via a recycled `id()`.
- **GGUF headers are read 11× faster and cached** (58 ms → 5.2 ms cold,
  ~0 ms warm): `read_metadata` can materialise only the keys a caller needs,
  string arrays (tokenizer vocabularies) are skipped in bulk instead of one
  seek per element, and profiles are cached per (path, size, mtime). Reading
  every model in the stack went from ~1.1 s to ~50 ms. Fixed `_skip_value`
  missing its scalar branch — dead code until the targeted path used it, and
  it desynchronised the stream.
- **Admission is memoised for the turn** (10 s TTL): quick tool rounds no
  longer pay a gateway inventory round-trip each; a long tool phase still
  re-checks.
- **The dashboard stops polling on a hidden tab** and refreshes on return;
  the vitals strip shows the daemon version and live jobs/hooks counts, and
  the chat footer lists the shortcuts.

### Added

- **`scripts/install.sh`**: dependency check (`--check`), Homebrew/uv install,
  launchd services and the first `hearth chat` in one idempotent command.

## 0.6.0 — 2026-09-28

### Added

- **Job-log retention:** job logs are pruned at startup and whenever a job
  starts — the newest 20 and anything younger than
  `[agent] job_log_retention_days` (default 7, 0 disables) survive; unbounded
  disk growth is not a feature.
- **`hearth up` is honest when a service already runs** (`up … (already
  running)` instead of silently swallowing the bootstrap error).

### Fixed

- A refused model load keeps its reason **readable in the model card**
  (reserve, resident model, ceiling) instead of a vanishing `alert()` with the
  raw `{"detail": …}`; API error bodies are parsed into the sentence a human
  should read (pure `errorMessage` helper, unit-tested).

- `hearth advise` now resolves model aliases (like warm/tune) and reports
  unknown ids instead of silently planning around them.
- `hearth advise` can no longer contradict the residency policy: the same
  one-large-model rule the warm gate enforces is checked against every
  change-set, with an explicit note (`the RAM budget may allow this set, but
  [memory].max_large_models=1 does…`) in both human and `--json` output.

- `turn_end` hooks no longer fire for requests that never became a turn
  (client disconnect before consuming the stream, refused turns).

### Changed

- Develop-mode instructions now carry a short workflow line — plan first,
  mark steps done, delegate exploratory reading to `task`, run long checks
  with `background=true` — so a local model picks the right tool instead of
  discovering it. It is part of the pinned system prefix, cached after the
  first request.

- Paging to older messages no longer yanks the transcript to the bottom: the
  log lands at the top of the page you asked for (verified in the browser).

- **Verification-aware auto-continue:** when the previous turn edited files
  without running a check, the automatic continuation asks to run the pending
  verification before the next plan step instead of blindly moving on.
- **`hearth chats delete <id>`** and **`hearth status --json`** (the daemon's
  full snapshot, or a local memory-only fallback when it is down).

- **`hearth chats`**: list, full-text search and export conversations from the
  terminal (markdown or `--json`), reading the SQLite store directly — no
  daemon, no model tokens. `sessions` was taken by the loadout history.
- The rolling compaction summary now waits for the chat to fall idle
  (bounded) instead of racing a user turn: with `--parallel 1` it used to sit
  in front of your next round.

### Fixed

- Test isolation: the autouse fixture now *writes* a sandbox config, so a
  default `stack_dir` can never resolve to `~/.hearthia` and collect test data
  (a new test did exactly that; the rows were removed and the guard added).

- **Friendlier failure surfaces:** a gateway that is down now says
  `gateway is not answering at … — start it with 'hearth up gateway'` instead
  of a bare `Errno 61`, timeouts and HTTP errors get their own sentence, and
  the text is recorded in the transcript. `GET /api/status` reports
  `jobs_running` and `hooks_configured`.
- **Chat shortcuts:** `Esc` stops a running turn and `Cmd/Ctrl+K` starts a new
  conversation (never while a turn is streaming).

- **Opt-in model compaction (`[agent] compaction = "model"`)**: dropped turns
  get a rolling 200-word summary produced by the model in a background task
  (input ≤10 KB, output ≤400 tokens), merged with the previous summary and
  stored in the conversation metadata; the deterministic digest remains the
  fallback and still covers the newest drops. Packing prefers the model
  summary when present, bounded at 2.6 KB alongside a tail of recent stubs.

- **Per-conversation usage panel** in the chat: turns, last/peak input vs the
  allowance, prompt/output totals, growth per turn, projected turns until
  trimming, and the last round's prefill/cache/tok-s (pure `usageSummary`
  module, unit-tested; zero model tokens). Markdown exports now name the
  model (and its tok/s) for every assistant message, and `hearth doctor`
  reports recorded background jobs from the state file.

- **Long-run token economy:** `update_plan` echoes the step list only when it
  changes (status-only updates return a compact ack), and tool results carry a
  short `pacing` note during the last three rounds of a turn so the model
  wraps up instead of being cut off. Both measured against a real autonomous
  run.

- **Hooks (`[[agent.hooks]]`)**: fire-and-forget commands on `turn_end` and
  `edit` with a JSON payload on stdin. Never delay a turn, never touch the
  conversation, bounded (4 concurrent, timeout+process-group kill, 20-run
  ring), failing hooks recorded and ignored, unknown event names rejected at
  load. Visible via `GET /api/hooks` and `hearth doctor`; hot-reloadable.

- **Human visibility for background jobs:** `GET /api/jobs`,
  `GET /api/jobs/{id}` (bounded tail) and `POST /api/jobs/{id}/stop`; a
  `hearth jobs [list|status|stop]` CLI; and a chat line with a stop button per
  running job, polled only while something runs. Previously only the model
  could see or stop jobs.
- Search's LIKE fallback escapes `%` and `_`, so a query like `100%` is
  literal instead of matching everything.

- **Transcript transparency:** every assistant message records which model
  produced it (mixed-model sessions are otherwise indistinguishable) and the
  chat shows it with the per-turn tok/s; JSON exports carry it too. Tool
  results for `task`, `job`, `plan` and MCP now render as structured cards
  (subagent header as title, job state/exit/tail, plan checklist with
  progress) instead of generic text.
- **`hearth tune` suggests a subagent helper** when `[agent] subagent_model`
  is unset and a model fits within the helper cap — on a real sparse GGUF it
  reads, for example: `'qwen2.5-coder-1.5b' (2.1 GiB) fits as a helper…`.

- **Dedicated subagent model:** `[agent] subagent_model` (id or alias, e.g.
  the 1.5B helper) is used for `task` subagents when the RAM policy allows a
  helper alongside the one large model; otherwise the task falls back to the
  main model and the report says why. The report header names the model that
  actually ran.
- **`hearth chat --new`:** deep-links a fresh conversation instead of resuming
  the last one. Fixed the demo/daemon job-state write when no job directory
  exists yet.

- **Background jobs:** `run_command(background=true)` starts a detached
  command that logs to a file, plus a `job` tool (`list`/`status`/`wait` ≤120 s
  /`stop`) that only ever shows a bounded 2 KB tail. Bounded by concurrent-job
  count, a hard lifetime (state `timeout`), a capped log that is still drained,
  process-group kills, shutdown cleanup and startup reaping of a previous
  daemon's leftovers (pid reuse guarded by spawn time).
- **Hot config reload:** `POST /api/config/reload` re-reads config.toml,
  applies agent/memory/brain sections, rebuilds MCP when its servers changed,
  updates job limits, reports the diff, and leaves the running settings
  untouched when the file does not parse.

- **Configured per-edit checks:** `[[agent.checks]]` maps extensions to a
  command (with `{file}`), run after every successful edit under the same
  process-group/RSS/deadline limits as any command. A pass costs ~10 tokens; a
  failure adds a bounded (2 KB) `check` dict to the edit result so the model
  fixes it in the same round. Not an LSP, never blocks an edit, and automatic
  checks deliberately do not satisfy the verification gate.

- **Verification gate on `update_plan`:** marking steps done while the current
  turn edited files with no command afterwards now adds a structured
  `warning` to the tool result (default) or **refuses the update** with
  `[agent] require_verification = true`, leaving the plan untouched until a
  check runs. Reads-only and verified turns are unaffected; zero token cost
  unless there is a mismatch to report.

- **Subagents (`task` tool, Develop mode):** delegate read-only exploration to
  a nested loop with a disposable context (own small message list, rounds cap
  6/12, 40 KB context, 12 KB per tool result, 4,000-char report labelled with
  how it ended). Edits, nesting and MCP are refused by the executor, not just
  hidden from the schemas. Same model, sequential, token usage accounted to
  the turn. Raw dumps and test logs never enter the main prompt — the main
  window is what local prefill makes expensive. See `docs/SUBAGENTS.md`.

- **Long-run quality:** the compaction digest now keeps a 150-char conclusion
  stub per dropped assistant turn (findings survive, not just tool history);
  `update_plan` accepts a `done` list of finished steps, shown in the
  dashboard checklist (`n/m done`, ✔/○) and usable for **auto-continue** — an
  opt-in toggle that submits continuation turns while the plan has pending
  steps, capped at 8, stopped by any failure/Stop and reset by a manual
  message. The pinned plan block stays steps-only, so status flips never shift
  the cached prefix.

- **`hearth doctor` (zero model tokens, read-only)**: full-system health check
  with honest ok/warn/fail findings — config and YAML parsing, model files on
  disk, configured binary, llama-swap on PATH, models dir, gateway and daemon
  reachability + launchd state, live memory/swap/wired ceiling with a
  cold-start fit for the largest model, disk space, conversations DB integrity
  and search-mirror sync, ledger validity, configured MCP servers (with
  `--deep` to actually spawn them and list tools), and loadout drift warnings.
  `--json` for machines, exit code 1 on any failure. Replaces the old,
  narrower doctor and keeps its checks.

- **Re-read shortcut (token-free):** full-file reads whose byte-identical
  content is already in the prompt (sha256 match) are answered with a
  three-line note instead of resending the file. Range reads, changed files
  and files known only through an edit diff always read in full, so
  exact-match editing is never taken away. The probe is a bounded worker call
  and never an inference.

- **Full-text search across all conversations** (zero model tokens): an FTS5
  mirror kept in sync on append/checkpoint/delete with a one-time backfill,
  a LIKE fallback when FTS5 is unavailable, sanitised AND-ed queries,
  per-conversation grouping with highlighted snippets and a jump-to-message
  result list in the dashboard (`GET /api/conversations/search`).

- **Cheap syntax feedback on edits:** `.py`/`.json`/`.toml` files are parsed
  locally after every edit; failures add a compact `syntax_error` note to the
  tool result (no inference, no tokens when clean) so the model fixes a broken
  file in the same round. The edit is never blocked and the dashboard marks
  the card as failed. Plans are now shown as a collapsible checklist in the
  chat, reading the stored metadata at zero token cost.

- **Retry and fork.** `POST /api/conversations/{id}/retry` forks without the
  last user turn and returns the message for re-submission; `POST …/fork`
  copies any prefix into an independent conversation with `forked_from`
  provenance. Both refuse while the source is running; the original is never
  rewritten. Dashboard buttons included.
- **Verification marker.** Each turn records edited paths and whether a
  command ran after the last edit; the chat shows an amber "Unverified" line
  when an edit ended the turn without a check. Stored in conversation
  metadata (`last_turn`) and documented as "command ran after edit" — not as
  proof the change was tested.

- **Long-session guards** (`docs/LONG-SESSIONS.md`): a keepalive ping while
  tools run so llama-swap's TTL cannot evict the model mid-turn (measured:
  countdown reset from 4m17s to 4m58s with pings), configurable
  `max_tool_rounds` (1..48) and `turn_budget_minutes` with a graceful forced
  final round, a persistent `update_plan` tool whose steps stay pinned across
  context trimming, and a no-progress guard that flags the third identical
  execution of a failing call.

- **Pinned project block:** the workspace snapshot (repo map, AGENTS.md) now
  sits at a stable position after the system message and is reused
  byte-for-byte while its signature is unchanged; a change rebuilds it once.
  Previously it trailed each turn, so ~1,600 tokens were re-prefilled per
  turn. Memory is bounded to 64 sessions, oldest evicted.
- **Per-element prompt accounting** (`element_tokens`, `tools_schema_tokens`)
  in the context event and the chat tooltip, plus the `hearth tune` advisor:
  spec-decode acceptance, KV size vs `--cache-ram`, prompt-cache reuse,
  observed peak context and calibration, all read-only with honest
  "no data yet" answers.

- **Harness compaction:** dropped turns leave a bounded deterministic digest
  (question stubs plus tool outcomes: command exit codes, edited paths, MCP
  calls) instead of vanishing, merges across packings, and is the last thing
  to give way. Drops now free room down to 85% of the budget (hysteresis) so
  the cached prefix is not shifted on the next turn.
- **Window usage ledger per conversation:** each turn records input/allowance,
  prompt and output tokens in the conversation metadata; `GET
  /api/conversations/{id}` returns occupancy, remaining tokens, observed
  growth per turn and projected turns until trimming. The chat shows it under
  the activity line.
- Token-aware room for appended tool results (same ratio as the window budget).

- **MCP client:** `[mcp.servers.*]` stdio servers are discovered at turn start
  and exposed to the chat as `mcp__<server>__<tool>` (64-char wire names with
  stable hash suffixes). Consult mode only sees servers marked
  `read_only = true`; the executor refuses mutating servers there even if the
  model calls them. Per-request deadlines restart a stuck server, crashes are
  reported with stderr context and recovered on the next call, and shutdown
  reaps every process group. See `docs/MCP-CLIENT.md`.
- Independent read-only tools in one round now run **concurrently** (bounded to
  4, results kept in call order); mutating tools stay strictly sequential and
  are never deduplicated.

- **Residency policy** (`[memory]` in config.toml): one large model resident at
  a time, a size cap for small helpers (embeddings/autocomplete), a non-wired
  OS/apps reserve checked after every load, swap-in-use warnings, and
  fail-closed refusal when the gateway inventory is unknown. An already-warm
  candidate is never charged twice and is always allowed. Wired into the CLI,
  daemon lifecycle followers, loadouts, MCP warm tool, rehearsal and chat.
- `hearth warm` and the dashboard now print/return the policy line, measured
  headroom and swap figure with each decision; `GET /api/status` exposes
  `memory_policy`, `swap_total` and the `sampled_at` instant; the vitals strip
  shows swap used/total, data age and the active policy.
- Measured baseline for the production `qwen3.8-27b-rvn` at 64K: 19.33 GiB
  resident, 8.67 GiB under the 28 GiB wired ceiling, 16.67 GiB left for
  macOS/apps (6 GiB reserve), swap 0 — see `docs/MEMORY-POLICY.md`.

- **Cache-friendly prompt structure:** project context (repo map, AGENTS.md,
  workspace note) moved from the system message to the newest user turn;
  context is packed once per turn and every later round is append-only; the
  turn never rewrites already-sent messages (oversized new tool results are
  truncated with an explicit note, and a spilled turn runs one final tool-less
  round before failing honestly). llama.cpp `timings` (`prompt_n`, `cache_n`,
  tok/s) are stored per assistant message and shown in the chat status line.
- **Machine-readable export:** `GET /api/conversations/{id}/export?format=json`
  returns every message, tool call, timing and error for other tools.
- `scripts/bench-prompt-cache.py` measures real prompt-cache reuse of a warm
  model through the gateway (refuses to load anything itself).

### Fixed

- A background job whose log hit the size cap lost the *end* of its output —
  exactly where errors are. A bounded 8 KB in-memory tail now survives the cap
  and `job(status)` prefers it when the log was truncated.
- An MCP server that was down when the daemon started was never retried until
  a restart. Discovery now re-attempts failed servers with a 60 s cooldown, so
  a hanging server cannot cost every turn its timeout, and recovery is automatic.
- An oversized stored compaction summary could exceed the digest budget; it is
  truncated at use and stored capped at 2.4 KB.

- The `job` tool now requires Develop mode like every other background-capable
  tool: a model in Consult mode could previously stop jobs it could not start.
- `GET /api/status` reports the daemon's code version and `hearth doctor`
  compares it with the installed package, flagging a daemon left running stale
  code after an update (`hearth restart daemon`) — found the hard way while
  developing, now a one-line warning.

- **Command PATH under launchd:** spawned commands prepend the workspace's
  `.venv/bin`, `~/.local/bin`, `/opt/homebrew/bin` and `/usr/local/bin`, so a
  plain `python` resolves as it does in your shell; a missing executable now
  reports `not found` with the searched PATH and exits 127 instead of dumping
  an exec traceback. Both found by a real autonomous run on this machine.

- Process-group cleanup tolerates `EPERM` as well as `EPLR` from `killpg`
  (zombie leaders and groups the OS refuses to signal) — cleanup must never
  crash — and the hook runner's timeout path no longer lets a `CancelledError`
  escape from a cancelled subprocess wait: the wait is shielded, the output
  drain is its own task, and `close()` kills running hook processes before
  cancelling their tasks. A transient spawn failure under load now gets one
  quiet retry. Verified with eight consecutive full-suite runs.

- Exact prompt measurement behind llama-swap: `/tokenize` and
  `/apply-template` are only exposed under `/upstream/<model>/…`, so the
  measurement now routes there (falling back to the byte estimate when the
  stack does not offer them at all). Verified against a live session.
- `scripts/bench-prompt-cache.py` cache-hit maths (divided by the delta
  instead of the total prompt) and a status test that only passed because the
  real gateway port happened to be closed.

- `plan_warm` costed an already-resident candidate twice (once as measured
  resident, once as new estimate), which could refuse a warm that needed no
  new allocation at all — chat mid-turn checks were the most exposed path.

- Explicit Develop mode with workspace-bound exact edits, atomic no-overwrite
  creation, persisted diffs and foreground executable/argument tools.
- Command exit codes, bounded head/tail output, CPU/wall limits, configurable
  sampled RSS allowance and process-group cleanup on Stop or completion.
- Read hashes, failed/passed command cards, and a real edit/fail/fix/test browser
  workflow using scripted model decisions and real subprocesses.
- Durable SQLite chat sessions with revision checks, streaming checkpoints,
  interrupted-turn recovery, paginated history and complete Markdown export.
- Explicit, idempotent import of legacy browser conversations without deleting
  the originals; the dashboard now keeps only one message page in memory.
- Workspace selection, root AGENTS.md context and line-range file reads.
- Disposable filesystem workers with CPU/wall deadlines, bounded output and
  process-group termination on cancellation.
- Model-aware conservative context packing, preserved tool-call/result pairs,
  visible omissions and tool shortening without rewriting stored transcripts.
- Structured tool activity and a real-browser test using an isolated demo daemon.

### Fixed

- Mutating tools invalidate read deduplication; repeat commands run again after
  fixes. Malformed duplicate tool IDs cannot execute a batch of edits.
- Browser-history imports retain their original IDs across the mode-schema
  upgrade and cannot implicitly grant Develop capabilities.
- Chat checks memory admission per inference round, limits concurrent turns,
  avoids implicit semantic indexing, and persists upstream stream failures.
- Streaming handles split Unicode and bounded buffers; attachment reads and
  transcript rendering are bounded, and Stop releases tool processes.
- Log streaming rejects gateway HTTP errors instead of displaying their response
  bodies as log lines; the dashboard shows the existing unavailable-gateway notice.

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
