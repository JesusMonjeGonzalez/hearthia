# Hearthia — Roadmap

What remains to reach the approved v1 design (spec:
`~/llm-stack/docs/superpowers/specs/2026-07-11-hearthia-design.md`), plus
improvements found while building. Publishing-related work (CI, releases,
Homebrew tap) is tracked separately and excluded here.

Updated: 2026-07-11 (evening). **Done since the morning list:** P1 items 1, 2,
3, 5, 6 and P2 items 8, 9, 10, plus persistent chat sampling from P3. Later: download resume (P1.4) shipped with hearth pull CDN-redirect fix; E2E smoke script, README and CHANGELOG shipped. Still open: configurable brain filing (P2.7), pull progress, status TTL/tok_s, numpy fallback decision, parallel embed batches.

## P1 — spec features still missing

### 1. "Add to config" after a download (spec §7)
`library.model_block_template()` exists but nothing calls it. After `hearth
pull` or a dashboard download, the user still hand-edits llama-swap.yaml.
- API: `POST /api/models/add` → registry inserts a generated block
  (ruamel, comments preserved), returns `restart_required`.
- UI: an "Add to config" button on downloaded files in the Library tab,
  with a small form (id, name, ctx, ttl, roles) pre-filled from the filename.
- CLI: `hearth pull --add` does the same non-interactively.
- Write `metadata.roles` in the generated block so lifecycle rules match
  without the fallback heuristic.

### 2. `hearth logs [-f]` (spec §5)
The only spec'd CLI command not implemented. Stream
`Gateway.logs_stream()` to stdout; without `-f`, print a recent window and
exit (llama-swap's `/logs` endpoint serves history).

### 3. Retire the old `brain` shim (spec §3)
`~/llm-stack/bin/brain` on PATH is still the pre-Hearthia standalone script
with duplicated capture logic (own prompt, own filing rules — it will
drift). Replace with a two-line shim that execs `hearth brain capture "$@"`.

### 4. Download resume support (spec §7)
`download_file` restarts from zero. Send a `Range` header when a `.tmp`
exists, hash the existing prefix into the running SHA-256, and keep the
atomic-rename guarantee. The dashboard Cancel button currently deletes the
tmp — keep it for "Dismiss", add "Pause".

### 5. Dashboard banners for failures (spec §6, §12)
Crash-loop detection notifies via macOS only; the spec also wants a
dashboard banner, and §12 wants every subsystem failure surfaced with the
failing subsystem named. One banner component fed by a new
`GET /api/health` (gateway up? event watcher connected? last crash-loop?)
covers: gateway down, SSE disconnected, crash loop, config write failures.

### 6. Structured logging in hearthd (spec §12)
The daemon logs nothing; broad `except` blocks stay silent (event watcher,
metrics poller, lifecycle tick). Add a module-tagged `logging` setup in
`daemon.create_app`, log swallowed exceptions as warnings, write to
`logs_dir/hearthd.log` alongside uvicorn's output.

## P2 — spec refinements

### 7. Configurable brain filing (spec §8)
`capture.FOLDERS` and the filing prompt are hardcoded to one vault layout.
Move to `[brain]` settings: `folders = [...]`, `prompt_path = "..."`
(optional override), keeping current values as defaults.

### 8. Line-anchored config validation (spec §9)
`PUT /api/config` returns ruamel's message which embeds line/column, but
the editor doesn't use it. Parse `line N column M` out of the 400 body and
scroll/highlight that line in `#cfg-editor` (a plain textarea can still
`setSelectionRange` + scroll).

### 9. Vault name hardcoded in Brain links
`brain.js` builds `obsidian://open?vault=Brain&…`. Derive the vault name
from `/api/brain/status` (basename of the configured vault path) — the spec
explicitly says no `~/Brain` assumption.

### 10. Allow *adding* ttl from the settings editor
`Registry.set_ttl` raises if the model has no `ttl` key, so the dashboard
can't give a TTL to a lifecycle-managed model. Insert the key instead
(ruamel handles it); the UI already sends the field.

## P3 — polish and hardening

- **Chat**: persist sampling settings (temperature/top_p/max_tokens) and
  the selected model in localStorage; a way to open the conversation list
  on narrow viewports (it's `display:none` under 760px); export
  conversation as Markdown.
- **`hearth pull` progress**: stream a progress line (bytes/total, MB/s)
  from the download loop; today it prints "pulling…" and blocks silently
  for a 20 GB file.
- **`hearth status`**: show per-model TTL countdown and tok/s (the daemon
  already tracks both; the CLI shows neither).
- **Brain fallback path**: the non-sqlite-vec cosine fallback is a
  pure-Python loop over every chunk; batch it with numpy (spec §8) or
  drop the fallback and make sqlite-vec a hard dependency (it already is
  in pyproject) — decide and delete dead code.
- **Parallel embedding batches** during reindex (spec §8): gather 2–3
  batch requests concurrently; llama.cpp batches embeddings well.
- **Playwright smoke script** under `tests/e2e/` (not in pytest): boot the
  daemon against a mock gateway, click through the six tabs, assert no
  console errors. This caught the dead Chat tab — worth keeping runnable.
- **CHANGELOG.md** started at 0.1.0 so changes accumulate from now.
- **README**: replace the stub with real install/usage docs (second-machine
  story: `brew install llama.cpp llama-swap`, `uv tool install hearthia`,
  `hearth install && hearth doctor`), a screenshot, and the editor/Obsidian
  integration recipes currently only in `~/llm-stack/README.md`.

## Explicitly out (per spec §13 / current decision)

Multi-user/auth · Linux/Windows · v2 single binary · auto-configuring
Continue/OpenCode/Obsidian · MLX backend · anything requiring GitHub
(Actions CI, releases, Homebrew tap).
