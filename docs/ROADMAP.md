# Hearthia — Roadmap

What remains beyond the shipped 0.4.0 feature set. Publishing-related work
(CI, releases, Homebrew tap) is tracked separately and excluded here.

Updated: 2026-08-31. **Shipped in 0.4.0:**

- **TreePact facade.** Human-operated `hearth treepact` commands launch and
  inspect governed runs through a version-pinned subprocess. The dashboard and
  API expose only TreePact's bounded, read-only review contract.

**Unreleased:**

- Loadout membership is projected into registry metadata, exposed in the model
  API/dashboard, and shared models are preserved during loadout cooling.

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

### 1. Loadout lifecycle metadata
`[loadouts]` is config.toml-side only; llama-swap never sees it. Optional:
write `metadata.loadout = "coding"` into each member's block so the
dashboard could badge loadout membership, and let `hearth loadout cool`
release only members not shared with other active loadouts.

### 2. MCP: expose daemon SSE as resources
Today the server answers point-in-time status. Exposing the event stream
(and `logs/stream`) as MCP resources/subscriptions would let agents react to
crash loops instead of polling.

## Remaining P3 — polish and hardening

- **Chat**: the conversation list is `display:none` under 760px with no way
  to open it; export conversation as Markdown.
- **Brain fallback path**: the non-sqlite-vec cosine fallback is a
  pure-Python loop over every chunk; batch it with numpy (spec §8) or drop
  the fallback — sqlite-vec is already a hard dependency. Decide and delete.
- **`hearth est --json` / `advise --json`**: machine-readable output for
  scripts (MCP already returns text designed for agents).
- **Playwright smoke script** under `tests/e2e/` (not in pytest): boot the
  daemon against a mock gateway, click through the six tabs, assert no
  console errors. This caught the dead Chat tab — worth keeping runnable.

## Explicitly out (per spec §13 / current decision)

Multi-user/auth · Linux/Windows · v2 single binary · auto-configuring
Continue/OpenCode/Obsidian · MLX backend · anything requiring GitHub
(Actions CI, releases, Homebrew tap).
