# Security Policy

## Scope

Hearthia is a local single-user control plane for llama.cpp and llama-swap on
Apple Silicon. Model files, gateway configuration, logs and an optional Brain
vault can contain sensitive information.

The repository must not contain model weights, local configurations, vault
indexes, logs or credentials.

## Boundary

- The daemon is loopback-only and rejects non-loopback bind addresses.
- It has no authentication and must not be exposed as a remote or multi-user service.
- Context tools can read files available to the local process; use a dedicated local user if needed.
- Model downloads require a file published by Hugging Face with a verifiable SHA-256.
- Configuration replacement is atomic and keeps a local backup.
- The MCP server (`hearth mcp`) has no network listener: it speaks stdio with
  the client that launched it and acts with that user's permissions — the same
  single-user boundary as the CLI. Warm tools enforce the RAM budget gate;
  there are no file-write tools.
- Brain filing constrains the model-chosen folder to the configured
  `[brain].folders` list and sanitises the title, so model output cannot
  traverse paths when a note is written.
- `hearth treepact` delegates only `doctor`, `validate` and human-authorized
  `run` operations to a separately installed, version-pinned TreePact CLI. It
  uses a fixed argument vector without a shell and does not ingest TreePact
  tasks, diffs or evidence.
- Integrated TreePact runs require a named Hearthia loadout. Its declared model
  set is checked and warmed under the unified-memory budget before TreePact is
  invoked; a missing or refused loadout fails closed. The operator must keep
  that declaration aligned with TreePact's selected provider profile.
- TreePact run, resume and cleanup operations are not exposed through MCP. An
  accepted TreePact decision is evidence of Pact conformance, not proof of
  correctness or authorization to merge or publish.
- TreePact CLI review commands (`status`, `diff`, `evidence`, `verify`) are
  transparent subprocess views. Hearthia does not parse their output, open
  TreePact SQLite or offer export paths that would duplicate restricted
  artifacts.
- The dashboard's TreePact tab and `GET /api/treepact/runs[/{run_id}]` are
  read-only and use only TreePact's dedicated `review` contract: a
  version-pinned subprocess with an allowlisted environment (no inherited
  secrets or agent sockets), a five-second timeout, a one-megabyte output
  ceiling, and validation of `schema`/`schema_version` before use. On
  TreePact's side, `review` opens SQLite `mode=ro` with `query_only` and never
  migrates, discovers runs, recalculates gates, generates a bundle or writes
  to disk. The response is discarded after rendering; nothing is cached to a
  second store. These routes are absent from `hearth demo`.
- Each `treepact review` subprocess runs in a worker thread
  (`asyncio.to_thread`), never on the daemon's event loop, and at most two run
  concurrently (a shared semaphore). Without this, a slow or hung TreePact
  process would stall every other route — chat, warm/cool, logs — and an
  unbounded burst of dashboard polls could spawn unlimited `treepact`
  processes.
- The TreePact projection excludes task text, repository and worktree paths,
  artifact content, prompts, provider payloads, logs and diffs; the dashboard
  renders every value with `textContent`, never as HTML or Markdown.

## Reporting

Do not open a public issue containing model URLs with credentials, local paths,
logs or vault content. Use a private GitHub security advisory or contact the
repository owner through GitHub with redacted details.

## Release rule

Real model loading, restart recovery, memory-pressure behavior and browser
smoke tests remain release evidence gates. Unit tests alone do not establish
safe operation under real model workloads.
