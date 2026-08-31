# TreePact integration

Hearthia provides a narrow, human-operated facade over TreePact. Hearthia
manages local model availability and memory. TreePact remains the authority for
Pacts, worktrees, checks, gates, decisions and evidence.

## Install

Install TreePact separately so its dependencies, data and release lifecycle do
not become part of Hearthia:

```bash
uv tool install /path/to/TreePact
treepact --version
```

Hearthia currently requires TreePact `0.1.0`. If it is not on `PATH`, configure
the absolute executable path:

```toml
[treepact]
executable = "/absolute/path/to/treepact"
expected_version = "0.1.0"
loadout = "treepact-coding"

[loadouts.treepact-coding]
description = "Models used by TreePact provider profiles"
models = ["code-model"]
```

TreePact owns its provider configuration. Point it at Hearthia's loopback
gateway and use model IDs present in `llama-swap.yaml`:

```toml
[provider]
endpoint = "http://127.0.0.1:9292/v1"
profiles = { classify = "fast-model", fast-code = "code-model", deep-code = "deep-model" }
```

The operator must keep the loadout aligned with the model behind the Pact's
selected TreePact provider profile. Before an integrated run, Hearthia checks
the declared set and warms it through the GGUF-derived memory gate. A missing
loadout or a set that does not fit fails closed before TreePact creates a run.
Standalone `treepact` usage remains available without this Hearthia-specific
preflight.

Hearthia ships a conservative `.treepact.yaml`. Coding runs may change only
`src/` and `tests/`; the final decision requires the Python suite, Ruff and
mypy. Packaging, configuration, documentation and the Pact remain protected
inputs unless a human deliberately changes that contract.

## Use

```bash
hearth treepact doctor --repo ~/src/project --deep
hearth treepact validate --repo ~/src/project
hearth treepact run "Fix the failing unit test" \
  --repo ~/src/project --mode repair --max-attempts 2 --max-minutes 30
hearth treepact status RUN_ID
hearth treepact diff RUN_ID
hearth treepact evidence RUN_ID --verify-hashes
hearth treepact verify RUN_ID
```

The facade fixes the TreePact runtime to `native`, invokes an absolute
executable with a fixed argument list and never uses a shell. It preserves
TreePact's exit codes. Hearthia does not capture tasks, diffs, prompts or
evidence.

Hearthia delegates review commands directly to TreePact without parsing or
storing their output. It deliberately omits export paths, watch loops and
arbitrary subcommands. Recovery and state-changing operations continue through
TreePact itself:

```bash
treepact resume RUN_ID
treepact cancel RUN_ID
treepact cleanup RUN_ID
```

An `accepted` decision means the captured facts passed the repository's Pact.
It is not proof of correctness and does not authorize merge, commit, push or
publication.

## Dashboard and API (read-only)

Hearthia's dashboard has a **TreePact** tab, backed by two loopback `GET`
endpoints:

```text
GET /api/treepact/runs?limit=20
GET /api/treepact/runs/{run_id}
```

Both endpoints call TreePact's own strict, read-only contract —
`treepact review --schema-version 1` — through a version-pinned subprocess
with a minimized environment, a five-second timeout and a one-megabyte output
ceiling. TreePact opens its database with SQLite `mode=ro` and `query_only`;
`review` never migrates, discovers interrupted runs, recalculates gates,
generates a Decision Bundle, or writes to disk. The parsed document is
validated against `schema` and `schema_version` before Hearthia uses it, then
discarded — nothing is cached or persisted a second time.

The projection is deliberately minimized: run identity, project identity,
state, decision, reason code, assurance level, timestamps, stored gate
results and evidence metadata (bundle availability, event-chain head,
artifact ID/kind/digest/size/media type). It never includes task text,
repository or worktree paths, artifact content, prompts, provider payloads,
logs or diffs — those fields do not exist in the contract TreePact emits.

The dashboard panel can only list runs and show one run's detail. It has no
button to start, resume, cancel or clean up anything; those actions exist
solely in the two CLIs, under an explicit human invocation. The panel renders
every value with `textContent`, never as HTML or Markdown, because TreePact
output is outside Hearthia's own trust boundary. Both endpoints are `GET`
only, return `Cache-Control: no-store`, and are absent from `hearth demo` so a
demo session never touches a real user's runs.

## Security boundary

- Starting a run requires an explicit human CLI invocation.
- No TreePact mutation tool is exposed through Hearthia MCP or the dashboard.
- CLI review commands (`status`, `diff`, `evidence`, `verify`) inherit stdio
  and preserve TreePact exit codes; Hearthia does not parse, cache or
  reclassify their decisions.
- The dashboard and `/api/treepact/*` use only `treepact review`, TreePact's
  dedicated read-only contract, captured with a bounded timeout and output
  size, and validated before use.
- TreePact owns its SQLite database, artifacts, event chain and worktrees;
  Hearthia never opens that database directly.
- Checks still execute with the local user's authority; TreePact is not a
  sandbox.
- Models and repository contents remain untrusted.
- Interrupted runs are inspected or resumed explicitly; Hearthia never retries
  them automatically.
