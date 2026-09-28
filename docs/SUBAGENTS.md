# Subagents: isolated context for exploration

The main loop's window is the scarce resource. On this Mac every prompt token
is prefilled at ~145 tok/s, and every file dumped into the conversation costs
tokens *for the rest of the session* — the digest only bounds the loss. The
`task` tool delegates focused, read-only work to a **subagent with its own
disposable message list**: it reads, searches and runs commands, then returns
a bounded report. The raw material never enters the main prompt.

## What it can and cannot do

| Capability | Subagent |
|---|---|
| `read_file`, `read_files`, `search`, `list_dir`, `glob` | yes |
| `run_command` (same workspace, same RSS/wall limits) | yes |
| `edit_file`, `create_file` | **no** — the workspace must not change behind the main loop |
| `update_plan`, `task` (nesting) | **no** — depth is bounded by construction |
| MCP tools | not in v1 (no side effects from nested runs) |

Schemas inform the model; the executor enforces the allowlist again, so a
hallucinated `edit_file` call is refused and recorded, never executed.

## Hard bounds

- **Rounds**: `max_rounds` per call, default 6, hard cap 12, plus a forced
  final round without tools.
- **Its own context**: 40 KB of messages; each tool result is capped at 12 KB.
  When the budget fills, the subagent gets an explicit “answer now” turn and
  its report is labelled `budget` instead of `complete`.
- **Report**: 4,000 characters, prefixed with how it ended:
  `Subagent report (complete · 2 rounds · tools: read_file, run_command):`
- **Inference**: sequential on the same model (the one-large-model policy
  holds; nothing runs in parallel). Its `prompt_n`/`predicted_n` are added to
  the turn's token usage, so projections stay honest.

Cancellation propagates: Stop kills the parent turn, the nested call and any
worker processes it started.

## Use a small model for subagents (optional)

Subagent work is exploration: with `[agent] subagent_model` pointing at a
helper-sized model (the 1.5B coder, ~2 GiB, within `memory.helper_max_mib`),
exploration runs at a fraction of the cost of the 27B and the one-large-model
policy still holds — a helper may share the slot. Hearthia checks the RAM
policy before using it; if it would not fit (or the id is unknown), the task
falls back to the main model and the report says so:

```toml
[agent]
subagent_model = "qwen2.5-coder-1.5b"
```

The report header always names the model that actually ran:
`Subagent report (complete · 3 rounds · model qwen2.5-coder-1.5b · tools: …)`.

## Using it

The model decides. In Develop mode the `task` tool appears with:

```json
{"description": "inspect the payment module",
 "prompt": "Read src/payments/*.py and report where retries are configured, with file:line evidence.",
 "max_rounds": 6}
```

Good delegation: search-and-summarise over many files, running a test suite
and reporting which tests fail and why, reading a large log. Bad delegation:
anything needing edits, anything that must remember this conversation's full
history (the subagent only sees its prompt), or trivial one-file reads.

## Honest limits

- Same model, same speed: a subagent does not make inference faster; it makes
  the **main** context cheaper and cleaner. It pays its own prompt tokens.
- No MCP, no edits, no nesting in v1.
- The report is lossy by design: if the caller needs raw evidence, it must ask
  for specific paths and read them itself.
