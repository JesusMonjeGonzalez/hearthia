# Develop mode: read → edit → verify

Hearthia can now run a complete coding-tool loop from its persistent chat.
Choose **Chat → project directory → Develop**, select a model and describe the
task. **Consult** remains the default and exposes only the existing read/search
tools. Develop is an explicit per-turn capability selected by the client and
saved in the conversation; a model tool call cannot promote its own mode.

Develop requires a persistent conversation and an explicit workspace. It adds:

| Tool | Contract |
|---|---|
| `edit_file` | Replace one exact, unique `old_text` block with `new_text`; optional `expected_sha256` rejects stale reads |
| `create_file` | Publish a new UTF-8 file without overwriting an existing file; parent must exist |
| `run_command` | Run `argv` in the workspace or an existing subdirectory; no implicit shell expansion |

Read tools include a SHA-256 for files within the 512 KB editing limit. The
agent can supply this hash with its edit. Existing read/search tools continue
to execute in short-lived workers.

## Editing behavior

- File tools reject paths outside the workspace, symlink paths, `.git` metadata,
  non-regular/binary files, ambiguous replacement text and oversized results.
- An edit preserves text outside its exact match, original line endings and
  permission bits. The target is staged in its own directory, flushed and
  atomically replaced after checking the original bytes and inode again.
- Creating a file uses atomic no-overwrite publication, including when another
  process creates the target while the tool is preparing its content.
- These checks are optimistic concurrency protection, not a filesystem sandbox
  or a universal atomic compare-and-swap against hostile concurrent writers.
- Each successful tool result records a unified diff, before/after hashes, byte
  count and whether content changed. Diff display is limited to 20,000 characters
  and explicitly labelled when truncated. Files are limited to 512 KB.
- The transcript shows the diff with added/removed lines. No automatic commit,
  staging or rollback is performed. If a tool is interrupted around publication,
  inspect the file before retrying: cancellation does not undo applied changes.
  A hard-killed editing worker can leave a `.hearthia-edit-*` staging file.

## Command environment

Commands run with the daemon's environment plus a **prepended PATH**: the
workspace's `.venv/bin` (or `venv/bin`), `~/.local/bin`, `/opt/homebrew/bin`
and `/usr/local/bin`. The daemon runs under launchd with a minimal PATH, so
without this a plain `python` that works in your shell failed with an opaque
exec traceback (found in a real autonomous run). A missing executable now
reports `python: not found` with the searched PATH and exits with the shell
convention **127**, so the model adapts instead of parsing a traceback.

## Configured per-edit checks

After a successful edit, Hearthia can run the check command you configured for
that extension — the project's own linter or checker, not a guess:

```toml
[[agent.checks]]
extensions = ["py"]
command = ["/path/to/venv/bin/ruff", "check", "--no-cache", "--quiet", "--output-format", "concise", "{file}"]
timeout_seconds = 20

[[agent.checks]]
extensions = ["ts", "tsx"]
command = ["npx", "tsc", "--noEmit", "--pretty", "false"]
```

- `{file}` is the workspace-relative path, substituted literally: no shell.
- The command runs with the same process-group cleanup, RSS sampling and wall
  deadline as any foreground command (default 20 s, max 120 s).
- **A pass costs ~10 tokens** (`check: "<command> (ok)"`). **A failure** carries
  a bounded dict (`command`, `exit_code`, and up to 2,000 characters of
  output), so the model can fix it in the same round. This is the cheapest
  verification loop available: measured with ruff on a real file, a clean run
  returns one line and a dirty one returns the exact diagnostics.
- Edits are never blocked by a check; failures are information.
- Nothing runs unless you configure it — the command is your explicit trust
  decision, like MCP servers.

Not an LSP: no incremental diagnostics, no type inference, no editor protocol.
It runs the command your project already trusts, once per edited file.

**Relationship to the verification gate:** automatic checks do *not* count as
the model's verification. The gate (`require_verification`) still expects a
model-initiated command after edits, so a passing lint cannot silently satisfy
a "run the tests" step.

## Cheap local verification

Every `edit_file`/`create_file` result runs a local parse of the written file:
`ast` for `.py`, `json` for `.json`, `tomllib` for `.toml`. It costs no
inference and **no tokens unless it fails** — a broken file adds a compact
`syntax_error` plus a note to the tool result, so the model can fix it in the
same round instead of spending a whole extra round discovering it. The edit is
always applied as written and never blocked; the check is information, not a
gate. The dashboard flags the edit card as failed and shows the error.

Note what this is not: it is a parse check, not a compiler, type checker or
test. Passing it says nothing about whether the code is correct.

## Foreground commands and resource controls

Example tool arguments:

```json
{"argv": [".venv/bin/python", "-m", "pytest", "-q"], "timeout_seconds": 60}
```

The command must be appropriate to the selected project; the harness tells the
agent to derive checks from its instructions and manifests. A string such as
`$(...)` is passed literally in an argument. Shell features require explicitly
invoking a shell in `argv`.

- stdin is closed to interactive input.
- stdout/stderr share a bounded 32 KiB buffer retaining the beginning and end.
  Remaining output is drained rather than stored; total byte count and truncation
  are reported. Output never becomes executable HTML in the dashboard.
- Default deadline: 60 seconds; maximum per call: 120 seconds. Stop and deadlines
  kill the foreground process group. Cleanup also kills ordinary background
  descendants remaining in that group after the leader exits.
- The exec wrapper lowers process priority, limits CPU time to 120/121 seconds
  and caps open descriptors at 512 (or the inherited lower hard limit).
- Common BLAS/OpenMP thread environment variables are set to 2. This is a hint
  for those libraries, not a universal two-thread limit.
- Parent/descendant RSS is sampled every 50 ms while the leader runs. Exceeding
  its allowance stops the group and returns `limit: "memory_limit"`. Sampling
  can miss brief peaks, double-count shared pages or miss detached descendants;
  it is not an OS-enforced memory ceiling and does not measure GPU memory.

Configure the RSS allowance in Hearthia's `config.toml`:

```toml
[agent]
command_memory_mib = 1024  # default; accepted range 128..8192
```

The environment override is `HEARTHIA_AGENT__COMMAND_MEMORY_MIB`. The model
cannot increase this allowance through tool arguments.

**Commands run with the local user's permissions.** A working directory is not
a sandbox: commands can access other paths, use the network and run project
scripts. The workspace boundary applies to Hearthia's file-editing tools, not
to arbitrary programs. Jobs which detach into other sessions/daemon processes
are not supported by foreground process-group cleanup. No background service
manager is implemented.

## Harness and transcript

Command results contain `exit_code`, duration, bounded output, output-byte count,
limit status and sampled peak RSS. The dashboard distinguishes passed/failed
commands and keeps their output expandable after reload. Exit code zero means
the program exited successfully; it does not independently prove task quality.

Commands and edits are never deduplicated as if they were immutable reads.
Invoking a mutating tool clears the turn's read-dedup cache, so re-reading a file
or running the same tests after a fix observes the new state. Invalid/missing or
duplicate tool-call IDs are rejected before executing any tool in the batch.

Interrupted calls remain in the durable transcript, with an explicit note that
effects may have occurred. Recovery does not replay mutations. Importing legacy
chats keeps their pre-mode identity and never silently grants Develop mode.

The API uses `mode: "build"` with `session_id`, current `revision`, `workspace`,
model/system settings and one new user message. Omitting mode selects `read`.
The tool executor also enforces this mode, independently of which schemas the
model was shown.

## Evidence and next gaps

The automated fixture exercises real tools on a temporary Python project:

1. Read a faulty addition function.
2. Run its real unittest suite and observe failure.
3. Replace the faulty expression while preserving an unrelated user comment.
4. Read the same file again and observe the changed source.
5. Run the identical command again and observe success.
6. Reload the browser and review the persisted diff and both command results.

The model decisions in this fixture are scripted; edits, subprocesses, tests,
SQLite persistence and browser interactions are real. A separate browser path
starts a long-running command, clicks Stop and verifies its PID no longer exists.
Unit tests cover ordinary descendant cleanup, deadlines, sampled memory refusal,
literal arguments, output truncation, stale/ambiguous edits and publication races.

Observed verification: **624 Python tests**, **14 JavaScript tests**, static
checks and the Chrome end-to-end workflow passed. No real-model coding-quality
benchmark or long-running GPU/RSS comparison with Pi, Crush or OpenCode was run.

For long/autonomous runs see [`LONG-SESSIONS.md`](LONG-SESSIONS.md): keepalive against
TTL eviction, budgets, pinned plans and the no-progress guard.

Remaining major blocks: LSP/hooks/permissions, provider/tool compatibility across real
models, durable plans/compaction for long tasks, and unified admission across
model-loading surfaces. Current tasks still have eight tool rounds and one
final inference; background jobs and distributed agent execution are not included.
