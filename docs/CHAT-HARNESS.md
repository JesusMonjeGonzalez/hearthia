# Persistent project chat

Hearthia's dashboard now uses server-side conversations, disposable filesystem
workers and a model-aware context budget. **Consult** is read-only; explicit
**Develop** mode adds editing and foreground commands. See
[`CODING-AGENT.md`](CODING-AGENT.md) for its tools, limits and tested workflow.

## Using it

1. Open **Chat → New chat** and choose a model.
2. Enter a project directory, for example `/Users/me/project`. Relative tool
   paths resolve there. Hearthia reads a bounded project map and the root
   `AGENTS.md`; it does not build a vector index or warm embeddings just because
   a project was selected. Deeper instructions must be read explicitly by the agent.
3. Send a message. Tool activity is shown separately, and saved tool results
   can be expanded in the transcript. **Stop** cancels the connection and any
   active filesystem worker.
4. Choose **Develop** to allow the agent to edit project files and run checks.
   Diffs and command results are preserved in the same transcript.
5. Reload the page or restart Hearthia: saved messages and partial replies are
   still available. Interrupted turns are labelled; no tool is automatically
   rerun on restart. **Refresh** reloads a session being used from another tab.
6. Use **Older messages / Latest messages** to page through history and
   **Export .md** for the complete transcript, including tool calls and results.

**Import browser chats** imports the previous `hearthia.convs` localStorage data.
Imports are idempotent by content and leave the original browser data intact.
An import over 1 MB or 1,000 messages is rejected rather than silently truncated.
If an import fails halfway through a batch, retrying skips identical imports.

## Searching every conversation

The dashboard has a **Search all messages** box. It queries a SQLite FTS5
index kept in sync with every append, streaming checkpoint and deletion, and
backfilled on first open (so conversations written before the index existed
are searchable too). Results are grouped per conversation with a highlighted
snippet and a hit count; clicking one opens that conversation at the matching
message. When FTS5 is unavailable the store degrades to a bounded LIKE scan
instead of failing.

Search costs **zero model tokens** — it never triggers an inference — and it
is available over the API as
`GET /api/conversations/search?q=...&limit=...`. Query tokens are sanitised
and AND-ed; quotes cannot break the query.

## Reloading configuration

`POST /api/config/reload` re-reads `config.toml` without restarting the daemon
and reports exactly what changed. It applies the sections read per use
(`agent`, `memory`, `brain`), rebuilds the MCP manager when its servers
changed, and updates `max_jobs`/`job_max_minutes`. A config that does not
parse is rejected (422) and the running settings are left untouched. Model
registry and service settings still need a restart.

## Retry, fork and the verification marker

- **Retry last turn** (`POST /api/conversations/{id}/retry`, the Retry button)
  forks the conversation *without* its last user turn and returns the message,
  which the dashboard re-submits in the copy. The original stays intact for
  comparison; nothing is rewritten in place.
- **Fork** copies a conversation (or any prefix, `from_seq`) into an
  independent one with `forked_from` provenance. Forks are refused while the
  source is running, so a copy is never taken mid-turn.
- **Verification marker**: each turn records whether files were edited and
  whether a command ran *after the last edit*. The chat shows an amber
  "Unverified" line when edits ended the turn without a check. The marker says
  exactly that — a command ran after the edit — not that the command tested
  the change.

## Persistence and recovery

- Data lives in `<stack_dir>/conversations.sqlite3` (normally
  `~/.hearthia/conversations.sqlite3`), with SQLite WAL sidecar files while in use.
  This is separate from `sessions.json`, which describes model loadouts.
- The user message is committed when the turn starts. Assistant output is
  checkpointed when chunks arrive at intervals of at least 0.5 seconds, at stream
  end and on cancellation. Tool-call messages and results are recorded separately.
- Abrupt process termination can lose output since the last checkpoint; recovery
  marks running conversations as interrupted. Missing tool results are represented
  as interrupted in subsequent model context, not silently reexecuted.
- Revision checks prevent a stale browser tab from appending to a changed session.
  A running session cannot be deleted. A rejected revision does not append a user
  message. Only one chat turn runs per daemon process.
- Session lists contain metadata only. The browser retains one page of up to 40
  messages, not all conversations. API history pages also have a 512 KB payload
  budget, except that one larger message is returned whole. Exports are streamed
  in bounded batches with database connections closed between batches.
- Transcripts include user text, retrieved files and model reasoning when
  supplied. They are local plaintext SQLite data, not encrypted storage. No
  automatic retention policy or cloud synchronization is enabled.

## Tool runtime

Filesystem calls run in a fresh Python process with the selected workspace as
its working directory. No persistent pool or model-per-tool process is retained.

| Limit | Current value |
|---|---|
| Filesystem worker wall time | 10 seconds |
| Filesystem/edit worker CPU time | 5 seconds soft / 6 seconds hard |
| Worker protocol output | 256 KiB |
| Tool result | 60,000 characters maximum before context packing |
| Tool calls | 16 per round; 8 tool rounds followed by a final inference |
| `read_file` | Line-numbered ranges, up to 500 lines, 12,000 content characters, bounded scanning |
| `read_files` | Up to 24 files, bounded prefixes |
| Notes search | Separate async path with a 30-second deadline and memory admission check |
| Develop commands | Up to 120 seconds, 32 KiB output buffer, configurable sampled RSS allowance |

Timeout/cancellation kills and reaps the filesystem worker's process group.
This provides real termination for slow volumes or pathological regex searches;
it is not merely a timeout around an unstoppable thread. The worker is **not a
sandbox or an OS memory limit**: absolute paths remain readable with the local
process's permissions. Workspace sets cwd/instructions, not filesystem isolation.

## Context management

The configured model context supplies the working window; unknown/automatic
context uses an 8,192-token fallback. Output is capped to one quarter of that
window, up to the requested `max_tokens`. Tool schemas and a 1,024-unit template
allowance are reserved before packing messages.

The input estimate uses UTF-8 bytes conservatively, **not an exact tokenizer**.
Configured context may also differ from runtime context per slot. The UI labels
this as context bytes and shows the output cap, omitted turns and shortened tool
results. Real llama.cpp tokenization/slot discovery remains future work.

Old complete user turns are removed from the model-facing copy when necessary;
tool-call/result pairs stay together. Oversized tool results are shortened with
an explicit note. A current user request that still cannot fit is rejected.
The saved transcript is never compacted or deleted by this process. A bounded
history suffix is loaded from SQLite before packing; the UI identifies that
condition. This is a sliding context window, not a model-generated summary.

Browser attachments are capped at 8 files, 200 KB each and 400 KB total, checked
before reading file contents. Such an attachment may still exceed the selected
model's context: use project tools and line ranges for large sources.

## API

- `POST /api/conversations`: create metadata or import user/assistant messages.
- `GET /api/conversations?limit=50&offset=0`: paginated metadata.
- `GET /api/conversations/{id}?limit=40&before={seq}`: message page and revision.
- `GET /api/conversations/{id}/export`: streamed Markdown transcript.
- `DELETE /api/conversations/{id}`: delete an inactive conversation.
- `POST /api/chat`: persistent turns include `session_id`, `revision`, `model`,
  `workspace`, `system`, `mode` (`read` or `build`) and exactly one new user message. The server reconstructs
  tool-aware history. Without `session_id`, the original stateless interface works.

Chat streams include content deltas plus `context`, `tool_event`, `error` and
final `session` events. Normal completion has one `[DONE]`. Upstream error events
and streams that end without a completion signal are recorded as errors.

## Verification

```sh
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
node --test tests/web/*.test.mjs
node --input-type=module --check < src/hearthia/web/chat.js
uvx --from playwright python tests/e2e/chat_sessions.py
```

The browser test starts its own temporary demo daemon on a free loopback port,
uses an isolated browser context and cleans up the daemon. It verifies persistence
after reload, workspace retention, idempotent browser migration, bounded message
pages, full export and Stop. It uses installed Chrome on macOS or an installed
Playwright Chromium. No real model or existing user service is used.

Observed after adding Develop mode: **624 Python tests passed**, **14 JavaScript
tests passed**, and the Chrome end-to-end session/coding test passed. The fixture
scripts model decisions but executes real edits and unittest subprocesses. These
checks do not measure real-model GPU memory or long-session RSS.

Remaining major gaps: client MCP, coordination of memory
admission across every loading surface, exact token budgeting, and real-model
long-session resource measurements. Deploy one daemon process for this runtime;
multi-worker shared execution is not supported.
