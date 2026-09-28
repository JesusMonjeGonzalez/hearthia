# MCP client: let the chat use your other tools

Hearthia already exposes **itself** as an MCP server (`hearth mcp`) so other
agents can manage models. This is the other direction: the chat can now
**consume** tools from stdio MCP servers — the same capability Crush and
OpenCode use to reach the rest of a team's tooling.

Servers are opt-in. Nothing is spawned unless it is declared in
`config.toml`:

```toml
[mcp.servers.fetch]
command = ".venv/bin/python"
args = ["-m", "mcp_server_fetch"]
read_only = true          # safe to expose in Consult mode too
timeout_seconds = 30      # 1..300, per request

[mcp.servers.github]
command = "github-mcp-server"
args = ["stdio"]
env = { GITHUB_TOKEN_CMD = "security find-generic-password -w -s gh" }
```

## How tools reach the model

- At the start of a turn Hearthia starts each configured server, performs the
  MCP `initialize` handshake and calls `tools/list` (cursor pagination, capped
  at 64 tools per server and 8 servers).
- Tools are exposed as `mcp__<server>__<tool>`. Names longer than 64 characters
  are shortened with a stable hash suffix; the mapping is kept internally so
  results are attributed to the real server and tool.
- **Develop mode** sees every configured server. **Consult mode** only sees
  servers explicitly marked `read_only = true`. A model that calls a mutating
  MCP tool in Consult mode gets a refusal — the executor enforces it, not the
  UI.
- Mutating MCP calls are never deduplicated, are serialized (never run in the
  parallel read batch), and their results are stored in the conversation like
  any other tool result.

## Limits and behaviour

| Bound | Value |
|---|---|
| Servers | 8, names `[A-Za-z0-9_]+` (no `mcp__` prefix) |
| Tools per server | 64, schema ≤ 8 KB, description ≤ 500 chars |
| Arguments | 32 KB, must be a JSON object |
| Output | 64 KB of text blocks; `isError` surfaces as `ok: false` |
| Timeout | per server, 1–300 s, default 30 s |
| Framing | newline-delimited JSON-RPC 2.0 over stdin/stdout, one request in flight |

- A hung request **restarts the server** (killing the process group) instead of
  leaving a stale reply in the pipe; the next call starts it again.
- A server that exits mid-request returns an explicit error including its last
  stderr lines; the next call transparently restarts it. A discovery failure
  is surfaced once per turn in the chat activity, never hidden.
- Shutdown kills every server process group. No server is left running after
  the daemon stops.

## Security boundary

MCP servers are ordinary programs run with **this user's permissions**, in a
new session, with no shell interpolation. Declaring one is equivalent to
trusting that program — the same trust level as `run_command` in Develop mode,
but outside the workspace restriction. The `read_only` flag is Hearthia's
promise to the *model* (which modes may call the tools); it does not restrict
what the server process itself can do. Declare servers you would run yourself
in a terminal, keep secrets in the server's own keychain/env, and prefer
receiving agents (fetching, search, issue trackers) over anything that writes
to production systems.

## Verifying

```sh
.venv/bin/pytest -q tests/test_mcp_client.py tests/test_chat_mcp.py
```

The tests run a real MCP server subprocess over stdio
(`tests/fake_mcp_server.py`): discovery, namespacing, read-only gating, an
error result, a timeout that restarts the server, a crash with recovery, and
process cleanup on `close()`. The chat integration tests prove tools are
exposed per mode, refused when they should be, recorded in the transcript, and
that two independent reads in one round run concurrently while keeping result
order.
