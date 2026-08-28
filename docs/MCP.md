# Hearthia MCP server

Hearthia speaks the [Model Context Protocol](https://modelcontextprotocol.io):
an AI agent running on the same Mac can manage the local model lifecycle as
self-service — ask what fits before it needs a model, warm and cool models
itself, load named loadouts, and search the local Brain.

The RAM budget gate is enforced on every tool, so an agent can never wire
more than the GPU ceiling. The server speaks JSON-RPC 2.0 over stdio with the
standard library only — `uvx` not required at call time, no extra packages.

## Tools

| Tool | What it does |
|---|---|
| `hearthia_status` | Gateway health, warm models, resident RAM, tok/s, budget lines |
| `hearthia_models` | Configured models with state, estimated resident RAM, roles |
| `hearthia_warm` | Budget-checked load — returns the arithmetic, or fit options when blocked |
| `hearthia_cool` | Unload one model, or everything with `model_id: "__all__"` |
| `hearthia_est` | What-if: would these models fit together? Nothing loads |
| `hearthia_advise` | Change-sets that make a set fit: KV quantisation, lower ctx, cooling |
| `hearthia_loadout` | Warm a named `[loadouts]` set; `name: "__list__"` lists them |
| `hearthia_brain_search` | Semantic search over the Obsidian vault |

## Client setup

The server command is the Hearthia CLI itself:

```json
{ "command": "hearth", "args": ["mcp"] }
```

### Claude Desktop — `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hearthia": { "command": "hearth", "args": ["mcp"] }
  }
}
```

### Claude Code

```bash
claude mcp add hearthia -- hearth mcp
```

### OpenCode — `opencode.json`

```json
{
  "mcp": {
    "hearthia": {
      "type": "local",
      "command": ["hearth", "mcp"]
    }
  }
}
```

### Zed — `settings.json`

```json
{
  "context_servers": {
    "hearthia": {
      "source": "custom",
      "command": "hearth",
      "args": ["mcp"]
    }
  }
}
```

## Example agent loop

The point: the agent plans memory before it spends it.

```text
agent → hearthia_est      ["qwen-coder-30b", "qwen3-embedding-0.6b"]
agent ← total 22.8 GiB of 28.0 GiB wired — FITS
agent → hearthia_warm      { "model_id": "qwen-coder-30b" }
agent ← qwen-coder-30b is warm … total 22.8 GiB of 28.0 GiB wired
agent → (inference via the gateway at http://127.0.0.1:9292/v1)
agent → hearthia_cool      { "model_id": "__all__" }
```

When a warm is refused the response includes `hearthia_advise`-style
options — quantise the KV cache, drop context, or cool a running model —
so the agent can adapt instead of failing.

## Loadouts

Declare sets in `~/.config/hearthia/config.toml`:

```toml
[loadouts.coding]
description = "Flagship coder + embeddings helper"
models = ["qwen-coder-30b", "qwen3-embedding-0.6b"]

[loadouts.notes]
models = ["gemma-notes-12b", "qwen3-embedding-0.6b"]
```

Load them from the shell the same way: `hearth loadout load coding`,
inspect with `hearth loadout show coding`, release with
`hearth loadout cool coding`.
