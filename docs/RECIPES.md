# Ecosystem recipes

Hearthia speaks the OpenAI API on `http://127.0.0.1:9292/v1`, so anything
that can use a local endpoint can use Hearthia — with lifecycle and memory
budgeting that plain llama.cpp doesn't have.

Pick any model you have configured (or `hearth adopt-ollama --add` to bring
your Ollama models over) and plug the base URL in below.

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:9292/v1` |
| API key | any non-empty string (local only) |
| Model | an id or alias from `llama-swap.yaml` |

## Code editors

### Zed

`settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "Hearthia": {
        "api_url": "http://127.0.0.1:9292/v1",
        "available_models": [
          {
            "name": "local-model",
            "display_name": "Hearthia local",
            "max_tokens": 32768
          }
        ]
      }
    }
  }
}
```

### Continue.dev (VS Code / JetBrains)

`~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Hearthia local",
      "provider": "openai",
      "apiBase": "http://127.0.0.1:9292/v1",
      "model": "local-model",
      "apiKey": "hearthia"
    }
  ]
}
```

### OpenCode

`opencode.json`:

```json
{
  "provider": {
    "hearthia": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:9292/v1" },
      "models": { "local-model": {} }
    }
  }
}
```

## Knowledge and notes

### Obsidian + the Brain

Point `[brain].vault` in `~/.config/hearthia/config.toml` at your vault, then:

```bash
hearth brain reindex     # embed new/changed notes locally
hearth brain search "how did I split strings in zsh"
hearth brain capture "the idea I just had"
```

In the dashboard's Chat tab, ask anything; the chat pipeline searches your
notes semantically and cites the note paths when they're relevant.

### Anything with a "custom OpenAI endpoint" box

Typing simulators, shells (`aichat`, `llm`, `mods`), Raycast AI proxies,
Dropover-style helpers — they all want a base URL and a model name. Give them
Hearthia's and every model it manages is one TTL away from freeing your RAM.

## Migration recipes

### From Ollama

```bash
hearth adopt-ollama          # see what Ollama already pulled, with real RAM costs
hearth adopt-ollama --add    # write config blocks for all of them
hearth restart gateway       # they're now OpenAI-compatible, budget-managed
```

Ollama keeps models resident until *you* kill them; Hearthia's TTL and RAM
budget gate decide for you, per model.

### From LM Studio (or any folder of GGUFs)

```bash
hearth scan ~/.lmstudio/models
hearth scan ~/.lmstudio/models --add
```

### From a hand-rolled llama.cpp alias farm

```bash
hearth migrate               # adopt an existing ~/llm-stack in place
hearth doctor                # verify services, ports, wired limit, disk
```

## Verifying the ecosystem story in one minute

```bash
hearth demo   # full dashboard + chat, zero setup — the product, on synthetic models
```
