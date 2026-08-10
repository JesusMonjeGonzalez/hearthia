# Hearthia

**The self-tending fire for local models** — a single-user, Mac-native control
plane for llama.cpp on Apple Silicon.

Models are **cold** (on disk), **kindling** (loading), or **warm** (in RAM).
Hearthia tends the fire so you don't: models load on first request, unload
after a TTL, and small helper models follow the thing they serve — the
autocomplete model lives while your editor runs, the embeddings model lives
while a chat model is warm.

## Status

Hearthia is under active development for Apple Silicon Macs. The CLI, daemon,
dashboard and model-management core are covered by an automated test suite;
real model loading still depends on a local llama.cpp/llama-swap installation.

## What you get

- **Gateway** — [llama-swap](https://github.com/mostlygeek/llama-swap) serving
  an OpenAI-compatible endpoint at `http://localhost:9292/v1` (plus Anthropic
  `/v1/messages`), managed as a launchd service.
- **`hearth` CLI** — status, warm/cool, verified resumable downloads from
  Hugging Face (`hearth pull --add` writes the config block for you), live
  logs, service management, `hearth doctor` triage.
- **Dashboard** at `http://localhost:9300` — unified-memory map with the
  GPU-wired limit, model cards with TTL countdown rings, streaming chat with
  conversations, a model library with fit checks, YAML config editor with
  line-anchored errors, live logs, and failure banners that name the broken
  subsystem.
- **Brain (optional)** — semantic search over an Obsidian vault using the
  local embeddings model (sqlite-vec); `brain "some thought"` captures a note,
  auto-titled and filed by a local model. Never fails: if models are down the
  raw text lands in the inbox.

## Install

```sh
brew install llama.cpp llama-swap
uv tool install hearthia        # or: uv tool install --editable .
hearth install                  # render + bootstrap launchd services
hearth doctor                   # verify the setup
```

Already running the pre-Hearthia `~/llm-stack`? `hearth migrate` adopts it in
place — weights and YAML never move, old services are booted out.

## Everyday use

```sh
hearth status                   # gateway health, warm models, RAM
hearth models                   # configured models with ember states
hearth warm coder               # force-load
hearth cool --all               # free all model RAM now
hearth pull unsloth/Qwen3.6-35B-A3B-GGUF --quant Q4_K_XL --add
hearth logs -f                  # follow gateway + model server logs
```

Configuration lives in `~/.config/hearthia/config.toml` (`HEARTHIA_*` env
vars override); model definitions stay in `llama-swap.yaml`, edited
structurally (comments survive, timestamped backups rotate).

## Development

```sh
uv run pytest                   # full suite, no gateway needed
uv run ruff check . && uv run mypy src
uv run hearth daemon --reload   # dev daemon
uvx --from playwright python tests/e2e/smoke.py   # UI smoke vs live daemon
```

Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md) · Changelog:
[`CHANGELOG.md`](CHANGELOG.md)
