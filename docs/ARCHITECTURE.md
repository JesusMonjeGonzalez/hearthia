# Architecture

How Hearthia is built, and why the pieces are where they are. For the
user-facing story, read the [README](../README.md); this document is for
people changing the code.

```
┌─────────────┐   OpenAI API   ┌──────────────────┐   spawns    ┌────────────────┐
│ any client  │ ─────────────▶ │ llama-swap :9292 │ ──────────▶ │ llama-server   │
└─────────────┘                └────────┬─────────┘             │ (per model)    │
                                        │ SSE events, /running  └────────────────┘
┌─────────────┐   HTTP         ┌────────▼─────────┐
│ hearth CLI  │ ─────────────▶ │ hearthd :9300    │──▶ registry (llama-swap.yaml,
└─────────────┘                │                  │      ruamel round-trip + backups)
┌─────────────┐                │  · telemetry     │──▶ RAM budget (GGUF headers)
│ dashboard   │ ─────────────▶ │  · lifecycle     │──▶ Brain (sqlite-vec + vault)
└─────────────┘                └──────────────────┘
```

## The RAM budget (`budget.py` + `gguf.py`)

The core claim: **whether a model fits is arithmetic, not vibes.**

1. `gguf.model_ram_profile(path)` parses the GGUF header — block count,
   KV-head geometry, head dimensions, training context. Arrays are skipped
   with seeks, so the read costs kilobytes. No third-party parser.
2. `library.kv_cache_bytes()` computes the exact KV cache for the
   *configured* context (from `llama-swap.yaml`'s `--ctx-size`):

   ```
   bytes = n_layers × (k_len + v_len) × n_kv_heads × bytes_per_element × ctx
   ```

   `bytes_per_element` accounts for quantised cache formats (q8_0 carries
   block scales: 8.5 bytes per 8 elements). The cache type comes from the
   model's own `--cache-type-k/v` flags.
3. `library.estimate_resident_ram()` adds compute-buffer overhead
   (`max(5% of weights, 256 MiB)`).
4. `budget.plan_warm()` sums the candidate with everything already resident
   (measured RSS from `/running` when available, estimates otherwise) and
   compares against two ceilings: the GPU-wired limit (`sysctl
   iogpu.wired_limit_mb`, else 75% of RAM) and available memory.

**Failure modes are explicit.** An unreadable GGUF header degrades to a
file-size guess *with a visible warning* — never a silent one. The policy
is `[memory] mode`: `enforce` refuses the load, `warn` allows with the
verdict attached, `off` is advisory. `--force` on the CLI and the demo
(`warn`) are the documented escapes.

**The gate lives at all three entry points**: `hearth warm` (CLI),
`POST /api/models/{id}/load` (dashboard) and `LifecycleEngine._spawn`
(followers never spawn over budget). One planner, three doors.

## Lifecycle (`lifecycle.py`)

Every 10 s, follow rules from `config.toml` decide who should be warm:

- `app:Visual Studio Code` — a helper model lives while an app runs.
- `role:chat` — an embedding model lives while any chat model serves.

Rules declare intent; the engine warms and cools to converge. The
`role:` path has a 300 s grace period so a momentary chat unload doesn't
tear the embeddings model down with it. Crash loops (3+ exits in 5 min)
raise a macOS notification and the dashboard banner.

## Telemetry (`telemetry.py`)

The daemon watches llama-swap's SSE stream for activity and crash signals,
and polls each model server's **own** `/metrics` port for throughput. The
port comes from `/running`'s `proxy` field — deliberately *not* through the
gateway, because proxied requests count as activity and would reset the
TTL forever (models would never unload). This regression has a test.

## Registry (`registry.py`)

`llama-swap.yaml` is edited in place with ruamel's round-trip loader so
user comments survive, validated as safe-YAML before an atomic rename, and
backed up (10 rolling copies) before every write. Generated model blocks
carry `metadata.roles` so lifecycle rules match without heuristics.

## Demo (`demo.py`)

`hearth demo` serves the *real* routers against synthetic state:
`DemoGateway` mutates an in-memory running set on warm/cool, and the demo
GGUFs are sparse files with valid headers — so `gguf.model_ram_profile`
and the whole budget planner run for real during a demo. `memory.mode` is
`warn` in demo: it must never refuse its audience. `scripts/capture_demo.py`
drives the demo with Playwright to regenerate the README GIF.

## Invariants worth keeping

1. **Loopback-only.** `daemon.bind` refuses non-loopback IPs; foreign
   browser `Origin`s are rejected with 403. No auth by design — which is
   exactly why remote exposure is out of scope (see SECURITY.md).
2. **One HTTP client to the gateway.** Everything that talks to llama-swap
   goes through `gateway.py`; nothing else opens sockets to it.
3. **Estimates are honest.** Every number shown to a user is either
   measured (RSS, tok/s) or derived from the file (GGUF header) — and
   guesses are labelled as guesses.
4. **The suite stays hermetic.** No test talks to a real gateway, network
   or model; the demo provides the end-to-end path without hardware
   (see `tests/test_demo.py`).
