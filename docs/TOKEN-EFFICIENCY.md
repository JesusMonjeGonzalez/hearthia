# Token efficiency and cache reuse

Local inference has no per-token invoice, but it has two real costs: **time**
(prefill and decode are the wall clock you feel) and **context** (a 64K window
is the scarce resource for long tasks). This page describes what Hearthia
actually sends, what was changed to make llama.cpp's prompt cache reusable,
and how to measure both on this Mac instead of trusting a claim.

## What a turn sends

1. **Tool schemas** — stable JSON, identical within a mode (Consult/Develop).
2. **System message** — harness rules and mode instructions; static for the
   conversation, so it caches.
3. **History** — reloaded from SQLite, cleaned of UI-only fields, append-only.
4. **Project context** — repo map, `AGENTS.md`, workspace note. **Appended as
   the prompt's trailing message, never merged into the system prefix and never
   woven into the stored user message.** The durable transcript must replay
   byte-for-byte next turn, or the cached prefix breaks inside the first user
   message; as a trailing block only the block itself is re-prefilled, and
   packing never mistakes it for a new question. The internal tag is stripped
   before the wire.
5. **Tool results** — bounded twice: each tool result is capped, and what a
   round appends is truncated to the remaining turn budget with an explicit
   note. Already-sent messages are never rewritten to make room.

## Element treatment

Every prompt element has an explicit cache class and an owner:

| Element | Position | Cache class | Policy |
|---|---|---|---|
| Tool schemas | request field | pinned | stable per mode; measured (1,031 tok read+coding) |
| System prompt | first message | pinned | static per mode (296 tok) |
| Project block | right after system | pinned, signature-keyed | reused **byte-identical** while the workspace is unchanged; rebuilt once when files change; ~1,600 tok for a Hearthia-sized repo |
| Digest | after the project block | bounded | deterministic memory of dropped turns (≤1.5 KB) |
| History | append-only | cached | never rewritten mid-turn |
| Tool results | tail of the current turn | ephemeral | bounded per call, token-aware room |
| Re-reads | — | replaced by a note | byte-identical content already in the prompt is not resent (sha256 match) |
| Question | each turn | new | never dropped, never merged with evidence |

Pinning the project block is what changed the prefill maths: it used to be
attached to the newest turn, so the ~1,600-token repo map shifted and was
re-prefilled *every turn*. Now the whole previous prompt is a prefix of the
next one while the workspace is unchanged (asserted in the tests), and only a
real project change costs one prefix shift.

The context event reports `element_tokens` per bucket (system, project context,
history, tool results, digest, tool schemas); the chat shows it in the
tooltip of the window line. `hearth tune` reads the same numbers back —
acceptance, KV vs `--cache-ram`, observed peak context — and advises without
touching the config.

## Measured prompt costs (this Mac, RVN Q4_K_M tokenizer)

| Component | Tokens | Bytes | Bytes/token |
|---|---:|---:|---:|
| System prompt (Develop mode) | 296 | 1,421 | 4.80 |
| Tool schemas (read + coding) | 1,031 | 3,946 | 3.83 |
| Repo map (real Hearthia checkout) | 1,613 | 4,087 | 2.53 |
| Python tool result (4 KB slice) | 1,121 | 4,012 | 3.58 |
| Spanish user sentence | 25 | 101 | 4.04 |

The window is 65,536 tokens and the KV cache for it is allocated at load time
(1.2 GiB at q4_0), so every unused token is RAM already paid for. The budget is
therefore derived from the window: `window − output reserve − template/tool
overhead`, converted with an observed bytes-per-token ratio. The previous fixed
60 KB byte cap corresponds to only ~18,000 tokens — roughly **28% of the 64K
window** — and is gone.

Before sending, the packed prompt is measured with the server's own tokenizer
(`/apply-template` + `/tokenize`, falling back to a JSON serialization that
over-counts — the safe direction). If it exceeds the allowance, Hearthia
re-packs with a tightened ratio and measures again, up to two corrections.
Each round's real `prompt_n` then recalibrates the per-model bytes/token seed,
so normal turns start from a ratio that matches the content mix actually used.
When the server cannot tokenize at all, the byte estimate stands and nothing
is blocked by the measurement itself.

## Re-reads are not resent

When the model asks for a full-file read whose exact bytes are already present
in the prompt, the harness answers with a three-line note instead of
thousands of tokens: the file is probed by size and sha256 (one cheap worker,
one round-trip, no inference), and the note is only produced when the sha
matches a copy **currently in this prompt** — a trimmed history or a changed
file reads in full. Only content the model has actually *seen as a file* is
shortcut: a file known only through the diff of its own edit stays readable in
full, because `edit_file` needs an exact `old_text` from the current bytes and
a note would take away the ability to edit it again.

Measured context: a 5.6 KB file is ~1,600 tokens; at ~145 tok/s prefill that
is ~11 s of work avoided per redundant re-read, and the tokens stay out of the
window entirely. Line-range reads are never shortcut, so the escape hatch
always works.

## One packing decision per turn

The context is packed **once** when the turn starts and is append-only
afterwards. When it must shrink, the order is: complete old turns first, then
the volatile project-context block (re-readable from disk), then the newest
tool results, and the digest last. The current question is never dropped and a
context block is never split from the question it belongs to.

Two harness policies make this stable instead of thrashing:

- **Dropped turns leave a digest.** Before a turn is dropped, its question
  (one-line stub) and each tool outcome (command exit codes, edited paths, MCP
  calls) are folded into a single bounded (~1.5 KB) deterministic memory
  message after the system prompt. No extra inference is spent, continuity
  survives, and repeated packings merge into the same digest instead of
  accumulating copies.
- **Hysteresis.** A drop frees room down to 85% of the budget, so the cached
  prefix is not shifted again on the very next turn. Each shift forces a
  re-prefill of everything after the cut, which is the dominant cost of a long
  session.

## Measured on this Mac (2026-09-28, RVN 64K, MTP)

Real numbers from a warm `qwen3.8-27b-rvn` through Hearthia's own chat:

| Measurement | Value |
|---|---|
| Prefill, cold prompt | 145 tok/s (1,569 tok in 10.8 s) |
| Prefill, cached prefix | 0.23 s for a fully reused 1,565-token prompt |
| Decode with MTP (`--spec-draft-n-max 3`) | 9–16 tok/s · acceptance **65%** |
| Cost of new context | ≈ 7 ms per prompt token, ≈ 70–110 ms per generated token |
| Resident with `--cache-ram 1536` | 20.9–21.0 GiB vs 20.3 GiB estimate (~3% under) |
| Cooled back | system at 10.8 of 36 GiB, 0 GiB committed |

A six-round tool session confirmed the cache structure end to end: reused
prefix grew monotonically (5,941 → 9,544 tokens) with only the appended
tokens evaluated — every round was append-only, no prefix shifts, and the
large prefill times (20 s) were exactly the new tool-result tokens at
~140 tok/s. `hearth tune` then recommended `--spec-draft-n-max 4` from the
measured 65% acceptance.

llama-swap does not proxy `/tokenize` or `/apply-template` at the root;
Hearthia reaches them through `/upstream/<model>/…`, which is what makes the
exact measurement path work behind the proxy (it falls back to the byte
estimate when even that is unavailable).

## How long until the 64K fills

Measured components (Develop mode, real RVN tokenizer): system 296 tok, tool
schemas 1,031 tok, a Hearthia-sized repo map 1,613 tok, template ~100 tok.
That is **~3,000 tokens before the first question**, against an input
allowance of ~59,400 (64K minus the 4K output reserve and overhead).

| Session shape | Growth per turn | Turns until trimming |
|---|---:|---:|
| Prose chat (one answer, no tools) | ~450 tok | **~125** |
| Coding with 3 file reads (12 KB each) | ~10,900 tok | **~5** |
| One 48 KB batch read in a round | ~14,000 tok | **~4** |
| Two 60 KB tool results | ~35,000 tok | **~2** |

The lesson: tool results, not conversation, fill the window. Hearthia reports
the live numbers per conversation (`#chat-context` in the UI, `projection` in
`GET /api/conversations/{id}`): occupancy, free tokens, observed growth per
turn and the estimated turns until the oldest turns start being omitted.

What happens when it fills: the oldest complete turns are replaced by the
digest and the session continues with a sliding window (the UI says how many
turns were omitted). Inside a turn, if the budget runs out mid-way the harness
forces one final tool-less round so the model answers with the evidence it
already gathered; only if even that cannot fit does the turn stop with an
explicit message. Nothing is corrupted silently.

The cost of a full window is prefill time, not RAM: reusing the cache after an
append costs only the appended tokens, while a prefix shift (a drop) forces
the retained history to be processed again — at ~59K tokens that is the whole
window. Measure the real prefill rate with `scripts/bench-prompt-cache.py`. Mid-turn, if the turn
outgrows its budget, Hearthia stops calling tools and runs one final round
without them; if even that cannot fit, it stops with an explicit message
instead of silently shifting the prompt. This is what lets llama.cpp reuse the
KV cache from round to round (`cache_prompt: true` plus `--cache-reuse` and
`--cache-ram` in the model `cmd`), instead of re-prefilling a changed context
on every tool round.

## Measuring it on this Mac

With the model already warm (the script refuses to load anything itself):

```sh
hearth warm qwen3.8-27b-rvn
.venv/bin/python scripts/bench-prompt-cache.py --model qwen3.8-27b-rvn --coding
```

The chat status line shows the same accounting live:
`input 18,204/59,418 tok (31% of allowance · 64K window) · 4096 output reserve
· estimated` becomes `measured` once the server tokenized the exact prompt.

The script sends the same agent-shaped request three times and prints, per
run, the llama.cpp `timings`: `prompt_n`, `cache_n` (tokens served from cache),
`prompt_ms` (prefill time) and generation tokens/second. A healthy cached
agent loop shows `cache_n` close to `prompt_n` on runs after the first and a
prefill time far below the cold run. Hearthia's own UI shows the same numbers
in the chat status line, and the per-message transcript stores the final
`timings` (visible in `?format=json` exports).

Cumulative usage per model (prompt tokens, completion tokens, largest context
observed) is already tracked in `~/.hearthia/usage_ledger.json` from
llama.cpp's Prometheus counters when `--metrics` is enabled.

## Honest comparison with Pi, Crush and OpenCode

| Dimension | Hearthia | Pi / Crush / OpenCode |
|---|---|---|
| Money cost | $0 (local weights) | $0 if pointed at the same local endpoint; otherwise provider pricing |
| Time cost | Same physics: prefill + decode on this Mac | Same physics; ordering of the prompt decides who re-prefills more |
| Prompt cache | `cache_prompt: true`, append-only turn, volatile context in the tail, cache figures surfaced | Provider-dependent; for local llama.cpp it depends on each tool's prompt shape (not published) |
| Token intake control | Conservative byte budget, bounded tool results, one packing per turn, spill→final round | Comparable classes of control (compaction, output budgets, context windows) |
| Export | Markdown + machine-readable JSON (`?format=json`) of every message, tool call, timing and error | Pi: JSONL session files; Crush/OpenCode: their own stores; interop is per-tool |
| Sessions | SQLite, revision checks, partial recovery, pagination, import | Pi: session trees/fork/resume; Crush: session storage + LSP |
| Agent features | Read/edit/command loop, diffs, RSS/wall limits, RAM policy | MCP client, LSP diagnostics, hooks, permissions UX, skills |
| Model/RAM management | **Unique**: GGUF-header budget gate, one-large-model policy, OS reserve, calibration, loadouts | Not provided; they assume the endpoint is safe |

Verdict, said plainly:

- **Money**: identical ($0) when all agents point at the same local model.
- **Speed per turn**: decided by prompt-cache reuse and prompt size. Hearthia
  now has the cache-friendly structure (stable prefix, append-only turn,
  volatile tail) but this document does not claim it beats Pi/Crush until
  `bench-prompt-cache.py` numbers exist for both on this Mac.
- **Long-context tasks**: 64K is a property of the model `cmd`; Hearthia's job
  is refusing loads that would overflow RAM and keeping the turn inside the
  window honestly. Pi/Crush do not do the RAM half.
- **Agentic completeness**: Pi/Crush/OpenCode are ahead on MCP client, LSP,
  hooks and session forking. Hearthia is ahead on memory safety, local-model
  lifecycle and durable, inspectable transcripts.

## What is still missing (no optimistic rounding)

- Runtime slot/context discovery per model is still not implemented: the
  window comes from the configured `--ctx-size`, not from querying a live
  slot. Measurement covers the prompt, not a mismatched slot configuration.
- No model-generated compaction summary yet: old turns are dropped, not
  summarised, so a very long conversation loses earlier detail.
- No LSP integration, hooks or permissions UI (MCP client landed: see MCP-CLIENT.md).
- No session forking/tree (Pi-style) and no JSONL session format.
- No side-by-side benchmark against Pi/Crush on this Mac yet; the script
  exists so that comparison can be numbers instead of opinion.
