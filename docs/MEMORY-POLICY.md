# Memory policy: one large model, live data, honest limits

Hearthia's admission gate answers one question before anything loads: does this
warm fit, given what is already resident and what macOS and every other app
need to keep working? The residence policy on top of the raw fit math is:

1. **One large model at a time.** A model whose resident estimate exceeds
   `helper_max_mib` occupies the single large slot. Loading a second large
   model is refused with the exact `hearth cool <id>` command that unblocks it.
2. **Small helpers share the slot.** Models under the helper cap (the 0.6B
   embedding, a 1.5B autocomplete) may stay warm alongside the large model,
   but only while the same ceiling and reserve checks keep passing.
3. **A reserve for macOS and apps.** After the load, at least
   `os_reserve_mib` of non-wired RAM must remain. Wired memory cannot be
   paged out; this reserve is what prevents the stutter-then-swap spiral that
   freezes the whole Mac.
4. **Swap is treated as evidence.** Swap already in use above
   `swap_warn_mib` is surfaced in the decision output and the warning line —
   the system is under real pressure before a load is even attempted.
5. **Fail closed.** If the gateway inventory cannot be read, what is resident
   is unknown and the warm is refused in `enforce` mode. An empty machine and
   an unreachable gateway are different facts.

Every decision is printed with its inputs — per-model estimates, measured RSS,
the wired ceiling, available RAM, the reserve, the swap figure and the policy
in force — so the numbers can be checked against Activity Monitor rather than
trusted blindly.

## Configuration

```toml
[memory]
mode = "enforce"          # enforce | warn | off
max_large_models = 1      # 1..4; number of large models allowed resident
helper_max_mib = 3072     # small-helper cap; 0 = strict single model
os_reserve_mib = 6144     # non-wired RAM kept free for macOS/apps
swap_warn_mib = 512       # swap above this is warned about
```

Environment overrides use the `HEARTHIA_MEMORY__` prefix, e.g.
`HEARTHIA_MEMORY__MAX_LARGE_MODELS=1`. Invalid values (zero or negative where
not allowed) are rejected at startup.

`warn` mode still runs every check and still reports the residency policy; it
just converts refusals into loud warnings. `--force` on `hearth warm` bypasses
the gate deliberately and is the only path that does.

## Measured on this Mac (2026-09-28)

Hardware and live state at the time of measurement (a snapshot, not a
guarantee — re-check with `hearth status`):

| Fact | Value |
|---|---|
| Machine | Apple M4 Max, 36 GiB unified memory |
| Wired GPU ceiling | 28.0 GiB (`sysctl iogpu.wired_limit_mb = 28672`) |
| RAM available at sample | 22.2 GiB · swap 0.0 GiB |
| Model used with Pi (`qwen3.8-27b-rvn`) | `RVN-Q4_K_M-multilingual-mtp.gguf`, 15.8 GiB weights |
| Resident estimate at 64K, q4_0 KV | **19.33 GiB** (weights + KV + mmproj + prompt cache) |
| Headroom vs wired ceiling | **8.67 GiB** |
| Headroom for macOS/apps after load | **16.67 GiB** (reserve: 6 GiB) |
| Verdict | Allowed with real margin; swap 0 |

The same arithmetic for the official `qwen3.8-27b` (UD-Q4_K_XL, 16.4 GiB
weights) gives 19.88 GiB — also inside the ceiling and the reserve.

Two caveats are part of the measurement, not footnotes:

- The MTP/speculative-decoding draft cache is **not modelled** in the
  estimate, and RSS includes things the header cannot see. Hearthia records
  measured RSS after real runs (see `hearth calibration`) and corrects the
  estimate with the learned factor; until two measurements exist per model,
  treat the estimate as a floor.
- `available` at any instant includes reclaimable inactive pages. The reserve
  is honest about what is promised: non-wired RAM left free for everything
  else. If other apps grow past it, macOS compresses and swaps.

## Verifying a load, in order

```sh
hearth status                     # live RAM, swap, wired ceiling, running set
hearth warm qwen3.8-27b-rvn       # prints the full decision lines, then loads
hearth calibration                # measured vs estimated factor, once runs exist
```

After a warm, Activity Monitor (or `hearth status`) is the source of truth for
what the model actually holds. If measured RSS runs materially above the
estimate, calibration folds that in on the next runs; the gate stays as
conservative as the data.

## What this does not promise

- It is not an OS-level memory cap. It refuses unsafe loads; it cannot stop
  a program you launch separately from allocating more.
- Direct clients of llama-swap (anything talking to port 9292 without going
  through Hearthia's warm path) bypass the gate. Deploy one loading surface.
- Estimates come from GGUF headers and file sizes; `warn`/`off` modes and
  `--force` deliberately trade this protection away.
- A 64K context is a property of the model's configured `cmd`, not of this
  policy: the policy only decides whether the resulting load fits the budget.
