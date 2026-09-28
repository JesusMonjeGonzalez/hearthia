# Long and autonomous sessions

Four harness mechanisms keep long Develop-mode turns alive on a local Mac,
where a five-minute test suite is longer than the model's own TTL.

## 1. Keepalive while tools run

llama-swap evicts a model after its `ttl` (300 s in this stack) without
activity. A verification command that runs longer than the TTL used to unload
the model mid-turn: the next round then paid a reload plus a full re-prefill
(tens of seconds to minutes at ~145 tok/s). While a Develop-mode turn is
running, Hearthia sends a one-token, cache-hit completion every
`agent.keepalive_seconds` (default 60), which resets llama-swap's activity
timer. The loop is cancelled the moment the turn ends.

Measured on the real stack: after 40 s idle the countdown read 4m17s; after
47 s more *with* the keepalive it read **4m58s**. The eviction risk is gone
for tool runs of any length.

## 2. Round and time budgets

```toml
[agent]
max_tool_rounds = 8        # 1..48 tool rounds per turn
turn_budget_minutes = 0    # 0 = no wall cap; otherwise a per-turn deadline
keepalive_seconds = 60     # 0 disables the keepalive
```

When a budget runs out mid-turn the harness stops calling tools and forces one
final answer round — the same graceful path as a context spill, never a silent
failure. Raise `max_tool_rounds` for genuinely long autonomous tasks; the
context digest and hysteresis keep the window stable meanwhile.

## 3. The plan survives context trimming

`update_plan` (Develop mode) records 1..24 short steps in the conversation
metadata. The harness pins them as a stable block near the top of every later
prompt, so the model keeps its spine even after old turns are dropped into the
digest. A 150-char stub of the model's own conclusion is kept per dropped
turn, so findings survive the drop, not only tool history. Mid-turn updates are returned in the tool result immediately and
pinned from the next turn on; the digest also records `plan updated (N steps)`.

Plans are bounded (24 steps × 160 chars), validated, persist across daemon
restarts, and are stripped of internal tags before the wire.

## Plan checklist in the dashboard

The persisted plan is rendered as a collapsible checklist under the chat
(`#chat-plan`): step count and age, no extra tokens spent — the panel reads
the same metadata the pinned plan block uses.

## 3c. The verification gate

Marking a step done is a claim. When this turn edited files and **no command
ran afterwards**, `update_plan` reacts:

- default: the plan is saved but the tool result carries
  `warning: "unverified edits this turn…"`, so the model sees the gap;
- `[agent] require_verification = true`: the call is **refused** (nothing is
  persisted) until a command runs, with the exact next move in the error.

Reads-only turns and verified turns pass untouched, and the escape hatch is
explicit: run any relevant check, or turn the setting off. The cost is zero
tokens unless there is a mismatch to report.

## 3d. Background jobs

A verification command longer than the turn budget no longer forces a choice
between blocking the model and skipping the check. `run_command` accepts
`background: true`: the command starts detached, writes its **full output to a
log file**, and the model continues. `job(action=...)` then:

- `list` — every job with state, duration and log size;
- `status` — bounded status plus a **2 KB tail** (the log never enters the
  conversation whole, same idea as subagents);
- `wait` — block up to 120 s for completion (efficient polling without
  burning rounds);
- `stop` — kill the whole process group.

Bounds: `[agent] max_jobs` (default 3) concurrent jobs refused with guidance
when full, `job_max_minutes` (default 30) after which the group is killed and
the state becomes `timeout`, and a capped log file (20 MB default) that is
drained past the cap so the child never blocks. Jobs are killed on daemon
shutdown — they do not outlive the daemon — and a restarted daemon reaps
leftovers from a previous one, guarding against pid reuse by comparing the
recorded spawn time with the process' creation time.

Jobs are visible outside the model too: `GET /api/jobs`, `GET /api/jobs/{id}`
and `POST /api/jobs/{id}/stop`, the `hearth jobs [list|status|stop]` CLI, and a
status line in the chat with a **stop** button per running job (polled every
10 s only while something runs).

`Stop` cancels the *turn*, not jobs already started: a long suite keeps
running and can be checked from the next turn (that is the point of detaching
it). Use `job(action="stop")` to end it deliberately.

## 3e. Hooks: your automation on agent events

```toml
[[agent.hooks]]
events = ["turn_end"]
command = ["/usr/bin/osascript", "-e", "display notification \"turn finished\""]

[[agent.hooks]]
events = ["edit"]
command = ["/path/to/formatter", "--file", "stdin-is-payload"]   # payload arrives on stdin
```

Two events: `turn_end` (status, model, workspace, mode, token totals, edited
paths, verified flag) and `edit` (path, conversation, workspace). The payload
is JSON on **stdin**; the command is an argv array (no shell).

Semantics, deliberately narrow: hooks are **fire-and-forget automation** — a
notification, a formatter, a log shipper. They never delay a turn, their
output never enters the conversation (it goes to the daemon log and a ring of
the last 20 runs), a failing hook is recorded and ignored, a slow one is
killed at its `timeout_seconds`, and at most 4 run at once (extra fires are
dropped with a log line instead of queuing). If you need the *model* to see a
result, use `run_command`, per-edit `checks`, or the verification gate
instead.

Visibility: `GET /api/hooks` (configured + recent runs) and a `hooks` line in
`hearth doctor`. Unknown event names are rejected at config load, so a typo
fails loudly instead of silently never firing. `POST /api/config/reload`
applies hook changes without a restart.

## 3e-bis. Model compaction (opt-in)

```toml
[agent]
compaction = "digest"   # or "model"
```

With `model`, dropped turns are summarised by the model itself and the summary
**rolls forward**: each flush merges the previous summary with the newly
dropped turns (≤200 words, input capped at 10 KB, output capped at 400
tokens). Summaries are produced by a **background task**, never blocking a
turn; the deterministic digest still covers the newest drops and is the
fallback whenever the summary call fails, times out, or the model is not
loaded. The stored summary lives in the conversation metadata
(`compaction_summary`) and is re-used by every later packing.

Honest costs: it is one extra inference per drop event (an opt-in trade of
tokens for semantic continuity), and the summary replaces detail with prose —
the raw transcript always keeps everything.

## 3f. Token economy for long runs

Two small policies keep long autonomous runs from spending context on
bookkeeping:

- **Compact plan acks.** `update_plan` echoes the full step list only when
  the list actually changes. Status-only updates (the common case) return
  `{ok, done, pending}` — measured on a real run, five of six updates were
  status-only, so most of the ~80 tokens per update disappear, and the plan
  still reaches the model through its own pinned block and the UI.
- **Round pacing.** Once a turn has three or fewer tool rounds left, every
  tool result carries a short note (`tool round 6/8; wrap up, verify, or
  answer with what you have`) — inside the JSON as a `pacing` key for
  structured results, as a trailing line for text. The model can pace itself
  instead of being cut off at the cap.

## 4. No-progress guard

Reads are deduplicated, but a runaway agent repeating a failing command or
edit had no guard. From the third identical execution in one turn the tool
result carries an explicit harness note telling the model to change approach
or arguments. Counts are per turn and bounded; the note is visible in the
transcript and in the digest's tool outcomes.

## What is deliberately not here yet

- Model-generated compaction summaries (the digest is deterministic and free;
  a summary costs a full extra inference over the dropped history).
- Auto-continuation across turns: a task that exhausts its budget answers and
  stops; continuing is the user's call.
- A verification gate: edits are not *required* to be followed by a command;
  the transcript simply shows what ran.
