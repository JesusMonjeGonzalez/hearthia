# Public Release Status

Status: **active development, public source release v0.6.0**.

The v0.6.0 release turns the dashboard into a daily agent harness: durable
SQLite conversations with retry/fork/search/export, a Develop mode with exact
edits, foreground and background commands, configured per-edit checks, an MCP
client, subagents on a disposable context, hooks, plan tracking with a
verification gate, a one-large-model residency policy with an OS reserve, and
token/cache budgeting measured with the real tokenizer. Hearthia remains a
local single-user tool, not a remote service or a `1.0` multi-user product.

Measured on the development machine (M4 Max, 36 GiB): a real autonomous
task (read → run failing tests → edit → re-run tests) completed in 104 s over
7 inference rounds with the 64K window at 3.7% occupancy and 98% prompt-cache
reuse by the end.

Still open before a distributable binary release:

- real llama.cpp/llama-swap model loading and restart evidence;
- memory-pressure and download recovery tests;
- browser smoke and accessibility checks;
- signed packaging and a third-party dependency/vendor inventory.
