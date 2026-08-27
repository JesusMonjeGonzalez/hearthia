# Contributing to Hearthia

Thanks for helping tend the fire. Hearthia is a focused tool — a single-user,
local, Apple-Silicon control plane for llama.cpp — so contributions that keep
that focus are the ones that land.

## Ground rules

1. **Local-only stays local.** No features that require binding to a
   non-loopback interface, accounts, or a server component.
2. **Memory estimates must be honest.** Anything that changes the RAM budget
   maths needs a test that pins the arithmetic.
3. **The config round-trips.** Edits to `llama-swap.yaml` must preserve
   comments and formatting (ruamel handles this; don't reach for plain YAML
   dumps).

## Development

```bash
git clone https://github.com/JesusMonjeGonzalez/hearthia.git
cd hearthia
uv sync
uv run pytest          # the suite must stay green
uv run ruff check .    # style
uv run ruff format --check .
uv run mypy src        # types
```

If you don't have an Apple Silicon Mac or any models installed, `hearth demo`
runs the full product against a synthetic stack — most UI and API work can be
verified against it.

## Pull requests

- One logical change per PR.
- Add or update tests for anything you touch.
- Update `CHANGELOG.md` under an "Unreleased" heading if the change is
  user-visible.
- CI must pass: ruff, mypy and pytest on macOS.

## Reporting bugs

Open an issue with the bug template. `hearth doctor` output (redact nothing —
Hearthia never leaves your machine, but strip anything you consider private)
and `hearth status` make reports actionable.
