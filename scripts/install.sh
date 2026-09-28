#!/usr/bin/env bash
# Hearthia installer: dependencies, the tool itself and the launchd services.
#
#   ./scripts/install.sh            install/upgrade and start everything
#   ./scripts/install.sh --check    report what is missing, change nothing
#
# Idempotent: safe to re-run after a git pull or a new release.
set -euo pipefail

check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

say()  { printf '  %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  say "Hearthia targets macOS on Apple Silicon (found $(uname -s)/$(uname -m))."
  exit 1
fi

missing=()
have brew || missing+=("homebrew (https://brew.sh)")
have uv   || missing+=("uv (brew install uv)")
have llama-server || have llama.cpp || missing+=("llama.cpp (brew install llama.cpp)")
have llama-swap || missing+=("llama-swap (brew install llama-swap)")

if (( ${#missing[@]} )); then
  say "missing:"
  for item in "${missing[@]}"; do say "  - $item"; done
  if (( check_only )); then exit 1; fi
  say "installing the Homebrew ones now (uv included)…"
  have brew || { say "install Homebrew first: https://brew.sh"; exit 1; }
  brew install uv llama.cpp llama-swap
else
  say "dependencies: brew, uv, llama.cpp, llama-swap present"
fi

if (( check_only )); then
  say "check complete — run without --check to install"
  exit 0
fi

say "installing Hearthia…"
uv tool install --force "git+https://github.com/JesusMonjeGonzalez/hearthia.git"

say "installing the background services (gateway + daemon)…"
hearth install

say "starting everything and opening the chat…"
hearth chat --no-warm

say "done. Next time: 'hearth chat -m <model> -w <project>' or just 'hearth'."
