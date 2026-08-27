#!/bin/sh
# Hearthia installer: https://github.com/JesusMonjeGonzalez/hearthia
#
#   curl -fsSL https://raw.githubusercontent.com/JesusMonjeGonzalez/hearthia/main/packaging/install.sh | sh
#
# Installs uv (via Homebrew when available, else the standalone installer)
# and Hearthia as a uv tool. Nothing else is touched.
set -eu

REPO="https://github.com/JesusMonjeGonzalez/hearthia.git"

say() { printf '\033[1;33mhearthia\033[0m %s\n' "$1"; }

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    say "Hearthia targets Apple Silicon Macs — this machine is not supported."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    say "installing uv…"
    if command -v brew >/dev/null 2>&1; then
        brew install uv
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

say "installing Hearthia from $REPO…"
uv tool install --force "$REPO"

say "done. try the demo with no setup:"
say "  hearth demo"
say "or wire up a real stack:"
say "  brew install llama.cpp llama-swap && hearth doctor"
