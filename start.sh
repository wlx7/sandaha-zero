#!/usr/bin/env bash
# One-click launcher for macOS / Linux
set -e
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[first run] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
exec uv run --python 3.12 sandaha "$@"
