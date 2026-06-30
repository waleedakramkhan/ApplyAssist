#!/usr/bin/env bash
# Convenience launcher for ApplyAssist.
# Activates Node 18+ (via nvm) and the project venv, then runs applyassist.
# Usage:  ./applyassist.sh <command> [args...]
#   e.g.  ./applyassist.sh doctor
#         ./applyassist.sh run
#         ./applyassist.sh apply
#
# This avoids touching your global nvm default (kept at v14 for other projects).
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use Node 22 (or any installed 18+) for this command only.
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
  nvm use 22 >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1 || true
fi

exec "$HERE/.venv/bin/applyassist" "$@"
