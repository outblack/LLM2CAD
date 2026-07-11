#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "cadgen02 requires 'uv' so the verified lockfile is used." >&2
  exit 127
fi

exec uv run --locked --quiet python -m cadgen.cli "$@"
