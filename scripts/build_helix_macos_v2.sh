#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/studio"

exec npx --prefix . tauri build --bundles app \
  --config src-tauri/tauri.helix-v2.conf.json "$@"
