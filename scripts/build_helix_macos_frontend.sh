#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/scripts/stage_helix_backend_overlay.py"
npm --prefix "$ROOT/studio/frontend" run build
