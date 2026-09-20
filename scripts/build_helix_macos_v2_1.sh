#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/studio"

# Rust otherwise embeds the builder's checkout and Cargo registry paths in
# panic/diagnostic strings. Keep release binaries reproducible and free of
# machine-local account identity without suppressing useful source locations.
build_cargo_home="${CARGO_HOME:-${HOME}/.cargo}"
release_remap="--remap-path-prefix=$ROOT=/helix-source"
if [[ -d "$build_cargo_home" ]]; then
  release_remap+=" --remap-path-prefix=$build_cargo_home=/cargo"
fi
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$release_remap"

exec npx --prefix . tauri build --bundles app \
  --config src-tauri/tauri.helix-v2-1.conf.json "$@"
