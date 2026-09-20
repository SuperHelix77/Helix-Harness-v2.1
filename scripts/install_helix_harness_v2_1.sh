#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

release_tag="v2.1.0"
asset_name="Helix-Harness-v2.1.0-macos-arm64.zip"
repository="SuperHelix77/Helix-Harness-v2.1"
install_root="${HELIX_INSTALL_DIR:-${HOME}/Applications}"
target_app="$install_root/Helix Harness v2.1.app"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Helix Harness v2.1 currently requires Apple Silicon macOS." >&2
  exit 1
fi

task_tmp="$(mktemp -d /private/tmp/helix-harness-v2.1-install.XXXXXX)"
cleanup() {
  rm -rf "$task_tmp"
}
trap cleanup EXIT INT TERM

release_base="https://github.com/$repository/releases/download/$release_tag"
archive="$task_tmp/$asset_name"
checksums="$task_tmp/SHA256SUMS.txt"

echo "Downloading Helix Harness v2.1…"
curl -fL --retry 3 --proto '=https' --tlsv1.2 \
  "$release_base/$asset_name" -o "$archive"
curl -fL --retry 3 --proto '=https' --tlsv1.2 \
  "$release_base/SHA256SUMS.txt" -o "$checksums"

expected="$(awk -v name="$asset_name" '$2 == name { print $1 }' "$checksums")"
if [[ ! "$expected" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "The release checksum manifest does not contain $asset_name." >&2
  exit 1
fi
actual="$(shasum -a 256 "$archive" | awk '{ print $1 }')"
if [[ "$actual" != "$expected" ]]; then
  echo "Checksum verification failed; the app was not installed." >&2
  exit 1
fi

ditto -x -k "$archive" "$task_tmp/unpacked"
source_app="$task_tmp/unpacked/Helix Harness v2.1.app"
if [[ ! -d "$source_app" ]]; then
  echo "The verified archive does not contain Helix Harness v2.1.app." >&2
  exit 1
fi

bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$source_app/Contents/Info.plist")"
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$source_app/Contents/Info.plist")"
if [[ "$bundle_id" != "ai.helix.harness.v21" || "$version" != "2.1.0" ]]; then
  echo "The verified archive has an unexpected application identity." >&2
  exit 1
fi

mkdir -p "$install_root"
if [[ -e "$target_app" ]]; then
  backup_app="$install_root/Helix Harness v2.1.backup-$(date +%Y%m%d-%H%M%S).app"
  echo "Preserving the existing v2.1 app at: $backup_app"
  mv "$target_app" "$backup_app"
fi
ditto "$source_app" "$target_app"
codesign --verify --deep --strict "$target_app"

echo "Installed: $target_app"
echo "SHA-256:  $actual"
if [[ "${HELIX_NO_LAUNCH:-0}" != "1" ]]; then
  open "$target_app"
fi
