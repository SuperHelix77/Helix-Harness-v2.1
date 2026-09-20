# SPDX-License-Identifier: AGPL-3.0-only
"""Runtime proof that the loaded backend matches a bundled Helix overlay.

The desktop overlay installer writes .helix_backend_manifest.json into the
installed backend only after it has verified the bundle manifest and copied the
owned files into an atomic staging tree. A backend process re-hashes those owned
files before publishing its identity. This prevents a same-install but stale or
locally modified terminal backend from being mistaken for the exact backend
bundled with the desktop app.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from state.tool_policy import HELIX_HARNESS_BACKEND_CONTRACT

_MANIFEST_NAME = ".helix_backend_manifest.json"


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _empty(reason: str) -> dict[str, Any]:
    return {
        "contract": HELIX_HARNESS_BACKEND_CONTRACT,
        "tree_sha256": None,
        "verified": False,
        "reason": reason,
    }


@lru_cache(maxsize=1)
def helix_backend_identity() -> dict[str, Any]:
    """Return the verified runtime overlay identity, or an explicit unverified state."""

    root = _backend_root()
    manifest_path = root / _MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty("runtime_manifest_missing")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _empty("runtime_manifest_unreadable")
    if not isinstance(raw, dict):
        return _empty("runtime_manifest_invalid")

    contract = str(raw.get("contract") or "")
    expected = str(raw.get("tree_sha256") or "")
    files = raw.get("files")
    if contract != HELIX_HARNESS_BACKEND_CONTRACT:
        return _empty("runtime_contract_mismatch")
    if len(expected) != 64 or not all(ch in "0123456789abcdef" for ch in expected):
        return _empty("runtime_tree_digest_invalid")
    if not isinstance(files, list) or not files or raw.get("file_count") != len(files):
        return _empty("runtime_file_manifest_invalid")

    digest = hashlib.sha256()
    seen: set[str] = set()
    try:
        for item in files:
            if not isinstance(item, str) or not item or item in seen:
                return _empty("runtime_file_manifest_invalid")
            rel = Path(item)
            if rel.is_absolute() or rel.as_posix() != item or any(
                part in ("", ".", "..") for part in rel.parts
            ):
                return _empty("runtime_file_manifest_invalid")
            seen.add(item)
            path = root / rel
            if not path.is_file() or path.is_symlink():
                return _empty("runtime_owned_file_missing_or_unsafe")
            digest.update(
                item.encode()
                + b"\0"
                + hashlib.sha256(path.read_bytes()).hexdigest().encode()
                + b"\n"
            )
    except OSError:
        return _empty("runtime_owned_file_unreadable")

    actual = digest.hexdigest()
    if actual != expected:
        return {
            "contract": contract,
            "tree_sha256": actual,
            "verified": False,
            "reason": "runtime_tree_digest_mismatch",
        }
    return {
        "contract": contract,
        "tree_sha256": actual,
        "verified": True,
        "reason": None,
    }
