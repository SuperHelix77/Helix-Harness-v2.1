# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json


def _manifest(files: dict[str, bytes]) -> dict:
    digest = hashlib.sha256()
    for rel, content in sorted(files.items()):
        digest.update(
            rel.encode()
            + b"\0"
            + hashlib.sha256(content).hexdigest().encode()
            + b"\n"
        )
    return {
        "schema_version": "helix.backend-overlay.v1",
        "contract": "helix.adaptive.backend.v1",
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
        "files": sorted(files),
    }


def test_runtime_identity_verifies_exact_owned_tree(tmp_path, monkeypatch):
    from state import helix_build_identity

    root = tmp_path / "backend"
    files = {
        "state/tool_policy.py": b"policy\n",
        "core/helix_engine/controller.py": b"controller\n",
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / ".helix_backend_manifest.json").write_text(
        json.dumps(_manifest(files)),
        encoding="utf-8",
    )
    monkeypatch.setattr(helix_build_identity, "_backend_root", lambda: root)
    helix_build_identity.helix_backend_identity.cache_clear()
    try:
        identity = helix_build_identity.helix_backend_identity()
        assert identity["verified"] is True
        assert identity["tree_sha256"] == _manifest(files)["tree_sha256"]
    finally:
        helix_build_identity.helix_backend_identity.cache_clear()


def test_runtime_identity_detects_owned_file_mutation(tmp_path, monkeypatch):
    from state import helix_build_identity

    root = tmp_path / "backend"
    files = {"state/tool_policy.py": b"policy\n"}
    path = root / "state/tool_policy.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(files["state/tool_policy.py"])
    (root / ".helix_backend_manifest.json").write_text(
        json.dumps(_manifest(files)),
        encoding="utf-8",
    )
    path.write_bytes(b"tampered\n")
    monkeypatch.setattr(helix_build_identity, "_backend_root", lambda: root)
    helix_build_identity.helix_backend_identity.cache_clear()
    try:
        identity = helix_build_identity.helix_backend_identity()
        assert identity["verified"] is False
        assert identity["reason"] == "runtime_tree_digest_mismatch"
    finally:
        helix_build_identity.helix_backend_identity.cache_clear()
