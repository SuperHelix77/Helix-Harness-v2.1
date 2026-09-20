from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/apply_helix_backend_overlay.py"


def load_overlay_module():
    spec = importlib.util.spec_from_file_location("helix_backend_overlay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_for(files: dict[str, bytes]) -> dict:
    digest = hashlib.sha256()
    for rel, content in sorted(files.items()):
        digest.update(rel.encode() + b"\0" + hashlib.sha256(content).hexdigest().encode() + b"\n")
    return {
        "contract": "helix.adaptive.backend.v1",
        "file_count": len(files),
        "files": sorted(files),
        "tree_sha256": digest.hexdigest(),
    }


def test_overlay_manifest_rejects_tampering_and_path_escape(tmp_path: Path) -> None:
    overlay_module = load_overlay_module()
    files = {"state/tool_policy.py": b"contract\n"}
    overlay = tmp_path / "overlay"
    (overlay / "state").mkdir(parents=True)
    (overlay / "state/tool_policy.py").write_bytes(files["state/tool_policy.py"])
    meta = manifest_for(files)

    assert overlay_module._manifest_files(overlay, meta) == ["state/tool_policy.py"]

    (overlay / "state/tool_policy.py").write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        overlay_module._manifest_files(overlay, meta)

    escaped = manifest_for(files)
    escaped["files"] = ["../outside.py"]
    with pytest.raises(RuntimeError, match="unsafe.*path"):
        overlay_module._manifest_files(overlay, escaped)


def test_failed_post_swap_validation_restores_previous_backend(tmp_path: Path) -> None:
    overlay_module = load_overlay_module()
    studio_root = tmp_path / "studio"
    current = studio_root / "backend"
    stage = studio_root / ".backend-helix-stage-test"
    backup = studio_root / ".backend-pre-helix-test"
    current.mkdir(parents=True)
    stage.mkdir(parents=True)
    (current / "marker.txt").write_text("previous", encoding="utf-8")
    (stage / "marker.txt").write_text("candidate", encoding="utf-8")

    def reject_candidate(candidate: Path) -> None:
        assert (candidate / "marker.txt").read_text(encoding="utf-8") == "candidate"
        raise RuntimeError("candidate contract failed")

    with pytest.raises(RuntimeError, match="candidate contract failed"):
        overlay_module._activate_stage(
            current,
            stage,
            backup,
            studio_root,
            validate=reject_candidate,
        )

    assert (current / "marker.txt").read_text(encoding="utf-8") == "previous"
    assert not backup.exists()
    assert not stage.exists()
    assert not list(studio_root.glob(".backend-helix-failed-*"))


def test_installed_tree_digest_matches_only_manifest_owned_files(tmp_path: Path) -> None:
    overlay_module = load_overlay_module()
    files = {
        "core/helix_engine/adaptive_checkpoint.py": b"adaptive\n",
        "routes/helix_engine.py": b"routes\n",
    }
    meta = manifest_for(files)
    current = tmp_path / "backend"
    for rel, content in files.items():
        path = current / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    # Upstream-managed files outside the overlay manifest do not invalidate the
    # Helix contract fast path.
    extra = current / "upstream_only.py"
    extra.write_text("unchanged upstream runtime\n", encoding="utf-8")
    assert (
        overlay_module._installed_tree_digest(current, meta["files"])
        == meta["tree_sha256"]
    )

    # A manifest-owned mutation is stale and must trigger the atomic repair.
    (current / "routes/helix_engine.py").write_bytes(b"stale routes\n")
    assert (
        overlay_module._installed_tree_digest(current, meta["files"])
        != meta["tree_sha256"]
    )

    # Missing or symlinked owned files are stale rather than trusted.
    (current / "routes/helix_engine.py").unlink()
    assert overlay_module._installed_tree_digest(current, meta["files"]) is None
    (current / "routes/helix_engine.py").symlink_to(extra)
    assert overlay_module._installed_tree_digest(current, meta["files"]) is None


def test_runtime_manifest_is_written_atomically_from_verified_bundle_metadata(
    tmp_path: Path,
) -> None:
    overlay_module = load_overlay_module()
    current = tmp_path / "backend"
    current.mkdir()
    files = {"state/tool_policy.py": b"policy\n"}
    meta = manifest_for(files)
    meta["schema_version"] = "helix.backend-overlay.v1"

    overlay_module._write_runtime_manifest(current, meta)

    written = json.loads(
        (current / overlay_module.RUNTIME_MANIFEST).read_text(encoding="utf-8")
    )
    assert written == {
        "schema_version": "helix.backend-overlay.v1",
        "contract": meta["contract"],
        "file_count": meta["file_count"],
        "tree_sha256": meta["tree_sha256"],
        "files": meta["files"],
    }
    assert not list(current.glob("..helix_backend_manifest.json.*.tmp"))


def test_normal_overlay_application_never_installs_packages_from_network() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pip install" not in source
    assert "subprocess.run" not in source
