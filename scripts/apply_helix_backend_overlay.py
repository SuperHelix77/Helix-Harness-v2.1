#!/usr/bin/env python3
"""Atomically apply a partial Helix backend overlay onto an installed Studio backend."""
from __future__ import annotations
import hashlib, importlib.metadata, importlib.util, json, os, shutil, sys, uuid
from pathlib import Path

EXPECTED = "helix.adaptive.backend.v1"
MEMORY_REQUIREMENT = "mem0ai==2.0.20; qdrant-client==1.19.1"
RUNTIME_MANIFEST = ".helix_backend_manifest.json"


def _manifest_files(overlay: Path, meta: dict) -> list[str]:
    files = meta.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Helix backend manifest has no files")
    if meta.get("file_count") != len(files):
        raise RuntimeError(
            f"Helix backend manifest file_count mismatch: {meta.get('file_count')!r} != {len(files)}"
        )

    normalized: list[str] = []
    digest = hashlib.sha256()
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"invalid Helix backend manifest path: {raw!r}")
        rel_path = Path(raw)
        if rel_path.is_absolute() or any(part in ("", ".", "..") for part in rel_path.parts):
            raise RuntimeError(f"unsafe Helix backend manifest path: {raw!r}")
        rel = rel_path.as_posix()
        if rel != raw or rel in seen:
            raise RuntimeError(f"non-canonical Helix backend manifest path: {raw!r}")
        seen.add(rel)
        src = overlay / rel_path
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f"overlay file missing or unsafe: {rel}")
        digest.update(rel.encode() + b"\0" + hashlib.sha256(src.read_bytes()).hexdigest().encode() + b"\n")
        normalized.append(rel)

    expected_digest = meta.get("tree_sha256")
    actual_digest = digest.hexdigest()
    if expected_digest != actual_digest:
        raise RuntimeError(
            f"Helix backend overlay digest mismatch: {actual_digest} != {expected_digest!r}"
        )
    return normalized


def _installed_tree_digest(current: Path, files: list[str]) -> str | None:
    """Hash the installed overlay-owned files with the manifest's exact scheme.

    The managed Studio backend contains additional upstream files that are not
    owned by Helix Harness, so only manifest entries participate.  A missing,
    non-regular, or symlinked entry is stale rather than an error: the atomic
    overlay application below will repair it.
    """
    digest = hashlib.sha256()
    for rel in files:
        path = current / Path(rel)
        if not path.is_file() or path.is_symlink():
            return None
        digest.update(
            rel.encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode()
            + b"\n"
        )
    return digest.hexdigest()


def _write_runtime_manifest(current: Path, meta: dict) -> None:
    """Publish the verified bundle identity inside the installed backend atomically."""
    payload = {
        "schema_version": meta.get("schema_version"),
        "contract": meta.get("contract"),
        "file_count": meta.get("file_count"),
        "tree_sha256": meta.get("tree_sha256"),
        "files": meta.get("files"),
    }
    temporary = current / f".{RUNTIME_MANIFEST}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, current / RUNTIME_MANIFEST)
    try:
        descriptor = os.open(current, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _validate_installed_backend(current: Path) -> None:
    sys.path.insert(0, str(current))
    try:
        from state.tool_policy import HELIX_HARNESS_BACKEND_CONTRACT, require_tool_access
        from core.helix_engine.controller import run_closed_loop

        if HELIX_HARNESS_BACKEND_CONTRACT != EXPECTED:
            raise RuntimeError(
                f"installed Helix backend contract mismatch: {HELIX_HARNESS_BACKEND_CONTRACT!r}"
            )
        if not callable(require_tool_access) or not callable(run_closed_loop):
            raise RuntimeError("installed Helix backend contract callables are unavailable")
    finally:
        try:
            sys.path.remove(str(current))
        except ValueError:
            pass


def _restore_backup(current: Path, backup: Path, studio_root: Path) -> None:
    failed = studio_root / f".backend-helix-failed-{uuid.uuid4().hex}"
    os.replace(current, failed)
    try:
        os.replace(backup, current)
    except BaseException:
        os.replace(failed, current)
        raise
    shutil.rmtree(failed, ignore_errors=True)


def _activate_stage(
    current: Path,
    stage: Path,
    backup: Path,
    studio_root: Path,
    validate=_validate_installed_backend,
) -> None:
    os.replace(current, backup)
    try:
        os.replace(stage, current)
    except BaseException:
        os.replace(backup, current)
        raise
    try:
        validate(current)
    except BaseException:
        _restore_backup(current, backup, studio_root)
        raise


def _memory_runtime_ready() -> bool:
    try:
        from packaging.version import Version

        version = Version(importlib.metadata.version("mem0ai"))
        qdrant_version = Version(importlib.metadata.version("qdrant-client"))
        if version != Version("2.0.20") or qdrant_version != Version("1.19.1"):
            return False
        import qdrant_client  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_optional_memory_runtime() -> bool:
    """Verify optional Mem0/Qdrant support without mutating the runtime at launch.

    Release startup must be reproducible and offline-safe. Dependencies are
    installed by the managed installer/update path, never selected from PyPI by
    an ordinary app launch. If the exact supported runtime is unavailable, Helix
    keeps using its bounded local graph fallback.
    """
    ready = _memory_runtime_ready()
    if not ready:
        print(
            f"[helix] optional memory runtime unavailable ({MEMORY_REQUIREMENT}); "
            "bounded local graph remains active; repair/update the managed runtime "
            "to install the packaged dependency set",
            file=sys.stderr,
        )
    return ready


def main() -> None:
    if len(sys.argv) != 2: raise SystemExit("usage: apply_helix_backend_overlay.py <overlay-dir>")
    overlay = Path(sys.argv[1]).resolve()
    meta_path = overlay / "HELIX_BACKEND_MANIFEST.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("contract") != EXPECTED: raise SystemExit(f"unexpected Helix backend contract: {meta.get('contract')!r}")
    files = _manifest_files(overlay, meta)
    spec = importlib.util.find_spec("studio")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if not locations: raise SystemExit("installed studio package not found")
    studio_root = Path(locations[0]).resolve()
    current = studio_root / "backend"
    if not current.is_dir(): raise SystemExit(f"installed backend not found: {current}")
    expected_digest = str(meta.get("tree_sha256") or "")
    if expected_digest and _installed_tree_digest(current, files) == expected_digest:
        _write_runtime_manifest(current, meta)
        _validate_installed_backend(current)
        print(
            f"already current {EXPECTED} ({meta.get('file_count')} overrides); "
            f"mem0={'ready' if _memory_runtime_ready() else 'fallback'}"
        )
        return
    stage = studio_root / f".backend-helix-{uuid.uuid4().hex}"
    backup = studio_root / f".backend-pre-helix-{uuid.uuid4().hex}"
    shutil.copytree(current, stage, symlinks=False)
    try:
        for rel in files:
            src, dst = overlay / rel, stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        _write_runtime_manifest(stage, meta)
        _activate_stage(current, stage, backup, studio_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)
    memory_ready = _ensure_optional_memory_runtime()
    print(
        f"applied {EXPECTED} ({meta.get('file_count')} overrides); "
        f"mem0={'ready' if memory_ready else 'fallback'}"
    )
if __name__ == "__main__": main()
