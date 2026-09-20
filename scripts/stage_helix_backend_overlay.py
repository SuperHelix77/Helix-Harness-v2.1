#!/usr/bin/env python3
"""Stage the complete Helix backend runtime without upstream install-integrity inputs."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "studio" / "backend"
DEST = ROOT / "studio" / "src-tauri" / "artifacts" / "helix-backend"
EXCLUDED_DIRS = {"tests", "requirements", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

def contract() -> str:
    for line in (SOURCE / "state" / "tool_policy.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("HELIX_HARNESS_BACKEND_CONTRACT = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("HELIX_HARNESS_BACKEND_CONTRACT is missing")

def runtime_files() -> list[str]:
    files: list[str] = []
    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name == ".DS_Store":
            continue
        files.append(rel.as_posix())
    return sorted(files)

def main() -> None:
    files = runtime_files()
    if DEST.exists(): shutil.rmtree(DEST)
    digest = hashlib.sha256()
    for rel in files:
        src = SOURCE / rel
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest.update(rel.encode() + b"\0" + hashlib.sha256(src.read_bytes()).hexdigest().encode() + b"\n")
    meta = {"schema_version":"helix.backend-overlay.v1","contract":contract(),"file_count":len(files),"tree_sha256":digest.hexdigest(),"files":files,"excluded_dirs":sorted(EXCLUDED_DIRS)}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "HELIX_BACKEND_MANIFEST.json").write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(f"staged {meta['contract']}: {len(files)} runtime files, sha256={meta['tree_sha256']}")
if __name__ == "__main__": main()
