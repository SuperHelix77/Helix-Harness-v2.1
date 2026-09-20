# SPDX-License-Identifier: AGPL-3.0-only
"""Small local-model hub: list / select / load GGUF without companion files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

_COMPANION_MARKERS = ("mmproj", "imatrix", "dflash", "mtp-")


def list_local_gguf(root: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if not root.exists():
        return models
    for path in sorted(root.rglob("*.gguf")):
        name = path.name.lower()
        if any(marker in name for marker in _COMPANION_MARKERS):
            continue
        models.append({"name": path.name, "path": str(path), "bytes": path.stat().st_size})
    return models


def select_local_model(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return {"ok": False, "error": "GGUF file not found"}
    return {"ok": True, "path": str(path.resolve()), "name": path.name}


def download_local_model(source: str, dest_dir: Path) -> dict[str, Any]:
    """Copy an already-local GGUF into dest. Remote pulls are explicit and local-only here."""
    src = Path(source).expanduser()
    if not src.is_file() or src.suffix.lower() != ".gguf":
        return {"ok": False, "error": "source GGUF not found"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    selected = select_local_model(str(dest))
    selected["downloaded"] = True
    return selected


def load_local_model(raw_path: str, load_fn) -> dict[str, Any]:
    """Validate a local GGUF, then hand the resolved path to the inference load entry."""
    selected = select_local_model(raw_path)
    if not selected.get("ok"):
        return selected
    loaded = load_fn(selected["path"])
    if isinstance(loaded, dict) and loaded.get("ok") is False:
        return {
            "ok": False,
            "error": str(loaded.get("error") or "load failed"),
            "path": selected["path"],
            "loaded": False,
        }
    model = (
        loaded.get("model")
        if isinstance(loaded, dict)
        else getattr(loaded, "model", selected["name"])
    )
    status = (
        loaded.get("status")
        if isinstance(loaded, dict)
        else getattr(loaded, "status", "loaded")
    )
    return {
        "ok": True,
        "path": selected["path"],
        "name": selected["name"],
        "loaded": True,
        "model": model,
        "status": status,
    }
