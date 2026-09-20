# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded append-only provenance ledger. All writes are optional/fail-open."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_MAX_LINE_BYTES = 256 * 1024
_MAX_RECENT_READ_BYTES = 8 * 1024 * 1024


def _root() -> Path:
    override = os.environ.get("HELIX_ENGINE_LEDGER_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
    if current_test:
        token = hashlib.sha256(current_test.encode("utf-8")).hexdigest()[:16]
        return Path(os.environ.get("TMPDIR", "/tmp")) / "helix-engine-pytest" / token
    try:
        from utils.paths import account_path

        return Path(account_path("learning/helix-engine"))
    except Exception:
        return Path.home() / ".unsloth" / "studio" / "learning" / "helix-engine"


def _path(kind: str) -> Path:
    safe = "".join(ch for ch in kind.lower() if ch.isalnum() or ch in {"-", "_"}) or "records"
    return _root() / f"{safe}.jsonl"


def append_record(kind: str, record: dict[str, Any]) -> bool:
    try:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        encoded = payload.encode("utf-8")
        if len(encoded) > _MAX_LINE_BYTES:
            payload = json.dumps({"schema_version": record.get("schema_version"), "truncated": True, "kind": kind})
        path = _path(kind)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        try:
            from .provenance_index import index_record

            index_record(kind, record)
        except Exception:
            # The JSONL ledger remains authoritative and indexing is rebuildable.
            pass
        return True
    except Exception:
        return False


def recent_records(kind: str, limit: int = 200) -> list[dict[str, Any]]:
    try:
        path = _path(kind)
        if not path.is_file():
            return []
        row_limit = max(1, min(limit, 2_000))
        # This ledger is append-only and can live for the lifetime of the app.
        # Reading the whole file before slicing makes a "recent" query scale with
        # total history and can spike memory on long-running autonomous sessions.
        # Writes are individually capped, so a fixed tail window is enough to
        # recover a useful bounded suffix while failing safely on older history.
        with _LOCK:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                start = max(0, size - _MAX_RECENT_READ_BYTES)
                handle.seek(start)
                raw = handle.read(_MAX_RECENT_READ_BYTES)
        if start > 0:
            newline = raw.find(b"\n")
            if newline < 0:
                return []
            raw = raw[newline + 1 :]
        lines = raw.decode("utf-8", errors="replace").splitlines()[-row_limit:]
        rows = []
        for line in lines:
            try:
                value = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
    except Exception:
        return []


def pattern_fingerprint(labels: list[str]) -> str:
    normalized = "|".join(sorted(set(item.strip().lower() for item in labels if item.strip())))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else "none"


def recurrence_sources(fingerprint: str, trajectory_id: str) -> tuple[int, list[str]]:
    """Return distinct trajectory count + provenance for one objective behavior signature.

    Re-analysis of the same trajectory is idempotent and cannot manufacture recurrence.
    """
    if not fingerprint or fingerprint == "none" or not trajectory_id:
        return 1, [trajectory_id] if trajectory_id else []
    prior = recent_records("patterns", limit=500)
    matching_ids = {
        str(item.get("trajectory_id") or "")
        for item in prior
        if item.get("fingerprint") == fingerprint and item.get("trajectory_id")
    }
    if trajectory_id not in matching_ids:
        append_record(
            "patterns",
            {
                "schema_version": "helix.pattern.v1",
                "fingerprint": fingerprint,
                "trajectory_id": trajectory_id,
            },
        )
        matching_ids.add(trajectory_id)
    ordered = sorted(matching_ids)
    return max(1, len(ordered)), ordered


def recurrence_count(fingerprint: str, trajectory_id: str) -> int:
    return recurrence_sources(fingerprint, trajectory_id)[0]


def storage_bytes() -> int:
    try:
        return sum(path.stat().st_size for path in _root().glob("*.jsonl") if path.is_file())
    except Exception:
        return 0
