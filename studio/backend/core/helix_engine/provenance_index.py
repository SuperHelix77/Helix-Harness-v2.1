# SPDX-License-Identifier: AGPL-3.0-only
"""Durable correlation index for the append-only Helix provenance ledgers.

JSONL remains the human-auditable source log. This SQLite index gives Execution
Graph exact historical lookups without scanning an ever-growing ledger tail.
Indexing is post-effect and fail-open at the ledger call site, so it never enters
the inference hot path or becomes learning authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_DB_INIT_LOCK = threading.Lock()


def _path() -> Path:
    override = os.environ.get("HELIX_PROVENANCE_INDEX_DB", "").strip()
    if override:
        return Path(override).expanduser()
    ledger_root = os.environ.get("HELIX_ENGINE_LEDGER_ROOT", "").strip()
    if ledger_root:
        return Path(ledger_root).expanduser() / "provenance-index.sqlite3"
    from utils.paths import account_path

    return Path(account_path("learning/helix-engine/provenance-index.sqlite3"))


def _open() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    with _DB_INIT_LOCK:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance_records (
                kind TEXT NOT NULL,
                record_key TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                thread_id TEXT NOT NULL DEFAULT '',
                turn_id TEXT NOT NULL DEFAULT '',
                run_id TEXT NOT NULL DEFAULT '',
                trajectory_id TEXT NOT NULL DEFAULT '',
                created_at_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (kind, record_key)
            )
            """
        )
        for name, column in (
            ("session", "session_id"),
            ("thread", "thread_id"),
            ("turn", "turn_id"),
            ("run", "run_id"),
            ("trajectory", "trajectory_id"),
        ):
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS provenance_{name}_idx "
                f"ON provenance_records(kind, {column}, created_at_ms)"
            )
    return conn


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _record_key(kind: str, record: dict[str, Any], payload: str) -> str:
    identity = {
        "kind": kind,
        "checkpoint_id": _text(record.get("checkpoint_id"), 240),
        "decision_id": _text(record.get("decision_id"), 240),
        "receipt_id": _text(record.get("receipt_id"), 240),
        "session_id": _text(record.get("session_id"), 500),
        "thread_id": _text(
            record.get("thread_id") or record.get("source_thread_id"),
            240,
        ),
        "turn_id": _text(record.get("turn_id"), 240),
        "run_id": _text(record.get("run_id"), 240),
        "trajectory_id": _text(
            record.get("trajectory_id") or record.get("source_trajectory_id"),
            240,
        ),
    }
    stable = any(value for key, value in identity.items() if key != "kind")
    material = (
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if stable
        else payload
    )
    # Multiple records may legitimately share a trajectory (e.g. decision
    # outcomes). Include the payload when there is no uniquely identifying
    # receipt/checkpoint/decision/turn key.
    unique_named = any(
        identity[name]
        for name in ("checkpoint_id", "decision_id", "receipt_id", "turn_id", "run_id")
    )
    if stable and not unique_named:
        material += "\0" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def index_record(kind: str, record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    normalized_kind = _text(kind, 120)
    if not normalized_kind:
        return False
    payload = _canonical(record)
    created = record.get("created_at_ms")
    try:
        created_at_ms = int(created or 0)
    except (TypeError, ValueError, OverflowError):
        created_at_ms = 0
    if created_at_ms <= 0:
        created_at_ms = int(time.time() * 1000)
    values = {
        "session_id": _text(record.get("session_id"), 500),
        "thread_id": _text(
            record.get("thread_id") or record.get("source_thread_id"),
            240,
        ),
        "turn_id": _text(record.get("turn_id"), 240),
        "run_id": _text(record.get("run_id"), 240),
        "trajectory_id": _text(
            record.get("trajectory_id") or record.get("source_trajectory_id"),
            240,
        ),
    }
    key = _record_key(normalized_kind, record, payload)
    with _open() as conn:
        conn.execute(
            """
            INSERT INTO provenance_records (
                kind, record_key, session_id, thread_id, turn_id, run_id,
                trajectory_id, created_at_ms, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, record_key) DO UPDATE SET
                session_id=excluded.session_id,
                thread_id=excluded.thread_id,
                turn_id=excluded.turn_id,
                run_id=excluded.run_id,
                trajectory_id=excluded.trajectory_id,
                created_at_ms=MAX(provenance_records.created_at_ms, excluded.created_at_ms),
                payload_json=excluded.payload_json
            """,
            (
                normalized_kind,
                key,
                values["session_id"],
                values["thread_id"],
                values["turn_id"],
                values["run_id"],
                values["trajectory_id"],
                created_at_ms,
                payload,
            ),
        )
    return True


def query_records(
    kind: str,
    *,
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
    run_id: str | None = None,
    trajectory_id: str | None = None,
    limit: int = 384,
) -> list[dict[str, Any]]:
    clauses = ["kind=?"]
    args: list[Any] = [_text(kind, 120)]
    for column, raw, cap in (
        ("session_id", session_id, 500),
        ("thread_id", thread_id, 240),
        ("turn_id", turn_id, 240),
        ("run_id", run_id, 240),
        ("trajectory_id", trajectory_id, 240),
    ):
        value = _text(raw, cap)
        if value:
            clauses.append(f"{column}=?")
            args.append(value)
    cap = max(1, min(int(limit), 2_000))
    args.append(cap)
    with _open() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM provenance_records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at_ms DESC, rowid DESC LIMIT ?",
            args,
        ).fetchall()
    decoded: list[dict[str, Any]] = []
    for row in reversed(rows):
        try:
            value = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            decoded.append(value)
    return decoded
