# SPDX-License-Identifier: AGPL-3.0-only
"""Durable idempotency for completed-turn Mem0 experience writes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


_PROCESS_INSTANCE_ID = uuid4().hex
_REPLAY_WAIT_SECONDS = 20.0
_REPLAY_POLL_SECONDS = 0.02
_DB_INIT_LOCK = threading.Lock()


class MemoryExperienceIdempotencyConflict(RuntimeError):
    """The same account/thread/key was reused for a different experience."""


class MemoryExperienceReplayUnavailable(RuntimeError):
    """A prior keyed write cannot be retried without risking duplicate effects."""


def _account_id() -> str:
    try:
        from utils.account_context import current_account_id

        value = str(current_account_id() or "").strip()
        if value:
            return value
    except Exception:
        pass
    return "owner"


def _db_path() -> Path:
    override = os.environ.get("HELIX_MEMORY_EXPERIENCE_DB", "").strip()
    if override:
        return Path(override).expanduser()
    from utils.paths import account_path

    return Path(account_path("learning/mem0/experience-idempotency.sqlite3"))


def _open_db() -> sqlite3.Connection:
    path = _db_path()
    try:
        from utils.paths import ensure_dir
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ensure_dir(path.parent)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    with _DB_INIT_LOCK:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_experience_receipts (
                dedupe_key TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                claim_pid INTEGER NOT NULL,
                claim_instance TEXT NOT NULL,
                receipt_json TEXT,
                failure_text TEXT NOT NULL DEFAULT '',
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            )
            """
        )
    return conn


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _identity(
    *,
    idempotency_key: str,
    thread_id: str | None,
    payload: dict[str, Any],
) -> dict[str, str]:
    account_id = _account_id()
    thread = str(thread_id or "")[:200]
    key = str(idempotency_key or "").strip()[:240]
    scope = _canonical_json(
        {
            "account_id": account_id,
            "thread_id": thread,
            "idempotency_key": key,
        }
    )
    return {
        "dedupe_key": hashlib.sha256(scope.encode("utf-8")).hexdigest(),
        "account_id": account_id,
        "thread_id": thread,
        "idempotency_key": key,
        "payload_sha256": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def _decode_receipt(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") != "completed":
        return None
    try:
        value = json.loads(str(row.get("receipt_json") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _assert_payload_matches(row: dict[str, Any], identity: dict[str, str]) -> None:
    if str(row.get("payload_sha256") or "") == identity["payload_sha256"]:
        return
    raise MemoryExperienceIdempotencyConflict(
        "memory experience idempotency key was reused in the same account/thread "
        "scope with a different payload"
    )


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError, OverflowError):
        return False
    if value <= 0:
        return False
    if value == os.getpid():
        return True
    try:
        os.kill(value, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _orphaned(row: dict[str, Any]) -> bool:
    pid = int(row.get("claim_pid") or 0)
    instance = str(row.get("claim_instance") or "")
    if pid == os.getpid():
        return instance != _PROCESS_INSTANCE_ID
    return not _pid_alive(pid)


def _load_row(dedupe_key: str) -> dict[str, Any] | None:
    conn = _open_db()
    try:
        row = conn.execute(
            "SELECT * FROM memory_experience_receipts WHERE dedupe_key=?",
            (dedupe_key,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _mark_interrupted(dedupe_key: str) -> None:
    conn = _open_db()
    try:
        conn.execute(
            """
            UPDATE memory_experience_receipts
               SET status='interrupted',
                   failure_text='writer disappeared before durable receipt completion',
                   updated_at_ms=?
             WHERE dedupe_key=? AND status='running'
            """,
            (int(time.time() * 1000), dedupe_key),
        )
    finally:
        conn.close()


def _persist_recovered_receipt(
    identity: dict[str, str],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    raw = _canonical_json(receipt)
    canonical = json.loads(raw)
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM memory_experience_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if current is None:
            conn.rollback()
            raise MemoryExperienceReplayUnavailable("memory experience durable claim disappeared")
        current_dict = dict(current)
        _assert_payload_matches(current_dict, identity)
        existing = _decode_receipt(current_dict)
        if existing is not None:
            conn.commit()
            return existing
        conn.execute(
            """UPDATE memory_experience_receipts
               SET status='completed', claim_pid=?, claim_instance=?,
                   receipt_json=?, failure_text='', updated_at_ms=?
             WHERE dedupe_key=? AND status IN ('running','failed','interrupted')""",
            (
                os.getpid(),
                _PROCESS_INSTANCE_ID,
                raw,
                int(time.time() * 1000),
                identity["dedupe_key"],
            ),
        )
        conn.commit()
        return canonical
    finally:
        conn.close()


def _reclaim_interrupted(identity: dict[str, str]) -> dict[str, str]:
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM memory_experience_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise MemoryExperienceReplayUnavailable("memory experience durable claim disappeared")
        current = dict(row)
        _assert_payload_matches(current, identity)
        receipt = _decode_receipt(current)
        if receipt is not None:
            conn.commit()
            raise MemoryExperienceReplayUnavailable(
                "memory experience completed while interrupted claim was being reclaimed"
            )
        if current.get("status") not in {"failed", "interrupted"}:
            conn.rollback()
            raise MemoryExperienceReplayUnavailable(
                "memory experience claim is not safely reclaimable"
            )
        conn.execute(
            """UPDATE memory_experience_receipts
               SET status='running', claim_pid=?, claim_instance=?,
                   failure_text='', updated_at_ms=?
             WHERE dedupe_key=? AND status IN ('failed','interrupted')""",
            (
                os.getpid(),
                _PROCESS_INSTANCE_ID,
                int(time.time() * 1000),
                identity["dedupe_key"],
            ),
        )
        conn.commit()
        return identity
    finally:
        conn.close()


def _existing_or_wait(
    row: dict[str, Any],
    identity: dict[str, str],
    recovery_probe: Callable[[], dict[str, Any] | None] | None,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    _assert_payload_matches(row, identity)
    receipt = _decode_receipt(row)
    if receipt is not None:
        return None, receipt
    if row.get("status") in {"failed", "interrupted"}:
        if recovery_probe is None:
            reason = str(row.get("failure_text") or "prior memory experience write did not complete")[:800]
            raise MemoryExperienceReplayUnavailable(reason)
        recovered = recovery_probe()
        if recovered is not None:
            return None, _persist_recovered_receipt(identity, recovered)
        return _reclaim_interrupted(identity), None

    deadline = time.monotonic() + _REPLAY_WAIT_SECONDS
    current = row
    while current.get("status") == "running":
        if _orphaned(current):
            _mark_interrupted(identity["dedupe_key"])
            if recovery_probe is None:
                raise MemoryExperienceReplayUnavailable(
                    "prior memory experience write was interrupted after its durable claim; "
                    "replaying the vector side effect is unsafe"
                )
            recovered = recovery_probe()
            if recovered is not None:
                return None, _persist_recovered_receipt(identity, recovered)
            return _reclaim_interrupted(identity), None
        if time.monotonic() >= deadline:
            raise MemoryExperienceReplayUnavailable(
                "duplicate memory experience write is still owned by a live request"
            )
        time.sleep(_REPLAY_POLL_SECONDS)
        current = _load_row(identity["dedupe_key"]) or current
        _assert_payload_matches(current, identity)
        receipt = _decode_receipt(current)
        if receipt is not None:
            return None, receipt

    receipt = _decode_receipt(current)
    if receipt is not None:
        return None, receipt
    if current.get("status") in {"failed", "interrupted"} and recovery_probe is not None:
        recovered = recovery_probe()
        if recovered is not None:
            return None, _persist_recovered_receipt(identity, recovered)
        return _reclaim_interrupted(identity), None
    reason = str(current.get("failure_text") or "prior memory experience write did not complete")[:800]
    raise MemoryExperienceReplayUnavailable(reason)


def _claim(
    *,
    idempotency_key: str,
    thread_id: str | None,
    payload: dict[str, Any],
    recovery_probe: Callable[[], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
    identity = _identity(
        idempotency_key=idempotency_key,
        thread_id=thread_id,
        payload=payload,
    )
    now_ms = int(time.time() * 1000)
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM memory_experience_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if row is not None:
            conn.commit()
            return _existing_or_wait(dict(row), identity, recovery_probe)
        conn.execute(
            """
            INSERT INTO memory_experience_receipts (
                dedupe_key, account_id, thread_id, idempotency_key, payload_sha256,
                status, claim_pid, claim_instance, receipt_json, failure_text,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, NULL, '', ?, ?)
            """,
            (
                identity["dedupe_key"],
                identity["account_id"],
                identity["thread_id"],
                identity["idempotency_key"],
                identity["payload_sha256"],
                os.getpid(),
                _PROCESS_INSTANCE_ID,
                now_ms,
                now_ms,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return identity, None


def _complete(identity: dict[str, str], receipt: dict[str, Any]) -> dict[str, Any]:
    raw = _canonical_json(receipt)
    canonical = json.loads(raw)
    conn = _open_db()
    try:
        cursor = conn.execute(
            """
            UPDATE memory_experience_receipts
               SET status='completed', receipt_json=?, failure_text='', updated_at_ms=?
             WHERE dedupe_key=? AND status='running' AND claim_instance=?
            """,
            (
                raw,
                int(time.time() * 1000),
                identity["dedupe_key"],
                _PROCESS_INSTANCE_ID,
            ),
        )
        if cursor.rowcount == 1:
            return canonical
        row = conn.execute(
            "SELECT * FROM memory_experience_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        existing = _decode_receipt(dict(row)) if row is not None else None
        if existing is not None:
            return existing
        raise MemoryExperienceReplayUnavailable(
            "memory experience durable receipt ownership was lost before completion"
        )
    finally:
        conn.close()


def _fail(identity: dict[str, str], exc: BaseException) -> None:
    try:
        conn = _open_db()
        try:
            conn.execute(
                """
                UPDATE memory_experience_receipts
                   SET status='failed', failure_text=?, updated_at_ms=?
                 WHERE dedupe_key=? AND status='running' AND claim_instance=?
                """,
                (
                    f"{type(exc).__name__}: {exc}"[:800],
                    int(time.time() * 1000),
                    identity["dedupe_key"],
                    _PROCESS_INSTANCE_ID,
                ),
            )
        finally:
            conn.close()
    except Exception:
        pass


def run_idempotent_memory_experience(
    *,
    idempotency_key: str,
    thread_id: str | None,
    payload: dict[str, Any],
    operation: Callable[[], dict[str, Any]],
    recovery_probe: Callable[[], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Run one keyed experience effect or return its exact prior durable receipt."""

    claim, replay = _claim(
        idempotency_key=idempotency_key,
        thread_id=thread_id,
        payload=payload,
        recovery_probe=recovery_probe,
    )
    if replay is not None:
        return replay
    assert claim is not None
    try:
        receipt = operation()
        return _complete(claim, receipt)
    except BaseException as exc:
        _fail(claim, exc)
        raise
