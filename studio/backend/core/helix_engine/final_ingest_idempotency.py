# SPDX-License-Identifier: AGPL-3.0-only
"""Durable idempotency for completed Helix turn ingest.

Final ingest performs several externally visible learning side effects.  A durable
chat/runtime reload may replay the request after losing the original HTTP response,
so keyed requests must acquire a persistent claim before any of those effects run.
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
from uuid import uuid4


_PROCESS_INSTANCE_ID = uuid4().hex
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_KEYS: set[str] = set()
_DB_INIT_LOCK = threading.Lock()
_REPLAY_WAIT_SECONDS = 20.0
_REPLAY_POLL_SECONDS = 0.02


class FinalIngestIdempotencyConflict(RuntimeError):
    """The same key/scope was reused for a different request payload."""


class FinalIngestReplayUnavailable(RuntimeError):
    """A prior keyed ingest cannot be safely replayed without repeating effects."""


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
    override = os.environ.get("HELIX_FINAL_INGEST_DB", "").strip()
    if override:
        return Path(override).expanduser()
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
    if current_test:
        token = hashlib.sha256(
            f"{current_test}\0{_account_id()}\0{os.getpid()}".encode("utf-8")
        ).hexdigest()[:16]
        return (
            Path(os.environ.get("TMPDIR", "/tmp"))
            / "helix-engine-pytest"
            / token
            / "final-ingest-idempotency.sqlite3"
        )
    from utils.paths import account_path

    return Path(account_path("learning/helix-engine/final-ingest-idempotency.sqlite3"))


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
            CREATE TABLE IF NOT EXISTS final_ingest_receipts (
            dedupe_key TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL,
            account_id TEXT NOT NULL,
            owner_subject TEXT NOT NULL,
            session_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            claim_pid INTEGER NOT NULL,
            claim_instance TEXT NOT NULL,
            capture_json TEXT,
            receipt_json TEXT,
            failure_text TEXT NOT NULL DEFAULT '',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
            )
            """
        )
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(final_ingest_receipts)").fetchall()
        }
        if "capture_json" not in columns:
            conn.execute("ALTER TABLE final_ingest_receipts ADD COLUMN capture_json TEXT")
    return conn


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("idempotency_key", None)
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _identity(
    *,
    idempotency_key: str,
    session_id: str,
    thread_id: str | None,
    turn_id: str | None,
    subject: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    account_id = _account_id()
    key = _text(idempotency_key, 240)
    session_scope = str(session_id or "").strip()
    thread_scope = str(thread_id or "").strip()
    turn_scope = str(turn_id or "").strip()
    scope = _canonical_json(
        {
            "account_id": account_id,
            "session_id": session_scope,
            "thread_id": thread_scope,
            "turn_id": turn_scope,
            "idempotency_key": key,
        }
    )
    return {
        "dedupe_key": hashlib.sha256(scope.encode("utf-8")).hexdigest(),
        "idempotency_key": key,
        "account_id": account_id,
        "owner_subject": _text(subject, 240),
        "session_id": _text(session_scope, 500),
        "thread_id": _text(thread_scope, 240),
        "turn_id": _text(turn_scope, 240),
        "payload_sha256": _payload_sha256(payload),
    }


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _decode_receipt(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") != "completed":
        return None
    try:
        value = json.loads(str(row.get("receipt_json") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _decode_capture(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("capture_json")
    if not raw:
        return None
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _with_stored_capture(
    identity: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(identity)
    capture = _decode_capture(row)
    if capture is not None:
        merged["capture_snapshot"] = capture
    return merged


def _assert_payload_matches(row: dict[str, Any], identity: dict[str, Any]) -> None:
    if str(row.get("payload_sha256") or "") == identity["payload_sha256"]:
        return
    raise FinalIngestIdempotencyConflict(
        "final ingest idempotency key was reused in the same logical turn scope "
        "with a different payload"
    )


def _active_locally(dedupe_key: str) -> bool:
    with _ACTIVE_LOCK:
        return dedupe_key in _ACTIVE_KEYS


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
        return instance != _PROCESS_INSTANCE_ID or not _active_locally(
            str(row.get("dedupe_key") or "")
        )
    return not _pid_alive(pid)


def _load_row(dedupe_key: str) -> dict[str, Any] | None:
    conn = _open_db()
    try:
        row = conn.execute(
            "SELECT * FROM final_ingest_receipts WHERE dedupe_key=?",
            (dedupe_key,),
        ).fetchone()
        return _row_dict(row)
    finally:
        conn.close()


def _interrupt_orphan(row: dict[str, Any]) -> None:
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM final_ingest_receipts WHERE dedupe_key=?",
            (row["dedupe_key"],),
        ).fetchone()
        if current is not None and current["status"] == "running":
            conn.execute(
                """
                UPDATE final_ingest_receipts
                   SET status='interrupted',
                       failure_text='owner disappeared before durable receipt completion',
                       updated_at_ms=?
                 WHERE dedupe_key=? AND status='running'
                """,
                (int(time.time() * 1000), row["dedupe_key"]),
            )
        conn.commit()
    finally:
        conn.close()


def _reclaim_interrupted(identity: dict[str, Any]) -> dict[str, Any]:
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM final_ingest_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise FinalIngestReplayUnavailable("final ingest durable claim disappeared")
        current = dict(row)
        _assert_payload_matches(current, identity)
        if _decode_receipt(current) is not None:
            conn.rollback()
            raise FinalIngestReplayUnavailable(
                "final ingest completed while interrupted claim was being reclaimed"
            )
        if current.get("status") not in {"failed", "interrupted"}:
            conn.rollback()
            raise FinalIngestReplayUnavailable("final ingest claim is not safely reclaimable")
        conn.execute(
            """UPDATE final_ingest_receipts
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
        with _ACTIVE_LOCK:
            _ACTIVE_KEYS.add(identity["dedupe_key"])
        return _with_stored_capture(identity, current)
    finally:
        conn.close()


def _existing_or_wait(
    row: dict[str, Any],
    identity: dict[str, Any],
    *,
    replay_safe: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    _assert_payload_matches(row, identity)
    receipt = _decode_receipt(row)
    if receipt is not None:
        return None, receipt
    if row.get("status") in {"failed", "interrupted"}:
        if replay_safe:
            return _reclaim_interrupted(_with_stored_capture(identity, row)), None
        reason = _text(row.get("failure_text"), 800) or "prior final ingest did not complete"
        raise FinalIngestReplayUnavailable(reason)

    deadline = time.monotonic() + _REPLAY_WAIT_SECONDS
    current = row
    while current.get("status") == "running" and not _orphaned(current):
        if time.monotonic() >= deadline:
            raise FinalIngestReplayUnavailable(
                "duplicate final ingest is still owned by a live request"
            )
        time.sleep(_REPLAY_POLL_SECONDS)
        current = _load_row(str(row["dedupe_key"])) or current
        _assert_payload_matches(current, identity)
        receipt = _decode_receipt(current)
        if receipt is not None:
            return None, receipt

    receipt = _decode_receipt(current)
    if receipt is not None:
        return None, receipt
    if current.get("status") == "running" and _orphaned(current):
        _interrupt_orphan(current)
        if replay_safe:
            return _reclaim_interrupted(_with_stored_capture(identity, current)), None
        raise FinalIngestReplayUnavailable(
            "prior final ingest was interrupted after acquiring its durable claim; "
            "replaying side effects is unsafe"
        )
    if current.get("status") in {"failed", "interrupted"} and replay_safe:
        return _reclaim_interrupted(_with_stored_capture(identity, current)), None
    reason = _text(current.get("failure_text"), 800) or "prior final ingest did not complete"
    raise FinalIngestReplayUnavailable(reason)


def claim_final_ingest(
    *,
    idempotency_key: str,
    session_id: str,
    thread_id: str | None,
    turn_id: str | None,
    subject: str,
    payload: dict[str, Any],
    replay_safe: bool = False,
    capture_snapshot: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Acquire the unique side-effect claim or return the prior completed receipt."""

    identity = _identity(
        idempotency_key=idempotency_key,
        session_id=session_id,
        thread_id=thread_id,
        turn_id=turn_id,
        subject=subject,
        payload=payload,
    )
    now_ms = int(time.time() * 1000)
    capture_json = (
        _canonical_json(capture_snapshot)
        if isinstance(capture_snapshot, dict)
        else None
    )
    conn = _open_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM final_ingest_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if row is not None:
            conn.commit()
            return _existing_or_wait(
                dict(row),
                identity,
                replay_safe=replay_safe,
            )
        conn.execute(
            """
            INSERT INTO final_ingest_receipts (
                dedupe_key, idempotency_key, account_id, owner_subject,
                session_id, thread_id, turn_id, payload_sha256,
                status, claim_pid, claim_instance, capture_json, receipt_json, failure_text,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, NULL, '', ?, ?)
            """,
            (
                identity["dedupe_key"],
                identity["idempotency_key"],
                identity["account_id"],
                identity["owner_subject"],
                identity["session_id"],
                identity["thread_id"],
                identity["turn_id"],
                identity["payload_sha256"],
                os.getpid(),
                _PROCESS_INSTANCE_ID,
                capture_json,
                now_ms,
                now_ms,
            ),
        )
        with _ACTIVE_LOCK:
            _ACTIVE_KEYS.add(identity["dedupe_key"])
        try:
            conn.commit()
        except BaseException:
            with _ACTIVE_LOCK:
                _ACTIVE_KEYS.discard(identity["dedupe_key"])
            raise
    finally:
        conn.close()
    if isinstance(capture_snapshot, dict):
        identity["capture_snapshot"] = capture_snapshot
    return identity, None


def complete_final_ingest(
    identity: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    """Persist the exact response before releasing the unique side-effect claim."""

    raw = _canonical_json(receipt)
    canonical = json.loads(raw)
    try:
        conn = _open_db()
        try:
            cursor = conn.execute(
                """
                UPDATE final_ingest_receipts
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
                "SELECT * FROM final_ingest_receipts WHERE dedupe_key=?",
                (identity["dedupe_key"],),
            ).fetchone()
            existing = _decode_receipt(dict(row)) if row is not None else None
            if existing is not None:
                return existing
            raise FinalIngestReplayUnavailable(
                "final ingest durable receipt ownership was lost before completion"
            )
        finally:
            conn.close()
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_KEYS.discard(identity["dedupe_key"])


def fail_final_ingest(identity: dict[str, Any], exc: BaseException) -> None:
    """Persist a terminal failure so a retry never repeats partially applied effects."""

    try:
        conn = _open_db()
        try:
            conn.execute(
                """
                UPDATE final_ingest_receipts
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
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_KEYS.discard(identity["dedupe_key"])
