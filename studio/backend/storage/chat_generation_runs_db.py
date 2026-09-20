# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transactional state and cursor events for durable Studio chat generations."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from uuid import uuid4
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from storage.studio_db import get_connection
from utils.account_context import current_account_id
from utils.paths import studio_db_path

ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
FINALIZATION_ACTIVE_STATUSES = frozenset({"pending", "running"})
FINALIZATION_TERMINAL_STATUSES = frozenset({"completed", "failed", "skipped"})
FINALIZATION_STATUSES = frozenset({"none"}) | FINALIZATION_ACTIVE_STATUSES | FINALIZATION_TERMINAL_STATUSES
_EVENTS_CHANGED = threading.Condition()
_RUN_TOMBSTONE_PREFIX = "chat-generation-run-tombstone:"
ChatGenerationEventInput = Union[tuple[str, dict[str, Any]], tuple[str, dict[str, Any], int]]

FINALIZATION_LEASE_MS = 60_000
FINALIZATION_RETRY_BASE_MS = 1_000
FINALIZATION_RETRY_MAX_MS = 60_000
_MIGRATION_LOCK_RETRIES = 3
_MIGRATION_LOCK_RETRY_SECONDS = 0.05
_ADDITIVE_COLUMNS = (
    ("progress_at", "INTEGER"),
    ("progress_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ("observation_pack_version", "INTEGER NOT NULL DEFAULT 0 CHECK (observation_pack_version IN (0, 1))"),
    ("resume_request_json", "TEXT"),
    ("finalization_status", "TEXT NOT NULL DEFAULT 'none'"),
    ("finalization_error", "TEXT"),
    ("finalization_worker_token", "TEXT"),
    ("finalization_started_at", "INTEGER"),
    ("finalization_completed_at", "INTEGER"),
    ("finalization_lease_expires_at", "INTEGER"),
    ("finalization_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("finalization_next_attempt_at", "INTEGER"),
)
_APPROVAL_TERMINAL_COLUMNS = (
    ("backend_account_id", "TEXT"),
    ("owner_subject", "TEXT"),
    ("thread_id", "TEXT"),
    ("authority_kind", "TEXT"),
    ("producer_receipt_json", "TEXT"),
    ("completion_json", "TEXT"),
    ("controller_is_error", "INTEGER"),
    ("completion_annotations_json", "TEXT"),
    ("post_controller_checkpoint_json", "TEXT"),
    ("terminal_seq", "INTEGER"),
    ("receipt_ref", "TEXT"),
    ("receipt_digest", "TEXT"),
)
_GENERIC_EXECUTION_ADDITIVE_COLUMNS = (
    ("pre_tool_checkpoint_json", "TEXT"),
    ("pre_tool_checkpoint_version", "INTEGER"),
    ("pre_tool_checkpoint_digest", "TEXT"),
)
_GENERIC_EXECUTION_COLUMNS = frozenset(
    {
        "run_id",
        "execution_id",
        "backend_account_id",
        "owner_subject",
        "session_id",
        "thread_id",
        "tool_name",
        "tool_call_id",
        "card_call_id",
        "arguments_json",
        "arguments_fingerprint",
        "pre_tool_checkpoint_json",
        "pre_tool_checkpoint_version",
        "pre_tool_checkpoint_digest",
        "authority_kind",
        "approval_id",
        "execution_state",
        "claim_token",
        "worker_token",
        "claimed_at",
        "started_at",
        "finished_at",
        "result_json",
        "error_message",
        "producer_receipt_json",
        "completion_json",
        "controller_is_error",
        "completion_annotations_json",
        "post_controller_checkpoint_json",
        "terminal_seq",
        "receipt_ref",
        "receipt_digest",
    }
)

# Progress lease columns live here rather than in _ensure_schema so the base table stays owned by
_schema_ready: set[Path] = set()
_schema_lock = threading.Lock()


class ChatGenerationConflictError(RuntimeError):
    pass


class ToolApprovalConflictError(RuntimeError):
    pass


class ToolApprovalExpiredError(RuntimeError):
    pass


class ToolApprovalFencedError(RuntimeError):
    pass


class ToolExecutionConflictError(RuntimeError):
    pass


class ObservationBindingConflictError(RuntimeError):
    """An ObservationPack execution identity was reused inconsistently."""


class ObservationBindingFencedError(RuntimeError):
    """The run, execution or approval identity no longer admits staging."""


class ObservationQuotaExceededError(RuntimeError):
    """An ObservationPack reservation would exceed the account quota."""


# Short aliases match the storage naming used by the optional packer while
# retaining the more explicit error names for callers of this module.
ObservationBindingConflict = ObservationBindingConflictError
ObservationQuotaExceeded = ObservationQuotaExceededError

_OBSERVATION_BINDING_STATES = frozenset(
    {"staged", "available", "unavailable", "missing", "corrupt", "missing_key"}
)
_OBSERVATION_AUTHORITY_KINDS = frozenset({"approved", "ungated"})
_OBSERVATION_DEFAULT_FORMAT_VERSION = 1
_OBSERVATION_DEFAULT_CANONICALIZATION_VERSION = "helix.tool-text.v1"
_OBSERVATION_DEFAULT_PROJECTION_VERSION = "helix.observation-projection.v1"
_OBSERVATION_KEY_ID_RE = re.compile(r"^observation-key-v1:sha256:[0-9a-f]{64}$")
_OBSERVATION_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_MIN_PLAINTEXT_BYTES = 10 * 1024
_OBSERVATION_MAX_PLAINTEXT_BYTES = 8 * 1024 * 1024
_OBSERVATION_CIPHERTEXT_OVERHEAD = 12 + 16
_OBSERVATION_MAX_PROJECTION_CHARS = 4096


def now_ms() -> int:
    return int(time.time() * 1000)


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def _database_path(conn: sqlite3.Connection) -> Path:
    """The file this connection opened, so polling paths do not resolve the account root twice."""
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(row[2]) if row and row[2] else studio_db_path()


def _connect() -> sqlite3.Connection:
    """Open a connection only after the complete additive schema is available.

    A connection that can see half of this migration is more dangerous than a
    transient lock error: callers may then make ownership decisions without the
    lease columns that fence them.  Retry lock contention briefly, verify the
    complete column set, and otherwise fail closed.
    """
    conn = get_connection()
    db_path = _database_path(conn)
    # Warm account connections already use the exact path that completed the
    # migration.  Avoid resolving it again: resolving can touch unavailable
    # network/removable parents and is unnecessary for the common path.
    with _schema_lock:
        if db_path in _schema_ready:
            return conn
    schema_path = db_path.resolve()
    with _schema_lock:
        if schema_path in _schema_ready:
            _schema_ready.add(db_path)
            return conn
    try:
        with _schema_lock:
            if schema_path not in _schema_ready:
                required = {column for column, _spec in _ADDITIVE_COLUMNS}
                required_approval = {
                    column for column, _spec in _APPROVAL_TERMINAL_COLUMNS
                }
                for attempt in range(_MIGRATION_LOCK_RETRIES):
                    try:
                        columns = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_runs)"
                            ).fetchall()
                        }
                        if not columns:
                            raise sqlite3.OperationalError(
                                "chat_generation_runs schema is unavailable"
                            )
                        for column, spec in _ADDITIVE_COLUMNS:
                            if column not in columns:
                                try:
                                    conn.execute(
                                        f"ALTER TABLE chat_generation_runs ADD COLUMN {column} {spec}"
                                    )
                                except sqlite3.OperationalError as exc:
                                    # A concurrent migrator may win after our
                                    # PRAGMA. Verification below is authoritative.
                                    if "duplicate column" not in str(exc).lower():
                                        raise
                        # A finalizer claimed by an older build has no explicit
                        # lease. Treat its last ownership stamp as the expiry so
                        # it is recoverable rather than permanently stuck.
                        conn.execute(
                            """UPDATE chat_generation_runs
                               SET finalization_lease_expires_at=COALESCE(
                                   finalization_started_at, updated_at, 0
                               )
                               WHERE finalization_status='running'
                                 AND finalization_lease_expires_at IS NULL"""
                        )
                        conn.execute(
                            """CREATE TABLE IF NOT EXISTS chat_generation_tool_approvals (
                                run_id TEXT NOT NULL REFERENCES chat_generation_runs(id)
                                    ON DELETE CASCADE,
                                approval_id TEXT NOT NULL,
                                session_id TEXT NOT NULL DEFAULT '',
                                tool_name TEXT NOT NULL,
                                tool_call_id TEXT NOT NULL DEFAULT '',
                                card_call_id TEXT NOT NULL DEFAULT '',
                                execution_id TEXT NOT NULL,
                                arguments_json TEXT NOT NULL,
                                arguments_fingerprint TEXT NOT NULL,
                                pre_tool_checkpoint_json TEXT NOT NULL,
                                pre_tool_checkpoint_version INTEGER NOT NULL,
                                pre_tool_checkpoint_digest TEXT NOT NULL,
                                resume_checkpoint_json TEXT NOT NULL,
                                checkpoint_version INTEGER NOT NULL,
                                status TEXT NOT NULL CHECK(status IN ('pending','allow','deny')),
                                expires_at INTEGER NOT NULL,
                                proposed_at INTEGER NOT NULL,
                                decision TEXT CHECK(decision IS NULL OR decision IN ('allow','deny')),
                                decision_source TEXT,
                                decided_at INTEGER,
                                decision_seq INTEGER,
                                execution_state TEXT NOT NULL CHECK(execution_state IN (
                                    'unclaimed','claimed','started','finished','ambiguous','cancelled'
                                )),
                                claim_token TEXT,
                                worker_token TEXT,
                                claimed_at INTEGER,
                                started_at INTEGER,
                                finished_at INTEGER,
                                result_json TEXT,
                                error_message TEXT,
                                PRIMARY KEY(run_id, approval_id)
                            ) WITHOUT ROWID"""
                        )
                        execution_columns = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_tool_executions)"
                            ).fetchall()
                        }
                        for column, spec in _GENERIC_EXECUTION_ADDITIVE_COLUMNS:
                            if column in execution_columns:
                                continue
                            try:
                                conn.execute(
                                    "ALTER TABLE chat_generation_tool_executions "
                                    f"ADD COLUMN {column} {spec}"
                                )
                            except sqlite3.OperationalError as exc:
                                if "duplicate column" not in str(exc).lower():
                                    raise
                        approval_columns = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_tool_approvals)"
                            ).fetchall()
                        }
                        for column, spec in _APPROVAL_TERMINAL_COLUMNS:
                            if column in approval_columns:
                                continue
                            try:
                                conn.execute(
                                    "ALTER TABLE chat_generation_tool_approvals "
                                    f"ADD COLUMN {column} {spec}"
                                )
                            except sqlite3.OperationalError as exc:
                                if "duplicate column" not in str(exc).lower():
                                    raise
                        conn.execute(
                            """CREATE TABLE IF NOT EXISTS chat_generation_tool_executions (
                                run_id TEXT NOT NULL REFERENCES chat_generation_runs(id)
                                    ON DELETE CASCADE,
                                execution_id TEXT NOT NULL,
                                backend_account_id TEXT NOT NULL,
                                owner_subject TEXT NOT NULL,
                                session_id TEXT NOT NULL DEFAULT '',
                                thread_id TEXT NOT NULL,
                                tool_name TEXT NOT NULL,
                                tool_call_id TEXT NOT NULL DEFAULT '',
                                card_call_id TEXT NOT NULL DEFAULT '',
                                arguments_json TEXT NOT NULL,
                                arguments_fingerprint TEXT NOT NULL,
                                authority_kind TEXT NOT NULL CHECK(authority_kind='ungated'),
                                approval_id TEXT,
                                execution_state TEXT NOT NULL CHECK(execution_state IN (
                                    'claimed','started','finished','ambiguous','cancelled'
                                )),
                                claim_token TEXT NOT NULL,
                                worker_token TEXT NOT NULL,
                                claimed_at INTEGER NOT NULL,
                                started_at INTEGER,
                                finished_at INTEGER,
                                result_json TEXT,
                                error_message TEXT,
                                producer_receipt_json TEXT,
                                completion_json TEXT,
                                controller_is_error INTEGER,
                                completion_annotations_json TEXT,
                                post_controller_checkpoint_json TEXT,
                                terminal_seq INTEGER,
                                receipt_ref TEXT,
                                receipt_digest TEXT,
                                PRIMARY KEY(run_id, execution_id)
                            ) WITHOUT ROWID"""
                        )
                        conn.execute(
                            """CREATE INDEX IF NOT EXISTS
                               idx_chat_generation_tool_executions_state
                               ON chat_generation_tool_executions(
                                   run_id, execution_state, claimed_at
                               )"""
                        )
                        conn.execute(
                            """CREATE INDEX IF NOT EXISTS
                               idx_chat_generation_tool_approvals_state
                               ON chat_generation_tool_approvals(
                                   run_id, status, execution_state, expires_at
                               )"""
                        )
                        conn.commit()
                        verified = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_runs)"
                            ).fetchall()
                        }
                        missing = required - verified
                        if missing:
                            raise sqlite3.OperationalError(
                                "incomplete chat generation migration: missing "
                                + ", ".join(sorted(missing))
                            )
                        verified_approval = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_tool_approvals)"
                            ).fetchall()
                        }
                        missing_approval = required_approval - verified_approval
                        if missing_approval:
                            raise sqlite3.OperationalError(
                                "incomplete tool approval migration: missing "
                                + ", ".join(sorted(missing_approval))
                            )
                        verified_execution = {
                            row[1]
                            for row in conn.execute(
                                "PRAGMA table_info(chat_generation_tool_executions)"
                            ).fetchall()
                        }
                        missing_execution = (
                            _GENERIC_EXECUTION_COLUMNS - verified_execution
                        )
                        if missing_execution:
                            raise sqlite3.OperationalError(
                                "incomplete generic tool execution migration: missing "
                                + ", ".join(sorted(missing_execution))
                            )
                        _schema_ready.update((schema_path, db_path))
                        break
                    except sqlite3.OperationalError as exc:
                        conn.rollback()
                        locked = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                        if not locked or attempt + 1 >= _MIGRATION_LOCK_RETRIES:
                            raise
                        time.sleep(_MIGRATION_LOCK_RETRY_SECONDS * (attempt + 1))
    except Exception:
        conn.close()
        raise
    return conn


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _validated_observation_pack_version(value: Any) -> int:
    """Validate the backend-owned feature pin without accepting bools or coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("observation_pack_version must be 0 or 1")
    return value


def _validated_observation_format_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("observation format_version must be exactly 1")
    return value


def _observation_text(value: Any, name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or "\x00" in value:
        raise ValueError(f"{name} must be NUL-free text")
    return value


def _observation_optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _observation_text(value, name)


def _terminal_completion(
    *,
    execution_id: str,
    authority_kind: str,
    terminal_state: str,
    result: Any,
    error: BaseException | str | None,
    producer_receipt: dict[str, Any] | None,
    controller_is_error: bool | None,
    completion_annotations: dict[str, Any] | None,
    post_controller_checkpoint: dict[str, Any] | None,
    binding: dict[str, Any],
) -> tuple[str, str, str, str | None, str]:
    """Return canonical completion JSON, digest, ref, result JSON, and error."""

    error_message = None if error is None else str(error)[:4000]
    result_json = None if error is not None else _canonical_json(result)
    receipt_ref = f"tool-receipt:{execution_id}"
    payload = {
        "schema_version": "helix.tool-completion.v1",
        "execution_id": execution_id,
        "authority_kind": authority_kind,
        "terminal_state": terminal_state,
        "selected_result": None if error is not None else result,
        "error": error_message,
        "producer_receipt": producer_receipt,
        "controller_is_error": controller_is_error,
        "completion_annotations": completion_annotations,
        "post_controller_checkpoint": post_controller_checkpoint,
        "receipt_ref": receipt_ref,
        "binding": binding,
    }
    completion_json = _canonical_json(payload)
    digest = hashlib.sha256(completion_json.encode("utf-8")).hexdigest()
    return completion_json, digest, receipt_ref, result_json, error_message


def canonical_request(
    *,
    thread_id: str,
    user_message_id: str,
    assistant_message_id: str,
    request_payload: dict[str, Any],
) -> tuple[str, str]:
    request_json = json.dumps(
        request_payload,
        sort_keys = True,
        separators = (",", ":"),
        ensure_ascii = False,
    )
    identity = json.dumps(
        {
            "threadId": thread_id,
            "userMessageId": user_message_id,
            "assistantMessageId": assistant_message_id,
            "requestPayload": request_payload,
        },
        sort_keys = True,
        separators = (",", ":"),
        ensure_ascii = False,
    )
    return request_json, hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())
    observation_pack_version = (
        int(row["observation_pack_version"])
        if "observation_pack_version" in keys
        else 0
    )
    _validated_observation_pack_version(observation_pack_version)
    return {
        "id": row["id"],
        "threadId": row["thread_id"],
        "userMessageId": row["user_message_id"],
        "assistantMessageId": row["assistant_message_id"],
        "observationPackVersion": observation_pack_version,
        "requestHash": row["request_hash"],
        "requestPayload": _loads(row["request_json"], {}),
        "resumeRequestPayload": (
            _loads(row["resume_request_json"], None)
            if "resume_request_json" in keys
            else None
        ),
        "status": row["status"],
        "cancelRequested": bool(row["cancel_requested"]),
        "lastEventSeq": int(row["last_event_seq"]),
        "finishReason": row["finish_reason"],
        "error": row["error_message"],
        "createdAt": int(row["created_at"]),
        "updatedAt": int(row["updated_at"]),
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
        "finalizationStatus": (
            row["finalization_status"] if "finalization_status" in keys else "none"
        ),
        "finalizationError": (
            row["finalization_error"] if "finalization_error" in keys else None
        ),
        "finalizationStartedAt": (
            row["finalization_started_at"] if "finalization_started_at" in keys else None
        ),
        "finalizationCompletedAt": (
            row["finalization_completed_at"] if "finalization_completed_at" in keys else None
        ),
        "finalizationLeaseExpiresAt": (
            row["finalization_lease_expires_at"]
            if "finalization_lease_expires_at" in keys
            else None
        ),
        "finalizationAttempts": (
            int(row["finalization_attempts"] or 0) if "finalization_attempts" in keys else 0
        ),
        "finalizationNextAttemptAt": (
            row["finalization_next_attempt_at"]
            if "finalization_next_attempt_at" in keys
            else None
        ),
        "pendingApprovals": [],
    }


def _approval_from_row(row: sqlite3.Row, *, include_checkpoint: bool = False) -> dict[str, Any]:
    keys = set(row.keys())
    value: dict[str, Any] = {
        "runId": str(row["run_id"]),
        "approvalId": str(row["approval_id"]),
        "sessionId": str(row["session_id"] or ""),
        "toolName": str(row["tool_name"]),
        "toolCallId": str(row["tool_call_id"] or ""),
        "cardCallId": str(row["card_call_id"] or ""),
        "executionId": str(row["execution_id"]),
        "arguments": _loads(row["arguments_json"], {}),
        "argumentsFingerprint": str(row["arguments_fingerprint"]),
        "status": str(row["status"]),
        "decision": row["decision"],
        "decisionSource": row["decision_source"],
        "decisionSeq": row["decision_seq"],
        "expiresAt": int(row["expires_at"]),
        "proposedAt": int(row["proposed_at"]),
        "decidedAt": row["decided_at"],
        "executionState": str(row["execution_state"]),
        "claimedAt": row["claimed_at"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "receiptRef": row["receipt_ref"] if "receipt_ref" in keys else None,
        "receiptDigest": row["receipt_digest"] if "receipt_digest" in keys else None,
        "terminalSeq": row["terminal_seq"] if "terminal_seq" in keys else None,
    }
    if include_checkpoint:
        value["checkpointVersion"] = int(row["checkpoint_version"])
        value["resumeCheckpoint"] = _loads(row["resume_checkpoint_json"], None)
        value["claimToken"] = row["claim_token"]
        value["workerToken"] = row["worker_token"]
        value["result"] = _loads(row["result_json"], None)
        value["error"] = row["error_message"]
        value["producerReceipt"] = (
            _loads(row["producer_receipt_json"], None)
            if "producer_receipt_json" in keys
            else None
        )
        value["completion"] = (
            _loads(row["completion_json"], None) if "completion_json" in keys else None
        )
        value["controllerIsError"] = (
            bool(row["controller_is_error"])
            if "controller_is_error" in keys and row["controller_is_error"] is not None
            else None
        )
        value["completionAnnotations"] = (
            _loads(row["completion_annotations_json"], None)
            if "completion_annotations_json" in keys
            else None
        )
        value["postControllerCheckpoint"] = (
            _loads(row["post_controller_checkpoint_json"], None)
            if "post_controller_checkpoint_json" in keys
            else None
        )
        value["authorityKind"] = (
            row["authority_kind"] if "authority_kind" in keys else None
        )
        value["backendAccountId"] = (
            row["backend_account_id"] if "backend_account_id" in keys else None
        )
        value["ownerSubject"] = row["owner_subject"] if "owner_subject" in keys else None
        value["threadId"] = row["thread_id"] if "thread_id" in keys else None
    return value


def _execution_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "runId": str(row["run_id"]),
        "executionId": str(row["execution_id"]),
        "backendAccountId": str(row["backend_account_id"]),
        "ownerSubject": str(row["owner_subject"]),
        "sessionId": str(row["session_id"] or ""),
        "threadId": str(row["thread_id"]),
        "toolName": str(row["tool_name"]),
        "toolCallId": str(row["tool_call_id"] or ""),
        "cardCallId": str(row["card_call_id"] or ""),
        "arguments": _loads(row["arguments_json"], {}),
        "argumentsFingerprint": str(row["arguments_fingerprint"]),
        "preToolCheckpoint": _loads(row["pre_tool_checkpoint_json"], None),
        "preToolCheckpointVersion": row["pre_tool_checkpoint_version"],
        "preToolCheckpointDigest": row["pre_tool_checkpoint_digest"],
        "authorityKind": str(row["authority_kind"]),
        "approvalId": row["approval_id"],
        "executionState": str(row["execution_state"]),
        "claimToken": str(row["claim_token"]),
        "workerToken": str(row["worker_token"]),
        "claimedAt": int(row["claimed_at"]),
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "result": _loads(row["result_json"], None),
        "error": row["error_message"],
        "producerReceipt": _loads(row["producer_receipt_json"], None),
        "completion": _loads(row["completion_json"], None),
        "controllerIsError": (
            bool(row["controller_is_error"])
            if row["controller_is_error"] is not None
            else None
        ),
        "completionAnnotations": _loads(row["completion_annotations_json"], None),
        "postControllerCheckpoint": _loads(
            row["post_controller_checkpoint_json"], None
        ),
        "terminalSeq": row["terminal_seq"],
        "receiptRef": row["receipt_ref"],
        "receiptDigest": row["receipt_digest"],
    }


def _attach_pending_approvals(
    conn: sqlite3.Connection, run: dict[str, Any]
) -> dict[str, Any]:
    if run.get("status") not in ACTIVE_STATUSES:
        run["pendingApprovals"] = []
        return run
    rows = conn.execute(
        """SELECT * FROM chat_generation_tool_approvals
           WHERE run_id=? AND execution_state NOT IN ('finished','cancelled')
           ORDER BY proposed_at, approval_id""",
        (run["id"],),
    ).fetchall()
    run["pendingApprovals"] = [_approval_from_row(row) for row in rows]
    return run


def _finalization_key(request_json: str | None) -> str:
    payload = _loads(request_json, {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("finalization_idempotency_key") or "").strip()


def _append_events_locked(
    conn: sqlite3.Connection, run_id: str, events: Iterable[ChatGenerationEventInput]
) -> list[int]:
    row = conn.execute(
        "SELECT last_event_seq FROM chat_generation_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    seq = int(row["last_event_seq"])
    batch_created = now_ms()
    sequences: list[int] = []
    for event in events:
        event_type, payload = event[:2]
        created = event[2] if len(event) == 3 else batch_created
        seq += 1
        conn.execute(
            """INSERT INTO chat_generation_events
               (run_id, seq, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                seq,
                event_type,
                json.dumps(payload, ensure_ascii = False, separators = (",", ":")),
                created,
            ),
        )
        sequences.append(seq)
    if sequences:
        conn.execute(
            "UPDATE chat_generation_runs SET last_event_seq=?, updated_at=? WHERE id=?",
            (seq, batch_created, run_id),
        )
    return sequences


def _terminal_safe_pending_events(
    events: Iterable[ChatGenerationEventInput],
    *,
    terminal_status: str,
) -> list[ChatGenerationEventInput]:
    """Fail closed private trace evidence against the effective terminal state.

    ``finish_run`` resolves a racing durable Stop while holding the same writer
    transaction that appends pending events.  Producer-side classification can
    therefore be stale by the time cancellation wins.  Rewrite only private
    trace rows here, after that resolution; public/tool payloads remain exact.
    """

    batch = list(events)
    if terminal_status == "completed":
        return batch
    safe: list[ChatGenerationEventInput] = []
    for event in batch:
        if event[0] != "optimization.trace":
            safe.append(event)
            continue
        original = event[1]
        payload = dict(original) if isinstance(original, dict) else {}
        payload["status"] = "unavailable"
        payload["trace"] = None
        payload["error"] = "generation_attempt_incomplete"
        payload["aggregation"] = {
            "admissible": False,
            "reason": "generation_attempt_incomplete",
        }
        safe.append(
            (event[0], payload, event[2])
            if len(event) == 3
            else (event[0], payload)
        )
    return safe


def _missing_lease_columns(exc: sqlite3.OperationalError) -> bool:
    """Whether `exc` is this database still waiting on the progress-lease migration. _connect lets a call through
    when contention blocks the ALTER, so every statement naming progress_at or progress_tokens can meet a table
    that predates them. Degrading to the pre-migration behaviour keeps that window harmless: without it a
    blocked migration would abort a generation with `no such column` the moment the writer let go.
    """
    message = str(exc).lower()
    return "no such column" in message and (
        "progress_at" in message or "progress_tokens" in message
    )


def _touch_progress_locked(
    conn: sqlite3.Connection,
    run_id: str,
    tokens: int,
    worker_token: str | None = None,
) -> None:
    """Stamp the progress lease for one flush of streamed output. Monotonic in both fields, the same
    rule studio_db._safe_generation_assistant_update applies to the assistant row this run owns: the
    token counter only ever accumulates, and progress_at takes MAX(stored, now) so a wall-clock step
    backwards (NTP, suspend) cannot age a live run into the sweep below. One chunk carries at most
    one token delta, so the count of chunk events is the token count. updated_at moves with it, as
    it already does on every event append. That is what the follower's snapshot poll compares, so a
    client watching a run through a long model preparation or an admission wait, neither of which
    emits events, sees the server is alive and rearms its own no-progress deadline instead of
    reporting an interruption over healthy work."""
    now = now_ms()
    try:
        sql = """UPDATE chat_generation_runs
               SET progress_at=MAX(COALESCE(progress_at, 0), ?),
                   updated_at=MAX(COALESCE(updated_at, 0), ?),
                   progress_tokens=COALESCE(progress_tokens, 0) + ?
               WHERE id=?"""
        args: tuple[Any, ...] = (now, now, max(0, int(tokens)), run_id)
        if worker_token is not None:
            sql += " AND worker_token=? AND status IN ('queued','running')"
            args += (worker_token,)
        conn.execute(
            sql,
            args,
        )
    except sqlite3.OperationalError as exc:
        # The migration has not landed yet, so the run ages out on started_at/created_at, which the sweep
        # already falls back to.
        if not _missing_lease_columns(exc):
            raise


def _commit(conn: sqlite3.Connection, *, notify: bool = False) -> None:
    conn.commit()
    if notify:
        with _EVENTS_CHANGED:
            _EVENTS_CHANGED.notify_all()


def _sync_assistant_status_locked(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    row = conn.execute(
        """SELECT r.assistant_message_id, r.finish_reason, m.metadata_json
           FROM chat_generation_runs r
           LEFT JOIN chat_messages m ON m.id=r.assistant_message_id
           WHERE r.id=?""",
        (run_id,),
    ).fetchone()
    if row is None or row["metadata_json"] is None:
        return
    metadata = _loads(row["metadata_json"], {})
    if not isinstance(metadata, dict) or metadata.get("generationRunId") not in (None, run_id):
        return
    metadata.update(
        {
            "generationRunId": run_id,
            "generationStatus": status,
            "serverManaged": True,
        }
    )
    if status == "cancelled":
        metadata["incomplete"] = {"reason": "cancelled"}
    elif status == "failed":
        metadata["incomplete"] = {"reason": "interrupted"}
    elif status == "completed":
        if row["finish_reason"] == "length":
            metadata["incomplete"] = {"reason": "length"}
        else:
            metadata.pop("incomplete", None)
    conn.execute(
        "UPDATE chat_messages SET metadata_json=? WHERE id=?",
        (json.dumps(metadata, ensure_ascii = False), row["assistant_message_id"]),
    )


def _sync_assistant_finalization_locked(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    row = conn.execute(
        """SELECT r.assistant_message_id, m.metadata_json
           FROM chat_generation_runs r
           LEFT JOIN chat_messages m ON m.id=r.assistant_message_id
           WHERE r.id=?""",
        (run_id,),
    ).fetchone()
    if row is None or row["metadata_json"] is None:
        return
    metadata = _loads(row["metadata_json"], {})
    if not isinstance(metadata, dict) or metadata.get("generationRunId") not in (None, run_id):
        return
    metadata.update(
        {
            "turnFinalizationStatus": status,
            "turnFinalizationIdempotencyKey": run_id,
        }
    )
    if error:
        metadata["turnFinalizationError"] = str(error)[:1000]
    else:
        metadata.pop("turnFinalizationError", None)
    conn.execute(
        "UPDATE chat_messages SET metadata_json=? WHERE id=?",
        (json.dumps(metadata, ensure_ascii = False), row["assistant_message_id"]),
    )


def create_run(
    *,
    run_id: str,
    owner_subject: str,
    thread_id: str,
    user_message_id: str,
    assistant_message_id: str,
    request_payload: dict[str, Any],
    observation_pack_version: int = 0,
) -> tuple[dict[str, Any], bool]:
    # This is deliberately outside request_payload/canonical_request.  The
    # admission pin is backend policy, not model/client input and must remain
    # stable across an idempotent create replay.
    observation_pack_version = _validated_observation_pack_version(
        observation_pack_version
    )
    request_json, request_hash = canonical_request(
        thread_id = thread_id,
        user_message_id = user_message_id,
        assistant_message_id = assistant_message_id,
        request_payload = request_payload,
    )
    created = now_ms()
    worker_token = secrets.token_hex(16)
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["owner_subject"] != owner_subject
                or existing["request_hash"] != request_hash
            ):
                raise ChatGenerationConflictError("Run ID is already bound to another request")
            conn.commit()
            return _attach_pending_approvals(conn, _run_from_row(existing)), False
        tombstone = conn.execute(
            "SELECT 1 FROM app_settings WHERE key=?",
            (f"{_RUN_TOMBSTONE_PREFIX}{run_id}",),
        ).fetchone()
        if tombstone is not None:
            raise ChatGenerationConflictError("Run ID has already been used")

        thread = conn.execute("SELECT 1 FROM chat_threads WHERE id=?", (thread_id,)).fetchone()
        user_message = conn.execute(
            "SELECT thread_id, role FROM chat_messages WHERE id=?",
            (user_message_id,),
        ).fetchone()
        if thread is None:
            raise KeyError("thread")
        if (
            user_message is None
            or user_message["thread_id"] != thread_id
            or user_message["role"] != "user"
        ):
            raise ValueError("userMessageId must identify a user message in the thread")
        active = conn.execute(
            """SELECT 1 FROM chat_generation_runs
               WHERE thread_id=? AND status IN ('queued','running','cancelling')""",
            (thread_id,),
        ).fetchone()
        if active is not None:
            raise ChatGenerationConflictError("This thread already has an active generation")

        metadata = {
            "generationRunId": run_id,
            "generationSeq": 0,
            "generationStatus": "queued",
            "serverManaged": True,
        }
        assistant = conn.execute(
            "SELECT * FROM chat_messages WHERE id=?",
            (assistant_message_id,),
        ).fetchone()
        if assistant is None:
            conn.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, parent_id, role, content_json, metadata_json, created_at)
                   VALUES (?, ?, ?, 'assistant', '[]', ?, ?)""",
                (
                    assistant_message_id,
                    thread_id,
                    user_message_id,
                    json.dumps(metadata, ensure_ascii = False),
                    created,
                ),
            )
        else:
            assistant_metadata = _loads(assistant["metadata_json"], {})
            existing_run_id = (
                assistant_metadata.get("generationRunId")
                if isinstance(assistant_metadata, dict)
                else None
            )
            content = _loads(assistant["content_json"], [])
            has_content = isinstance(content, list) and any(
                isinstance(part, dict)
                and (
                    (part.get("type") == "text" and str(part.get("text") or "").strip())
                    or part.get("type") not in (None, "text")
                )
                for part in content
            )
            if (
                assistant["thread_id"] != thread_id
                or assistant["parent_id"] != user_message_id
                or assistant["role"] != "assistant"
                or existing_run_id not in (None, run_id)
                or (existing_run_id is None and has_content)
            ):
                raise ChatGenerationConflictError(
                    "Assistant message does not match this generation run"
                )
            merged_metadata = (
                dict(assistant_metadata) if isinstance(assistant_metadata, dict) else {}
            )
            merged_metadata.update(metadata)
            conn.execute(
                "UPDATE chat_messages SET metadata_json=? WHERE id=?",
                (json.dumps(merged_metadata, ensure_ascii = False), assistant_message_id),
            )

        try:
            conn.execute(
                """INSERT INTO chat_generation_runs
                   (id, owner_subject, thread_id, user_message_id, assistant_message_id,
                    request_hash, request_json, worker_token, status, cancel_requested, last_event_seq,
                    observation_pack_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?, ?, ?)""",
                (
                    run_id,
                    owner_subject,
                    thread_id,
                    user_message_id,
                    assistant_message_id,
                    request_hash,
                    request_json,
                    worker_token,
                    observation_pack_version,
                    created,
                    created,
                ),
            )
        except sqlite3.IntegrityError as exc:
            active = conn.execute(
                """SELECT 1 FROM chat_generation_runs
                   WHERE thread_id=?
                     AND status IN ('queued','running','cancelling')""",
                (thread_id,),
            ).fetchone()
            if active is not None:
                raise ChatGenerationConflictError(
                    "This thread already has an active generation"
                ) from exc
            raise
        _append_events_locked(conn, run_id, [("run.created", {"status": "queued"})])
        row = conn.execute("SELECT * FROM chat_generation_runs WHERE id=?", (run_id,)).fetchone()
        _commit(conn, notify = True)
        return _attach_pending_approvals(conn, _run_from_row(row)), True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_run(run_id: str, owner_subject: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
    try:
        if owner_subject is None:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=? AND owner_subject=?",
                (run_id, owner_subject),
            ).fetchone()
        return (
            _attach_pending_approvals(conn, _run_from_row(row))
            if row is not None
            else None
        )
    finally:
        conn.close()


def _observation_binding_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    value = dict(row)
    # The contract calls this the source run in the evidence layer while the
    # relational identity deliberately remains ``run_id`` for the FK/UNIQUE.
    value["source_run_id"] = value["run_id"]
    return value


def _observation_binding_id(
    account_id: str, run_id: str, execution_id: str, arguments_fingerprint: str
) -> str:
    payload = "\0".join(
        (
            "helix.observation.binding.v1",
            account_id,
            run_id,
            execution_id,
            arguments_fingerprint,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _observation_check_account(account_id: str) -> None:
    # The account context is the authority.  An account string supplied by a
    # caller is only an identity assertion that must match it, never a switch
    # that grants access to another account's rows.
    if str(account_id) != str(current_account_id()):
        raise ObservationBindingFencedError("observation account binding mismatch")


def _observation_payload_fields(
    payload_json: str | dict[str, Any], payload_digest: str | None,
) -> tuple[str, str]:
    if isinstance(payload_json, dict):
        canonical = _canonical_json(payload_json)
    elif isinstance(payload_json, str):
        canonical = _observation_text(payload_json, "payload_json")
    else:
        raise TypeError("payload_json must be text or an object")
    calculated = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if payload_digest is not None and payload_digest != calculated:
        raise ObservationBindingFencedError("observation payload digest mismatch")
    return canonical, calculated


def _observation_connection(
    connection: sqlite3.Connection | None,
) -> tuple[sqlite3.Connection, bool, bool]:
    """Return (connection, owns_connection, owns_transaction).

    A supplied connection may already be inside the caller's finish
    transaction.  In that case no commit/rollback is performed here.
    """
    owns_connection = connection is None
    conn = _connect() if owns_connection else connection
    assert conn is not None
    owns_transaction = False
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
        owns_transaction = True
    return conn, owns_connection, owns_transaction


def _observation_close(
    conn: sqlite3.Connection,
    *,
    owns_connection: bool,
    owns_transaction: bool,
    success: bool,
) -> None:
    if owns_transaction:
        if success:
            conn.commit()
        else:
            conn.rollback()
    if owns_connection:
        conn.close()


def _observation_execution_identity(
    conn: sqlite3.Connection,
    *,
    run: sqlite3.Row,
    execution_id: str,
    authority_kind: str,
    approval_id: str | None,
) -> sqlite3.Row:
    """Load the durable execution/approval row that authorizes staging metadata.

    This checks identity only.  It intentionally does not grant execution
    authority and permits a ``started`` row because the later available
    transition is fenced by the committed terminal sequence.
    """
    if authority_kind == "approved":
        if not approval_id:
            raise ObservationBindingFencedError("approved observation needs approval_id")
        row = conn.execute(
            """SELECT a.*, r.owner_subject AS run_owner_subject,
                      r.thread_id AS run_thread_id
                 FROM chat_generation_tool_approvals a
                 JOIN chat_generation_runs r ON r.id=a.run_id
                WHERE a.run_id=? AND a.approval_id=? AND a.execution_id=?""",
            (run["id"], approval_id, execution_id),
        ).fetchone()
    elif authority_kind == "ungated":
        if approval_id is not None:
            raise ObservationBindingFencedError("ungated observation cannot have approval_id")
        row = conn.execute(
            """SELECT e.*, r.owner_subject AS run_owner_subject,
                      r.thread_id AS run_thread_id
                 FROM chat_generation_tool_executions e
                 JOIN chat_generation_runs r ON r.id=e.run_id
                WHERE e.run_id=? AND e.execution_id=?""",
            (run["id"], execution_id),
        ).fetchone()
    else:
        raise ObservationBindingFencedError("invalid observation authority kind")
    if row is None:
        raise ObservationBindingFencedError("durable execution identity is unavailable")
    return row


def _observation_compare_execution_identity(
    row: sqlite3.Row,
    *,
    account_id: str,
    owner_subject: str,
    thread_id: str,
    session_id: str,
    tool_name: str,
    tool_call_id: str,
    card_call_id: str,
    approval_id: str | None,
    authority_kind: str,
    payload_json: str,
    arguments_fingerprint: str,
    claim_token: str | None,
    worker_token: str | None,
) -> None:
    actual = {
        "account_id": str(row["backend_account_id"] or ""),
        "owner_subject": str(row["owner_subject"] or row["run_owner_subject"] or ""),
        "thread_id": str(row["thread_id"] or row["run_thread_id"] or ""),
        "session_id": str(row["session_id"] or ""),
        "tool_name": str(row["tool_name"] or ""),
        "tool_call_id": str(row["tool_call_id"] or ""),
        "card_call_id": str(row["card_call_id"] or ""),
        "approval_id": (
            str(row["approval_id"] or "")
            if "approval_id" in row.keys()
            else ""
        ),
        "authority_kind": (
            str(row["authority_kind"] or "")
            if "authority_kind" in row.keys()
            else ""
        ),
        "payload_json": str(row["arguments_json"] or ""),
        "arguments_fingerprint": str(row["arguments_fingerprint"] or ""),
        "claim_token": (
            str(row["claim_token"] or "") if "claim_token" in row.keys() else ""
        ),
        "worker_token": (
            str(row["worker_token"] or "") if "worker_token" in row.keys() else ""
        ),
    }
    expected = {
        "account_id": account_id,
        "owner_subject": owner_subject,
        "thread_id": thread_id,
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "card_call_id": card_call_id,
        "approval_id": str(approval_id or ""),
        "authority_kind": authority_kind,
        "payload_json": payload_json,
        "arguments_fingerprint": arguments_fingerprint,
        "claim_token": str(claim_token or ""),
        "worker_token": str(worker_token or ""),
    }
    if actual != expected:
        raise ObservationBindingFencedError("durable execution identity mismatch")
    if str(row["execution_state"] or "") != "started":
        raise ObservationBindingFencedError(
            "observation staging requires an exactly started execution"
        )


_OBSERVATION_IMMUTABLE_FIELDS = (
    "binding_id",
    "account_id",
    "run_id",
    "execution_id",
    "owner_subject",
    "session_id",
    "thread_id",
    "tool_name",
    "tool_call_id",
    "card_call_id",
    "approval_id",
    "authority_kind",
    "payload_json",
    "payload_digest",
    "arguments_fingerprint",
    "claim_token",
    "worker_token",
    "format_version",
    "canonicalization_version",
    "projection_version",
    "plaintext_length",
    "plaintext_digest",
    "reserved_bytes",
)


def _observation_assert_same(
    row: sqlite3.Row,
    expected: Mapping[str, Any],
    *,
    fields: tuple[str, ...] = _OBSERVATION_IMMUTABLE_FIELDS,
) -> None:
    for field in fields:
        actual = row[field]
        wanted = expected[field]
        if actual != wanted:
            raise ObservationBindingConflictError(
                f"observation binding field conflicts: {field}"
            )


def _observation_expected_row(
    *,
    binding_id: str,
    account_id: str,
    run_id: str,
    execution_id: str,
    owner_subject: str,
    session_id: str,
    thread_id: str,
    tool_name: str,
    tool_call_id: str,
    card_call_id: str,
    approval_id: str | None,
    authority_kind: str,
    payload_json: str,
    payload_digest: str,
    arguments_fingerprint: str,
    claim_token: str | None,
    worker_token: str | None,
    format_version: int,
    canonicalization_version: str,
    projection_version: str,
    plaintext_length: int | None,
    plaintext_digest: str | None,
    reserved_bytes: int,
) -> dict[str, Any]:
    return {
        "binding_id": binding_id,
        "account_id": account_id,
        "run_id": run_id,
        "execution_id": execution_id,
        "owner_subject": owner_subject,
        "session_id": session_id,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "card_call_id": card_call_id,
        "approval_id": approval_id,
        "authority_kind": authority_kind,
        "payload_json": payload_json,
        "payload_digest": payload_digest,
        "arguments_fingerprint": arguments_fingerprint,
        "claim_token": claim_token,
        "worker_token": worker_token,
        "format_version": format_version,
        "canonicalization_version": canonicalization_version,
        "projection_version": projection_version,
        "plaintext_length": plaintext_length,
        "plaintext_digest": plaintext_digest,
        "reserved_bytes": reserved_bytes,
    }


def stage_observation_binding(
    *,
    binding_id: str | None = None,
    account_id: str,
    run_id: str,
    execution_id: str,
    owner_subject: str | None = None,
    thread_id: str,
    session_id: str = "",
    tool_name: str,
    tool_call_id: str = "",
    card_call_id: str = "",
    approval_id: str | None = None,
    authority_kind: str = "ungated",
    payload_json: str | dict[str, Any],
    payload_digest: str | None = None,
    arguments_fingerprint: str,
    claim_token: str | None = None,
    worker_token: str | None = None,
    format_version: int = _OBSERVATION_DEFAULT_FORMAT_VERSION,
    canonicalization_version: str = _OBSERVATION_DEFAULT_CANONICALIZATION_VERSION,
    projection_version: str = _OBSERVATION_DEFAULT_PROJECTION_VERSION,
    plaintext_length: int | None = None,
    plaintext_digest: str | None = None,
    reserved_bytes: int = 0,
    quota_bytes: int = 64 * 1024 * 1024,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Reserve one account-local binding, idempotently and without authority.

    The durable execution/approval row is re-read in the same transaction.
    A supplied connection can therefore be the future finish transaction;
    this function never claims, starts, or authorizes execution itself.
    """
    _observation_check_account(account_id)
    account_id = _observation_text(account_id, "account_id", allow_empty=False)
    run_id = _observation_text(run_id, "run_id", allow_empty=False)
    execution_id = _observation_text(execution_id, "execution_id", allow_empty=False)
    thread_id = _observation_text(thread_id, "thread_id", allow_empty=False)
    session_id = _observation_text(session_id, "session_id")
    tool_name = _observation_text(tool_name, "tool_name", allow_empty=False)
    tool_call_id = _observation_text(tool_call_id, "tool_call_id")
    card_call_id = _observation_text(card_call_id, "card_call_id")
    owner_subject = None if owner_subject is None else _observation_text(owner_subject, "owner_subject", allow_empty=False)
    approval_id = _observation_optional_text(approval_id, "approval_id")
    claim_token = _observation_optional_text(claim_token, "claim_token")
    worker_token = _observation_optional_text(worker_token, "worker_token")
    arguments_fingerprint = _observation_text(
        arguments_fingerprint, "arguments_fingerprint", allow_empty=False
    )
    authority_kind = _observation_text(authority_kind, "authority_kind", allow_empty=False)
    if authority_kind not in _OBSERVATION_AUTHORITY_KINDS:
        raise ValueError("invalid observation authority kind")
    payload_json, payload_digest = _observation_payload_fields(payload_json, payload_digest)
    expected_binding_id = _observation_binding_id(
        account_id, run_id, execution_id, arguments_fingerprint
    )
    if binding_id is None:
        binding_id = expected_binding_id
    else:
        binding_id = _observation_text(binding_id, "binding_id", allow_empty=False)
        if _OBSERVATION_DIGEST_RE.fullmatch(binding_id) is None:
            raise ValueError("observation binding_id must be a lowercase SHA-256 digest")
        if binding_id != expected_binding_id:
            raise ObservationBindingFencedError("observation binding id mismatch")
    format_version = _validated_observation_format_version(format_version)
    if canonicalization_version != _OBSERVATION_DEFAULT_CANONICALIZATION_VERSION:
        raise ValueError("observation canonicalization_version must be helix.tool-text.v1")
    if projection_version != _OBSERVATION_DEFAULT_PROJECTION_VERSION:
        raise ValueError(
            "observation projection_version must be helix.observation-projection.v1"
        )
    if (
        isinstance(plaintext_length, bool)
        or not isinstance(plaintext_length, int)
        or not _OBSERVATION_MIN_PLAINTEXT_BYTES
        <= plaintext_length
        <= _OBSERVATION_MAX_PLAINTEXT_BYTES
    ):
        raise ValueError("observation plaintext length is outside the bounded range")
    if (
        not isinstance(plaintext_digest, str)
        or _OBSERVATION_DIGEST_RE.fullmatch(plaintext_digest) is None
    ):
        raise ValueError("observation plaintext digest must be a lowercase SHA-256 digest")
    if (
        isinstance(reserved_bytes, bool)
        or not isinstance(reserved_bytes, int)
        or reserved_bytes != plaintext_length + _OBSERVATION_CIPHERTEXT_OVERHEAD
    ):
        raise ValueError("observation reservation must equal ciphertext length")
    if isinstance(quota_bytes, bool) or not isinstance(quota_bytes, int) or quota_bytes < 0:
        raise ValueError("invalid observation quota")

    conn, owns_connection, owns_transaction = _observation_connection(connection)
    success = False
    try:
        run = conn.execute(
            """SELECT id, owner_subject, thread_id, observation_pack_version
                 FROM chat_generation_runs WHERE id=?""",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ObservationBindingFencedError("observation run is unavailable")
        if int(run["observation_pack_version"] or 0) != 1:
            raise ObservationBindingFencedError(
                "observation staging requires an admitted v1 run"
            )
        stored_owner = str(run["owner_subject"])
        if owner_subject is None:
            owner_subject = stored_owner
        if owner_subject != stored_owner or thread_id != str(run["thread_id"]):
            raise ObservationBindingFencedError("observation run identity mismatch")
        execution = _observation_execution_identity(
            conn,
            run=run,
            execution_id=execution_id,
            authority_kind=authority_kind,
            approval_id=approval_id,
        )
        _observation_compare_execution_identity(
            execution,
            account_id=account_id,
            owner_subject=owner_subject,
            thread_id=thread_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            card_call_id=card_call_id,
            approval_id=approval_id,
            authority_kind=authority_kind,
            payload_json=payload_json,
            arguments_fingerprint=arguments_fingerprint,
            claim_token=claim_token,
            worker_token=worker_token,
        )
        expected = _observation_expected_row(
            binding_id=binding_id,
            account_id=account_id,
            run_id=run_id,
            execution_id=execution_id,
            owner_subject=owner_subject,
            session_id=session_id,
            thread_id=thread_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            card_call_id=card_call_id,
            approval_id=approval_id,
            authority_kind=authority_kind,
            payload_json=payload_json,
            payload_digest=payload_digest,
            arguments_fingerprint=arguments_fingerprint,
            claim_token=claim_token,
            worker_token=worker_token,
            format_version=format_version,
            canonicalization_version=canonicalization_version,
            projection_version=projection_version,
            plaintext_length=plaintext_length,
            plaintext_digest=plaintext_digest,
            reserved_bytes=reserved_bytes,
        )
        existing = conn.execute(
            """SELECT * FROM chat_generation_observation_bindings
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        if existing is not None:
            _observation_assert_same(existing, expected)
            success = True
            return _observation_binding_dict(existing)
        used = conn.execute(
            """SELECT COALESCE(SUM(reserved_bytes), 0)
                 FROM chat_generation_observation_bindings
                WHERE account_id=? AND state IN ('staged', 'available')""",
            (account_id,),
        ).fetchone()[0]
        if int(used or 0) + reserved_bytes > quota_bytes:
            raise ObservationQuotaExceededError("observation quota would be exceeded")
        now = now_ms()
        conn.execute(
            """INSERT INTO chat_generation_observation_bindings (
                   binding_id, account_id, run_id, execution_id, owner_subject,
                   session_id, thread_id, tool_name, tool_call_id, card_call_id,
                   approval_id, authority_kind, payload_json, payload_digest,
                   arguments_fingerprint, claim_token, worker_token, state,
                   format_version, canonicalization_version, projection_version,
                   plaintext_length, plaintext_digest, reserved_bytes,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                         'staged', ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                binding_id,
                account_id,
                run_id,
                execution_id,
                owner_subject,
                session_id,
                thread_id,
                tool_name,
                tool_call_id,
                card_call_id,
                approval_id,
                authority_kind,
                payload_json,
                payload_digest,
                arguments_fingerprint,
                claim_token,
                worker_token,
                format_version,
                canonicalization_version,
                projection_version,
                plaintext_length,
                plaintext_digest,
                reserved_bytes,
                now,
                now,
            ),
        )
        inserted = conn.execute(
            "SELECT * FROM chat_generation_observation_bindings WHERE binding_id=?",
            (binding_id,),
        ).fetchone()
        success = True
        return _observation_binding_dict(inserted)
    except Exception:
        raise
    finally:
        _observation_close(
            conn,
            owns_connection=owns_connection,
            owns_transaction=owns_transaction,
            success=success,
        )


def get_observation_binding(
    binding_id: str,
    *,
    account_id: str | None = None,
    thread_id: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Read one binding under the current account; never a capability grant."""
    account_id = str(current_account_id()) if account_id is None else account_id
    _observation_check_account(account_id)
    binding_id = _observation_text(binding_id, "binding_id", allow_empty=False)
    if thread_id is not None:
        thread_id = _observation_text(thread_id, "thread_id", allow_empty=False)
    conn = connection or _connect()
    owns = connection is None
    try:
        sql = "SELECT * FROM chat_generation_observation_bindings WHERE binding_id=? AND account_id=?"
        args: tuple[Any, ...] = (binding_id, account_id)
        if thread_id is not None:
            sql += " AND thread_id=?"
            args += (thread_id,)
        return _observation_binding_dict(conn.execute(sql, args).fetchone())
    finally:
        if owns:
            conn.close()


def find_available_observation_binding(
    *,
    account_id: str,
    thread_id: str,
    plaintext_digest: str,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Return only an authorized available row, never a newer staged shadow."""
    _observation_check_account(account_id)
    account_id = _observation_text(account_id, "account_id", allow_empty=False)
    thread_id = _observation_text(thread_id, "thread_id", allow_empty=False)
    plaintext_digest = _observation_text(
        plaintext_digest, "plaintext_digest", allow_empty=False
    )
    conn = connection or _connect()
    owns = connection is None
    try:
        row = conn.execute(
            """SELECT * FROM chat_generation_observation_bindings
                WHERE account_id=? AND thread_id=? AND plaintext_digest=?
                  AND state='available'
                ORDER BY created_at ASC, binding_id ASC LIMIT 1""",
            (account_id, thread_id, plaintext_digest),
        ).fetchone()
        return _observation_binding_dict(row)
    finally:
        if owns:
            conn.close()


def _observation_complete_identity(
    *,
    binding_id: str,
    key_id: str,
    blob_relpath: str,
    plaintext_length: int,
    plaintext_digest: str,
    ciphertext_length: int,
    ciphertext_digest: str,
    blob_dev: int | None,
    blob_ino: int | None,
    projection: str,
    finished_event_seq: int,
) -> dict[str, Any]:
    if (
        isinstance(finished_event_seq, bool)
        or not isinstance(finished_event_seq, int)
        or finished_event_seq <= 0
    ):
        raise ObservationBindingFencedError("observation terminal sequence is required")
    if (
        isinstance(binding_id, str) is False
        or _OBSERVATION_DIGEST_RE.fullmatch(binding_id) is None
    ):
        raise ObservationBindingFencedError("invalid observation binding id")
    if not isinstance(key_id, str) or _OBSERVATION_KEY_ID_RE.fullmatch(key_id) is None:
        raise ValueError("invalid observation key id")
    if not isinstance(plaintext_digest, str) or _OBSERVATION_DIGEST_RE.fullmatch(plaintext_digest) is None:
        raise ValueError("invalid observation plaintext digest")
    if not isinstance(ciphertext_digest, str) or _OBSERVATION_DIGEST_RE.fullmatch(ciphertext_digest) is None:
        raise ValueError("invalid observation ciphertext digest")
    if not isinstance(plaintext_length, int) or isinstance(plaintext_length, bool):
        raise ValueError("invalid observation plaintext length")
    if not _OBSERVATION_MIN_PLAINTEXT_BYTES <= plaintext_length <= _OBSERVATION_MAX_PLAINTEXT_BYTES:
        raise ValueError("observation plaintext length is outside the bounded range")
    if not isinstance(ciphertext_length, int) or isinstance(ciphertext_length, bool):
        raise ValueError("invalid observation ciphertext length")
    if ciphertext_length != plaintext_length + _OBSERVATION_CIPHERTEXT_OVERHEAD:
        raise ValueError("observation ciphertext length does not match plaintext")
    if (
        not isinstance(blob_dev, int)
        or isinstance(blob_dev, bool)
        or blob_dev <= 0
        or not isinstance(blob_ino, int)
        or isinstance(blob_ino, bool)
        or blob_ino <= 0
    ):
        raise ValueError("observation blob identity must be positive")
    if not isinstance(blob_relpath, str) or blob_relpath != f"blobs/{binding_id}.blob":
        raise ObservationBindingFencedError("observation blob path does not match binding")
    if not isinstance(projection, str) or not projection:
        raise ValueError("observation projection must be nonempty")
    if len(projection) > _OBSERVATION_MAX_PROJECTION_CHARS:
        raise ValueError("observation projection exceeds the bounded length")
    return {
        "key_id": key_id,
        "blob_relpath": blob_relpath,
        "plaintext_length": plaintext_length,
        "plaintext_digest": plaintext_digest,
        "ciphertext_length": ciphertext_length,
        "ciphertext_digest": ciphertext_digest,
        "blob_dev": blob_dev,
        "blob_ino": blob_ino,
        "projection": projection,
        "finished_event_seq": finished_event_seq,
    }


_OBSERVATION_CANDIDATE_LOCAL_ERRORS = (
    ObservationBindingConflictError,
    ObservationBindingFencedError,
    ObservationQuotaExceededError,
    KeyError,
    TypeError,
    ValueError,
)


def _observation_candidate_field(
    candidate: Mapping[str, Any],
    name: str,
    *aliases: str,
    required: bool = True,
) -> Any:
    """Read a private candidate field without granting it durable authority."""
    for key in (name, *aliases):
        if key in candidate:
            return candidate[key]
    if required:
        raise ObservationBindingFencedError(
            f"observation candidate is missing {name}"
        )
    return None


def _observation_candidate_projection(
    candidate: Mapping[str, Any],
) -> str:
    if not isinstance(candidate, Mapping):
        raise TypeError("observation finish candidate must be a mapping")
    projection = _observation_candidate_field(candidate, "projection")
    if not isinstance(projection, str) or not projection:
        raise ValueError("observation candidate projection must be nonempty")
    if len(projection) > _OBSERVATION_MAX_PROJECTION_CHARS:
        raise ValueError("observation candidate projection exceeds the bounded length")
    return projection


def _observation_candidate_metadata(
    candidate: Mapping[str, Any],
    *,
    terminal_row: sqlite3.Row,
    authority_kind: str,
    fallback_result: Any,
) -> dict[str, Any]:
    """Normalize candidate metadata for the single-connection promotion gate.

    The mapping is deliberately only a value supplied by the producer.  All
    authority fields are derived again from the locked durable row and the
    staged binding below; candidate values can only match or be rejected.
    """
    if not isinstance(candidate, Mapping):
        raise TypeError("observation finish candidate must be a mapping")
    projection = _observation_candidate_projection(candidate)
    fallback = _observation_candidate_field(candidate, "fallback_result")
    if not isinstance(fallback, str):
        raise TypeError("observation candidate fallback_result must be text")
    expected_fallback = fallback_result if isinstance(fallback_result, str) else str(fallback_result)
    if fallback != expected_fallback:
        raise ObservationBindingConflictError(
            "observation candidate fallback result does not match the durable finish"
        )
    is_error = _observation_candidate_field(candidate, "is_error", required=False)
    if not isinstance(is_error, bool):
        raise TypeError("observation candidate is_error must be boolean")

    account_id = _observation_text(
        _observation_candidate_field(candidate, "account_id"),
        "candidate account_id",
        allow_empty=False,
    )
    source_run_id = _observation_text(
        _observation_candidate_field(candidate, "source_run_id", "run_id"),
        "candidate source_run_id",
        allow_empty=False,
    )
    execution_id = _observation_text(
        _observation_candidate_field(candidate, "execution_id"),
        "candidate execution_id",
        allow_empty=False,
    )
    tool_name = _observation_text(
        _observation_candidate_field(candidate, "tool_name"),
        "candidate tool_name",
        allow_empty=False,
    )
    tool_call_id = _observation_text(
        _observation_candidate_field(candidate, "tool_call_id"),
        "candidate tool_call_id",
    )
    approval_id = _observation_candidate_field(
        candidate, "approval_id", required=False
    )
    approval_id = _observation_optional_text(approval_id, "candidate approval_id")
    claim_token = _observation_candidate_field(candidate, "claim_token", required=False)
    claim_token = _observation_optional_text(claim_token, "candidate claim_token")
    worker_token = _observation_candidate_field(
        candidate, "worker_token", "worker_id", required=False
    )
    worker_token = _observation_optional_text(worker_token, "candidate worker_token")
    arguments_fingerprint = _observation_text(
        _observation_candidate_field(candidate, "arguments_fingerprint"),
        "candidate arguments_fingerprint",
        allow_empty=False,
    )
    handle = _observation_candidate_field(candidate, "handle")
    if not isinstance(handle, str) or not handle or "\x00" in handle:
        raise ValueError("observation candidate handle is invalid")

    # The handle is not an authority grant, but its public digest suffix is a
    # cheap consistency check against a candidate that was built for another
    # blob.  The account/thread keyed HMAC remains owned by the recall layer.
    plaintext_digest = _observation_candidate_field(candidate, "plaintext_digest")
    if (
        not isinstance(plaintext_digest, str)
        or _OBSERVATION_DIGEST_RE.fullmatch(plaintext_digest) is None
        or not handle.endswith(f":sha256:{plaintext_digest}")
    ):
        raise ObservationBindingFencedError(
            "observation candidate handle does not bind plaintext"
        )

    if authority_kind not in _OBSERVATION_AUTHORITY_KINDS:
        raise ObservationBindingFencedError("invalid observation finish authority")
    expected_authority = str(terminal_row["authority_kind"] or authority_kind)
    if expected_authority != authority_kind:
        raise ObservationBindingFencedError("observation authority kind mismatch")
    expected_approval = (
        str(terminal_row["approval_id"] or "")
        if "approval_id" in terminal_row.keys()
        else ""
    )
    if str(approval_id or "") != expected_approval:
        raise ObservationBindingFencedError("observation approval identity mismatch")

    return {
        "binding_id": _observation_candidate_field(candidate, "binding_id"),
        "account_id": account_id,
        "source_run_id": source_run_id,
        "execution_id": execution_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "approval_id": approval_id,
        "claim_token": claim_token,
        "worker_token": worker_token,
        "arguments_fingerprint": arguments_fingerprint,
        "handle": handle,
        "format_version": _observation_candidate_field(candidate, "format_version"),
        "canonicalization_version": _observation_candidate_field(
            candidate, "canonicalization_version"
        ),
        "projection_version": _observation_candidate_field(
            candidate, "projection_version"
        ),
        "plaintext_length": _observation_candidate_field(candidate, "plaintext_length"),
        "plaintext_digest": plaintext_digest,
        "ciphertext_length": _observation_candidate_field(
            candidate, "ciphertext_length"
        ),
        "ciphertext_digest": _observation_candidate_field(
            candidate, "ciphertext_digest"
        ),
        "key_id": _observation_candidate_field(candidate, "key_id"),
        "blob_relpath": _observation_candidate_field(candidate, "blob_relpath"),
        "blob_dev": _observation_candidate_field(candidate, "blob_dev"),
        "blob_ino": _observation_candidate_field(candidate, "blob_ino"),
        "projection": projection,
        "fallback_result": fallback,
        "is_error": is_error,
    }


def _observation_terminal_binding(
    row: sqlite3.Row,
    *,
    authority_kind: str,
) -> dict[str, Any]:
    binding = {
        "backend_account_id": str(row["backend_account_id"] or ""),
        "owner_subject": str(row["owner_subject"] or ""),
        "run_id": str(row["run_id"]),
        "session_id": str(row["session_id"] or ""),
        "thread_id": str(row["thread_id"] or ""),
        "execution_id": str(row["execution_id"]),
        "tool_name": str(row["tool_name"] or ""),
        "tool_call_id": str(row["tool_call_id"] or ""),
        "card_call_id": str(row["card_call_id"] or ""),
        "arguments_fingerprint": str(row["arguments_fingerprint"] or ""),
        "authority_kind": authority_kind,
        "approval_id": (
            str(row["approval_id"] or "")
            if "approval_id" in row.keys()
            else None
        ),
    }
    if authority_kind == "ungated":
        binding["pre_tool_checkpoint_digest"] = str(
            row["pre_tool_checkpoint_digest"] or ""
        )
        binding["approval_id"] = None
    return binding


def _observation_check_terminal_event_locked(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    authority_kind: str,
) -> None:
    terminal_seq = row["terminal_seq"]
    if (
        isinstance(terminal_seq, bool)
        or not isinstance(terminal_seq, int)
        or terminal_seq <= 0
    ):
        raise ObservationBindingFencedError(
            "observation terminal row has no durable event sequence"
        )
    event = conn.execute(
        """SELECT event_type, payload_json FROM chat_generation_events
           WHERE run_id=? AND seq=?""",
        (str(row["run_id"]), terminal_seq),
    ).fetchone()
    if event is None or event["event_type"] != "tool_execution.finished":
        raise ObservationBindingFencedError(
            "observation terminal event is unavailable or not finished"
        )
    payload = _loads(event["payload_json"], None)
    expected: dict[str, Any] = {
        "schema_version": (
            "helix.tool-execution.v2"
            if authority_kind == "approved"
            else "helix.tool-execution.v3"
        ),
        "execution_id": str(row["execution_id"]),
        "tool_name": str(row["tool_name"] or ""),
        "tool_call_id": str(
            row["tool_call_id"] or ""
            if authority_kind == "approved"
            else row["card_call_id"] or row["tool_call_id"] or ""
        ),
        "effect_state": "finished",
        "receipt_ref": str(row["receipt_ref"] or ""),
        "receipt_digest": str(row["receipt_digest"] or ""),
    }
    if authority_kind == "approved":
        expected["approval_id"] = str(row["approval_id"] or "")
    else:
        expected["authority_kind"] = "ungated"
    if row["error_message"] is not None:
        expected["error"] = str(row["error_message"])
    else:
        expected["result"] = _loads(row["result_json"], None)
    if payload != expected:
        raise ObservationBindingFencedError(
            "observation terminal event does not match the terminal row"
        )


def _observation_mark_unavailable_locked(
    conn: sqlite3.Connection,
    *,
    binding_id: str,
    account_id: str,
    reason: str,
) -> None:
    """Release a staged reservation without opening a second transaction."""
    conn.execute(
        """UPDATE chat_generation_observation_bindings
              SET state='unavailable', reserved_bytes=0,
                  unavailable_reason=?, updated_at=?
            WHERE binding_id=? AND account_id=? AND state='staged'""",
        (reason[:1000], now_ms(), binding_id, account_id),
    )


def _observation_best_effort_unavailable_locked(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    """Best-effort cleanup that can never prevent the ordinary finish."""
    if not isinstance(candidate, Mapping):
        return
    binding_id = candidate.get("binding_id")
    if not isinstance(binding_id, str) or not binding_id:
        return
    savepoint = "observation_candidate_cleanup"
    try:
        conn.execute(f"SAVEPOINT {savepoint}")
        _observation_mark_unavailable_locked(
            conn,
            binding_id=binding_id,
            account_id=str(current_account_id()),
            reason=reason,
        )
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        # SQLite permits a failed statement to be recovered with a savepoint;
        # if even that recovery is unavailable, the outer finish transaction
        # still owns the mandatory terminal write and will report its own
        # commit/fence error rather than turning optional cleanup into one.
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            pass


def _observation_promote_available_locked(
    conn: sqlite3.Connection,
    *,
    candidate: Mapping[str, Any],
    terminal_row: sqlite3.Row,
    authority_kind: str,
    fallback_result: Any,
) -> sqlite3.Row:
    """Atomically admit a published candidate after a provisional finish.

    Every identity is read from the locked terminal/staged rows and compared
    with the candidate.  The candidate can supply blob/projection facts, but
    it cannot choose account, run, execution, approval, claim or worker
    authority.
    """
    if str(terminal_row["execution_state"] or "") != "finished":
        raise ObservationBindingFencedError(
            "observation promotion requires a finished terminal row"
        )
    _observation_check_account(str(terminal_row["backend_account_id"] or ""))
    candidate_values = _observation_candidate_metadata(
        candidate,
        terminal_row=terminal_row,
        authority_kind=authority_kind,
        fallback_result=fallback_result,
    )
    run = conn.execute(
        """SELECT id, owner_subject, thread_id, observation_pack_version
             FROM chat_generation_runs WHERE id=?""",
        (str(terminal_row["run_id"]),),
    ).fetchone()
    if run is None or int(run["observation_pack_version"] or 0) != 1:
        raise ObservationBindingFencedError(
            "observation promotion requires the admitted run pin"
        )
    if (
        str(run["owner_subject"] or "") != str(terminal_row["owner_subject"] or "")
        or str(run["thread_id"] or "") != str(terminal_row["thread_id"] or "")
    ):
        raise ObservationBindingFencedError("observation run identity changed")

    expected_binding_id = _observation_binding_id(
        str(terminal_row["backend_account_id"]),
        str(terminal_row["run_id"]),
        str(terminal_row["execution_id"]),
        str(terminal_row["arguments_fingerprint"]),
    )
    if candidate_values["binding_id"] != expected_binding_id:
        raise ObservationBindingFencedError("observation candidate binding id mismatch")
    if (
        candidate_values["account_id"] != str(terminal_row["backend_account_id"])
        or candidate_values["source_run_id"] != str(terminal_row["run_id"])
        or candidate_values["execution_id"] != str(terminal_row["execution_id"])
        or candidate_values["tool_name"] != str(terminal_row["tool_name"] or "")
        or candidate_values["tool_call_id"] != str(terminal_row["tool_call_id"] or "")
        or candidate_values["arguments_fingerprint"]
        != str(terminal_row["arguments_fingerprint"] or "")
    ):
        raise ObservationBindingFencedError(
            "observation candidate does not match terminal identity"
        )
    if (
        candidate_values["claim_token"] is not None
        and candidate_values["claim_token"] != str(terminal_row["claim_token"] or "")
    ) or (
        candidate_values["worker_token"] is not None
        and candidate_values["worker_token"] != str(terminal_row["worker_token"] or "")
    ):
        raise ObservationBindingFencedError(
            "observation candidate worker fence does not match terminal row"
        )
    if candidate_values["is_error"] != bool(terminal_row["controller_is_error"]):
        raise ObservationBindingConflictError(
            "observation candidate error classification does not match terminal row"
        )
    if terminal_row["error_message"] is not None:
        raise ObservationBindingFencedError(
            "observation projection cannot replace an errored terminal result"
        )
    if _loads(terminal_row["result_json"], None) != candidate_values["projection"]:
        raise ObservationBindingConflictError(
            "observation projection does not match terminal result"
        )
    completion = _loads(terminal_row["completion_json"], None)
    if not isinstance(completion, dict):
        raise ObservationBindingFencedError("terminal completion is unavailable")
    if completion.get("binding") != _observation_terminal_binding(
        terminal_row, authority_kind=authority_kind
    ):
        raise ObservationBindingFencedError(
            "terminal completion binding does not match durable identity"
        )
    if (
        completion.get("selected_result") != candidate_values["projection"]
        or completion.get("error") is not None
        or completion.get("receipt_ref") != str(terminal_row["receipt_ref"] or "")
    ):
        raise ObservationBindingFencedError(
            "terminal completion does not match projected result"
        )
    _observation_check_terminal_event_locked(
        conn, row=terminal_row, authority_kind=authority_kind
    )
    terminal_seq = int(terminal_row["terminal_seq"])
    staged = conn.execute(
        """SELECT * FROM chat_generation_observation_bindings
             WHERE binding_id=? AND account_id=?""",
        (candidate_values["binding_id"], str(current_account_id())),
    ).fetchone()
    if staged is None or str(staged["state"] or "") != "staged":
        raise ObservationBindingFencedError(
            "observation candidate is not in the staged state"
        )
    expected_staged = {
        "binding_id": candidate_values["binding_id"],
        "account_id": candidate_values["account_id"],
        "run_id": candidate_values["source_run_id"],
        "execution_id": candidate_values["execution_id"],
        "owner_subject": str(terminal_row["owner_subject"] or ""),
        "session_id": str(terminal_row["session_id"] or ""),
        "thread_id": str(terminal_row["thread_id"] or ""),
        "tool_name": candidate_values["tool_name"],
        "tool_call_id": candidate_values["tool_call_id"],
        "card_call_id": str(terminal_row["card_call_id"] or ""),
        "approval_id": candidate_values["approval_id"],
        "authority_kind": authority_kind,
        "payload_json": str(terminal_row["arguments_json"] or ""),
        "arguments_fingerprint": candidate_values["arguments_fingerprint"],
        "claim_token": str(terminal_row["claim_token"] or ""),
        "worker_token": str(terminal_row["worker_token"] or ""),
        "format_version": candidate_values["format_version"],
        "canonicalization_version": candidate_values["canonicalization_version"],
        "projection_version": candidate_values["projection_version"],
        "plaintext_length": candidate_values["plaintext_length"],
        "plaintext_digest": candidate_values["plaintext_digest"],
        "reserved_bytes": candidate_values["ciphertext_length"],
    }
    for field, wanted in expected_staged.items():
        if field == "claim_token" or field == "worker_token":
            actual = str(staged[field] or "")
            if actual != str(wanted or ""):
                raise ObservationBindingFencedError(
                    f"observation staged identity mismatch: {field}"
                )
        elif staged[field] != wanted:
            raise ObservationBindingConflictError(
                f"observation staged identity mismatch: {field}"
            )
    complete = _observation_complete_identity(
        binding_id=candidate_values["binding_id"],
        key_id=candidate_values["key_id"],
        blob_relpath=candidate_values["blob_relpath"],
        plaintext_length=candidate_values["plaintext_length"],
        plaintext_digest=candidate_values["plaintext_digest"],
        ciphertext_length=candidate_values["ciphertext_length"],
        ciphertext_digest=candidate_values["ciphertext_digest"],
        blob_dev=candidate_values["blob_dev"],
        blob_ino=candidate_values["blob_ino"],
        projection=candidate_values["projection"],
        finished_event_seq=terminal_seq,
    )
    cursor = conn.execute(
        """UPDATE chat_generation_observation_bindings
              SET state='available', ciphertext_length=?, ciphertext_digest=?,
                  key_id=?, blob_relpath=?, blob_dev=?, blob_ino=?, projection=?,
                  finished_event_seq=?, unavailable_reason=NULL, updated_at=?
            WHERE binding_id=? AND account_id=? AND run_id=? AND execution_id=?
              AND state='staged'""",
        (
            complete["ciphertext_length"],
            complete["ciphertext_digest"],
            complete["key_id"],
            complete["blob_relpath"],
            complete["blob_dev"],
            complete["blob_ino"],
            complete["projection"],
            complete["finished_event_seq"],
            now_ms(),
            candidate_values["binding_id"],
            str(current_account_id()),
            str(terminal_row["run_id"]),
            str(terminal_row["execution_id"]),
        ),
    )
    if cursor.rowcount != 1:
        raise ObservationBindingFencedError(
            "observation staged row changed during promotion"
        )
    promoted = conn.execute(
        """SELECT * FROM chat_generation_observation_bindings
             WHERE binding_id=? AND account_id=?""",
        (candidate_values["binding_id"], str(current_account_id())),
    ).fetchone()
    if promoted is None or str(promoted["state"] or "") != "available":
        raise ObservationBindingFencedError(
            "observation availability write was not durable"
        )
    return promoted


def _finish_with_observation_candidate_locked(
    conn: sqlite3.Connection,
    *,
    candidate: Mapping[str, Any] | None,
    terminal_row: sqlite3.Row,
    authority_kind: str,
    terminal_state: str,
    fallback_result: Any,
    write_terminal: Any,
) -> sqlite3.Row | None:
    """Write a terminal projection or fall back exactly once under a savepoint."""
    if candidate is None:
        return write_terminal(fallback_result)

    # Ambiguous finishes are never eligible for evidence publication.  They
    # still get ordinary terminal semantics, with staged quota cleanup best
    # effort after the mandatory writer succeeds.
    if terminal_state != "finished":
        fallback = write_terminal(fallback_result)
        if fallback is not None:
            _observation_best_effort_unavailable_locked(
                conn, candidate, reason="terminal_state_not_publishable"
            )
        return fallback

    try:
        projection = _observation_candidate_projection(candidate)
    except _OBSERVATION_CANDIDATE_LOCAL_ERRORS:
        fallback = write_terminal(fallback_result)
        if fallback is not None:
            _observation_best_effort_unavailable_locked(
                conn, candidate, reason="candidate_invalid"
            )
        return fallback

    savepoint = "observation_candidate"
    conn.execute(f"SAVEPOINT {savepoint}")
    projected = write_terminal(projection)
    # A lost started/fence CAS is a mandatory terminal-writer failure, not a
    # candidate-local failure eligible for a second side effect.
    if projected is None:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        return None
    try:
        _observation_promote_available_locked(
            conn,
            candidate=candidate,
            terminal_row=projected,
            authority_kind=authority_kind,
            fallback_result=fallback_result,
        )
    except _OBSERVATION_CANDIDATE_LOCAL_ERRORS:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        fallback = write_terminal(fallback_result)
        if fallback is not None:
            _observation_best_effort_unavailable_locked(
                conn, candidate, reason="candidate_rejected"
            )
        return fallback
    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    return projected


def transition_observation_binding_available(
    *,
    binding_id: str,
    account_id: str,
    run_id: str,
    execution_id: str,
    owner_subject: str,
    thread_id: str,
    key_id: str,
    blob_relpath: str,
    plaintext_length: int,
    plaintext_digest: str,
    ciphertext_length: int,
    ciphertext_digest: str,
    blob_dev: int | None = None,
    blob_ino: int | None = None,
    projection: str,
    finished_event_seq: int,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Reject standalone publication until the durable finish helper exists.

    Availability is a terminal-state side effect.  Allowing this public seam
    to publish after an ordinary fallback finish would make a late publisher
    capable of promoting a candidate without the finish transaction's
    continuation/terminal identity.  The eventual locked finish helper will
    perform the actual promotion on its sole durable finish path.
    """
    raise ObservationBindingFencedError(
        "observation availability requires the durable finish transaction"
    )


def mark_observation_available(**kwargs: Any) -> dict[str, Any]:
    """Compatibility name for the disabled transition seam."""
    return transition_observation_binding_available(**kwargs)


def mark_observation_unavailable(
    *,
    binding_id: str,
    reason: str,
    account_id: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Release a staged reservation after a failed optional publication."""
    account_id = str(current_account_id()) if account_id is None else account_id
    _observation_check_account(account_id)
    reason = _observation_text(reason, "reason", allow_empty=False)[:1000]
    conn, owns_connection, owns_transaction = _observation_connection(connection)
    success = False
    try:
        row = conn.execute(
            "SELECT * FROM chat_generation_observation_bindings WHERE binding_id=? AND account_id=?",
            (binding_id, account_id),
        ).fetchone()
        if row is None:
            success = True
            return None
        state = str(row["state"])
        if state == "available":
            success = True
            return _observation_binding_dict(row)
        if state not in {"staged", "unavailable"}:
            raise ObservationBindingConflictError("observation binding cannot become unavailable")
        if state == "staged":
            conn.execute(
                """UPDATE chat_generation_observation_bindings
                      SET state='unavailable', reserved_bytes=0,
                          unavailable_reason=?, updated_at=?
                    WHERE binding_id=? AND account_id=? AND state='staged'""",
                (reason, now_ms(), binding_id, account_id),
            )
        else:
            conn.execute(
                """UPDATE chat_generation_observation_bindings
                      SET unavailable_reason=?, updated_at=?
                    WHERE binding_id=? AND account_id=?""",
                (reason, now_ms(), binding_id, account_id),
            )
        updated = conn.execute(
            "SELECT * FROM chat_generation_observation_bindings WHERE binding_id=?",
            (binding_id,),
        ).fetchone()
        success = True
        return _observation_binding_dict(updated)
    finally:
        _observation_close(
            conn,
            owns_connection=owns_connection,
            owns_transaction=owns_transaction,
            success=success,
        )


def get_worker_token(run_id: str) -> str | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT worker_token FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        return str(row["worker_token"]) if row is not None else None
    finally:
        conn.close()


def get_worker_run(
    run_id: str, worker_token: str | None = None
) -> tuple[dict[str, Any], str, str] | None:
    """Return one fenced producer snapshot and its owner from the same row read."""
    conn = _connect()
    try:
        if worker_token is None:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=?",
                (run_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM chat_generation_runs WHERE id=? AND worker_token=?",
                (run_id, worker_token),
            ).fetchone()
        if row is None:
            return None
        return (
            _attach_pending_approvals(conn, _run_from_row(row)),
            str(row["owner_subject"]),
            str(row["worker_token"]),
        )
    finally:
        conn.close()


def touch_progress(run_id: str, worker_token: str | None = None) -> None:
    """Renew one run's progress lease without recording any streamed output. For work the lease cannot
    see: automatic model loading, idle reload and auto-download all happen between mark_running and
    the first token, and the engine's own first-token budget does not start until after them, so
    ageing a run from mark_running could reap a legitimate load followed by a legitimate prefill."""
    conn = _connect()
    try:
        _touch_progress_locked(conn, run_id, 0, worker_token)
        conn.commit()
    finally:
        conn.close()


def get_progress(run_id: str) -> tuple[int | None, int] | None:
    """(last progress timestamp, tokens streamed) for one run, or None if unknown."""
    conn = _connect()
    try:
        try:
            row = conn.execute(
                """SELECT COALESCE(progress_at, started_at, created_at) AS progress_at,
                          COALESCE(progress_tokens, 0) AS progress_tokens
                   FROM chat_generation_runs WHERE id=?""",
                (run_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if not _missing_lease_columns(exc):
                raise
            row = conn.execute(
                """SELECT COALESCE(started_at, created_at) AS progress_at,
                          0 AS progress_tokens
                   FROM chat_generation_runs WHERE id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        progress_at = row["progress_at"]
        return (int(progress_at) if progress_at is not None else None, int(row["progress_tokens"]))
    finally:
        conn.close()


def list_active(thread_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE thread_id=?
                 AND status IN ('queued','running','cancelling')
               ORDER BY created_at, id""",
            (thread_id,),
        ).fetchall()
        return [_attach_pending_approvals(conn, _run_from_row(row)) for row in rows]
    finally:
        conn.close()


def list_all_active() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE status IN ('queued','running','cancelling')
               ORDER BY created_at, id"""
        ).fetchall()
        return [_attach_pending_approvals(conn, _run_from_row(row)) for row in rows]
    finally:
        conn.close()


def get_recovery_snapshot(
    run_id: str,
) -> tuple[dict[str, Any], str, int] | None:
    """Return the run and the exact ownership lease a recovery plan observed."""
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT *, COALESCE(progress_at, started_at, created_at) AS recovery_progress_at
               FROM chat_generation_runs
               WHERE id=? AND status IN ('queued','running') AND cancel_requested=0""",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return (
            _attach_pending_approvals(conn, _run_from_row(row)),
            str(row["worker_token"]),
            int(row["recovery_progress_at"]),
        )
    finally:
        conn.close()


def requeue_run_for_restart(
    run_id: str,
    resume_request_payload: dict[str, Any],
    *,
    reason: str = "process_restart",
    expected_worker_token: str | None = None,
    expected_progress_at: int | None = None,
    expected_last_event_seq: int | None = None,
    stale_before_ms: int | None = None,
) -> dict[str, Any] | None:
    """Rotate the worker lease and requeue one proven-safe orphaned run.

    The original request/hash stay immutable for API idempotency. The rebuilt
    continuation request lives in ``resume_request_json`` and is used only by the
    server-owned worker.
    """

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT *, COALESCE(progress_at, started_at, created_at) AS recovery_progress_at
               FROM chat_generation_runs WHERE id=?""",
            (run_id,),
        ).fetchone()
        if row is None or row["status"] not in ACTIVE_STATUSES:
            conn.commit()
            return None
        if row["status"] == "cancelling" or bool(row["cancel_requested"]):
            conn.commit()
            return None
        # Recovery is allowed to rotate only the exact, unchanged ownership
        # snapshot the planner inspected.  The legacy direct helper remains
        # available only for a never-started queued row; a running/live worker
        # can never be stolen without both fences.
        if (
            expected_worker_token is None
            or expected_progress_at is None
        ):
            if row["status"] != "queued" or row["started_at"] is not None:
                conn.commit()
                return None
            expected_worker_token = str(row["worker_token"])
            expected_progress_at = int(row["recovery_progress_at"])
            if expected_last_event_seq is None:
                expected_last_event_seq = int(row["last_event_seq"])
        elif stale_before_ms is None:
            # A planner snapshot without an expired lease is not positive proof
            # of orphaning: a live producer can be between heartbeat writes.
            conn.commit()
            return None
        if (
            str(row["worker_token"]) != expected_worker_token
            or int(row["recovery_progress_at"]) != int(expected_progress_at)
            or (
                expected_last_event_seq is not None
                and int(row["last_event_seq"]) != int(expected_last_event_seq)
            )
            or (
                stale_before_ms is not None
                and int(row["recovery_progress_at"]) > int(stale_before_ms)
            )
        ):
            conn.commit()
            return None
        resumed = now_ms()
        worker_token = secrets.token_hex(16)
        sql = """UPDATE chat_generation_runs
               SET status='queued', cancel_requested=0, worker_token=?,
                   resume_request_json=?, finish_reason=NULL, error_message=NULL,
                   progress_at=?, progress_tokens=0, updated_at=?,
                   started_at=NULL, completed_at=NULL,
                   finalization_status='none', finalization_error=NULL,
                   finalization_worker_token=NULL,
                   finalization_started_at=NULL, finalization_completed_at=NULL,
                   finalization_lease_expires_at=NULL,
                   finalization_attempts=0, finalization_next_attempt_at=NULL
               WHERE id=? AND worker_token=?
                 AND COALESCE(progress_at, started_at, created_at)=?
                 AND (? IS NULL OR last_event_seq=?)"""
        args: tuple[Any, ...] = (
            worker_token,
            json.dumps(resume_request_payload, ensure_ascii=False, separators=(",", ":")),
            resumed,
            resumed,
            run_id,
            expected_worker_token,
            int(expected_progress_at),
            expected_last_event_seq,
            expected_last_event_seq,
        )
        if stale_before_ms is not None:
            sql += " AND COALESCE(progress_at, started_at, created_at)<=?"
            args += (int(stale_before_ms),)
        cursor = conn.execute(
            sql,
            args,
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return None
        # A claim that never crossed the started CAS is recoverable only as part
        # of this fenced owner takeover.  Preserve the stable execution id while
        # releasing the dead owner's token.  Started calls are never reclaimed.
        conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET execution_state='unclaimed', claim_token=NULL,
                   worker_token=?, claimed_at=NULL
               WHERE run_id=? AND execution_state='claimed'
                 AND worker_token=?""",
            (worker_token, run_id, expected_worker_token),
        )
        conn.execute(
            """UPDATE chat_generation_tool_executions
               SET claim_token=lower(hex(randomblob(24))), worker_token=?, claimed_at=?
               WHERE run_id=? AND execution_state='claimed'
                 AND worker_token=?""",
            (worker_token, resumed, run_id, expected_worker_token),
        )
        _append_events_locked(
            conn,
            run_id,
            [
                (
                    "run.recovered",
                    {
                        "status": "queued",
                        "reason": reason[:240],
                        "resume": True,
                    },
                )
            ],
        )
        _sync_assistant_status_locked(conn, run_id, "queued")
        updated = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify=True)
        result = _attach_pending_approvals(conn, _run_from_row(updated))
        result["_workerToken"] = worker_token
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_events(
    run_id: str, worker_token: str, events: Iterable[ChatGenerationEventInput]
) -> list[int]:
    batch = list(events)
    if not batch:
        return []
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM chat_generation_runs WHERE id=? AND worker_token=?",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] not in ACTIVE_STATUSES:
            conn.commit()
            return []
        sequences = _append_events_locked(conn, run_id, batch)
        # The producer's only regular write, so it is also the lease renewal: output reaching the database
        # is the definition of progress this sweep reaps on.
        _touch_progress_locked(
            conn,
            run_id,
            sum(1 for event in batch if event[0] == "chunk"),
            worker_token,
        )
        _commit(conn, notify = True)
        return sequences
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_events_with_tool_proposal(
    run_id: str,
    worker_token: str,
    events: Iterable[ChatGenerationEventInput],
    proposal: dict[str, Any],
) -> list[int]:
    """Commit buffered output, the public ``tool_start`` and its private checkpoint.

    This is one writer transaction so a public approval card can never outrun the
    server-only state needed to resolve or recover it.  A retry is accepted only
    when every stable identity and the arguments fingerprint match.
    """

    batch = list(events)
    approval_id = str(proposal.get("approval_id") or "")
    if not approval_id:
        raise ValueError("durable tool proposal is missing approval_id")
    checkpoint = proposal.get("resume_checkpoint")
    checkpoint_version = int(proposal.get("checkpoint_version") or 0)
    if checkpoint_version <= 0 or not isinstance(checkpoint, dict):
        raise ValueError("durable tool proposal has no compatible checkpoint")
    arguments = proposal.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("durable tool proposal arguments must be an object")
    arguments_json = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    fingerprint = hashlib.sha256(arguments_json.encode("utf-8")).hexdigest()
    supplied_fingerprint = str(proposal.get("arguments_fingerprint") or fingerprint)
    if not secrets.compare_digest(supplied_fingerprint, fingerprint):
        raise ToolApprovalFencedError("tool proposal arguments fingerprint mismatch")
    checkpoint_json = json.dumps(
        checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    current_call = checkpoint.get("current_call")
    checkpoint_call = (
        current_call.get("tool_call") if isinstance(current_call, dict) else None
    )
    checkpoint_function = (
        checkpoint_call.get("function") if isinstance(checkpoint_call, dict) else None
    )
    checkpoint_arguments = (
        checkpoint_function.get("arguments")
        if isinstance(checkpoint_function, dict)
        else None
    )
    if isinstance(checkpoint_arguments, str):
        try:
            checkpoint_arguments = json.loads(checkpoint_arguments)
        except (TypeError, ValueError):
            checkpoint_arguments = None
    if (
        int(checkpoint.get("version") or 0) != checkpoint_version
        or str(checkpoint.get("backend") or "") not in {"gguf", "safetensors"}
        or not isinstance(checkpoint.get("conversation"), list)
        or not isinstance(checkpoint.get("controller"), dict)
        or not isinstance(checkpoint.get("remaining_calls"), list)
        or not isinstance(checkpoint_call, dict)
        or not isinstance(checkpoint_function, dict)
        or not isinstance(checkpoint_arguments, dict)
        or str(checkpoint_function.get("name") or "")
        != str(proposal.get("tool_name") or "unknown")[:240]
        or str(checkpoint_call.get("id") or "")
        != str(proposal.get("tool_call_id") or "")[:500]
        or str(current_call.get("card_call_id") or checkpoint_call.get("id") or "")
        != str(proposal.get("card_call_id") or proposal.get("tool_call_id") or "")[:500]
        or json.dumps(
            checkpoint_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        != arguments_json
    ):
        raise ToolApprovalFencedError(
            "durable tool checkpoint does not match the proposed call"
        )
    proposed_at = int(proposal.get("proposed_at") or now_ms())
    expires_at = int(proposal.get("expires_at") or 0)
    if expires_at <= proposed_at:
        raise ValueError("durable tool proposal expiry must be absolute and in the future")
    execution_id = str(proposal.get("execution_id") or f"tool-exec-{uuid4().hex}")
    stable = (
        str(proposal.get("session_id") or ""),
        str(proposal.get("tool_name") or "unknown")[:240],
        str(proposal.get("tool_call_id") or "")[:500],
        str(proposal.get("card_call_id") or proposal.get("tool_call_id") or "")[:500],
        execution_id,
        arguments_json,
        fingerprint,
        checkpoint_json,
        checkpoint_version,
        expires_at,
    )
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            """SELECT status, cancel_requested, owner_subject, thread_id
               FROM chat_generation_runs
               WHERE id=? AND worker_token=?""",
            (run_id, worker_token),
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if run["status"] not in ACTIVE_STATUSES or bool(run["cancel_requested"]):
            raise ToolApprovalFencedError("durable run no longer admits tool proposals")
        existing = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? AND approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO chat_generation_tool_approvals (
                       run_id, approval_id, session_id, tool_name, tool_call_id,
                       card_call_id, execution_id, arguments_json,
                       arguments_fingerprint, resume_checkpoint_json,
                       checkpoint_version, status, expires_at, proposed_at,
                       execution_state, worker_token, backend_account_id,
                       owner_subject, thread_id, authority_kind
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?,
                             'unclaimed', ?, ?, ?, ?, 'approved')""",
                (
                    run_id,
                    approval_id,
                    *stable,
                    proposed_at,
                    worker_token,
                    current_account_id(),
                    str(run["owner_subject"]),
                    str(run["thread_id"]),
                ),
            )
            sequences = _append_events_locked(conn, run_id, batch)
            proposal_payload = {
                "schema_version": "helix.tool-approval.v2",
                "approval_id": approval_id,
                "tool_name": stable[1],
                "tool_call_id": stable[2],
                "card_call_id": stable[3],
                "execution_id": stable[4],
                "arguments_fingerprint": fingerprint,
                "expires_at": expires_at,
                "status": "pending",
            }
            sequences.extend(
                _append_events_locked(conn, run_id, [("approval.proposed", proposal_payload)])
            )
        else:
            existing_stable = (
                str(existing["session_id"] or ""),
                str(existing["tool_name"]),
                str(existing["tool_call_id"] or ""),
                str(existing["card_call_id"] or ""),
                str(existing["execution_id"]),
                str(existing["arguments_json"]),
                str(existing["arguments_fingerprint"]),
                str(existing["resume_checkpoint_json"]),
                int(existing["checkpoint_version"]),
                int(existing["expires_at"]),
            )
            if existing_stable != stable:
                raise ToolApprovalFencedError(
                    "approval id is already bound to a different tool proposal"
                )
            # A replayed private envelope must not duplicate public output.
            sequences = []
        _touch_progress_locked(
            conn,
            run_id,
            sum(1 for event in batch if event[0] == "chunk"),
            worker_token,
        )
        _commit(conn, notify=True)
        return sequences
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_tool_approval(
    run_id: str,
    approval_id: str,
    *,
    owner_subject: str | None = None,
    include_checkpoint: bool = True,
) -> dict[str, Any] | None:
    conn = _connect()
    try:
        sql = (
            """SELECT a.* FROM chat_generation_tool_approvals a
               JOIN chat_generation_runs r ON r.id=a.run_id
               WHERE a.run_id=? AND a.approval_id=?"""
        )
        args: tuple[Any, ...] = (run_id, approval_id)
        if owner_subject is not None:
            sql += " AND r.owner_subject=?"
            args += (owner_subject,)
        row = conn.execute(sql, args).fetchone()
        return _approval_from_row(row, include_checkpoint=include_checkpoint) if row else None
    finally:
        conn.close()


def list_tool_approvals(
    run_id: str, *, include_checkpoint: bool = True
) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? ORDER BY proposed_at, approval_id""",
            (run_id,),
        ).fetchall()
        return [
            _approval_from_row(row, include_checkpoint=include_checkpoint) for row in rows
        ]
    finally:
        conn.close()


def decide_tool_approval(
    run_id: str,
    approval_id: str,
    *,
    owner_subject: str,
    decision: str,
    session_id: str | None = None,
    source: str = "user",
) -> dict[str, Any]:
    """First-writer decision CAS plus ``approval.decided`` in one commit."""

    if decision not in {"allow", "deny"}:
        raise ValueError("invalid tool approval decision")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT a.*, r.owner_subject, r.status AS run_status,
                      r.cancel_requested
               FROM chat_generation_tool_approvals a
               JOIN chat_generation_runs r ON r.id=a.run_id
               WHERE a.run_id=? AND a.approval_id=? AND r.owner_subject=?""",
            (run_id, approval_id, owner_subject),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, approval_id))
        # The authenticated endpoint is scoped by both the durable run owner and
        # the originating session.  Omitting a non-empty session is not a way to
        # weaken that scope. Server-originated expiry is intentionally exempt.
        if source == "user" and str(row["session_id"] or "") != str(session_id or ""):
            raise KeyError((run_id, approval_id))
        current = str(row["status"])
        if current in {"allow", "deny"}:
            if current != decision:
                raise ToolApprovalConflictError("the opposite approval decision already won")
            conn.commit()
            return _approval_from_row(row, include_checkpoint=True)
        decided_at = now_ms()
        if int(row["expires_at"]) <= decided_at:
            decision = "deny"
            source = "expiry"
        if bool(row["cancel_requested"]) or row["run_status"] in TERMINAL_STATUSES | {"cancelling"}:
            decision = "deny"
            source = "stop"
        cursor = conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET status=?, decision=?, decision_source=?, decided_at=?
               WHERE run_id=? AND approval_id=? AND status='pending'""",
            (decision, decision, source[:80], decided_at, run_id, approval_id),
        )
        if cursor.rowcount != 1:
            raise ToolApprovalConflictError("tool approval decision raced")
        [seq] = _append_events_locked(
            conn,
            run_id,
            [
                (
                    "approval.decided",
                    {
                        "schema_version": "helix.tool-approval.v2",
                        "approval_id": approval_id,
                        "tool_name": str(row["tool_name"]),
                        "tool_call_id": str(row["tool_call_id"] or ""),
                        "execution_id": str(row["execution_id"]),
                        "decision": decision,
                        "source": source[:80],
                    },
                )
            ],
        )
        conn.execute(
            """UPDATE chat_generation_tool_approvals SET decision_seq=?
               WHERE run_id=? AND approval_id=?""",
            (seq, run_id, approval_id),
        )
        updated = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? AND approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        _commit(conn, notify=True)
        result = _approval_from_row(updated, include_checkpoint=True)
        if source == "expiry":
            result["expired"] = True
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def expire_tool_approval(run_id: str, approval_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        owner = conn.execute(
            "SELECT owner_subject FROM chat_generation_runs WHERE id=?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    if owner is None:
        return None
    try:
        return decide_tool_approval(
            run_id,
            approval_id,
            owner_subject=str(owner["owner_subject"]),
            decision="deny",
            source="expiry",
        )
    except KeyError:
        return None


def claim_tool_execution(
    run_id: str,
    approval_id: str,
    *,
    worker_token: str,
    backend_account_id: str | None = None,
    session_id: str | None = None,
    thread_id: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    card_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Claim an allowed call without starting it; only one token can win."""

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT a.*, r.status AS run_status, r.cancel_requested,
                      r.worker_token AS run_worker_token,
                      r.owner_subject AS run_owner_subject,
                      r.thread_id AS run_thread_id
               FROM chat_generation_tool_approvals a
               JOIN chat_generation_runs r ON r.id=a.run_id
               WHERE a.run_id=? AND a.approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, approval_id))
        if arguments is not None:
            encoded = _canonical_json(arguments)
            fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            expected = (
                str(backend_account_id or current_account_id()),
                str(session_id or ""),
                str(thread_id or row["run_thread_id"]),
                str(tool_name or "unknown")[:240],
                str(tool_call_id or "")[:500],
                str(card_call_id or tool_call_id or "")[:500],
                encoded,
                fingerprint,
            )
            actual = (
                str(row["backend_account_id"] or ""),
                str(row["session_id"] or ""),
                str(row["thread_id"] or row["run_thread_id"]),
                str(row["tool_name"]),
                str(row["tool_call_id"] or ""),
                str(row["card_call_id"] or ""),
                str(row["arguments_json"]),
                str(row["arguments_fingerprint"]),
            )
            if actual != expected:
                raise ToolApprovalFencedError(
                    "approved tool invocation does not match its durable proposal"
                )
        if (
            str(row["run_worker_token"]) != worker_token
            or row["run_status"] not in ACTIVE_STATUSES
            or bool(row["cancel_requested"])
            or row["status"] != "allow"
        ):
            conn.commit()
            return None
        state = str(row["execution_state"])
        if state in {"finished", "ambiguous"}:
            conn.commit()
            return _approval_from_row(row, include_checkpoint=True)
        if state == "claimed" and str(row["worker_token"] or "") == worker_token:
            conn.commit()
            return _approval_from_row(row, include_checkpoint=True)
        if state != "unclaimed":
            conn.commit()
            return None
        claimed_at = now_ms()
        claim_token = secrets.token_hex(24)
        cursor = conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET execution_state='claimed', claim_token=?, worker_token=?, claimed_at=?
               WHERE run_id=? AND approval_id=? AND execution_state='unclaimed'""",
            (claim_token, worker_token, claimed_at, run_id, approval_id),
        )
        if cursor.rowcount != 1:
            conn.commit()
            return None
        _append_events_locked(
            conn,
            run_id,
            [
                (
                    "tool_execution.claimed",
                    {
                        "schema_version": "helix.tool-execution.v2",
                        "approval_id": approval_id,
                        "execution_id": str(row["execution_id"]),
                    },
                )
            ],
        )
        updated = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? AND approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        _commit(conn, notify=True)
        return _approval_from_row(updated, include_checkpoint=True)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_tool_execution_started(
    run_id: str,
    approval_id: str,
    *,
    worker_token: str,
    claim_token: str,
) -> dict[str, Any] | None:
    """Final stop/fence CAS immediately before ``Thread.start``."""

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT a.*, r.status AS run_status, r.cancel_requested,
                      r.worker_token AS run_worker_token
               FROM chat_generation_tool_approvals a
               JOIN chat_generation_runs r ON r.id=a.run_id
               WHERE a.run_id=? AND a.approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, approval_id))
        if (
            str(row["run_worker_token"]) != worker_token
            or str(row["worker_token"] or "") != worker_token
            or not secrets.compare_digest(str(row["claim_token"] or ""), str(claim_token))
            or row["run_status"] not in ACTIVE_STATUSES
            or bool(row["cancel_requested"])
            or row["status"] != "allow"
            or row["execution_state"] != "claimed"
        ):
            conn.commit()
            return None
        started_at = now_ms()
        cursor = conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET execution_state='started', started_at=?
               WHERE run_id=? AND approval_id=? AND execution_state='claimed'
                 AND claim_token=? AND worker_token=?""",
            (started_at, run_id, approval_id, claim_token, worker_token),
        )
        if cursor.rowcount != 1:
            conn.commit()
            return None
        _append_events_locked(
            conn,
            run_id,
            [
                (
                    "tool_execution.started",
                    {
                        "schema_version": "helix.tool-execution.v2",
                        "approval_id": approval_id,
                        "execution_id": str(row["execution_id"]),
                        "tool_name": str(row["tool_name"]),
                        "tool_call_id": str(row["tool_call_id"] or ""),
                        "effect_state": "started",
                    },
                )
            ],
        )
        updated = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? AND approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        _commit(conn, notify=True)
        return _approval_from_row(updated, include_checkpoint=True)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _finish_tool_execution_locked(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    approval_id: str,
    row: sqlite3.Row,
    terminal_state: str,
    finished_at: int,
    result_json: str | None,
    error_message: str | None,
    producer_receipt: dict[str, Any] | None,
    completion_json: str,
    controller_is_error: bool | None,
    completion_annotations: dict[str, Any] | None,
    post_controller_checkpoint: dict[str, Any] | None,
    receipt_ref: str,
    receipt_digest: str,
    worker_token: str,
    claim_token: str,
) -> sqlite3.Row | None:
    """Finish one approved execution while the caller owns its transaction.

    This helper deliberately has no transaction, connection-lifetime, or
    notification responsibilities.  The outer public function retains the
    worker/claim fences and commit/replay behavior; this locked body is the
    single place that mutates the terminal row, appends its terminal event,
    records the event sequence, and reads the authoritative row back.
    """
    cursor = conn.execute(
        """UPDATE chat_generation_tool_approvals
           SET execution_state=?, finished_at=?, result_json=?, error_message=?,
               producer_receipt_json=?, completion_json=?, controller_is_error=?,
               completion_annotations_json=?, post_controller_checkpoint_json=?,
               receipt_ref=?, receipt_digest=?
           WHERE run_id=? AND approval_id=? AND execution_state='started'
             AND worker_token=? AND claim_token=?""",
        (
            terminal_state,
            finished_at,
            result_json,
            error_message,
            _canonical_json(producer_receipt) if producer_receipt is not None else None,
            completion_json,
            None if controller_is_error is None else int(controller_is_error),
            (
                _canonical_json(completion_annotations)
                if completion_annotations is not None
                else None
            ),
            (
                _canonical_json(post_controller_checkpoint)
                if post_controller_checkpoint is not None
                else None
            ),
            receipt_ref,
            receipt_digest,
            run_id,
            approval_id,
            worker_token,
            claim_token,
        ),
    )
    if cursor.rowcount != 1:
        return None
    payload: dict[str, Any] = {
        "schema_version": "helix.tool-execution.v2",
        "approval_id": approval_id,
        "execution_id": str(row["execution_id"]),
        "tool_name": str(row["tool_name"]),
        "tool_call_id": str(row["tool_call_id"] or ""),
        "effect_state": terminal_state,
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
    }
    if error_message is not None:
        payload["error"] = error_message
    else:
        payload["result"] = _loads(result_json, None)
    [terminal_seq] = _append_events_locked(
        conn,
        run_id,
        [(f"tool_execution.{terminal_state}", payload)],
    )
    conn.execute(
        """UPDATE chat_generation_tool_approvals SET terminal_seq=?
           WHERE run_id=? AND approval_id=? AND receipt_digest=?""",
        (terminal_seq, run_id, approval_id, receipt_digest),
    )
    return conn.execute(
        """SELECT * FROM chat_generation_tool_approvals
           WHERE run_id=? AND approval_id=?""",
        (run_id, approval_id),
    ).fetchone()


def finish_tool_execution(
    run_id: str,
    approval_id: str,
    *,
    worker_token: str,
    claim_token: str,
    result: Any = None,
    error: BaseException | str | None = None,
    ambiguous: bool = False,
    producer_receipt: dict[str, Any] | None = None,
    controller_is_error: bool | None = None,
    completion_annotations: dict[str, Any] | None = None,
    post_controller_checkpoint: dict[str, Any] | None = None,
    finish_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    terminal_state = "ambiguous" if ambiguous else "finished"
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT * FROM chat_generation_tool_approvals
               WHERE run_id=? AND approval_id=?""",
            (run_id, approval_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, approval_id))
        # A retry after the commit boundary is replay-only.  Validate the
        # durable execution owner first, then return the stored receipt as the
        # authority.  In particular, do not rebuild a completion from a fresh
        # result/candidate supplied by a caller that may have lost the original
        # response after the database commit.
        if row["execution_state"] in {"finished", "ambiguous"}:
            if (
                not row["worker_token"]
                or str(row["worker_token"]) != str(worker_token)
                or not row["claim_token"]
                or not secrets.compare_digest(
                    str(row["claim_token"]), str(claim_token)
                )
            ):
                conn.commit()
                return None
            conn.commit()
            return _approval_from_row(row, include_checkpoint=True)
        if (
            row["execution_state"] != "started"
            or str(row["worker_token"] or "") != worker_token
            or not secrets.compare_digest(str(row["claim_token"] or ""), str(claim_token))
        ):
            conn.commit()
            return None
        binding = {
            "backend_account_id": row["backend_account_id"],
            "owner_subject": row["owner_subject"],
            "run_id": str(row["run_id"]),
            "session_id": str(row["session_id"] or ""),
            "thread_id": row["thread_id"],
            "execution_id": str(row["execution_id"]),
            "tool_name": str(row["tool_name"]),
            "tool_call_id": str(row["tool_call_id"] or ""),
            "card_call_id": str(row["card_call_id"] or ""),
            "arguments_fingerprint": str(row["arguments_fingerprint"]),
            "authority_kind": str(row["authority_kind"] or "approved"),
            "approval_id": str(row["approval_id"]),
        }
        def _write_terminal(result_value: Any) -> sqlite3.Row | None:
            (
                completion_json,
                receipt_digest,
                receipt_ref,
                result_json,
                error_message,
            ) = _terminal_completion(
                execution_id=str(row["execution_id"]),
                authority_kind="approved",
                terminal_state=terminal_state,
                result=result_value,
                error=error,
                producer_receipt=producer_receipt,
                controller_is_error=controller_is_error,
                completion_annotations=completion_annotations,
                post_controller_checkpoint=post_controller_checkpoint,
                binding=binding,
            )
            return _finish_tool_execution_locked(
                conn,
                run_id=run_id,
                approval_id=approval_id,
                row=row,
                terminal_state=terminal_state,
                finished_at=now_ms(),
                result_json=result_json,
                error_message=error_message,
                producer_receipt=producer_receipt,
                completion_json=completion_json,
                controller_is_error=controller_is_error,
                completion_annotations=completion_annotations,
                post_controller_checkpoint=post_controller_checkpoint,
                receipt_ref=receipt_ref,
                receipt_digest=receipt_digest,
                worker_token=worker_token,
                claim_token=claim_token,
            )

        updated = _finish_with_observation_candidate_locked(
            conn,
            candidate=finish_candidate,
            terminal_row=row,
            authority_kind="approved",
            terminal_state=terminal_state,
            fallback_result=result,
            write_terminal=_write_terminal,
        )
        if updated is None:
            conn.rollback()
            return None
        _commit(conn, notify=True)
        return _approval_from_row(updated, include_checkpoint=True)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def claim_ungated_tool_execution(
    run_id: str,
    execution_id: str,
    *,
    worker_token: str,
    backend_account_id: str,
    session_id: str | None,
    thread_id: str,
    tool_name: str,
    tool_call_id: str,
    card_call_id: str,
    arguments: dict[str, Any],
    pre_tool_checkpoint: dict[str, Any],
    pre_tool_checkpoint_digest: str,
) -> dict[str, Any] | None:
    """Create or replay one account-local, transactionally idempotent call claim."""

    arguments_json = _canonical_json(arguments)
    fingerprint = hashlib.sha256(arguments_json.encode("utf-8")).hexdigest()
    checkpoint_json = _canonical_json(pre_tool_checkpoint)
    checkpoint_digest = hashlib.sha256(checkpoint_json.encode("utf-8")).hexdigest()
    checkpoint_version = int(pre_tool_checkpoint.get("version") or 0)
    if checkpoint_version != 1 or not secrets.compare_digest(
        checkpoint_digest, str(pre_tool_checkpoint_digest)
    ):
        raise ToolApprovalFencedError("tool execution checkpoint binding mismatch")
    stable = (
        str(backend_account_id),
        str(session_id or ""),
        str(thread_id),
        str(tool_name or "unknown")[:240],
        str(tool_call_id or "")[:500],
        str(card_call_id or tool_call_id or "")[:500],
        arguments_json,
        fingerprint,
        checkpoint_json,
        checkpoint_version,
        checkpoint_digest,
    )
    if str(backend_account_id) != str(current_account_id()):
        raise ToolApprovalFencedError("tool execution account binding mismatch")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            """SELECT owner_subject, thread_id, status, cancel_requested,
                      worker_token AS run_worker_token
               FROM chat_generation_runs WHERE id=?""",
            (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if str(run["thread_id"]) != str(thread_id):
            raise ToolApprovalFencedError("tool execution thread binding mismatch")
        row = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        if row is not None:
            actual = (
                str(row["backend_account_id"]),
                str(row["session_id"] or ""),
                str(row["thread_id"]),
                str(row["tool_name"]),
                str(row["tool_call_id"] or ""),
                str(row["card_call_id"] or ""),
                str(row["arguments_json"]),
                str(row["arguments_fingerprint"]),
                str(row["pre_tool_checkpoint_json"] or ""),
                int(row["pre_tool_checkpoint_version"] or 0),
                str(row["pre_tool_checkpoint_digest"] or ""),
            )
            if actual != stable or str(row["owner_subject"]) != str(run["owner_subject"]):
                raise ToolExecutionConflictError(
                    "execution id is already bound to a different invocation"
                )
            state = str(row["execution_state"])
            if state in {"finished", "ambiguous"}:
                conn.commit()
                return _execution_from_row(row)
            if state == "claimed" and str(row["worker_token"]) == worker_token:
                conn.commit()
                return _execution_from_row(row)
            if (
                state == "claimed"
                and str(run["run_worker_token"]) == worker_token
                and run["status"] in ACTIVE_STATUSES
                and not bool(run["cancel_requested"])
            ):
                claim_token = secrets.token_hex(24)
                claimed_at = now_ms()
                cursor = conn.execute(
                    """UPDATE chat_generation_tool_executions
                       SET claim_token=?, worker_token=?, claimed_at=?
                       WHERE run_id=? AND execution_id=? AND execution_state='claimed'
                         AND pre_tool_checkpoint_digest=?""",
                    (
                        claim_token,
                        worker_token,
                        claimed_at,
                        run_id,
                        execution_id,
                        checkpoint_digest,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    return None
                row = conn.execute(
                    """SELECT * FROM chat_generation_tool_executions
                       WHERE run_id=? AND execution_id=?""",
                    (run_id, execution_id),
                ).fetchone()
                _commit(conn, notify=True)
                return _execution_from_row(row)
            conn.commit()
            return None
        if (
            str(run["run_worker_token"]) != worker_token
            or run["status"] not in ACTIVE_STATUSES
            or bool(run["cancel_requested"])
        ):
            conn.commit()
            return None
        claim_token = secrets.token_hex(24)
        claimed_at = now_ms()
        conn.execute(
            """INSERT INTO chat_generation_tool_executions (
                   run_id, execution_id, backend_account_id, owner_subject,
                   session_id, thread_id, tool_name, tool_call_id, card_call_id,
                   arguments_json, arguments_fingerprint, pre_tool_checkpoint_json,
                   pre_tool_checkpoint_version, pre_tool_checkpoint_digest, authority_kind,
                   execution_state, claim_token, worker_token, claimed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ungated',
                         'claimed', ?, ?, ?)""",
            (
                run_id,
                execution_id,
                stable[0],
                str(run["owner_subject"]),
                *stable[1:6],
                arguments_json,
                fingerprint,
                checkpoint_json,
                checkpoint_version,
                checkpoint_digest,
                claim_token,
                worker_token,
                claimed_at,
            ),
        )
        _append_events_locked(
            conn,
            run_id,
            [
                (
                    "tool_execution.claimed",
                    {
                        "schema_version": "helix.tool-execution.v3",
                        "execution_id": execution_id,
                        "authority_kind": "ungated",
                        "arguments_fingerprint": fingerprint,
                        "pre_tool_checkpoint_digest": checkpoint_digest,
                    },
                )
            ],
        )
        row = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        _commit(conn, notify=True)
        return _execution_from_row(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_ungated_tool_execution_started(
    run_id: str,
    execution_id: str,
    *,
    worker_token: str,
    claim_token: str,
    pre_tool_checkpoint_digest: str,
) -> dict[str, Any] | None:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT e.*, r.status AS run_status, r.cancel_requested,
                      r.worker_token AS run_worker_token
               FROM chat_generation_tool_executions e
               JOIN chat_generation_runs r ON r.id=e.run_id
               WHERE e.run_id=? AND e.execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, execution_id))
        if not secrets.compare_digest(
            str(row["pre_tool_checkpoint_digest"] or ""),
            str(pre_tool_checkpoint_digest),
        ):
            raise ToolExecutionConflictError(
                "tool execution checkpoint digest does not match its claim"
            )
        if row["execution_state"] in {"finished", "ambiguous"}:
            conn.commit()
            return _execution_from_row(row)
        if (
            row["execution_state"] != "claimed"
            or str(row["worker_token"]) != worker_token
            or str(row["run_worker_token"]) != worker_token
            or not secrets.compare_digest(str(row["claim_token"]), str(claim_token))
            or not secrets.compare_digest(
                str(row["pre_tool_checkpoint_digest"] or ""),
                str(pre_tool_checkpoint_digest),
            )
            or row["run_status"] not in ACTIVE_STATUSES
            or bool(row["cancel_requested"])
        ):
            conn.commit()
            return None
        started_at = now_ms()
        cursor = conn.execute(
            """UPDATE chat_generation_tool_executions
               SET execution_state='started', started_at=?
               WHERE run_id=? AND execution_id=? AND execution_state='claimed'
                 AND worker_token=? AND claim_token=?""",
            (started_at, run_id, execution_id, worker_token, claim_token),
        )
        if cursor.rowcount != 1:
            conn.commit()
            return None
        _append_events_locked(
            conn,
            run_id,
            [
                (
                    "tool_execution.started",
                    {
                        "schema_version": "helix.tool-execution.v3",
                        "execution_id": execution_id,
                        "tool_name": str(row["tool_name"]),
                        "tool_call_id": str(row["card_call_id"] or row["tool_call_id"] or ""),
                        "effect_state": "started",
                        "authority_kind": "ungated",
                    },
                )
            ],
        )
        updated = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        _commit(conn, notify=True)
        return _execution_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _finish_ungated_tool_execution_locked(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    execution_id: str,
    row: sqlite3.Row,
    terminal_state: str,
    finished_at: int,
    result_json: str | None,
    error_message: str | None,
    producer_receipt: dict[str, Any] | None,
    completion_json: str,
    controller_is_error: bool | None,
    completion_annotations: dict[str, Any] | None,
    post_controller_checkpoint: dict[str, Any] | None,
    receipt_ref: str,
    receipt_digest: str,
    worker_token: str,
    claim_token: str,
) -> sqlite3.Row | None:
    """Finish one ungated execution while the caller owns its transaction."""
    cursor = conn.execute(
        """UPDATE chat_generation_tool_executions
           SET execution_state=?, finished_at=?, result_json=?, error_message=?,
               producer_receipt_json=?, completion_json=?, controller_is_error=?,
               completion_annotations_json=?, post_controller_checkpoint_json=?,
               receipt_ref=?, receipt_digest=?
           WHERE run_id=? AND execution_id=? AND execution_state='started'
             AND worker_token=? AND claim_token=?""",
        (
            terminal_state,
            finished_at,
            result_json,
            error_message,
            _canonical_json(producer_receipt) if producer_receipt is not None else None,
            completion_json,
            None if controller_is_error is None else int(controller_is_error),
            (
                _canonical_json(completion_annotations)
                if completion_annotations is not None
                else None
            ),
            (
                _canonical_json(post_controller_checkpoint)
                if post_controller_checkpoint is not None
                else None
            ),
            receipt_ref,
            receipt_digest,
            run_id,
            execution_id,
            worker_token,
            claim_token,
        ),
    )
    if cursor.rowcount != 1:
        return None
    payload: dict[str, Any] = {
        "schema_version": "helix.tool-execution.v3",
        "execution_id": execution_id,
        "tool_name": str(row["tool_name"]),
        "tool_call_id": str(row["card_call_id"] or row["tool_call_id"] or ""),
        "effect_state": terminal_state,
        "authority_kind": "ungated",
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
    }
    if error_message is not None:
        payload["error"] = error_message
    else:
        payload["result"] = _loads(result_json, None)
    [terminal_seq] = _append_events_locked(
        conn, run_id, [(f"tool_execution.{terminal_state}", payload)]
    )
    conn.execute(
        """UPDATE chat_generation_tool_executions SET terminal_seq=?
           WHERE run_id=? AND execution_id=? AND receipt_digest=?""",
        (terminal_seq, run_id, execution_id, receipt_digest),
    )
    return conn.execute(
        """SELECT * FROM chat_generation_tool_executions
           WHERE run_id=? AND execution_id=?""",
        (run_id, execution_id),
    ).fetchone()


def finish_ungated_tool_execution(
    run_id: str,
    execution_id: str,
    *,
    worker_token: str,
    claim_token: str,
    result: Any = None,
    error: BaseException | str | None = None,
    ambiguous: bool = False,
    producer_receipt: dict[str, Any] | None = None,
    controller_is_error: bool | None = None,
    completion_annotations: dict[str, Any] | None = None,
    post_controller_checkpoint: dict[str, Any] | None = None,
    pre_tool_checkpoint_digest: str,
    finish_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    terminal_state = "ambiguous" if ambiguous else "finished"
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, execution_id))
        if not secrets.compare_digest(
            str(row["pre_tool_checkpoint_digest"] or ""),
            str(pre_tool_checkpoint_digest),
        ):
            raise ToolExecutionConflictError(
                "tool completion checkpoint digest does not match its claim"
            )
        # The checkpoint digest above binds this retry to the exact prepared
        # call.  Once a terminal receipt exists, the worker/claim fence is the
        # remaining authority check; the caller's new result and optional
        # metadata are deliberately ignored during replay.
        if row["execution_state"] in {"finished", "ambiguous"}:
            if (
                not row["worker_token"]
                or str(row["worker_token"]) != str(worker_token)
                or not row["claim_token"]
                or not secrets.compare_digest(
                    str(row["claim_token"]), str(claim_token)
                )
            ):
                conn.commit()
                return None
            conn.commit()
            return _execution_from_row(row)
        if (
            row["execution_state"] != "started"
            or str(row["worker_token"]) != worker_token
            or not secrets.compare_digest(str(row["claim_token"]), str(claim_token))
            or not secrets.compare_digest(
                str(row["pre_tool_checkpoint_digest"] or ""),
                str(pre_tool_checkpoint_digest),
            )
        ):
            conn.commit()
            return None
        binding = {
            "backend_account_id": str(row["backend_account_id"]),
            "owner_subject": str(row["owner_subject"]),
            "run_id": str(row["run_id"]),
            "session_id": str(row["session_id"] or ""),
            "thread_id": str(row["thread_id"]),
            "execution_id": str(row["execution_id"]),
            "tool_name": str(row["tool_name"]),
            "tool_call_id": str(row["tool_call_id"] or ""),
            "card_call_id": str(row["card_call_id"] or ""),
            "arguments_fingerprint": str(row["arguments_fingerprint"]),
            "pre_tool_checkpoint_digest": str(row["pre_tool_checkpoint_digest"] or ""),
            "authority_kind": "ungated",
            "approval_id": None,
        }
        def _write_terminal(result_value: Any) -> sqlite3.Row | None:
            completion_json, digest, receipt_ref, result_json, error_message = (
                _terminal_completion(
                    execution_id=execution_id,
                    authority_kind="ungated",
                    terminal_state=terminal_state,
                    result=result_value,
                    error=error,
                    producer_receipt=producer_receipt,
                    controller_is_error=controller_is_error,
                    completion_annotations=completion_annotations,
                    post_controller_checkpoint=post_controller_checkpoint,
                    binding=binding,
                )
            )
            return _finish_ungated_tool_execution_locked(
                conn,
                run_id=run_id,
                execution_id=execution_id,
                row=row,
                terminal_state=terminal_state,
                finished_at=now_ms(),
                result_json=result_json,
                error_message=error_message,
                producer_receipt=producer_receipt,
                completion_json=completion_json,
                controller_is_error=controller_is_error,
                completion_annotations=completion_annotations,
                post_controller_checkpoint=post_controller_checkpoint,
                receipt_ref=receipt_ref,
                receipt_digest=digest,
                worker_token=worker_token,
                claim_token=claim_token,
            )

        updated = _finish_with_observation_candidate_locked(
            conn,
            candidate=finish_candidate,
            terminal_row=row,
            authority_kind="ungated",
            terminal_state=terminal_state,
            fallback_result=result,
            write_terminal=_write_terminal,
        )
        if updated is None:
            conn.rollback()
            return None
        _commit(conn, notify=True)
        return _execution_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_ungated_tool_execution(
    run_id: str, execution_id: str
) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? AND execution_id=?""",
            (run_id, execution_id),
        ).fetchone()
        return _execution_from_row(row) if row is not None else None
    finally:
        conn.close()


def list_ungated_tool_executions(run_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_tool_executions
               WHERE run_id=? ORDER BY claimed_at, execution_id""",
            (run_id,),
        ).fetchall()
        return [_execution_from_row(row) for row in rows]
    finally:
        conn.close()


def mark_running(run_id: str, worker_token: str) -> bool:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT status, cancel_requested FROM chat_generation_runs
               WHERE id=? AND worker_token=?""",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        if row["status"] == "running":
            conn.commit()
            return True
        if row["status"] != "queued" or bool(row["cancel_requested"]):
            conn.commit()
            return False
        started = now_ms()
        conn.execute(
            """UPDATE chat_generation_runs
               SET status='running', started_at=COALESCE(started_at, ?), updated_at=?
               WHERE id=?""",
            (started, started, run_id),
        )
        _sync_assistant_status_locked(conn, run_id, "running")
        _append_events_locked(conn, run_id, [("run.started", {"status": "running"})])
        _touch_progress_locked(conn, run_id, 0, worker_token)
        _commit(conn, notify = True)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def request_cancel(run_id: str, owner_subject: str | None = None) -> dict[str, Any] | None:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        sql = "SELECT * FROM chat_generation_runs WHERE id=?"
        args: tuple[Any, ...] = (run_id,)
        if owner_subject is not None:
            sql += " AND owner_subject=?"
            args += (owner_subject,)
        row = conn.execute(sql, args).fetchone()
        if row is None:
            conn.commit()
            return None
        status = row["status"]
        if status in TERMINAL_STATUSES or status == "cancelling":
            conn.commit()
            return _run_from_row(row)
        updated = now_ms()
        if status == "queued":
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status='cancelled', cancel_requested=1, finish_reason='cancelled',
                       updated_at=?, completed_at=? WHERE id=?""",
                (updated, updated, run_id),
            )
            _append_events_locked(
                conn,
                run_id,
                [("run.cancelled", {"status": "cancelled", "finishReason": "cancelled"})],
            )
            _sync_assistant_status_locked(conn, run_id, "cancelled")
        else:
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status='cancelling', cancel_requested=1, updated_at=? WHERE id=?""",
                (updated, run_id),
            )
            _append_events_locked(conn, run_id, [("run.cancelling", {"status": "cancelling"})])
            _sync_assistant_status_locked(conn, run_id, "cancelling")
        # Stop and decision serialize on the same BEGIN IMMEDIATE writer lock.
        # Pending prompts become durable denials; an allowed but not-started call
        # is cancelled so a racing claimant cannot cross the start CAS.  A call
        # already marked started is left started/ambiguous and may only be
        # interrupted cooperatively by the process-local cancel signal.
        approvals = conn.execute(
            """SELECT approval_id, status, execution_state, tool_name,
                      tool_call_id, execution_id
               FROM chat_generation_tool_approvals
               WHERE run_id=? AND execution_state!='finished'""",
            (run_id,),
        ).fetchall()
        for approval in approvals:
            if approval["status"] == "pending":
                [decision_seq] = _append_events_locked(
                    conn,
                    run_id,
                    [
                        (
                            "approval.decided",
                            {
                                "schema_version": "helix.tool-approval.v2",
                                "approval_id": str(approval["approval_id"]),
                                "tool_name": str(approval["tool_name"]),
                                "tool_call_id": str(approval["tool_call_id"] or ""),
                                "execution_id": str(approval["execution_id"]),
                                "decision": "deny",
                                "source": "stop",
                            },
                        )
                    ],
                )
                conn.execute(
                    """UPDATE chat_generation_tool_approvals
                       SET status='deny', decision='deny', decision_source='stop',
                           decided_at=?, decision_seq=?
                       WHERE run_id=? AND approval_id=? AND status='pending'""",
                    (updated, decision_seq, run_id, approval["approval_id"]),
                )
            if approval["execution_state"] in {"unclaimed", "claimed"}:
                conn.execute(
                    """UPDATE chat_generation_tool_approvals
                       SET execution_state='cancelled', finished_at=?
                       WHERE run_id=? AND approval_id=?
                         AND execution_state IN ('unclaimed','claimed')""",
                    (updated, run_id, approval["approval_id"]),
                )
        conn.execute(
            """UPDATE chat_generation_tool_executions
               SET execution_state='cancelled', finished_at=?
               WHERE run_id=? AND execution_state='claimed'""",
            (updated, run_id),
        )
        updated_row = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify = True)
        return _attach_pending_approvals(conn, _run_from_row(updated_row))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_run(
    run_id: str,
    *,
    worker_token: str,
    status: str,
    finish_reason: str | None = None,
    error: str | None = None,
    pending_events: Iterable[ChatGenerationEventInput] = (),
) -> dict[str, Any] | None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Invalid terminal status: {status}")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=? AND worker_token=?",
            (run_id, worker_token),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        if row["status"] in TERMINAL_STATUSES:
            conn.commit()
            return _run_from_row(row)
        if bool(row["cancel_requested"]):
            status = "cancelled"
            finish_reason = "cancelled"
            error = None
        terminal_events = _terminal_safe_pending_events(
            pending_events,
            terminal_status=status,
        )
        _append_events_locked(conn, run_id, terminal_events)
        terminal_payload: dict[str, Any] = {
            "status": status,
            "finishReason": finish_reason,
        }
        if error:
            terminal_payload["error"] = error
        _append_events_locked(conn, run_id, [(f"run.{status}", terminal_payload)])
        completed = now_ms()
        finalization_status = "none"
        if (
            status == "completed"
            and finish_reason != "length"
            and _finalization_key(row["request_json"]) == run_id
        ):
            finalization_status = "pending"
        conn.execute(
            """UPDATE chat_generation_runs
               SET status=?, finish_reason=?, error_message=?, updated_at=?, completed_at=?,
                   finalization_status=?, finalization_error=NULL,
                   finalization_worker_token=NULL,
                   finalization_started_at=NULL, finalization_completed_at=NULL,
                   finalization_lease_expires_at=NULL,
                   finalization_attempts=0, finalization_next_attempt_at=NULL
               WHERE id=?""",
            (
                status,
                finish_reason,
                error,
                completed,
                completed,
                finalization_status,
                run_id,
            ),
        )
        if finalization_status == "pending":
            _append_events_locked(
                conn,
                run_id,
                [
                    (
                        "turn_finalization.pending",
                        {"status": "pending", "idempotencyKey": run_id},
                    )
                ],
            )
            _sync_assistant_finalization_locked(conn, run_id, "pending")
        _sync_assistant_status_locked(conn, run_id, status)
        updated = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify = True)
        return _run_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def claim_finalization(
    run_id: str,
    *,
    lease_ms: int = FINALIZATION_LEASE_MS,
) -> tuple[dict[str, Any], str, str] | None:
    """Atomically claim one pending finalization for exactly one worker."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        started = now_ms()
        if (
            row is None
            or row["finalization_status"] != "pending"
            or (
                row["finalization_next_attempt_at"] is not None
                and int(row["finalization_next_attempt_at"]) > started
            )
        ):
            conn.commit()
            return None
        token = secrets.token_hex(16)
        lease_expires = started + max(1, int(lease_ms))
        cursor = conn.execute(
            """UPDATE chat_generation_runs
               SET finalization_status='running', finalization_worker_token=?,
                   finalization_error=NULL,
                   finalization_started_at=COALESCE(finalization_started_at, ?),
                   finalization_completed_at=NULL,
                   finalization_lease_expires_at=?,
                   finalization_next_attempt_at=NULL, updated_at=?
               WHERE id=? AND finalization_status='pending'
                 AND (finalization_next_attempt_at IS NULL
                      OR finalization_next_attempt_at<=?)""",
            (token, started, lease_expires, started, run_id, started),
        )
        if cursor.rowcount != 1:
            conn.commit()
            return None
        _append_events_locked(
            conn,
            run_id,
            [("turn_finalization.started", {"status": "running", "idempotencyKey": run_id})],
        )
        _sync_assistant_finalization_locked(conn, run_id, "running")
        claimed = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify=True)
        return _run_from_row(claimed), token, str(row["owner_subject"])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_finalization(
    run_id: str,
    worker_token: str,
    *,
    status: str,
    receipt: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    """Settle a claimed finalizer and persist its terminal receipt as a run event."""
    if status not in FINALIZATION_TERMINAL_STATUSES:
        raise ValueError(f"Invalid finalization terminal status: {status}")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        owned_at = now_ms()
        row = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_lease_expires_at>?""",
            (run_id, worker_token, owned_at),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        current = str(row["finalization_status"] or "none")
        if current in FINALIZATION_TERMINAL_STATUSES:
            conn.commit()
            return _run_from_row(row)
        if current != "running":
            conn.commit()
            return None
        completed = owned_at
        conn.execute(
            """UPDATE chat_generation_runs
               SET finalization_status=?, finalization_error=?,
                   finalization_worker_token=NULL,
                   finalization_completed_at=?,
                   finalization_lease_expires_at=NULL,
                   finalization_attempts=0, finalization_next_attempt_at=NULL,
                   updated_at=?
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_lease_expires_at>?""",
            (status, error, completed, completed, run_id, worker_token, owned_at),
        )
        payload: dict[str, Any] = {
            "status": status,
            "idempotencyKey": run_id,
        }
        if receipt is not None:
            payload["receipt"] = receipt
        if error:
            payload["error"] = str(error)[:1000]
        _append_events_locked(conn, run_id, [(f"turn_finalization.{status}", payload)])
        _sync_assistant_finalization_locked(conn, run_id, status, error=error)
        updated = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify=True)
        return _run_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_finalization_event(
    run_id: str,
    worker_token: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    lease_ms: int = FINALIZATION_LEASE_MS,
) -> int | None:
    """Append one phase receipt while the named finalizer still owns the run."""
    if not event_type.startswith("turn_finalization."):
        raise ValueError("Finalization event types must use the turn_finalization namespace")
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        renewed = now_ms()
        row = conn.execute(
            """SELECT finalization_status FROM chat_generation_runs
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_lease_expires_at>?""",
            (run_id, worker_token, renewed),
        ).fetchone()
        if row is None or row["finalization_status"] != "running":
            conn.commit()
            return None
        [seq] = _append_events_locked(conn, run_id, [(event_type, payload)])
        conn.execute(
            """UPDATE chat_generation_runs
               SET finalization_lease_expires_at=?, updated_at=MAX(updated_at, ?)
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_status='running'""",
            (
                renewed + max(1, int(lease_ms)),
                renewed,
                run_id,
                worker_token,
            ),
        )
        _commit(conn, notify=True)
        return seq
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def renew_finalization_claim(
    run_id: str,
    worker_token: str,
    *,
    lease_ms: int = FINALIZATION_LEASE_MS,
) -> bool:
    """Renew a live claim without allowing an expired owner to resurrect it."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        renewed = now_ms()
        cursor = conn.execute(
            """UPDATE chat_generation_runs
               SET finalization_lease_expires_at=?, updated_at=MAX(updated_at, ?)
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_status='running'
                 AND finalization_lease_expires_at>?""",
            (
                renewed + max(1, int(lease_ms)),
                renewed,
                run_id,
                worker_token,
                renewed,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retry_finalization(
    run_id: str,
    worker_token: str,
    *,
    error: str,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return an ordinarily failed phase to pending under the current claim."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        owned_at = now_ms()
        row = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE id=? AND finalization_worker_token=?
                 AND finalization_status='running'
                 AND finalization_lease_expires_at>?""",
            (run_id, worker_token, owned_at),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        previous_attempts = int(row["finalization_attempts"] or 0)
        attempts = previous_attempts + 1
        retry_delay_ms = min(
            max(1, int(FINALIZATION_RETRY_MAX_MS)),
            max(1, int(FINALIZATION_RETRY_BASE_MS)) * (2 ** min(previous_attempts, 30)),
        )
        next_attempt_at = owned_at + retry_delay_ms
        conn.execute(
            """UPDATE chat_generation_runs
               SET finalization_status='pending', finalization_error=?,
                   finalization_worker_token=NULL,
                   finalization_lease_expires_at=NULL,
                   finalization_completed_at=NULL,
                   finalization_attempts=?, finalization_next_attempt_at=?,
                   updated_at=?
               WHERE id=? AND finalization_worker_token=?""",
            (
                str(error)[:1000],
                attempts,
                next_attempt_at,
                owned_at,
                run_id,
                worker_token,
            ),
        )
        payload: dict[str, Any] = {
            "status": "pending",
            "idempotencyKey": run_id,
            "error": str(error)[:1000],
            "attempt": attempts,
            "nextAttemptAt": next_attempt_at,
            "retryDelayMs": retry_delay_ms,
        }
        if receipt:
            payload["receipt"] = receipt
        _append_events_locked(conn, run_id, [("turn_finalization.retry_pending", payload)])
        _sync_assistant_finalization_locked(conn, run_id, "pending", error=error)
        updated = conn.execute(
            "SELECT * FROM chat_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        _commit(conn, notify=True)
        return _run_from_row(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def requeue_interrupted_finalizations(run_id: str | None = None) -> list[str]:
    """Recover only claims whose explicit ownership lease has expired."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        expired_at = now_ms()
        sql = (
            """SELECT id FROM chat_generation_runs
               WHERE finalization_status='running'
                 AND finalization_lease_expires_at<=?"""
        )
        args: tuple[Any, ...] = (expired_at,)
        if run_id is not None:
            sql += " AND id=?"
            args += (run_id,)
        rows = conn.execute(sql + " ORDER BY updated_at, id", args).fetchall()
        run_ids = [str(row["id"]) for row in rows]
        if run_ids:
            updated = now_ms()
            conn.executemany(
                """UPDATE chat_generation_runs
                   SET finalization_status='pending', finalization_worker_token=NULL,
                       finalization_error=NULL, finalization_completed_at=NULL,
                       finalization_lease_expires_at=NULL,
                       finalization_next_attempt_at=NULL,
                       updated_at=? WHERE id=? AND finalization_status='running'
                         AND finalization_lease_expires_at<=?""",
                [(updated, recovered_id, expired_at) for recovered_id in run_ids],
            )
            for run_id in run_ids:
                _append_events_locked(
                    conn,
                    run_id,
                    [
                        (
                            "turn_finalization.recovered",
                            {"status": "pending", "idempotencyKey": run_id},
                        )
                    ],
                )
                _sync_assistant_finalization_locked(conn, run_id, "pending")
        _commit(conn, notify=bool(run_ids))
        return run_ids
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_running_finalizations() -> list[tuple[str, int]]:
    """Return running claim expiries so startup can schedule delayed recovery."""
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT id, finalization_lease_expires_at
               FROM chat_generation_runs
               WHERE finalization_status='running'
                 AND finalization_lease_expires_at IS NOT NULL
               ORDER BY finalization_lease_expires_at, id"""
        ).fetchall()
        return [
            (str(row["id"]), int(row["finalization_lease_expires_at"]))
            for row in rows
        ]
    finally:
        conn.close()


def list_pending_finalizations() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM chat_generation_runs
               WHERE finalization_status='pending'
               ORDER BY COALESCE(finalization_next_attempt_at, 0), completed_at, id"""
        ).fetchall()
        return [_run_from_row(row) for row in rows]
    finally:
        conn.close()


def list_events(
    run_id: str,
    after: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT seq, event_type, payload_json, created_at
               FROM chat_generation_events
               WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?""",
            (run_id, after, limit),
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "type": row["event_type"],
                "payload": _loads(row["payload_json"], {}),
                "createdAt": int(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def wait_for_events(
    run_id: str,
    after: int = 0,
    timeout: float = 15,
) -> list[dict[str, Any]]:
    events = list_events(run_id, after)
    if events:
        return events
    with _EVENTS_CHANGED:
        events = list_events(run_id, after)
        if events:
            return events
        _EVENTS_CHANGED.wait(timeout)
    return list_events(run_id, after)


def reconcile_runs(
    *,
    error: str = "Studio restarted during generation",
    stale_after_ms: int | None = None,
    preserve_run_ids: set[str] | None = None,
) -> list[str]:
    """Settle active runs, returning the ids settled. ``stale_after_ms`` is what makes this safe to run
    while Studio is serving: with it, only runs whose progress lease has not moved for that long are
    settled, so a slow but advancing generation is never touched. Without it (process boot) every
    active run is orphaned by definition and all of them are settled. Partial output survives either
    way: only the run row and the assistant message's status metadata are rewritten, never the
    streamed content or the event log."""
    conn = _connect()
    settled: list[str] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        completed = now_ms()
        sql = """SELECT id, status, cancel_requested FROM chat_generation_runs
                 WHERE status IN ('queued','running','cancelling')"""
        args: tuple[Any, ...] = ()
        if stale_after_ms is not None:
            # started_at/created_at carry a run that has not streamed anything yet, so a producer wedged before
            # its first token still ages out.
            sql += " AND COALESCE(progress_at, started_at, created_at) <= ?"
            args = (completed - int(stale_after_ms),)
        try:
            rows = conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
        except sqlite3.OperationalError as exc:
            if not _missing_lease_columns(exc):
                raise
            # Contention blocked the migration, so falling back to started_at/created_at is the opposite of
            # conservative: those stamps are older by the whole life of the run, so one that streamed moments ago is
            # reaped once its total AGE passes the timeout. Boot reconcile passes no stale_after_ms.
            if stale_after_ms is not None:
                conn.rollback()
                return []
            rows = conn.execute(
                sql.replace(" AND COALESCE(progress_at, started_at, created_at) <= ?", "")
                + " ORDER BY created_at, id",
                (),
            ).fetchall()
        for row in rows:
            run_id = row["id"]
            if preserve_run_ids and str(run_id) in preserve_run_ids:
                continue
            # A Stop that was already recorded outlives the restart, and reporting it as a backend failure
            # would tell the user Studio broke when they stopped it; finish_run settles this case as cancelled.
            if str(row["status"]) == "cancelling" or bool(row["cancel_requested"]):
                status, finish_reason, message = "cancelled", "cancelled", None
                terminal = ("run.cancelled", {"status": status, "finishReason": finish_reason})
            else:
                status, finish_reason, message = "failed", "interrupted", error
                terminal = ("run.failed", {"status": status, "error": error, "interrupted": True})
            _append_events_locked(conn, run_id, [terminal])
            conn.execute(
                """UPDATE chat_generation_runs
                   SET status=?, finish_reason=?, error_message=?,
                       updated_at=?, completed_at=? WHERE id=?""",
                (status, finish_reason, message, completed, completed, run_id),
            )
            # Stamps incomplete on the assistant message, which is what releases the frontend's "generating"
            # state and restores Send.
            _sync_assistant_status_locked(conn, run_id, status)
            settled.append(str(run_id))
        _commit(conn, notify = bool(settled))
        return settled
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reconcile_orphaned_runs(error: str = "Studio restarted during generation") -> int:
    return len(reconcile_runs(error = error))
