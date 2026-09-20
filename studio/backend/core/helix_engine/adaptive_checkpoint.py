# SPDX-License-Identifier: AGPL-3.0-only
"""Mid-answer adaptive checkpoint orchestration.

The foreground generation owns the model.  A context checkpoint therefore does
everything that can be decided from already-observed artifacts, persists those
decisions, and *never* starts a training run.  The final turn ingest remains the
only place that may hand a qualified QLoRA candidate to the training scheduler.

This module intentionally does not archive the live capture session.  A long
answer may checkpoint more than once and the final post-task audit still needs
the complete tool trajectory.
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

from core.helix_event_sequence import next_event_metadata

from .capture import capture_session_key, session_steps, trajectory_from_session
from .critic import critic_from_steps
from .ingest import apply_engine_actions
from .ledger import append_record
from .pipeline import run_adaptation_pipeline


_REPLAY_SCHEMA_VERSION = "helix.adaptive-checkpoint-replay.v1"
_PROCESS_INSTANCE_ID = uuid4().hex
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_KEYS: set[str] = set()
_REPLAY_WAIT_SECONDS = 20.0
_REPLAY_POLL_SECONDS = 0.02


def _account_id() -> str:
    """Immutable authenticated-account scope, with a conservative local fallback."""
    try:
        from utils.account_context import current_account_id

        value = str(current_account_id() or "").strip()
        if value:
            return value
    except Exception:
        pass
    return "owner"


def _checkpoint_db_path() -> Path:
    """Account-private durable idempotency store for adaptive checkpoints."""
    override = os.environ.get("HELIX_ADAPTIVE_CHECKPOINT_DB", "").strip()
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
            / "adaptive-checkpoints.sqlite3"
        )
    from utils.paths import account_path

    return Path(account_path("learning/helix-engine/adaptive-checkpoints.sqlite3"))


def _open_checkpoint_db() -> sqlite3.Connection:
    path = _checkpoint_db_path()
    try:
        from utils.paths import ensure_dir
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ensure_dir(path.parent)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_checkpoint_receipts (
            dedupe_key TEXT PRIMARY KEY,
            checkpoint_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            owner_subject TEXT NOT NULL,
            session_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            resume_round INTEGER NOT NULL,
            status TEXT NOT NULL,
            phase TEXT NOT NULL,
            claim_pid INTEGER NOT NULL,
            claim_instance TEXT NOT NULL,
            recovery_json TEXT NOT NULL,
            receipt_json TEXT,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )
        """
    )
    return conn


def _bounded_text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _resume_round(telemetry: dict[str, Any], explicit: int | None) -> int:
    value: Any = explicit
    if value is None:
        value = telemetry.get("adaptive_resume_round")
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        parsed = 0
    return max(0, parsed)


def _run_id(telemetry: dict[str, Any], explicit: str | None, turn_id: str | None) -> str:
    if str(explicit or "").strip():
        return _bounded_text(explicit, 240)
    for key in ("generation_run_id", "generationRunId", "run_id", "runId"):
        if str(telemetry.get(key) or "").strip():
            return _bounded_text(telemetry.get(key), 240)
    return _bounded_text(turn_id, 240)


def _identity(
    *,
    session_id: str,
    thread_id: str | None,
    turn_id: str | None,
    run_id: str | None,
    checkpoint_id: str | None,
    checkpoint_event_seq: int | None,
    resume_round: int | None,
    telemetry: dict[str, Any],
    partial_result: str,
    subject: str,
) -> dict[str, Any]:
    account_id = _account_id()
    session = _bounded_text(session_id, 500)
    thread = _bounded_text(thread_id, 240)
    turn = _bounded_text(turn_id, 240)
    run = _run_id(telemetry, run_id, turn_id)
    round_value = _resume_round(telemetry, resume_round)
    event_seq_value: Any = checkpoint_event_seq
    if event_seq_value is None:
        event_seq_value = telemetry.get("checkpoint_event_seq")
    if event_seq_value is None:
        event_seq_value = telemetry.get("checkpointEventSeq")
    try:
        event_seq = max(0, int(event_seq_value or 0))
    except (TypeError, ValueError, OverflowError):
        event_seq = 0
    marker = telemetry.get("adaptive_checkpoint")
    marker = marker if isinstance(marker, dict) else {}
    supplied = _bounded_text(checkpoint_id, 240)
    if event_seq > 0:
        public_id = supplied or f"acp-event-{event_seq}"
        replay_authority = f"event:{event_seq}"
    elif supplied:
        public_id = supplied
        replay_authority = f"checkpoint:{supplied}"
    else:
        marker_key = {
            "reason": marker.get("reason"),
            "ratio": marker.get("ratio"),
            "trigger_tokens": marker.get("trigger_tokens"),
            "prompt_tokens": marker.get("prompt_tokens"),
            "completion_tokens": marker.get("completion_tokens"),
            "occupancy_tokens": marker.get("occupancy_tokens"),
            "segment_max_tokens": marker.get("segment_max_tokens"),
            "partial_sha256": hashlib.sha256(
                str(partial_result or "").encode("utf-8")
            ).hexdigest(),
        }
        checkpoint_material = json.dumps(
            {
                "session_id": session,
                "thread_id": thread,
                "turn_id": turn,
                "run_id": run,
                "marker": marker_key,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        public_id = "acp-" + hashlib.sha256(checkpoint_material.encode("utf-8")).hexdigest()[:24]
        replay_authority = f"fallback:{public_id}"
    scope_material = json.dumps(
        {
            "account_id": account_id,
            "session_id": session,
            "thread_id": thread,
            "turn_id": turn,
            "run_id": run,
            "replay_authority": replay_authority,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "dedupe_key": hashlib.sha256(scope_material.encode("utf-8")).hexdigest(),
        "checkpoint_id": public_id,
        "account_id": account_id,
        "owner_subject": _bounded_text(subject, 240),
        "session_id": session,
        "thread_id": thread,
        "turn_id": turn,
        "run_id": run,
        "resume_round": round_value,
        "checkpoint_event_seq": event_seq or None,
        "replay_authority": replay_authority,
    }


def _initial_recovery(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _REPLAY_SCHEMA_VERSION,
        "checkpoint_id": identity["checkpoint_id"],
        "session_id": identity["session_id"],
        "thread_id": identity["thread_id"] or None,
        "turn_id": identity["turn_id"] or None,
        "run_id": identity["run_id"] or None,
        "resume_round": identity["resume_round"],
        "checkpoint_event_seq": identity.get("checkpoint_event_seq"),
        "replay_authority": identity.get("replay_authority"),
        "phase": "claimed",
        "boundaries": {},
    }


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _claim_checkpoint(identity: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    recovery = _initial_recovery(identity)
    with _open_checkpoint_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM adaptive_checkpoint_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        if row is not None:
            conn.commit()
            return False, dict(row)
        conn.execute(
            """
            INSERT INTO adaptive_checkpoint_receipts (
                dedupe_key, checkpoint_id, account_id, owner_subject,
                session_id, thread_id, turn_id, run_id, resume_round,
                status, phase, claim_pid, claim_instance, recovery_json,
                receipt_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 'claimed', ?, ?, ?, NULL, ?, ?)
            """,
            (
                identity["dedupe_key"],
                identity["checkpoint_id"],
                identity["account_id"],
                identity["owner_subject"],
                identity["session_id"],
                identity["thread_id"],
                identity["turn_id"],
                identity["run_id"],
                identity["resume_round"],
                os.getpid(),
                _PROCESS_INSTANCE_ID,
                json.dumps(recovery, ensure_ascii=False, sort_keys=True, default=str),
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
        return True, {
            **identity,
            "status": "running",
            "phase": "claimed",
            "claim_pid": os.getpid(),
            "claim_instance": _PROCESS_INSTANCE_ID,
            "recovery_json": json.dumps(
                recovery, ensure_ascii=False, sort_keys=True, default=str
            ),
        }


def _load_checkpoint_row(dedupe_key: str) -> dict[str, Any] | None:
    with _open_checkpoint_db() as conn:
        row = conn.execute(
            "SELECT * FROM adaptive_checkpoint_receipts WHERE dedupe_key=?",
            (dedupe_key,),
        ).fetchone()
    return _row_dict(row)


def _decode_json_dict(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _checkpoint_receipt(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") not in {"completed", "recovered"}:
        return None
    receipt = _decode_json_dict(row.get("receipt_json"))
    return receipt or None


def _phase_update(
    identity: dict[str, Any],
    phase: str,
    *,
    boundaries: dict[str, dict[str, int]] | None = None,
) -> None:
    """Persist recovery metadata before/after externally visible side effects."""
    with _open_checkpoint_db() as conn:
        row = conn.execute(
            "SELECT recovery_json FROM adaptive_checkpoint_receipts WHERE dedupe_key=?",
            (identity["dedupe_key"],),
        ).fetchone()
        recovery = _decode_json_dict(row["recovery_json"] if row is not None else None)
        recovery.update(
            {
                "schema_version": _REPLAY_SCHEMA_VERSION,
                "checkpoint_id": identity["checkpoint_id"],
                "run_id": identity["run_id"] or None,
                "resume_round": identity["resume_round"],
                "phase": phase,
            }
        )
        if boundaries:
            prior = recovery.get("boundaries")
            prior = dict(prior) if isinstance(prior, dict) else {}
            prior.update(boundaries)
            recovery["boundaries"] = prior
        conn.execute(
            """
            UPDATE adaptive_checkpoint_receipts
               SET phase=?, recovery_json=?, updated_at_ms=?
             WHERE dedupe_key=? AND status='running' AND claim_instance=?
            """,
            (
                phase,
                json.dumps(recovery, ensure_ascii=False, sort_keys=True, default=str),
                int(time.time() * 1000),
                identity["dedupe_key"],
                _PROCESS_INSTANCE_ID,
            ),
        )


def _canonical_receipt(receipt: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str)
    decoded = json.loads(raw)
    return raw, decoded


def _persist_completed_receipt(
    identity: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    raw, canonical = _canonical_receipt(receipt)
    with _open_checkpoint_db() as conn:
        cursor = conn.execute(
            """
            UPDATE adaptive_checkpoint_receipts
               SET status='completed', phase='completed', receipt_json=?, updated_at_ms=?
             WHERE dedupe_key=? AND status='running' AND claim_instance=?
            """,
            (
                raw,
                int(time.time() * 1000),
                identity["dedupe_key"],
                _PROCESS_INSTANCE_ID,
            ),
        )
        if cursor.rowcount != 1:
            row = conn.execute(
                "SELECT * FROM adaptive_checkpoint_receipts WHERE dedupe_key=?",
                (identity["dedupe_key"],),
            ).fetchone()
            existing = _checkpoint_receipt(dict(row)) if row is not None else None
            if existing is not None:
                return existing
            raise RuntimeError("adaptive checkpoint durable receipt ownership was lost")
    return canonical


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


def _active_locally(dedupe_key: str) -> bool:
    with _ACTIVE_LOCK:
        return dedupe_key in _ACTIVE_KEYS


def _orphaned(row: dict[str, Any]) -> bool:
    pid = int(row.get("claim_pid") or 0)
    instance = str(row.get("claim_instance") or "")
    if pid == os.getpid():
        return instance != _PROCESS_INSTANCE_ID or not _active_locally(str(row.get("dedupe_key") or ""))
    return not _pid_alive(pid)


def _recovery_receipt(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    recovery = _decode_json_dict(row.get("recovery_json"))
    boundaries = recovery.get("boundaries") if isinstance(recovery.get("boundaries"), dict) else {}
    receipt: dict[str, Any] = {
        "available": False,
        "schema_version": "helix.adaptive-checkpoint.v1",
        "checkpoint_id": str(row.get("checkpoint_id") or ""),
        "run_id": str(row.get("run_id") or "") or None,
        "resume_round": int(row.get("resume_round") or 0),
        "checkpoint_event_seq": recovery.get("checkpoint_event_seq"),
        "training_deferred": True,
        "qlora_candidate": False,
        "qlora_eligible": False,
        "actions": [],
        "recovery": {
            "status": "interrupted_checkpoint_recovered",
            "reason": reason,
            "phase": str(row.get("phase") or recovery.get("phase") or "running"),
            "boundaries": boundaries,
        },
    }
    for name in ("start", "audit", "resume"):
        boundary = boundaries.get(name) if isinstance(boundaries, dict) else None
        if isinstance(boundary, dict):
            sequence = int(boundary.get("sequence") or 0)
            created_at_ms = int(boundary.get("created_at_ms") or 0)
            if sequence > 0 and created_at_ms > 0:
                receipt[f"{name}_sequence"] = sequence
                receipt[f"{name}_created_at_ms"] = created_at_ms
    return receipt


def _persist_recovered_receipt(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    receipt = _recovery_receipt(row, reason=reason)
    raw, canonical = _canonical_receipt(receipt)
    with _open_checkpoint_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM adaptive_checkpoint_receipts WHERE dedupe_key=?",
            (row["dedupe_key"],),
        ).fetchone()
        if current is not None:
            current_dict = dict(current)
            existing = _checkpoint_receipt(current_dict)
            if existing is not None:
                conn.commit()
                return existing
            conn.execute(
                """
                UPDATE adaptive_checkpoint_receipts
                   SET status='recovered', phase='recovered', receipt_json=?, updated_at_ms=?
                 WHERE dedupe_key=? AND status='running'
                """,
                (raw, int(time.time() * 1000), row["dedupe_key"]),
            )
        conn.commit()
    return canonical


def _existing_or_wait(row: dict[str, Any]) -> dict[str, Any]:
    receipt = _checkpoint_receipt(row)
    if receipt is not None:
        return receipt
    deadline = time.monotonic() + _REPLAY_WAIT_SECONDS
    current = row
    while current.get("status") == "running" and not _orphaned(current):
        if time.monotonic() >= deadline:
            # The owner is still live. Do not steal the claim or repeat side effects.
            return _recovery_receipt(current, reason="duplicate_request_wait_timeout_owner_still_live")
        time.sleep(_REPLAY_POLL_SECONDS)
        current = _load_checkpoint_row(str(row["dedupe_key"])) or current
        receipt = _checkpoint_receipt(current)
        if receipt is not None:
            return receipt
    receipt = _checkpoint_receipt(current)
    if receipt is not None:
        return receipt
    return _persist_recovered_receipt(current, reason="orphaned_after_process_restart_or_crash")


def _storage_unavailable_receipt(identity: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "available": False,
        "schema_version": "helix.adaptive-checkpoint.v1",
        "checkpoint_id": identity["checkpoint_id"],
        "run_id": identity["run_id"] or None,
        "resume_round": identity["resume_round"],
        "checkpoint_event_seq": identity.get("checkpoint_event_seq"),
        "training_deferred": True,
        "qlora_candidate": False,
        "qlora_eligible": False,
        "actions": [],
        "recovery": {
            "status": "durable_store_unavailable",
            "reason": f"{type(exc).__name__}: {exc}"[:800],
        },
    }


def _mem0_retrieval_receipt() -> dict[str, Any]:
    """Describe retrieval ownership without adding a vector search to this hot path."""
    return {
        "status": "preflight_owned",
        "owner": "chat_preflight",
        "performed_at_checkpoint": False,
        "reason": (
            "Mem0 retrieval is owned by chat preflight before inference; the >=80% "
            "checkpoint records provenance and performs only the bounded update."
        ),
    }


def _event_boundary(scope_id: str) -> dict[str, int]:
    """Best-effort ordering metadata from the shared observable event clock."""
    try:
        metadata = next_event_metadata(scope_id)
        sequence = int(metadata.get("sequence") or 0)
        created_at_ms = int(metadata.get("created_at_ms") or 0)
    except Exception:
        return {}
    if sequence <= 0 or created_at_ms <= 0:
        return {}
    return {"sequence": sequence, "created_at_ms": created_at_ms}


def _temporary_skill_receipt(thread_id: str | None) -> dict[str, Any]:
    """Inspect current thread skills without deciding their final lifecycle mid-answer."""
    receipt: dict[str, Any] = {
        "inspected": False,
        "temporary_skill_count": 0,
        "temporary_skills": [],
        "skill_disposition": "defer_until_task_complete",
        "final_authority": "final_turn_ingest",
    }
    if not str(thread_id or "").strip():
        receipt["reason"] = "no_thread_id"
        return receipt
    try:
        from core.inference.skills import list_temp_skills

        temporary = list_temp_skills(str(thread_id))
        receipt["inspected"] = True
        receipt["temporary_skill_count"] = len(temporary)
        receipt["temporary_skills"] = [
            str(item.get("name") or "")[:120]
            for item in temporary[:16]
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        receipt["reason"] = (
            "Mid-answer checkpoints keep temporary skills available; promotion or discard "
            "requires the completed task outcome."
        )
    except Exception as exc:  # noqa: BLE001 -- skill inspection is advisory only
        receipt["error"] = f"{type(exc).__name__}: {exc}"[:800]
        receipt["reason"] = "temporary_skill_inspection_failed_fail_open"
    return receipt


def _qlora_consideration(
    adaptive: dict[str, Any], adaptation: dict[str, Any]
) -> dict[str, Any]:
    """Combine shadow candidacy with Hermes metadata without authorizing training."""
    shadow_decisions = (
        adaptive.get("shadow_decisions")
        if isinstance(adaptive.get("shadow_decisions"), dict)
        else {}
    )
    shadow = (
        shadow_decisions.get("QLORA_CANDIDATE")
        if isinstance(shadow_decisions.get("QLORA_CANDIDATE"), dict)
        else {}
    )
    shadow_candidate = str(shadow.get("choice") or "").upper() == "TAKE"
    adaptation_candidate = adaptation.get("action") == "QLORA_CANDIDATE"
    candidate = bool(shadow_candidate or adaptation_candidate)
    # Eligibility remains whatever the closed-loop Hermes path established. The
    # checkpoint never fabricates a corrected target or overrides receipt gating.
    eligible = bool(adaptation_candidate and adaptation.get("qlora_eligible") is True)
    return {
        "considered": True,
        "metadata_available": bool(shadow or adaptation),
        "candidate": candidate,
        "eligible": eligible,
        "shadow_candidate": shadow_candidate,
        "adaptation_candidate": adaptation_candidate,
        "shadow_decision": shadow,
        "adaptation_action": str(adaptation.get("action") or ""),
        "advisory_only": True,
        "training_deferred": True,
        "eligibility_source": "hermes_verified_target_gate",
        "verified_target_receipt_gate": "preserved_final_ingest_authority",
    }


def run_adaptive_checkpoint(
    *,
    session_id: str,
    prompt: str,
    partial_result: str,
    thread_id: str | None,
    turn_id: str | None,
    model_id: str,
    effective_model_id: str = "",
    adapter_state: bool | None = None,
    telemetry: dict[str, Any] | None = None,
    subject: str = "local",
    checkpoint_id: str | None = None,
    checkpoint_event_seq: int | None = None,
    run_id: str | None = None,
    resume_round: int | None = None,
) -> dict[str, Any]:
    """Run the non-destructive learn/compact checkpoint for one live answer.

    The order is deliberate: memory commit first, then Helix/Hermes analysis and
    skill actions.  If vector memory is unavailable the bounded graph write still
    succeeds and the cycle continues fail-open. QLoRA candidacy is considered from
    existing Hermes and shadow metadata but is reported as advisory/deferred; no
    training worker is started here.
    """

    observed = dict(telemetry or {})
    identity = _identity(
        session_id=session_id,
        thread_id=thread_id,
        turn_id=turn_id,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_event_seq=checkpoint_event_seq,
        resume_round=resume_round,
        telemetry=observed,
        partial_result=partial_result,
        subject=subject,
    )
    try:
        owns_claim, prior = _claim_checkpoint(identity)
    except Exception as exc:  # durable idempotency is required before any side effect
        return _storage_unavailable_receipt(identity, exc)
    if not owns_claim:
        return _existing_or_wait(prior)

    try:
        return _run_claimed_adaptive_checkpoint(
            identity=identity,
            session_id=session_id,
            prompt=prompt,
            partial_result=partial_result,
            thread_id=thread_id,
            turn_id=turn_id,
            model_id=model_id,
            effective_model_id=effective_model_id,
            adapter_state=adapter_state,
            telemetry=observed,
            subject=subject,
        )
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_KEYS.discard(identity["dedupe_key"])


def _run_claimed_adaptive_checkpoint(
    *,
    identity: dict[str, Any],
    session_id: str,
    prompt: str,
    partial_result: str,
    thread_id: str | None,
    turn_id: str | None,
    model_id: str,
    effective_model_id: str,
    adapter_state: bool | None,
    telemetry: dict[str, Any],
    subject: str,
) -> dict[str, Any]:
    """Execute side effects for the unique durable checkpoint claim owner."""

    capture_key = capture_session_key(session_id, thread_id, turn_id)
    _phase_update(identity, "start_boundary_pending")
    start_boundary = _event_boundary(capture_key)
    _phase_update(
        identity,
        "start_boundary_recorded",
        boundaries={"start": start_boundary} if start_boundary else None,
    )
    steps = session_steps(capture_key)
    observed = dict(telemetry or {})
    observed["adaptive_checkpoint"] = True
    observed["checkpoint_phase"] = "mid_answer"
    observed.setdefault("objective_verified", False)
    trajectory_id = str(turn_id or observed.get("trajectory_id") or capture_key)
    behavioral_model_id = str(effective_model_id or model_id or "").strip()

    critic = critic_from_steps(
        steps,
        # A checkpoint is intentionally not the completed task.  Marking it full
        # would let a partial answer masquerade as objective task completion.
        finished="partial",
    )
    trajectory = trajectory_from_session(
        capture_key,
        prompt_state=prompt,
        final_result=partial_result,
        latency_ms=float(observed.get("latency_ms") or 0),
        prompt_tokens=int(observed.get("prompt_tokens") or observed.get("promptTokens") or 0),
        completion_tokens=int(
            observed.get("completion_tokens") or observed.get("completionTokens") or 0
        ),
        verified=False,
        extras={
            "critic": critic.as_dict(),
            "telemetry": observed,
            "model_id": behavioral_model_id,
            "base_model_id": model_id,
            "adapter_state": adapter_state,
            "objective_verified": False,
            # Reuse the logical turn id so repeated checkpoints cannot manufacture
            # recurrence evidence for QLoRA admission.
            "trajectory_id": trajectory_id,
            "thread_id": thread_id,
            "capture_session_id": capture_key,
            "adaptive_checkpoint": True,
        },
    )

    memory_result: dict[str, Any]
    _phase_update(identity, "mem0_update_pending")
    try:
        from core.memory.mem0_store import add_experience

        memory_result = add_experience(
            subject,
            (
                f"User objective: {prompt[:3000]}\n\n"
                f"Checkpoint progress: {partial_result[-4000:]}"
            ),
            thread_id=thread_id,
            kind="adaptive-checkpoint",
            title=(prompt.strip()[:160] or "Adaptive context checkpoint"),
        )
    except Exception as exc:  # noqa: BLE001 -- learning must never strand the answer
        memory_result = {"stored": False, "reason": f"{type(exc).__name__}: {exc}"[:800]}
    _phase_update(identity, "mem0_update_completed")

    analysis_error = ""
    _phase_update(identity, "analysis_pending")
    try:
        analysis = run_adaptation_pipeline(trajectory, self_audit=None, claims=None)
    except Exception as exc:  # noqa: BLE001 -- checkpoint analysis must not strand the answer
        analysis_error = f"{type(exc).__name__}: {exc}"[:800]
        analysis = {
            "adaptive_cycle": {
                "schema_version": "helix.closed-loop.v1",
                "available": False,
                "fail_open": True,
                "error": analysis_error,
            }
        }
    _phase_update(identity, "analysis_completed")

    adaptive = analysis.get("adaptive_cycle") if isinstance(analysis.get("adaptive_cycle"), dict) else {}
    adaptation = adaptive.get("adaptation") if isinstance(adaptive.get("adaptation"), dict) else {}
    helix_hermes_receipt = {
        "analysis_attempted": True,
        "analysis_performed": not bool(analysis_error),
        "adaptive_cycle_available": adaptive.get("available") is not False,
        "hermes_adjudication_performed": bool(adaptation),
        "adaptation_action": str(adaptation.get("action") or ""),
    }
    if analysis_error:
        helix_hermes_receipt["error"] = analysis_error

    actions: list[str] = []
    if not analysis_error:
        _phase_update(identity, "engine_actions_pending")
        try:
            actions = apply_engine_actions(
                analysis,
                critic,
                prompt=prompt,
                thread_id=thread_id,
                subject=subject,
            )
        except Exception as exc:  # noqa: BLE001 -- deterministic checkpoint must fail open
            actions = ["adaptive_action_error"]
            observed["adaptive_action_error"] = f"{type(exc).__name__}: {exc}"[:800]
        _phase_update(identity, "engine_actions_completed")

    _phase_update(identity, "temporary_skill_inspection_pending")
    skill_receipt = _temporary_skill_receipt(thread_id)
    _phase_update(identity, "temporary_skill_inspection_completed")
    preflight = observed.get("preflight") if isinstance(observed.get("preflight"), dict) else {}
    preflight_mem0 = preflight.get("memory") if isinstance(preflight.get("memory"), dict) else (preflight.get("mem0") if isinstance(preflight.get("mem0"), dict) else None)
    preflight_skills = preflight.get("skills") if isinstance(preflight.get("skills"), dict) else None
    if preflight_skills is not None:
        skill_receipt["preflight"] = dict(preflight_skills)
    qlora_receipt = _qlora_consideration(adaptive, adaptation)
    qlora_candidate = bool(qlora_receipt["candidate"])
    qlora_eligible = bool(qlora_receipt["eligible"])
    # Even an eligible candidate is only staged here.  Starting training while
    # the answer still owns inference memory can evict the resident model/OOM.
    if qlora_candidate:
        actions.append("defer_qlora_until_answer_complete")

    mechanisms = {
        "mem0": {
            "retrieval": {
                **_mem0_retrieval_receipt(),
                **({"receipt": dict(preflight_mem0)} if preflight_mem0 is not None else {}),
            },
            "update": {"attempted": True, "result": memory_result},
        },
        "helix_hermes": helix_hermes_receipt,
        "temporary_skill": skill_receipt,
        "qlora": qlora_receipt,
    }
    # Persist an exact clock point for the completed checkpoint analysis itself.
    # Start/resume alone prove that a pause occurred but do not place the
    # Helix/Hermes policy receipt inside that causal interval.
    _phase_update(identity, "audit_boundary_pending")
    audit_boundary = _event_boundary(capture_key)
    _phase_update(
        identity,
        "audit_boundary_recorded",
        boundaries={"audit": audit_boundary} if audit_boundary else None,
    )
    _phase_update(identity, "resume_boundary_pending")
    resume_boundary = _event_boundary(capture_key)
    _phase_update(
        identity,
        "resume_boundary_recorded",
        boundaries={"resume": resume_boundary} if resume_boundary else None,
    )

    record = {
        "schema_version": "helix.adaptive-checkpoint.v1",
        "checkpoint_id": identity["checkpoint_id"],
        "trajectory_id": trajectory_id,
        "session_id": capture_key,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "run_id": identity["run_id"] or None,
        "resume_round": identity["resume_round"],
        "checkpoint_event_seq": identity.get("checkpoint_event_seq"),
        "model_id": behavioral_model_id,
        "memory_stored": bool(memory_result.get("stored")),
        "mechanisms": mechanisms,
        "adaptation": adaptation,
        "actions": list(dict.fromkeys(actions)),
        "qlora_candidate": qlora_candidate,
        "qlora_eligible": qlora_eligible,
        "training_deferred": True,
        "created_at_ms": int(
            resume_boundary.get("created_at_ms")
            or start_boundary.get("created_at_ms")
            or time.time() * 1000
        ),
    }
    if start_boundary:
        record["start_sequence"] = start_boundary["sequence"]
        record["start_created_at_ms"] = start_boundary["created_at_ms"]
    if audit_boundary:
        record["audit_sequence"] = audit_boundary["sequence"]
        record["audit_created_at_ms"] = audit_boundary["created_at_ms"]
    if resume_boundary:
        record["resume_sequence"] = resume_boundary["sequence"]
        record["resume_created_at_ms"] = resume_boundary["created_at_ms"]
    _phase_update(identity, "adaptive_ledger_append_pending")
    try:
        append_record("adaptive-checkpoints", record)
    except Exception:
        # A ledger failure cannot prevent the foreground answer from resuming.
        pass
    _phase_update(identity, "adaptive_ledger_append_completed")
    result = {
        "available": True,
        "schema_version": "helix.adaptive-checkpoint.v1",
        "checkpoint_id": identity["checkpoint_id"],
        "run_id": identity["run_id"] or None,
        "resume_round": identity["resume_round"],
        "checkpoint_event_seq": identity.get("checkpoint_event_seq"),
        "memory": memory_result,
        "mechanisms": mechanisms,
        "adaptive_cycle": adaptive,
        "actions": record["actions"],
        "training_deferred": True,
        "qlora_candidate": qlora_candidate,
        "qlora_eligible": qlora_eligible,
    }
    if start_boundary:
        result["start_sequence"] = start_boundary["sequence"]
        result["start_created_at_ms"] = start_boundary["created_at_ms"]
    if audit_boundary:
        result["audit_sequence"] = audit_boundary["sequence"]
        result["audit_created_at_ms"] = audit_boundary["created_at_ms"]
    if resume_boundary:
        result["resume_sequence"] = resume_boundary["sequence"]
        result["resume_created_at_ms"] = resume_boundary["created_at_ms"]
    result["control"] = {
        "resume_round": identity["resume_round"],
        "run_id": identity["run_id"] or None,
        "checkpoint_event_seq": identity.get("checkpoint_event_seq"),
        "boundaries": {
            **({"start": start_boundary} if start_boundary else {}),
            **({"audit": audit_boundary} if audit_boundary else {}),
            **({"resume": resume_boundary} if resume_boundary else {}),
        },
    }
    try:
        return _persist_completed_receipt(identity, result)
    except Exception as exc:
        row = _load_checkpoint_row(identity["dedupe_key"])
        if row is not None:
            return _persist_recovered_receipt(
                row,
                reason=f"completed_receipt_persist_failed:{type(exc).__name__}",
            )
        return _storage_unavailable_receipt(identity, exc)
