# SPDX-License-Identifier: AGPL-3.0-only

"""Focused account-local storage gates for disabled ObservationPack admission."""

from __future__ import annotations

import hashlib
import json

import pytest

from storage import chat_generation_runs_db as runs_db
from storage import studio_db

_MIN_PLAINTEXT_BYTES = 10 * 1024
_MAX_PLAINTEXT_BYTES = 8 * 1024 * 1024
_CIPHERTEXT_BYTES = _MIN_PLAINTEXT_BYTES + 28


def _seed_thread(thread_id: str = "thread-1", user_id: str = "user-1") -> None:
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "Observation storage",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": user_id,
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": "Collect evidence"}],
            "createdAt": 2,
        }
    )


def _create_run(
    run_id: str = "run-1",
    *,
    thread_id: str = "thread-1",
    user_id: str = "user-1",
    observation_pack_version: int = 0,
):
    _seed_thread(thread_id, user_id)
    return runs_db.create_run(
        run_id=run_id,
        owner_subject="alice",
        thread_id=thread_id,
        user_message_id=user_id,
        assistant_message_id=f"assistant-{run_id}",
        request_payload={"model": "local", "messages": [], "stream": True},
        observation_pack_version=observation_pack_version,
    )


def _start_execution(
    run_id: str = "run-1",
    *,
    thread_id: str = "thread-1",
    execution_id: str = "exec-1",
    tool_name: str = "lookup",
    arguments: dict | None = None,
):
    arguments = arguments or {"needle": "alpha"}
    run = runs_db.get_run(run_id)
    assert run is not None
    worker_token = runs_db.get_worker_token(run_id)
    assert worker_token
    assert runs_db.mark_running(run_id, worker_token)
    checkpoint = {"version": 1, "conversation": [], "controller": {}, "remaining_calls": []}
    checkpoint_json = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    checkpoint_digest = hashlib.sha256(checkpoint_json.encode()).hexdigest()
    claimed = runs_db.claim_ungated_tool_execution(
        run_id,
        execution_id,
        worker_token=worker_token,
        backend_account_id="owner",
        session_id="session-1",
        thread_id=thread_id,
        tool_name=tool_name,
        tool_call_id=f"call-{execution_id}",
        card_call_id=f"card-{execution_id}",
        arguments=arguments,
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    assert claimed is not None
    started = runs_db.mark_ungated_tool_execution_started(
        run_id,
        execution_id,
        worker_token=worker_token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    assert started is not None
    payload_json = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    arguments_fingerprint = hashlib.sha256(payload_json.encode()).hexdigest()
    return {
        "worker_token": worker_token,
        "claim_token": claimed["claimToken"],
        "payload_json": payload_json,
        "arguments_fingerprint": arguments_fingerprint,
        "checkpoint_digest": checkpoint_digest,
        "tool_call_id": f"call-{execution_id}",
        "card_call_id": f"card-{execution_id}",
    }


def _start_approved_execution():
    _create_run(observation_pack_version=1)
    worker_token = runs_db.get_worker_token("run-1")
    assert worker_token
    assert runs_db.mark_running("run-1", worker_token)
    arguments = {"needle": "alpha"}
    payload_json = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    arguments_fingerprint = hashlib.sha256(payload_json.encode()).hexdigest()
    checkpoint = {
        "version": 1,
        "backend": "gguf",
        "conversation": [],
        "current_call": {
            "tool_call": {
                "id": "call-exec-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps(arguments),
                },
            },
            "card_call_id": "card-exec-1",
            "provenance": {},
        },
        "remaining_calls": [],
        "controller": {},
    }
    proposal = {
        "approval_id": "approval-1",
        "session_id": "session-1",
        "tool_name": "lookup",
        "tool_call_id": "call-exec-1",
        "card_call_id": "card-exec-1",
        "execution_id": "exec-1",
        "arguments": arguments,
        "arguments_fingerprint": arguments_fingerprint,
        "checkpoint_version": 1,
        "resume_checkpoint": checkpoint,
        "proposed_at": 100,
        "expires_at": 10_000_000_000_000,
    }
    runs_db.append_events_with_tool_proposal("run-1", worker_token, [], proposal)
    runs_db.decide_tool_approval(
        "run-1",
        "approval-1",
        owner_subject="alice",
        session_id="session-1",
        decision="allow",
    )
    claimed = runs_db.claim_tool_execution(
        "run-1", "approval-1", worker_token=worker_token
    )
    assert claimed is not None
    started = runs_db.mark_tool_execution_started(
        "run-1",
        "approval-1",
        worker_token=worker_token,
        claim_token=claimed["claimToken"],
    )
    assert started is not None
    staged = runs_db.stage_observation_binding(
        account_id="owner",
        run_id="run-1",
        execution_id="exec-1",
        owner_subject="alice",
        thread_id="thread-1",
        session_id="session-1",
        tool_name="lookup",
        tool_call_id="call-exec-1",
        card_call_id="card-exec-1",
        authority_kind="approved",
        approval_id="approval-1",
        payload_json=payload_json,
        arguments_fingerprint=arguments_fingerprint,
        claim_token=claimed["claimToken"],
        worker_token=worker_token,
        plaintext_length=_MIN_PLAINTEXT_BYTES,
        plaintext_digest="a" * 64,
        reserved_bytes=_CIPHERTEXT_BYTES,
        quota_bytes=64 * 1024,
    )
    identity = {
        "worker_token": worker_token,
        "claim_token": claimed["claimToken"],
        "payload_json": payload_json,
        "arguments_fingerprint": arguments_fingerprint,
        "tool_call_id": "call-exec-1",
        "card_call_id": "card-exec-1",
    }
    return staged, identity


def _stage(
    run_id: str = "run-1",
    execution_id: str = "exec-1",
    *,
    identity: dict | None = None,
    **overrides,
):
    if identity is None:
        identity = _start_execution(
            run_id,
            thread_id=overrides.get("thread_id", "thread-1"),
            execution_id=execution_id,
        )
    values = {
        "account_id": "owner",
        "run_id": run_id,
        "execution_id": execution_id,
        "owner_subject": "alice",
        "thread_id": "thread-1",
        "session_id": "session-1",
        "tool_name": "lookup",
        "tool_call_id": identity["tool_call_id"],
        "card_call_id": identity["card_call_id"],
        "authority_kind": "ungated",
        "payload_json": identity["payload_json"],
        "arguments_fingerprint": identity["arguments_fingerprint"],
        "claim_token": identity["claim_token"],
        "worker_token": identity["worker_token"],
        "plaintext_length": _MIN_PLAINTEXT_BYTES,
        "plaintext_digest": "a" * 64,
        "reserved_bytes": _CIPHERTEXT_BYTES,
        "quota_bytes": 64 * 1024,
    }
    values.update(overrides)
    return runs_db.stage_observation_binding(**values), identity


def _make_available_fixture(binding: dict, *, finished_event_seq: int) -> dict:
    """Arrange an authorized row for read-order tests without calling promotion.

    ObservationPack publication is intentionally disabled until the durable
    finish transaction owns the state transition.  This SQL-only fixture is
    therefore limited to testing ``find_available_observation_binding`` and
    does not represent a supported production promotion path.
    """
    conn = runs_db._connect()
    try:
        cursor = conn.execute(
            """UPDATE chat_generation_observation_bindings
                  SET state='available', key_id=?, blob_relpath=?,
                      plaintext_length=?, plaintext_digest=?, ciphertext_length=?,
                      ciphertext_digest=?, blob_dev=?, blob_ino=?, projection=?,
                      finished_event_seq=?, unavailable_reason=NULL, updated_at=?
                WHERE binding_id=? AND state='staged'""",
            (
                "observation-key-v1:sha256:" + "c" * 64,
                "blobs/" + binding["binding_id"] + ".blob",
                _MIN_PLAINTEXT_BYTES,
                "a" * 64,
                _CIPHERTEXT_BYTES,
                "d" * 64,
                1,
                1,
                "projection",
                finished_event_seq,
                1,
                binding["binding_id"],
            ),
        )
        assert cursor.rowcount == 1
        conn.commit()
    finally:
        conn.close()
    available = runs_db.get_observation_binding(
        binding["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert available is not None
    assert available["state"] == "available"
    return available


def _promotion_kwargs(binding: dict, *, finished_event_seq: int) -> dict:
    return {
        "binding_id": binding["binding_id"],
        "account_id": "owner",
        "run_id": "run-1",
        "execution_id": "exec-1",
        "owner_subject": "alice",
        "thread_id": "thread-1",
        "key_id": "observation-key-v1:sha256:" + "e" * 64,
        "blob_relpath": "blobs/" + binding["binding_id"] + ".blob",
        "plaintext_length": _MIN_PLAINTEXT_BYTES,
        "plaintext_digest": "a" * 64,
        "ciphertext_length": _CIPHERTEXT_BYTES,
        "ciphertext_digest": "f" * 64,
        "blob_dev": 1,
        "blob_ino": 1,
        "projection": "projection",
        "finished_event_seq": finished_event_seq,
    }


def _finish_candidate(
    binding: dict,
    identity: dict,
    *,
    projection: str = "projection",
    fallback_result: str = "fallback",
    **overrides,
) -> dict:
    values = {
        "binding_id": binding["binding_id"],
        "account_id": "owner",
        "source_run_id": "run-1",
        "thread_id": "thread-1",
        "execution_id": "exec-1",
        "tool_name": "lookup",
        "tool_call_id": identity["tool_call_id"],
        "approval_id": None,
        "claim_token": identity["claim_token"],
        "worker_id": identity["worker_token"],
        "arguments_fingerprint": identity["arguments_fingerprint"],
        "handle": "obs:v1:test:sha256:" + "a" * 64,
        "format_version": 1,
        "canonicalization_version": "helix.tool-text.v1",
        "projection_version": "helix.observation-projection.v1",
        "plaintext_length": _MIN_PLAINTEXT_BYTES,
        "plaintext_digest": "a" * 64,
        "ciphertext_length": _CIPHERTEXT_BYTES,
        "ciphertext_digest": "d" * 64,
        "key_id": "observation-key-v1:sha256:" + "c" * 64,
        "blob_relpath": "blobs/" + binding["binding_id"] + ".blob",
        "blob_dev": 1,
        "blob_ino": 1,
        "projection": projection,
        "fallback_result": fallback_result,
        "is_error": False,
    }
    values.update(overrides)
    return values


def test_run_pin_is_additive_backend_owned_and_idempotent():
    run, created = _create_run(observation_pack_version=1)
    assert created is True
    assert run["observationPackVersion"] == 1

    replay, replay_created = runs_db.create_run(
        run_id="run-1",
        owner_subject="alice",
        thread_id="thread-1",
        user_message_id="user-1",
        assistant_message_id="assistant-run-1",
        request_payload={"model": "local", "messages": [], "stream": True},
        observation_pack_version=0,
    )
    assert replay_created is False
    assert replay["observationPackVersion"] == 1

    default_run, _ = _create_run(
        "run-default", thread_id="thread-default", user_id="user-default"
    )
    assert default_run["observationPackVersion"] == 0
    conn = runs_db._connect()
    try:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(chat_generation_runs)")}
        assert columns["observation_pack_version"][3] == 1
        assert columns["observation_pack_version"][4] == "0"
        binding_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(chat_generation_observation_bindings)"
            )
        }
        assert {
            "binding_id",
            "account_id",
            "run_id",
            "execution_id",
            "owner_subject",
            "thread_id",
            "tool_name",
            "tool_call_id",
            "card_call_id",
            "approval_id",
            "payload_json",
            "claim_token",
            "worker_token",
            "state",
            "format_version",
            "canonicalization_version",
            "projection_version",
            "plaintext_length",
            "plaintext_digest",
            "ciphertext_length",
            "ciphertext_digest",
            "key_id",
            "blob_relpath",
            "blob_dev",
            "blob_ino",
            "projection",
            "reserved_bytes",
            "finished_event_seq",
            "created_at",
            "updated_at",
            "unavailable_reason",
        } <= binding_columns
    finally:
        conn.close()


@pytest.mark.parametrize("value", [None, -1, 2, True, "1"])
def test_run_pin_rejects_values_outside_backend_enum(value):
    _seed_thread()
    with pytest.raises(ValueError, match="observation_pack_version"):
        runs_db.create_run(
            run_id="bad-pin",
            owner_subject="alice",
            thread_id="thread-1",
            user_message_id="user-1",
            assistant_message_id="assistant-bad-pin",
            request_payload={"model": "local"},
            observation_pack_version=value,
        )


def test_stage_is_identity_bound_idempotent_and_does_not_increase_replay_reservation():
    _create_run(observation_pack_version=1)
    first, identity = _stage()
    replay = runs_db.stage_observation_binding(
        account_id="owner",
        run_id="run-1",
        execution_id="exec-1",
        owner_subject="alice",
        thread_id="thread-1",
        session_id="session-1",
        tool_name="lookup",
        tool_call_id=identity["tool_call_id"],
        card_call_id=identity["card_call_id"],
        authority_kind="ungated",
        payload_json=identity["payload_json"],
        arguments_fingerprint=identity["arguments_fingerprint"],
        claim_token=identity["claim_token"],
        worker_token=identity["worker_token"],
        plaintext_length=_MIN_PLAINTEXT_BYTES,
        plaintext_digest="a" * 64,
        reserved_bytes=_CIPHERTEXT_BYTES,
        quota_bytes=_CIPHERTEXT_BYTES,
    )
    assert replay["binding_id"] == first["binding_id"]
    assert replay["reserved_bytes"] == _CIPHERTEXT_BYTES

    # A replay with a changed immutable reservation is a conflict, not a new
    # reservation and not an authority upgrade.
    with pytest.raises(runs_db.ObservationBindingConflictError, match="plaintext_digest"):
        runs_db.stage_observation_binding(
            account_id="owner",
            run_id="run-1",
            execution_id="exec-1",
            owner_subject="alice",
            thread_id="thread-1",
            session_id="session-1",
            tool_name="lookup",
            tool_call_id=identity["tool_call_id"],
            card_call_id=identity["card_call_id"],
            authority_kind="ungated",
            payload_json=identity["payload_json"],
            arguments_fingerprint=identity["arguments_fingerprint"],
            claim_token=identity["claim_token"],
            worker_token=identity["worker_token"],
            plaintext_length=_MIN_PLAINTEXT_BYTES,
            plaintext_digest="b" * 64,
            reserved_bytes=_CIPHERTEXT_BYTES,
            quota_bytes=1024 * 1024,
        )

    with pytest.raises(runs_db.ObservationBindingFencedError, match="identity"):
        runs_db.stage_observation_binding(
            account_id="owner",
            run_id="run-1",
            execution_id="exec-1",
            owner_subject="alice",
            thread_id="wrong-thread",
            session_id="session-1",
            tool_name="lookup",
            tool_call_id=identity["tool_call_id"],
            card_call_id=identity["card_call_id"],
            authority_kind="ungated",
            payload_json=identity["payload_json"],
            arguments_fingerprint=identity["arguments_fingerprint"],
            claim_token=identity["claim_token"],
            worker_token=identity["worker_token"],
            plaintext_length=_MIN_PLAINTEXT_BYTES,
            plaintext_digest="a" * 64,
            reserved_bytes=_CIPHERTEXT_BYTES,
            quota_bytes=1024,
        )


def test_stage_enforces_quota_and_account_context():
    _create_run(observation_pack_version=1)
    first, identity = _stage(
        reserved_bytes=_CIPHERTEXT_BYTES,
        quota_bytes=_CIPHERTEXT_BYTES,
    )
    # Replaying an existing identity does not re-check quota as a new reserve.
    replay = runs_db.stage_observation_binding(
        account_id="owner",
        run_id="run-1",
        execution_id="exec-1",
        owner_subject="alice",
        thread_id="thread-1",
        session_id="session-1",
        tool_name="lookup",
        tool_call_id=identity["tool_call_id"],
        card_call_id=identity["card_call_id"],
        authority_kind="ungated",
        payload_json=identity["payload_json"],
        arguments_fingerprint=identity["arguments_fingerprint"],
        claim_token=identity["claim_token"],
        worker_token=identity["worker_token"],
        plaintext_length=_MIN_PLAINTEXT_BYTES,
        plaintext_digest="a" * 64,
        reserved_bytes=_CIPHERTEXT_BYTES,
        quota_bytes=1,
    )
    assert replay["binding_id"] == first["binding_id"]
    assert identity["claim_token"]

    _create_run(
        "run-2",
        thread_id="thread-2",
        user_id="user-2",
        observation_pack_version=1,
    )
    with pytest.raises(runs_db.ObservationQuotaExceededError):
        _stage(
            "run-2",
            "exec-2",
            thread_id="thread-2",
            reserved_bytes=_CIPHERTEXT_BYTES,
            quota_bytes=_CIPHERTEXT_BYTES,
        )

    with pytest.raises(runs_db.ObservationBindingFencedError, match="account"):
        runs_db.stage_observation_binding(
            account_id="another-account",
            run_id="run-1",
            execution_id="exec-cross",
            owner_subject="alice",
            thread_id="thread-1",
            tool_name="lookup",
            payload_json="{}",
            arguments_fingerprint="b" * 64,
            reserved_bytes=1,
            quota_bytes=1024,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("binding_id", "A" * 64),
        ("binding_id", True),
        ("format_version", 2),
        ("format_version", True),
        ("canonicalization_version", "other"),
        ("projection_version", "other"),
        ("plaintext_length", None),
        ("plaintext_length", True),
        ("plaintext_length", _MIN_PLAINTEXT_BYTES - 1),
        ("plaintext_length", _MAX_PLAINTEXT_BYTES + 1),
        ("plaintext_digest", "A" * 64),
        ("plaintext_digest", "g" * 64),
        ("plaintext_digest", True),
        ("reserved_bytes", _CIPHERTEXT_BYTES - 1),
        ("reserved_bytes", True),
        ("quota_bytes", True),
    ],
)
def test_stage_rejects_malformed_candidate_metadata(field, value):
    _create_run(observation_pack_version=1)
    identity = _start_execution()
    with pytest.raises((ValueError, runs_db.ObservationBindingFencedError)):
        _stage(identity=identity, **{field: value})


def test_available_read_prefers_existing_available_over_newer_staged_duplicate():
    _create_run(observation_pack_version=1)
    first, _identity = _stage()
    run_token = runs_db.get_worker_token("run-1")
    assert run_token
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=run_token,
        claim_token=first["claim_token"],
        result={"value": "ok"},
        pre_tool_checkpoint_digest=_identity["checkpoint_digest"],
    )
    assert finished is not None and finished["terminalSeq"]
    # Arrange an authorized row directly so this test covers read ordering;
    # production promotion remains fenced to the future finish transaction.
    available = _make_available_fixture(
        first, finished_event_seq=finished["terminalSeq"]
    )
    assert runs_db.finish_run(
        "run-1", worker_token=run_token, status="completed", finish_reason="stop"
    ) is not None

    # A second source run can stage the same content, but a read must not let
    # that newer staged row shadow the authorized available binding.
    _create_run("run-2", thread_id="thread-1", user_id="user-2", observation_pack_version=1)
    staged, _ = _stage("run-2", "exec-2")
    assert staged["state"] == "staged"
    selected = runs_db.find_available_observation_binding(
        account_id="owner", thread_id="thread-1", plaintext_digest="a" * 64
    )
    assert selected is not None
    assert selected["binding_id"] == first["binding_id"]


@pytest.mark.parametrize(
    "publisher",
    [
        runs_db.transition_observation_binding_available,
        runs_db.mark_observation_available,
    ],
    ids=["transition", "mark"],
)
def test_late_public_publisher_is_rejected_without_promoting_staged_row(publisher):
    _create_run(observation_pack_version=1)
    staged, identity = _stage()
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result={"value": "fallback"},
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None
    assert runs_db.finish_run(
        "run-1",
        worker_token=identity["worker_token"],
        status="completed",
        finish_reason="stop",
    ) is not None

    with pytest.raises(
        runs_db.ObservationBindingFencedError,
        match="durable finish transaction",
    ):
        publisher(**_promotion_kwargs(staged, finished_event_seq=finished["terminalSeq"]))

    row = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert row is not None
    assert row["state"] == "staged"
    assert runs_db.find_available_observation_binding(
        account_id="owner", thread_id="thread-1", plaintext_digest="a" * 64
    ) is None


def test_stage_requires_admitted_v1_run():
    _create_run()
    identity = _start_execution()
    with pytest.raises(runs_db.ObservationBindingFencedError, match="admitted v1"):
        _stage(identity=identity)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_stage_rejects_finished_or_ambiguous_execution(ambiguous):
    _create_run(observation_pack_version=1)
    identity = _start_execution()
    worker_token = identity["worker_token"]
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=worker_token,
        claim_token=identity["claim_token"],
        result={"value": "ok"},
        ambiguous=ambiguous,
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None
    with pytest.raises(runs_db.ObservationBindingFencedError, match="exactly started"):
        _stage(identity=identity)


_VALID_COMPLETE_IDENTITY = {
    "binding_id": "a" * 64,
    "key_id": "observation-key-v1:sha256:" + "b" * 64,
    "blob_relpath": "blobs/" + "a" * 64 + ".blob",
    "plaintext_length": _MIN_PLAINTEXT_BYTES,
    "plaintext_digest": "c" * 64,
    "ciphertext_length": _CIPHERTEXT_BYTES,
    "ciphertext_digest": "d" * 64,
    "blob_dev": 1,
    "blob_ino": 1,
    "projection": "projection",
    "finished_event_seq": 1,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_id", "observation-key-v1:sha256:" + "A" * 64),
        ("plaintext_digest", "g" * 64),
        ("ciphertext_digest", "a" * 63),
        ("blob_relpath", "blobs/other.blob"),
        ("blob_dev", 0),
        ("blob_ino", None),
        ("plaintext_length", _MIN_PLAINTEXT_BYTES - 1),
        ("ciphertext_length", _CIPHERTEXT_BYTES - 1),
        ("projection", ""),
        ("projection", "x" * 4097),
        ("finished_event_seq", 0),
    ],
)
def test_complete_identity_rejects_malformed_metadata(field, value):
    identity = dict(_VALID_COMPLETE_IDENTITY)
    identity[field] = value
    with pytest.raises((ValueError, runs_db.ObservationBindingFencedError)):
        runs_db._observation_complete_identity(**identity)


def test_late_public_publisher_does_not_bypass_removed_admission_pin():
    _create_run(observation_pack_version=1)
    staged, identity = _stage()
    worker_token = identity["worker_token"]
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=worker_token,
        claim_token=identity["claim_token"],
        result={"value": "ok"},
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None
    conn = runs_db._connect()
    try:
        conn.execute(
            "UPDATE chat_generation_runs SET observation_pack_version=0 WHERE id=?",
            ("run-1",),
        )
        conn.commit()
    finally:
        conn.close()
    # Admission-pin validation belongs to the durable finish helper.  The
    # public late-publisher seam is fenced before it can inspect either the
    # run pin or the candidate row.
    with pytest.raises(
        runs_db.ObservationBindingFencedError,
        match="durable finish transaction",
    ):
        runs_db.transition_observation_binding_available(
            **_promotion_kwargs(staged, finished_event_seq=finished["terminalSeq"])
        )
    row = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert row is not None
    assert row["state"] == "staged"


def test_ungated_finish_promotes_candidate_once_and_replays_without_inspecting_it():
    _create_run(observation_pack_version=1)
    staged, identity = _stage()
    candidate = _finish_candidate(staged, identity)
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result="fallback",
        controller_is_error=False,
        finish_candidate=candidate,
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None
    assert finished["result"] == "projection"
    assert finished["completion"]["binding"] == {
        "approval_id": None,
        "arguments_fingerprint": identity["arguments_fingerprint"],
        "authority_kind": "ungated",
        "backend_account_id": "owner",
        "card_call_id": identity["card_call_id"],
        "execution_id": "exec-1",
        "owner_subject": "alice",
        "pre_tool_checkpoint_digest": identity["checkpoint_digest"],
        "run_id": "run-1",
        "session_id": "session-1",
        "thread_id": "thread-1",
        "tool_call_id": identity["tool_call_id"],
        "tool_name": "lookup",
    }
    binding = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert binding is not None and binding["state"] == "available"
    events = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(events) == 1
    assert events[0]["seq"] == finished["terminalSeq"]

    class _ExplodingCandidate(dict):
        def __contains__(self, _key):
            raise AssertionError("terminal replay inspected the candidate")

        def get(self, _key, _default=None):
            raise AssertionError("terminal replay inspected the candidate")

        def __getitem__(self, _key):
            raise AssertionError("terminal replay inspected the candidate")

    replay = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result="attacker",
        finish_candidate=_ExplodingCandidate(),
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert replay is not None and replay["result"] == "projection"


def test_approved_finish_promotes_candidate_and_preserves_one_terminal_event():
    staged, identity = _start_approved_execution()
    candidate = _finish_candidate(
        staged,
        identity,
        fallback_result="fallback",
        approval_id="approval-1",
    )
    finished = runs_db.finish_tool_execution(
        "run-1",
        "approval-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result="fallback",
        controller_is_error=False,
        finish_candidate=candidate,
    )
    assert finished is not None and finished["result"] == "projection"
    binding = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert binding is not None
    assert binding["state"] == "available"
    events = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(events) == 1
    assert events[0]["seq"] == finished["terminalSeq"]


def test_candidate_validation_rolls_back_projection_and_falls_back_once():
    _create_run(observation_pack_version=1)
    staged, identity = _stage()
    malformed = {
        "binding_id": staged["binding_id"],
        "projection": "projection",
        "fallback_result": "fallback",
    }
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result="fallback",
        finish_candidate=malformed,
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None and finished["result"] == "fallback"
    binding = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert binding is not None
    assert binding["state"] == "unavailable"
    assert binding["reserved_bytes"] == 0
    events = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["result"] == "fallback"


def test_candidate_promotion_injection_rolls_back_provisional_terminal_write(monkeypatch):
    _create_run(observation_pack_version=1)
    staged, identity = _stage()
    candidate = _finish_candidate(staged, identity)
    def reject(*args, **kwargs):
        raise runs_db.ObservationBindingFencedError("injected candidate rejection")

    monkeypatch.setattr(runs_db, "_observation_promote_available_locked", reject)
    finished = runs_db.finish_ungated_tool_execution(
        "run-1",
        "exec-1",
        worker_token=identity["worker_token"],
        claim_token=identity["claim_token"],
        result="fallback",
        finish_candidate=candidate,
        pre_tool_checkpoint_digest=identity["checkpoint_digest"],
    )
    assert finished is not None and finished["result"] == "fallback"
    events = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(events) == 1
    assert events[0]["seq"] == finished["terminalSeq"]
    binding = runs_db.get_observation_binding(
        staged["binding_id"], account_id="owner", thread_id="thread-1"
    )
    assert binding is not None and binding["state"] == "unavailable"
