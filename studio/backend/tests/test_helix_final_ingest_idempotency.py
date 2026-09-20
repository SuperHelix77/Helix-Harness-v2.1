# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import threading
import time

import pytest
from fastapi import BackgroundTasks, HTTPException


def _payload(**overrides):
    from routes.helix_engine import IngestTurnIn

    values = {
        "idempotency_key": "final-ingest-run-1",
        "session_id": "session-a",
        "thread_id": "thread-a",
        "turn_id": "turn-a",
        "prompt": "finish the task",
        "final_result": "done",
        "model_id": "qwen-test",
        "telemetry": {"generation_run_id": "run-a", "completion_tokens": 12},
    }
    values.update(overrides)
    return IngestTurnIn(**values)


def _patch_effects(monkeypatch, *, entered=None, release=None):
    from core.helix_engine import provenance
    from routes import helix_engine, self_training

    calls = {
        "final_learning": 0,
        "skill_retention": 0,
        "queue_decision": 0,
        "provenance": 0,
    }
    lock = threading.Lock()

    def ingest(**kwargs):
        with lock:
            calls["final_learning"] += 1
            calls["skill_retention"] += 1
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(3.0)
        return {
            "adaptive_cycle": {
                "adaptation": {"action": "QLORA_CANDIDATE", "qlora_eligible": True}
            },
            "actions": ["stage_qlora_candidate", "retain_temp_skill:focused-skill"],
            "skill_retention": [
                {
                    "skill_name": "focused-skill",
                    "disposition": "retained",
                    "reason": "verified final outcome",
                }
            ],
            "qlora_stage": {"stored": True, "reason": "verified_hermes_candidate_staged"},
            "trajectory_id": kwargs.get("turn_id") or "turn-a",
            "model_id": kwargs.get("effective_model_id") or kwargs.get("model_id") or "qwen-test",
            "base_model_id": kwargs.get("model_id") or "qwen-test",
        }

    def queue(**_kwargs):
        with lock:
            calls["queue_decision"] += 1
        return {"outcome": "denied", "reason": "automatic_qlora_training_disabled"}

    def append(**_kwargs):
        with lock:
            calls["provenance"] += 1
        return True

    monkeypatch.setattr(helix_engine, "ingest_turn", ingest)
    monkeypatch.setattr(self_training, "queue_hermes_training_with_provenance", queue)
    monkeypatch.setattr(provenance, "append_turn_receipt", append)
    return calls


def test_keyed_final_ingest_replays_exact_receipt_without_repeating_side_effects(
    tmp_path, monkeypatch
):
    from routes.helix_engine import ingest_completed_turn

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    calls = _patch_effects(monkeypatch)
    payload = _payload()

    first = ingest_completed_turn(payload, BackgroundTasks(), current_subject="subject-a")
    replay = ingest_completed_turn(payload, BackgroundTasks(), current_subject="subject-a")

    assert replay == first
    assert calls == {
        "final_learning": 1,
        "skill_retention": 1,
        "queue_decision": 1,
        "provenance": 1,
    }
    assert replay["qlora_outcome"] == {
        "outcome": "denied",
        "reason": "automatic_qlora_training_disabled",
    }


def test_keyed_final_ingest_concurrent_duplicates_have_one_side_effect_owner(
    tmp_path, monkeypatch
):
    from routes.helix_engine import ingest_completed_turn

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    entered = threading.Event()
    release = threading.Event()
    calls = _patch_effects(monkeypatch, entered=entered, release=release)
    payload = _payload(idempotency_key="concurrent-final-ingest")
    results = []
    errors = []

    def invoke():
        try:
            results.append(
                ingest_completed_turn(payload, BackgroundTasks(), current_subject="subject-a")
            )
        except BaseException as exc:  # pragma: no cover - assertions report thread failure
            errors.append(exc)

    first_thread = threading.Thread(target=invoke)
    second_thread = threading.Thread(target=invoke)
    first_thread.start()
    assert entered.wait(2.0)
    second_thread.start()
    time.sleep(0.05)
    release.set()
    first_thread.join(3.0)
    second_thread.join(3.0)

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert calls == {
        "final_learning": 1,
        "skill_retention": 1,
        "queue_decision": 1,
        "provenance": 1,
    }


def test_keyed_final_ingest_completed_receipt_survives_process_reload(
    tmp_path, monkeypatch
):
    from core.helix_engine import final_ingest_idempotency
    from core.helix_engine import provenance
    from routes import helix_engine, self_training

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    calls = _patch_effects(monkeypatch)
    payload = _payload(idempotency_key="lost-response-final-ingest")
    first = helix_engine.ingest_completed_turn(
        payload, BackgroundTasks(), current_subject="subject-a"
    )
    assert calls["final_learning"] == 1

    monkeypatch.setattr(final_ingest_idempotency, "_PROCESS_INSTANCE_ID", "reloaded-process")
    monkeypatch.setattr(
        helix_engine,
        "ingest_turn",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("final learning repeated")),
    )
    monkeypatch.setattr(
        self_training,
        "queue_hermes_training_with_provenance",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("queue decision repeated")),
    )
    monkeypatch.setattr(
        provenance,
        "append_turn_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provenance repeated")),
    )

    replay = helix_engine.ingest_completed_turn(
        payload, BackgroundTasks(), current_subject="subject-a"
    )
    assert replay == first


def test_interrupted_claim_before_effect_is_reclaimed_and_completed(
    tmp_path, monkeypatch
):
    from core.helix_engine import final_ingest_idempotency
    from routes import helix_engine

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    payload = _payload(idempotency_key="interrupted-before-effect")
    request_payload = payload.model_dump(mode="json", exclude={"idempotency_key"})
    claim, replay = final_ingest_idempotency.claim_final_ingest(
        idempotency_key=payload.idempotency_key or "",
        session_id=payload.session_id,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        subject="subject-a",
        payload=request_payload,
    )
    assert claim is not None and replay is None

    monkeypatch.setattr(
        final_ingest_idempotency,
        "_PROCESS_INSTANCE_ID",
        "restarted-process",
    )
    calls = _patch_effects(monkeypatch)
    result = helix_engine.ingest_completed_turn(
        payload,
        BackgroundTasks(),
        current_subject="subject-a",
    )

    assert result["trajectory_id"] == "turn-a"
    assert calls == {
        "final_learning": 1,
        "skill_retention": 1,
        "queue_decision": 1,
        "provenance": 1,
    }

    replayed = helix_engine.ingest_completed_turn(
        payload,
        BackgroundTasks(),
        current_subject="subject-a",
    )
    assert replayed == result
    assert calls["final_learning"] == 1


def test_interrupted_claim_restores_exact_tool_capture_after_process_restart(
    tmp_path, monkeypatch
):
    from core.helix_engine import capture, final_ingest_idempotency

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    payload = _payload(idempotency_key="capture-restart")
    key = capture.capture_session_key(
        payload.session_id,
        payload.thread_id,
        payload.turn_id,
    )
    capture.clear_session(key)
    capture.record_tool_execution(
        key,
        "terminal",
        {"command": "printf ok"},
        "ok",
        verification_kind="test",
        verification_claim="command succeeded",
        verification_subject="artifact-1",
    )
    snapshot = capture.capture_snapshot(key)
    request_payload = payload.model_dump(mode="json", exclude={"idempotency_key"})
    claim, replay = final_ingest_idempotency.claim_final_ingest(
        idempotency_key=payload.idempotency_key or "",
        session_id=payload.session_id,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        subject="subject-a",
        payload=request_payload,
        replay_safe=True,
        capture_snapshot=snapshot,
    )
    assert claim is not None and replay is None

    # Simulate a complete backend restart: process-local capture is gone, while
    # the durable final-ingest claim survives under a new process identity.
    capture.clear_session(key)
    monkeypatch.setattr(
        final_ingest_idempotency,
        "_PROCESS_INSTANCE_ID",
        "restarted-process",
    )
    reclaimed, replay = final_ingest_idempotency.claim_final_ingest(
        idempotency_key=payload.idempotency_key or "",
        session_id=payload.session_id,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        subject="subject-a",
        payload=request_payload,
        replay_safe=True,
        capture_snapshot=capture.capture_snapshot(key),
    )
    assert reclaimed is not None and replay is None
    assert reclaimed["capture_snapshot"] == snapshot
    assert capture.restore_capture_snapshot(key, reclaimed["capture_snapshot"]) is True

    restored = capture.session_steps(key)
    assert len(restored) == 1
    assert restored[0].name == "terminal"
    assert restored[0].arguments == '{"command": "printf ok"}'
    assert restored[0].result == "ok"
    assert restored[0].verification is not None
    assert restored[0].verification.subject == "artifact-1"
    final_ingest_idempotency.complete_final_ingest(reclaimed, {"completed": True})
    capture.clear_session(key)


def test_keyed_final_ingest_reused_key_with_different_payload_is_conflict(
    tmp_path, monkeypatch
):
    from routes.helix_engine import ingest_completed_turn

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "idempotency.sqlite3"))
    calls = _patch_effects(monkeypatch)
    first_payload = _payload(idempotency_key="payload-conflict")
    changed_payload = _payload(idempotency_key="payload-conflict", prompt="different task")

    ingest_completed_turn(first_payload, BackgroundTasks(), current_subject="subject-a")
    with pytest.raises(HTTPException) as exc_info:
        ingest_completed_turn(changed_payload, BackgroundTasks(), current_subject="subject-a")

    assert exc_info.value.status_code == 409
    assert "different payload" in str(exc_info.value.detail)
    assert calls == {
        "final_learning": 1,
        "skill_retention": 1,
        "queue_decision": 1,
        "provenance": 1,
    }


def test_keyed_final_ingest_same_key_is_independent_across_account_and_turn_scope(
    tmp_path, monkeypatch
):
    from routes.helix_engine import ingest_completed_turn
    from utils.account_context import AccountContext, run_as

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "shared-idempotency.sqlite3"))
    calls = _patch_effects(monkeypatch)
    account_a = AccountContext("account-a", "same-user")
    account_b = AccountContext("account-b", "same-user")

    def invoke(payload):
        return ingest_completed_turn(payload, BackgroundTasks(), current_subject="same-subject")

    run_as(account_a, invoke, _payload(idempotency_key="same-key"))
    run_as(account_b, invoke, _payload(idempotency_key="same-key"))
    run_as(
        account_a,
        invoke,
        _payload(idempotency_key="same-key", session_id="session-b"),
    )
    run_as(
        account_a,
        invoke,
        _payload(idempotency_key="same-key", thread_id="thread-b"),
    )
    run_as(
        account_a,
        invoke,
        _payload(idempotency_key="same-key", turn_id="turn-b"),
    )

    assert calls == {
        "final_learning": 5,
        "skill_retention": 5,
        "queue_decision": 5,
        "provenance": 5,
    }


def test_unkeyed_final_ingest_keeps_legacy_repeat_behavior(tmp_path, monkeypatch):
    from routes.helix_engine import ingest_completed_turn

    monkeypatch.setenv("HELIX_FINAL_INGEST_DB", str(tmp_path / "unused.sqlite3"))
    calls = _patch_effects(monkeypatch)
    payload = _payload(idempotency_key=None)

    ingest_completed_turn(payload, BackgroundTasks(), current_subject="subject-a")
    ingest_completed_turn(payload, BackgroundTasks(), current_subject="subject-a")

    assert calls == {
        "final_learning": 2,
        "skill_retention": 2,
        "queue_decision": 2,
        "provenance": 2,
    }
