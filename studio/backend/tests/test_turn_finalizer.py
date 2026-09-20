# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.inference.turn_finalizer import _learning_excerpt, finalize_run
from core.inference import chat_generation_runs as runs_mod
from core.inference.chat_generation_runs import ChatGenerationSupervisor, _run_key
from storage import chat_generation_runs_db as runs_db
from storage import studio_db


def _seed_run(*, prompt: str = "Fix the bug", run_id: str = "run-finalize") -> None:
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
            "createdAt": 2,
        }
    )
    runs_db.create_run(
        run_id=run_id,
        owner_subject="alice",
        thread_id="thread-1",
        user_message_id="user-1",
        assistant_message_id="assistant-1",
        request_payload={
            "model": "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "session_id": "project-1",
            "use_adapter": False,
            "finalization_idempotency_key": run_id,
        },
    )
    worker = runs_db.get_worker_run(run_id)
    assert worker is not None
    _run, _owner, token = worker
    assert runs_db.mark_running(run_id, token)
    runs_db.append_events(
        run_id,
        token,
        [
            (
                "chunk",
                {"choices": [{"delta": {"content": "Answer "}}]},
                runs_db.now_ms(),
            ),
            (
                "chunk",
                {
                    "choices": [{"delta": {"content": "complete"}}],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 2,
                        "total_tokens": 42,
                    },
                },
                runs_db.now_ms(),
            ),
        ],
    )
    terminal = runs_db.finish_run(
        run_id,
        worker_token=token,
        status="completed",
        finish_reason="stop",
    )
    assert terminal is not None
    assert terminal["finalizationStatus"] == "pending"


def test_backend_finalizer_owns_keyed_effects_and_replay_is_noop(monkeypatch):
    _seed_run()
    calls: list[tuple[str, object]] = []

    from routes import helix_engine, learning, memory, self_training

    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})

    def fake_memory(subject, text, **kwargs):
        calls.append(("memory", (subject, text, kwargs)))
        return {"stored": True, "node": {"id": "node-1"}}

    async def fake_example(payload, background_tasks, current_subject):
        calls.append(("self_training", (payload, current_subject)))
        return {
            "recorded": True,
            "idempotent": False,
            "exampleId": "example-1",
        }

    def fake_prepare(payload):
        calls.append(("prepare", payload))
        return {"available": True, "perform_deep_audit": False}

    def fake_ingest(payload, background_tasks, current_subject):
        calls.append(("ingest", (payload, current_subject)))
        return {
            "trajectory_id": payload.turn_id,
            "actions": ["memory_considered"],
            "qlora_outcome": {"outcome": "not_applicable", "reason": "none"},
        }

    monkeypatch.setattr(memory, "persist_memory_experience", fake_memory)
    monkeypatch.setattr(self_training, "record_self_training_example", fake_example)
    monkeypatch.setattr(helix_engine, "prepare_completed_turn_audit", fake_prepare)
    monkeypatch.setattr(helix_engine, "ingest_completed_turn", fake_ingest)

    settled = asyncio.run(finalize_run("run-finalize"))
    assert settled is not None
    assert settled["finalizationStatus"] == "completed"
    assert [name for name, _value in calls] == [
        "memory",
        "self_training",
        "prepare",
        "ingest",
    ]
    memory_call = calls[0][1]
    assert memory_call[2]["idempotency_key"] == "run-finalize"
    example_payload = calls[1][1][0]
    assert example_payload.idempotencyKey == "run-finalize"
    assert example_payload.prompt == "Fix the bug"
    assert example_payload.completion == "Answer complete"
    ingest_payload = calls[3][1][0]
    assert ingest_payload.idempotency_key == "run-finalize"
    assert ingest_payload.turn_id == "run-finalize"
    assert ingest_payload.telemetry["prompt_tokens"] == 40
    assert ingest_payload.telemetry["completion_tokens"] == 2

    before = len(calls)
    replay = asyncio.run(finalize_run("run-finalize"))
    assert replay is not None
    assert replay["finalizationStatus"] == "completed"
    assert len(calls) == before
    phases = [event["type"] for event in runs_db.list_events("run-finalize")]
    assert phases[-7:] == [
        "turn_finalization.started",
        "turn_finalization.memory",
        "turn_finalization.self_training",
        "turn_finalization.audit",
        "turn_finalization.helix_ingest",
        "turn_finalization.qlora_dispatch",
        "turn_finalization.completed",
    ]


def test_learning_protocol_turn_is_skipped_without_side_effects(monkeypatch):
    _seed_run(prompt="[HELIX_SELF_AUDIT] inspect", run_id="run-protocol")
    from routes import learning, memory

    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})
    monkeypatch.setattr(
        memory,
        "persist_memory_experience",
        lambda *_args, **_kwargs: pytest.fail("protocol turn wrote memory"),
    )
    settled = asyncio.run(finalize_run("run-protocol"))
    assert settled is not None
    assert settled["finalizationStatus"] == "skipped"


def test_learning_excerpt_preserves_final_correction_in_long_turns():
    text = "HEAD-" + ("x" * 5_000) + "-FINAL-CORRECTION"
    excerpt = _learning_excerpt(text, 400)
    assert len(excerpt) <= 400
    assert excerpt.startswith("HEAD-")
    assert excerpt.endswith("-FINAL-CORRECTION")
    assert "middle omitted by durable finalizer" in excerpt


def test_backend_finalizer_wires_optional_model_audit_into_helix_ingest(monkeypatch):
    _seed_run(run_id="run-audit")
    from core.inference import turn_finalizer_audit
    from routes import helix_engine, learning, memory, self_training

    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": False})
    monkeypatch.setattr(
        memory,
        "persist_memory_experience",
        lambda *_args, **_kwargs: {"stored": False, "reason": "disabled"},
    )

    async def fake_example(*_args, **_kwargs):
        return {"recorded": True, "exampleId": "example-audit"}

    monkeypatch.setattr(self_training, "record_self_training_example", fake_example)
    monkeypatch.setattr(
        helix_engine,
        "prepare_completed_turn_audit",
        lambda _payload: {
            "available": True,
            "perform_deep_audit": True,
            "artifacts": {"objective": "Fix the bug", "tests": [{"passed": True}]},
        },
    )

    async def fake_model_audit(**_kwargs):
        return (
            {
                "objective": "Fix the bug",
                "achieved": True,
                "recommendation": "IGNORE",
                "source": "model",
            },
            [{"claim_id": "c1", "claim": "test passed", "evidence_refs": []}],
            {"performed": True, "duration_ms": 12},
        )

    monkeypatch.setattr(turn_finalizer_audit, "run_optional_model_audit", fake_model_audit)
    captured = {}

    def fake_ingest(payload, background_tasks, current_subject):
        captured["payload"] = payload
        return {
            "trajectory_id": payload.turn_id,
            "actions": [],
            "qlora_outcome": {"outcome": "not_applicable", "reason": "none"},
        }

    monkeypatch.setattr(helix_engine, "ingest_completed_turn", fake_ingest)
    settled = asyncio.run(
        finalize_run("run-audit", app=SimpleNamespace(state=SimpleNamespace()))
    )
    assert settled is not None and settled["finalizationStatus"] == "completed"
    payload = captured["payload"]
    assert payload.self_audit["source"] == "model"
    assert payload.claims[0]["claim"] == "test passed"
    assert payload.telemetry["self_audit_performed"] is True


def test_retry_replays_committed_phase_receipt_without_reinvoking_effect(monkeypatch):
    _seed_run(run_id="run-retry")
    from routes import helix_engine, learning, memory, self_training

    calls = {"memory": 0, "training": 0, "ingest": 0}
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_BASE_MS", 1)
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_MAX_MS", 1)
    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})

    def fake_memory(*_args, **_kwargs):
        calls["memory"] += 1
        return {"stored": True, "node": {"id": "node-retry"}}

    async def flaky_training(*_args, **_kwargs):
        calls["training"] += 1
        if calls["training"] == 1:
            raise RuntimeError("transient training store failure")
        return {"recorded": True, "exampleId": "example-retry"}

    monkeypatch.setattr(memory, "persist_memory_experience", fake_memory)
    monkeypatch.setattr(self_training, "record_self_training_example", flaky_training)
    monkeypatch.setattr(
        helix_engine,
        "prepare_completed_turn_audit",
        lambda _payload: {"perform_deep_audit": False},
    )
    monkeypatch.setattr(
        helix_engine,
        "ingest_completed_turn",
        lambda payload, *_args: {
            "trajectory_id": payload.turn_id,
            "actions": [],
            "qlora_outcome": {"outcome": "not_applicable"},
        },
    )

    def fake_ingest(payload, *_args):
        calls["ingest"] += 1
        return {"trajectory_id": payload.turn_id, "actions": [], "qlora_outcome": {}}

    monkeypatch.setattr(helix_engine, "ingest_completed_turn", fake_ingest)

    first = asyncio.run(finalize_run("run-retry"))
    assert first is not None and first["finalizationStatus"] == "pending"
    assert calls == {"memory": 1, "training": 1, "ingest": 0}

    asyncio.run(asyncio.sleep(0.005))
    second = asyncio.run(finalize_run("run-retry"))
    assert second is not None and second["finalizationStatus"] == "completed"
    assert calls == {"memory": 1, "training": 2, "ingest": 1}


def test_lost_claim_stops_before_the_next_phase(monkeypatch):
    _seed_run(run_id="run-lost-claim")
    from routes import learning, memory, self_training

    calls: list[str] = []
    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})
    monkeypatch.setattr(
        memory,
        "persist_memory_experience",
        lambda *_args, **_kwargs: calls.append("memory") or {"stored": True},
    )

    async def forbidden_training(*_args, **_kwargs):
        calls.append("training")
        return {"recorded": True}

    monkeypatch.setattr(self_training, "record_self_training_example", forbidden_training)
    monkeypatch.setattr(runs_db, "append_finalization_event", lambda *_args, **_kwargs: None)

    settled = asyncio.run(finalize_run("run-lost-claim"))
    assert settled is not None and settled["finalizationStatus"] == "running"
    assert calls == ["memory"]


def test_queued_qlora_dispatch_replays_after_post_ingest_fault(monkeypatch):
    _seed_run(run_id="run-qlora-dispatch")
    from routes import helix_engine, learning, memory, self_training

    dispatches = {"count": 0}
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_BASE_MS", 1)
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_MAX_MS", 1)
    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": False})
    monkeypatch.setattr(
        memory,
        "persist_memory_experience",
        lambda *_args, **_kwargs: {"stored": False, "reason": "disabled"},
    )

    async def fake_example(*_args, **_kwargs):
        return {"recorded": True, "exampleId": "example-dispatch"}

    async def flaky_dispatch(_owner):
        dispatches["count"] += 1
        if dispatches["count"] == 1:
            raise RuntimeError("fault after durable ingest receipt")

    def fake_ingest(payload, background_tasks, _owner):
        background_tasks.add_task(flaky_dispatch, "alice")
        return {
            "trajectory_id": payload.turn_id,
            "actions": ["queue_qlora_training"],
            "qlora_outcome": {"outcome": "queued", "reason": "eligible"},
        }

    monkeypatch.setattr(self_training, "record_self_training_example", fake_example)
    monkeypatch.setattr(self_training, "_start_training_for_state", flaky_dispatch)
    monkeypatch.setattr(
        helix_engine,
        "prepare_completed_turn_audit",
        lambda _payload: {"perform_deep_audit": False},
    )
    monkeypatch.setattr(helix_engine, "ingest_completed_turn", fake_ingest)

    first = asyncio.run(finalize_run("run-qlora-dispatch"))
    assert first is not None and first["finalizationStatus"] == "pending"
    events = runs_db.list_events("run-qlora-dispatch")
    assert any(event["type"] == "turn_finalization.helix_ingest" for event in events)
    assert not any(event["type"] == "turn_finalization.qlora_dispatch" for event in events)

    asyncio.run(asyncio.sleep(0.005))
    second = asyncio.run(finalize_run("run-qlora-dispatch"))
    assert second is not None and second["finalizationStatus"] == "completed"
    assert dispatches["count"] == 2
    assert sum(
        event["type"] == "turn_finalization.helix_ingest"
        for event in runs_db.list_events("run-qlora-dispatch")
    ) == 1


async def _wait_for_status(run_id: str, status: str, *, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        run = runs_db.get_run(run_id)
        if run is not None and run["finalizationStatus"] == status:
            return run
        await asyncio.sleep(0.005)
    raise AssertionError(f"{run_id} did not reach finalization status {status}")


def _install_finalizer_effects(monkeypatch, memory_effect):
    from routes import helix_engine, learning, memory, self_training

    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})
    monkeypatch.setattr(memory, "persist_memory_experience", memory_effect)

    async def fake_example(*_args, **_kwargs):
        return {"recorded": True, "exampleId": "example-supervised-retry"}

    monkeypatch.setattr(self_training, "record_self_training_example", fake_example)
    monkeypatch.setattr(
        helix_engine,
        "prepare_completed_turn_audit",
        lambda _payload: {"perform_deep_audit": False},
    )


def _install_short_finalization_lease(monkeypatch, *, lease_ms: int = 100):
    real_claim = runs_db.claim_finalization
    real_append = runs_db.append_finalization_event
    real_renew = runs_db.renew_finalization_claim
    monkeypatch.setattr(
        runs_db,
        "claim_finalization",
        lambda run_id: real_claim(run_id, lease_ms=lease_ms),
    )
    monkeypatch.setattr(
        runs_db,
        "append_finalization_event",
        lambda run_id, token, event_type, payload: real_append(
            run_id,
            token,
            event_type,
            payload,
            lease_ms=lease_ms,
        ),
    )
    monkeypatch.setattr(
        runs_db,
        "renew_finalization_claim",
        lambda run_id, token: real_renew(run_id, token, lease_ms=lease_ms),
    )
    monkeypatch.setattr(runs_mod, "_FINALIZATION_WATCH_RETRY_BASE_SECONDS", 0.001)
    monkeypatch.setattr(runs_mod, "_FINALIZATION_WATCH_RETRY_MAX_SECONDS", 0.005)


def test_transient_phase_failure_self_retries_without_restart(monkeypatch):
    _seed_run(run_id="run-supervised-retry")
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_BASE_MS", 10)
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_MAX_MS", 20)
    calls = {"memory": 0}

    def flaky_memory(*_args, **_kwargs):
        calls["memory"] += 1
        if calls["memory"] == 1:
            raise RuntimeError("transient memory failure")
        return {"stored": True, "node": {"id": "node-supervised-retry"}}

    _install_finalizer_effects(monkeypatch, flaky_memory)

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        supervisor._schedule_finalization("run-supervised-retry")
        try:
            return await _wait_for_status("run-supervised-retry", "completed")
        finally:
            await supervisor.stop()

    settled = asyncio.run(scenario())
    assert calls["memory"] == 2
    assert settled["finalizationAttempts"] == 0
    assert settled["finalizationNextAttemptAt"] is None


def test_persistent_phase_failure_uses_bounded_backoff_without_hot_loop(monkeypatch):
    _seed_run(run_id="run-bounded-retry")
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_BASE_MS", 40)
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_MAX_MS", 80)
    call_times: list[float] = []

    # The effect runs in a worker thread, so use monotonic there rather than the
    # event-loop accessor.
    def timed_failure(*_args, **_kwargs):
        call_times.append(time.monotonic())
        raise RuntimeError("persistent memory failure")

    _install_finalizer_effects(monkeypatch, timed_failure)

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        supervisor._schedule_finalization("run-bounded-retry")
        try:
            deadline = asyncio.get_running_loop().time() + 2.0
            while asyncio.get_running_loop().time() < deadline:
                run = runs_db.get_run("run-bounded-retry")
                if run is not None and run["finalizationAttempts"] >= 3:
                    return run
                await asyncio.sleep(0.005)
            raise AssertionError("persistent finalizer did not make three bounded attempts")
        finally:
            await supervisor.stop()

    pending = asyncio.run(scenario())
    assert pending["finalizationStatus"] == "pending"
    assert pending["finalizationNextAttemptAt"] is not None
    assert len(call_times) == 3
    assert call_times[1] - call_times[0] >= 0.03
    assert call_times[2] - call_times[1] >= 0.06


def test_restart_schedules_not_yet_due_pending_finalization(monkeypatch):
    _seed_run(run_id="run-restart-backoff")
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_BASE_MS", 150)
    monkeypatch.setattr(runs_db, "FINALIZATION_RETRY_MAX_MS", 150)
    calls = {"memory": 0}

    def memory_effect(*_args, **_kwargs):
        calls["memory"] += 1
        if calls["memory"] == 1:
            raise RuntimeError("fail before restart")
        return {"stored": True, "node": {"id": "node-restart-backoff"}}

    _install_finalizer_effects(monkeypatch, memory_effect)
    first = asyncio.run(finalize_run("run-restart-backoff"))
    assert first is not None and first["finalizationStatus"] == "pending"
    assert first["finalizationNextAttemptAt"] > runs_db.now_ms()

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        resumed = supervisor.resume_pending_finalizations()
        assert resumed == []
        try:
            await asyncio.sleep(0.03)
            assert calls["memory"] == 1
            return await _wait_for_status("run-restart-backoff", "completed")
        finally:
            await supervisor.stop()

    settled = asyncio.run(scenario())
    assert calls["memory"] == 2
    assert settled["finalizationAttempts"] == 0


def test_running_finalization_recovery_tracks_a_renewed_live_lease(monkeypatch):
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    now = runs_db.now_ms()
    recovery_calls = {"count": 0}

    def recover(_run_id):
        recovery_calls["count"] += 1
        return ["run-renewed-lease"] if recovery_calls["count"] == 2 else []

    monkeypatch.setattr(runs_db, "requeue_interrupted_finalizations", recover)
    monkeypatch.setattr(
        runs_db,
        "get_run",
        lambda _run_id: {
            "finalizationStatus": "running",
            "finalizationLeaseExpiresAt": now + 5,
        },
    )

    async def completed(_run_id):
        return {"finalizationStatus": "completed"}

    monkeypatch.setattr(supervisor, "_finalize", completed)
    result = asyncio.run(
        supervisor._recover_finalization_after_lease("run-renewed-lease", now)
    )
    assert recovery_calls["count"] == 2
    assert result == {"finalizationStatus": "completed"}


def test_supervisor_recovers_a_lost_finalization_claim(monkeypatch):
    _seed_run(run_id="run-lost-claim-watch")
    _install_short_finalization_lease(monkeypatch)
    calls = {"memory": 0, "append": 0}

    def memory_effect(*_args, **_kwargs):
        calls["memory"] += 1
        return {"stored": True, "node": {"id": "node-lost-claim-watch"}}

    _install_finalizer_effects(monkeypatch, memory_effect)
    real_append = runs_db.append_finalization_event

    def lose_first_receipt(*args, **kwargs):
        calls["append"] += 1
        if calls["append"] == 1:
            return None
        return real_append(*args, **kwargs)

    monkeypatch.setattr(runs_db, "append_finalization_event", lose_first_receipt)

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        supervisor._schedule_finalization("run-lost-claim-watch")
        try:
            return await _wait_for_status("run-lost-claim-watch", "completed")
        finally:
            await supervisor.stop()

    settled = asyncio.run(scenario())
    assert settled["finalizationStatus"] == "completed"
    assert calls["memory"] == 2


def test_supervisor_recovers_when_retry_transition_raises(monkeypatch):
    _seed_run(run_id="run-retry-transition-error")
    _install_short_finalization_lease(monkeypatch)
    calls = {"memory": 0, "retry": 0}

    def flaky_memory(*_args, **_kwargs):
        calls["memory"] += 1
        if calls["memory"] == 1:
            raise RuntimeError("phase failure before retry transition")
        return {"stored": True, "node": {"id": "node-retry-transition-error"}}

    _install_finalizer_effects(monkeypatch, flaky_memory)
    real_retry = runs_db.retry_finalization

    def flaky_retry(*args, **kwargs):
        calls["retry"] += 1
        if calls["retry"] == 1:
            raise RuntimeError("database failed before pending transition")
        return real_retry(*args, **kwargs)

    monkeypatch.setattr(runs_db, "retry_finalization", flaky_retry)

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        supervisor._schedule_finalization("run-retry-transition-error")
        try:
            return await _wait_for_status("run-retry-transition-error", "completed")
        finally:
            await supervisor.stop()

    settled = asyncio.run(scenario())
    assert settled["finalizationStatus"] == "completed"
    assert calls == {"memory": 2, "retry": 1}


def test_finalization_watcher_stops_cleanly():
    _seed_run(run_id="run-watcher-stop")
    claim = runs_db.claim_finalization("run-watcher-stop", lease_ms=60_000)
    assert claim is not None

    async def scenario():
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        assert supervisor.resume_pending_finalizations() == []
        key = _run_key("run-watcher-stop")
        task = supervisor._finalization_tasks[key]
        await supervisor.stop()
        await asyncio.sleep(0)
        return supervisor, key, task

    supervisor, key, task = asyncio.run(scenario())
    assert task.cancelled() or task.done()
    assert key not in supervisor._finalization_tasks
