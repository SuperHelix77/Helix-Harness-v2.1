# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from core.memory import experience_idempotency, mem0_store
from routes import memory as memory_routes
from utils.account_context import AccountContext, bind_account, reset_account


def _payload(
    *,
    text: str = "Completed durable task",
    thread_id: str = "thread-1",
    key: str = "turn-1-experience",
) -> memory_routes.MemoryExperienceRequest:
    return memory_routes.MemoryExperienceRequest(
        text=text,
        threadId=thread_id,
        kind="experience",
        title="Durable task",
        idempotencyKey=key,
    )


def _enable_memory(monkeypatch) -> None:
    import routes.learning as learning

    monkeypatch.setattr(learning, "_read_state", lambda: {"mem0Enabled": True})


def test_same_key_replay_calls_mem0_once_and_returns_exact_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []

    def add(subject, text, **kwargs):
        calls.append((subject, text, kwargs))
        return {"stored": True, "result": {"id": "vector-1"}, "node": {"id": "node-1"}}

    monkeypatch.setattr(mem0_store, "add_experience", add)
    payload = _payload()

    first = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")
    replay = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    assert replay == first
    assert len(calls) == 1


def test_concurrent_same_key_retry_has_one_side_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = 0
    lock = threading.Lock()

    def add(_subject, _text, **_kwargs):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.08)
        return {"stored": True, "result": {"id": "vector-1"}}

    monkeypatch.setattr(mem0_store, "add_experience", add)
    payload = _payload()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: memory_routes.add_memory_experience(
                    payload, current_subject="alice@example.test"
                ),
                range(6),
            )
        )

    assert calls == 1
    assert all(result == results[0] for result in results)


def test_completed_receipt_survives_process_instance_restart(tmp_path, monkeypatch):
    db = tmp_path / "idempotency.sqlite3"
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(db))
    _enable_memory(monkeypatch)
    calls = 0

    def add(_subject, _text, **_kwargs):
        nonlocal calls
        calls += 1
        return {"stored": True, "result": {"id": "vector-persisted"}}

    monkeypatch.setattr(mem0_store, "add_experience", add)
    payload = _payload()
    first = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    monkeypatch.setattr(experience_idempotency, "_PROCESS_INSTANCE_ID", "restarted-process")
    replay = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    assert db.exists()
    assert replay == first
    assert calls == 1


def test_interrupted_claim_before_effect_is_safely_reclaimed_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []
    monkeypatch.setattr(
        mem0_store,
        "add_experience",
        lambda *_args, **_kwargs: calls.append(1) or {"stored": True},
    )
    payload = _payload()
    claim, replay = experience_idempotency._claim(
        idempotency_key=payload.idempotencyKey or "",
        thread_id=payload.threadId,
        payload={
            "text": payload.text,
            "threadId": payload.threadId,
            "kind": payload.kind,
            "title": payload.title,
        },
    )
    assert claim is not None
    assert replay is None

    monkeypatch.setattr(experience_idempotency, "_PROCESS_INSTANCE_ID", "restarted-process")
    replay = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    assert replay["stored"] is True
    assert calls == [1]


def test_interrupted_claim_after_durable_graph_write_recovers_without_vector_replay(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    graph_root = tmp_path / "mem0"
    graph_root.mkdir()
    monkeypatch.setattr(mem0_store, "_root", lambda: graph_root)
    _enable_memory(monkeypatch)
    payload = _payload(key="graph-witness")
    claim, replay = experience_idempotency._claim(
        idempotency_key=payload.idempotencyKey or "",
        thread_id=payload.threadId,
        payload={
            "text": payload.text,
            "threadId": payload.threadId,
            "kind": payload.kind,
            "title": payload.title,
        },
    )
    assert claim is not None and replay is None
    mem0_store._append_graph_node(
        title=payload.title or payload.text,
        text=payload.text,
        kind=payload.kind,
        thread_id=payload.threadId or "",
        entities=None,
        links=None,
        idempotency_key=payload.idempotencyKey,
    )
    monkeypatch.setattr(experience_idempotency, "_PROCESS_INSTANCE_ID", "restarted-process")
    monkeypatch.setattr(
        mem0_store,
        "add_experience",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("vector write must not replay after durable graph witness")
        ),
    )

    recovered = memory_routes.add_memory_experience(
        payload,
        current_subject="alice@example.test",
    )

    assert recovered["stored"] is True
    assert recovered["idempotent"] is True
    assert recovered["reason"] == "recovered_from_idempotent_local_graph"


def test_different_key_identical_text_remains_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []

    def add(_subject, text, **kwargs):
        calls.append((text, kwargs))
        return {"stored": True, "result": {"id": f"vector-{len(calls)}"}}

    monkeypatch.setattr(mem0_store, "add_experience", add)
    first = memory_routes.add_memory_experience(
        _payload(key="turn-1"), current_subject="alice@example.test"
    )
    second = memory_routes.add_memory_experience(
        _payload(key="turn-2"), current_subject="alice@example.test"
    )

    assert len(calls) == 2
    assert first != second


def test_same_key_is_separate_across_threads_and_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []

    def add(_subject, _text, **kwargs):
        calls.append(kwargs["thread_id"])
        return {"stored": True, "result": {"id": f"vector-{len(calls)}"}}

    monkeypatch.setattr(mem0_store, "add_experience", add)
    key = "stable-key"
    memory_routes.add_memory_experience(
        _payload(thread_id="thread-a", key=key), current_subject="alice@example.test"
    )
    memory_routes.add_memory_experience(
        _payload(thread_id="thread-b", key=key), current_subject="alice@example.test"
    )

    token = bind_account(AccountContext("account-2", "second-user"))
    try:
        memory_routes.add_memory_experience(
            _payload(thread_id="thread-a", key=key), current_subject="second@example.test"
        )
    finally:
        reset_account(token)

    assert calls == ["thread-a", "thread-b", "thread-a"]


def test_same_key_conflicting_payload_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []
    monkeypatch.setattr(
        mem0_store,
        "add_experience",
        lambda *_args, **_kwargs: calls.append(1) or {"stored": True},
    )

    memory_routes.add_memory_experience(_payload(), current_subject="alice@example.test")
    with pytest.raises(HTTPException) as raised:
        memory_routes.add_memory_experience(
            _payload(text="Different completed task"),
            current_subject="alice@example.test",
        )

    assert raised.value.status_code == 409
    assert "different payload" in str(raised.value.detail)
    assert len(calls) == 1


def test_mem0_unavailable_graph_fallback_is_replayed_without_second_vector_attempt(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    graph_root = tmp_path / "mem0"
    graph_root.mkdir()
    _enable_memory(monkeypatch)
    monkeypatch.setattr(mem0_store, "_root", lambda: graph_root)
    attempts = 0

    def unavailable():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("mem0 unavailable")

    monkeypatch.setattr(mem0_store, "_instance", unavailable)
    payload = _payload()

    first = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")
    replay = memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    assert first["stored"] is True
    assert "mem0 unavailable" in first["reason"]
    assert replay == first
    assert attempts == 1
    assert len(mem0_store.graph_snapshot()["nodes"]) == 1


def test_unkeyed_legacy_behavior_still_writes_each_time(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_MEMORY_EXPERIENCE_DB", str(tmp_path / "idempotency.sqlite3"))
    _enable_memory(monkeypatch)
    calls = []
    monkeypatch.setattr(
        mem0_store,
        "add_experience",
        lambda *_args, **_kwargs: calls.append(1) or {"stored": True},
    )
    payload = memory_routes.MemoryExperienceRequest(text="Legacy experience", threadId="thread-1")

    memory_routes.add_memory_experience(payload, current_subject="alice@example.test")
    memory_routes.add_memory_experience(payload, current_subject="alice@example.test")

    assert len(calls) == 2
