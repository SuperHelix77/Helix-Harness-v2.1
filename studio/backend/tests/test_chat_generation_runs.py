# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from core.inference.chat_generation_runs import _requires_durable_barrier
from core.inference.durable_tool_journal import durable_tool_run
from core.inference.tool_stream_exec import stream_tool_execution
from routes.chat_generation_runs import (
    CreateChatGenerationRun,
    _contains_sensitive_key,
    _event_cursor,
    _sanitize_request,
)
from storage import chat_generation_runs_db as runs_db
from storage import studio_db
from state.tool_policy import reset_tool_policy, set_tool_policy, set_tool_policy_default
from utils.paths import studio_db_path


@pytest.fixture(autouse = True)
def _clean_tool_policy():
    reset_tool_policy()
    yield
    reset_tool_policy()


def _seed_thread(
    thread_id = "thread-1",
    user_id = "user-1",
    text = "Hello",
    created_at = 1,
):
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "Chat",
            "modelType": "base",
            "modelId": "local",
            "createdAt": created_at,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": user_id,
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "createdAt": created_at + 1,
        }
    )


@pytest.fixture
def chat_home():
    _seed_thread()


def _request(**overrides):
    request = {"model": "local", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
    request.update(overrides)
    return request


def _model(**overrides):
    return CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(**overrides),
    )


def _tool_messages(arguments):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        }
    ]


def _assert_protected(message):
    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.upsert_chat_message(message)


def _create(
    run_id = "run-1",
    owner = "alice",
    request = None,
):
    return runs_db.create_run(
        run_id = run_id,
        owner_subject = owner,
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1" if run_id == "run-1" else f"assistant-{run_id}",
        request_payload = request or _request(),
    )


def test_create_is_owner_scoped_idempotent_and_binds_placeholder(chat_home):
    run, created = _create()
    replay, replay_created = _create()
    assert created is True and replay_created is False
    assert replay == run
    assert runs_db.get_run("run-1", "bob") is None
    assert [run["id"] for run in runs_db.list_active("thread-1")] == ["run-1"]
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"] == {
        "generationRunId": "run-1",
        "generationSeq": 0,
        "generationStatus": "queued",
        "serverManaged": True,
    }
    _assert_protected({**message, "content": [{"type": "text", "text": "stale overwrite"}]})
    synced = studio_db.sync_chat_messages("thread-1", [], prune_missing = True)
    assert {message["id"] for message in synced} == {"user-1", "assistant-1"}
    assert runs_db.get_run("run-1", "alice") is not None
    explicitly_pruned = studio_db.sync_chat_messages(
        "thread-1",
        [],
        prune_missing = True,
        deleted_message_ids = {"assistant-1"},
    )
    assert {message["id"] for message in explicitly_pruned} == {"user-1", "assistant-1"}
    assert runs_db.get_run("run-1", "alice") is not None
    with pytest.raises(runs_db.ChatGenerationConflictError):
        _create(owner = "bob")
    with pytest.raises(runs_db.ChatGenerationConflictError):
        _create(request = _request(max_tokens = 9))


def test_shared_thread_rejects_a_second_subjects_active_generation(chat_home):
    _create()
    with pytest.raises(runs_db.ChatGenerationConflictError, match = "active generation"):
        _create("run-2", owner = "bob")
    assert [run["id"] for run in runs_db.list_active("thread-1")] == ["run-1"]
    assert studio_db.get_chat_message("thread-1", "assistant-run-2") is None


def test_explicit_prune_deletes_terminal_generation_messages(chat_home):
    user = studio_db.get_chat_message("thread-1", "user-1")
    user["attachments"] = [{"id": "file-1", "name": "notes.txt"}]
    studio_db.upsert_chat_message(user)
    _create()
    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.delete_chat_attachment("user-1", "file-1")
    stale_assistant = studio_db.get_chat_message("thread-1", "assistant-1")
    active = studio_db.sync_chat_messages("thread-1", [], prune_missing = True)
    assert {message["id"] for message in active} == {"user-1", "assistant-1"}

    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    runs_db.finish_run("run-1", worker_token = token, status = "completed")
    assert studio_db.delete_chat_attachment("user-1", "file-1") is True
    user = studio_db.get_chat_message("thread-1", "user-1")
    assert user.get("attachments") == []

    retained = studio_db.sync_chat_messages("thread-1", [user], prune_missing = True)
    assert {message["id"] for message in retained} == {"user-1", "assistant-1"}
    assert runs_db.get_run("run-1", "alice") is not None

    deleted = studio_db.sync_chat_messages(
        "thread-1",
        [user],
        prune_missing = True,
        deleted_message_ids = {"assistant-1"},
    )
    assert [message["id"] for message in deleted] == ["user-1"]
    assert studio_db.get_chat_message("thread-1", "assistant-1") is None
    assert runs_db.get_run("run-1", "alice") is None

    stale_sync = studio_db.sync_chat_messages(
        "thread-1", [user, stale_assistant], prune_missing = True
    )
    assert [message["id"] for message in stale_sync] == ["user-1"]
    assert studio_db.get_chat_message("thread-1", "assistant-1") is None
    _assert_protected(stale_assistant)


def test_terminal_run_does_not_unprotect_a_message_shared_with_an_active_run(chat_home):
    user = studio_db.get_chat_message("thread-1", "user-1")
    user["attachments"] = [{"id": "file-1", "name": "notes.txt"}]
    studio_db.upsert_chat_message(user)
    _create()
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    runs_db.finish_run("run-1", worker_token = token, status = "completed")
    _create("run-2")

    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.delete_chat_attachment("user-1", "file-1")
    studio_db.sync_chat_messages(
        "thread-1",
        [],
        prune_missing = True,
        deleted_message_ids = {"user-1"},
    )

    assert studio_db.get_chat_message("thread-1", "user-1") is not None
    assert runs_db.get_run("run-2") is not None


def test_generation_message_writes_are_run_bound_and_monotonic(chat_home):
    _create()
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    runs_db.append_events("run-1", token, [("chunk", {"text": "A"})])
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    message["content"] = [{"type": "text", "text": "A"}]
    message["metadata"].update({"generationSeq": 3, "generationStatus": "running"})
    studio_db.upsert_chat_message(message)

    forged_terminal = {
        **message,
        "metadata": {
            **message["metadata"],
            "generationStatus": "completed",
            "generationSettled": True,
        },
    }
    _assert_protected(forged_terminal)
    for metadata in (
        {**message["metadata"], "generationSeq": 2},
        {**message["metadata"], "generationRunId": "other-run"},
    ):
        _assert_protected(
            {**message, "content": [{"type": "text", "text": "stale"}], "metadata": metadata}
        )
    runs_db.finish_run("run-1", worker_token = token, status = "completed", finish_reason = "length")
    stale = {
        **message,
        "content": [{"type": "text", "text": "downgraded"}],
        "metadata": {**message["metadata"], "generationStatus": "running"},
    }
    _assert_protected(stale)
    studio_db.sync_chat_messages("thread-1", [stale])
    stored = studio_db.get_chat_message("thread-1", "assistant-1")
    assert stored["content"] == [{"type": "text", "text": "A"}]
    assert stored["metadata"]["generationStatus"] == "completed"
    stored["metadata"]["generationSettled"] = True
    _assert_protected(stored)
    stored["metadata"].update(
        {"generationSeq": 4, "generationSettled": True, "responseDetails": {"durationMs": 1}}
    )
    studio_db.upsert_chat_message(stored)
    stored["metadata"]["generationSettled"] = False
    _assert_protected(stored)
    authoritative = studio_db.get_chat_message("thread-1", "assistant-1")
    stale = {
        **authoritative,
        "metadata": {
            key: value
            for key, value in authoritative["metadata"].items()
            if key not in {"incomplete", "responseDetails"}
        },
    }
    _assert_protected(stale)
    studio_db.sync_chat_messages("thread-1", [stale])
    preserved = studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]
    assert preserved["incomplete"] == {"reason": "length"}
    assert preserved["responseDetails"] == {"durationMs": 1}


def test_settled_generation_response_can_be_explicitly_edited(chat_home):
    _create()
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    assert runs_db.finish_run("run-1", worker_token = token, status = "completed", finish_reason = "stop")
    run = runs_db.get_run("run-1", "alice")
    stored = studio_db.get_chat_message("thread-1", "assistant-1")
    stored["metadata"].update(
        {
            "generationSeq": run["lastEventSeq"],
            "generationStatus": "completed",
            "generationSettled": True,
        }
    )
    studio_db.upsert_chat_message(stored)
    authoritative = studio_db.get_chat_message("thread-1", "assistant-1")
    edited = {key: value for key, value in authoritative.items() if key != "metadata"}
    edited["content"] = [{"type": "text", "text": "edited"}]

    _assert_protected(edited)
    saved = studio_db.upsert_chat_message(edited, allow_generation_edit = True)
    assert saved["content"] == [{"type": "text", "text": "edited"}]
    assert saved.get("metadata") is None
    assert runs_db.get_run("run-1", "alice") is None
    _assert_protected(authoritative)


def test_explicit_edit_keeps_display_metadata_but_not_the_run_claim(chat_home):
    """The shape the edit pencil sends: the details the reply is shown with, minus the run's claim.

    Passing the whole stored metadata back through is refused, because the record still says the
    run owns the turn and detaching it is the whole point of an explicit edit.
    """
    _create()
    token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", token)
    streamed = studio_db.get_chat_message("thread-1", "assistant-1")
    streamed["metadata"].update(
        {
            "generationStatus": "running",
            "timing": {"tokensPerSecond": 42.5, "durationMs": 1200},
            "contextUsage": {"promptTokens": 900, "contextLength": 4096},
        }
    )
    studio_db.upsert_chat_message(streamed)
    assert runs_db.finish_run(
        "run-1", worker_token = token, status = "completed", finish_reason = "length"
    )
    run = runs_db.get_run("run-1", "alice")
    settled = studio_db.get_chat_message("thread-1", "assistant-1")
    settled["metadata"].update(
        {
            "generationSeq": run["lastEventSeq"],
            "generationStatus": "completed",
            "generationSettled": True,
        }
    )
    studio_db.upsert_chat_message(settled)
    authoritative = studio_db.get_chat_message("thread-1", "assistant-1")
    assert authoritative["metadata"]["incomplete"] == {"reason": "length"}

    edited = {**authoritative, "content": [{"type": "text", "text": "edited"}]}
    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.upsert_chat_message(edited, allow_generation_edit = True)

    edited["metadata"] = {
        key: value
        for key, value in authoritative["metadata"].items()
        if key
        not in {
            "serverManaged",
            "generationRunId",
            "generationSeq",
            "generationStatus",
            "generationSettled",
        }
    }
    saved = studio_db.upsert_chat_message(edited, allow_generation_edit = True)
    assert saved["content"] == [{"type": "text", "text": "edited"}]
    assert saved["metadata"] == {
        "incomplete": {"reason": "length"},
        "timing": {"tokensPerSecond": 42.5, "durationMs": 1200},
        "contextUsage": {"promptTokens": 900, "contextLength": 4096},
    }
    assert runs_db.get_run("run-1", "alice") is None


def test_batched_events_have_gapless_cursor_and_terminal_flush(chat_home):
    run, _created = _create()
    worker_token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", worker_token) is True
    assert runs_db.append_events(
        "run-1", worker_token, [("chunk", {"i": 1}), ("chunk", {"i": 2})]
    ) == [3, 4]
    cancelling = runs_db.request_cancel("run-1", "alice")
    assert cancelling["status"] == "cancelling"
    terminal = runs_db.finish_run(
        "run-1",
        worker_token = worker_token,
        status = "completed",
        finish_reason = "stop",
        pending_events = [("chunk", {"i": 3})],
    )
    assert terminal["status"] == "cancelled"
    events = runs_db.list_events("run-1")
    assert [event["seq"] for event in events] == list(range(1, 8))
    assert [event["payload"].get("i") for event in events if event["type"] == "chunk"] == [1, 2, 3]
    assert runs_db.list_events("run-1", after = 4)[0]["seq"] == 5
    assert runs_db.request_cancel("run-1", "alice")["lastEventSeq"] == 7


def test_batched_events_preserve_receipt_timestamps(chat_home):
    _create()
    worker_token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", worker_token) is True
    runs_db.append_events(
        "run-1",
        worker_token,
        [("chunk", {"i": 1}, 1001), ("chunk", {"i": 2}, 1002)],
    )
    chunks = [event for event in runs_db.list_events("run-1") if event["type"] == "chunk"]
    assert [event["createdAt"] for event in chunks] == [1001, 1002]


def test_stream_batches_do_not_rewrite_studio_db_while_the_keeper_holds_the_wal(chat_home):
    """#9934: each event batch closed the last connection, checkpointing studio.db."""
    _create()
    worker_token = runs_db.get_worker_token("run-1")
    db_path = Path(str(studio_db_path()))
    wal_path = Path(f"{db_path}-wal")

    assert studio_db.open_wal_keeper() is True
    try:
        before = hashlib.sha256(db_path.read_bytes()).hexdigest()
        for index in range(5):
            runs_db.append_events("run-1", worker_token, [("chunk", {"text": str(index)})])
            assert wal_path.is_file()
        assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    finally:
        studio_db.close_wal_keeper()
    assert not wal_path.exists()
    assert len(runs_db.list_events("run-1")) >= 5


def test_cancel_before_registration_and_startup_orphan_reconciliation(chat_home):
    queued, _created = _create("queued")
    cancelled = runs_db.request_cancel("queued", "alice")
    assert cancelled["status"] == "cancelled"
    assert runs_db.mark_running("queued", runs_db.get_worker_token("queued")) is False
    orphan, _created = _create("orphan", request = _request(seed = 2))
    assert runs_db.mark_running("orphan", runs_db.get_worker_token("orphan")) is True
    assert runs_db.reconcile_orphaned_runs() == 1
    orphan = runs_db.get_run("orphan", "alice")
    assert (orphan["status"], orphan["finishReason"]) == ("failed", "interrupted")
    assert runs_db.list_events("orphan")[-1]["payload"]["interrupted"] is True


def test_a_stop_in_flight_survives_a_restart_as_a_cancellation(chat_home):
    """Stop recorded, worker not yet settled, Studio restarts.

    Reporting this as a backend failure would tell the user Studio broke when in fact they
    stopped it, and finish_run already settles the same case as cancelled.
    """
    _create()
    runs_db.mark_running("run-1", runs_db.get_worker_token("run-1"))
    assert runs_db.request_cancel("run-1", "alice")["status"] == "cancelling"

    assert runs_db.reconcile_orphaned_runs() == 1

    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("cancelled", "cancelled")
    assert run["error"] is None
    assert runs_db.list_events("run-1")[-1]["type"] == "run.cancelled"


def test_an_uncancelled_run_still_reconciles_as_interrupted(chat_home):
    """The cancellation branch above must not swallow a genuine restart."""
    _create()
    runs_db.mark_running("run-1", runs_db.get_worker_token("run-1"))

    assert runs_db.reconcile_orphaned_runs() == 1

    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
    assert run["error"] == "Studio restarted during generation"
    assert runs_db.list_events("run-1")[-1]["payload"]["interrupted"] is True


def test_deleted_run_id_is_tombstoned_against_stale_tabs(chat_home):
    _original, _created = _create()
    studio_db.delete_chat_threads(["thread-1"])
    _seed_thread("thread-2", "user-2", "Next", 3)
    with pytest.raises(runs_db.ChatGenerationConflictError, match = "already been used"):
        runs_db.create_run(
            run_id = "run-1",
            owner_subject = "alice",
            thread_id = "thread-2",
            user_message_id = "user-2",
            assistant_message_id = "assistant-2",
            request_payload = _request(seed = 2),
        )
    assert runs_db.get_run("run-1", "alice") is None


_SYNC_USER = {
    "id": "user-1",
    "threadId": "thread-1",
    "role": "user",
    "content": [{"type": "text", "text": "Hello"}],
    "createdAt": 2,
}


def _edited_generated_assistant():
    """Settle a generated assistant the way the pipeline does, then edit it by hand.

    Returns the stale pre-edit copy another tab would still be holding.
    """
    studio_db.upsert_chat_message(
        {
            "id": "assistant-1",
            "threadId": "thread-1",
            "role": "assistant",
            "parentId": "user-1",
            "content": [],
            "createdAt": 3,
            "metadata": {},
        }
    )
    _create()
    token = runs_db.get_worker_token("run-1")
    runs_db.mark_running("run-1", token)
    runs_db.finish_run("run-1", worker_token = token, status = "completed", finish_reason = "stop")

    metadata = dict(studio_db.get_chat_message("thread-1", "assistant-1")["metadata"])
    metadata["generationSeq"] = int(runs_db.get_run("run-1", "alice")["lastEventSeq"])
    metadata["generationSettled"] = True
    settled = {
        "id": "assistant-1",
        "threadId": "thread-1",
        "role": "assistant",
        "parentId": "user-1",
        "content": [{"type": "text", "text": "generated answer"}],
        "createdAt": 3,
        "metadata": metadata,
    }
    studio_db.sync_chat_messages("thread-1", [_SYNC_USER, settled])
    assert (
        studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]["generationSettled"]
        is True
    ), "the settle write did not land, so anything built on it would prove nothing"
    stale_tab_copy = json.loads(json.dumps(settled))

    studio_db.upsert_chat_message(
        {
            "id": "assistant-1",
            "threadId": "thread-1",
            "role": "assistant",
            "parentId": "user-1",
            "content": [{"type": "text", "text": "edited by hand"}],
            "createdAt": 3,
            "metadata": {},
        },
        allow_generation_edit = True,
    )
    assert runs_db.get_run("run-1", "alice") is None, "the edit did not detach the run"
    return stale_tab_copy


def test_a_stale_tab_sync_cannot_prune_an_edited_generated_assistant(chat_home):
    """Editing a settled generated answer detaches it; another open tab must not delete it.

    The edit drops the run row, so the id stops counting as generation-linked. A stale tab
    still holding the pre-edit copy has that copy filtered out as tombstoned, which would
    otherwise leave the id absent from the requested set and inside the prune.
    """
    stale_tab_copy = _edited_generated_assistant()

    studio_db.sync_chat_messages("thread-1", [_SYNC_USER, stale_tab_copy], prune_missing = True)

    survivor = studio_db.get_chat_message("thread-1", "assistant-1")
    assert survivor is not None, "the stale tab's sync deleted the user's edited message"
    # Kept, but still not writable by the stale copy.
    assert survivor["content"] == [{"type": "text", "text": "edited by hand"}]


def test_an_explicit_delete_still_removes_a_detached_generated_assistant(chat_home):
    """The retention above is snapshot-pruning only; naming the id must still delete it."""
    stale_tab_copy = _edited_generated_assistant()

    studio_db.sync_chat_messages(
        "thread-1",
        [_SYNC_USER, stale_tab_copy],
        prune_missing = True,
        deleted_message_ids = ["assistant-1"],
    )
    assert studio_db.get_chat_message("thread-1", "assistant-1") is None


def test_completed_keyed_run_enters_backend_finalization_state(chat_home):
    _create(request=_request(finalization_idempotency_key="run-1"))
    worker = runs_db.get_worker_run("run-1")
    assert worker is not None
    _run, _owner, token = worker
    assert runs_db.mark_running("run-1", token)
    terminal = runs_db.finish_run(
        "run-1",
        worker_token=token,
        status="completed",
        finish_reason="stop",
    )
    assert terminal is not None
    assert terminal["finalizationStatus"] == "pending"
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"]["turnFinalizationStatus"] == "pending"
    assert message["metadata"]["turnFinalizationIdempotencyKey"] == "run-1"
    assert [event["type"] for event in runs_db.list_events("run-1")][-2:] == [
        "run.completed",
        "turn_finalization.pending",
    ]


def test_length_and_unkeyed_runs_never_start_finalization(chat_home):
    for run_id, request, reason in (
        ("run-length", _request(finalization_idempotency_key="run-length"), "length"),
        ("run-old", _request(), "stop"),
    ):
        runs_db.create_run(
            run_id=run_id,
            owner_subject="alice",
            thread_id="thread-1",
            user_message_id="user-1",
            assistant_message_id=f"assistant-{run_id}",
            request_payload=request,
        )
        worker = runs_db.get_worker_run(run_id)
        assert worker is not None
        _run, _owner, token = worker
        assert runs_db.mark_running(run_id, token)
        terminal = runs_db.finish_run(
            run_id,
            worker_token=token,
            status="completed",
            finish_reason=reason,
        )
        assert terminal is not None
        assert terminal["finalizationStatus"] == "none"


def test_finalization_claim_is_atomic_and_recoverable(chat_home, monkeypatch):
    _create(request=_request(finalization_idempotency_key="run-1"))
    worker = runs_db.get_worker_run("run-1")
    assert worker is not None
    _run, _owner, generation_token = worker
    assert runs_db.mark_running("run-1", generation_token)
    runs_db.finish_run(
        "run-1",
        worker_token=generation_token,
        status="completed",
        finish_reason="stop",
    )
    claim = runs_db.claim_finalization("run-1")
    assert claim is not None
    run, finalization_token, owner = claim
    assert owner == "alice"
    assert run["finalizationStatus"] == "running"
    assert runs_db.claim_finalization("run-1") is None
    assert runs_db.append_finalization_event(
        "run-1",
        finalization_token,
        "turn_finalization.memory",
        {"stored": True},
    )
    conn = runs_db._connect()
    lease_expires = conn.execute(
        "SELECT finalization_lease_expires_at FROM chat_generation_runs WHERE id='run-1'"
    ).fetchone()[0]
    conn.close()
    monkeypatch.setattr(runs_db, "now_ms", lambda: int(lease_expires) + 1)
    assert runs_db.requeue_interrupted_finalizations() == ["run-1"]
    recovered = runs_db.get_run("run-1")
    assert recovered["finalizationStatus"] == "pending"
    replay = runs_db.claim_finalization("run-1")
    assert replay is not None
    _run, replay_token, _owner = replay
    settled = runs_db.finish_finalization(
        "run-1",
        replay_token,
        status="completed",
        receipt={"memory": {"stored": True}},
    )
    assert settled["finalizationStatus"] == "completed"
    assert runs_db.finish_finalization(
        "run-1",
        replay_token,
        status="completed",
    ) is None
    assert runs_db.get_run("run-1")["finalizationStatus"] == "completed"
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"]["turnFinalizationStatus"] == "completed"


def test_live_finalization_claim_is_not_requeued_and_stale_owner_is_fenced(
    chat_home,
    monkeypatch,
):
    _create(request=_request(finalization_idempotency_key="run-1"))
    _run, _owner, generation_token = runs_db.get_worker_run("run-1")
    assert runs_db.mark_running("run-1", generation_token)
    runs_db.finish_run(
        "run-1",
        worker_token=generation_token,
        status="completed",
        finish_reason="stop",
    )
    _run, token, _owner = runs_db.claim_finalization("run-1")
    assert runs_db.requeue_interrupted_finalizations() == []
    assert runs_db.renew_finalization_claim("run-1", token)

    conn = runs_db._connect()
    expires = conn.execute(
        "SELECT finalization_lease_expires_at FROM chat_generation_runs WHERE id='run-1'"
    ).fetchone()[0]
    conn.close()
    monkeypatch.setattr(runs_db, "now_ms", lambda: int(expires) + 1)
    assert runs_db.requeue_interrupted_finalizations() == ["run-1"]
    assert (
        runs_db.append_finalization_event(
            "run-1",
            token,
            "turn_finalization.memory",
            {"stored": True},
        )
        is None
    )


def test_generation_restart_requeue_is_fenced_by_token_and_progress(chat_home):
    _create(request=_request(finalization_idempotency_key="run-1"))
    worker = runs_db.get_worker_run("run-1")
    assert worker is not None
    _run, _owner, token = worker
    assert runs_db.mark_running("run-1", token)
    snapshot = runs_db.get_recovery_snapshot("run-1")
    assert snapshot is not None
    _run, expected_token, expected_progress = snapshot
    resume = _request(finalization_idempotency_key="run-1")

    # A live worker advanced after planning, so the stale plan cannot rotate it.
    conn = runs_db._connect()
    conn.execute(
        "UPDATE chat_generation_runs SET progress_at=? WHERE id='run-1'",
        (expected_progress + 1,),
    )
    conn.commit()
    conn.close()
    assert (
        runs_db.requeue_run_for_restart(
            "run-1",
            resume,
            expected_worker_token=expected_token,
            expected_progress_at=expected_progress,
            stale_before_ms=expected_progress,
        )
        is None
    )
    assert runs_db.get_worker_token("run-1") == token

    fresh = runs_db.get_recovery_snapshot("run-1")
    assert fresh is not None
    _run, expected_token, expected_progress = fresh
    recovered = runs_db.requeue_run_for_restart(
        "run-1",
        resume,
        expected_worker_token=expected_token,
        expected_progress_at=expected_progress,
        stale_before_ms=expected_progress,
    )
    assert recovered is not None and recovered["status"] == "queued"
    assert runs_db.get_worker_token("run-1") != token


def test_recovery_planner_cannot_requeue_a_run_that_progressed_after_planning(chat_home):
    from core.inference.durable_agent_recovery import plan_orphaned_runs, requeue_planned_runs

    request = _request(
        cancel_id="run-1",
        generation_run_id="run-1",
        finalization_idempotency_key="run-1",
    )
    _create(request=request)
    worker = runs_db.get_worker_run("run-1")
    assert worker is not None
    _run, _owner, token = worker
    assert runs_db.mark_running("run-1", token)
    plans = plan_orphaned_runs()
    assert len(plans) == 1 and plans[0].safe
    assert plans[0].expected_progress_at is not None

    # A second backend starting while this producer is between heartbeats sees
    # an unchanged token/snapshot, but cannot steal it before lease expiry.
    assert requeue_planned_runs(
        plans,
        stale_before_ms=int(plans[0].expected_progress_at) - 1,
    ) == []
    assert runs_db.get_worker_token("run-1") == token

    conn = runs_db._connect()
    conn.execute(
        "UPDATE chat_generation_runs SET progress_at=progress_at+1 WHERE id='run-1'"
    )
    conn.commit()
    conn.close()
    assert requeue_planned_runs(
        plans,
        stale_before_ms=plans[0].expected_progress_at,
    ) == []
    assert runs_db.get_worker_token("run-1") == token

    fresh = plan_orphaned_runs()
    recovered = requeue_planned_runs(
        fresh,
        stale_before_ms=fresh[0].expected_progress_at,
    )
    assert len(recovered) == 1 and recovered[0]["status"] == "queued"


@pytest.mark.asyncio
async def test_fresh_crash_waits_for_expiry_then_sweeper_recovers_it(
    chat_home,
    monkeypatch,
):
    from types import SimpleNamespace

    from core.inference import chat_generation_runs as runs_mod
    from utils.account_context import OWNER

    clock = {"now": runs_db.now_ms()}
    monkeypatch.setattr(runs_db, "now_ms", lambda: clock["now"])
    monkeypatch.setattr(runs_mod, "sweepable_job_accounts", lambda: [OWNER])
    request = _request(
        cancel_id="run-1",
        generation_run_id="run-1",
        finalization_idempotency_key="run-1",
    )
    _create(request=request)
    token = runs_db.get_worker_token("run-1")
    assert token is not None and runs_db.mark_running("run-1", token)

    started: list[str] = []
    supervisor = SimpleNamespace(
        _tasks={},
        start_recovered=lambda run_id, **_kwargs: started.append(run_id),
        cancel=lambda _run_id: None,
    )
    sweeper = runs_mod.ChatGenerationLeaseSweeper(
        SimpleNamespace(state=SimpleNamespace(chat_generation_supervisor=supervisor)),
        timeout_s=1.0,
    )
    assert await sweeper.sweep_once() == []
    assert runs_db.get_worker_token("run-1") == token

    clock["now"] += int(sweeper._timeout * 1000) + 1
    assert await sweeper.sweep_once() == ["run-1"]
    assert started == ["run-1"]
    assert runs_db.get_run("run-1")["status"] == "queued"
    assert runs_db.get_worker_token("run-1") != token


@pytest.mark.asyncio
async def test_disabled_sweeper_still_runs_deferred_recovery(chat_home, monkeypatch):
    from types import SimpleNamespace

    from core.inference import chat_generation_runs as runs_mod
    from core.inference.durable_agent_recovery import plan_orphaned_runs

    clock = {"now": runs_db.now_ms()}
    monkeypatch.setattr(runs_db, "now_ms", lambda: clock["now"])
    monkeypatch.setattr(runs_mod, "_applied_lease_timeout", lambda value: value)
    request = _request(
        cancel_id="run-1",
        generation_run_id="run-1",
        finalization_idempotency_key="run-1",
    )
    _create(request=request)
    token = runs_db.get_worker_token("run-1")
    assert token is not None and runs_db.mark_running("run-1", token)
    plan = plan_orphaned_runs()[0]
    assert plan.safe

    supervisor = runs_mod.ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    started: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "start_recovered",
        lambda run_id, **_kwargs: started.append(run_id),
    )
    supervisor.schedule_recovery_plans([plan], timeout_s=0.001)
    clock["now"] += 2
    await asyncio.gather(*list(supervisor._recovery_tasks.values()))
    assert started == ["run-1"]
    assert runs_db.get_run("run-1")["status"] == "queued"


def test_unsafe_ambiguous_recovery_plan_is_not_preserved_at_startup(chat_home):
    from core.inference.durable_agent_recovery import plan_orphaned_runs

    request = _request(
        cancel_id="run-1",
        generation_run_id="run-1",
        finalization_idempotency_key="run-1",
    )
    _create(request=request)
    token = runs_db.get_worker_token("run-1")
    assert token is not None and runs_db.mark_running("run-1", token)
    runs_db.append_events(
        "run-1",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "call-1",
                    "arguments": {"command": "touch marker"},
                },
            ),
            (
                "tool_execution.started",
                {
                    "execution_id": "exec-1",
                    "tool_name": "terminal",
                    "tool_call_id": "call-1",
                    "effect_state": "started",
                },
            ),
        ],
    )
    plans = plan_orphaned_runs()
    assert len(plans) == 1 and plans[0].safe is False
    preserve = {plan.run_id for plan in plans if plan.safe}
    assert preserve == set()
    assert runs_db.reconcile_runs(preserve_run_ids=preserve) == ["run-1"]
    assert runs_db.get_run("run-1")["status"] == "failed"


@pytest.mark.parametrize("message", ["database is locked", "database disk image is malformed"])
def test_additive_migration_never_exposes_partial_schema(chat_home, monkeypatch, message):
    real_get = runs_db.get_connection

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _BrokenMigration:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA table_info(chat_generation_runs)":
                rows = self._inner.execute(sql).fetchall()
                return _Rows([row for row in rows if row[1] != "finalization_lease_expires_at"])
            if "ADD COLUMN finalization_lease_expires_at" in sql:
                raise __import__("sqlite3").OperationalError(message)
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(runs_db, "_schema_ready", set())
    monkeypatch.setattr(runs_db, "_MIGRATION_LOCK_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(runs_db, "get_connection", lambda: _BrokenMigration(real_get()))
    with pytest.raises(__import__("sqlite3").OperationalError, match=message):
        runs_db._connect()


@pytest.mark.parametrize(
    "override,detail",
    [
        ({"provider_id": "external"}, "only for local"),
        ({"rag_scope": {"access_token": "secret"}}, "Credentials"),
        ({"rag_scope": {"signing_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"ssh_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"encryption_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"secret_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"api_token": "secret"}}, "Credentials"),
        (
            {
                "messages": [
                    {"role": "user", "content": "Hello", "extra_content": {"api_key": "secret"}}
                ]
            },
            "Credentials",
        ),
        (
            {"messages": _tool_messages('{"api_key":"secret"}')},
            "Credentials",
        ),
    ],
)
def test_request_sanitization_rejects_nonlocal_or_sensitive_payloads(override, detail):
    with pytest.raises(Exception, match = detail):
        _sanitize_request(_model(**override))


def test_request_sanitization_pins_server_owned_fields():
    sanitized = _sanitize_request(
        _model(
            stream = False,
            cancel_id = "legacy",
            thread_id = "wrong",
            finalization_idempotency_key = "run-1",
        )
    )
    assert sanitized["stream"] is True
    assert sanitized["cancel_id"] == "run-1"
    assert sanitized["thread_id"] == "thread-1"
    assert sanitized["finalization_idempotency_key"] == "run-1"


def test_effect_boundaries_are_durable_barriers():
    for event_type in ("tool_start", "tool_end", "adaptive_checkpoint"):
        assert _requires_durable_barrier({"type": event_type}) is True
    for event_type in ("tool_status", "tool_output", "reasoning_summary", ""):
        assert _requires_durable_barrier({"type": event_type}) is False


def test_durable_tool_execution_journals_started_and_finished(chat_home):
    runs_db.create_run(
        run_id="run-tool",
        owner_subject="alice",
        thread_id="thread-1",
        user_message_id="user-1",
        assistant_message_id="assistant-tool",
        request_payload=_request(enable_tools=True),
    )
    worker = runs_db.get_worker_run("run-tool")
    assert worker is not None
    _run, _owner, token = worker
    assert runs_db.mark_running("run-tool", token)

    with durable_tool_run("run-tool", token):
        generator = stream_tool_execution(
            lambda _callback: "effect-result",
            tool_name="terminal",
            tool_call_id="call-1",
        )
        with pytest.raises(StopIteration) as stopped:
            while True:
                next(generator)
        assert stopped.value.value == "effect-result"

    journal = [
        event
        for event in runs_db.list_events("run-tool")
        if event["type"].startswith("tool_execution.")
    ]
    assert [event["type"] for event in journal] == [
        "tool_execution.started",
        "tool_execution.finished",
    ]
    assert journal[0]["payload"]["tool_call_id"] == "call-1"
    assert journal[1]["payload"]["result"] == "effect-result"
    assert journal[0]["payload"]["execution_id"] == journal[1]["payload"]["execution_id"]


def test_request_sanitization_admits_durable_local_tool_runs():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    sanitized = _sanitize_request(
        _model(
            enable_tools=True,
            mcp_enabled=True,
            tools=tools,
        )
    )
    assert sanitized["enable_tools"] is True
    assert sanitized["mcp_enabled"] is True
    assert sanitized["tools"] == tools
    assert sanitized["generation_run_id"] == "run-1"
    assert sanitized["cancel_id"] == "run-1"


def test_request_sanitization_treats_message_text_as_data():
    sanitized = _sanitize_request(
        _model(messages = [{"role": "user", "content": '{"api_key":"example"}'}])
    )
    assert sanitized["messages"][0]["content"] == '{"api_key":"example"}'


@pytest.mark.parametrize("key", ["key", "lookup_key", "monkey", "hockey", "keyboard"])
def test_request_sanitization_accepts_benign_tool_argument_keys(key):
    arguments = json.dumps({key: "value"})
    sanitized = _sanitize_request(_model(messages = _tool_messages(arguments)))
    assert sanitized["messages"][0]["tool_calls"][0]["function"]["arguments"] == arguments


@pytest.mark.parametrize(
    "key",
    ["api_key", "access_key", "private_key", "secret_key", "signing_key", "ssh_key"],
)
def test_request_sanitization_rejects_known_credential_keys(key):
    arguments = json.dumps({key: "secret"})
    with pytest.raises(Exception, match = "Credentials"):
        _sanitize_request(_model(messages = _tool_messages(arguments)))


@pytest.mark.parametrize("value", ['{"api_key":"secret"', '"{\\"api_key\\":\\"secret\\"}"'])
def test_request_sanitization_scans_json_string_envelopes(value):
    assert _contains_sensitive_key(value) is True


def test_request_sanitization_bounds_nested_envelopes():
    nested = {"value": None}
    for _ in range(100):
        nested = {"value": nested}
    assert _contains_sensitive_key(nested) is True
    assert _contains_sensitive_key("[" * 5000 + "0" + "]" * 5000) is True
    with pytest.raises(Exception, match = "Credentials"):
        _sanitize_request(
            _model(messages = [{"role": "user", "content": "hello", "extra_content": nested}])
        )


@pytest.mark.parametrize("messages", [None, 1, [{"role": "user", "content": None}]])
def test_request_sanitization_returns_json_safe_validation_errors(messages):
    with pytest.raises(Exception) as exc_info:
        _sanitize_request(_model(messages = messages))
    assert exc_info.value.status_code == 422
    json.dumps(exc_info.value.detail)


@pytest.mark.parametrize(
    "overrides",
    [{"provider_id": ""}, {"provider_id": None, "encrypted_api_key": None}, {"tools": []}],
)
def test_request_sanitization_accepts_empty_optional_routing(overrides):
    assert _sanitize_request(_model(**overrides))["stream"] is True


def test_request_sanitization_admits_launcher_default_tool_capability():
    set_tool_policy_default(True)
    sanitized = _sanitize_request(_model())
    assert sanitized["stream"] is True
    assert sanitized["generation_run_id"] == "run-1"


def test_request_sanitization_admits_cli_tool_override_even_when_request_disables_tools():
    set_tool_policy(True)
    sanitized = _sanitize_request(_model(enable_tools = False))
    assert sanitized["enable_tools"] is False
    assert sanitized["generation_run_id"] == "run-1"


def test_request_sanitization_does_not_preflight_dynamic_checkpoint_tool_policy(monkeypatch):
    import routes.inference as inference_routes
    monkeypatch.setattr(
        inference_routes, "_checkpoint_recall_may_enable_tools", lambda request: True, raising = False
    )
    sanitized = _sanitize_request(_model(enable_tools = False))
    assert sanitized["enable_tools"] is False
    assert sanitized["generation_run_id"] == "run-1"


def test_event_cursor_rejects_values_outside_sqlite_integer_range():
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(10**30, None)
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(None, str(10**30))
    with pytest.raises(Exception, match = "must be an integer"):
        _event_cursor(None, "²")
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(None, "9" * 4301)


@pytest.mark.parametrize("field", ["image_base64", "audio_base64", "video_base64"])
def test_request_sanitization_rejects_inline_media(field):
    """Media stays on the legacy stream on the server too, not only in the composer.

    Recovery rebuilds text and reasoning deltas, and the request is persisted verbatim,
    so admitting one of these would park a base64 blob in request_json for the life of
    the thread and hand the client a transcript it has no way to replay.
    """
    with pytest.raises(Exception, match = "legacy streaming path"):
        _sanitize_request(_model(**{field: "iVBORw0KGgo="}))
