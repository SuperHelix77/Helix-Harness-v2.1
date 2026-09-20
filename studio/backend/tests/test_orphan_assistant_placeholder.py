# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from storage import studio_db


def _reset_studio_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())


def _thread():
    return {
        "id": "thread-1",
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _message(**overrides):
    row = {
        "id": "a1",
        "threadId": "thread-1",
        "parentId": None,
        "role": "assistant",
        "content": [],
        "createdAt": 2,
    }
    row.update(overrides)
    return row


def test_delete_orphan_assistant_placeholder(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(
        _message(id = "u1", role = "user", content = [{"type": "text", "text": "hi"}], createdAt = 1)
    )
    studio_db.upsert_chat_message(_message(id = "a1", parentId = "u1", content = []))
    assert studio_db.delete_orphan_assistant_placeholder("thread-1", "a1") is True
    assert studio_db.get_chat_message("thread-1", "a1") is None
    assert studio_db.get_chat_message("thread-1", "u1") is not None


def test_delete_orphan_rejects_nonempty_assistant(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(
        _message(content = [{"type": "text", "text": "hello"}])
    )
    assert studio_db.delete_orphan_assistant_placeholder("thread-1", "a1") is False
    assert studio_db.get_chat_message("thread-1", "a1") is not None


def test_delete_orphan_rejects_user_message(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(
        _message(id = "u1", role = "user", content = [{"type": "text", "text": "hi"}], createdAt = 1)
    )
    assert studio_db.delete_orphan_assistant_placeholder("thread-1", "u1") is False
