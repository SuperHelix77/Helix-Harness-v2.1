# SPDX-License-Identifier: AGPL-3.0-only

"""Adversarial storage-only gates for the disabled ObservationPack slice."""

from __future__ import annotations

import hashlib
import inspect
import os
import sqlite3
from pathlib import Path

import pytest

from auth import storage
from core.inference import observation_pack as op


@pytest.fixture
def isolated_observation_store(tmp_path, monkeypatch):
    auth_db = tmp_path / "auth" / "auth.db"
    monkeypatch.setattr(storage, "DB_PATH", auth_db)
    monkeypatch.setattr(storage, "_auth_schema_ready", set())
    monkeypatch.setattr(storage, "_observation_master_key_cache", None)
    monkeypatch.setattr(storage, "_observation_master_key_cache_path", None)
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    yield tmp_path


def _candidate(root: Path, *, execution = "exec-a", text = None, **extra):
    text = extra.pop("canonical_text", text)
    if text is None:
        text = ("αβγ-日本語\n" * 2500) + "tail"
    return op.build_observation_candidate(
        account_id = "acct-a",
        thread_id = "thread-a",
        source_run_id = "run-a",
        execution_id = execution,
        tool_name = "python",
        arguments_fingerprint = "args-v1",
        canonical_text = text,
        fallback_result = "ordinary bounded fallback",
        observation_pack_version = extra.pop("observation_pack_version", 1),
        **extra,
    )


def _published_binding(root: Path, *, execution = "exec-a", text = None):
    staged = _candidate(root, execution = execution, text = text)
    published = op.publish_observation_candidate(staged, storage_root = root / "observation-store")
    assert published.state == "published"
    assert published.available is False
    metadata = dict(published.binding_metadata())
    metadata["state"] = "available"  # the test simulates the later finish transaction
    metadata["finished_event_seq"] = 1
    return published, op.ObservationBinding(**metadata)


def test_observation_key_is_dedicated_stable_and_no_binding_schema(isolated_observation_store):
    credential = storage.get_or_create_credential_encryption_key()
    first = storage.get_or_create_observation_master_key()
    assert first != credential
    key_id = storage.get_observation_master_key_id()
    assert hashlib.sha256(first).hexdigest() in key_id

    storage._observation_master_key_cache = None
    assert storage.get_observation_master_key() == first
    rotated = b"r" * 32
    conn = storage.get_connection()
    try:
        conn.execute(
            "UPDATE app_secrets SET value = ? WHERE key = ?",
            (rotated.hex(), storage._OBSERVATION_MASTER_KEY_DB_KEY),
        )
        conn.commit()
    finally:
        conn.close()
    assert storage.get_or_create_observation_master_key() == rotated
    assert storage.get_observation_master_key_id() == storage.observation_master_key_id_for(rotated)

    # Restore the original key for the remainder of the row-lifecycle checks.
    conn = storage.get_connection()
    try:
        conn.execute(
            "UPDATE app_secrets SET value = ? WHERE key = ?",
            (first.hex(), storage._OBSERVATION_MASTER_KEY_DB_KEY),
        )
        conn.commit()
    finally:
        conn.close()
    conn = storage.get_connection()
    try:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "observation_bindings" not in tables
        conn.execute(
            "DELETE FROM app_secrets WHERE key = ?", (storage._OBSERVATION_MASTER_KEY_DB_KEY,)
        )
        conn.commit()
    finally:
        conn.close()
    storage._observation_master_key_cache = None
    assert storage.get_observation_master_key(refresh = True) is None
    replacement = storage.get_or_create_observation_master_key()
    assert replacement != first
    assert storage.get_observation_master_key_id() != key_id


def test_hkdf_domains_and_opaque_handles_bind_account_and_thread(isolated_observation_store):
    encryption = op.derive_observation_encryption_key("acct-a")
    handle_key = op.derive_observation_handle_key("acct-a")
    assert encryption != handle_key
    assert encryption != op.derive_observation_encryption_key("acct-b")
    digest = hashlib.sha256(b"same").hexdigest()
    one = op.make_observation_handle("acct-a", "thread-a", digest)
    assert one == op.make_observation_handle("acct-a", "thread-a", digest)
    assert one != op.make_observation_handle("acct-a", "thread-b", digest)
    assert one != op.make_observation_handle("acct-b", "thread-a", digest)
    assert op.parse_observation_handle(one) == (one.split(":")[2], digest)


@pytest.mark.parametrize("bad", ["", ".", "..", "acct/a", "acct\\a", "acct\x00a"])
def test_account_path_components_are_rejected(isolated_observation_store, bad):
    with pytest.raises(ValueError):
        op.observation_blob_root(bad, storage_root = isolated_observation_store / "store")
    with pytest.raises(ValueError):
        op.make_observation_handle(bad, "thread-a", "a" * 64)


def test_candidate_is_staged_only_and_publication_is_not_availability(isolated_observation_store):
    root = isolated_observation_store
    staged = _candidate(root)
    assert staged.state == "staged"
    assert staged.available is False
    assert staged.selected_result is None
    assert "state" not in staged.binding_metadata()
    assert not (root / "observation-store").exists()
    published = op.publish_observation_candidate(staged, storage_root = root / "observation-store")
    assert published.state == "published"
    assert published.available is False
    assert published.selected_result is None
    assert published.binding_metadata().get("state") is None
    assert published.projection != published.fallback_result

    conn = storage.get_connection()
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='observation_bindings'"
        ).fetchone() is None
    finally:
        conn.close()


def test_publish_recall_requires_caller_binding_and_preserves_utf8_boundaries(isolated_observation_store):
    root = isolated_observation_store
    published, binding = _published_binding(root)
    assert op.recall_observation(published).status != "available"
    assert op.recall_observation(
        binding, max_bytes = 4096, storage_root = root / "observation-store"
    ).status == "available"
    page = op.read_observation(
        binding, max_bytes = 4096, storage_root = root / "observation-store"
    )
    assert page.status == "available"
    pages = [page.text.encode("utf-8")]
    while page.next_offset is not None:
        page = op.read_observation(
            binding,
            offset = page.next_offset,
            max_bytes = 4096,
            storage_root = root / "observation-store",
        )
        assert page.status == "available"
        pages.append(page.text.encode("utf-8"))
    expected = ("αβγ-日本語\n" * 2500 + "tail").encode("utf-8")
    assert b"".join(pages) == expected
    assert "run_authorized" not in inspect.signature(op.recall_observation).parameters
    assert "current_run_id" not in inspect.signature(op.recall_observation).parameters


def test_recall_does_not_mutate_read_only_private_directories(isolated_observation_store, monkeypatch):
    root = isolated_observation_store
    _published, binding = _published_binding(root, execution = "read-only")

    def forbidden_fchmod(*_args, **_kwargs):
        raise AssertionError("recall must not chmod directory metadata")

    monkeypatch.setattr(op.os, "fchmod", forbidden_fchmod)
    page = op.recall_observation(
        binding, max_bytes = 4096, storage_root = root / "observation-store"
    )
    assert page.status == "available"


def test_recall_rejects_malformed_paging_and_tiny_budgets(isolated_observation_store):
    root = isolated_observation_store
    _published, binding = _published_binding(root, execution = "paging", text = "é\n" * 8000)
    for kwargs in (
        {"offset": -1},
        {"max_bytes": 0},
        {"max_bytes": True},
        {"max_lines": 0},
        {"max_lines": True},
        {"active_result_budget_bytes": 255},
        {"active_result_budget_bytes": True},
    ):
        page = op.recall_observation(
            binding, storage_root = root / "observation-store", **kwargs
        )
        assert page.status in {"invalid_handle", "unavailable"}
        assert page.next_offset is None
    page = op.recall_observation(
        binding,
        max_bytes = 16_384,
        max_lines = 400,
        active_result_budget_bytes = 512,
        storage_root = root / "observation-store",
    )
    assert page.status == "available"
    assert page.byte_length <= 512
    assert page.text.count("\n") <= 400
    assert page.text.startswith("é\n")


def test_recall_exact_byte_boundaries_do_not_split_utf8(isolated_observation_store):
    root = isolated_observation_store
    text = "é中a\n" * 3000
    _published, binding = _published_binding(root, execution = "exact", text = text)
    page = op.recall_observation(
        binding, max_bytes = 3, max_lines = 400, storage_root = root / "observation-store"
    )
    assert page.status == "available"
    assert page.text == "é"
    assert page.byte_length == len("é".encode("utf-8"))
    assert page.next_offset == page.end_offset
    assert op.recall_observation(
        binding,
        offset = 1,
        max_bytes = 3,
        storage_root = root / "observation-store",
    ).status == "invalid_handle"


def test_projection_is_deterministic_and_bounded(isolated_observation_store):
    digest = hashlib.sha256(b"unused").hexdigest()
    handle = op.make_observation_handle("acct-a", "thread-a", digest)
    text = "head\n" + ("x" * 9000) + "\ntail"
    projection = op.build_observation_projection(handle, text)
    assert projection == op.build_observation_projection(handle, text)
    assert len(projection) <= op.MAX_PROJECTION_CHARS
    assert "exact canonical text retained; excerpt is incomplete context" in projection
    assert "read_observation" in projection


def test_symlink_ancestor_and_parent_replacement_fail_closed(isolated_observation_store):
    root = isolated_observation_store
    outside = root / "outside"
    outside.mkdir()
    link_parent = root / "link-parent"
    link_parent.symlink_to(outside, target_is_directory = True)
    candidate = _candidate(root, execution = "symlink-ancestor")
    with pytest.raises(op.ObservationPathError):
        op.publish_observation_candidate(candidate, storage_root = link_parent / "nested")

    _published, binding = _published_binding(root, execution = "parent-replace")
    store = root / "observation-store"
    account_root = store / "acct-a"
    saved = account_root / "blobs"
    replacement = root / "replacement"
    replacement.mkdir()
    saved.rename(root / "blobs-saved")
    saved.symlink_to(replacement, target_is_directory = True)
    page = op.recall_observation(binding, storage_root = store)
    assert page.status == "corrupt"


def test_descriptor_relative_creation_rejects_injected_parent_replacement(
    isolated_observation_store, monkeypatch
):
    root = isolated_observation_store / "race-store"
    outside = isolated_observation_store / "race-outside"
    outside.mkdir()
    candidate = _candidate(isolated_observation_store, execution = "injected-parent")
    target = root / "acct-a"
    original_open = op.os.open
    calls = 0

    def replace_before_second_open(path, flags, *args, **kwargs):
        nonlocal calls
        if path == "acct-a":
            calls += 1
            if calls == 2:
                target.rename(root / "acct-a-retained")
                target.symlink_to(outside, target_is_directory = True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(op.os, "open", replace_before_second_open)
    with pytest.raises(op.ObservationPathError):
        op.publish_observation_candidate(candidate, storage_root = root)
    assert calls == 2
    assert not (outside / "blobs").exists()


def test_changed_inode_hardlink_and_fifo_are_rejected(isolated_observation_store):
    root = isolated_observation_store
    published, binding = _published_binding(root, execution = "identity")
    blob = root / "observation-store" / "acct-a" / binding.blob_relpath
    original = blob.read_bytes()
    blob.unlink()
    blob.write_bytes(original)
    assert op.recall_observation(binding, storage_root = root / "observation-store").status == "corrupt"

    blob.unlink()
    outside = root / "outside-blob"
    outside.write_bytes(original)
    blob.hardlink_to(outside)
    assert op.recall_observation(binding, storage_root = root / "observation-store").status == "corrupt"

    blob.unlink()
    os.mkfifo(blob, 0o600)
    try:
        assert op.recall_observation(binding, storage_root = root / "observation-store").status == "corrupt"
    finally:
        blob.unlink()


def test_key_replacement_is_missing_key_not_recreation(isolated_observation_store):
    root = isolated_observation_store
    _published, binding = _published_binding(root, execution = "key")
    conn = storage.get_connection()
    try:
        conn.execute(
            "DELETE FROM app_secrets WHERE key = ?", (storage._OBSERVATION_MASTER_KEY_DB_KEY,)
        )
        conn.commit()
    finally:
        conn.close()
    storage._observation_master_key_cache = None
    assert op.recall_observation(binding, storage_root = root / "observation-store").status == "missing_key"


def test_candidate_classification_preserves_fallback_and_rejects_ineligible_inputs(isolated_observation_store):
    root = isolated_observation_store
    for kwargs, reason in (
        ({"observation_pack_version": None}, "not_admitted"),
        ({"complete": False}, "incomplete"),
        ({"process_exit_code": 1}, "nonzero_exit"),
        ({"policy_eligible": False}, "policy_excluded"),
        ({"secret_bearing": True}, "policy_excluded"),
        ({"canonical_text": "short"}, "size_excluded"),
    ):
        candidate = _candidate(root, execution = reason, **kwargs)
        assert candidate.reason == reason
        assert candidate.fallback_result == "ordinary bounded fallback"
        assert candidate.selected_result is None
        assert candidate.available is False


def test_unpaired_surrogates_are_rejected_at_candidate_admission(isolated_observation_store):
    with pytest.raises(ValueError):
        op.canonicalize_observation_text("\ud800" * 6000)
    candidate = _candidate(
        isolated_observation_store,
        execution = "surrogate",
        canonical_text = "\ud800" * 6000,
    )
    assert candidate.reason == "invalid_utf8"
    assert candidate.available is False
    assert candidate.selected_result is None
