"""No-live-execution tests for the Phase 1 source-runtime benchmark runner."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark-helix-v3-source-runtime.py"
_SPEC = importlib.util.spec_from_file_location("helix_v3_source_runtime", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _identity(tree: str = "a" * 64) -> dict:
    return {
        "schema_version": "path-nul-sha256-newline-v2-artifacts-excluded",
        "repo": "/repo",
        "git_head": "head",
        "branch": "main",
        "dirty": True,
        "status_count": 1,
        "file_count": 2,
        "total_bytes": 10,
        "tree_sha256": tree,
    }


def _identity_runner(value: dict | None = None):
    identity = value or _identity()

    def run(command, **_kwargs):
        assert command[1].endswith("scripts/helix-v3-source-identity.py")
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity))

    return run


def _runtime_template(tmp_path: Path, *, password: str = "source-password-123") -> Path:
    template = tmp_path / "runtime-template"
    python = template / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    python.chmod(0o755)
    auth = template / ".unsloth" / "studio" / "auth"
    auth.mkdir(parents=True)
    (auth / ".bootstrap_password").write_text(password + "\n", encoding="utf-8")
    return template


def _config(tmp_path: Path, template: Path, *, execute: bool = False) -> _RUNNER.RuntimeConfig:
    model = tmp_path / "mlx-model"
    model.mkdir(exist_ok=True)
    model_config = tmp_path / "mlx-model.json"
    model_config.write_text(json.dumps({"backend": "mlx", "model_identity": {"repository": "mlx-model", "revision": "rev-1"}}), encoding="utf-8")
    return _RUNNER.RuntimeConfig(
        runtime_home_template=template,
        run_dir=tmp_path / "run-home",
        output_dir=tmp_path / "artifacts",
        repo_root=Path(__file__).resolve().parents[3],
        model_path=model,
        model_config_path=model_config,
        execute=execute,
    )


def _event(invocation_id: str = "invocation-1", *, model_identity: str = "local") -> dict:
    return {
        "invocation_id": invocation_id,
        "scope": "foreground",
        "provider_tier": "local",
        "model_identity": model_identity,
        "provider_identity": "local:mlx",
        "outcome": "completed",
        "semantic_contribution": "final_answer",
        "wasted": False,
        "wasted_reason": None,
        "retry_of": None,
        "escalation_from_tier": None,
    }


def _trace_payload(objective: dict[str, str], *, status: str = "available", binding: dict | None = None, events: list | None = None, model_identity: str = "local") -> dict:
    raw_events = events if events is not None else [_event(model_identity=model_identity)]
    aggregation = _RUNNER.aggregate_trace_events(raw_events)
    return {
        "schema_version": _RUNNER.TRACE_EVENT_SCHEMA,
        "status": status,
        "aggregation": {"admissible": status == "available", "reason": None if status == "available" else "missing_trace_envelope"},
        "binding": binding or {
            "run_id": objective["run_id"],
            "thread_id": objective["thread_id"],
            "user_message_id": objective["user_message_id"],
            "owner_subject": "unsloth",
            "account_id": "owner",
        },
        "trace": {
            "schema_version": _RUNNER.TRACE_SCHEMA,
            "status": status,
            "invocation_events": raw_events if status == "available" else [],
            "invocation_counts": aggregation if status == "available" else None,
            "all_handles_settled": True,
            "error": None if status == "available" else "missing_trace_envelope",
        } if status == "available" else None,
        "error": None if status == "available" else "missing_trace_envelope",
    }


def _create_db(
    home: Path,
    objective: dict[str, str],
    *,
    trace_payload: dict | None = None,
    trace_rows: int = 1,
    finalization_status: str = "none",
    assistant_content: object | None = None,
    model_name: str = "local",
) -> None:
    database = home / ".unsloth" / "studio" / "studio.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "model": model_name,
        "thread_id": objective["thread_id"],
        "generation_run_id": objective["run_id"],
        "messages": [{"role": "user", "content": "prompt"}],
    }
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE chat_threads (id TEXT PRIMARY KEY);
        CREATE TABLE chat_messages (
            id TEXT PRIMARY KEY, thread_id TEXT, parent_id TEXT, role TEXT,
            content_json TEXT, metadata_json TEXT
        );
        CREATE TABLE chat_generation_runs (
            id TEXT PRIMARY KEY, owner_subject TEXT, thread_id TEXT,
            user_message_id TEXT, assistant_message_id TEXT, status TEXT,
            finish_reason TEXT, request_json TEXT, request_hash TEXT,
            last_event_seq INTEGER, finalization_status TEXT NOT NULL DEFAULT 'none'
        );
        CREATE TABLE chat_generation_events (
            run_id TEXT, seq INTEGER, event_type TEXT, payload_json TEXT,
            created_at INTEGER
        );
        """
    )
    connection.execute("INSERT INTO chat_threads VALUES (?)", (objective["thread_id"],))
    connection.execute(
        "INSERT INTO chat_messages VALUES (?, ?, NULL, 'user', ?, NULL)",
        (objective["user_message_id"], objective["thread_id"], json.dumps([])),
    )
    connection.execute(
        "INSERT INTO chat_messages VALUES (?, ?, ?, 'assistant', ?, ?)",
        (
            objective["assistant_message_id"],
            objective["thread_id"],
            objective["user_message_id"],
            json.dumps(
                [{"type": "text", "text": _RUNNER.EXPECTED_ANSWER}]
                if assistant_content is None else assistant_content
            ),
            json.dumps(
                {
                    "generationRunId": objective["run_id"],
                    "generationSeq": 4,
                    "generationStatus": "completed",
                    "generationSettled": True,
                    "serverManaged": True,
                }
            ),
        ),
    )
    connection.execute(
        "INSERT INTO chat_generation_runs VALUES (?, 'unsloth', ?, ?, ?, 'completed', 'stop', ?, 'hash', 4, ?)",
        (
            objective["run_id"],
            objective["thread_id"],
            objective["user_message_id"],
            objective["assistant_message_id"],
            json.dumps(request),
            finalization_status,
        ),
    )
    connection.execute(
        "INSERT INTO chat_generation_events VALUES (?, 1, 'run.created', '{}', 1)",
        (objective["run_id"],),
    )
    connection.execute(
        "INSERT INTO chat_generation_events VALUES (?, 2, 'generation.chunk', ?, 2)",
        (
            objective["run_id"],
            json.dumps({"choices": [{"delta": {"content": _RUNNER.EXPECTED_ANSWER}}]}),
        ),
    )
    payload = trace_payload or _trace_payload(objective, model_identity=model_name)
    for _ in range(trace_rows):
        connection.execute(
            "INSERT INTO chat_generation_events VALUES (?, 3, 'optimization.trace', ?, 3)",
            (objective["run_id"], json.dumps(payload)),
        )
    connection.execute(
        "INSERT INTO chat_generation_events VALUES (?, 4, 'run.completed', ?, 4)",
        (objective["run_id"], json.dumps({"status": "completed", "finishReason": "stop"})),
    )
    connection.commit()
    connection.close()


class _FakeProcess:
    pid = 12345

    def __init__(self):
        self.stdin = io.StringIO()
        self.exited = False
        self.terminated = False

    def poll(self):
        return 0 if self.exited else None

    def wait(self, **_kwargs):
        self.exited = True
        return 0

    def terminate(self):
        self.terminated = True
        self.exited = True

    def kill(self):
        self.terminated = True
        self.exited = True


def _ps_runner(process: _FakeProcess):
    root = "12345 1 12345 10 Sun Sep 20 10:00:00 2026 source-run.py --owned\n"

    def run(command, **_kwargs):
        if command[0] == "/bin/ps":
            if process.exited:
                # A working global ps always has at least an unrelated process;
                # empty output is treated by the runner as an observation fault.
                return SimpleNamespace(returncode=0, stdout="1 0 1 1 Sun Sep 20 10:00:00 2026 launchd\n")
            return SimpleNamespace(returncode=0, stdout=root)
        if command[0] == "/usr/sbin/lsof":
            if process.exited:
                return SimpleNamespace(returncode=1, stdout="")
            return SimpleNamespace(returncode=0, stdout="12345\n")
        raise AssertionError(command)

    return run


class _Transport:
    def __init__(self, objective: dict[str, str], *, private: bool = False, database_path: Path | None = None):
        self.objective = objective
        self.private = private
        self.database_path = database_path
        self.requests: list[tuple[str, str, object]] = []
        self.assistant_message = {
            "id": objective["assistant_message_id"],
            "threadId": objective["thread_id"],
            "parentId": objective["user_message_id"],
            "role": "assistant",
            "content": [],
            "attachments": None,
            "metadata": {
                "generationRunId": objective["run_id"],
                "generationSeq": 0,
                "generationStatus": "completed",
                "serverManaged": True,
            },
            "createdAt": 1,
        }

    def request_json(self, method, url, *, headers, payload=None, timeout):
        self.requests.append((method, url, payload))
        if url.endswith("/api/health"):
            return {"helix_backend_contract": _RUNNER.EXPECTED_BACKEND_CONTRACT, "helix_backend_tree_sha256": None, "helix_backend_verified": False}
        if url.endswith("/api/auth/login"):
            return {"access_token": "access-token-secret", "account_id": "owner"}
        if url.endswith("/api/inference/load"):
            return {
                "status": "loaded",
                "model": payload["model_path"],
                "is_mlx": True,
                "is_local_model": True,
            }
        if url.endswith("/api/inference/unload"):
            return {"status": "unloaded", "model": payload["model_path"]}
        if url.endswith("/api/chat/threads"):
            return {"id": self.objective["thread_id"]}
        if "/messages/" in url:
            if url.endswith("/" + self.objective["assistant_message_id"]):
                if method == "GET":
                    return dict(self.assistant_message)
                assert payload["content"] == [{"type": "text", "text": _RUNNER.EXPECTED_ANSWER}]
                self.assistant_message = dict(payload)
                if self.database_path is not None:
                    connection = sqlite3.connect(self.database_path)
                    connection.execute(
                        "UPDATE chat_messages SET content_json=?, metadata_json=? WHERE id=?",
                        (
                            json.dumps(payload["content"]),
                            json.dumps(payload["metadata"]),
                            self.objective["assistant_message_id"],
                        ),
                    )
                    connection.commit()
                    connection.close()
                return dict(self.assistant_message)
            return {"id": self.objective["user_message_id"], "threadId": self.objective["thread_id"], "role": "user"}
        if url.endswith("/api/inference/chat-runs"):
            return {"id": self.objective["run_id"], "threadId": self.objective["thread_id"], "userMessageId": self.objective["user_message_id"], "assistantMessageId": self.objective["assistant_message_id"]}
        if url.endswith("/api/shutdown"):
            return {"ok": True}
        raise AssertionError(url)

    def open_stream(self, method, url, *, headers, payload, timeout):
        run = {"id": self.objective["run_id"], "status": "completed", "finishReason": "stop", "lastEventSeq": 4, "threadId": self.objective["thread_id"], "userMessageId": self.objective["user_message_id"], "assistantMessageId": self.objective["assistant_message_id"]}
        chunk = {
            "seq": 2,
            "type": "chunk",
            "payload": {
                "choices": [{"delta": {"content": _RUNNER.EXPECTED_ANSWER}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                "timings": {"prompt_ms": 4.5, "predicted_ms": 8.5, "predicted_per_second": 11.0},
            },
            "createdAt": 2,
        }
        terminal = {
            "seq": 4,
            "type": "run.completed",
            "payload": {"status": "completed", "finishReason": "stop"},
            "createdAt": 4,
            "run": run,
        }
        # The real route advances over the private optimization.trace row
        # without emitting it, so the public replay jumps from seq 2 to seq 4.
        body = (
            "id: 2\nevent: chunk\ndata: " + json.dumps(chunk) + "\n\n"
            "id: 4\nevent: run.completed\ndata: " + json.dumps(terminal) + "\n\n"
        )
        return io.BytesIO(body.encode())


def test_dry_run_validates_contracts_without_launch_or_artifact(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path)
    config = _config(tmp_path, template)
    launched = []
    monkeypatch.setattr(_RUNNER, "_launch_source", lambda *args, **kwargs: launched.append(True))
    result = _RUNNER.run_benchmark(config, source_identity_runner=_identity_runner())
    assert result["status"] == "dry_run"
    assert result["would_launch"] is False
    assert not launched
    assert not (tmp_path / "artifacts").exists()


def test_apfs_clone_does_not_mutate_source_and_uses_explicit_run_dir(tmp_path):
    template = _runtime_template(tmp_path)
    original = (template / ".unsloth" / "studio" / "auth" / ".bootstrap_password").read_bytes()

    def copy_runner(command, **_kwargs):
        assert command[:2] == ["/bin/cp", "-cR"]
        shutil.copytree(command[2], command[3], symlinks=True)
        return SimpleNamespace(returncode=0)

    clone = _RUNNER.clone_runtime_home(template, tmp_path / "explicit-run", copy_runner=copy_runner, system="Darwin")
    assert clone.run_dir == (tmp_path / "explicit-run").resolve()
    assert (template / ".unsloth" / "studio" / "auth" / ".bootstrap_password").read_bytes() == original
    assert clone.run_dir.exists()
    shutil.rmtree(clone.run_dir)


def test_failed_apfs_clone_removes_partial_run_dir_without_touching_source(tmp_path):
    template = _runtime_template(tmp_path)
    run_dir = tmp_path / "partial-run"
    original = (template / ".unsloth" / "studio" / "auth" / ".bootstrap_password").read_bytes()

    def failed_copy(command, **_kwargs):
        shutil.copytree(command[2], command[3], symlinks=True)
        return SimpleNamespace(returncode=1)

    with pytest.raises(_RUNNER.ContractError, match="clone failed"):
        _RUNNER.clone_runtime_home(template, run_dir, copy_runner=failed_copy, system="Darwin")
    assert not run_dir.exists()
    assert (template / ".unsloth" / "studio" / "auth" / ".bootstrap_password").read_bytes() == original


def test_launch_keeps_password_out_of_argv_environment_and_artifact(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path, password="never-in-argv-987")
    captured = {}

    def popen(argv, **kwargs):
        captured.update(argv=argv, kwargs=kwargs)
        return _FakeProcess()

    clone = _RUNNER.clone_runtime_home(
        template,
        tmp_path / "explicit-run",
        copy_runner=lambda command, **_kwargs: (shutil.copytree(command[2], command[3], symlinks=True), SimpleNamespace(returncode=0))[1],
        system="Darwin",
    )
    process = _RUNNER._launch_source(clone, _config(tmp_path, template, execute=True), 43123, password="never-in-argv-987", popen_factory=popen)
    assert "never-in-argv-987" not in " ".join(captured["argv"])
    assert "never-in-argv-987" not in captured["kwargs"]["env"].values()
    assert captured["argv"][-2:] == ["--password", "-"]
    assert process.stdin.closed
    output = _RUNNER._sanitize({"password": "never-in-argv-987", "detail": "never-in-argv-987"}, ["never-in-argv-987"])
    assert "never-in-argv-987" not in json.dumps(output)
    shutil.rmtree(clone.run_dir)


def test_default_launch_password_is_distinct_and_secret_safe(monkeypatch):
    monkeypatch.setattr(_RUNNER.secrets, "token_urlsafe", lambda _size: "new-clone-secret")
    generated = _RUNNER.ephemeral_launch_password("frozen-bootstrap-secret")
    assert generated == "helix-v3-new-clone-secret"
    assert generated != "frozen-bootstrap-secret"
    assert not any(char.isspace() for char in generated)


def test_frontend_sse_private_trace_is_suppressed_from_public_replay():
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    transport = _Transport(objective, private=True)
    result = _RUNNER.consume_frontend_sse(transport, "http://127.0.0.1:1", {}, "r", timeout=1)
    assert result["private_trace_suppressed"] is True
    assert [event["cursor"] for event in result["events"]] == [2, 4]
    assert all(event["event"] != _RUNNER.TRACE_EVENT_TYPE for event in result["events"])


def test_frontend_sse_rejects_private_trace_if_server_leaks_it():
    envelope = {
        "seq": 3,
        "type": _RUNNER.TRACE_EVENT_TYPE,
        "payload": {"_helix_optimization_trace_v1": {"status": "available"}},
        "createdAt": 3,
        "run": {"id": "r", "status": "running"},
    }
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="leaked"):
        _RUNNER.consume_frontend_sse(
            _StreamOnlyTransport(
                ("id: 3\nevent: " + _RUNNER.TRACE_EVENT_TYPE + "\ndata: " + json.dumps(envelope) + "\n\n").encode()
            ),
            "http://127.0.0.1:1",
            {},
            "r",
            timeout=1,
        )


def test_source_health_requires_explicit_unverified_manifest_missing_mode():
    valid = {
        "helix_backend_contract": _RUNNER.EXPECTED_BACKEND_CONTRACT,
        "helix_backend_tree_sha256": None,
        "helix_backend_verified": False,
    }
    assert _RUNNER._validate_health_identity(valid)["verified"] is False
    assert _RUNNER._validate_health_identity(valid)["reason"] == "not_exposed_source_manifest_absence_preflight"
    assert _RUNNER._validate_health_identity({**valid, "reason": "runtime_manifest_missing"})["reason"] == "runtime_manifest_missing"
    for invalid in (
        {**valid, "helix_backend_verified": True, "helix_backend_tree_sha256": "a" * 64},
        {**valid, "reason": "runtime_manifest_unreadable"},
        {**valid, "helix_backend_tree_sha256": "b" * 64},
    ):
        with pytest.raises(_RUNNER.IdentityMismatchError):
            _RUNNER._validate_health_identity(invalid)


def test_source_run_rejects_packaged_overlay_manifest(tmp_path):
    repo = tmp_path / "repo"
    backend = repo / "studio" / "backend"
    backend.mkdir(parents=True)
    assert _RUNNER._validate_source_manifest_absent(repo) == backend / ".helix_backend_manifest.json"
    manifest = backend / ".helix_backend_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(_RUNNER.IdentityMismatchError, match="packaged overlay manifest"):
        _RUNNER._validate_source_manifest_absent(repo)


def test_nested_model_identity_and_load_payload_are_authoritative(tmp_path):
    from models.inference import ChatCompletionRequest

    config = _config(tmp_path, _runtime_template(tmp_path))
    identity = _RUNNER.model_identity(config)
    assert identity["repository"] == "mlx-model"
    assert identity["revision"] == "rev-1"
    spec = _RUNNER.validate_offline_model_spec(config.model_path, config.model_config_path)
    payload = _RUNNER.build_load_payload(spec, seed=7)
    assert payload["model_path"] == str(config.model_path.resolve())
    assert payload["load_request_id"] == "helix-v3-7"
    request = _RUNNER.build_request_payload(config, "run", "thread")
    assert request["enable_thinking"] is False
    assert request["reasoning_effort"] == "none"
    assert "generation_run_id" not in request
    assert not (set(request) - set(ChatCompletionRequest.model_fields))
    assert _RUNNER.public_model_id(str(config.model_path.resolve())) == "mlx-model"
    assert _RUNNER.derive_public_model_identity(
        config.model_path,
        spec["config"],
    ) == "mlx-model"
    assert _RUNNER.select_model_request_name(
        {
            "status": "loaded",
            "model": str(config.model_path.resolve()),
            "is_mlx": True,
            "is_local_model": True,
        },
        config.model_path,
        allowed_names={"local"},
    ) == str(config.model_path.resolve())
    for wrong in (
        {"status": "loaded", "model": "local", "is_mlx": True, "is_local_model": True},
        {"status": "loaded", "model": str(config.model_path.resolve()), "is_mlx": False, "is_local_model": True},
        {"status": "loaded", "model": str(config.model_path.resolve()), "is_mlx": True, "is_local_model": False},
        {"status": "loaded", "model": "", "is_mlx": True, "is_local_model": True},
    ):
        with pytest.raises(_RUNNER.ContractError, match="identity|contract|local MLX"):
            _RUNNER.select_model_request_name(wrong, config.model_path, allowed_names={"local"})
    with pytest.raises(_RUNNER.IdentityMismatchError, match="path"):
        _RUNNER.select_model_request_name(
            {"status": "loaded", "model": str(config.model_path.resolve()), "model_path": str((tmp_path / "other-model").resolve()), "is_mlx": True, "is_local_model": True},
            config.model_path,
            allowed_names={"local"},
        )


def test_public_model_identity_matches_backend_hf_cache_contract(tmp_path):
    snapshot = (
        tmp_path
        / "hub"
        / "models--mlx-community--Qwen3.8-27B-4bit"
        / "snapshots"
        / ("a" * 40)
    )
    snapshot.mkdir(parents=True)
    config = {
        "model_identity": {
            "repository": "mlx-community/Qwen3.8-27B-4bit",
        }
    }
    assert _RUNNER.public_model_id(str(snapshot)) == "mlx-community/Qwen3.8-27B-4bit"
    assert _RUNNER.derive_public_model_identity(snapshot, config) == "mlx-community/Qwen3.8-27B-4bit"
    with pytest.raises(_RUNNER.IdentityMismatchError, match="configured model repository"):
        _RUNNER.derive_public_model_identity(
            snapshot,
            {"model_identity": {"repository": "other/model"}},
        )


def test_unload_identity_must_match_admitted_load_model(tmp_path):
    model_path = tmp_path / "mlx-model"
    model_path.mkdir()
    valid = {
        "status": "unloaded",
        "model": str(model_path.resolve()),
    }
    assert _RUNNER.validate_unload_response(
        valid,
        admitted_model_name=str(model_path.resolve()),
        model_path=model_path,
    ) == str(model_path.resolve())
    for wrong in (
        {**valid, "model": "other"},
        {**valid, "model": ""},
        {"status": "unloaded"},
        {**valid, "model_path": str((tmp_path / "other-model").resolve())},
    ):
        with pytest.raises(_RUNNER.ContractError):
            _RUNNER.validate_unload_response(
                wrong,
                admitted_model_name=str(model_path.resolve()),
                model_path=model_path,
            )


def test_child_environment_does_not_disable_dns_pinning(tmp_path):
    env = _RUNNER.build_environment(tmp_path / "clone", Path("/repo"), base={"PATH": "/bin"})
    assert "UNSLOTH_STUDIO_DISABLE_DNS_PINNING" not in env
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"


class _StreamOnlyTransport:
    def __init__(self, body: bytes):
        self.body = body

    def open_stream(self, *_args, **_kwargs):
        return io.BytesIO(self.body)


def test_frontend_sse_bounds_cursor_and_terminal_proof():
    terminal = {
        "seq": 2,
        "type": "run.completed",
        "payload": {"status": "completed", "finishReason": "stop"},
        "createdAt": 2,
        "run": {"id": "r", "status": "completed", "finishReason": "stop", "lastEventSeq": 2},
    }
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="monotonic"):
        _RUNNER.consume_frontend_sse(
            _StreamOnlyTransport(
                ("id: 1\nevent: run.started\ndata: " + json.dumps({"seq": 1, "type": "run.started", "payload": {"status": "running"}, "createdAt": 1, "run": {"id": "r", "status": "running"}}) + "\n\n"
                 "id: 1\nevent: run.completed\ndata: " + json.dumps({**terminal, "seq": 1}) + "\n\n").encode()
            ),
            "http://127.0.0.1:1", {}, "r", timeout=1,
        )
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="lastEventSeq"):
        bad_terminal = {**terminal, "run": {**terminal["run"], "lastEventSeq": 3}}
        _RUNNER.consume_frontend_sse(
            _StreamOnlyTransport(("id: 2\nevent: run.completed\ndata: " + json.dumps(bad_terminal) + "\n\n").encode()),
            "http://127.0.0.1:1", {}, "r", timeout=1,
        )
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="line exceeded"):
        _RUNNER.consume_frontend_sse(
            _StreamOnlyTransport(("data: " + ("x" * (_RUNNER.MAX_SSE_LINE_BYTES + 1)) + "\n\n").encode()),
            "http://127.0.0.1:1", {}, "r", timeout=1,
        )


@pytest.mark.parametrize(
    ("wire_id", "wire_event", "envelope", "message"),
    [
        (
            "3",
            "chunk",
            {"seq": 2, "type": "chunk", "payload": {}, "createdAt": 2},
            "wire cursor",
        ),
        (
            "2",
            "run.completed",
            {"seq": 2, "type": "chunk", "payload": {}, "createdAt": 2, "run": {}},
            "wire event",
        ),
        (
            "2",
            "chunk",
            {"seq": 2, "type": "chunk", "payload": [], "createdAt": 2},
            "inner payload",
        ),
    ],
)
def test_frontend_sse_requires_canonical_wire_envelope(wire_id, wire_event, envelope, message):
    body = f"id: {wire_id}\nevent: {wire_event}\ndata: {json.dumps(envelope)}\n\n"
    with pytest.raises(_RUNNER.ProtocolMismatchError, match=message):
        _RUNNER.consume_frontend_sse(
            _StreamOnlyTransport(body.encode()),
            "http://127.0.0.1:1",
            {},
            "r",
            timeout=1,
        )


def test_visible_answer_reconstruction_requires_exact_text_and_stop():
    valid = {
        "events": [
            {
                "payload": {
                    "choices": [
                        {"delta": {"content": _RUNNER.EXPECTED_ANSWER}}
                    ]
                }
            }
        ],
        "terminal_run": {"finishReason": "stop"},
    }
    assert _RUNNER.reconstruct_visible_answer(valid) == _RUNNER.EXPECTED_ANSWER

    for invalid, message in (
        ({**valid, "events": []}, "exact expected answer"),
        ({**valid, "terminal_run": {"finishReason": "length"}}, "finish with stop"),
        (
            {
                **valid,
                "events": [
                    {"payload": {"choices": [{"delta": {"content": "wrong"}}]}}
                ],
            },
            "exact expected answer",
        ),
        (
            {
                **valid,
                "events": [
                    {"payload": {"choices": [{"delta": {"tool_calls": [{}]}}]}}
                ],
            },
            "tool proposal",
        ),
    ):
        with pytest.raises(_RUNNER.ProtocolMismatchError, match=message):
            _RUNNER.reconstruct_visible_answer(invalid)


def test_persist_assistant_answer_validates_decoded_api_content_and_placeholder():
    objective = {
        "run_id": "r",
        "thread_id": "t",
        "user_message_id": "u",
        "assistant_message_id": "a",
    }
    transport = _Transport(objective)
    result = _RUNNER.persist_assistant_answer(
        transport,
        "http://127.0.0.1:1",
        {},
        objective=objective,
        sse={"last_event_seq": 4},
        answer=_RUNNER.EXPECTED_ANSWER,
        timeout=1,
    )
    assert result["status"] == "pass"
    assert result["generation_seq"] == 4
    assert transport.assistant_message["content"] == [
        {"type": "text", "text": _RUNNER.EXPECTED_ANSWER}
    ]

    occupied = _Transport(objective)
    occupied.assistant_message["content"] = [{"type": "text", "text": "existing"}]
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="already contained"):
        _RUNNER.persist_assistant_answer(
            occupied,
            "http://127.0.0.1:1",
            {},
            objective=objective,
            sse={"last_event_seq": 4},
            answer=_RUNNER.EXPECTED_ANSWER,
            timeout=1,
        )


def test_wrong_account_binding_fails_closed():
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    payload = _trace_payload(objective, binding={"run_id": "r", "thread_id": "t", "user_message_id": "u", "owner_subject": "other", "account_id": "other"})
    with pytest.raises(_RUNNER.AccountBindingError):
        _RUNNER.validate_trace_payload(payload, objective=objective, provider_tier="local", model_name="local", provider_identity="local:mlx")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update(status="unavailable", trace=None, error="missing_trace_envelope"),
        lambda payload: payload["aggregation"].update(admissible=False, reason="not_admissible"),
    ],
)
def test_missing_or_inadmissible_trace_fails_closed(mutator):
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    payload = _trace_payload(objective)
    mutator(payload)
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.validate_trace_payload(payload, objective=objective, provider_tier="local", model_name="local", provider_identity="local:mlx")


def test_duplicate_trace_rows_fail_closed(tmp_path):
    template = _runtime_template(tmp_path)
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    _create_db(template, objective, trace_rows=2)
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="missing or duplicated"):
        _RUNNER.read_durable_evidence(template / ".unsloth" / "studio" / "studio.db", objective=objective, model_name="local", provider_tier="local", provider_identity="local:mlx")


@pytest.mark.parametrize(
    "assistant_content",
    [
        [{"type": "text", "text": "WRONG"}],
        [],
    ],
)
def test_assistant_content_must_prove_exact_expected_answer(tmp_path, assistant_content):
    template = _runtime_template(tmp_path)
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    _create_db(template, objective, assistant_content=assistant_content)
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="exact expected answer|empty"):
        _RUNNER.read_durable_evidence(
            template / ".unsloth" / "studio" / "studio.db",
            objective=objective,
            model_name="local",
            provider_tier="local",
            provider_identity="local:mlx",
        )


def test_trace_must_precede_terminal_and_finalization_must_be_none(tmp_path):
    template = _runtime_template(tmp_path)
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    _create_db(template, objective, finalization_status="pending")
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="finalization"):
        _RUNNER.read_durable_evidence(template / ".unsloth" / "studio" / "studio.db", objective=objective, model_name="local", provider_tier="local", provider_identity="local:mlx")

    template2 = _runtime_template(tmp_path / "bad-sequence")
    _create_db(template2, objective)
    database = template2 / ".unsloth" / "studio" / "studio.db"
    connection = sqlite3.connect(database)
    connection.execute("UPDATE chat_generation_events SET seq=4 WHERE event_type='optimization.trace'")
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError, match="immediately before"):
        _RUNNER.read_durable_evidence(database, objective=objective, model_name="local", provider_tier="local", provider_identity="local:mlx")


def test_sqlite_boundary_is_standalone_read_only_uri(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path)
    objective = {"run_id": "r", "thread_id": "t", "user_message_id": "u", "assistant_message_id": "a"}
    _create_db(template, objective)
    real_connect = _RUNNER.sqlite3.connect
    observed = {}

    def recording_connect(database, *args, **kwargs):
        observed.update(database=database, args=args, kwargs=kwargs)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(_RUNNER.sqlite3, "connect", recording_connect)
    evidence = _RUNNER.read_durable_evidence(
        template / ".unsloth" / "studio" / "studio.db",
        objective=objective,
        model_name="local",
        provider_tier="local",
        provider_identity="local:mlx",
    )
    assert "mode=ro" in observed["database"]
    assert observed["kwargs"]["uri"] is True
    assert evidence["sqlite"] == {"mode": "ro", "query_only": True}
    assert evidence["task_success"]["status"] == "pass"
    assert evidence["task_success"]["objective_correctness"] == "unverified"
    assert evidence["task_success"]["expected_answer_sha256"] == evidence["task_success"]["observed_answer_sha256"]
    assert evidence["task_success"]["provenance"] == "client_observed_sse+authenticated_message_put+sqlite_ro"
    assert evidence["trace_availability"]["status"] == "available"


def test_source_identity_drift_fails_closed():
    with pytest.raises(_RUNNER.IdentityMismatchError):
        _RUNNER.verify_source_identity(_identity(), _identity("c" * 64))


def test_group_reuse_is_not_signalled():
    captured = _RUNNER._parse_ps_snapshot("12345 1 12345 10 Sun Sep 20 10:00:00 2026 source --owned\n", session_id_getter=lambda pid: pid)
    reused = "12345 1 12346 10 Sun Sep 20 11:00:00 2026 unrelated --reused\n"

    def command_runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=reused)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    result = _RUNNER.verify_cleanup(12345, {12345}, 43123, command_runner=command_runner, owned_identities=captured, owned_pgid=12345, session_id_getter=lambda pid: pid)
    assert result["status"] == "fail"
    assert result["identity_mismatches"] == [12345]


def test_source_process_identity_requires_exact_module_origin(tmp_path):
    clone = _RUNNER.CloneHandle(
        source=tmp_path,
        run_dir=tmp_path / "run",
        studio_home=tmp_path / "run" / ".unsloth" / "studio",
        venv_python=tmp_path / "run" / "venv" / "bin" / "python",
        source_manifest="a" * 64,
    )
    config = _config(tmp_path, _runtime_template(tmp_path), execute=True)
    command = _RUNNER._source_launch_command(clone, config, 43123)
    identity = _RUNNER._parse_ps_snapshot(
        "12345 1 12345 10 Sun Sep 20 10:00:00 2026 " + " ".join(command) + "\n",
        session_id_getter=lambda pid: pid,
    )[12345]
    _RUNNER._validate_source_process_identity(identity, command, config.repo_root)
    identity["command"] = identity["command"].replace("run.py", "other.py")
    with pytest.raises(_RUNNER.IdentityMismatchError, match="command/module origin"):
        _RUNNER._validate_source_process_identity(identity, command, config.repo_root)


def _capture_ps_rows(rows: list[tuple[int, int, int]], *, missing_session_pids: set[int] | None = None):
    missing_session_pids = missing_session_pids or set()
    output = "".join(
        f"{pid} {ppid} {pgid} 10 Sun Sep 20 10:00:00 2026 process-{pid}\n"
        for pid, ppid, pgid in rows
    )

    def command_runner(command, **_kwargs):
        assert command[0] == "/bin/ps"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    def session_id_getter(pid: int) -> int:
        if pid in missing_session_pids:
            raise ProcessLookupError(pid)
        return pid

    return command_runner, session_id_getter


def test_capture_protects_raw_intermediate_descendants_before_getsid():
    command_runner, session_id_getter = _capture_ps_rows(
        [(100, 1, 100), (101, 100, 101), (102, 101, 101)],
        missing_session_pids={101},
    )
    assert _RUNNER.capture_process_identities(100, command_runner=command_runner, session_id_getter=session_id_getter) == {}


def test_capture_protects_same_and_reparented_owned_pgid_rows():
    command_runner, session_id_getter = _capture_ps_rows(
        [(100, 1, 100), (102, 1, 100)],
        missing_session_pids={102},
    )
    assert _RUNNER.capture_process_identities(100, command_runner=command_runner, session_id_getter=session_id_getter) == {}


def test_capture_allows_unrelated_getsid_churn_only():
    command_runner, session_id_getter = _capture_ps_rows(
        [(100, 1, 100), (200, 1, 200)],
        missing_session_pids={200},
    )
    captured = _RUNNER.capture_process_identities(100, command_runner=command_runner, session_id_getter=session_id_getter)
    assert set(captured) == {100}


def test_group_signal_is_skipped_when_identity_is_reused():
    process = _FakeProcess()
    captured = _RUNNER._parse_ps_snapshot(
        "12345 1 12345 10 Sun Sep 20 10:00:00 2026 source --owned\n",
        session_id_getter=lambda pid: pid,
    )
    signalled = []
    result = _RUNNER.terminate_owned_processes(
        process,
        {12345},
        owned_identities=captured,
        identity_conflicts={12345},
        owned_pgid=12345,
        session_id_getter=lambda pid: pid,
        group_signal_sender=lambda *args: signalled.append(args),
    )
    assert result["status"] == "fail"
    assert result["group_signal_attempted"] is False
    assert not signalled


def test_empty_global_process_observation_fails_closed():
    captured = _RUNNER._parse_ps_snapshot(
        "12345 1 12345 10 Sun Sep 20 10:00:00 2026 source --owned\n",
        session_id_getter=lambda pid: pid,
    )

    def command_runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout="")
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        raise AssertionError(command)

    result = _RUNNER.verify_cleanup(
        12345,
        {12345},
        43123,
        command_runner=command_runner,
        owned_identities=captured,
        owned_pgid=12345,
        session_id_getter=lambda pid: pid,
    )
    assert result["status"] == "unavailable"
    assert result["provenance"] == "ps+lsof_unavailable"


def test_full_mocked_run_reads_trace_offline_and_removes_clone(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path, password="in-memory-only-555")
    objective = {"run_id": "run-fixed", "thread_id": "thread-fixed", "user_message_id": "user-fixed", "assistant_message_id": "assistant-fixed"}
    config = _config(tmp_path, template, execute=True)
    authoritative_model = str(config.model_path.resolve())
    public_model = _RUNNER.public_model_id(authoritative_model)
    _create_db(template, objective, assistant_content=[], model_name=public_model)
    process = _FakeProcess()
    transport = _Transport(
        objective,
        database_path=tmp_path / "run-home" / ".unsloth" / "studio" / "studio.db",
    )
    monkeypatch.setattr(_RUNNER, "create_objective", lambda *args, **kwargs: {**objective, "request_payload": _RUNNER.build_request_payload(args[3], objective["run_id"], objective["thread_id"])})
    monkeypatch.setattr(_RUNNER, "clone_runtime_home", lambda *args, **kwargs: _clone_for_test(template, tmp_path / "run-home"))
    monkeypatch.setattr(_RUNNER, "_launch_source", lambda *args, **kwargs: process)
    # The process is mocked, so its ps command cannot contain the dynamically
    # cloned interpreter/module paths; the dedicated identity regression tests
    # exercise that check directly.
    monkeypatch.setattr(_RUNNER, "_validate_source_process_identity", lambda *args, **kwargs: None)
    group_signals = []
    result = _RUNNER.run_benchmark(
        config,
        transport=transport,
        command_runner=_ps_runner(process),
        session_id_getter=lambda pid: pid,
        waitid_fn=lambda *_args: SimpleNamespace(si_pid=0),
        group_signal_sender=lambda *args: group_signals.append(args),
        source_identity_runner=_identity_runner(),
        port_allocator=lambda: 43123,
    )
    assert result["status"] == "complete", result.get("error")
    assert result["task_success"]["status"] == "pass"
    assert result["durable_evidence"]["trace_availability"]["status"] == "available"
    assert result["durable_evidence"]["trace"]["aggregation"]["semantic_turns"]["counts"]["foreground"]["local"] == 1
    assert result["metrics"]["quality_and_work"]["input_tokens"]["value"] == 7
    assert result["metrics"]["quality_and_work"]["output_tokens"]["value"] == 3
    assert result["metrics"]["latency"]["prefill_ms"]["value"] == 4.5
    assert result["sse"]["event_types"] == {"chunk": 1, "run.completed": 1}
    assert "events" not in result["sse"]
    assert "terminal_run" not in result["sse"]
    assert result["runtime"]["clone_removed"] is True
    assert group_signals
    request_paths = [url.rsplit("/", 1)[-1] for _method, url, _payload in transport.requests]
    assert "load" in request_paths
    assert "unload" in request_paths
    assert "assistant-fixed" in request_paths
    assert request_paths.index("unload") < request_paths.index("shutdown")
    load_payload = next(payload for method, url, payload in transport.requests if url.endswith("/api/inference/load"))
    unload_payload = next(payload for method, url, payload in transport.requests if url.endswith("/api/inference/unload"))
    assert load_payload["model_path"] == authoritative_model
    assert unload_payload["model_path"] == load_payload["model_path"]
    assert result["objective"]["request_summary"]["model"] == public_model
    assert result["model"]["name"] == public_model
    assert result["load"]["model"] == authoritative_model
    assert result["load"]["public_model_id"] == public_model
    artifact = Path(result["artifact_path"])
    rendered = artifact.read_text(encoding="utf-8")
    assert "in-memory-only-555" not in rendered
    assert "access-token-secret" not in rendered
    assert config.prompt not in rendered
    assert _RUNNER.EXPECTED_ANSWER not in rendered
    serialized_result = json.dumps(result, sort_keys=True)
    assert config.prompt not in serialized_result
    assert _RUNNER.EXPECTED_ANSWER not in serialized_result
    assert transport.assistant_message["content"] == [
        {"type": "text", "text": _RUNNER.EXPECTED_ANSWER}
    ]


def _clone_for_test(template: Path, run_dir: Path) -> _RUNNER.CloneHandle:
    shutil.copytree(template, run_dir, symlinks=True)
    return _RUNNER.CloneHandle(
        source=template.resolve(),
        run_dir=run_dir.resolve(),
        studio_home=(run_dir / ".unsloth" / "studio").resolve(),
        venv_python=(run_dir / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python").resolve(),
        source_manifest=_RUNNER._file_manifest(template),
    )
