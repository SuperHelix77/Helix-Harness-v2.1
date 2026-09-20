"""No-live-execution tests for the bounded P1-RO-REPO harness."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
from types import SimpleNamespace

import pytest


_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "benchmark-helix-v3-tool-scenarios.py"
_SPEC = importlib.util.spec_from_file_location("helix_v3_tool_scenarios", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _fixture() -> _RUNNER.FixtureSpec:
    return _RUNNER.load_fixture(
        (_REPO / "artifacts" / "helix-v3" / "phase1" / "fixtures" / _RUNNER.FIXTURE_ID).resolve()
    )


def _runtime_template(tmp_path: Path, password: str = "frozen-password-123") -> Path:
    template = tmp_path / "runtime-template"
    python = template / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    python.chmod(0o755)
    auth = template / ".unsloth" / "studio" / "auth"
    auth.mkdir(parents=True)
    (auth / ".bootstrap_password").write_text(password + "\n", encoding="utf-8")
    return template


def _config(tmp_path: Path, template: Path, *, execute: bool = False) -> _RUNNER.ToolScenarioConfig:
    model = tmp_path / "mlx-model"
    model.mkdir(exist_ok=True)
    model_config = tmp_path / "mlx-model.json"
    model_config.write_text(json.dumps({"backend": "mlx", "model_identity": {"repository": "mlx-model", "revision": "rev-1"}}), encoding="utf-8")
    return _RUNNER.ToolScenarioConfig(
        runtime_home_template=template,
        run_dir=tmp_path / "run-home",
        output_dir=tmp_path / "artifacts",
        repo_root=_REPO,
        fixture_root=_fixture().root,
        model_path=model,
        model_config_path=model_config,
        execute=execute,
    )


def _objective(request: dict, *, project_id: str = "project-test") -> dict[str, str]:
    value = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message_id": "user-1",
        "assistant_message_id": "assistant-1",
        "project_id": project_id,
        "session_id": f"project-{project_id}",
    }
    value["request_hash"] = _RUNNER._durable_request_hash(value, request)
    return value


def test_fixture_is_pinned_and_paths_ranges_and_shell_policy_are_exact():
    assert _RUNNER.SCRIPT_VERSION == "helix-v3-tool-scenarios.v2"
    assert "without Markdown fences" in _RUNNER.DEFAULT_PROMPT
    assert (
        "exact order and no others: alpha "
        in _RUNNER.DEFAULT_PROMPT
    )
    fixture = _fixture()
    assert fixture.fixture_sha256 == _RUNNER.CANONICAL_FIXTURE_SHA256
    assert fixture.manifest_sha256 == _RUNNER.CANONICAL_MANIFEST_SHA256
    assert _RUNNER.fixture_tree_hash(fixture.root) == _RUNNER.CANONICAL_FIXTURE_SHA256
    for index, item in enumerate(fixture.expected_sequence):
        summary = _RUNNER.validate_tool_proposal("terminal", {"command": item["command"]}, fixture, sequence_index=index)
        assert summary["arguments_sha256"] == item["arguments_sha256"]
        assert summary["proposal_payload_sha256"] == item["proposal_payload_sha256"]
    for command in (
        "cat README.md; rm -f README.md",
        "curl https://example.invalid README.md",
        "sed -i 's/a/b/' src/alpha.txt",
        "grep -R -- 'signal:' README.md",
        "grep -n -- signal ../README.md",
        "echo safe > README.md",
        "python3 -c 'print(1)'",
        "grep -n -- \"signal:\" ./README.md",
        "grep -n -- \"signal:\" README.md $HOME",
    ):
        with pytest.raises(_RUNNER.ProtocolMismatchError):
            _RUNNER._validate_command_text(command)


def test_modified_self_consistent_fixture_is_rejected_by_pinned_manifest(tmp_path):
    source = _fixture().root
    modified = tmp_path / "fixture"
    import shutil

    shutil.copytree(source, modified)
    manifest_path = modified / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest["fixture_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(_RUNNER.ContractError):
        _RUNNER.load_fixture(modified.resolve())


def test_request_matches_current_route_sanitizer_and_omits_unsupported_fields(tmp_path):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    payload = _RUNNER.build_tool_request_payload(config, "run", "thread", fixture, project_id="project-test")
    assert {"acceptance_criteria_sha256", "fixture_id", "fixture_sha256", "network_enabled"}.isdisjoint(payload)
    assert payload["session_id"] == "project-project-test"
    assert payload["permission_mode"] == "auto"
    assert payload["confirm_tool_calls"] is True
    backend_root = str(_REPO / "studio" / "backend")
    sys.path.insert(0, backend_root)
    try:
        from models.inference import ChatCompletionRequest

        assert set(payload) <= set(ChatCompletionRequest.model_fields)
        ChatCompletionRequest.model_validate(payload)
    finally:
        sys.path.remove(backend_root)


def test_tool_prompt_is_pinned_and_custom_prompt_is_rejected(tmp_path):
    config = _config(tmp_path, _runtime_template(tmp_path))
    custom = _RUNNER.ToolScenarioConfig(**{**config.__dict__, "prompt": "run something else"})
    with pytest.raises(_RUNNER.ContractError):
        _RUNNER.build_tool_request_payload(custom, "run", "thread", _fixture(), project_id="project-test")
    assert _RUNNER._prompt_content_sha256() == _RUNNER._sha256_json(_RUNNER._canonical_user_message_content())


def test_tool_request_is_terminal_only_and_launch_contract_does_not_disable_tools(tmp_path):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    payload = _RUNNER.build_tool_request_payload(config, "run", "thread", fixture, project_id="project-test")
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["terminal"]
    assert payload["enable_tools"] is True
    assert payload["enabled_tools"] == ["terminal"]
    assert payload["mcp_enabled"] is False
    assert payload["bypass_permissions"] is False
    assert payload["permission_mode"] == "auto"
    assert payload["confirm_tool_calls"] is True
    clone = SimpleNamespace(venv_python=tmp_path / "python", run_dir=tmp_path / "clone")
    launch = _RUNNER._tool_launch_command(clone, config, 43123)
    assert "--disable-tools" not in launch
    assert launch[-2:] == ["--password", "-"]


def test_dry_run_is_control_only_and_does_not_launch_or_write_artifact(tmp_path, monkeypatch):
    config = _config(tmp_path, _runtime_template(tmp_path))
    launched: list[bool] = []
    monkeypatch.setattr(_RUNNER, "_launch_tool_runtime", lambda *args, **kwargs: launched.append(True))
    identity = {"schema_version": "path-nul-sha256-newline-v2-artifacts-excluded", "repo": str(config.repo_root), "git_head": "head", "branch": "main", "dirty": True, "status_count": 1, "file_count": 2, "total_bytes": 10, "tree_sha256": "a" * 64}

    def identity_runner(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity))

    result = _RUNNER.run_benchmark(config, source_identity_runner=identity_runner)
    assert result["status"] == "dry_run"
    assert result["would_launch"] is False
    assert result["artifact_role"] == "baseline_control_only"
    assert not launched
    assert not (tmp_path / "artifacts").exists()
    assert result["privacy"]["raw_prompt_retained"] is False
    assert result["mandatory_flags"]["h2_repo_eligible"] is False
    assert result["benchmark_metadata"]["observation_pack"]["enabled"] is False
    assert result["benchmark_metadata"]["action_fusion"]["enabled"] is False
    assert all(measurement["value"] is None for group in result["metrics"].values() for measurement in group.values())


class _ProjectTransport:
    def __init__(self, root: Path):
        self.root = root

    def request_json(self, method, url, *, headers, payload=None, timeout):
        assert method == "POST" and url.endswith("/api/chat/projects")
        root = self.root / "project"
        sandbox = root / "sandbox"
        sandbox.mkdir(parents=True)
        return {"id": payload["id"], "rootPath": str(root), "sandboxPath": str(sandbox)}


def test_project_sandbox_is_clone_bound_empty_staged_readonly_and_integrity_checked(tmp_path):
    fixture = _fixture()
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    project = _RUNNER.create_managed_project(_ProjectTransport(clone_root), "http://loopback", {}, clone_root, timeout=1)
    staged = _RUNNER.stage_fixture_in_project(project, fixture)
    assert staged["fixture_sha256"] == _RUNNER.CANONICAL_FIXTURE_SHA256
    assert _RUNNER.verify_staged_fixture(project, fixture)["manifest_sha256"] == _RUNNER.CANONICAL_MANIFEST_SHA256
    sandbox = Path(project["sandbox"])
    assert stat.S_IMODE(sandbox.stat().st_mode) & 0o200
    assert not stat.S_IMODE((sandbox / "README.md").stat().st_mode) & 0o200
    assert not stat.S_IMODE((sandbox / "src").stat().st_mode) & 0o200
    outside = tmp_path / "outside"
    outside.mkdir()

    class OutsideTransport(_ProjectTransport):
        def request_json(self, method, url, *, headers, payload=None, timeout):
            return {"id": payload["id"], "rootPath": str(outside), "sandboxPath": str(outside)}

    with pytest.raises(_RUNNER.ContractError):
        _RUNNER.create_managed_project(OutsideTransport(clone_root), "http://loopback", {}, clone_root, timeout=1)


def test_exact_staged_fixture_can_restore_directory_permissions_for_clone_cleanup(tmp_path):
    fixture = _fixture()
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    project = _RUNNER.create_managed_project(_ProjectTransport(clone_root), "http://loopback", {}, clone_root, timeout=1)
    _RUNNER.stage_fixture_in_project(project, fixture)
    sandbox = Path(project["sandbox"])
    assert not stat.S_IMODE((sandbox / "src").stat().st_mode) & stat.S_IWUSR

    prepared = _RUNNER.prepare_staged_fixture_for_cleanup(project, fixture)

    assert prepared == {"directories_restored": 3, "fixture_sha256": fixture.fixture_sha256}
    assert stat.S_IMODE((sandbox / "src").stat().st_mode) & stat.S_IWUSR
    assert stat.S_IMODE((sandbox / "diagnostics").stat().st_mode) & stat.S_IWUSR
    assert _RUNNER.fixture_tree_hash(fixture.root) == fixture.fixture_sha256
    shutil.rmtree(clone_root)
    assert not clone_root.exists()


def _trace_payload(objective: dict[str, str], *, account_id: str = "owner", model_name: str = "local") -> dict:
    event = {
        "invocation_id": "inv-1",
        "scope": "foreground",
        "provider_tier": "local",
        "model_identity": model_name,
        "provider_identity": "local:mlx",
        "outcome": "completed",
        "semantic_contribution": "final_answer",
        "wasted": False,
        "wasted_reason": None,
        "retry_of": None,
        "escalation_from_tier": None,
    }
    envelope = {
        "schema_version": _RUNNER._SOURCE.TRACE_SCHEMA,
        "status": "available",
        "invocation_events": [event],
        "invocation_counts": _RUNNER._SOURCE.aggregate_trace_events([event]),
        "all_handles_settled": True,
        "error": None,
    }
    return {
        "schema_version": _RUNNER._SOURCE.TRACE_EVENT_SCHEMA,
        "status": "available",
        "aggregation": {"admissible": True, "reason": None},
        "binding": {"run_id": objective["run_id"], "thread_id": objective["thread_id"], "user_message_id": objective["user_message_id"], "owner_subject": "unsloth", "account_id": account_id},
        "trace": envelope,
        "error": None,
    }


def _create_evidence_db(path: Path, objective: dict[str, str], request: dict, fixture: _RUNNER.FixtureSpec, *, event_reorder: bool = False, blank_id: bool = False, hash_only: bool = False, wrong_account: bool = False, extra_request: bool = False):
    answer = _RUNNER._expected_answer_from_manifest(json.loads(fixture.manifest_path.read_text(encoding="utf-8")))
    if extra_request:
        request = {**request, "unexpected_field": True}
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE chat_generation_runs (id TEXT PRIMARY KEY, owner_subject TEXT, thread_id TEXT, user_message_id TEXT, assistant_message_id TEXT, status TEXT, finish_reason TEXT, request_json TEXT, request_hash TEXT, last_event_seq INTEGER, finalization_status TEXT);
        CREATE TABLE chat_threads (id TEXT PRIMARY KEY, project_id TEXT);
        CREATE TABLE chat_messages (id TEXT PRIMARY KEY, thread_id TEXT, parent_id TEXT, role TEXT, content_json TEXT, metadata_json TEXT);
        CREATE TABLE chat_generation_events (run_id TEXT, seq INTEGER, event_type TEXT, payload_json TEXT, created_at INTEGER);
        CREATE TABLE chat_generation_tool_approvals (run_id TEXT, approval_id TEXT);
        CREATE TABLE chat_generation_tool_executions (
            run_id TEXT, execution_id TEXT, backend_account_id TEXT, owner_subject TEXT,
            session_id TEXT, thread_id TEXT, tool_name TEXT, tool_call_id TEXT, card_call_id TEXT,
            arguments_json TEXT, arguments_fingerprint TEXT, pre_tool_checkpoint_json TEXT,
            pre_tool_checkpoint_version INTEGER, pre_tool_checkpoint_digest TEXT,
            authority_kind TEXT, approval_id TEXT, execution_state TEXT, claim_token TEXT,
            worker_token TEXT, claimed_at INTEGER, started_at INTEGER, finished_at INTEGER,
            result_json TEXT, error_message TEXT, producer_receipt_json TEXT, completion_json TEXT,
            controller_is_error INTEGER, completion_annotations_json TEXT,
            post_controller_checkpoint_json TEXT, terminal_seq INTEGER, receipt_ref TEXT,
            receipt_digest TEXT, PRIMARY KEY (run_id, execution_id)
        );
        """
    )
    connection.execute("INSERT INTO chat_threads VALUES (?, ?)", (objective["thread_id"], objective["project_id"]))
    run_request = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    request_hash = _RUNNER._durable_request_hash(objective, request)
    connection.execute("INSERT INTO chat_generation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (objective["run_id"], "unsloth", objective["thread_id"], objective["user_message_id"], objective["assistant_message_id"], "completed", "stop", run_request, request_hash, 0, "none"))
    sequence = 0
    command_outputs = ["3:signal: alpha\n4:signal: beta\n", "alpha=one\n", "records=3\n"]
    execution_rows = []
    execution_events = []
    for index, (item, output) in enumerate(zip(fixture.expected_sequence, command_outputs)):
        card_id = "" if blank_id and index == 0 else f"card-{index + 1}"
        model_call_id = f"call-{index + 1}"
        execution_id = f"execution-{index + 1}"
        args = {"command": item["command"]}
        checkpoint = {"version": 1, "backend": "safetensors", "conversation": [{"role": "assistant", "content": "", "tool_calls": [{"id": model_call_id, "type": "function", "function": {"name": "terminal", "arguments": json.dumps(args, sort_keys=True, separators=(",", ":"))}}]}], "controller": {}, "remaining_calls": [], "current_call": {"tool_call": {"id": model_call_id, "type": "function", "function": {"name": "terminal", "arguments": json.dumps(args, sort_keys=True, separators=(",", ":"))}}, "card_call_id": card_id, "provenance": {}}}
        checkpoint_json = _RUNNER._canonical_json(checkpoint)
        checkpoint_digest = _RUNNER._sha256_json(checkpoint)
        fingerprint = _RUNNER._sha256_json(args)
        sequence += 1
        proposal_seq = sequence
        start_payload = {"type": "tool_start", "tool_name": "terminal", "tool_call_id": card_id, "arguments": args, "arguments_text": _RUNNER._canonical_json(args), "provenance": {}, "approval_id": "", "awaiting_confirmation": False}
        connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "chunk", json.dumps(start_payload), sequence))
        sequence += 1
        claim_seq = sequence
        claim_payload = {"schema_version": "helix.tool-execution.v3", "execution_id": execution_id, "authority_kind": "ungated", "arguments_fingerprint": fingerprint, "pre_tool_checkpoint_digest": checkpoint_digest}
        connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "tool_execution.claimed", json.dumps(claim_payload), sequence))
        sequence += 1
        started_seq = sequence
        started_payload = {"schema_version": "helix.tool-execution.v3", "execution_id": execution_id, "tool_name": "terminal", "tool_call_id": card_id, "effect_state": "started", "authority_kind": "ungated"}
        connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "tool_execution.started", json.dumps(started_payload), sequence))
        sequence += 1
        finished_seq = sequence
        raw_bytes = output.encode("utf-8", "surrogatepass")
        producer = {"schema_version": "helix.tool-producer-receipt.v1", "producer_version": 1, "tool": "terminal", "resolved_launch_cwd_observed_before_spawn": "/tmp/fixture-sandbox", "cwd_identity_observed_before_spawn": {"device": 1, "inode": 2}, "confinement": "sandboxed-owner-direct", "process_outcome_kind": "exited", "return_code": 0, "return_code_available": True, "timed_out": False, "cancelled": False, "spawn_error": False, "decoded_output_utf8_surrogatepass_sha256": _RUNNER._sha256_bytes(raw_bytes), "decoded_output_utf8_surrogatepass_byte_length": len(raw_bytes), "capture_complete": True, "captured_byte_length": len(raw_bytes), "discarded_byte_length": 0, "capture_budget_chars": 4096, "read_error": None, "fallback_result_utf8_surrogatepass_sha256": _RUNNER._sha256_bytes(raw_bytes), "fallback_result_utf8_surrogatepass_byte_length": len(raw_bytes), "fallback_budget_chars": 4096}
        post_checkpoint = {"version": 1, "controller_state": {"executed_count": index + 1}}
        annotations = {}
        binding = {"backend_account_id": "wrong" if wrong_account else "owner", "owner_subject": "unsloth", "run_id": objective["run_id"], "session_id": objective["session_id"], "thread_id": objective["thread_id"], "execution_id": execution_id, "tool_name": "terminal", "tool_call_id": model_call_id, "card_call_id": card_id, "arguments_fingerprint": fingerprint, "pre_tool_checkpoint_digest": checkpoint_digest, "authority_kind": "ungated", "approval_id": None}
        completion = {"schema_version": "helix.tool-completion.v1", "execution_id": execution_id, "authority_kind": "ungated", "terminal_state": "finished", "selected_result": output, "error": None, "producer_receipt": producer, "controller_is_error": False, "completion_annotations": annotations, "post_controller_checkpoint": post_checkpoint, "receipt_ref": f"tool-receipt:{execution_id}", "binding": binding}
        receipt_digest = _RUNNER._sha256_json(completion)
        finish_result = {"stdout_sha256": item["result_sha256"], "stdout_length": item["result_length"]} if hash_only else output
        finish_payload = {"schema_version": "helix.tool-execution.v3", "execution_id": execution_id, "tool_name": "terminal", "tool_call_id": card_id, "effect_state": "finished", "authority_kind": "ungated", "receipt_ref": f"tool-receipt:{execution_id}", "receipt_digest": receipt_digest, "result": finish_result}
        connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "tool_execution.finished", json.dumps(finish_payload), sequence))
        sequence += 1
        connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "chunk", json.dumps({"type": "tool_end", "tool_name": "terminal", "tool_call_id": card_id, "result": output, "provenance": annotations}), sequence))
        execution_rows.append((objective["run_id"], execution_id, "wrong" if wrong_account else "owner", "unsloth", objective["session_id"], objective["thread_id"], "terminal", model_call_id, card_id, _RUNNER._canonical_json(args), fingerprint, checkpoint_json, 1, checkpoint_digest, "ungated", None, "finished", f"claim-{index}", "worker-1", 1000 + index * 10, 1001 + index * 10, 1002 + index * 10, json.dumps(output), None, json.dumps(producer, sort_keys=True, separators=(",", ":")), json.dumps(completion, sort_keys=True, separators=(",", ":")), 0, json.dumps(annotations), json.dumps(post_checkpoint), finished_seq, f"tool-receipt:{execution_id}", receipt_digest))
    trace = _trace_payload(objective, account_id="wrong" if wrong_account else "owner")
    sequence += 1
    connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "optimization.trace", json.dumps(trace), sequence))
    sequence += 1
    connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], sequence, "run.completed", json.dumps({"status": "completed", "finishReason": "stop"}), sequence))
    connection.execute("UPDATE chat_generation_runs SET last_event_seq=? WHERE id=?", (sequence, objective["run_id"]))
    connection.executemany("INSERT INTO chat_generation_tool_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", execution_rows)
    metadata = {"generationRunId": objective["run_id"], "generationSeq": sequence, "generationStatus": "completed", "generationSettled": True, "serverManaged": True}
    connection.execute("INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?)", (objective["user_message_id"], objective["thread_id"], None, "user", json.dumps(_RUNNER._canonical_user_message_content()), json.dumps({})))
    connection.execute("INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?)", (objective["assistant_message_id"], objective["thread_id"], objective["user_message_id"], "assistant", json.dumps([{"type": "text", "text": _RUNNER._canonical_json(answer)}]), json.dumps(metadata)))
    connection.commit()
    connection.close()
    if event_reorder:
        connection = sqlite3.connect(path)
        connection.execute("UPDATE chat_generation_events SET seq=999 WHERE run_id=? AND event_type='tool_execution.started' AND seq=(SELECT MIN(seq) FROM chat_generation_events WHERE run_id=? AND event_type='tool_execution.started')", (objective["run_id"], objective["run_id"]))
        connection.commit()
        connection.close()


def test_sqlite_verifier_requires_v3_durable_evidence_raw_results_and_privacy(tmp_path):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture)
    result = _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")
    assert result["status"] == "pass"
    assert result["raw_evidence_retained"] is True
    assert result["trace"]["invocation_counts"]["model_invocations"]["counts"]["foreground"]["local"] == 1
    assert result["final_answer"]["length"] == 98
    assert result["approval_rows"] == result["approval_events"] == 0
    assert result["coverage"]["status"] == "derived"
    assert result["coverage"]["filesystem_access"] == "unknown"
    assert "syscall" not in json.dumps(result["coverage"], sort_keys=True).lower()
    rendered = json.dumps(result, sort_keys=True)
    assert "signal: alpha" not in rendered and "alpha=one" not in rendered and "records=3" not in rendered


def test_sqlite_verifier_requires_zero_approval_rows_and_checks_stored_prompt(tmp_path):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO chat_generation_tool_approvals VALUES (?, ?)", (objective["run_id"], "unexpected-approval"))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM chat_generation_tool_approvals")
    connection.execute("UPDATE chat_messages SET content_json=? WHERE id=?", (json.dumps(_RUNNER._canonical_user_message_content("tampered")), objective["user_message_id"]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


@pytest.mark.parametrize(
    "event_type,field,value",
    [
        ("tool_execution.claimed", "schema_version", "helix.tool-execution.v2"),
        ("tool_execution.claimed", "execution_id", "wrong-execution"),
        ("tool_execution.started", "effect_state", "finished"),
        ("tool_execution.finished", "receipt_digest", "0" * 64),
    ],
)
def test_sqlite_verifier_rejects_incomplete_or_mismatched_v3_receipts(tmp_path, event_type, field, value):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture)
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT rowid, payload_json FROM chat_generation_events WHERE event_type=? ORDER BY seq LIMIT 1", (event_type,)).fetchone()
    payload = json.loads(row[1])
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    connection.execute("UPDATE chat_generation_events SET payload_json=? WHERE rowid=?", (json.dumps(payload), row[0]))
    connection.commit()
    connection.close()
    with pytest.raises((_RUNNER.ProtocolMismatchError, _RUNNER.AccountBindingError)):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_sqlite_verifier_requires_trace_immediately_before_final_completion(tmp_path):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture)
    connection = sqlite3.connect(database)
    terminal_seq = connection.execute("SELECT seq FROM chat_generation_events WHERE event_type='run.completed'").fetchone()[0]
    connection.execute("UPDATE chat_generation_events SET seq=? WHERE event_type='run.completed'", (terminal_seq + 1,))
    connection.execute("INSERT INTO chat_generation_events VALUES (?, ?, ?, ?, ?)", (objective["run_id"], terminal_seq, "chunk", json.dumps({"type": "status"}), terminal_seq))
    connection.execute("UPDATE chat_generation_runs SET last_event_seq=? WHERE id=?", (terminal_seq + 1, objective["run_id"]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


@pytest.mark.parametrize("kwargs", [{"wrong_account": True}, {"event_reorder": True}, {"blank_id": True}, {"hash_only": True}, {"extra_request": True}])
def test_sqlite_verifier_rejects_wrong_account_reorder_blank_ids_hash_only_and_extra_request(tmp_path, kwargs):
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture, **kwargs)
    with pytest.raises((_RUNNER.ProtocolMismatchError, _RUNNER.AccountBindingError)):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def _database_for_corruption(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = _fixture()
    config = _config(tmp_path, _runtime_template(tmp_path))
    request = _RUNNER.build_tool_request_payload(config, "run-1", "thread-1", fixture, project_id="project-test")
    objective = _objective(request)
    database = tmp_path / "studio.db"
    _create_evidence_db(database, objective, request, fixture)
    return fixture, request, objective, database


def test_auto_safe_public_proposal_never_calls_tool_confirmation(tmp_path):
    fixture = _fixture()
    calls = []

    class Transport:
        def request_json(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("auto-safe proposal must not call tool-confirm")

    payload = {
        "type": "tool_start",
        "tool_name": "terminal",
        "tool_call_id": "card-1",
        "arguments": {"command": fixture.expected_sequence[0]["command"]},
        "approval_id": "",
        "awaiting_confirmation": False,
    }
    result = _RUNNER._allow_tool_start(Transport(), "http://loopback", {}, "run-1", "session-1", payload, fixture, 0, 1)
    assert result["decision"] == "ungated"
    assert result["approval_id_sha256"] == _RUNNER._sha256_bytes(b"")
    assert calls == []


@pytest.mark.parametrize(
    "updates",
    [{"awaiting_confirmation": True}, {"approval_id": "approval-1"}],
)
def test_auto_safe_public_proposal_rejects_confirmation_state(tmp_path, updates):
    fixture = _fixture()
    payload = {
        "type": "tool_start",
        "tool_name": "terminal",
        "tool_call_id": "card-1",
        "arguments": {"command": fixture.expected_sequence[0]["command"]},
        "approval_id": "",
        "awaiting_confirmation": False,
    }
    payload.update(updates)
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER._allow_tool_start(object(), "http://loopback", {}, "run-1", "session-1", payload, fixture, 0, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("authority_kind", "approved"),
        ("approval_id", "approval-1"),
        ("pre_tool_checkpoint_digest", "0" * 64),
        ("receipt_ref", "tool-receipt:other"),
        ("receipt_digest", "0" * 64),
    ],
)
def test_ungated_execution_identity_and_checkpoint_binding_are_strict(tmp_path, field, value):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(f"UPDATE chat_generation_tool_executions SET {field}=? WHERE run_id=? AND execution_id=?", (value, objective["run_id"], "execution-1"))
    connection.commit()
    connection.close()
    with pytest.raises((_RUNNER.ProtocolMismatchError, _RUNNER.AccountBindingError)):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_launch_cwd_observed_before_spawn", "relative-cwd"),
        ("cwd_identity_observed_before_spawn", {"device": "bad", "inode": 2}),
        ("confinement", "sandboxed-unavailable"),
        ("process_outcome_kind", "spawn_error"),
        ("decoded_output_utf8_surrogatepass_sha256", "0" * 64),
        ("fallback_result_utf8_surrogatepass_sha256", "0" * 64),
    ],
)
def test_producer_receipt_cwd_confinement_outcome_and_output_hashes_are_strict(tmp_path, field, value):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT producer_receipt_json FROM chat_generation_tool_executions WHERE run_id=? AND execution_id=?", (objective["run_id"], "execution-1")).fetchone()
    receipt = json.loads(row[0])
    receipt[field] = value
    connection.execute("UPDATE chat_generation_tool_executions SET producer_receipt_json=? WHERE run_id=? AND execution_id=?", (json.dumps(receipt, sort_keys=True, separators=(",", ":")), objective["run_id"], "execution-1"))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


@pytest.mark.parametrize("event_kind", ["tool_execution.claimed", "tool_execution.started", "tool_execution.finished"])
def test_ungated_v3_missing_or_ambiguous_execution_events_fail_closed(tmp_path, event_kind):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT seq FROM chat_generation_events WHERE run_id=? AND event_type=? ORDER BY seq LIMIT 1", (objective["run_id"], event_kind)).fetchone()
    connection.execute("DELETE FROM chat_generation_events WHERE run_id=? AND seq=?", (objective["run_id"], row[0]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_ungated_v3_ambiguous_event_type_and_corrupt_completion_fail_closed(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE chat_generation_events SET event_type='tool_execution.ambiguous' WHERE run_id=? AND event_type='tool_execution.finished' AND seq=(SELECT MIN(seq) FROM chat_generation_events WHERE run_id=? AND event_type='tool_execution.finished')", (objective["run_id"], objective["run_id"]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")

    fixture, request, objective, database = _database_for_corruption(tmp_path / "completion")
    connection = sqlite3.connect(database)
    connection.execute("UPDATE chat_generation_tool_executions SET completion_json=? WHERE run_id=? AND execution_id=?", (json.dumps({"schema_version": "corrupt"}), objective["run_id"], "execution-1"))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_public_end_mismatch_and_duplicate_fail_closed(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT rowid, payload_json FROM chat_generation_events WHERE run_id=? AND event_type='chunk' AND payload_json LIKE '%tool_end%' ORDER BY seq LIMIT 1", (objective["run_id"],)).fetchone()
    payload = json.loads(row[1])
    payload["result"] = "tampered"
    connection.execute("UPDATE chat_generation_events SET payload_json=? WHERE rowid=?", (json.dumps(payload), row[0]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")

    fixture, request, objective, database = _database_for_corruption(tmp_path / "duplicate")
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT seq, payload_json FROM chat_generation_events WHERE run_id=? AND event_type='chunk' AND payload_json LIKE '%tool_end%' ORDER BY seq LIMIT 1", (objective["run_id"],)).fetchone()
    connection.execute("UPDATE chat_generation_events SET payload_json=? WHERE run_id=? AND event_type='chunk' AND payload_json LIKE '%tool_end%' AND seq=(SELECT MAX(seq) FROM chat_generation_events WHERE run_id=? AND event_type='chunk' AND payload_json LIKE '%tool_end%')", (row[1], objective["run_id"], objective["run_id"]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_checkpoint_completion_and_producer_receipt_corruption_is_rejected(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE chat_generation_tool_executions SET pre_tool_checkpoint_json=? WHERE run_id=? AND execution_id=?", (json.dumps({"version": 1}), objective["run_id"], "execution-1"))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_checkpoint_backend_must_be_a_recoverable_local_backend(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    row = connection.execute(
        "SELECT pre_tool_checkpoint_json FROM chat_generation_tool_executions "
        "WHERE run_id=? AND execution_id=?",
        (objective["run_id"], "execution-1"),
    ).fetchone()
    checkpoint = json.loads(row[0])
    checkpoint["backend"] = "studio"
    connection.execute(
        "UPDATE chat_generation_tool_executions "
        "SET pre_tool_checkpoint_json=?, pre_tool_checkpoint_digest=? "
        "WHERE run_id=? AND execution_id=?",
        (
            _RUNNER._canonical_json(checkpoint),
            _RUNNER._sha256_json(checkpoint),
            objective["run_id"],
            "execution-1",
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(
            database,
            objective=objective,
            fixture=fixture,
            request_payload=request,
            model_name="local",
            provider_identity="local:mlx",
        )


def test_public_start_command_mutation_is_rejected_even_if_allowlisted(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT rowid, payload_json FROM chat_generation_events WHERE run_id=? AND event_type='chunk' ORDER BY seq LIMIT 1", (objective["run_id"],)).fetchone()
    payload = json.loads(row[1])
    payload["arguments"] = {"command": fixture.expected_sequence[1]["command"]}
    connection.execute("UPDATE chat_generation_events SET payload_json=? WHERE rowid=?", (json.dumps(payload), row[0]))
    connection.commit()
    connection.close()
    with pytest.raises(_RUNNER.ProtocolMismatchError):
        _RUNNER.read_tool_durable_evidence(database, objective=objective, fixture=fixture, request_payload=request, model_name="local", provider_identity="local:mlx")


def test_output_or_run_overlap_fixture_is_rejected(tmp_path):
    config = _config(tmp_path, _runtime_template(tmp_path))
    overlap = _fixture().root / "nested-run"
    bad = _RUNNER.ToolScenarioConfig(**{**config.__dict__, "run_dir": overlap})
    with pytest.raises(_RUNNER.ContractError):
        _RUNNER.run_benchmark(bad, source_identity_runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps({"schema_version": "path-nul-sha256-newline-v2-artifacts-excluded", "repo": str(config.repo_root), "git_head": "h", "branch": "b", "dirty": True, "status_count": 0, "file_count": 0, "total_bytes": 0, "tree_sha256": "a" * 64})))


def test_public_sse_metrics_delegate_to_source_and_keep_unavailable_explicit(monkeypatch):
    calls = []

    def collect(sse, *, elapsed_ms):
        calls.append(elapsed_ms)
        return {
            "metrics": _RUNNER.unavailable_metrics(),
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "visible_output_chars": 1,
            "visible_output_sha256": "a" * 64,
        }

    monkeypatch.setattr(_RUNNER._SOURCE, "collect_metrics", collect)
    result = _RUNNER.collect_metrics(
        {"events": [{"received_elapsed_ms": 4.5, "payload": {"choices": [{"delta": {"content": "{"}}]}}]},
        elapsed_ms=12.0,
    )
    assert calls == [12.0]
    assert result["metrics"]["quality_and_work"]["total_tokens"]["value"] == 10
    assert result["metrics"]["latency"]["ttft_first_protocol_event_ms"]["value"] == 4.5
    assert result["metrics"]["latency"]["ttft_first_user_visible_token_ms"]["value"] == 4.5
    assert result["metrics"]["quality_and_work"]["tool_calls"]["value"] is None


def test_durable_tool_metrics_require_verified_timestamps_and_never_fake_zeroes():
    metrics = _RUNNER.unavailable_metrics()
    evidence = {"executions": [{"started_at": 10, "finished_at": 13}, {"started_at": 20, "finished_at": 22}, {"started_at": 30, "finished_at": 31}]}
    admitted = _RUNNER._admit_durable_tool_metrics(metrics, evidence, expected_calls=3)
    assert admitted["quality_and_work"]["tool_calls"]["value"] == 3
    assert admitted["quality_and_work"]["failed_tool_calls"]["value"] == 0
    assert admitted["quality_and_work"]["prevented_tool_calls"]["value"] == 0
    assert admitted["quality_and_work"]["redundant_tool_calls"]["value"] == 0
    assert admitted["latency"]["tool_ms"]["value"] == 6
    assert "sqlite:chat_generation_tool_executions" in admitted["quality_and_work"]["tool_calls"]["exposure"]
    assert "sqlite:chat_generation_tool_approvals" not in admitted["quality_and_work"]["tool_calls"]["exposure"]
    unavailable = _RUNNER._admit_durable_tool_metrics(_RUNNER.unavailable_metrics(), {"executions": [{"started_at": None, "finished_at": 13}] * 3}, expected_calls=3)
    assert unavailable["latency"]["tool_ms"]["value"] is None


def test_atomic_artifact_refuses_collision_and_does_not_claim_self_path(tmp_path, monkeypatch):
    output = tmp_path / "artifacts"
    record = {"status": "failed"}
    monkeypatch.setattr(_RUNNER._datetime, "datetime", SimpleNamespace(now=lambda _tz: SimpleNamespace(strftime=lambda _fmt: "20260920T000000Z"), timezone=_RUNNER._datetime.timezone))
    first = _RUNNER.atomic_artifact(output, record)
    original = first.read_bytes()
    with pytest.raises(_RUNNER.ContractError):
        _RUNNER.atomic_artifact(output, {"status": "different"})
    assert first.read_bytes() == original
    assert "artifact_path" not in json.loads(original)


def test_tool_child_environment_disables_bytecode(tmp_path):
    config = _config(tmp_path, _runtime_template(tmp_path))
    clone = SimpleNamespace(run_dir=tmp_path / "clone")
    environment = _RUNNER._tool_launch_environment(clone, config)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_structured_answer_requires_exact_saved_content_and_no_model_verified_claim():
    fixture = _fixture()
    expected = _RUNNER._canonical_json(_RUNNER._expected_answer_from_manifest(json.loads(fixture.manifest_path.read_text(encoding="utf-8"))))
    assert "verified" not in expected
    assert expected == '{"alpha":"one","fixture_id":"p1-ro-repo-v1","read_count":3,"records":3,"signals":["alpha","beta"]}'

    exact = _RUNNER._assess_answer_text(expected, fixture)
    fenced = _RUNNER._assess_answer_text(f"```json\n{expected}\n```", fixture)
    assert exact["exact"] is True and exact["semantic_match"] is True
    assert fenced["exact"] is False and fenced["semantic_match"] is True


def test_sqlite_verifier_retains_nonexact_semantic_answer_as_task_failure(tmp_path):
    fixture, request, objective, database = _database_for_corruption(tmp_path)
    expected = _RUNNER._canonical_json(
        _RUNNER._expected_answer_from_manifest(
            json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
        )
    )
    fenced = f"```json\n{expected}\n```"
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE chat_messages SET content_json=? WHERE id=?",
        (
            json.dumps([{"type": "text", "text": fenced}]),
            objective["assistant_message_id"],
        ),
    )
    connection.commit()
    connection.close()

    evidence = _RUNNER.read_tool_durable_evidence(
        database,
        objective=objective,
        fixture=fixture,
        request_payload=request,
        model_name="local",
        provider_identity="local:mlx",
    )
    assert evidence["status"] == "pass"
    assert evidence["final_answer"]["exact"] is False
    assert evidence["final_answer"]["semantic_match"] is True
    assert fenced not in json.dumps(evidence, sort_keys=True)


def test_cleanup_uncertainty_is_a_failure_boundary_and_clone_is_retained(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path)
    config = _config(tmp_path, template, execute=True)
    fixture = _fixture()
    clone_dir = tmp_path / "clone"
    clone_dir.mkdir()
    clone = SimpleNamespace(source=template, run_dir=clone_dir, source_manifest="a" * 64)
    removed: list[bool] = []
    monkeypatch.setattr(_RUNNER._SOURCE, "remove_clone_if_proven", lambda _clone, *, cleanup_proven: removed.append(cleanup_proven) or cleanup_proven)
    assert _RUNNER._SOURCE.remove_clone_if_proven(clone, cleanup_proven=False) is False
    assert removed == [False]
    assert _RUNNER.fixture_tree_hash(fixture.root) == fixture.fixture_sha256


class _FakeProcess:
    pid = 4242

    def __init__(self):
        self.stdin = io.StringIO()
        self.exited = False

    def poll(self):
        return 0 if self.exited else None

    def terminate(self):
        self.exited = True

    def kill(self):
        self.exited = True

    def wait(self, **_kwargs):
        self.exited = True
        return 0


def test_mocked_lifecycle_classifies_verifier_failure_retains_uncertain_clone_and_redacts_artifact(tmp_path, monkeypatch):
    template = _runtime_template(tmp_path, password="private-bootstrap-value")
    config = _config(tmp_path, template, execute=True)
    identity = {"schema_version": "path-nul-sha256-newline-v2-artifacts-excluded", "repo": str(config.repo_root), "git_head": "head", "branch": "main", "dirty": True, "status_count": 1, "file_count": 2, "total_bytes": 10, "tree_sha256": "a" * 64}

    def identity_runner(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity))

    objective = {"run_id": "run-fixed", "thread_id": "thread-fixed", "user_message_id": "user-fixed", "assistant_message_id": "assistant-fixed", "project_id": "project-fixed", "session_id": "project-project-fixed"}
    fixture = _fixture()
    request = _RUNNER.build_tool_request_payload(config, objective["run_id"], objective["thread_id"], fixture, project_id="project-fixed")
    process = _FakeProcess()
    project_root = config.run_dir / "project"
    sandbox = project_root / "sandbox"

    class Transport:
        def __init__(self):
            self.assistant = {"id": objective["assistant_message_id"], "threadId": objective["thread_id"], "parentId": objective["user_message_id"], "role": "assistant", "content": [], "attachments": None, "metadata": {"generationRunId": objective["run_id"], "generationStatus": "completed", "serverManaged": True}, "createdAt": 1}

        def request_json(self, method, url, *, headers, payload=None, timeout):
            if url.endswith("/api/auth/login"):
                return {"access_token": "private-access-token", "account_id": "owner"}
            if url.endswith("/api/chat/projects"):
                sandbox.mkdir(parents=True)
                return {"id": payload["id"], "rootPath": str(project_root), "sandboxPath": str(sandbox)}
            if url.endswith("/api/inference/load"):
                return {"status": "loaded", "model": payload["model_path"], "is_mlx": True, "is_local_model": True}
            if url.endswith("/api/inference/unload"):
                return {"status": "unloaded", "model": payload["model_path"]}
            if url.endswith("/api/inference/chat-runs"):
                return {"id": objective["run_id"], "threadId": objective["thread_id"], "userMessageId": objective["user_message_id"], "assistantMessageId": objective["assistant_message_id"], "requestHash": _RUNNER._durable_request_hash(objective, payload["requestPayload"]), "requestPayload": payload["requestPayload"]}
            if url.endswith("/api/inference/tool-confirm"):
                return {"resolved": True}
            if url.endswith("/api/shutdown"):
                return {"ok": True}
            if url.endswith("/" + objective["assistant_message_id"]):
                if method == "GET":
                    return dict(self.assistant)
                self.assistant = dict(payload)
                return dict(self.assistant)
            if url.endswith("/api/chat/threads"):
                return {"id": objective["thread_id"], "projectId": payload["projectId"]}
            return {"id": objective["user_message_id"], "threadId": objective["thread_id"], "role": "user"}

        def open_stream(self, *_args, **_kwargs):
            answer = _RUNNER._canonical_json(_RUNNER._expected_answer_from_manifest(json.loads(fixture.manifest_path.read_text(encoding="utf-8"))))
            proposal = {"type": "tool_start", "tool_name": "terminal", "tool_call_id": "card-1", "arguments": {"command": fixture.expected_sequence[0]["command"]}, "approval_id": "approval-1", "awaiting_confirmation": True}
            chunk = {"seq": 1, "type": "chunk", "payload": proposal, "createdAt": 1}
            terminal = {"seq": 2, "type": "run.completed", "payload": {"status": "completed", "finishReason": "stop"}, "createdAt": 2, "run": {"id": objective["run_id"], "status": "completed", "finishReason": "stop", "lastEventSeq": 2}}
            return io.BytesIO(("id: 1\nevent: chunk\ndata: " + json.dumps(chunk) + "\n\n" + "id: 2\nevent: run.completed\ndata: " + json.dumps(terminal) + "\n\n").encode())

    def clone_runtime_home(_source, run_dir, **_kwargs):
        Path(run_dir).mkdir(parents=True)
        return _RUNNER._SOURCE.CloneHandle(source=template, run_dir=Path(run_dir), studio_home=Path(run_dir) / ".unsloth" / "studio", venv_python=Path(run_dir) / "python", source_manifest="a" * 64)

    monkeypatch.setattr(_RUNNER._SOURCE, "clone_runtime_home", clone_runtime_home)
    monkeypatch.setattr(_RUNNER._SOURCE, "read_password", lambda _path: "private-bootstrap-value")
    monkeypatch.setattr(_RUNNER._SOURCE, "ephemeral_launch_password", lambda _password: "private-launch-value")
    monkeypatch.setattr(_RUNNER._SOURCE, "_reaper_boundary_status", lambda **_kwargs: (True, "ready"))
    monkeypatch.setattr(_RUNNER, "_launch_tool_runtime", lambda *args, **kwargs: process)
    monkeypatch.setattr(_RUNNER._SOURCE, "capture_process_identities", lambda *_args, **_kwargs: {4242: {"pid": 4242, "ppid": 1, "pgid": 4242, "identity_token": "token", "command": "owned"}})
    monkeypatch.setattr(_RUNNER._SOURCE, "_validate_source_process_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(_RUNNER._SOURCE, "_validate_owned_process_group", lambda *_args, **_kwargs: (True, "owned"))
    monkeypatch.setattr(_RUNNER._SOURCE, "_merge_process_identities", lambda owned, incoming: (owned.update(incoming) or set()))
    monkeypatch.setattr(_RUNNER._SOURCE, "_wait_for_health", lambda *_args, **_kwargs: {"contract": _RUNNER.EXPECTED_BACKEND_CONTRACT, "verified": False, "tree_sha256": None})
    monkeypatch.setattr(_RUNNER._SOURCE, "verify_cleanup", lambda *_args, **_kwargs: {"status": "unavailable", "provenance": "mock_uncertain"})
    monkeypatch.setattr(_RUNNER._SOURCE, "terminate_owned_processes", lambda process, *_args, **_kwargs: (process.terminate() or {"status": "fail"}))
    monkeypatch.setattr(_RUNNER, "read_tool_durable_evidence", lambda *_args, **_kwargs: (_ for _ in ()).throw(_RUNNER.ProtocolMismatchError("missing durable receipt")))
    monkeypatch.setattr(_RUNNER._SOURCE, "remove_clone_if_proven", lambda _clone, *, cleanup_proven: cleanup_proven)

    result = _RUNNER.run_benchmark(config, transport=Transport(), source_identity_runner=identity_runner, port_allocator=lambda: 43123)
    assert result["status"] == "failed"
    assert result["correctness"]["outcome"] == "unverified"
    assert result["cleanup"]["retained_on_uncertainty"] is True
    assert result["runtime"]["clone_removed"] is False
    artifact_text = Path(result["artifact_path"]).read_text(encoding="utf-8")
    for secret in ("private-bootstrap-value", "private-launch-value", "private-access-token"):
        assert secret not in artifact_text
    assert config.prompt not in artifact_text
    assert "signal: alpha" not in artifact_text
