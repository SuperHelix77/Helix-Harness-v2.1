# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import hashlib
import os
import sqlite3

import pytest

from core.inference.durable_tool_journal import (
    DurableExecutionHandle,
    durable_result_value,
    durable_tool_run,
    prepared_result,
    replace_durable_result_value,
    resume_checkpoint_calls,
    settle_controller_completion,
)
from core.inference import durable_agent_recovery as recovery
from core.inference import tools as tools_module
from core.inference.tool_loop_controller import ToolLoopController
from core.inference.tool_producer_receipt import (
    OBSERVATION_SEED_MAX_BYTES,
    OBSERVATION_SEED_MIN_BYTES,
    ObservationSeed,
    ProcessOutputCapture,
    ProducedToolResult,
    ResultBudgetExposure,
    observe_launch_cwd,
    observation_seed_for_text,
    produced_result,
    receipt_dict,
)
from core.inference.tool_stream_exec import stream_tool_execution
from core.inference.tools import _bash_exec, _python_exec
from storage import chat_generation_runs_db as db
from storage import studio_db
from utils.account_context import AccountContext, current_account_id, run_as


def _checkpoint(
    *,
    call_id: str = "call-1",
    card_id: str = "card-1",
    arguments: dict | None = None,
    controller: dict | None = None,
    backend: str = "gguf",
) -> tuple[dict, str]:
    arguments = arguments or {"command": "printf ok"}
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps(arguments)},
    }
    value = {
        "version": 1,
        "backend": backend,
        "conversation": [{"role": "assistant", "content": "", "tool_calls": [call]}],
        "current_call": {
            "tool_call": call,
            "card_call_id": card_id,
            "provenance": {},
        },
        "remaining_calls": [],
        "controller": controller or {},
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _seed(run_id: str = "run-receipt") -> str:
    suffix = run_id.removeprefix("run-")
    thread_id = f"thread-{suffix}"
    user_id = f"user-{suffix}"
    assistant_id = f"assistant-{suffix}"
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "Receipt",
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
            "content": [{"type": "text", "text": "run"}],
            "createdAt": 2,
        }
    )
    db.create_run(
        run_id=run_id,
        owner_subject="owner",
        thread_id=thread_id,
        user_message_id=user_id,
        assistant_message_id=assistant_id,
        request_payload={
            "model": "local",
            "messages": [{"role": "user", "content": "run"}],
            "stream": True,
            "cancel_id": run_id,
            "generation_run_id": run_id,
            "finalization_idempotency_key": run_id,
        },
    )
    worker = db.get_worker_run(run_id)
    assert worker is not None
    token = worker[2]
    assert db.mark_running(run_id, token)
    return token


def _claim(token: str, *, arguments: dict | None = None):
    actual_arguments = arguments or {"command": "printf ok"}
    checkpoint, checkpoint_digest = _checkpoint(arguments=actual_arguments)
    return db.claim_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt",
        tool_name="terminal",
        tool_call_id="call-1",
        card_call_id="card-1",
        arguments=actual_arguments,
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )


def test_ungated_terminal_retry_replays_stored_completion_and_fences_authority():
    token = _seed()
    claimed = _claim(token)
    assert claimed and claimed["executionState"] == "claimed"
    assert _claim(token)["claimToken"] == claimed["claimToken"]
    with pytest.raises(db.ToolExecutionConflictError):
        _claim(token, arguments={"command": "printf changed"})
    assert db.mark_ungated_tool_execution_started(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    ) is not None
    completion = {
        "loop_progress": {"executed_calls": 1},
    }
    finished = db.finish_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="ok",
        controller_is_error=False,
        completion_annotations=completion,
        post_controller_checkpoint={"version": 1},
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )
    assert finished and finished["result"] == "ok"
    retry = db.finish_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="ok",
        controller_is_error=False,
        completion_annotations=completion,
        post_controller_checkpoint={"version": 1},
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )
    assert retry["receiptDigest"] == finished["receiptDigest"]
    assert (
        db.finish_ungated_tool_execution(
            "run-receipt",
            "execution-1",
            worker_token="wrong-worker",
            claim_token=claimed["claimToken"],
            result="attacker-result",
            pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
        )
        is None
    )
    assert (
        db.finish_ungated_tool_execution(
            "run-receipt",
            "execution-1",
            worker_token=token,
            claim_token="wrong-claim",
            result="attacker-result",
            pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
        )
        is None
    )
    replay_with_changed_candidate = db.finish_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="different",
        error="stale optional error",
        ambiguous=True,
        producer_receipt={"candidate": "different"},
        controller_is_error=True,
        completion_annotations={"changed": True},
        post_controller_checkpoint={"version": 99},
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )
    assert replay_with_changed_candidate["receiptDigest"] == finished["receiptDigest"]
    assert replay_with_changed_candidate["result"] == "ok"
    assert replay_with_changed_candidate["error"] is None
    assert replay_with_changed_candidate["executionState"] == "finished"
    terminal = [
        event
        for event in db.list_events("run-receipt")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(terminal) == 1


def test_stop_before_start_fences_and_stop_after_start_preserves_latched_finish():
    token = _seed()
    claimed = _claim(token)
    db.request_cancel("run-receipt", "owner")
    assert db.mark_ungated_tool_execution_started(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    ) is None
    assert db.get_ungated_tool_execution("run-receipt", "execution-1")[
        "executionState"
    ] == "cancelled"

    token = _seed("run-receipt-2")
    checkpoint, checkpoint_digest = _checkpoint(
        call_id="call-2", card_id="card-2"
    )
    claimed = db.claim_ungated_tool_execution(
        "run-receipt-2",
        "execution-2",
        worker_token=token,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt-2",
        tool_name="terminal",
        tool_call_id="call-2",
        card_call_id="card-2",
        arguments={"command": "printf ok"},
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    assert db.mark_ungated_tool_execution_started(
        "run-receipt-2",
        "execution-2",
        worker_token=token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )
    db.request_cancel("run-receipt-2", "owner")
    assert db.finish_ungated_tool_execution(
        "run-receipt-2",
        "execution-2",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="latched",
        controller_is_error=False,
        completion_annotations={},
        post_controller_checkpoint={"version": 1},
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )["executionState"] == "finished"


def test_stream_commits_selected_controller_completion_and_replays_without_recording(
    monkeypatch,
):
    # The finisher returns its committed row.  A second lookup would make a
    # lost-response recovery depend on a new read rather than the transaction
    # that established the completion.
    monkeypatch.setattr(
        db,
        "get_ungated_tool_execution",
        lambda *_args, **_kwargs: pytest.fail(
            "settlement must consume the finisher's committed row"
        ),
    )
    token = _seed()
    tools = [{"type": "function", "function": {"name": "terminal"}}]
    decision_call = {
        "id": "call-1",
        "card_id": "card-1",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf Error"}),
        },
    }
    controller = ToolLoopController(tools=tools)
    decision = controller.prepare_call(decision_call)
    checkpoint, _checkpoint_digest = _checkpoint(
        arguments={"command": "printf Error"},
        controller=controller.export_state(),
    )
    with durable_tool_run("run-receipt", token):
        generator = stream_tool_execution(
            lambda _callback: "Error: printed by a zero-exit process",
            tool_name="terminal",
            tool_call_id="card-1",
            journal_tool_call_id="call-1",
            arguments={"command": "printf Error"},
            session_id="session-1",
            thread_id="thread-receipt",
            pre_tool_checkpoint=checkpoint,
        )
        with pytest.raises(StopIteration) as stopped:
            next(generator)
        prepared = stopped.value.value
        assert durable_result_value(prepared) == "Error: printed by a zero-exit process"
        prepared = replace_durable_result_value(
            prepared, "Error: selected after a pre-controller window transform"
        )
        completion = settle_controller_completion(prepared, decision, controller)
    assert completion.is_error is True
    [stored] = db.list_ungated_tool_executions("run-receipt")
    assert stored["result"] == completion.result
    assert stored["controllerIsError"] is True
    assert stored["postControllerCheckpoint"]["executed_count"] == 1

    replay_controller = ToolLoopController(tools=tools)
    replay_decision = replay_controller.prepare_call(decision_call)
    replay_checkpoint, _replay_digest = _checkpoint(
        arguments={"command": "printf Error"},
        controller=replay_controller.export_state(),
    )
    with durable_tool_run("run-receipt", token):
        replay = stream_tool_execution(
            lambda _callback: pytest.fail("finished execution was invoked again"),
            tool_name="terminal",
            tool_call_id="card-1",
            journal_tool_call_id="call-1",
            arguments={"command": "printf Error"},
            session_id="session-1",
            thread_id="thread-receipt",
            pre_tool_checkpoint=replay_checkpoint,
        )
        with pytest.raises(StopIteration) as replayed:
            next(replay)
        replay_completion = settle_controller_completion(
            replayed.value.value, replay_decision, replay_controller
        )
    assert replay_completion.result == completion.result
    assert replay_controller.export_state() == controller.export_state()


def test_terminal_and_python_results_carry_backend_receipts():
    terminal = _bash_exec("printf 'line\\n'; exit 7", timeout=5)
    python = _python_exec("print('ok')", timeout=5)
    assert isinstance(terminal, ProducedToolResult)
    assert terminal.producer_receipt.process_outcome_kind == "nonzero_exit"
    assert terminal.producer_receipt.return_code == 7
    assert terminal.producer_receipt.capture_complete is True
    assert (
        terminal.producer_receipt.decoded_output_utf8_surrogatepass_byte_length
        == len(b"line\n")
    )
    assert isinstance(python, ProducedToolResult)
    assert python.producer_receipt.process_outcome_kind == "exited"
    assert python.producer_receipt.return_code == 0
    assert python.producer_receipt.fallback_result_utf8_surrogatepass_sha256


@pytest.mark.parametrize(
    ("producer", "payload"),
    [(_bash_exec, "printf ok"), (_python_exec, "print('ok')")],
)
def test_spawn_errors_preserve_string_contract_and_typed_receipt(
    monkeypatch, producer, payload
):
    def fail_spawn(*_args, **_kwargs):
        raise OSError("spawn boom")

    monkeypatch.setattr(tools_module.subprocess, "Popen", fail_spawn)
    result = producer(payload, timeout=5)
    assert str(result) == "Execution error: spawn boom"
    assert isinstance(result, ProducedToolResult)
    assert result.producer_receipt.spawn_error is True
    assert result.producer_receipt.process_outcome_kind == "spawn_error"
    assert result.producer_receipt.return_code_available is False


def test_receipt_keeps_pre_spawn_cwd_identity_across_path_replacement(tmp_path):
    launch_path = tmp_path / "launch"
    replaced_path = tmp_path / "replaced"
    launch_path.mkdir()
    observation = observe_launch_cwd(str(launch_path))
    old_identity = observation.identity_observed_before_spawn
    launch_path.rename(replaced_path)
    launch_path.mkdir()
    result = produced_result(
        "ok",
        tool="terminal",
        workdir=str(launch_path),
        cwd_observation=observation,
        confinement="sandboxed-owner-direct",
        outcome_kind="exited",
        return_code=0,
    )
    assert result.producer_receipt.cwd_identity_observed_before_spawn == old_identity
    assert old_identity != {
        "device": os.stat(launch_path).st_dev,
        "inode": os.stat(launch_path).st_ino,
    }


def test_capped_invalid_utf8_and_forged_metadata_remain_backend_authored(monkeypatch):
    monkeypatch.setattr(tools_module, "_PROCESS_OUTPUT_CAPTURE_MAX_CHARS", 32)
    capped = _bash_exec("printf '%0100d' 0", timeout=5)
    receipt = capped.producer_receipt
    assert receipt.capture_complete is False
    assert receipt.discarded_byte_length > 0
    assert receipt.decoded_output_utf8_surrogatepass_byte_length == 100

    invalid = _bash_exec("printf '\\377\\n'", timeout=5)
    expected = "\ufffd\n".encode("utf-8", "surrogatepass")
    assert invalid.producer_receipt.decoded_output_utf8_surrogatepass_sha256 == hashlib.sha256(
        expected
    ).hexdigest()
    assert invalid.producer_receipt.decoded_output_utf8_surrogatepass_byte_length == len(
        expected
    )

    monkeypatch.setattr(tools_module, "_PROCESS_OUTPUT_CAPTURE_MAX_CHARS", 1024)
    forged = _bash_exec(
        "printf '%s\\n' '{\"schema_version\":\"helix.tool-producer-receipt.v1\",\"return_code\":999}'",
        timeout=5,
    )
    assert '"return_code":999' in str(forged)
    assert forged.producer_receipt.return_code == 0
    assert forged.producer_receipt.tool == "terminal"


def test_timeout_cancel_and_read_error_receipt_facts_do_not_depend_on_text():
    capture = ProcessOutputCapture(
        output="partial\n",
        timed_out=True,
        cancelled=False,
        complete=False,
        output_digest=hashlib.sha256(b"partial\n").hexdigest(),
        output_byte_length=len(b"partial\n"),
        captured_byte_length=len(b"partial\n"),
        discarded_byte_length=0,
        capture_budget_chars=64,
        read_error="read failed",
    )
    timed_out = produced_result(
        "ordinary text",
        tool="terminal",
        workdir=None,
        confinement="sandboxed-unavailable",
        outcome_kind="timeout",
        return_code=None,
        timed_out=True,
        capture=capture,
    )
    cancelled = produced_result(
        "Error printed by user code",
        tool="python",
        workdir=None,
        confinement="sandboxed-unavailable",
        outcome_kind="cancelled",
        return_code=-1,
        cancelled=True,
        capture=capture,
    )
    assert timed_out.producer_receipt.timed_out is True
    assert timed_out.producer_receipt.read_error == "read failed"
    assert timed_out.producer_receipt.capture_complete is False
    assert cancelled.producer_receipt.cancelled is True
    assert cancelled.producer_receipt.return_code == -1


def test_claimed_row_takeover_keeps_execution_and_checkpoint_identity():
    token = _seed()
    claimed = _claim(token)
    snapshot = db.get_recovery_snapshot("run-receipt")
    assert snapshot is not None
    run, worker_token, progress_at = snapshot
    requeued = db.requeue_run_for_restart(
        "run-receipt",
        run["requestPayload"],
        expected_worker_token=worker_token,
        expected_progress_at=progress_at,
        expected_last_event_seq=run["lastEventSeq"],
        stale_before_ms=db.now_ms() + 1000,
    )
    assert requeued is not None
    replacement_worker = db.get_worker_run("run-receipt")
    assert replacement_worker is not None
    replacement = replacement_worker[2]
    assert db.mark_running("run-receipt", replacement)
    checkpoint = claimed["preToolCheckpoint"]
    replay = db.claim_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=replacement,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt",
        tool_name="terminal",
        tool_call_id="call-1",
        card_call_id="card-1",
        arguments={"command": "printf ok"},
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=claimed["preToolCheckpointDigest"],
    )
    assert replay["executionId"] == claimed["executionId"]
    assert replay["preToolCheckpointDigest"] == claimed["preToolCheckpointDigest"]
    assert replay["claimToken"] != claimed["claimToken"]


def test_wrong_account_identity_checkpoint_and_worker_are_fenced():
    token = _seed()
    claimed = _claim(token)
    checkpoint = claimed["preToolCheckpoint"]
    digest = claimed["preToolCheckpointDigest"]
    with pytest.raises(db.ToolApprovalFencedError):
        db.claim_ungated_tool_execution(
            "run-receipt",
            "execution-foreign",
            worker_token=token,
            backend_account_id="f" * 32,
            session_id="session-1",
            thread_id="thread-receipt",
            tool_name="terminal",
            tool_call_id="call-foreign",
            card_call_id="card-foreign",
            arguments={"command": "printf ok"},
            pre_tool_checkpoint=checkpoint,
            pre_tool_checkpoint_digest=digest,
        )
    changed_checkpoint = json.loads(json.dumps(checkpoint))
    changed_checkpoint["backend"] = "safetensors"
    changed_encoded = json.dumps(
        changed_checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(db.ToolExecutionConflictError):
        db.claim_ungated_tool_execution(
            "run-receipt",
            "execution-1",
            worker_token=token,
            backend_account_id=current_account_id(),
            session_id="session-1",
            thread_id="thread-receipt",
            tool_name="terminal",
            tool_call_id="call-1",
            card_call_id="card-1",
            arguments={"command": "printf ok"},
            pre_tool_checkpoint=changed_checkpoint,
            pre_tool_checkpoint_digest=hashlib.sha256(
                changed_encoded.encode("utf-8")
            ).hexdigest(),
        )
    with pytest.raises(db.ToolExecutionConflictError):
        db.claim_ungated_tool_execution(
            "run-receipt",
            "execution-1",
            worker_token=token,
            backend_account_id=current_account_id(),
            session_id="session-1",
            thread_id="thread-receipt",
            tool_name="python",
            tool_call_id="call-1",
            card_call_id="card-1",
            arguments={"command": "printf ok"},
            pre_tool_checkpoint=checkpoint,
            pre_tool_checkpoint_digest=digest,
        )
    with pytest.raises(db.ToolApprovalFencedError):
        db.claim_ungated_tool_execution(
            "run-receipt",
            "execution-wrong-thread",
            worker_token=token,
            backend_account_id=current_account_id(),
            session_id="session-1",
            thread_id="another-thread",
            tool_name="terminal",
            tool_call_id="call-x",
            card_call_id="card-x",
            arguments={"command": "printf ok"},
            pre_tool_checkpoint=checkpoint,
            pre_tool_checkpoint_digest=digest,
        )
    assert (
        db.mark_ungated_tool_execution_started(
            "run-receipt",
            "execution-1",
            worker_token="wrong-worker",
            claim_token=claimed["claimToken"],
            pre_tool_checkpoint_digest=digest,
        )
        is None
    )
    with pytest.raises(db.ToolExecutionConflictError):
        db.mark_ungated_tool_execution_started(
            "run-receipt",
            "execution-1",
            worker_token=token,
            claim_token=claimed["claimToken"],
            pre_tool_checkpoint_digest="0" * 64,
        )


def test_same_run_and_execution_ids_are_isolated_across_accounts():
    alice = AccountContext("a" * 32, "alice")
    bob = AccountContext("b" * 32, "bob")
    alice_token = run_as(alice, _seed)
    bob_token = run_as(bob, _seed)
    alice_claim = run_as(alice, _claim, alice_token)
    bob_claim = run_as(bob, _claim, bob_token)
    assert alice_claim["executionId"] == bob_claim["executionId"] == "execution-1"
    assert alice_claim["backendAccountId"] == alice.account_id
    assert bob_claim["backendAccountId"] == bob.account_id
    assert run_as(
        alice, db.get_ungated_tool_execution, "run-receipt", "execution-1"
    )["backendAccountId"] == alice.account_id
    assert run_as(
        bob, db.get_ungated_tool_execution, "run-receipt", "execution-1"
    )["backendAccountId"] == bob.account_id


def test_finished_before_public_end_recovers_completion_without_rerun():
    token = _seed()
    tools = [{"type": "function", "function": {"name": "terminal"}}]
    raw_call = {
        "id": "call-1",
        "card_id": "card-1",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf ok"}),
        },
    }
    before = ToolLoopController(tools=tools)
    decision = before.prepare_call(raw_call)
    checkpoint, checkpoint_digest = _checkpoint(controller=before.export_state())
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "arguments": {"command": "printf ok"},
                    "arguments_text": json.dumps({"command": "printf ok"}),
                    "approval_id": "",
                },
                db.now_ms(),
            )
        ],
    )
    claimed = db.claim_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt",
        tool_name="terminal",
        tool_call_id="call-1",
        card_call_id="card-1",
        arguments={"command": "printf ok"},
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    db.mark_ungated_tool_execution_started(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    selected = before.record_result(decision, "settled once")
    db.finish_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result=selected.result,
        controller_is_error=selected.is_error,
        completion_annotations=dict(decision.provenance),
        post_controller_checkpoint=before.export_state(),
        pre_tool_checkpoint_digest=checkpoint_digest,
    )

    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is True
    assert plan.reason == "resume_finished_ungated_tool_receipt"
    assert plan.action == "resume_tool"
    assert plan.resume_checkpoint["recovered_receipt"]["completion"][
        "selected_result"
    ] == selected.result

    replay_controller = ToolLoopController(tools=tools)
    replay_controller.record_result = lambda *_args, **_kwargs: pytest.fail(
        "controller selection reran"
    )
    conversation = list(plan.resume_checkpoint["conversation"])
    generator = resume_checkpoint_calls(
        plan.resume_checkpoint,
        backend="gguf",
        controller=replay_controller,
        conversation=conversation,
        execute_tool=lambda *_args, **_kwargs: pytest.fail("producer reran"),
        stream_tool_execution=lambda *_args, **_kwargs: pytest.fail("stream reran"),
        cancel_event=None,
        tool_call_timeout=10,
        session_id="session-1",
        thread_id="thread-receipt",
        rag_scope=None,
        bypass_permissions=False,
        permission_mode="off",
        confirm_tool_calls=False,
    )
    events = []
    with pytest.raises(StopIteration):
        while True:
            events.append(next(generator))
    assert [event["type"] for event in events] == ["tool_end"]
    assert events[0]["result"] == selected.result


def test_public_start_without_authoritative_row_fails_closed():
    token = _seed()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "card-missing",
                    "arguments": {"command": "printf missing"},
                    "approval_id": "",
                },
                db.now_ms(),
            )
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "ungated_public_start_has_no_durable_row"


def _commit_finished_open(
    *,
    remaining_calls: list[dict] | None = None,
    result: str = "settled once",
    checkpoint_extras: dict | None = None,
) -> tuple[str, dict, dict, dict]:
    token = _seed()
    tools = [{"type": "function", "function": {"name": "terminal"}}]
    raw_call = {
        "id": "call-1",
        "card_id": "card-1",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf ok"}),
        },
    }
    controller = ToolLoopController(tools=tools)
    decision = controller.prepare_call(raw_call)
    checkpoint, _unused = _checkpoint(controller=controller.export_state())
    checkpoint["remaining_calls"] = list(remaining_calls or [])
    checkpoint.update(checkpoint_extras or {})
    checkpoint_json = json.dumps(
        checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    checkpoint_digest = hashlib.sha256(checkpoint_json.encode("utf-8")).hexdigest()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "arguments": {"command": "printf ok"},
                    "approval_id": "",
                },
                db.now_ms(),
            )
        ],
    )
    claimed = db.claim_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt",
        tool_name="terminal",
        tool_call_id="call-1",
        card_call_id="card-1",
        arguments={"command": "printf ok"},
        pre_tool_checkpoint=checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    db.mark_ungated_tool_execution_started(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    selected = controller.record_result(decision, result)
    finished = db.finish_ungated_tool_execution(
        "run-receipt",
        "execution-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result=selected.result,
        controller_is_error=selected.is_error,
        completion_annotations=dict(decision.provenance),
        post_controller_checkpoint=controller.export_state(),
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    return token, checkpoint, finished, controller.export_state()


@pytest.mark.parametrize(
    "corruption",
    [
        "completion",
        "receipt_digest",
        "receipt_ref",
        "terminal_event",
        "terminal_type",
        "missing_terminal_event",
    ],
)
def test_finished_replay_integrity_corruption_fails_closed(corruption):
    _token, _checkpoint_value, finished, _controller_state = _commit_finished_open()
    conn = db._connect()
    try:
        if corruption == "completion":
            conn.execute(
                """UPDATE chat_generation_tool_executions
                   SET completion_json=? WHERE run_id=? AND execution_id=?""",
                ('{"schema_version":"corrupt"}', "run-receipt", "execution-1"),
            )
        elif corruption == "receipt_digest":
            conn.execute(
                """UPDATE chat_generation_tool_executions
                   SET receipt_digest=? WHERE run_id=? AND execution_id=?""",
                ("0" * 64, "run-receipt", "execution-1"),
            )
        elif corruption == "receipt_ref":
            conn.execute(
                """UPDATE chat_generation_tool_executions
                   SET receipt_ref=? WHERE run_id=? AND execution_id=?""",
                ("tool-receipt:other", "run-receipt", "execution-1"),
            )
        elif corruption == "terminal_event":
            event = conn.execute(
                """SELECT payload_json FROM chat_generation_events
                   WHERE run_id=? AND seq=?""",
                ("run-receipt", finished["terminalSeq"]),
            ).fetchone()
            payload = json.loads(event["payload_json"])
            payload["receipt_digest"] = "f" * 64
            conn.execute(
                """UPDATE chat_generation_events SET payload_json=?
                   WHERE run_id=? AND seq=?""",
                (
                    json.dumps(payload, separators=(",", ":")),
                    "run-receipt",
                    finished["terminalSeq"],
                ),
            )
        elif corruption == "terminal_type":
            conn.execute(
                """UPDATE chat_generation_events
                   SET event_type='tool_execution.ambiguous'
                   WHERE run_id=? AND seq=?""",
                ("run-receipt", finished["terminalSeq"]),
            )
        else:
            conn.execute(
                "DELETE FROM chat_generation_events WHERE run_id=? AND seq=?",
                ("run-receipt", finished["terminalSeq"]),
            )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert db.get_ungated_tool_execution("run-receipt", "execution-1")[
        "executionState"
    ] == "finished"


def test_persisted_public_end_is_not_replayed_or_duplicated():
    token, _checkpoint_value, finished, controller_state = _commit_finished_open(
        checkpoint_extras={"tool_iters_done": 4, "iteration": 2}
    )
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": finished["result"],
                    "provenance": finished["completionAnnotations"],
                },
                db.now_ms(),
            )
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.reason == "resume_after_published_ungated_completion"
    assert plan.resume_checkpoint["current_public_end_persisted"] is True
    assert plan.resume_checkpoint["tool_iters_done"] == 4
    assert plan.resume_checkpoint["iteration"] == 2
    replay_controller = ToolLoopController(
        tools=[{"type": "function", "function": {"name": "terminal"}}]
    )
    replay_controller.record_result = lambda *_args, **_kwargs: pytest.fail(
        "controller selection reran"
    )
    conversation = list(plan.resume_checkpoint["conversation"])
    generator = resume_checkpoint_calls(
        plan.resume_checkpoint,
        backend="gguf",
        controller=replay_controller,
        conversation=conversation,
        execute_tool=lambda *_args, **_kwargs: pytest.fail("producer reran"),
        stream_tool_execution=lambda *_args, **_kwargs: pytest.fail("stream reran"),
        cancel_event=None,
        tool_call_timeout=10,
        session_id="session-1",
        thread_id="thread-receipt",
        rag_scope=None,
        bypass_permissions=False,
        permission_mode="off",
        confirm_tool_calls=False,
    )
    with pytest.raises(StopIteration):
        next(generator)
    assert replay_controller.export_state() == controller_state
    assert conversation[-1]["role"] == "tool"
    assert conversation[-1]["content"] == finished["result"]
    public_ends = [
        event
        for event in db.list_events("run-receipt")
        if event["type"] == "chunk"
        and event["payload"].get("type") == "tool_end"
    ]
    assert len(public_ends) == 1


def test_progress_after_published_completion_fails_closed_without_counter_reset():
    token, _checkpoint_value, finished, _controller_state = _commit_finished_open()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": finished["result"],
                },
                db.now_ms(),
            ),
            (
                "chunk",
                {"choices": [{"delta": {"content": "newer model output"}}]},
                db.now_ms(),
            ),
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.action == "fail_closed"
    assert plan.reason == "progress_after_published_tool_completion"


def test_altered_public_arguments_fail_closed_before_replay():
    _commit_finished_open()
    conn = db._connect()
    try:
        row = conn.execute(
            """SELECT seq, payload_json FROM chat_generation_events
               WHERE run_id=? AND event_type='chunk' ORDER BY seq LIMIT 1""",
            ("run-receipt",),
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["arguments"] = {"command": "printf altered"}
        conn.execute(
            "UPDATE chat_generation_events SET payload_json=? WHERE run_id=? AND seq=?",
            (json.dumps(payload, separators=(",", ":")), "run-receipt", row["seq"]),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "ungated_tool_event_causality_mismatch"


def test_reordered_finish_and_public_end_fail_closed():
    token, _checkpoint_value, finished, _controller_state = _commit_finished_open()
    [end_seq] = db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": finished["result"],
                },
                db.now_ms(),
            )
        ],
    )
    terminal_seq = finished["terminalSeq"]
    conn = db._connect()
    try:
        conn.execute(
            "UPDATE chat_generation_events SET seq=-1 WHERE run_id=? AND seq=?",
            ("run-receipt", terminal_seq),
        )
        conn.execute(
            "UPDATE chat_generation_events SET seq=? WHERE run_id=? AND seq=?",
            (terminal_seq, "run-receipt", end_seq),
        )
        conn.execute(
            "UPDATE chat_generation_events SET seq=? WHERE run_id=? AND seq=-1",
            (end_seq, "run-receipt"),
        )
        conn.execute(
            """UPDATE chat_generation_tool_executions SET terminal_seq=?
               WHERE run_id=? AND execution_id=?""",
            (end_seq, "run-receipt", "execution-1"),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "ungated_tool_event_causality_mismatch"


def test_duplicate_public_end_with_pending_suffix_fails_closed():
    second_call = {
        "id": "call-2",
        "card_id": "card-2",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf second"}),
        },
    }
    token, _checkpoint_value, finished, _controller_state = _commit_finished_open(
        remaining_calls=[second_call]
    )
    end = (
        "chunk",
        {
            "type": "tool_end",
            "tool_name": "terminal",
            "tool_call_id": "card-1",
            "result": finished["result"],
        },
        db.now_ms(),
    )
    db.append_events("run-receipt", token, [end, end])
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_public_frontier_is_ambiguous"


def test_stop_after_committed_receipt_blocks_recovery_and_preserves_evidence():
    _token, _checkpoint_value, finished, _controller_state = _commit_finished_open()
    assert db.request_cancel("run-receipt", "owner") is not None
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "explicit_stop_was_pending"
    stored = db.get_ungated_tool_execution("run-receipt", "execution-1")
    assert stored["receiptDigest"] == finished["receiptDigest"]
    assert stored["completion"] == finished["completion"]


def test_mixed_approved_and_ungated_open_frontiers_fail_closed():
    token = _seed()
    approved_arguments = {"command": "printf approved"}
    approved_checkpoint, approved_digest = _checkpoint(
        call_id="approved-call",
        card_id="approved-card",
        arguments=approved_arguments,
    )
    proposal = {
        "approval_id": "approval-1",
        "session_id": "session-1",
        "tool_name": "terminal",
        "tool_call_id": "approved-call",
        "card_call_id": "approved-card",
        "execution_id": "approved-execution",
        "arguments": approved_arguments,
        "arguments_fingerprint": hashlib.sha256(
            json.dumps(
                approved_arguments, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "checkpoint_version": 1,
        "resume_checkpoint": approved_checkpoint,
        "proposed_at": 100,
        "expires_at": 10_000_000_000_000,
    }
    assert approved_digest
    db.append_events_with_tool_proposal(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "approved-card",
                    "arguments": approved_arguments,
                    "arguments_text": json.dumps(approved_arguments),
                    "approval_id": "approval-1",
                    "awaiting_confirmation": True,
                },
                db.now_ms(),
            )
        ],
        proposal,
    )

    ungated_arguments = {"command": "printf ungated"}
    ungated_checkpoint, ungated_digest = _checkpoint(
        call_id="ungated-call",
        card_id="ungated-card",
        arguments=ungated_arguments,
    )
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "ungated-card",
                    "approval_id": "",
                },
                db.now_ms(),
            )
        ],
    )
    ungated = db.claim_ungated_tool_execution(
        "run-receipt",
        "ungated-execution",
        worker_token=token,
        backend_account_id=current_account_id(),
        session_id="session-1",
        thread_id="thread-receipt",
        tool_name="terminal",
        tool_call_id="ungated-call",
        card_call_id="ungated-card",
        arguments=ungated_arguments,
        pre_tool_checkpoint=ungated_checkpoint,
        pre_tool_checkpoint_digest=ungated_digest,
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "mixed_tool_authority_frontier_is_ambiguous"
    assert db.get_tool_approval("run-receipt", "approval-1")[
        "executionState"
    ] == "unclaimed"
    assert ungated["executionState"] == "claimed"


def _append_approved_call(
    token: str,
    *,
    approval_id: str = "approval-2",
    call_id: str = "approved-call",
    card_id: str = "approved-card",
    execution_id: str = "approved-execution",
    remaining_calls: list[dict] | None = None,
) -> tuple[dict, ToolLoopController, object]:
    arguments = {"command": "printf approved"}
    raw_call = {
        "id": call_id,
        "card_id": card_id,
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps(arguments)},
    }
    controller = ToolLoopController(
        tools=[{"type": "function", "function": {"name": "terminal"}}]
    )
    decision = controller.prepare_call(raw_call)
    checkpoint, _digest = _checkpoint(
        call_id=call_id,
        card_id=card_id,
        arguments=arguments,
        controller=controller.export_state(),
    )
    checkpoint["remaining_calls"] = list(remaining_calls or [])
    checkpoint["tool_iters_done"] = 9
    proposal = {
        "approval_id": approval_id,
        "session_id": "session-1",
        "tool_name": "terminal",
        "tool_call_id": call_id,
        "card_call_id": card_id,
        "execution_id": execution_id,
        "arguments": arguments,
        "arguments_fingerprint": hashlib.sha256(
            json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "checkpoint_version": 1,
        "resume_checkpoint": checkpoint,
        "proposed_at": 100,
        "expires_at": 10_000_000_000_000,
    }
    db.append_events_with_tool_proposal(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": card_id,
                    "arguments": arguments,
                    "arguments_text": json.dumps(arguments),
                    "approval_id": approval_id,
                    "awaiting_confirmation": True,
                },
                db.now_ms(),
            )
        ],
        proposal,
    )
    return checkpoint, controller, decision


def _finish_appended_approved_call(
    token: str,
    controller: ToolLoopController,
    decision: object,
    *,
    result: str = "approved-result",
) -> dict:
    db.decide_tool_approval(
        "run-receipt",
        "approval-2",
        owner_subject="owner",
        session_id="session-1",
        decision="allow",
    )
    approved = db.claim_tool_execution(
        "run-receipt", "approval-2", worker_token=token
    )
    assert approved is not None
    assert db.mark_tool_execution_started(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
    )
    selected = controller.record_result(decision, result)
    finished = db.finish_tool_execution(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
        result=selected.result,
        controller_is_error=selected.is_error,
        completion_annotations=dict(decision.provenance),
        post_controller_checkpoint=controller.export_state(),
    )
    assert finished is not None
    return finished


@pytest.mark.parametrize("corruption", ["completion", "digest", "erase"])
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("has_suffix", [False, True])
def test_corrupt_modern_approved_receipt_never_downgrades_to_legacy(
    corruption, closed, has_suffix
):
    token = _seed()
    suffix = [
        {
            "id": "call-3",
            "card_id": "card-3",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({"command": "printf third"}),
            },
        }
    ] if has_suffix else []
    _checkpoint_value, controller, decision = _append_approved_call(
        token, remaining_calls=suffix
    )
    finished = _finish_appended_approved_call(token, controller, decision)
    if closed:
        db.append_events(
            "run-receipt",
            token,
            [
                (
                    "chunk",
                    {
                        "type": "tool_end",
                        "tool_name": "terminal",
                        "tool_call_id": "approved-card",
                        "result": finished["result"],
                    },
                    db.now_ms(),
                )
            ],
        )
    conn = db._connect()
    try:
        if corruption == "completion":
            conn.execute(
                """UPDATE chat_generation_tool_approvals SET completion_json=?
                   WHERE run_id=? AND approval_id=?""",
                ('{"schema_version":"corrupt"}', "run-receipt", "approval-2"),
            )
        elif corruption == "digest":
            conn.execute(
                """UPDATE chat_generation_tool_approvals SET receipt_digest=?
                   WHERE run_id=? AND approval_id=?""",
                ("0" * 64, "run-receipt", "approval-2"),
            )
        else:
            conn.execute(
                """UPDATE chat_generation_tool_approvals
                   SET receipt_ref=NULL, receipt_digest=NULL, terminal_seq=NULL,
                       producer_receipt_json=NULL, completion_json=NULL,
                       controller_is_error=NULL, completion_annotations_json=NULL,
                       post_controller_checkpoint_json=NULL
                   WHERE run_id=? AND approval_id=?""",
                ("run-receipt", "approval-2"),
            )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "finished_approved_tool_completion_missing_or_corrupt"


@pytest.mark.parametrize("closed", [False, True])
def test_modern_approved_terminal_metadata_rejects_nonfinished_row(closed):
    token = _seed()
    _checkpoint_value, controller, decision = _append_approved_call(token)
    finished = _finish_appended_approved_call(token, controller, decision)
    if closed:
        db.append_events(
            "run-receipt",
            token,
            [
                (
                    "chunk",
                    {
                        "type": "tool_end",
                        "tool_name": "terminal",
                        "tool_call_id": "approved-card",
                        "result": finished["result"],
                    },
                    db.now_ms(),
                )
            ],
        )
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE chat_generation_tool_approvals SET execution_state='claimed'
               WHERE run_id=? AND approval_id=?""",
            ("run-receipt", "approval-2"),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "approved_row_event_frontier_is_inconsistent"


@pytest.mark.parametrize("has_suffix", [False, True])
def test_closed_modern_approved_receipt_requires_valid_checkpoint(has_suffix):
    token = _seed()
    suffix = [
        {
            "id": "call-3",
            "card_id": "card-3",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({"command": "printf third"}),
            },
        }
    ] if has_suffix else []
    _checkpoint_value, controller, decision = _append_approved_call(
        token, remaining_calls=suffix
    )
    finished = _finish_appended_approved_call(token, controller, decision)
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "approved-card",
                    "result": finished["result"],
                },
                db.now_ms(),
            )
        ],
    )
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE chat_generation_tool_approvals SET resume_checkpoint_json='{}'
               WHERE run_id=? AND approval_id=?""",
            ("run-receipt", "approval-2"),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_checkpoint_missing_or_incompatible"


def test_erased_receipt_and_rolled_back_row_state_cannot_reinvoke_approved_tool():
    token = _seed()
    _checkpoint_value, controller, decision = _append_approved_call(token)
    _finish_appended_approved_call(token, controller, decision)
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET execution_state='claimed', receipt_ref=NULL,
                   receipt_digest=NULL, terminal_seq=NULL,
                   producer_receipt_json=NULL, completion_json=NULL,
                   controller_is_error=NULL, completion_annotations_json=NULL,
                   post_controller_checkpoint_json=NULL
               WHERE run_id=? AND approval_id=?""",
            ("run-receipt", "approval-2"),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.action == "fail_closed"
    assert plan.reason == "approved_row_event_frontier_is_inconsistent"


def test_ungated_finished_plus_ambiguous_terminal_evidence_fails_closed():
    token, _checkpoint_value, _finished, _controller_state = _commit_finished_open()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "tool_execution.ambiguous",
                {
                    "execution_id": "execution-1",
                    "effect_state": "ambiguous",
                },
                db.now_ms(),
            )
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "ungated_tool_event_causality_mismatch"


def test_approved_finished_plus_ambiguous_terminal_evidence_fails_closed():
    token = _seed()
    _checkpoint_value, controller, decision = _append_approved_call(token)
    _finish_appended_approved_call(token, controller, decision)
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "tool_execution.ambiguous",
                {
                    "execution_id": "approved-execution",
                    "effect_state": "ambiguous",
                },
                db.now_ms(),
            )
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "approved_row_event_frontier_is_inconsistent"


def test_closed_ungated_checkpoint_never_supersedes_later_pending_approval():
    approved_call = {
        "id": "approved-call",
        "card_id": "approved-card",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf approved"}),
        },
    }
    token, _checkpoint_value, first, _controller_state = _commit_finished_open(
        remaining_calls=[approved_call]
    )
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": first["result"],
                },
                db.now_ms(),
            )
        ],
    )
    _append_approved_call(token)
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "wait_approval"
    assert plan.approval_id == "approval-2"


def test_closed_ungated_checkpoint_never_supersedes_later_finished_approval():
    token, _checkpoint_value, first, _controller_state = _commit_finished_open()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": first["result"],
                },
                db.now_ms(),
            )
        ],
    )
    _checkpoint_value, controller, decision = _append_approved_call(token)
    db.decide_tool_approval(
        "run-receipt",
        "approval-2",
        owner_subject="owner",
        session_id="session-1",
        decision="allow",
    )
    approved = db.claim_tool_execution(
        "run-receipt", "approval-2", worker_token=token
    )
    assert approved is not None
    assert db.mark_tool_execution_started(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
    )
    selected = controller.record_result(decision, "approved-result")
    db.finish_tool_execution(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
        result=selected.result,
        controller_is_error=selected.is_error,
        completion_annotations=dict(decision.provenance),
        post_controller_checkpoint=controller.export_state(),
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.reason == "resume_finished_tool_receipt"
    assert plan.approval_id == "approval-2"
    assert plan.resume_checkpoint["recovered_receipt"]["result"] == "approved-result"


def test_latest_closed_approved_completion_replays_once_and_continues_suffix():
    token, _checkpoint_value, first, _controller_state = _commit_finished_open()
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": first["result"],
                },
                db.now_ms(),
            )
        ],
    )
    third_call = {
        "id": "call-3",
        "card_id": "card-3",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf third"}),
        },
    }
    _checkpoint_value, approved_controller, approved_decision = _append_approved_call(
        token, remaining_calls=[third_call]
    )
    db.decide_tool_approval(
        "run-receipt",
        "approval-2",
        owner_subject="owner",
        session_id="session-1",
        decision="allow",
    )
    approved = db.claim_tool_execution(
        "run-receipt", "approval-2", worker_token=token
    )
    assert approved is not None
    assert db.mark_tool_execution_started(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
    )
    selected = approved_controller.record_result(approved_decision, "approved-result")
    finished = db.finish_tool_execution(
        "run-receipt",
        "approval-2",
        worker_token=token,
        claim_token=approved["claimToken"],
        result=selected.result,
        controller_is_error=selected.is_error,
        completion_annotations=dict(approved_decision.provenance),
        post_controller_checkpoint=approved_controller.export_state(),
    )
    assert finished is not None
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "approved-card",
                    "result": selected.result,
                },
                db.now_ms(),
            )
        ],
    )
    [plan] = recovery.plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.reason == "resume_after_published_approved_completion"
    assert plan.approval_id == "approval-2"
    assert plan.resume_checkpoint["current_public_end_persisted"] is True
    assert plan.resume_checkpoint["remaining_calls"] == [third_call]

    replay_controller = ToolLoopController(
        tools=[{"type": "function", "function": {"name": "terminal"}}]
    )
    record_calls: list[str] = []
    original_record = replay_controller.record_result

    def record_once(decision, value):
        record_calls.append(decision.tool_call_id)
        return original_record(decision, value)

    replay_controller.record_result = record_once
    producer_calls: list[str] = []
    conversation = list(plan.resume_checkpoint["conversation"])
    generator = resume_checkpoint_calls(
        plan.resume_checkpoint,
        backend="gguf",
        controller=replay_controller,
        conversation=conversation,
        execute_tool=lambda _name, arguments, **_kwargs: (
            producer_calls.append(arguments["command"]) or "third-result"
        ),
        stream_tool_execution=stream_tool_execution,
        cancel_event=None,
        tool_call_timeout=10,
        session_id="session-1",
        thread_id="thread-receipt",
        rag_scope=None,
        bypass_permissions=False,
        permission_mode="off",
        confirm_tool_calls=False,
    )
    replay_events = list(generator)
    assert producer_calls == ["printf third"]
    assert record_calls == ["call-3"]
    assert [event["tool_call_id"] for event in replay_events if event["type"] == "tool_end"] == ["card-3"]
    assert not any(
        event.get("type") == "tool_end"
        and event.get("tool_call_id") == "approved-card"
        for event in replay_events
    )


def test_double_restart_advances_sibling_without_duplicate_end_or_rerun():
    second_call = {
        "id": "call-2",
        "card_id": "card-2",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "printf second"}),
        },
    }
    token, checkpoint, first_finished, _first_state = _commit_finished_open(
        remaining_calls=[second_call],
        result="first-result",
        checkpoint_extras={"tool_iters_done": 7, "iteration": 3},
    )
    db.append_events(
        "run-receipt",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": "card-1",
                    "result": first_finished["result"],
                    "provenance": first_finished["completionAnnotations"],
                },
                db.now_ms(),
            )
        ],
    )
    [first_plan] = recovery.plan_orphaned_runs()
    assert first_plan.safe is True
    assert first_plan.reason == "resume_after_published_ungated_completion"
    assert first_plan.resume_checkpoint["current_public_end_persisted"] is True

    [requeued] = recovery.requeue_planned_runs(
        [first_plan], stale_before_ms=db.now_ms() + 1000
    )
    first_worker = db.get_worker_run("run-receipt")
    assert first_worker is not None
    first_worker_token = first_worker[2]
    assert db.mark_running("run-receipt", first_worker_token)
    resumed_checkpoint = requeued["_resumeCheckpoint"]
    conversation = list(resumed_checkpoint["conversation"])
    controller = ToolLoopController(
        tools=[{"type": "function", "function": {"name": "terminal"}}]
    )
    record_calls: list[str] = []
    original_record = controller.record_result

    def count_record(decision, value):
        record_calls.append(decision.tool_call_id)
        return original_record(decision, value)

    controller.record_result = count_record
    producer_calls: list[str] = []

    def execute_tool(name, arguments, **_kwargs):
        producer_calls.append(arguments["command"])
        return "second-result"

    with durable_tool_run("run-receipt", first_worker_token):
        first_resume = resume_checkpoint_calls(
            resumed_checkpoint,
            backend="gguf",
            controller=controller,
            conversation=conversation,
            execute_tool=execute_tool,
            stream_tool_execution=stream_tool_execution,
            cancel_event=None,
            tool_call_timeout=10,
            session_id="session-1",
            thread_id="thread-receipt",
            rag_scope=None,
            bypass_permissions=False,
            permission_mode="off",
            confirm_tool_calls=False,
        )
        second_end = None
        while second_end is None:
            event = next(first_resume)
            if event.get("type") == "tool_start":
                db.append_events(
                    "run-receipt",
                    first_worker_token,
                    [("chunk", event, db.now_ms())],
                )
            elif event.get("type") == "tool_end":
                second_end = event
        first_resume.close()
    assert producer_calls == ["printf second"]
    assert record_calls == ["call-2"]
    assert second_end["tool_call_id"] == "card-2"
    assert second_end["result"] == "second-result"
    public_ends = [
        event
        for event in db.list_events("run-receipt")
        if event["type"] == "chunk"
        and event["payload"].get("type") == "tool_end"
    ]
    assert [event["payload"]["tool_call_id"] for event in public_ends] == ["card-1"]

    second_rows = [
        row
        for row in db.list_ungated_tool_executions("run-receipt")
        if row["cardCallId"] == "card-2"
    ]
    assert len(second_rows) == 1
    second_row = second_rows[0]
    assert second_row["executionState"] == "finished"
    assert second_row["preToolCheckpoint"]["current_call"]["card_call_id"] == "card-2"
    assert second_row["preToolCheckpoint"]["remaining_calls"] == []
    assert second_row["preToolCheckpoint"]["tool_iters_done"] == 7
    assert second_row["preToolCheckpoint"]["iteration"] == 3
    assert "recovered_receipt" not in second_row["preToolCheckpoint"]
    assert "current_public_end_persisted" not in second_row["preToolCheckpoint"]
    state_after_second = controller.export_state()

    [second_plan] = recovery.plan_orphaned_runs()
    assert second_plan.safe is True
    assert second_plan.reason == "resume_finished_ungated_tool_receipt"
    [requeued_again] = recovery.requeue_planned_runs(
        [second_plan], stale_before_ms=db.now_ms() + 1000
    )
    second_worker = db.get_worker_run("run-receipt")
    assert second_worker is not None
    second_worker_token = second_worker[2]
    assert db.mark_running("run-receipt", second_worker_token)
    replay_controller = ToolLoopController(
        tools=[{"type": "function", "function": {"name": "terminal"}}]
    )
    replay_controller.record_result = lambda *_args, **_kwargs: pytest.fail(
        "record_result reran on the second restart"
    )
    replay_conversation = list(requeued_again["_resumeCheckpoint"]["conversation"])
    with durable_tool_run("run-receipt", second_worker_token):
        second_resume = resume_checkpoint_calls(
            requeued_again["_resumeCheckpoint"],
            backend="gguf",
            controller=replay_controller,
            conversation=replay_conversation,
            execute_tool=lambda *_args, **_kwargs: pytest.fail("producer reran"),
            stream_tool_execution=lambda *_args, **_kwargs: pytest.fail("stream reran"),
            cancel_event=None,
            tool_call_timeout=10,
            session_id="session-1",
            thread_id="thread-receipt",
            rag_scope=None,
            bypass_permissions=False,
            permission_mode="off",
            confirm_tool_calls=False,
        )
        replay_events = []
        with pytest.raises(StopIteration):
            while True:
                replay_events.append(next(second_resume))
    assert [event["type"] for event in replay_events] == ["tool_end"]
    assert replay_events[0]["tool_call_id"] == "card-2"
    assert replay_controller.export_state() == state_after_second


def _legacy_run_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE chat_generation_runs (
               id TEXT PRIMARY KEY, owner_subject TEXT NOT NULL,
               thread_id TEXT NOT NULL, status TEXT NOT NULL,
               cancel_requested INTEGER NOT NULL DEFAULT 0,
               worker_token TEXT NOT NULL, request_json TEXT NOT NULL,
               created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
               started_at INTEGER, completed_at INTEGER,
               last_event_seq INTEGER NOT NULL DEFAULT 0
           )"""
    )


def _legacy_execution_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE chat_generation_tool_executions (
               run_id TEXT NOT NULL, execution_id TEXT NOT NULL,
               backend_account_id TEXT NOT NULL, owner_subject TEXT NOT NULL,
               session_id TEXT NOT NULL DEFAULT '', thread_id TEXT NOT NULL,
               tool_name TEXT NOT NULL, tool_call_id TEXT NOT NULL DEFAULT '',
               card_call_id TEXT NOT NULL DEFAULT '', arguments_json TEXT NOT NULL,
               arguments_fingerprint TEXT NOT NULL, authority_kind TEXT NOT NULL,
               approval_id TEXT, execution_state TEXT NOT NULL,
               claim_token TEXT NOT NULL, worker_token TEXT NOT NULL,
               claimed_at INTEGER NOT NULL, started_at INTEGER, finished_at INTEGER,
               result_json TEXT, error_message TEXT, producer_receipt_json TEXT,
               completion_json TEXT, controller_is_error INTEGER,
               completion_annotations_json TEXT, post_controller_checkpoint_json TEXT,
               terminal_seq INTEGER, receipt_ref TEXT, receipt_digest TEXT,
               PRIMARY KEY(run_id, execution_id)
           ) WITHOUT ROWID"""
    )


def _temporary_connection_factory(path):
    def connect():
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    return connect


def test_existing_generic_table_migrates_checkpoint_columns_additively(
    tmp_path, monkeypatch
):
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    _legacy_run_schema(conn)
    _legacy_execution_schema(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "get_connection", _temporary_connection_factory(path))
    db._schema_ready.clear()
    migrated = db._connect()
    try:
        columns = {
            row[1]
            for row in migrated.execute(
                "PRAGMA table_info(chat_generation_tool_executions)"
            ).fetchall()
        }
    finally:
        migrated.close()
        db._schema_ready.clear()
    assert {
        "pre_tool_checkpoint_json",
        "pre_tool_checkpoint_version",
        "pre_tool_checkpoint_digest",
    } <= columns


def test_incomplete_existing_generic_table_never_marks_schema_ready(
    tmp_path, monkeypatch
):
    path = tmp_path / "incomplete.sqlite3"
    conn = sqlite3.connect(path)
    _legacy_run_schema(conn)
    conn.execute(
        "CREATE TABLE chat_generation_tool_executions (run_id TEXT, execution_id TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "get_connection", _temporary_connection_factory(path))
    db._schema_ready.clear()
    with pytest.raises(sqlite3.OperationalError):
        db._connect()
    assert path not in db._schema_ready
    assert path.resolve() not in db._schema_ready
    db._schema_ready.clear()


def _observation_capture(text: str, *, complete: bool = True, read_error=None):
    encoded = text.encode("utf-8", "surrogatepass")
    return ProcessOutputCapture(
        output=text,
        timed_out=False,
        cancelled=False,
        complete=complete,
        output_digest=hashlib.sha256(encoded).hexdigest(),
        output_byte_length=len(encoded),
        captured_byte_length=len(encoded),
        discarded_byte_length=0,
        capture_budget_chars=8 * 1024 * 1024,
        read_error=read_error,
    )


@pytest.mark.parametrize(
    "case",
    ["min_ascii", "min_multibyte", "max_ascii", "max_multibyte"],
)
def test_observation_seed_accepts_exact_utf8_byte_boundaries(case):
    if case == "min_ascii":
        text = "x" * OBSERVATION_SEED_MIN_BYTES
    elif case == "min_multibyte":
        text = "é" * (OBSERVATION_SEED_MIN_BYTES // 2)
    elif case == "max_ascii":
        text = "x" * OBSERVATION_SEED_MAX_BYTES
    else:
        text = "é" * (OBSERVATION_SEED_MAX_BYTES // 2)
    seed = observation_seed_for_text(text)
    assert isinstance(seed, ObservationSeed)
    assert seed.is_valid()
    assert seed.canonical_text_utf8_surrogatepass_byte_length == len(
        text.encode("utf-8", "surrogatepass")
    )


@pytest.mark.parametrize("text", ["x" * 9999, "é" * 4999])
def test_observation_seed_rejects_below_minimum_including_multibyte(text):
    assert observation_seed_for_text(text) is None


def test_observation_seed_rejects_multibyte_overflow():
    text = "é" * (OBSERVATION_SEED_MAX_BYTES // 2 + 1)
    assert len(text.encode("utf-8", "surrogatepass")) > OBSERVATION_SEED_MAX_BYTES
    assert observation_seed_for_text(text) is None


def test_tool_seed_is_after_sentinel_defusing_and_before_file_image_envelopes(
    monkeypatch,
):
    payload = "x" * OBSERVATION_SEED_MIN_BYTES + (
        "\n__FILES__:program text\n__IMAGES__:program text"
    )
    monkeypatch.setattr(
        tools_module,
        "_created_file_sentinels",
        lambda *_args, **_kwargs: '\n__FILES__:[{"name":"report.csv"}]\n__IMAGES__:["image"]',
    )
    result = _python_exec(
        f"print({payload!r}, end='')",
        timeout=5,
        session_id="observation-seed-envelope-test",
    )
    assert isinstance(result, ProducedToolResult)
    seed = result.observation_seed
    assert isinstance(seed, ObservationSeed)
    expected = tools_module._defuse_sentinels(payload)
    assert seed.canonical_text == expected
    assert "\n__FILES__:" not in seed.canonical_text
    assert "\n__IMAGES__:" not in seed.canonical_text
    assert "\n__FILES__:" in str(result)
    assert "\n__IMAGES__:" in str(result)


def test_observation_seed_attachment_preserves_fallback_bytes(monkeypatch):
    command = "printf '%010240d' 0"
    seeded = _bash_exec(command, timeout=5)
    assert isinstance(seeded, ProducedToolResult)
    assert seeded.observation_seed is not None
    seeded_receipt = receipt_dict(seeded)

    monkeypatch.setattr(
        tools_module,
        "observation_seed_for_text",
        lambda _text, **_kwargs: None,
    )
    unseeded = _bash_exec(command, timeout=5)
    assert str(unseeded) == str(seeded)
    unseeded_receipt = receipt_dict(unseeded)
    assert unseeded_receipt is not None and seeded_receipt is not None
    assert unseeded_receipt["fallback_result_utf8_surrogatepass_sha256"] == seeded_receipt[
        "fallback_result_utf8_surrogatepass_sha256"
    ]
    assert unseeded_receipt["fallback_result_utf8_surrogatepass_byte_length"] == seeded_receipt[
        "fallback_result_utf8_surrogatepass_byte_length"
    ]


@pytest.mark.parametrize(
    ("budget", "expected_mode"),
    [(0, "conservative_estimate"), (None, "unpriced")],
)
def test_observation_seed_latches_active_budget_and_pricing_context(
    monkeypatch, budget, expected_mode
):
    # Force the ordinary fit down the non-tokenizer branch so this test does
    # not depend on a resident model or perform a second tokenizer probe.
    monkeypatch.setattr(tools_module, "_can_measure_tokens", lambda *_args: False)
    context_token = tools_module._REQUEST_CONTEXT_TOKENS.set(4096)
    budget_token = tools_module._REQUEST_RESULT_BUDGET.set(budget)
    try:
        result = _bash_exec("printf '%010240d' 0", timeout=5)
    finally:
        tools_module._REQUEST_RESULT_BUDGET.reset(budget_token)
        tools_module._REQUEST_CONTEXT_TOKENS.reset(context_token)

    assert isinstance(result, ProducedToolResult)
    seed = result.observation_seed
    assert isinstance(seed, ObservationSeed)
    exposure = seed.result_budget_exposure
    assert isinstance(exposure, ResultBudgetExposure)
    assert exposure.active_result_budget_tokens == budget
    assert exposure.served_context_tokens == 4096
    assert exposure.pricing_mode == expected_mode

    # Later request context cannot rewrite the private evidence captured at
    # the producer fit.
    later_context = tools_module._REQUEST_CONTEXT_TOKENS.set(32768)
    try:
        assert seed.result_budget_exposure.served_context_tokens == 4096
    finally:
        tools_module._REQUEST_CONTEXT_TOKENS.reset(later_context)


def test_observation_seed_records_measured_pricing_branch_without_new_probe(
    monkeypatch,
):
    monkeypatch.setattr(tools_module, "_can_measure_tokens", lambda *_args: True)
    monkeypatch.setattr(
        tools_module,
        "_exact_prefix_chars",
        lambda text, *_args: min(len(text), 128),
    )
    context_token = tools_module._REQUEST_CONTEXT_TOKENS.set(4096)
    budget_token = tools_module._REQUEST_RESULT_BUDGET.set(256)
    try:
        result = _bash_exec("printf '%010240d' 0", timeout=5)
    finally:
        tools_module._REQUEST_RESULT_BUDGET.reset(budget_token)
        tools_module._REQUEST_CONTEXT_TOKENS.reset(context_token)

    assert isinstance(result, ProducedToolResult)
    assert result.observation_seed is not None
    assert result.observation_seed.result_budget_exposure == ResultBudgetExposure(
        active_result_budget_tokens=256,
        served_context_tokens=4096,
        pricing_mode="measured_model",
    )


def test_observation_seed_rejects_invalid_private_budget_exposure():
    text = "é" * (OBSERVATION_SEED_MIN_BYTES // 2)
    invalid = [
        ResultBudgetExposure(
            active_result_budget_tokens=1,
            served_context_tokens=4096,
            pricing_mode="not-a-pricing-branch",
        ),
        ResultBudgetExposure(
            active_result_budget_tokens=True,
            served_context_tokens=4096,
            pricing_mode="conservative_estimate",
        ),
        ResultBudgetExposure(
            active_result_budget_tokens=1,
            served_context_tokens=True,
            pricing_mode="conservative_estimate",
        ),
        ResultBudgetExposure(
            active_result_budget_tokens=1,
            served_context_tokens=None,
            pricing_mode="measured_model",
        ),
    ]
    for exposure in invalid:
        assert observation_seed_for_text(text, result_budget_exposure=exposure) is None


def test_plain_result_fit_does_not_leak_pricing_mode_into_future_seed_latch(
    monkeypatch,
):
    monkeypatch.setattr(tools_module, "_can_measure_tokens", lambda *_args: False)
    assert (
        tools_module._RESULT_BUDGET_PRICING_MODE.get()
        is tools_module._RESULT_BUDGET_PRICING_INACTIVE
    )
    tools_module._truncate("x" * 256, limit=128)
    assert (
        tools_module._RESULT_BUDGET_PRICING_MODE.get()
        is tools_module._RESULT_BUDGET_PRICING_INACTIVE
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"tool": "other"},
        {"confinement": "bypass"},
        {"outcome_kind": "nonzero_exit", "return_code": 1},
        {"outcome_kind": "timeout", "timed_out": True},
        {"outcome_kind": "cancelled", "cancelled": True},
        {"outcome_kind": "spawn_error", "spawn_error": True, "return_code": None},
        {"capture": "incomplete"},
        {"capture": "read_error"},
    ],
)
def test_produced_result_rejects_ineligible_observation_seed(mutation):
    text = "x" * OBSERVATION_SEED_MIN_BYTES
    capture = _observation_capture(text)
    if mutation.get("capture") == "incomplete":
        capture = _observation_capture(text, complete=False)
    elif mutation.get("capture") == "read_error":
        capture = _observation_capture(text, complete=False, read_error="read failed")
    kwargs = {
        "tool": "terminal",
        "workdir": None,
        "confinement": "sandboxed-owner-direct",
        "outcome_kind": "exited",
        "return_code": 0,
        "capture": capture,
        "observation_seed": observation_seed_for_text(text),
    }
    kwargs.update({key: value for key, value in mutation.items() if key != "capture"})
    result = produced_result("fallback", **kwargs)
    assert result.observation_seed is None


def test_observation_seed_is_private_and_absent_from_receipt_serialization():
    text = "x" * OBSERVATION_SEED_MIN_BYTES
    result = produced_result(
        text,
        tool="terminal",
        workdir=None,
        confinement="sandboxed-owner-direct",
        outcome_kind="exited",
        return_code=0,
        capture=_observation_capture(text),
        observation_seed=observation_seed_for_text(
            text,
            result_budget_exposure=ResultBudgetExposure(
                active_result_budget_tokens=123,
                served_context_tokens=4096,
                pricing_mode="conservative_estimate",
            ),
        ),
    )
    assert isinstance(result.observation_seed, ObservationSeed)
    receipt = receipt_dict(result)
    assert receipt is not None
    assert "observation_seed" not in receipt
    assert "result_budget_exposure" not in receipt
    assert "active_result_budget_tokens" not in receipt
    assert "canonical_text" not in receipt
    json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    assert str(result) == text


def test_prepared_result_preserves_private_seed_through_replacement():
    text = "x" * OBSERVATION_SEED_MIN_BYTES
    value = produced_result(
        text,
        tool="terminal",
        workdir=None,
        confinement="sandboxed-owner-direct",
        outcome_kind="exited",
        return_code=0,
        capture=_observation_capture(text),
        observation_seed=observation_seed_for_text(text),
    )
    execution = DurableExecutionHandle(execution_id="seed-execution")
    prepared = prepared_result(value, execution)
    replacement = replace_durable_result_value(prepared, "controller-selected")
    assert prepared.observation_seed is value.observation_seed
    assert replacement.observation_seed is value.observation_seed
    assert replacement.value == "controller-selected"


def test_replayed_prepared_result_does_not_recreate_private_seed():
    execution = DurableExecutionHandle(
        execution_id="replayed-seed-execution",
        replay_completion={"selected_result": "persisted"},
    )
    replayed = prepared_result("persisted", execution)
    assert replayed.observation_seed is None
    assert replayed.replay_completion == {"selected_result": "persisted"}
