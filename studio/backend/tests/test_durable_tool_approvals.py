# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import threading

import pytest

from core.inference import durable_agent_recovery as recovery
from core.inference.durable_agent_recovery import plan_orphaned_runs
from core.inference.durable_tool_journal import durable_tool_run
from state import tool_approvals
from storage import chat_generation_runs_db as db
from storage import studio_db
from utils.account_context import AccountContext, run_as


def _seed(run_id: str = "run-approval", owner: str = "alice") -> str:
    studio_db.upsert_chat_thread(
        {
            "id": "thread-approval",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-approval",
            "threadId": "thread-approval",
            "role": "user",
            "content": [{"type": "text", "text": "run it"}],
            "createdAt": 2,
        }
    )
    db.create_run(
        run_id=run_id,
        owner_subject=owner,
        thread_id="thread-approval",
        user_message_id="user-approval",
        assistant_message_id="assistant-approval",
        request_payload={
            "model": "local",
            "messages": [{"role": "user", "content": "run it"}],
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


def _checkpoint(command: str = "pwd", call_id: str = "call-1") -> dict:
    return {
        "version": 1,
        "backend": "gguf",
        "conversation": [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            },
        ],
        "current_call": {
            "tool_call": {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command}),
                },
            },
            "card_call_id": call_id,
            "provenance": {},
        },
        "remaining_calls": [],
        "controller": {"version": 1},
        "iteration": 0,
    }


def _proposal(
    command: str = "pwd",
    *,
    approval_id: str = "approval-1",
    call_id: str = "call-1",
    execution_id: str = "execution-1",
) -> dict:
    arguments = {"command": command}
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return {
        "approval_id": approval_id,
        "session_id": "session-1",
        "tool_name": "terminal",
        "tool_call_id": call_id,
        "card_call_id": call_id,
        "execution_id": execution_id,
        "arguments": arguments,
        "arguments_fingerprint": hashlib.sha256(encoded.encode()).hexdigest(),
        "checkpoint_version": 1,
        "resume_checkpoint": _checkpoint(command, call_id),
        "proposed_at": 100,
        "expires_at": 10_000_000_000_000,
    }


def _persist(token: str, proposal: dict | None = None) -> None:
    proposal = proposal or _proposal()
    db.append_events_with_tool_proposal(
        "run-approval",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": proposal["tool_name"],
                    "tool_call_id": proposal["card_call_id"],
                    "arguments": proposal["arguments"],
                    "arguments_text": json.dumps(proposal["arguments"]),
                    "approval_id": proposal["approval_id"],
                    "awaiting_confirmation": True,
                },
                100,
            )
        ],
        proposal,
    )


def _append_tool_end(token: str, call_id: str, result: str) -> None:
    db.append_events(
        "run-approval",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "terminal",
                    "tool_call_id": call_id,
                    "result": result,
                },
            )
        ],
    )


def _finish_approval(
    token: str,
    *,
    approval_id: str = "approval-1",
    result: str = "/workspace",
) -> None:
    db.decide_tool_approval(
        "run-approval",
        approval_id,
        owner_subject="alice",
        session_id="session-1",
        decision="allow",
    )
    claimed = db.claim_tool_execution(
        "run-approval", approval_id, worker_token=token
    )
    assert claimed is not None
    assert db.mark_tool_execution_started(
        "run-approval",
        approval_id,
        worker_token=token,
        claim_token=claimed["claimToken"],
    ) is not None
    assert db.finish_tool_execution(
        "run-approval",
        approval_id,
        worker_token=token,
        claim_token=claimed["claimToken"],
        result=result,
        controller_is_error=False,
        completion_annotations={},
        post_controller_checkpoint={"version": 1},
    ) is not None


def test_proposal_and_public_start_commit_together_and_snapshot_is_display_safe():
    token = _seed()
    _persist(token)
    events = db.list_events("run-approval")
    assert [event["type"] for event in events][-2:] == ["chunk", "approval.proposed"]
    run = db.get_run("run-approval")
    assert run is not None
    [approval] = run["pendingApprovals"]
    assert approval["approvalId"] == "approval-1"
    assert approval["arguments"] == {"command": "pwd"}
    assert "resumeCheckpoint" not in approval
    assert db.get_tool_approval("run-approval", "approval-1")["resumeCheckpoint"]


def test_proposal_fingerprint_mismatch_rolls_back_public_event():
    token = _seed()
    proposal = _proposal()
    proposal["arguments_fingerprint"] = "0" * 64
    before = db.get_run("run-approval")["lastEventSeq"]
    with pytest.raises(db.ToolApprovalFencedError):
        _persist(token, proposal)
    assert db.get_run("run-approval")["lastEventSeq"] == before
    assert db.get_tool_approval("run-approval", "approval-1") is None


def test_checkpoint_call_mismatch_rolls_back_public_event():
    token = _seed()
    proposal = _proposal()
    proposal["resume_checkpoint"]["current_call"]["tool_call"]["function"][
        "arguments"
    ] = json.dumps({"command": "whoami"})
    before = db.get_run("run-approval")["lastEventSeq"]
    with pytest.raises(db.ToolApprovalFencedError):
        _persist(token, proposal)
    assert db.get_run("run-approval")["lastEventSeq"] == before
    assert db.get_tool_approval("run-approval", "approval-1") is None


def test_exact_run_and_approval_ids_are_account_scoped():
    alice = AccountContext("a" * 32, "alice")
    bob = AccountContext("b" * 32, "bob")
    alice_token = run_as(alice, _seed)
    bob_token = run_as(bob, _seed)
    run_as(alice, _persist, alice_token)
    run_as(bob, _persist, bob_token)
    run_as(
        alice,
        db.decide_tool_approval,
        "run-approval",
        "approval-1",
        owner_subject="alice",
        session_id="session-1",
        decision="allow",
    )
    run_as(
        bob,
        db.decide_tool_approval,
        "run-approval",
        "approval-1",
        owner_subject="alice",
        session_id="session-1",
        decision="deny",
    )
    assert run_as(alice, db.get_tool_approval, "run-approval", "approval-1")[
        "decision"
    ] == "allow"
    assert run_as(bob, db.get_tool_approval, "run-approval", "approval-1")[
        "decision"
    ] == "deny"


def test_concurrent_opposite_decisions_have_one_winner_and_same_retry_is_idempotent():
    token = _seed()
    _persist(token)

    def decide(value: str):
        try:
            return db.decide_tool_approval(
                "run-approval",
                "approval-1",
                owner_subject="alice",
                session_id="session-1",
                decision=value,
            )["decision"]
        except db.ToolApprovalConflictError:
            return "conflict"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, ["allow", "deny"]))
    assert sorted(results) in (["allow", "conflict"], ["conflict", "deny"])
    winner = next(value for value in results if value != "conflict")
    assert decide(winner) == winner
    assert len([e for e in db.list_events("run-approval") if e["type"] == "approval.decided"]) == 1


def test_decision_committed_without_notification_is_observed_before_wait():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    with durable_tool_run("run-approval", token):
        slot = tool_approvals.begin_tool_decision("session-1", "approval-1")
        assert tool_approvals.wait_tool_decision(slot, "approval-1", timeout=0.01) == "allow"


def test_shutdown_detaches_without_denial_or_expiry_change():
    token = _seed()
    _persist(token)
    before = db.get_tool_approval("run-approval", "approval-1")
    stopped = threading.Event()
    stopped.set()
    with durable_tool_run("run-approval", token):
        slot = tool_approvals.begin_tool_decision("session-1", "approval-1")
        with pytest.raises(tool_approvals.ToolApprovalDetached):
            tool_approvals.wait_tool_decision(
                slot, "approval-1", cancel_event=stopped, timeout=0.01
            )
    after = db.get_tool_approval("run-approval", "approval-1")
    assert after["status"] == "pending"
    assert after["expiresAt"] == before["expiresAt"]


def test_claim_and_start_race_admits_exactly_one_start(monkeypatch):
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    monkeypatch.setattr(db, "now_ms", lambda: 123456)

    def claim():
        return db.claim_tool_execution(
            "run-approval", "approval-1", worker_token=token
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _value: claim(), range(2)))
    assert all(item is not None for item in claims)
    assert len({item["claimToken"] for item in claims}) == 1

    def start(claimed):
        return db.mark_tool_execution_started(
            "run-approval",
            "approval-1",
            worker_token=token,
            claim_token=claimed["claimToken"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        starts = list(pool.map(start, claims))
    assert sum(item is not None for item in starts) == 1
    event_types = [event["type"] for event in db.list_events("run-approval")]
    assert event_types.count("tool_execution.claimed") == 1
    assert event_types.count("tool_execution.started") == 1


def test_claim_without_start_is_reclaimed_only_by_fenced_takeover():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    claimed = db.claim_tool_execution("run-approval", "approval-1", worker_token=token)
    execution_id = claimed["executionId"]
    [plan] = plan_orphaned_runs()
    from core.inference.durable_agent_recovery import requeue_planned_runs

    [recovered] = requeue_planned_runs(
        [plan], stale_before_ms=plan.expected_progress_at
    )
    new_token = recovered["_workerToken"]
    reset = db.get_tool_approval("run-approval", "approval-1")
    assert reset["executionState"] == "unclaimed"
    assert reset["executionId"] == execution_id
    assert db.mark_tool_execution_started(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
    ) is None
    reclaimed = db.claim_tool_execution(
        "run-approval", "approval-1", worker_token=new_token
    )
    assert reclaimed["executionId"] == execution_id


def test_denial_without_tool_end_recovers_as_replay_only():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="deny"
    )
    [plan] = plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.resume_checkpoint["recovered_receipt"] == {"state": "denied"}


def test_commit_failure_never_notifies_or_resolves(monkeypatch):
    token = _seed()
    _persist(token)
    notified: list[tuple[str, str]] = []

    def fail_commit(*_args, **_kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(db, "decide_tool_approval", fail_commit)
    monkeypatch.setattr(
        tool_approvals,
        "notify_tool_decision",
        lambda run_id, approval_id: notified.append((run_id, approval_id)),
    )
    with pytest.raises(OSError, match="database unavailable"):
        tool_approvals.resolve_tool_decision(
            "approval-1",
            "allow",
            "session-1",
            run_id="run-approval",
            owner_subject="alice",
        )
    assert notified == []
    assert db.get_tool_approval("run-approval", "approval-1")["status"] == "pending"


def test_wrong_owner_and_session_are_indistinguishable_from_unknown():
    token = _seed()
    _persist(token)
    with pytest.raises(KeyError):
        db.decide_tool_approval(
            "run-approval", "approval-1", owner_subject="bob", decision="allow"
        )
    with pytest.raises(KeyError):
        db.decide_tool_approval(
            "run-approval",
            "approval-1",
            owner_subject="alice",
            session_id="other",
            decision="allow",
        )
    with pytest.raises(KeyError):
        db.decide_tool_approval(
            "run-approval", "approval-1", owner_subject="alice", decision="allow"
        )


def test_stop_wins_before_execution_start_even_after_allow():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    claimed = db.claim_tool_execution("run-approval", "approval-1", worker_token=token)
    assert claimed is not None
    db.request_cancel("run-approval", owner_subject="alice")
    assert (
        db.mark_tool_execution_started(
            "run-approval",
            "approval-1",
            worker_token=token,
            claim_token=claimed["claimToken"],
        )
        is None
    )


def test_finished_receipt_is_recovery_replay_and_started_without_finish_fails_closed():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    claimed = db.claim_tool_execution("run-approval", "approval-1", worker_token=token)
    started = db.mark_tool_execution_started(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
    )
    assert started is not None
    [ambiguous] = plan_orphaned_runs()
    assert ambiguous.safe is False
    assert ambiguous.reason == "tool_effect_started_without_durable_finish"
    db.finish_tool_execution(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="/workspace",
        controller_is_error=False,
        completion_annotations={},
        post_controller_checkpoint={"version": 1},
    )
    [finished] = plan_orphaned_runs()
    assert finished.safe is True
    assert finished.action == "resume_tool"
    receipt = finished.resume_checkpoint["recovered_receipt"]
    assert receipt["state"] == "finished"
    assert receipt["result"] == "/workspace"
    assert receipt["error"] is None
    assert receipt["completion"]["post_controller_checkpoint"] == {"version": 1}


def test_approved_terminal_retry_replays_stored_completion_and_fences_authority():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval",
        "approval-1",
        owner_subject="alice",
        session_id="session-1",
        decision="allow",
    )
    claimed = db.claim_tool_execution("run-approval", "approval-1", worker_token=token)
    assert claimed is not None
    assert db.mark_tool_execution_started(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
    ) is not None
    finished = db.finish_tool_execution(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="committed-result",
        producer_receipt={"candidate": "committed"},
        controller_is_error=False,
        completion_annotations={"committed": True},
        post_controller_checkpoint={"version": 1},
    )
    assert finished is not None
    assert (
        db.finish_tool_execution(
            "run-approval",
            "approval-1",
            worker_token="wrong-worker",
            claim_token=claimed["claimToken"],
            result="attacker-result",
        )
        is None
    )
    assert (
        db.finish_tool_execution(
            "run-approval",
            "approval-1",
            worker_token=token,
            claim_token="wrong-claim",
            result="attacker-result",
        )
        is None
    )
    replay = db.finish_tool_execution(
        "run-approval",
        "approval-1",
        worker_token=token,
        claim_token=claimed["claimToken"],
        result="different",
        error="stale optional error",
        ambiguous=True,
        producer_receipt={"candidate": "different"},
        controller_is_error=True,
        completion_annotations={"changed": True},
        post_controller_checkpoint={"version": 99},
    )
    assert replay is not None
    assert replay["receiptDigest"] == finished["receiptDigest"]
    assert replay["result"] == "committed-result"
    assert replay["error"] is None
    assert replay["executionState"] == "finished"
    terminal = [
        event
        for event in db.list_events("run-approval")
        if event["type"] == "tool_execution.finished"
    ]
    assert len(terminal) == 1


def test_public_tool_end_then_trailing_partial_fails_closed_without_checkpoint():
    token = _seed()
    _persist(token)
    _finish_approval(token)
    _append_tool_end(token, "call-1", "/workspace")
    db.append_events(
        "run-approval",
        token,
        [
            (
                "chunk",
                {"choices": [{"delta": {"content": "Trailing partial"}}]},
            )
        ],
    )

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.action == "fail_closed"
    assert plan.reason == "progress_after_published_tool_completion"


def test_two_publicly_closed_finished_approvals_resume_after_both_results():
    token = _seed()
    _persist(token)
    _finish_approval(token, result="first")
    _append_tool_end(token, "call-1", "first")

    second = _proposal(
        "echo second",
        approval_id="approval-2",
        call_id="call-2",
        execution_id="execution-2",
    )
    _persist(token, second)
    _finish_approval(token, approval_id="approval-2", result="second")
    _append_tool_end(token, "call-2", "second")

    [plan] = plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.reason == "resume_after_published_approved_completion"
    assert plan.approval_id == "approval-2"
    assert plan.resume_checkpoint["recovered_receipt"]["result"] == "second"
    assert plan.resume_checkpoint["current_public_end_persisted"] is True


def test_publicly_closed_denial_resumes_model_but_open_denial_resumes_tool():
    token = _seed()
    _persist(token)
    db.decide_tool_approval(
        "run-approval",
        "approval-1",
        owner_subject="alice",
        session_id="session-1",
        decision="deny",
    )
    [open_plan] = plan_orphaned_runs()
    assert open_plan.action == "resume_tool"
    assert open_plan.resume_checkpoint["recovered_receipt"] == {"state": "denied"}

    _append_tool_end(token, "call-1", tool_approvals.TOOL_REJECTED_MESSAGE)
    [closed_plan] = plan_orphaned_runs()
    assert closed_plan.safe is True
    assert closed_plan.action == "resume_model"
    assert closed_plan.reason == "resume_after_durable_tool_result"
    assert closed_plan.resume_checkpoint is None


def test_cancelled_open_approval_is_never_a_checkpoint_candidate():
    token = _seed()
    _persist(token)
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET execution_state='cancelled'
               WHERE run_id='run-approval' AND approval_id='approval-1'"""
        )
        conn.commit()
    finally:
        conn.close()

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_approval_cancelled_while_public_span_open"
    assert plan.resume_checkpoint is None


def test_duplicate_public_approval_start_fails_closed():
    token = _seed()
    _persist(token)
    db.append_events(
        "run-approval",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "terminal",
                    "tool_call_id": "call-1",
                    "approval_id": "approval-1",
                    "awaiting_confirmation": True,
                },
            )
        ],
    )

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_approval_public_span_is_ambiguous"


@pytest.mark.parametrize(
    "tool_name,tool_call_id",
    [("wrong_tool", "call-1"), ("terminal", "wrong-call")],
)
def test_unmatched_public_end_while_approval_open_fails_closed(
    tool_name, tool_call_id
):
    token = _seed()
    _persist(token)
    db.append_events(
        "run-approval",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "result": "wrong span",
                },
            )
        ],
    )

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_approval_public_end_mismatch"


def test_nested_ungated_span_makes_mixed_authority_frontier_ambiguous():
    token = _seed()
    _persist(token)
    db.append_events(
        "run-approval",
        token,
        [
            (
                "chunk",
                {
                    "type": "tool_start",
                    "tool_name": "inference_performance",
                    "tool_call_id": "ungated-1",
                    "approval_id": "",
                    "awaiting_confirmation": False,
                },
            ),
            (
                "chunk",
                {
                    "type": "tool_end",
                    "tool_name": "inference_performance",
                    "tool_call_id": "ungated-1",
                    "result": "fast",
                },
            ),
        ],
    )

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "mixed_tool_authority_frontier_is_ambiguous"


def test_recovery_reads_tool_end_and_trailing_output_beyond_100k_events():
    token = _seed()
    _persist(token)
    db.append_events("run-approval", token, [("noop", {})] * 100_000)
    _finish_approval(token)
    _append_tool_end(token, "call-1", "/workspace")
    db.append_events(
        "run-approval",
        token,
        [("chunk", {"choices": [{"delta": {"content": "after cap"}}]})],
    )
    assert db.get_run("run-approval")["lastEventSeq"] < recovery._RECOVERY_MAX_EVENT_COUNT

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "progress_after_published_tool_completion"


def test_recovery_event_count_ceiling_fails_closed(monkeypatch):
    token = _seed()
    _persist(token)
    before = db.get_run("run-approval")["lastEventSeq"]
    db.append_events("run-approval", token, [("noop", {})] * 3)
    monkeypatch.setattr(recovery, "_RECOVERY_MAX_EVENT_COUNT", before + 2)

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "recovery_event_count_limit_exceeded"


@pytest.mark.parametrize("payload_sizes", [[1_200], [450, 450, 450]])
def test_recovery_serialized_byte_ceiling_fails_closed(monkeypatch, payload_sizes):
    token = _seed()
    _persist(token)
    existing = db.list_events("run-approval", after=0, limit=100)
    existing_bytes = sum(
        len(
            json.dumps(
                event,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        for event in existing
    )
    db.append_events(
        "run-approval",
        token,
        [("oversized", {"blob": "x" * size}) for size in payload_sizes],
    )
    monkeypatch.setattr(
        recovery, "_RECOVERY_MAX_SERIALIZED_BYTES", existing_bytes + 1_000
    )

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "recovery_event_byte_limit_exceeded"


@pytest.mark.parametrize("missing", ["middle", "tail"])
def test_missing_or_gapped_recovery_event_snapshot_fails_closed(missing):
    token = _seed()
    _persist(token)
    db.append_events(
        "run-approval",
        token,
        [("noop", {}), ("noop", {}), ("noop", {})],
    )
    run = db.get_run("run-approval")
    missing_seq = 2 if missing == "middle" else run["lastEventSeq"]
    conn = db._connect()
    try:
        conn.execute(
            "DELETE FROM chat_generation_events WHERE run_id=? AND seq=?",
            ("run-approval", missing_seq),
        )
        conn.commit()
    finally:
        conn.close()

    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "recovery_event_log_incomplete"


def test_closed_modern_checkpoint_with_remaining_calls_continues_siblings():
    token = _seed()
    proposal = _proposal()
    proposal["resume_checkpoint"]["remaining_calls"] = [
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"command":"echo sibling"}'},
        }
    ]
    _persist(token, proposal)
    _finish_approval(token)
    _append_tool_end(token, "call-1", "/workspace")

    [plan] = plan_orphaned_runs()
    assert plan.safe is True
    assert plan.action == "resume_tool"
    assert plan.reason == "resume_after_published_approved_completion"
    assert plan.resume_checkpoint["remaining_calls"][0]["id"] == "call-2"


def test_corrupt_or_old_checkpoint_fails_closed():
    token = _seed()
    _persist(token)
    checkpoint = _checkpoint()
    checkpoint["version"] = 0
    conn = db._connect()
    try:
        conn.execute(
            """UPDATE chat_generation_tool_approvals
               SET checkpoint_version=0, resume_checkpoint_json=?
               WHERE run_id='run-approval' AND approval_id='approval-1'""",
            (json.dumps(checkpoint),),
        )
        conn.commit()
    finally:
        conn.close()
    [plan] = plan_orphaned_runs()
    assert plan.safe is False
    assert plan.reason == "tool_checkpoint_missing_or_incompatible"


def test_safetensors_resume_executes_exact_call_before_any_model_sample():
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    token = _seed()
    proposal = _proposal()
    proposal["resume_checkpoint"]["backend"] = "safetensors"
    proposal["resume_checkpoint"]["remaining_calls"] = [
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "inference_performance", "arguments": "{}"},
        }
    ]
    _persist(token, proposal)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    checkpoint = db.get_tool_approval("run-approval", "approval-1")["resumeCheckpoint"]
    checkpoint["recovery_approval_id"] = "approval-1"
    order: list[str] = []

    def single_turn(_messages):
        order.append("model")
        yield "done"

    def execute(name, _arguments, **_kwargs):
        order.append(f"tool:{name}")
        return "/workspace" if name == "terminal" else "fast"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inference_performance",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    with durable_tool_run("run-approval", token, resume_checkpoint=checkpoint):
        list(
            run_safetensors_tool_loop(
                single_turn=single_turn,
                messages=[{"role": "user", "content": "run it"}],
                tools=tools,
                execute_tool=execute,
                confirm_tool_calls=True,
                permission_mode="auto",
                max_tool_iterations=2,
                session_id="session-1",
            )
        )
    assert order[:3] == ["tool:terminal", "tool:inference_performance", "model"]
    approval = db.get_tool_approval("run-approval", "approval-1")
    assert approval["executionState"] == "finished"


def test_gguf_resume_executes_exact_call_before_any_model_sample(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    token = _seed()
    proposal = _proposal()
    proposal["resume_checkpoint"]["remaining_calls"] = [
        {
            "id": "call-2",
            "type": "function",
            "function": {"name": "inference_performance", "arguments": "{}"},
        }
    ]
    _persist(token, proposal)
    db.decide_tool_approval(
        "run-approval", "approval-1", owner_subject="alice", session_id="session-1", decision="allow"
    )
    checkpoint = db.get_tool_approval("run-approval", "approval-1")["resumeCheckpoint"]
    checkpoint["recovery_approval_id"] = "approval-1"
    order: list[str] = []

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = None
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream(_client, _url, _payload, _cancel, **_kwargs):
        order.append("model")
        chunks = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {"content": "done"}, "finish_reason": "stop"}
                    ]
                }
            )
            + "\n",
            "data: [DONE]\n",
        ]
        yield type("Response", (), {"status_code": 200, "chunks": chunks})()

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream)
    monkeypatch.setattr(
        backend, "_iter_text_cancellable", lambda response, *_a, **_k: iter(response.chunks)
    )
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)

    def execute(name, _arguments, **_kwargs):
        order.append(f"tool:{name}")
        return "/workspace" if name == "terminal" else "fast"

    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inference_performance",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    with durable_tool_run("run-approval", token, resume_checkpoint=checkpoint):
        list(
            backend.generate_chat_completion_with_tools(
                messages=[{"role": "user", "content": "run it"}],
                tools=tools,
                confirm_tool_calls=True,
                permission_mode="auto",
                max_tool_iterations=2,
                session_id="session-1",
            )
        )
    assert order[:3] == ["tool:terminal", "tool:inference_performance", "model"]
    approval = db.get_tool_approval("run-approval", "approval-1")
    assert approval["executionState"] == "finished"
