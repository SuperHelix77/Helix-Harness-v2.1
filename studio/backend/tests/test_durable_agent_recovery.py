# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json

import pytest

from core.inference import durable_agent_recovery as recovery
from core.inference.durable_agent_recovery import build_restart_request


def _request(run_id: str = "run-1") -> dict:
    return {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Inspect and fix it"}],
        "stream": True,
        "cancel_id": run_id,
        "generation_run_id": run_id,
        "finalization_idempotency_key": run_id,
    }


def _event(seq: int, event_type: str, payload: dict) -> dict:
    return {"seq": seq, "type": event_type, "payload": payload, "createdAt": seq}


def _chunk(seq: int, payload: dict) -> dict:
    return _event(seq, "chunk", payload)


def _tool_start(*, awaiting_confirmation: bool = False) -> dict:
    return {
        "type": "tool_start",
        "tool_name": "terminal",
        "tool_call_id": "call-1",
        "arguments": {"command": "pwd"},
        "arguments_text": '{"command":"pwd"}',
        "approval_id": "approval-1" if awaiting_confirmation else "",
        "awaiting_confirmation": awaiting_confirmation,
    }


def test_restart_replays_finished_effect_without_reexecuting_it():
    events = [
        _chunk(1, {"choices": [{"delta": {"content": "Checking. "}}]}),
        _chunk(2, _tool_start()),
        _event(
            3,
            "tool_execution.started",
            {
                "execution_id": "exec-1",
                "tool_name": "terminal",
                "tool_call_id": "call-1",
                "effect_state": "started",
            },
        ),
        _event(
            4,
            "tool_execution.finished",
            {
                "execution_id": "exec-1",
                "tool_name": "terminal",
                "tool_call_id": "call-1",
                "effect_state": "finished",
                "result": "/workspace",
            },
        ),
        # Crash here: the effect receipt is durable but tool_end/model continuation is not.
    ]
    plan = build_restart_request(_request(), events)
    assert plan.safe is True
    assert plan.reason == "resume_after_durable_tool_result"
    assert plan.request_payload is not None
    messages = plan.request_payload["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"] == "Checking. "
    assert messages[-2]["tool_calls"][0]["id"] == "call-1"
    assert messages[-1] == {
        "role": "tool",
        "name": "terminal",
        "content": "/workspace",
        "tool_call_id": "call-1",
    }
    assert plan.request_payload.get("continue_final_message") is None


def test_restart_never_reexecutes_effect_with_started_but_no_finish_receipt():
    events = [
        _chunk(1, _tool_start()),
        _event(
            2,
            "tool_execution.started",
            {
                "execution_id": "exec-1",
                "tool_name": "terminal",
                "tool_call_id": "call-1",
                "effect_state": "started",
            },
        ),
    ]
    plan = build_restart_request(_request(), events)
    assert plan.safe is False
    assert plan.reason == "tool_effect_started_without_durable_finish"
    assert plan.request_payload is None


def test_restart_can_replay_durable_denial_without_executing_tool():
    events = [
        _chunk(1, _tool_start(awaiting_confirmation=True)),
        _event(
            2,
            "approval.decided",
            {
                "tool_name": "terminal",
                "tool_call_id": "call-1",
                "approval_id": "approval-1",
                "decision": "deny",
            },
        ),
    ]
    plan = build_restart_request(_request(), events)
    assert plan.safe is True
    assert plan.request_payload is not None
    assert plan.request_payload["messages"][-1]["role"] == "tool"
    assert "declined" in plan.request_payload["messages"][-1]["content"].lower()


def test_restart_does_not_guess_an_unanswered_approval():
    plan = build_restart_request(
        _request(),
        [_chunk(1, _tool_start(awaiting_confirmation=True))],
    )
    assert plan.safe is False
    assert plan.reason == "tool_approval_interrupted_before_decision"


def test_plain_partial_text_resumes_token_exactly_without_new_tool_effects():
    plan = build_restart_request(
        _request(),
        [
            _chunk(1, {"choices": [{"delta": {"content": "Part one"}}]}),
            _chunk(2, {"choices": [{"delta": {"content": " and two"}}]}),
        ],
    )
    assert plan.safe is True
    assert plan.reason == "resume_trailing_assistant_partial"
    assert plan.request_payload is not None
    assert plan.request_payload["messages"][-1] == {
        "role": "assistant",
        "content": "Part one and two",
    }
    assert plan.request_payload["continue_final_message"] is True


def test_pre_v3_run_is_never_recovered_under_invented_identity():
    request = _request()
    request.pop("finalization_idempotency_key")
    plan = build_restart_request(request, [])
    assert plan.safe is False
    assert plan.reason == "not_a_v3_durable_turn"


@pytest.mark.parametrize(
    "extra_operational,expected_error",
    [(0, None), (1, "recovery_event_count_limit_exceeded")],
)
def test_private_trace_does_not_consume_exact_operational_event_ceiling(
    monkeypatch, extra_operational, expected_error
):
    operational_count = recovery._RECOVERY_MAX_EVENT_COUNT + extra_operational
    trace_sequence = operational_count + 1

    def list_events(_run_id, *, after, limit):
        upper = min(trace_sequence, after + limit)
        events = []
        for sequence in range(after + 1, upper + 1):
            if sequence == trace_sequence:
                events.append(
                    _event(sequence, "optimization.trace", {"trace": "private"})
                )
            else:
                events.append(_event(sequence, "noop", {}))
        return events

    monkeypatch.setattr(recovery.runs_db, "list_events", list_events)
    events, error = recovery._load_exact_event_snapshot("run-1", trace_sequence)

    assert error == expected_error
    if expected_error is None:
        assert events is not None
        assert len(events) == recovery._RECOVERY_MAX_EVENT_COUNT
        assert all(event["type"] != "optimization.trace" for event in events)
    else:
        assert events is None


@pytest.mark.parametrize(
    "extra_byte,expected_error",
    [(0, None), (1, "recovery_event_byte_limit_exceeded")],
)
def test_private_trace_does_not_consume_exact_operational_byte_ceiling(
    monkeypatch, extra_byte, expected_error
):
    ordinary = _event(1, "noop", {"blob": ""})
    encoded_empty = len(
        json.dumps(
            ordinary,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    ordinary["payload"]["blob"] = "x" * (
        recovery._RECOVERY_MAX_SERIALIZED_BYTES - encoded_empty + extra_byte
    )
    private = _event(2, "optimization.trace", {"trace": "private"})

    def list_events(_run_id, *, after, limit):
        return [ordinary, private][after : after + limit]

    monkeypatch.setattr(recovery.runs_db, "list_events", list_events)
    events, error = recovery._load_exact_event_snapshot("run-1", 2)

    assert error == expected_error
    if expected_error is None:
        assert events == [ordinary]
    else:
        assert events is None


def test_private_trace_has_independent_fail_closed_recovery_bound(monkeypatch):
    private = _event(1, "optimization.trace", {"trace": "x" * 200})
    encoded = len(
        json.dumps(
            private,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    monkeypatch.setattr(recovery, "_RECOVERY_MAX_PRIVATE_SERIALIZED_BYTES", encoded - 1)
    monkeypatch.setattr(
        recovery.runs_db,
        "list_events",
        lambda _run_id, *, after, limit: [private][after : after + limit],
    )

    events, error = recovery._load_exact_event_snapshot("run-1", 1)

    assert events is None
    assert error == "recovery_private_event_byte_limit_exceeded"


def test_snapshot_without_private_evidence_preserves_ordinary_recovery(monkeypatch):
    ordinary = _chunk(1, {"choices": [{"delta": {"content": "partial"}}]})
    monkeypatch.setattr(
        recovery.runs_db,
        "list_events",
        lambda _run_id, *, after, limit: [ordinary][after : after + limit],
    )

    events, error = recovery._load_exact_event_snapshot("run-1", 1)
    plan = build_restart_request(_request(), events or [])

    assert error is None
    assert plan.safe is True
    assert plan.reason == "resume_trailing_assistant_partial"
