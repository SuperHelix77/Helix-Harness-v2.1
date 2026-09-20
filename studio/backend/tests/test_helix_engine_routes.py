# SPDX-License-Identifier: AGPL-3.0-only
from routes.helix_engine import (
    CorrectionIn,
    IngestTurnIn,
    ToolStepIn,
    TrajectoryIn,
    analyze_trajectory,
    prepare_completed_turn_audit,
    session_trace,
)


def test_session_trace_preserves_latest_turn_id_for_provenance_lookup(tmp_path, monkeypatch):
    from core.helix_engine.capture import (
        archive_session,
        capture_session_key,
        clear_session,
        record_tool_execution,
    )
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    session_id = "graph-session"
    thread_id = "graph-thread"
    turn_id = "graph-turn"
    key = capture_session_key(session_id, thread_id, turn_id)
    clear_session(key)
    record_tool_execution(key, "search_memory", {"query": "x"}, "memory hit")
    archive_session(key)
    result = session_trace(session_id, thread_id=thread_id)

    assert result["turn_id"] == turn_id
    assert result["steps"][0]["name"] == "search_memory"
    assert result["steps"][0]["sequence"] > 0
    assert result["steps"][0]["created_at_ms"] > 0


def test_session_trace_never_labels_archived_events_as_a_newer_live_turn():
    from core.helix_engine.capture import (
        archive_session,
        capture_session_key,
        clear_session,
        record_tool_control_event,
        record_tool_execution,
    )

    session_id = "graph-correlation-session"
    thread_id = "graph-correlation-thread"
    base = capture_session_key(session_id, thread_id)
    first_key = capture_session_key(session_id, thread_id, "turn-one")
    second_key = capture_session_key(session_id, thread_id, "turn-two")
    clear_session(base)

    record_tool_execution(first_key, "read_file", {"path": "old.py"}, "old")
    record_tool_control_event(
        first_key,
        action="skip",
        tool_name="read_file",
        arguments={"path": "old.py"},
        reason="old duplicate",
    )
    archive_session(first_key)
    record_tool_execution(second_key, "run_tests", {"cmd": "pytest"}, "new")

    result = session_trace(session_id, thread_id=thread_id)

    assert result["turn_id"] == "turn-two"
    assert [step["name"] for step in result["steps"]] == ["run_tests"]
    assert result["control_events"] == []


def test_session_trace_advances_to_completed_empty_turn_instead_of_reusing_prior_trace():
    from core.helix_engine.capture import (
        archive_session,
        capture_session_key,
        clear_session,
        record_tool_execution,
    )

    session_id = "graph-empty-session"
    thread_id = "graph-empty-thread"
    base = capture_session_key(session_id, thread_id)
    first_key = capture_session_key(session_id, thread_id, "turn-with-tool")
    empty_key = capture_session_key(session_id, thread_id, "turn-without-tools")
    clear_session(base)

    record_tool_execution(first_key, "read_file", {"path": "old.py"}, "old")
    archive_session(first_key)
    archive_session(empty_key)

    result = session_trace(session_id, thread_id=thread_id)

    assert result["turn_id"] == "turn-without-tools"
    assert result["steps"] == []
    assert result["control_events"] == []


def test_analyze_route_does_not_promote_qlora_on_success_alone():
    payload = TrajectoryIn(
        prompt_state="fix helper",
        retrieved_context="tests",
        steps=[
            ToolStepIn(name="read_file", arguments="a.py", result="ok", useful_hint="evidence"),
            ToolStepIn(name="read_file", arguments="a.py", result="ok", useful_hint="redundant"),
            ToolStepIn(name="search_repo", arguments="a.py", result="same", useful_hint="redundant"),
            ToolStepIn(name="edit_file", arguments="bad", result="broke", useful_hint="wrong", error="fail"),
            ToolStepIn(name="grep", arguments="x", result="hit", useful_hint="useful"),
            ToolStepIn(name="run_tests", arguments="pytest", result="FAIL", useful_hint="useful"),
            ToolStepIn(name="read_file", arguments="b.py", result="ok", useful_hint="evidence"),
            ToolStepIn(name="edit_file", arguments="good", result="pass", useful_hint="useful"),
        ],
        user_corrections=[
            CorrectionIn(
                state="failed",
                bad_action="edit_file bad",
                failure_evidence="FAIL",
                correct_action="edit_file good",
                why="user correction",
            )
        ],
        final_result="tests pass",
        verified=True,
        allow_qlora=False,
    )
    result = analyze_trajectory(payload)
    assert result["success_authorizes_weight_update"] is False
    assert result["route"] == "hermes"
    assert result["compressed"]["tool_calls"] < 8
    assert result["corrections"][0]["correct_action"] == "edit_file good"
    assert result["product"] == "Helix Harness"
    assert result["decision"] == "promote_hermes"
    assert result["gates"][-1] == "promote"


def test_prepare_audit_route_fails_open_to_historical_deep_audit(monkeypatch):
    from core.helix_engine import audit

    monkeypatch.setattr(
        audit,
        "prepare_observable_self_audit",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("optional audit preparation died")),
    )
    result = prepare_completed_turn_audit(
        IngestTurnIn(
            session_id="s",
            turn_id="t",
            prompt="hello",
            final_result="hi",
            telemetry={"prompt_tokens": 3},
        )
    )
    assert result["available"] is False
    assert result["perform_deep_audit"] is True
    assert result["fail_open"] is True


def test_prepare_audit_discovers_visible_high_impact_claim_without_client_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    result = prepare_completed_turn_audit(
        IngestTurnIn(
            session_id="claim-session",
            turn_id="claim-turn",
            prompt="report the result",
            final_result="Helix is at least 5x faster.",
            telemetry={"prompt_tokens": 10, "cached_tokens": 4},
        )
    )
    assert result["available"] is True
    assert result["perform_deep_audit"] is True
    assert "high_impact_claim" in result["artifacts"]["deep_audit_forced_reasons"]
    speed = next(item for item in result["artifacts"]["evidence"] if "5x" in item["claim"])
    assert speed["status"] == "UNVERIFIED"
    assert any("benchmark" in item for item in speed["missing_evidence"])


def test_prepare_audit_ignores_client_objective_verified_telemetry(monkeypatch):
    from core.helix_engine import audit

    captured = {}

    def inspect(traj, *, claims=None):
        captured["verified"] = traj.verified
        captured["objective_verified"] = traj.extras.get("objective_verified")
        captured["telemetry"] = dict(traj.extras.get("telemetry") or {})
        return {"available": True, "perform_deep_audit": True, "artifacts": {}}

    monkeypatch.setattr(audit, "prepare_observable_self_audit", inspect)

    result = prepare_completed_turn_audit(
        IngestTurnIn(
            session_id="client-proof-session",
            turn_id="client-proof-turn",
            prompt="claim success",
            final_result="done",
            telemetry={"objective_verified": True, "prompt_tokens": 10},
        )
    )

    assert result["available"] is True
    assert captured["verified"] is False
    assert captured["objective_verified"] is False
    assert "objective_verified" not in captured["telemetry"]
