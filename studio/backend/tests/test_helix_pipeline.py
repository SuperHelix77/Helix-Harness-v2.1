# SPDX-License-Identifier: AGPL-3.0-only

from core.helix_engine import (
    Correction,
    GATES,
    ToolStep,
    Trajectory,
    adapter_record,
    cluster_deficits,
    rollback_adapter,
    run_adaptation_pipeline,
    success_authorizes_weight_update,
)
from core.helix_engine.capture import clear_session, finalize_session, record_tool_execution, session_steps


def _traj(**overrides) -> Trajectory:
    steps = [
        ToolStep(name="read_file", arguments="a.py", result="def a(): pass", useful_hint="evidence"),
        ToolStep(name="read_file", arguments="a.py", result="def a(): pass", useful_hint="redundant"),
        ToolStep(name="search_repo", arguments="a.py", result="same as read", useful_hint="redundant"),
        ToolStep(name="grep", arguments="TODO", result="none", useful_hint="useful"),
        ToolStep(name="read_file", arguments="b.py", result="def b(): pass", useful_hint="evidence"),
        ToolStep(name="edit_file", arguments="a.py bad", result="broke tests", useful_hint="wrong", error="tests failed"),
        ToolStep(name="run_tests", arguments="pytest", result="FAIL", useful_hint="useful"),
        ToolStep(name="edit_file", arguments="a.py good", result="tests pass", useful_hint="useful"),
    ]
    base = dict(
        prompt_state="fix the failing helper",
        retrieved_context="a.py helper tests",
        reasoning="try a patch",
        steps=steps,
        user_corrections=[
            Correction(
                state="tests failed after bad edit",
                bad_action="edit_file a.py bad",
                failure_evidence="FAIL",
                correct_action="edit_file a.py good",
                why="user restored the helper invariant",
            )
        ],
        final_result="tests pass",
        latency_ms=12_000,
        prompt_tokens=70_000,
        completion_tokens=4_000,
        verified=True,
    )
    base.update(overrides)
    return Trajectory(**base)


def test_pipeline_runs_full_gate_chain_and_does_not_qlora_on_success():
    result = run_adaptation_pipeline(_traj())
    assert result["gates"] == list(GATES)
    assert result["success_authorizes_weight_update"] is False
    assert result["route"] == "hermes"
    assert result["decision"] == "promote_hermes"
    assert result["compressed"]["tool_calls"] < 8
    assert result["corrections"][0]["correct_action"] == "edit_file a.py good"
    traj = _traj()
    names = [f"{traj.steps[i].name}:{traj.steps[i].arguments}" for i in result["credit"]]
    assert "edit_file:a.py bad" not in names


def test_qlora_candidate_requires_every_gate_and_stays_reversible():
    ready = _traj(
        allow_qlora=True,
        frequent_behavior=True,
        holdout_passed=True,
        regression_passed=True,
        dataset_hash="sha256:fixture",
        adapter_version="qlora-1",
    )
    result = run_adaptation_pipeline(ready)
    assert result["route"] == "qlora"
    assert result["decision"] == "promote_qlora_candidate"
    assert result["adapter"]["reversible"] is True
    assert result["adapter"]["applied"] is False
    assert result["success_authorizes_weight_update"] is False
    missing_holdout = _traj(
        allow_qlora=True,
        frequent_behavior=True,
        holdout_passed=False,
        regression_passed=True,
        dataset_hash="sha256:fixture",
    )
    blocked = run_adaptation_pipeline(missing_holdout)
    assert blocked["route"] != "qlora" or blocked["decision"] != "promote_qlora_candidate"


def test_adapter_rollback_is_reversible():
    current = adapter_record(
        version="qlora-2",
        dataset_hash="sha256:b",
        provenance="helix-engine-pipeline",
        eval_delta=-0.1,
        parent_version="qlora-1",
    )
    previous = adapter_record(
        version="qlora-1",
        dataset_hash="sha256:a",
        provenance="helix-engine-pipeline",
        eval_delta=0.2,
    )
    rolled = rollback_adapter(current, previous)
    assert rolled["rolled_back"] == "qlora-2"
    assert rolled["active"] == "qlora-1"
    assert rolled["reversible"] is True


def test_q38_deficit_clusters_from_repeated_bad_steps():
    clusters = cluster_deficits([_traj(), _traj()])
    assert clusters["rereads_files"] >= 1
    assert clusters["unnecessary_repository_searches"] >= 1
    assert clusters["wrong_tool_under_ambiguity"] >= 1


def test_tool_loop_capture_feeds_the_pipeline():
    clear_session("goal-session")
    record_tool_execution("goal-session", "read_file", {"path": "a.py"}, "def a(): pass")
    record_tool_execution("goal-session", "read_file", {"path": "a.py"}, "def a(): pass")
    record_tool_execution("goal-session", "edit_file", {"path": "a.py"}, "Error: tests failed")
    assert len(session_steps("goal-session")) == 3
    assert session_steps("goal-session")[1].useful_hint == "redundant"
    assert session_steps("goal-session")[2].useful_hint == "wrong"
    result = finalize_session(
        "goal-session",
        prompt_state="fix helper",
        final_result="ok",
        verified=True,
    )
    assert result["success_authorizes_weight_update"] is False
    assert success_authorizes_weight_update(_traj()) is False
    assert result["route"] == "hermes"


def test_execute_tool_wrapper_appends_a_trajectory_step(monkeypatch):
    from core.inference import tools

    clear_session("wrap-session")
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *args, **kwargs: "ok-from-tool")
    output = tools.execute_tool("read_file", {"path": "a.py"}, session_id="wrap-session")
    assert output == "ok-from-tool"
    steps = session_steps("wrap-session")
    assert steps[-1].name == "read_file"
    assert "a.py" in steps[-1].arguments


def test_execute_tool_capture_isolates_session_and_thread(monkeypatch):
    from core.helix_engine.capture import capture_session_key
    from core.inference import tools

    key = capture_session_key("sess-1", "thread-9")
    clear_session(key)
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *args, **kwargs: "ok")
    tools.execute_tool(
        "read_file",
        {"path": "a.py"},
        session_id="sess-1",
        thread_id="thread-9",
    )
    assert session_steps(key)[-1].name == "read_file"
    assert session_steps("sess-1") == []
    assert session_steps("thread-9") == []
