# SPDX-License-Identifier: AGPL-3.0-only
"""Helix Engine is a learning control plane over trajectories, not conversations."""

from core.helix_engine import (
    Correction,
    ToolStep,
    Trajectory,
    assign_credit,
    compress_counterfactual,
    mine_corrections,
    monitor_semantic_turns,
    route_adaptation,
    success_authorizes_weight_update,
)


def _fixture() -> Trajectory:
    steps = [
        ToolStep(name="read_file", arguments="a.py", result="def a(): pass", useful_hint="evidence"),
        ToolStep(name="read_file", arguments="b.py", result="def b(): pass", useful_hint="evidence"),
        ToolStep(name="grep", arguments="TODO", result="none", useful_hint="useful"),
        ToolStep(name="read_file", arguments="a.py", result="def a(): pass", useful_hint="redundant"),
        ToolStep(name="search_repo", arguments="a.py", result="same as read", useful_hint="redundant"),
        ToolStep(name="edit_file", arguments="a.py bad", result="broke tests", useful_hint="wrong", error="tests failed"),
        ToolStep(name="run_tests", arguments="pytest", result="FAIL", useful_hint="useful"),
        ToolStep(name="edit_file", arguments="a.py good", result="tests pass", useful_hint="useful"),
    ]
    return Trajectory(
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
        semantic_turns=3,
        holdout_passed=False,
        allow_qlora=False,
        frequent_behavior=False,
    )


def test_credit_assignment_keeps_useful_subsequence_not_the_whole_success():
    traj = _fixture()
    kept = assign_credit(traj)
    names = [traj.steps[i].name + ":" + traj.steps[i].arguments for i in kept]
    assert "read_file:a.py" in names
    assert "edit_file:a.py good" in names
    assert "read_file:a.py" in names
    assert "search_repo:a.py" not in names
    assert "edit_file:a.py bad" not in names
    assert len(kept) < len(traj.steps)


def test_counterfactual_compression_shortens_a_successful_trajectory():
    compressed = compress_counterfactual(_fixture())
    assert compressed["tool_calls"] < 8
    assert compressed["state_summary_tokens"] <= 2_000
    assert compressed["necessary_evidence_tokens"] <= 4_000
    assert "tests pass" in compressed["final_decision"]


def test_error_mining_builds_contrast_examples_from_user_corrections():
    examples = mine_corrections(_fixture())
    assert examples[0]["bad_action"] == "edit_file a.py bad"
    assert examples[0]["correct_action"] == "edit_file a.py good"
    assert examples[0]["failure_evidence"] == "FAIL"
    assert "invariant" in examples[0]["why"]


def test_success_never_authorizes_weight_updates_by_itself():
    traj = _fixture()
    assert traj.verified is True
    assert success_authorizes_weight_update(traj) is False
    assert route_adaptation(traj) != "qlora"


def test_routing_prefers_hermes_for_reusable_procedure_and_mem0_for_one_off():
    traj = _fixture()
    assert route_adaptation(traj) == "hermes"
    one_off = Trajectory(
        prompt_state="the wifi password is x",
        retrieved_context="",
        reasoning="",
        steps=[],
        user_corrections=[],
        final_result="ok",
        latency_ms=10,
        prompt_tokens=20,
        completion_tokens=5,
        verified=True,
        semantic_turns=0,
        holdout_passed=False,
        allow_qlora=False,
        frequent_behavior=False,
        one_off_fact=True,
    )
    assert route_adaptation(one_off) == "discard"
    qlora_ready = Trajectory(
        **{
            **traj.__dict__,
            "allow_qlora": True,
            "frequent_behavior": True,
            "holdout_passed": True,
            "regression_passed": True,
            "dataset_hash": "sha256:fixture",
        }
    )
    assert route_adaptation(qlora_ready) == "qlora"


def test_semantic_turn_monitor_flags_unnecessary_chat_and_feeds_self_improvement():
    signal = monitor_semantic_turns(["ok", "sure", "let me think about that again", "as I said"])
    assert signal["unnecessary_turns"] >= 1
    assert signal["feed_self_improvement"] is True
