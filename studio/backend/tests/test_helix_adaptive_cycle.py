# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import asyncio

import pytest
from fastapi import BackgroundTasks

from core.helix_engine.audit import (
    fallback_self_audit,
    observable_audit_payload,
    parse_self_audit,
    prepare_observable_self_audit,
)
from core.helix_engine.cache_integrity import build_cache_integrity_report
from core.helix_engine.compress import build_counterfactual_candidate
from core.helix_engine.decision_controller import (
    calibration_summary,
    parse_decision_kind,
    safe_advisory_decision,
    shadow_decision,
)
from core.helix_engine.evidence import build_evidence_claims, discover_important_claims, evaluate_claim
from core.helix_engine.hermes import adjudicate
from core.helix_engine.pipeline import run_adaptation_pipeline
from core.helix_engine.quality import build_quality_vector
from core.helix_engine.schemas import (
    AdaptationKind,
    DecisionChoice,
    EvidenceStatus,
    SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
)
from core.helix_engine.trajectory import ToolStep, Trajectory, trajectory_record


def _traj(**overrides) -> Trajectory:
    base = dict(
        prompt_state="verify the helper",
        retrieved_context="helper.py and tests",
        reasoning="private reasoning must not enter audit payloads",
        steps=[ToolStep(name="read_file", arguments="helper.py", result="ok", useful_hint="evidence")],
        user_corrections=[],
        final_result="tests pass",
        latency_ms=100,
        prompt_tokens=1_000,
        completion_tokens=100,
        verified=True,
        extras={"trajectory_id": "traj-1", "telemetry": {"prompt_tokens": 1_000}, "objective_verified": True},
    )
    base.update(overrides)
    return Trajectory(**base)


def test_closed_loop_failure_cannot_break_legacy_pipeline(monkeypatch):
    from core.helix_engine import controller

    def explode(*args, **kwargs):
        raise RuntimeError("optional controller died")

    monkeypatch.setattr(controller, "run_closed_loop", explode)
    result = run_adaptation_pipeline(_traj())
    assert result["product"] == "Helix Harness"
    assert result["adaptive_cycle"]["fail_open"] is True
    assert "optional controller died" in result["adaptive_cycle"]["error"]


def test_cache_controller_failure_is_fail_open(monkeypatch):
    from core.helix_engine import controller

    monkeypatch.setattr(
        controller,
        "build_cache_integrity_report",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("cache telemetry malformed")),
    )
    result = run_adaptation_pipeline(_traj())
    assert result["adaptive_cycle"]["available"] is False
    assert result["decision"] in {"promote_hermes", "discard", "reject"}


def test_invalid_model_audit_uses_observable_fallback_without_changing_task_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    result = run_adaptation_pipeline(_traj(), self_audit={"recommendation": "nonsense"})
    assert result["adaptive_cycle"]["self_audit"]["source"] == "deterministic_fallback"
    assert result["adaptive_cycle"]["self_audit"]["schema_version"] == SCHEMA_VERSION
    assert result["compressed"]["final_decision"] == "tests pass"


def test_hermes_failure_cannot_alter_completed_result(monkeypatch):
    from core.helix_engine import controller

    monkeypatch.setattr(
        controller,
        "adjudicate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Hermes unavailable")),
    )
    result = run_adaptation_pipeline(_traj())
    assert result["adaptive_cycle"]["fail_open"] is True
    assert result["compressed"]["final_decision"] == "tests pass"


def test_typed_decisions_reject_unknown_values_and_are_advisory():
    with pytest.raises(ValueError):
        parse_decision_kind("WRITE_ARBITRARY_PROSE")
    decision = shadow_decision("RUN_ANOTHER_TEST", {"missing_evidence_count": 1, "test_can_resolve": True})
    assert decision.advisory_only is True
    assert decision.probability >= 0.5
    assert decision.to_dict()["decision"] == "RUN_ANOTHER_TEST"


def test_decision_controller_failure_falls_back_to_existing_deep_audit_behavior(monkeypatch):
    from core.helix_engine import decision_controller

    monkeypatch.setattr(
        decision_controller,
        "shadow_decision",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decision backend unavailable")),
    )
    decision = safe_advisory_decision(
        "DEEP_SELF_AUDIT",
        {"meaningful_task": False},
        decision_id="fallback-decision",
        trajectory_id="traj-fallback",
        fallback_choice=DecisionChoice.TAKE,
    )
    assert decision.choice == DecisionChoice.TAKE
    assert decision.confidence == 0.0
    assert decision.policy_version == "helix-fallback-v1"
    assert "decision backend unavailable" in str(decision.evidence_features.get("controller_fallback"))


def test_pre_audit_gate_can_skip_trivial_turn_but_forces_meaningful_or_failed_work(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    trivial = _traj(
        steps=[],
        final_result="hello",
        extras={"trajectory_id": "trivial", "telemetry": {}, "objective_verified": False},
    )
    prepared = prepare_observable_self_audit(trivial)
    assert prepared["perform_deep_audit"] is False
    assert prepared["decision"]["decision"] == "DEEP_SELF_AUDIT"
    assert prepared["decision"]["choice"] == "SKIP"

    failed = _traj(
        steps=[ToolStep(name="run_test", arguments="", result="failed", error="boom")],
        extras={"trajectory_id": "failed", "telemetry": {}, "objective_verified": False},
    )
    prepared_failed = prepare_observable_self_audit(failed)
    assert prepared_failed["perform_deep_audit"] is True
    assert "failed_tool" in prepared_failed["artifacts"]["deep_audit_forced_reasons"]


def test_evidence_distinguishes_supported_from_unverified_speed_claim():
    unsupported_report = evaluate_claim({"claim": "unit tests pass", "evidence": ["pytest: 12 passed"]})
    assert unsupported_report.status == EvidenceStatus.UNVERIFIED
    untyped = evaluate_claim(
        {"claim": "unit tests pass", "evidence_refs": ["tool:0:verification"]},
        objective_evidence={"tool:0:verification": "pytest: 12 passed"},
    )
    assert untyped.status == EvidenceStatus.UNVERIFIED
    assert any("typed backend provenance" in item for item in untyped.missing_evidence)
    traj = _traj(extras={"trajectory_id": "t-speed", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    speed = build_evidence_claims(traj, cache, [{"claim_id": "speed", "claim": "DFlash achieves 5x speed"}])[0]
    assert speed.status == EvidenceStatus.UNVERIFIED
    assert any("benchmark" in item for item in speed.missing_evidence)


def test_backend_discovers_high_impact_final_claim_without_frontend_claim_array():
    traj = _traj(
        final_result="The helper is fixed. Helix is at least 5x faster. No benchmark was supplied for another idea.",
        extras={"trajectory_id": "claim-discovery", "telemetry": {}, "objective_verified": False},
    )
    discovered = discover_important_claims(traj)
    claims = [item["claim"] for item in discovered]
    assert "The helper is fixed." in claims
    assert "Helix is at least 5x faster." in claims
    assert all("No benchmark" not in item for item in claims)
    cache = build_cache_integrity_report({}, traj.steps)
    evidence = build_evidence_claims(traj, cache, discovered)
    speed = next(item for item in evidence if "5x" in item.claim)
    assert speed.status == EvidenceStatus.UNVERIFIED
    assert any("benchmark" in item for item in speed.missing_evidence)


def test_backend_claim_discovery_does_not_promote_explicit_uncertainty_as_positive_claim():
    traj = _traj(
        final_result=(
            "Performance is not measured. No benchmark was run. "
            "The fix is not verified. The test did not pass."
        ),
        extras={"trajectory_id": "claim-caveat", "telemetry": {}, "objective_verified": False},
    )
    assert discover_important_claims(traj) == []


def test_single_self_audit_cannot_make_qlora_eligible(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    audit = {
        "objective": "fix recurring tool choice",
        "achieved": True,
        "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE",
        "recommendation_reason": "I think this should be trained",
        "self_assessment_confidence": 0.9,
    }
    result = run_adaptation_pipeline(_traj(), self_audit=audit)
    adaptation = result["adaptive_cycle"]["adaptation"]
    assert adaptation["qlora_eligible"] is False
    assert adaptation["action"] != "QLORA_CANDIDATE"


def test_cache_miss_does_not_reduce_task_quality_or_evidence_quality():
    traj = _traj(extras={"trajectory_id": "quality", "telemetry": {"prompt_tokens": 1000, "cached_tokens": 0}, "objective_verified": True})
    cache = build_cache_integrity_report(traj.extras["telemetry"], traj.steps)
    evidence = build_evidence_claims(traj, cache)
    audit = fallback_self_audit(traj, cache, evidence)
    quality = build_quality_vector(traj, cache, evidence, audit)
    assert quality.task_quality == 1.0
    assert quality.evidentiary_completeness == 1.0
    assert quality.computational_efficiency <= 1.0


def test_failed_tool_call_is_recorded_and_same_exception_is_reraised(monkeypatch):
    from core.helix_engine.capture import clear_session, session_steps
    from core.inference import tools

    clear_session("failure-session")
    original = RuntimeError("boom")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", fail)
    with pytest.raises(RuntimeError) as exc:
        tools.execute_tool("terminal", {"command": "false"}, session_id="failure-session")
    assert exc.value is original
    step = session_steps("failure-session")[-1]
    assert step.error and "RuntimeError" in step.error


def test_successful_tool_call_preserves_exact_return_object(monkeypatch):
    from core.helix_engine.capture import clear_session, session_steps
    from core.inference import tools

    clear_session("return-session")
    marker = {"exact": [1, 2, 3]}
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: marker)
    result = tools.execute_tool("read_file", {"path": "x"}, session_id="return-session")
    assert result is marker
    assert session_steps("return-session")[-1].name == "read_file"


def test_cancellation_remains_cancellation_and_is_observed(monkeypatch):
    from core.helix_engine.capture import clear_session, session_steps
    from core.inference import tools

    clear_session("cancel-session")
    original = asyncio.CancelledError("cancelled")

    def cancel(*args, **kwargs):
        raise original

    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", cancel)
    with pytest.raises(asyncio.CancelledError) as exc:
        tools.execute_tool("terminal", {"command": "sleep"}, session_id="cancel-session")
    assert exc.value is original
    assert "CancelledError" in (session_steps("cancel-session")[-1].error or "")


def test_capture_failure_cannot_change_tool_return(monkeypatch):
    from core.helix_engine import capture
    from core.inference import tools

    marker = object()
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: marker)
    monkeypatch.setattr(capture, "record_tool_execution", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recorder dead")))
    assert tools.execute_tool("read_file", {}, session_id="recorder-dead") is marker


def test_cache_report_records_provenance_and_avoidable_repeat():
    steps = [
        ToolStep(name="read_file", arguments="a.py", result="same"),
        ToolStep(name="read_file", arguments="a.py", result="same"),
    ]
    report = build_cache_integrity_report(
        {"prompt_tokens": 100, "cached_tokens": 60, "prefill_ms": 10, "ttft_ms": 15},
        steps,
    )
    assert report.stable_prefix_tokens == 60
    assert report.newly_evaluated_tokens == 40
    assert report.cache_reuse_ratio == pytest.approx(0.6)
    assert any(item.cause.value == "MODEL_CAUSED" and not item.necessary for item in report.disruptions)
    assert report.telemetry_provenance["stable_prefix_tokens"] == "inferred_from_cached_tokens"


def test_cache_report_marks_tool_context_reinjection_and_unknown_kv_reset_with_provenance():
    report = build_cache_integrity_report(
        {
            "prompt_tokens": 200,
            "cached_tokens": 50,
            "kv_cache_resets": 1,
            "context_insertions": [
                {
                    "kind": "tool_output",
                    "repeated": True,
                    "estimated_tokens": 25,
                    "provenance": "frontend_outbound_tool_history",
                }
            ],
            "telemetry_provenance": {"kv_cache_resets": "runtime_event"},
        },
        [],
    )
    assert report.kv_cache_resets == 1
    assert report.repeated_context_insertions == 1
    reset = next(item for item in report.disruptions if item.action == "kv_cache_reset")
    assert reset.cause.value == "UNKNOWN"
    assert reset.provenance == "runtime_event"
    repeated = next(item for item in report.disruptions if item.action == "repeat_context_insertion:tool_output")
    assert repeated.cause.value == "TOOL_CAUSED"
    assert repeated.necessary is False


def test_unavailable_runtime_counter_is_not_also_claimed_observable():
    report = build_cache_integrity_report(
        {
            "prompt_tokens": 100,
            "cached_tokens": 80,
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "speculative_counter_scope": "unavailable",
            "telemetry_provenance": {"kv_cache_resets": "unavailable"},
        },
        [ToolStep(name="search_memory", arguments="q", result="same")],
    )
    assert "kv_cache_resets" in report.unavailable_fields
    assert "kv_cache_resets" not in report.observable_fields
    assert "accepted_drafts" in report.unavailable_fields
    assert "accepted_drafts" not in report.observable_fields
    assert "rejected_drafts" in report.unavailable_fields
    assert "context_insertions" in report.observable_fields
    assert report.context_insertions == ["tool_output:search_memory"]


def test_cache_report_treats_search_memory_aliases_as_duplicate_retrieval():
    steps = [
        ToolStep(name="search_memory", arguments='{"query": "same"}', result="same evidence"),
        ToolStep(name="search_conversation", arguments='{"query": "same"}', result="same evidence"),
    ]
    report = build_cache_integrity_report({}, steps)
    assert report.repeated_context_insertions == 1
    assert len(report.disruptions) == 1
    assert report.disruptions[0].action == "repeat_tool:search_conversation"
    assert report.disruptions[0].necessary is False
    assert report.disruptions[0].evidence == [
        "equivalent retrieval path, arguments, and leading result already occurred in this trajectory"
    ]


def test_alias_duplicate_drives_quality_and_counterfactual_even_without_redundant_hint():
    steps = [
        ToolStep(name="search_memory", arguments='{"query": "same"}', result="same evidence", useful_hint="useful"),
        ToolStep(name="search_conversation", arguments='{"query": "same"}', result="same evidence", useful_hint="useful"),
    ]
    traj = _traj(steps=steps, extras={"trajectory_id": "alias-cf", "telemetry": {}, "objective_verified": False})
    cache = build_cache_integrity_report({}, steps)
    evidence = build_evidence_claims(traj, cache)
    audit = fallback_self_audit(traj, cache, evidence)
    quality = build_quality_vector(traj, cache, evidence, audit)
    candidate = build_counterfactual_candidate(
        traj,
        quality=quality,
        evidence_ids=[item.claim_id for item in evidence],
    )

    assert quality.raw_metrics["redundant_tool_calls"] == 1
    assert candidate.actual_tool_calls == 2
    assert candidate.proposed_tool_calls == 1
    assert candidate.estimated_tool_calls_saved == 1
    assert candidate.equivalence_status == EvidenceStatus.UNVERIFIED
    assert candidate.training_pair_eligible is False


def test_self_audit_is_machine_readable_and_versioned():
    audit = parse_self_audit({"objective": "x", "achieved": True, "recommendation": "SKILL"}, model_id="qwen")
    assert audit is not None
    payload = audit.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["recommendation"] == "SKILL"
    assert payload["model_id"] == "qwen"


def test_trajectory_wire_record_is_versioned_and_excludes_hidden_reasoning():
    record = trajectory_record(_traj())
    payload = record.to_dict()
    assert payload["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert payload["trajectory_id"] == "traj-1"
    assert payload["verified"] is True
    assert payload["objective_verified"] is True
    assert "reasoning" not in payload
    assert payload["tool_steps"][0]["name"] == "read_file"


def test_trajectory_wire_separates_completed_loop_from_objective_verification():
    record = trajectory_record(
        _traj(
            verified=True,
            extras={"trajectory_id": "completion-only", "telemetry": {}, "objective_verified": False},
        )
    )
    payload = record.to_dict()
    assert payload["verified"] is True
    assert payload["objective_verified"] is False


def test_observable_audit_payload_contains_resolved_artifacts_not_hidden_reasoning():
    traj = _traj()
    cache = build_cache_integrity_report(traj.extras["telemetry"], traj.steps)
    evidence = build_evidence_claims(traj, cache)
    payload = observable_audit_payload(traj, cache, evidence)
    assert payload["schema_version"] == "helix.audit-input.v1"
    assert payload["trajectory"]["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert "reasoning" not in payload["trajectory"]
    assert payload["evidence"][0]["claim_id"] == "task-outcome"
    assert "cache_integrity" in payload


def test_counterfactual_candidate_is_versioned_and_shorter_is_not_equivalence_proof():
    traj = _traj(
        steps=[
            ToolStep(name="read_file", arguments="a", result="same", useful_hint="evidence"),
            ToolStep(name="read_file", arguments="a", result="same", useful_hint="redundant"),
        ]
    )
    cache = build_cache_integrity_report(traj.extras["telemetry"], traj.steps)
    evidence = build_evidence_claims(traj, cache)
    audit = fallback_self_audit(traj, cache, evidence)
    quality = build_quality_vector(traj, cache, evidence, audit)
    # A client/model cannot spoof equivalence through ordinary telemetry.
    traj.extras["telemetry"]["counterfactual_equivalent_verified"] = True
    candidate = build_counterfactual_candidate(
        traj,
        quality=quality,
        evidence_ids=[item.claim_id for item in evidence],
    )
    assert candidate.schema_version == "helix.counterfactual.v1"
    assert candidate.proposed_tool_calls < candidate.actual_tool_calls
    assert candidate.equivalence_status == EvidenceStatus.UNVERIFIED
    assert candidate.equivalence_verified is False
    assert candidate.training_pair_eligible is False


def test_closed_loop_records_objective_decision_outcomes_and_measured_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    result = run_adaptation_pipeline(_traj())
    adaptive = result["adaptive_cycle"]
    assert adaptive["decision_outcomes"]["EVIDENCE_SUFFICIENT"] is True
    assert adaptive["decision_calibration"]["calibration_measured"] is True
    assert adaptive["decision_calibration"]["labelled_decisions"] >= 2
    assert adaptive["decision_calibration"]["brier_score"] is not None
    assert adaptive["decision_calibration"]["calibrated"] is False
    assert calibration_summary()["labelled_decisions"] >= 2


def test_adaptation_retains_trajectory_and_evidence_provenance():
    traj = _traj(extras={"trajectory_id": "prov-123", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    evidence = build_evidence_claims(traj, cache)
    audit = parse_self_audit({"objective": "x", "achieved": True, "recommendation": "SKILL", "reusable_lessons": ["reuse this"]})
    assert audit is not None
    quality = build_quality_vector(traj, cache, evidence, audit)
    decision = adjudicate(traj, audit, evidence, cache, quality, recurrence=1)
    assert decision.action == AdaptationKind.SKILL
    assert decision.source_trajectory_ids == ["prov-123"]
    assert "task-outcome" in decision.evidence_ids


def test_hermes_records_adaptation_disagreement_with_model_self_audit():
    repeated_result = "same memory result"
    traj = _traj(
        steps=[
            ToolStep(name="search_memory", arguments='{"query":"x"}', result=repeated_result),
            ToolStep(name="search_conversation", arguments='{"query":"x"}', result=repeated_result),
        ],
        extras={"trajectory_id": "disagreement-1", "telemetry": {}, "objective_verified": False},
    )
    cache = build_cache_integrity_report({}, traj.steps)
    evidence = build_evidence_claims(traj, cache)
    audit = parse_self_audit({"objective": "x", "achieved": True, "recommendation": "IGNORE"})
    assert audit is not None
    quality = build_quality_vector(traj, cache, evidence, audit)

    decision = adjudicate(traj, audit, evidence, cache, quality, recurrence=1)

    assert decision.action == AdaptationKind.RUNTIME_POLICY
    assert decision.self_assessment_disagreement is True


def test_ordinary_examples_do_not_autostart_qlora(tmp_path, monkeypatch):
    from routes import self_training

    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")
    state = self_training._empty_state()
    state.update({"autoTrain": True, "minExamples": 4, "status": "idle"})
    self_training._write_state(state)

    async def record(index: int):
        return await self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                modelId="local-model",
                prompt=f"p{index}",
                completion=f"c{index}",
            ),
            BackgroundTasks(),
            current_subject="tester",
        )

    for index in range(4):
        asyncio.run(record(index))
    after = self_training._read_state()
    assert after["status"] == "idle"
    assert len(self_training._qualified_examples(after)) == 0


def test_reanalyzing_same_trajectory_cannot_manufacture_recurrence(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    steps = [
        ToolStep(name="read_file", arguments="a", result="same"),
        ToolStep(name="read_file", arguments="a", result="same", useful_hint="redundant"),
    ]
    traj = _traj(steps=steps, extras={"trajectory_id": "same-id", "model_id": "qwen", "telemetry": {}, "objective_verified": True})
    audit = {
        "objective": "x", "achieved": True, "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE", "self_assessment_confidence": 0.9,
    }
    results = [run_adaptation_pipeline(traj, self_audit=audit)["adaptive_cycle"]["adaptation"] for _ in range(3)]
    assert [item["recurrence_count"] for item in results] == [1, 1, 1]
    assert all(item["qlora_eligible"] is False for item in results)


def test_unrelated_self_reported_patterns_do_not_collide_into_qlora(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    audit = {
        "objective": "x", "achieved": True, "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE", "self_assessment_confidence": 0.9,
    }
    outcomes = []
    for index in range(3):
        traj = _traj(
            prompt_state=f"unrelated objective {index}",
            extras={"trajectory_id": f"unrelated-{index}", "model_id": "qwen", "telemetry": {}, "objective_verified": True},
        )
        outcomes.append(run_adaptation_pipeline(traj, self_audit=audit)["adaptive_cycle"]["adaptation"] )
    assert all(item["recurrence_count"] == 1 for item in outcomes)
    assert all(item["qlora_eligible"] is False for item in outcomes)


def test_speed_claim_with_real_ratio_and_drafts_can_be_supported():
    traj = _traj(
        extras={
            "trajectory_id": "speed-ok",
            "telemetry": {"speed_ratio": 5.2, "speed_benchmark_present": True, "accepted_drafts": 12},
        }
    )
    cache = build_cache_integrity_report(traj.extras["telemetry"], traj.steps)
    claim = build_evidence_claims(traj, cache, [{"claim_id": "speed", "claim": "DFlash achieves 5x speed"}])[0]
    assert claim.status == EvidenceStatus.SUPPORTED
    assert any("5.200x" in item for item in claim.supporting_evidence)



def test_turn_id_isolates_overlapping_generations(monkeypatch):
    from core.helix_engine.capture import capture_session_key, clear_session, session_steps
    from core.inference import tools

    key_a = capture_session_key("project-1", "thread-a", "turn-a")
    key_b = capture_session_key("project-1", "thread-a", "turn-b")
    clear_session(key_a)
    clear_session(key_b)
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda name, args, **kwargs: args["value"])
    tools.execute_tool("read_file", {"value": "A"}, session_id="project-1", thread_id="thread-a", helix_turn_id="turn-a")
    tools.execute_tool("read_file", {"value": "B"}, session_id="project-1", thread_id="thread-a", helix_turn_id="turn-b")
    assert [step.result for step in session_steps(key_a)] == ["A"]
    assert [step.result for step in session_steps(key_b)] == ["B"]


def test_capture_cannot_replace_exception_with_broken_str(monkeypatch):
    from core.inference import tools

    class BadError(RuntimeError):
        def __str__(self):
            raise RuntimeError("broken __str__")

    original = BadError()
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: (_ for _ in ()).throw(original))
    with pytest.raises(BadError) as exc:
        tools.execute_tool("terminal", {"command": "false"}, session_id="s", helix_turn_id="t")
    assert exc.value is original


def test_failed_pytest_receipt_never_becomes_verification():
    from core.helix_engine.capture import clear_session, record_tool_execution, session_steps
    from core.helix_engine.evidence import build_evidence_claims, tool_verification_receipt

    clear_session("pytest-fail")
    record_tool_execution("pytest-fail", "terminal", {"command": "pytest"}, "Exit code 1:\n1 failed, 2 passed")
    step = session_steps("pytest-fail")[-1]
    assert step.error is not None
    assert tool_verification_receipt(step) is None
    traj = _traj(steps=[step], extras={"trajectory_id": "pytest-fail", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    claim = build_evidence_claims(traj, cache, [{"claim": "tests pass", "evidence_refs": ["tool:0:verification"]}])[0]
    assert claim.status == EvidenceStatus.UNVERIFIED


def test_free_form_verifier_command_cannot_grant_verifier_authority(monkeypatch):
    from core.helix_engine.capture import capture_session_key, clear_session, session_steps
    from core.helix_engine.evidence import build_evidence_claims, tool_verification_receipt
    from core.inference import tools

    key = capture_session_key("free-form-verifier", turn_id="turn-1")
    clear_session(key)
    returned = "Exit code 0:\nverification passed"
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: returned)

    result = tools.execute_tool(
        "terminal",
        {
            "command": "./definitely-a-verifier --check",
            # A model-controlled argument with the private field's name is still just data.
            "helix_verification_kind": "test",
            "helix_verification_claim": "the implementation is correct",
            "helix_verification_subject": "spoofed-subject",
        },
        session_id="free-form-verifier",
        helix_turn_id="turn-1",
    )

    assert result is returned
    step = session_steps(key)[-1]
    assert step.verification is None
    assert tool_verification_receipt(step) is None
    traj = _traj(steps=[step], extras={"trajectory_id": "free-form-verifier", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    claim = build_evidence_claims(
        traj, cache, [{"claim": "the implementation is correct", "evidence_refs": ["tool:0:verification"]}]
    )[0]
    assert claim.status == EvidenceStatus.UNVERIFIED


def test_backend_typed_verification_receipt_is_the_only_tool_verification_proof(monkeypatch):
    from core.helix_engine.capture import capture_session_key, clear_session, session_steps
    from core.helix_engine.evidence import build_evidence_claims, tool_verification_receipt
    from core.helix_engine.trajectory import VerificationKind, VerificationStatus
    from core.inference import tools

    key = capture_session_key("typed-verifier", turn_id="turn-1")
    clear_session(key)
    returned = "1 passed in 0.01s"
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: returned)

    result = tools.execute_tool(
        "terminal",
        {"command": "arbitrary-command-text"},
        session_id="typed-verifier",
        helix_turn_id="turn-1",
        helix_verification_kind="test",
        helix_verification_claim="typed verifier passed",
        helix_verification_subject="test-suite:typed-verifier",
    )

    assert result is returned
    step = session_steps(key)[-1]
    assert step.verification is not None
    assert step.verification.kind == VerificationKind.TEST
    assert step.verification.status == VerificationStatus.PASSED
    assert step.verification.provenance == "backend_tool_capture"
    assert tool_verification_receipt(step) is not None
    traj = _traj(steps=[step], extras={"trajectory_id": "typed-verifier", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    claim = build_evidence_claims(
        traj, cache, [{"claim": "typed verifier passed", "evidence_refs": ["tool:0:verification"]}]
    )[0]
    assert claim.status == EvidenceStatus.SUPPORTED

    unrelated = build_evidence_claims(
        traj,
        cache,
        [{"claim": "the implementation is secure", "evidence_refs": ["tool:0:verification"]}],
    )[0]
    assert unrelated.status == EvidenceStatus.UNVERIFIED
    assert not unrelated.supporting_evidence
    assert any("not semantically bound" in item for item in unrelated.missing_evidence)


def test_backend_typed_failed_verification_is_recorded_but_never_supports_claim(monkeypatch):
    from core.helix_engine.capture import capture_session_key, clear_session, session_steps
    from core.helix_engine.evidence import build_evidence_claims, tool_verification_receipt
    from core.helix_engine.trajectory import VerificationStatus
    from core.inference import tools

    key = capture_session_key("typed-verifier-fail", turn_id="turn-1")
    clear_session(key)
    returned = "Exit code 1:\n1 failed, 2 passed"
    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", lambda *a, **k: returned)

    result = tools.execute_tool(
        "terminal",
        {"command": "anything"},
        session_id="typed-verifier-fail",
        helix_turn_id="turn-1",
        helix_verification_kind="test",
        helix_verification_claim="tests pass",
        helix_verification_subject="test-suite:typed-verifier-fail",
    )

    assert result is returned
    step = session_steps(key)[-1]
    assert step.error is not None
    assert step.verification is not None
    assert step.verification.status == VerificationStatus.FAILED
    assert tool_verification_receipt(step) is None
    traj = _traj(steps=[step], extras={"trajectory_id": "typed-verifier-fail", "telemetry": {}})
    cache = build_cache_integrity_report({}, traj.steps)
    claim = build_evidence_claims(
        traj, cache, [{"claim": "tests pass", "evidence_refs": ["tool:0:verification"]}]
    )[0]
    assert claim.status == EvidenceStatus.UNVERIFIED


def test_typed_verification_preserves_exact_cancellation(monkeypatch):
    from core.helix_engine.capture import capture_session_key, clear_session, session_steps
    from core.helix_engine.trajectory import VerificationStatus
    from core.inference import tools

    key = capture_session_key("typed-verifier-cancel", turn_id="turn-1")
    clear_session(key)
    original = asyncio.CancelledError("cancelled")

    def cancel(*args, **kwargs):
        raise original

    monkeypatch.setattr(tools, "_EXECUTE_TOOL_IMPL", cancel)
    with pytest.raises(asyncio.CancelledError) as exc:
        tools.execute_tool(
            "terminal",
            {"command": "anything"},
            session_id="typed-verifier-cancel",
            helix_turn_id="turn-1",
            helix_verification_kind="test",
            helix_verification_claim="tests pass",
            helix_verification_subject="test-suite:typed-verifier-cancel",
        )

    assert exc.value is original
    step = session_steps(key)[-1]
    assert step.verification is not None
    assert step.verification.status == VerificationStatus.FAILED


def test_resolved_raw_tool_result_is_context_not_proof():
    claim = evaluate_claim(
        {"claim": "behavior improved", "evidence_refs": ["tool:0:result"]},
        objective_evidence={"tool:0:result": "ordinary file contents"},
    )
    assert claim.status == EvidenceStatus.UNVERIFIED
    assert not claim.supporting_evidence


def test_three_repeated_objective_patterns_still_need_verified_training_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    audit = {
        "objective": "avoid duplicate reads",
        "achieved": True,
        "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE",
        "self_assessment_confidence": 0.9,
    }
    last = None
    for index in range(3):
        steps = [
            ToolStep(name="read_file", arguments="a.py", result="same", useful_hint="useful"),
            ToolStep(name="read_file", arguments="a.py", result="same", useful_hint="redundant"),
        ]
        traj = _traj(
            steps=steps,
            extras={
                "trajectory_id": f"obj-{index}",
                "model_id": "qwen",
                "telemetry": {},
                "objective_verified": True,
                # Legacy/raw flags are client-spoofable and must never substitute
                # for the backend-issued corrected-target receipt.
                "verified_training_target": "raw model self-audit target",
                "training_target_verified": True,
            },
        )
        last = run_adaptation_pipeline(traj, self_audit=audit)["adaptive_cycle"]["adaptation"]
    assert last is not None
    assert last["action"] == "QLORA_CANDIDATE"
    assert last["recurrence_count"] == 3
    assert last["qlora_eligible"] is False
    assert "corrected target" in last["reason"]


def test_repeated_self_report_without_objective_pattern_never_recurs(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    audit = {
        "objective": "x",
        "achieved": True,
        "continued_too_long": True,
        "overclaimed_claims": ["I overclaimed"],
        "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE",
        "self_assessment_confidence": 0.9,
    }
    decisions = []
    for index in range(3):
        traj = _traj(
            steps=[ToolStep(name="read_file", arguments=str(index), result="ordinary")],
            extras={"trajectory_id": f"self-{index}", "model_id": "qwen", "telemetry": {}, "objective_verified": True},
        )
        decisions.append(run_adaptation_pipeline(traj, self_audit=audit)["adaptive_cycle"]["adaptation"])
    assert all(item["recurrence_count"] == 1 for item in decisions)
    assert all(item["qlora_eligible"] is False for item in decisions)


def test_public_example_cannot_spoof_training_qualification(tmp_path, monkeypatch):
    from routes import self_training

    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")
    state = self_training._empty_state()
    state.update({"autoTrain": True, "minExamples": 1, "status": "idle"})
    self_training._write_state(state)
    payload = self_training.SelfTrainingExampleRequest(
        modelId="local-model",
        prompt="p",
        completion="c",
        eligibleForTraining=True,
        sourceTrajectoryId="spoof",
        evidenceIds=["fake"],
    )
    asyncio.run(self_training.record_self_training_example(payload, BackgroundTasks(), current_subject="tester"))
    after = self_training._read_state()
    assert len(self_training._qualified_examples(after)) == 0
    assert after["examples"][-1]["eligibleForTraining"] is False
    assert after["status"] == "idle"


def test_training_snapshot_is_immutable_after_collection_changes(tmp_path, monkeypatch):
    from routes import self_training

    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")
    monkeypatch.setattr(self_training, "account_path", lambda suffix: tmp_path / suffix)
    state = self_training._empty_state()
    state["examples"] = [
        {"prompt": "p1", "completion": "c1", "eligibleForTraining": True},
        {"prompt": "p2", "completion": "c2", "eligibleForTraining": False},
    ]
    snapshot, digest, count = self_training._snapshot_training_dataset(state, qualified_only=True)
    before = snapshot.read_bytes()
    assert count == 1
    assert b"p1" in before and b"p2" not in before
    # Mutable collection view may change, but the admitted run path never does.
    state["examples"].append({"prompt": "p3", "completion": "c3", "eligibleForTraining": False})
    self_training._write_dataset(state)
    assert snapshot.read_bytes() == before
    import hashlib
    assert hashlib.sha256(before).hexdigest() == digest
