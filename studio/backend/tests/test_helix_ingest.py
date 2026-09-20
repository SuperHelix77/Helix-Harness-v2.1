# SPDX-License-Identifier: AGPL-3.0-only

from dataclasses import replace
import hashlib

import pytest
from fastapi import BackgroundTasks

from core.helix_engine.capture import (
    capture_session_key,
    clear_session,
    record_tool_control_event,
    record_tool_execution,
    session_control_events,
    session_steps,
    trajectory_from_session,
)
from core.helix_engine.critic import critic_from_steps, parse_self_critic
from core.helix_engine.ingest import ingest_turn
from core.helix_engine.trajectory import ToolStep, Trajectory, trajectory_record
from core.helix_engine.routing import route_adaptation
from core.helix_engine.training_targets import (
    issue_verified_training_target_receipt,
    validated_training_target_receipt,
)


def test_parse_self_critic_reads_finished_and_tool_judgement():
    critic = parse_self_critic(
        {
            "finished": "partial",
            "rightTools": False,
            "rightSkills": False,
            "tooManyTools": True,
            "toolCount": 9,
            "usedSkills": [],
            "notes": "reread the same file",
            "recommendation": "skill",
            "skillTitle": "Use the existing json skill",
            "skillContent": "Call read_skill before inventing a workflow.",
            "reason": "The task already has a skill.",
        }
    )
    assert critic.finished == "partial"
    assert critic.right_tools is False
    assert critic.too_many_tools is True
    assert critic.recommendation == "skill"


def test_unfinished_or_wrong_tools_route_hermes_not_qlora():
    traj = Trajectory(
        prompt_state="fix helper",
        retrieved_context="",
        reasoning="",
        steps=[ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="useful")],
        user_corrections=[],
        final_result="still broken",
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        verified=False,
        extras={
            "critic": {
                "finished": "partial",
                "right_tools": False,
                "right_skills": False,
                "too_many_tools": True,
                "recommendation": "qlora",
            }
        },
    )
    assert route_adaptation(traj) == "hermes"


def test_full_finish_with_right_tools_does_not_train():
    traj = Trajectory(
        prompt_state="what time is it",
        retrieved_context="",
        reasoning="",
        steps=[],
        user_corrections=[],
        final_result="afternoon",
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        verified=True,
        extras={
            "critic": {
                "finished": "full",
                "right_tools": True,
                "right_skills": True,
                "too_many_tools": False,
                "recommendation": "none",
            }
        },
    )
    assert route_adaptation(traj) == "discard"


def test_deterministic_critic_flags_redundant_tool_spam():
    steps = [
        ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="useful"),
        ToolStep(name="read_file", arguments="a.py", result="ok", useful_hint="redundant"),
        ToolStep(name="search_repo", arguments="a.py", result="same", useful_hint="redundant"),
        ToolStep(name="grep", arguments="a", result="same", useful_hint="redundant"),
    ]
    critic = critic_from_steps(steps, finished="partial")
    assert critic.too_many_tools is True
    assert critic.recommendation == "skill"


def test_live_capture_records_objective_retry_index_for_same_execution():
    key = "retry-capture"
    clear_session(key)
    first = record_tool_execution(key, "web_search", {"query": "x"}, "Error: temporary")
    second = record_tool_execution(key, "web_search", {"query": "x"}, "Error: temporary")
    changed = record_tool_execution(key, "web_search", {"query": "y"}, "ok")

    assert first.retry == 0
    assert second.retry == 1
    assert changed.retry == 0
    assert [step.retry for step in session_steps(key)] == [0, 1, 0]


def test_prevented_tool_call_is_a_versioned_control_event_not_a_fake_tool_step():
    key = capture_session_key("control-session", "thread", "turn-1")
    clear_session(key)
    event = record_tool_control_event(
        key,
        action="equivalent_duplicate",
        tool_name="search_conversation",
        arguments={"query": "same"},
        reason="search_memory already completed the equivalent retrieval",
        equivalent_to="search_memory",
        progress={"executed_calls": 1, "suppressed_equivalent_duplicates": 1},
    )

    assert event.schema_version == "helix.tool-control.v1"
    assert session_steps(key) == []
    assert session_control_events(key)[0].action == "equivalent_duplicate"

    traj = trajectory_from_session(
        key,
        prompt_state="retrieve same memory once",
        final_result="done",
        verified=False,
        extras={"trajectory_id": "turn-1", "objective_verified": False},
    )
    record = trajectory_record(traj)
    assert record.tool_steps == []
    assert len(record.control_events) == 1
    assert record.control_events[0]["schema_version"] == "helix.tool-control.v1"
    assert record.control_events[0]["equivalent_to"] == "search_memory"


def test_ingest_turn_stages_hermes_skill_and_never_auto_qlora(tmp_path, monkeypatch):
    from routes import learning, self_training

    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning.json")
    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "qlora.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")

    clear_session("critic-session")
    record_tool_execution("critic-session", "read_file", {"path": "a.py"}, "ok")
    record_tool_execution("critic-session", "read_file", {"path": "a.py"}, "ok")
    result = ingest_turn(
        session_id="critic-session",
        prompt="verify the json contract",
        final_result="still missing a check",
        thread_id="thread-1",
        subject="tester",
        critic={
            "finished": "partial",
            "rightTools": False,
            "rightSkills": False,
            "tooManyTools": True,
            "toolCount": 2,
            "usedSkills": [],
            "recommendation": "qlora",
            "skillTitle": "json-checks",
            "skillContent": "Read the json-checks skill and follow it.",
            "reason": "Existing skill covers this.",
        },
    )
    assert result["success_authorizes_weight_update"] is False
    assert result["route"] == "hermes"
    assert result["decision"] == "promote_hermes"
    # The adaptive controller is authoritative when healthy: repeated mechanical
    # reads route to runtime policy, while the raw legacy qlora recommendation is ignored.
    assert "advise_runtime_fix" in result["actions"]
    assert "stage_skill" not in result["actions"]
    assert "start_qlora" not in result["actions"]
    state = learning._read_state()
    assert state["pending"] == []
    qlora = self_training._read_state()
    assert (qlora.get("lastRecommendation") or {}).get("action") == "runtime-fix"
    assert (qlora.get("lastRecommendation") or {}).get("advisoryOnly") is True
    assert qlora.get("status") != "queued"


def test_unverified_hermes_memory_stays_pending_even_in_autonomous_mode(tmp_path, monkeypatch):
    from core.helix_engine import ingest as ingest_module
    from routes import learning

    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning.json")
    state = learning._empty_state()
    state["decisionMode"] = "autonomous"
    learning._write_state(state)
    clear_session("unverified-memory")
    monkeypatch.setattr(
        ingest_module,
        "run_adaptation_pipeline",
        lambda *_args, **_kwargs: {
            "success_authorizes_weight_update": False,
            "decision": "promote_hermes",
            "adaptive_cycle": {
                "available": True,
                "adaptation": {"action": "MEMORY", "qlora_eligible": False},
                "self_audit": {
                    "reusable_lessons": ["Model-proposed lesson without objective outcome proof."],
                },
                "evidence": [
                    {
                        "claim_id": "task-outcome",
                        "status": "UNVERIFIED",
                    }
                ],
            },
        },
    )

    result = ingest_module.ingest_turn(
        session_id="unverified-memory",
        prompt="remember this",
        final_result="done",
        self_audit={"recommendation": "MEMORY"},
        critic={"finished": "full", "rightTools": True, "rightSkills": True},
        thread_id="thread-memory",
        subject="tester",
    )

    saved = learning._read_state()
    assert "stage_memory" in result["actions"]
    assert saved["memory"] == []
    assert len(saved["pending"]) == 1
    assert saved["pending"][0]["kind"] == "memory"


def test_final_ingest_ignores_client_objective_verified_telemetry(monkeypatch):
    from core.helix_engine import ingest as ingest_module

    key = "client-objective-proof"
    clear_session(key)
    captured = {}

    def pipeline(traj, **_kwargs):
        captured["objective_verified"] = traj.extras.get("objective_verified")
        captured["telemetry"] = dict(traj.extras.get("telemetry") or {})
        return {
            "success_authorizes_weight_update": False,
            "decision": "discard",
            "adaptive_cycle": {"available": False},
        }

    monkeypatch.setattr(ingest_module, "run_adaptation_pipeline", pipeline)
    monkeypatch.setattr(ingest_module, "apply_engine_actions", lambda *_args, **_kwargs: [])

    ingest_module.ingest_turn(
        session_id=key,
        prompt="client says verified",
        final_result="done",
        telemetry={"objective_verified": True, "prompt_tokens": 10},
        thread_id="thread-proof",
        subject="tester",
    )

    assert captured["objective_verified"] is False
    assert "objective_verified" not in captured["telemetry"]


def test_skill_retention_uses_observed_final_turn_not_model_claimed_completion(monkeypatch):
    from core.helix_engine import ingest as ingest_module

    key = "retention-authority"
    clear_session(key)
    record_tool_execution(
        key,
        "learn_skill",
        {"name": "observed-skill", "description": "x", "instructions": "y"},
        "created",
    )
    record_tool_execution(key, "read_skill", {"name": "observed-skill"}, "read")
    promoted = []
    discarded = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda _thread: [{"name": "observed-skill"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda name, thread_id: promoted.append((name, thread_id)),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda name, thread_id: discarded.append((name, thread_id)),
    )
    monkeypatch.setattr(
        ingest_module,
        "run_adaptation_pipeline",
        lambda *_args, **_kwargs: {
            "success_authorizes_weight_update": False,
            "decision": "discard",
            "adaptive_cycle": {
                "available": True,
                "adaptation": {"action": "IGNORE", "qlora_eligible": False},
                "evidence": [
                    {"claim_id": "task-outcome", "status": "UNVERIFIED"},
                ],
            },
        },
    )

    result = ingest_module.ingest_turn(
        session_id=key,
        prompt="perform task",
        final_result="",
        critic={
            "finished": "full",
            "rightTools": True,
            "rightSkills": True,
        },
        thread_id="thread-retention",
        subject="tester",
    )

    assert promoted == []
    assert discarded == [("observed-skill", "thread-retention")]
    assert "discard_temp_skill:observed-skill" in result["actions"]
    assert result["skill_retention"][0]["evidence"]["task_finished_full"] is False



def test_verified_training_target_receipt_is_backend_owned_and_tamper_evident(tmp_path, monkeypatch):
    from core.helix_engine.ledger import recent_records

    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    receipt = issue_verified_training_target_receipt(
        trajectory_id="human-target-1",
        target="Use the corrected response.",
        source="human_correction",
        verifier_subject="tester",
        source_ref="user-correction-message-17",
        source_thread_id="thread-1",
    )
    assert receipt is not None
    assert validated_training_target_receipt(receipt, trajectory_id="human-target-1") is receipt
    assert receipt.source.value == "human_correction"
    assert receipt.target_sha256 == hashlib.sha256(b"Use the corrected response.").hexdigest()
    assert len(receipt.receipt_sha256) == 64
    replayed_receipt = issue_verified_training_target_receipt(
        trajectory_id="human-target-1",
        target="Use the corrected response.",
        source="human_correction",
        verifier_subject="tester",
        source_ref="user-correction-message-17",
        source_thread_id="thread-1",
    )
    assert replayed_receipt is not None
    assert replayed_receipt.receipt_id == receipt.receipt_id
    assert replayed_receipt.receipt_sha256 == receipt.receipt_sha256

    tampered = replace(receipt, target="model-supplied replacement")
    assert validated_training_target_receipt(tampered, trajectory_id="human-target-1") is None
    assert validated_training_target_receipt(receipt.metadata(), trajectory_id="human-target-1") is None

    with pytest.raises(ValueError, match="human_correction or objective_correction"):
        issue_verified_training_target_receipt(
            trajectory_id="self-audit-target",
            target="train on my own answer",
            source="model_self_audit",
            verifier_subject="tester",
        )

    stored = recent_records("training-target-receipts")
    assert len(stored) == 1
    assert stored[0]["source"] == "human_correction"
    assert stored[0]["provenance"] == "backend_verified_target_ingress"
    assert stored[0]["target_sha256"] == receipt.target_sha256
    assert stored[0]["receipt_sha256"] == receipt.receipt_sha256
    assert "target" not in stored[0]


def test_objective_training_target_requires_backend_typed_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    corrected_target = "Corrected objective answer"
    plain_step = ToolStep(name="terminal", arguments="pytest", result="12 passed")
    with pytest.raises(ValueError, match="unverified evidence"):
        issue_verified_training_target_receipt(
            trajectory_id="objective-1",
            target=corrected_target,
            source="objective_correction",
            verifier_subject="tester",
            evidence_refs=["tool:0:verification"],
            steps=[plain_step],
        )

    clear_session("objective-wrong-subject")
    record_tool_execution(
        "objective-wrong-subject",
        "terminal",
        {"command": "pytest"},
        "12 passed",
        verification_kind="test",
        verification_claim="corrected objective answer verified",
        verification_subject="target:sha256:not-the-corrected-target",
    )
    with pytest.raises(ValueError, match="not bound to the corrected target"):
        issue_verified_training_target_receipt(
            trajectory_id="objective-wrong-subject",
            target=corrected_target,
            source="objective_correction",
            verifier_subject="tester",
            evidence_refs=["tool:0:verification"],
            steps=session_steps("objective-wrong-subject"),
        )

    clear_session("objective-receipt")
    record_tool_execution(
        "objective-receipt",
        "terminal",
        {"command": "pytest"},
        "12 passed",
        verification_kind="test",
        verification_claim="corrected objective answer verified",
        verification_subject=f"target:sha256:{hashlib.sha256(corrected_target.encode()).hexdigest()}",
    )
    receipt = issue_verified_training_target_receipt(
        trajectory_id="objective-2",
        target=corrected_target,
        source="objective_correction",
        verifier_subject="tester",
        evidence_refs=["tool:0:verification"],
        steps=session_steps("objective-receipt"),
    )
    assert receipt is not None
    assert receipt.source.value == "objective_correction"
    assert receipt.evidence_refs == ("tool:0:verification",)


def test_verified_target_ingress_stages_only_after_third_repeated_pattern(tmp_path, monkeypatch):
    from core.helix_engine.ledger import recent_records
    from routes import self_training
    from routes.helix_engine import (
        IngestTurnIn,
        VerifiedTrainingTargetIn,
        ingest_completed_turn,
    )

    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path / "ledger"))
    monkeypatch.setattr(self_training, "_state_path", lambda: tmp_path / "qlora.json")
    monkeypatch.setattr(self_training, "_dataset_path", lambda: tmp_path / "examples.jsonl")

    audit = {
        "objective": "avoid duplicate reads",
        "achieved": True,
        "likely_behavioral_pattern": True,
        "recommendation": "QLORA_CANDIDATE",
        "recommendation_reason": "repeated duplicate read behavior",
        "self_assessment_confidence": 0.9,
    }
    results = []
    qualified_counts = []
    for index in range(3):
        turn_id = f"verified-target-turn-{index}"
        corrected_target = "Read a.py once, use the evidence, then stop."
        key = capture_session_key("verified-target-session", "thread-1", turn_id)
        clear_session(key)
        for _ in range(2):
            record_tool_execution(
                key,
                "read_file",
                {"path": "a.py"},
                "verification passed",
                verification_kind="test",
                verification_claim="the verifier passed",
                verification_subject=f"target:sha256:{hashlib.sha256(corrected_target.encode()).hexdigest()}",
            )
        result = ingest_completed_turn(
            IngestTurnIn(
                session_id="verified-target-session",
                thread_id="thread-1",
                turn_id=turn_id,
                prompt="inspect a.py once and stop",
                final_result="done",
                self_audit=audit,
                claims=[
                    {
                        "claim_id": "objective-check",
                        "claim": "the verifier passed",
                        "evidence_refs": ["tool:0:verification"],
                    }
                ],
                model_id="qwen-test",
                verified_training_target=VerifiedTrainingTargetIn(
                    target=corrected_target,
                    source="objective_correction",
                    source_ref="pytest-objective-verifier",
                    evidence_refs=["tool:0:verification"],
                ),
            ),
            BackgroundTasks(),
            current_subject="tester",
        )
        results.append(result)
        qualified_counts.append(len(self_training._qualified_examples(self_training._read_state())))

    assert [item["adaptive_cycle"]["adaptation"]["recurrence_count"] for item in results] == [1, 2, 3]
    assert [item["adaptive_cycle"]["adaptation"]["qlora_eligible"] for item in results] == [False, False, True]
    assert qualified_counts == [0, 0, 1]
    assert "stage_qlora_candidate" not in results[0]["actions"]
    assert "stage_qlora_candidate" not in results[1]["actions"]
    assert "stage_qlora_candidate" in results[2]["actions"]
    assert results[2]["qlora_stage"]["stored"] is True
    assert results[2]["qlora_stage"]["reason"] == "verified_hermes_candidate_staged"
    assert results[2]["qlora_outcome"] == {
        "outcome": "denied",
        "reason": "automatic_qlora_training_disabled",
    }
    assert all(item["training_target_receipt"]["accepted"] is True for item in results)

    receipts = recent_records("training-target-receipts")
    assert len(receipts) == 3
    assert all(item["source"] == "objective_correction" for item in receipts)
    assert all(len(item["target_sha256"]) == 64 and len(item["receipt_sha256"]) == 64 for item in receipts)
    admissions = recent_records("qlora-admissions")
    assert len(admissions) == 1
    assert admissions[0]["training_target_verified"] is True
    assert admissions[0]["training_target_receipt"]["source"] == "objective_correction"

    state = self_training._read_state()
    qualified = self_training._qualified_examples(state)
    assert len(qualified) == 1
    assert qualified[0]["completion"] == "Read a.py once, use the evidence, then stop."
    assert qualified[0]["sourceTrajectoryId"] == "verified-target-turn-2"


def test_ingest_preserves_effective_adapter_identity_separately_from_base_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    key = "effective-model-identity"
    clear_session(key)
    record_tool_execution(key, "read_file", {"path": "a.py"}, "ok")

    result = ingest_turn(
        session_id=key,
        prompt="inspect a.py",
        final_result="done",
        model_id="qwen/base-checkpoint",
        effective_model_id="qwen/base-checkpoint::adapter=enabled",
        adapter_state=True,
        turn_id="effective-model-turn",
        subject="tester",
    )

    adaptive = result["adaptive_cycle"]
    assert adaptive["self_audit"]["model_id"] == "qwen/base-checkpoint::adapter=enabled"
    assert "model:qwen/base-checkpoint::adapter=enabled" not in adaptive["pattern_fingerprint"]
    # Fingerprints are hashes, but the behavioral identity must influence them when
    # an objective recurring pattern exists. A base-only trajectory must differ.
    clear_session("base-model-identity")
    for _ in range(2):
        record_tool_execution("base-model-identity", "read_file", {"path": "a.py"}, "ok")
    base = ingest_turn(
        session_id="base-model-identity",
        prompt="inspect a.py",
        final_result="done",
        model_id="qwen/base-checkpoint",
        effective_model_id="qwen/base-checkpoint::adapter=disabled",
        adapter_state=False,
        turn_id="base-model-turn",
        subject="tester",
    )
    clear_session("adapter-model-identity")
    for _ in range(2):
        record_tool_execution("adapter-model-identity", "read_file", {"path": "a.py"}, "ok")
    adapter = ingest_turn(
        session_id="adapter-model-identity",
        prompt="inspect a.py",
        final_result="done",
        model_id="qwen/base-checkpoint",
        effective_model_id="qwen/base-checkpoint::adapter=enabled",
        adapter_state=True,
        turn_id="adapter-model-turn",
        subject="tester",
    )
    assert base["adaptive_cycle"]["pattern_fingerprint"] != adapter["adaptive_cycle"]["pattern_fingerprint"]
