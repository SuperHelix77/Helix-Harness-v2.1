# SPDX-License-Identifier: AGPL-3.0-only

import json

from core.helix_engine.capture import capture_session_key
from core.helix_engine.ledger import append_record
from core.helix_engine import provenance as module


def _checkpoint(session_id: str, thread_id: str, turn_id: str, *, marker: str = "") -> dict:
    return {
        "schema_version": "helix.adaptive-checkpoint.v1",
        "trajectory_id": turn_id,
        "session_id": capture_session_key(session_id, thread_id, turn_id),
        "thread_id": thread_id,
        "model_id": "effective-model",
        "mechanisms": {
            "mem0": {
                "retrieval": {
                    "status": "preflight_owned",
                    "owner": "chat_preflight",
                    "performed_at_checkpoint": False,
                },
                "update": {"attempted": True, "result": {"stored": True}},
            },
            "marker": marker,
        },
        "actions": ["defer_qlora_until_answer_complete"],
        "created_at_ms": 100,
    }


def test_provenance_correlates_final_turn_and_all_supported_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    trajectory_id = "turn-1"
    append_record("adaptive-checkpoints", _checkpoint("session", "thread-a", trajectory_id))
    append_record("trajectories", {"trajectory_id": trajectory_id, "objective": "finish task"})
    append_record("evidence", {"trajectory_id": trajectory_id, "claims": [{"claim_id": "e-1"}]})
    append_record("self-audits", {"trajectory_id": trajectory_id, "objective": "finish task"})
    append_record("adaptations", {"trajectory_id": trajectory_id, "action": "QLORA_CANDIDATE"})
    append_record("quality", {"trajectory_id": trajectory_id, "score": 0.9})
    append_record(
        "decisions",
        {
            "trajectory_id": trajectory_id,
            "decision_id": "turn-1:QLORA_CANDIDATE",
            "decision": "QLORA_CANDIDATE",
            "choice": "TAKE",
            "eventual_outcome": True,
        },
    )
    append_record(
        "counterfactuals",
        {"source_trajectory_id": trajectory_id, "candidate_id": "cf-1"},
    )
    target = {
        "trajectory_id": trajectory_id,
        "receipt_id": "target-1",
        "source_thread_id": "thread-a",
    }
    append_record("training-target-receipts", target)
    append_record(
        "qlora-admissions",
        {
            "trajectory_id": trajectory_id,
            "training_target_verified": True,
            "training_target_receipt": target,
        },
    )
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="effective-model",
        base_model_id="base-model",
        actions=["stage_qlora_candidate", "queue_qlora_training"],
        skill_retention=[
            {
                "skill_name": "repo-audit",
                "disposition": "promoted",
                "reason": "created and read successfully",
                "evidence": {"read_successfully": True},
            }
        ],
        qlora_outcome={
            "outcome": "queued",
            "reason": "qualified_candidate_passed_final_autonomous_queue_gate",
        },
    )

    result = module.read_provenance(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
    )

    assert result["available"] is True
    assert result["selection"]["source"] == "turn_receipt"
    assert result["selection"]["trajectory_id"] == trajectory_id
    assert result["turn_receipt"]["model"] == "effective-model"
    assert result["adaptive_checkpoints"][0]["mechanisms"]["mem0"]["retrieval"]["status"] == "preflight_owned"
    assert result["mechanisms"]["mem0"]["retrieval"]["owner"] == "chat_preflight"
    assert result["trajectory"]["objective"] == "finish task"
    assert result["evidence"]["claims"][0]["claim_id"] == "e-1"
    assert result["self_audit"]["objective"] == "finish task"
    assert result["adaptation"]["action"] == "QLORA_CANDIDATE"
    assert result["quality"]["score"] == 0.9
    assert result["decisions"][0]["decision_id"] == "turn-1:QLORA_CANDIDATE"
    assert result["decision_outcomes"][0]["eventual_outcome"] is True
    assert result["counterfactual"]["candidate_id"] == "cf-1"
    assert result["qlora_admission"]["training_target_verified"] is True
    assert result["training_target_receipt"]["receipt_id"] == "target-1"
    assert result["skill_retention"][0]["skill_name"] == "repo-audit"
    assert result["skill_retention"][0]["disposition"] == "promoted"
    assert result["qlora_outcome"]["outcome"] == "queued"
    assert "queue_gate" in result["qlora_outcome"]["reason"]
    assert result["final_actions"] == ["stage_qlora_candidate", "queue_qlora_training"]
    assert "mem0_preflight_retrieval_details" in result["missing_sources"]
    assert "evidence" not in result["missing_sources"]


def test_provenance_selects_latest_matching_turn_or_live_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    times = iter([1.0, 2.0])
    monkeypatch.setattr(module.time, "time", lambda: next(times))
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id="turn-old",
        trajectory_id="turn-old",
        model="model",
        actions=["old"],
    )
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id="turn-new",
        trajectory_id="turn-new",
        model="model",
        actions=["new"],
    )
    append_record("adaptations", {"trajectory_id": "turn-old", "action": "MEMORY"})
    append_record("adaptations", {"trajectory_id": "turn-new", "action": "SKILL"})

    latest = module.read_provenance(session_id="session", thread_id="thread-a")
    assert latest["selection"]["turn_id"] == "turn-new"
    assert latest["adaptation"]["action"] == "SKILL"
    assert latest["final_actions"] == ["new"]

    live = _checkpoint("live-session", "thread-live", "turn-live")
    live["created_at_ms"] = 300
    append_record("adaptive-checkpoints", live)
    result = module.read_provenance(
        session_id="live-session",
        thread_id="thread-live",
        turn_id="turn-live",
    )
    assert result["available"] is True
    assert result["selection"]["source"] == "adaptive_checkpoint"
    assert result["selection"]["trajectory_id"] == "turn-live"
    assert result["turn_receipt"] is None
    assert result["final_actions"] == []
    assert "turn_receipt" in result["missing_sources"]

    newer_live = _checkpoint("session", "thread-a", "turn-live-newer")
    newer_live["created_at_ms"] = 3_000
    append_record("adaptive-checkpoints", newer_live)
    newest = module.read_provenance(session_id="session", thread_id="thread-a")
    assert newest["selection"]["source"] == "adaptive_checkpoint"
    assert newest["selection"]["trajectory_id"] == "turn-live-newer"
    assert newest["turn_receipt"] is None
    assert newest["final_actions"] == []


def test_provenance_reports_missing_sources_instead_of_inventing_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id="turn-empty",
        trajectory_id="turn-empty",
        model="model",
        actions=[],
    )

    result = module.read_provenance(
        session_id="session",
        thread_id="thread-a",
        turn_id="turn-empty",
    )

    assert result["available"] is True
    assert result["trajectory"] is None
    assert result["evidence"] is None
    assert result["self_audit"] is None
    assert result["adaptation"] is None
    assert result["quality"] is None
    assert result["decisions"] == []
    assert result["counterfactual"] is None
    assert result["qlora_admission"] is None
    assert result["training_target_receipt"] is None
    for source in (
        "adaptive_checkpoints",
        "mechanisms",
        "trajectory",
        "evidence",
        "self_audit",
        "adaptation",
        "quality",
        "decisions",
        "counterfactual",
        "qlora_admission",
        "training_target_receipt",
    ):
        assert source in result["missing_sources"]


def test_exact_historical_turn_survives_beyond_jsonl_tail_scan_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    for index in range(module._SCAN_LIMIT + 40):
        turn_id = f"turn-{index:04d}"
        assert module.append_turn_receipt(
            session_id="long-session",
            thread_id="long-thread",
            turn_id=turn_id,
            trajectory_id=turn_id,
            model="model",
            actions=[f"action-{index}"],
        )

    result = module.read_provenance(
        session_id="long-session",
        thread_id="long-thread",
        turn_id="turn-0000",
    )

    assert result["available"] is True
    assert result["selection"]["turn_id"] == "turn-0000"
    assert result["final_actions"] == ["action-0"]


def test_provenance_fails_closed_on_reused_trajectory_id_across_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    trajectory_id = "shared-turn-id"
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="model-a",
        actions=["thread-a-action"],
    )
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-b",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="model-b",
        actions=["thread-b-secret-action"],
    )
    append_record("adaptive-checkpoints", _checkpoint("session", "thread-a", trajectory_id, marker="A"))
    append_record("adaptive-checkpoints", _checkpoint("session", "thread-b", trajectory_id, marker="B-SECRET"))
    append_record(
        "evidence",
        {"trajectory_id": trajectory_id, "claims": [{"claim": "THREAD-B-SECRET"}]},
    )
    append_record(
        "training-target-receipts",
        {
            "trajectory_id": trajectory_id,
            "receipt_id": "thread-b-target-secret",
            "source_thread_id": "thread-b",
        },
    )
    append_record(
        "qlora-admissions",
        {
            "trajectory_id": trajectory_id,
            "training_target_receipt": {"source_thread_id": "thread-b"},
            "secret": "THREAD-B-QLORA-SECRET",
        },
    )

    result = module.read_provenance(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
    )
    encoded = json.dumps(result, sort_keys=True)

    assert result["selection"]["thread_id"] == "thread-a"
    assert result["selection"]["scope_ambiguous"] is True
    assert result["final_actions"] == ["thread-a-action"]
    assert len(result["adaptive_checkpoints"]) == 1
    assert result["adaptive_checkpoints"][0]["mechanisms"]["marker"] == "A"
    assert result["evidence"] is None
    assert result["training_target_receipt"] is None
    assert result["qlora_admission"] is None
    assert "trajectory_scope_ambiguous_across_sessions_or_threads" in result["missing_sources"]
    assert "THREAD-B-SECRET" not in encoded
    assert "thread-b-target-secret" not in encoded
    assert "THREAD-B-QLORA-SECRET" not in encoded
    assert "B-SECRET" not in encoded


def test_provenance_fails_closed_on_reused_trajectory_id_across_sessions_same_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    trajectory_id = "same-turn-id"
    assert module.append_turn_receipt(
        session_id="session-a",
        thread_id="shared-thread",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="model-a",
        actions=["session-a-action"],
    )
    assert module.append_turn_receipt(
        session_id="session-b",
        thread_id="shared-thread",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="model-b",
        actions=["session-b-secret-action"],
    )
    append_record(
        "evidence",
        {"trajectory_id": trajectory_id, "claims": [{"claim": "SESSION-B-SECRET"}]},
    )

    result = module.read_provenance(
        session_id="session-a",
        thread_id="shared-thread",
        turn_id=trajectory_id,
    )
    encoded = json.dumps(result, sort_keys=True)

    assert result["selection"]["session_id"] == "session-a"
    assert result["selection"]["scope_ambiguous"] is True
    assert result["final_actions"] == ["session-a-action"]
    assert result["evidence"] is None
    assert "trajectory_scope_ambiguous_across_sessions_or_threads" in result["missing_sources"]
    assert "SESSION-B-SECRET" not in encoded


def test_ingest_route_indexes_turn_receipt_after_final_queue_action(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    from fastapi import BackgroundTasks
    from routes import helix_engine, self_training

    monkeypatch.setattr(
        helix_engine,
        "ingest_turn",
        lambda **kwargs: {
            "actions": ["stage_qlora_candidate"],
            "skill_retention": [
                {
                    "skill_name": "repo-audit",
                    "disposition": "promoted",
                    "reason": "created and read successfully",
                }
            ],
            "trajectory_id": "route-turn",
            "turn_id": "route-turn",
            "thread_id": "thread-route",
            "model_id": "effective-model",
            "base_model_id": "base-model",
        },
    )
    monkeypatch.setattr(
        self_training,
        "queue_hermes_training_with_provenance",
        lambda **kwargs: {
            "outcome": "queued",
            "reason": "qualified_candidate_passed_final_autonomous_queue_gate",
        },
    )
    monkeypatch.setattr(self_training, "_start_training_for_state", lambda subject: None)

    result = helix_engine.ingest_completed_turn(
        helix_engine.IngestTurnIn(
            session_id="route-session",
            thread_id="thread-route",
            turn_id="route-turn",
            prompt="finish it",
            final_result="done",
            model_id="base-model",
            effective_model_id="effective-model",
        ),
        BackgroundTasks(),
        current_subject="tester",
    )
    indexed = module.read_provenance(
        session_id="route-session",
        thread_id="thread-route",
        turn_id="route-turn",
    )
    endpoint = helix_engine.provenance(
        session_id="route-session",
        thread_id="thread-route",
        turn_id="route-turn",
        current_subject="tester",
    )

    assert result["actions"] == ["stage_qlora_candidate", "queue_qlora_training"]
    assert result["qlora_outcome"]["outcome"] == "queued"
    assert indexed["turn_receipt"]["actions"] == result["actions"]
    assert indexed["turn_receipt"]["model"] == "effective-model"
    assert indexed["skill_retention"][0]["skill_name"] == "repo-audit"
    assert indexed["qlora_outcome"]["outcome"] == "queued"
    assert endpoint["selection"]["trajectory_id"] == "route-turn"
    assert endpoint["final_actions"] == result["actions"]


def test_ingest_route_persists_deferred_qlora_queue_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    from fastapi import BackgroundTasks
    from routes import helix_engine, self_training

    monkeypatch.setattr(
        helix_engine,
        "ingest_turn",
        lambda **kwargs: {
            "adaptive_cycle": {
                "adaptation": {
                    "action": "QLORA_CANDIDATE",
                    "qlora_eligible": True,
                }
            },
            "actions": ["stage_qlora_candidate"],
            "trajectory_id": "route-deferred",
            "turn_id": "route-deferred",
            "thread_id": "thread-route",
            "model_id": "effective-model",
            "base_model_id": "base-model",
        },
    )
    monkeypatch.setattr(
        self_training,
        "queue_hermes_training_with_provenance",
        lambda **kwargs: {
            "outcome": "deferred",
            "reason": "needs_more_verified_examples:3/8",
        },
    )

    result = helix_engine.ingest_completed_turn(
        helix_engine.IngestTurnIn(
            session_id="route-session",
            thread_id="thread-route",
            turn_id="route-deferred",
            prompt="finish it",
            final_result="done",
            model_id="base-model",
            effective_model_id="effective-model",
        ),
        BackgroundTasks(),
        current_subject="tester",
    )
    indexed = module.read_provenance(
        session_id="route-session",
        thread_id="thread-route",
        turn_id="route-deferred",
    )

    assert result["actions"] == ["stage_qlora_candidate"]
    assert result["qlora_outcome"] == {
        "outcome": "deferred",
        "reason": "needs_more_verified_examples:3/8",
    }
    assert indexed["qlora_outcome"] == result["qlora_outcome"]
    assert "qlora_outcome" not in indexed["missing_sources"]


def test_provenance_response_is_size_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    trajectory_id = "bounded-turn"
    assert module.append_turn_receipt(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
        trajectory_id=trajectory_id,
        model="model",
        actions=["done"],
    )
    payload = "x" * 120_000
    append_record("trajectories", {"trajectory_id": trajectory_id, "objective": payload})
    append_record("evidence", {"trajectory_id": trajectory_id, "claims": [{"claim": payload}]})
    append_record("self-audits", {"trajectory_id": trajectory_id, "notes": payload})
    append_record("adaptations", {"trajectory_id": trajectory_id, "reason": payload})
    append_record("quality", {"trajectory_id": trajectory_id, "notes": payload})

    result = module.read_provenance(
        session_id="session",
        thread_id="thread-a",
        turn_id=trajectory_id,
    )

    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
    assert len(encoded) <= module._MAX_RESPONSE_BYTES
    assert len(result["trajectory"]["objective"]) <= module._MAX_STRING_CHARS


def test_provenance_response_hard_cap_survives_adversarial_nested_receipts():
    leaf = "x" * module._MAX_STRING_CHARS
    wide = {str(index): [leaf] * module._MAX_LIST_ITEMS for index in range(module._MAX_DICT_ITEMS)}
    payload = {
        "schema_version": "helix.provenance.v1",
        "available": True,
        "selection": {
            "source": "turn_receipt",
            "session_id": "session",
            "thread_id": "thread",
            "turn_id": "turn",
            "trajectory_id": "turn",
        },
        "turn_receipt": {
            "session_id": "session",
            "thread_id": "thread",
            "turn_id": "turn",
            "trajectory_id": "turn",
            "actions": ["done"],
        },
        "adaptive_checkpoints": [wide] * module._MAX_CHECKPOINTS,
        "mechanisms": wide,
        "trajectory": wide,
        "evidence": wide,
        "self_audit": wide,
        "adaptation": wide,
        "quality": wide,
        "decisions": [wide] * module._MAX_DECISIONS,
        "decision_outcomes": [wide] * module._MAX_DECISIONS,
        "counterfactual": wide,
        "qlora_admission": wide,
        "training_target_receipt": wide,
        "skill_retention": [],
        "qlora_outcome": None,
        "final_actions": ["done"],
        "missing_sources": [],
    }

    result = module._bound_response(payload)
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")

    assert len(encoded) <= module._MAX_RESPONSE_BYTES
    assert result["response_truncated"] is True
    assert "response_size_limit" in result["missing_sources"]
    assert result.get("truncated_sources")
