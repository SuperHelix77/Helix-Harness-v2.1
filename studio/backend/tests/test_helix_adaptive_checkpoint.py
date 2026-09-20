# SPDX-License-Identifier: AGPL-3.0-only

import threading
import time
from types import SimpleNamespace


def _patch_checkpoint_basics(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "session_steps", lambda key: [])
    monkeypatch.setattr(
        module,
        "trajectory_from_session",
        lambda *args, **kwargs: SimpleNamespace(extras=kwargs.get("extras", {})),
    )
    monkeypatch.setattr(
        module,
        "critic_from_steps",
        lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: {}),
    )


def _run(module, **overrides):
    kwargs = dict(
        session_id="session",
        thread_id="thread",
        turn_id="turn",
        prompt="finish the task",
        partial_result="halfway there",
        model_id="qwen",
        telemetry={"prompt_tokens": 800, "completion_tokens": 20},
        subject="acct",
    )
    kwargs.update(overrides)
    return module.run_adaptive_checkpoint(**kwargs)


def _patch_successful_checkpoint(monkeypatch, module):
    _patch_checkpoint_basics(monkeypatch, module)
    calls = {"memory": 0, "analysis": 0, "actions": 0, "ledger": 0, "events": 0}
    event_lock = threading.Lock()

    def next_event(_scope):
        with event_lock:
            calls["events"] += 1
            seq = calls["events"]
        return {"sequence": seq, "created_at_ms": 10_000 + seq}

    def add_memory(*args, **kwargs):
        calls["memory"] += 1
        return {"stored": True, "node": {"id": "checkpoint-memory"}}

    def analyze(*args, **kwargs):
        calls["analysis"] += 1
        return {
            "adaptive_cycle": {
                "adaptation": {"action": "SKILL", "qlora_eligible": False},
            }
        }

    def apply_actions(*args, **kwargs):
        calls["actions"] += 1
        return ["stage_skill"]

    def append_ledger(*args, **kwargs):
        calls["ledger"] += 1
        return True

    monkeypatch.setattr(module, "next_event_metadata", next_event)
    monkeypatch.setattr("core.memory.mem0_store.add_experience", add_memory)
    monkeypatch.setattr("core.inference.skills.list_temp_skills", lambda thread_id: [])
    monkeypatch.setattr(module, "run_adaptation_pipeline", analyze)
    monkeypatch.setattr(module, "apply_engine_actions", apply_actions)
    monkeypatch.setattr(module, "append_record", append_ledger)
    return calls


def test_checkpoint_receipt_coordinates_all_four_mechanisms_without_hot_path_mem0_search(
    monkeypatch,
):
    from core.helix_engine import adaptive_checkpoint as module

    _patch_checkpoint_basics(monkeypatch, module)
    boundaries = iter(
        [
            {"sequence": 41, "created_at_ms": 1_000},
            {"sequence": 42, "created_at_ms": 1_100},
            {"sequence": 43, "created_at_ms": 1_200},
        ]
    )
    monkeypatch.setattr(module, "next_event_metadata", lambda _scope: next(boundaries))
    memory_calls = []
    search_calls = []
    ledger_records = []
    monkeypatch.setattr(
        "core.memory.mem0_store.add_experience",
        lambda subject, text, **kwargs: memory_calls.append((subject, text, kwargs))
        or {"stored": True, "node": {"id": "checkpoint-memory"}},
    )

    def forbidden_search(*args, **kwargs):
        search_calls.append((args, kwargs))
        raise AssertionError("adaptive checkpoint must not add a hot-path Mem0 search")

    monkeypatch.setattr("core.memory.mem0_store.search", forbidden_search)
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread_id: [
            {"name": "temporary-one", "temporary": True, "thread_id": thread_id},
            {"name": "temporary-two", "temporary": True, "thread_id": thread_id},
        ],
    )
    monkeypatch.setattr(
        module,
        "run_adaptation_pipeline",
        lambda *args, **kwargs: {
            "adaptive_cycle": {
                "adaptation": {"action": "SKILL", "qlora_eligible": False},
                "shadow_decisions": {
                    "QLORA_CANDIDATE": {
                        "choice": "TAKE",
                        "probability": 0.78,
                        "advisory_only": True,
                        "evidence_features": {
                            "recurrence_count": 3,
                            "behavior_gap": True,
                            "evidence_supported": True,
                        },
                    }
                },
            }
        },
    )
    monkeypatch.setattr(module, "apply_engine_actions", lambda *args, **kwargs: ["stage_skill"])
    monkeypatch.setattr(
        module,
        "append_record",
        lambda category, record: ledger_records.append((category, record)) or True,
    )

    result = _run(module)
    mechanisms = result["mechanisms"]

    assert memory_calls and memory_calls[0][0] == "acct"
    assert search_calls == []
    assert mechanisms["mem0"]["retrieval"] == {
        "status": "preflight_owned",
        "owner": "chat_preflight",
        "performed_at_checkpoint": False,
        "reason": mechanisms["mem0"]["retrieval"]["reason"],
    }
    assert mechanisms["mem0"]["update"]["attempted"] is True
    assert mechanisms["mem0"]["update"]["result"]["stored"] is True

    assert mechanisms["helix_hermes"]["analysis_performed"] is True
    assert mechanisms["helix_hermes"]["hermes_adjudication_performed"] is True
    assert mechanisms["helix_hermes"]["adaptation_action"] == "SKILL"

    skill = mechanisms["temporary_skill"]
    assert skill["inspected"] is True
    assert skill["temporary_skill_count"] == 2
    assert skill["temporary_skills"] == ["temporary-one", "temporary-two"]
    assert skill["skill_disposition"] == "defer_until_task_complete"
    assert skill["final_authority"] == "final_turn_ingest"

    qlora = mechanisms["qlora"]
    assert qlora["considered"] is True
    assert qlora["metadata_available"] is True
    assert qlora["shadow_candidate"] is True
    assert qlora["adaptation_candidate"] is False
    assert qlora["candidate"] is True
    assert qlora["eligible"] is False
    assert qlora["advisory_only"] is True
    assert qlora["training_deferred"] is True
    assert qlora["eligibility_source"] == "hermes_verified_target_gate"
    assert qlora["verified_target_receipt_gate"] == "preserved_final_ingest_authority"

    assert result["qlora_candidate"] is True
    assert result["qlora_eligible"] is False
    assert result["training_deferred"] is True
    assert "defer_qlora_until_answer_complete" in result["actions"]
    assert not any("start_qlora" in action or "queue_qlora" in action for action in result["actions"])
    assert ledger_records[0][0] == "adaptive-checkpoints"
    assert ledger_records[0][1]["mechanisms"] == mechanisms
    assert ledger_records[0][1]["start_sequence"] == 41
    assert ledger_records[0][1]["audit_sequence"] == 42
    assert ledger_records[0][1]["resume_sequence"] == 43
    assert ledger_records[0][1]["start_created_at_ms"] == 1_000
    assert ledger_records[0][1]["audit_created_at_ms"] == 1_100
    assert ledger_records[0][1]["resume_created_at_ms"] == 1_200
    assert result["start_sequence"] == 41
    assert result["audit_sequence"] == 42
    assert result["resume_sequence"] == 43


def test_checkpoint_qlora_candidate_preserves_verified_target_gate_and_never_trains(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    _patch_checkpoint_basics(monkeypatch, module)
    monkeypatch.setattr(
        "core.memory.mem0_store.add_experience",
        lambda *args, **kwargs: {"stored": True},
    )
    monkeypatch.setattr("core.inference.skills.list_temp_skills", lambda thread_id: [])
    monkeypatch.setattr(
        module,
        "run_adaptation_pipeline",
        lambda *args, **kwargs: {
            "adaptive_cycle": {
                "adaptation": {
                    "action": "QLORA_CANDIDATE",
                    "qlora_eligible": False,
                    "reason": "verified corrected-target receipt still required",
                },
                "shadow_decisions": {
                    "QLORA_CANDIDATE": {
                        "choice": "TAKE",
                        "probability": 0.78,
                        "advisory_only": True,
                    }
                },
            }
        },
    )
    monkeypatch.setattr(module, "apply_engine_actions", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "append_record", lambda *args, **kwargs: True)

    result = _run(module)
    qlora = result["mechanisms"]["qlora"]

    assert qlora["candidate"] is True
    assert qlora["adaptation_candidate"] is True
    assert qlora["eligible"] is False
    assert result["qlora_eligible"] is False
    assert qlora["verified_target_receipt_gate"] == "preserved_final_ingest_authority"
    assert qlora["advisory_only"] is True
    assert result["training_deferred"] is True
    assert result["actions"] == ["defer_qlora_until_answer_complete"]


def test_checkpoint_mechanism_receipt_fails_open_when_analysis_skill_and_ledger_fail(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    _patch_checkpoint_basics(monkeypatch, module)

    def fail_memory(*args, **kwargs):
        raise RuntimeError("memory unavailable")

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("controller unavailable")

    def fail_skills(*args, **kwargs):
        raise RuntimeError("skills unavailable")

    def forbidden_actions(*args, **kwargs):
        raise AssertionError("actions must not run after checkpoint analysis failed")

    monkeypatch.setattr("core.memory.mem0_store.add_experience", fail_memory)
    monkeypatch.setattr("core.inference.skills.list_temp_skills", fail_skills)
    monkeypatch.setattr(
        module,
        "next_event_metadata",
        lambda _scope: (_ for _ in ()).throw(RuntimeError("event clock unavailable")),
    )
    monkeypatch.setattr(module, "run_adaptation_pipeline", fail_analysis)
    monkeypatch.setattr(module, "apply_engine_actions", forbidden_actions)
    monkeypatch.setattr(
        module,
        "append_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ledger unavailable")),
    )

    result = _run(module)
    mechanisms = result["mechanisms"]

    assert result["available"] is True
    assert mechanisms["mem0"]["update"]["attempted"] is True
    assert mechanisms["mem0"]["update"]["result"]["stored"] is False
    assert "memory unavailable" in mechanisms["mem0"]["update"]["result"]["reason"]
    assert mechanisms["helix_hermes"]["analysis_attempted"] is True
    assert mechanisms["helix_hermes"]["analysis_performed"] is False
    assert mechanisms["helix_hermes"]["hermes_adjudication_performed"] is False
    assert "controller unavailable" in mechanisms["helix_hermes"]["error"]
    assert mechanisms["temporary_skill"]["skill_disposition"] == "defer_until_task_complete"
    assert "skills unavailable" in mechanisms["temporary_skill"]["error"]
    assert mechanisms["qlora"]["considered"] is True
    assert mechanisms["qlora"]["metadata_available"] is False
    assert mechanisms["qlora"]["candidate"] is False
    assert mechanisms["qlora"]["eligible"] is False
    assert result["training_deferred"] is True
    assert "start_sequence" not in result
    assert "audit_sequence" not in result
    assert "resume_sequence" not in result


def test_checkpoint_first_call_executes_once_and_same_event_seq_replays_exact_receipt(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    calls = _patch_successful_checkpoint(monkeypatch, module)
    first = _run(
        module,
        checkpoint_event_seq=7001,
        run_id="generation-run-1",
        resume_round=3,
    )
    replay = _run(
        module,
        checkpoint_event_seq=7001,
        run_id="generation-run-1",
        # Client-carried round metadata may differ after a reload; the durable
        # event sequence remains the identity authority.
        resume_round=99,
    )

    assert replay == first
    assert first["checkpoint_event_seq"] == 7001
    assert first["resume_round"] == 3
    assert calls == {"memory": 1, "analysis": 1, "actions": 1, "ledger": 1, "events": 3}


def test_checkpoint_concurrent_duplicate_requests_execute_side_effects_once(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    calls = _patch_successful_checkpoint(monkeypatch, module)
    entered = threading.Event()
    release = threading.Event()
    original_add = __import__("core.memory.mem0_store", fromlist=["add_experience"]).add_experience

    def blocked_add(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return original_add(*args, **kwargs)

    monkeypatch.setattr("core.memory.mem0_store.add_experience", blocked_add)
    results = []
    errors = []

    def invoke():
        try:
            results.append(
                _run(
                    module,
                    checkpoint_event_seq=7002,
                    run_id="generation-run-race",
                    resume_round=1,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            errors.append(exc)

    first_thread = threading.Thread(target=invoke)
    second_thread = threading.Thread(target=invoke)
    first_thread.start()
    assert entered.wait(2.0)
    second_thread.start()
    time.sleep(0.05)
    release.set()
    first_thread.join(3.0)
    second_thread.join(3.0)

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert calls == {"memory": 1, "analysis": 1, "actions": 1, "ledger": 1, "events": 3}


def test_checkpoint_account_and_logical_scope_collisions_do_not_dedupe(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module
    from utils.account_context import AccountContext, run_as

    calls = _patch_successful_checkpoint(monkeypatch, module)
    account_a = AccountContext("acct-a", "same-user")
    account_b = AccountContext("acct-b", "same-user")

    def invoke(**overrides):
        values = {
            "checkpoint_event_seq": 8001,
            "checkpoint_id": "same-client-id",
            "run_id": "same-run",
            "resume_round": 0,
        }
        values.update(overrides)
        return _run(module, **values)

    run_as(account_a, invoke)
    run_as(account_b, invoke)
    run_as(account_a, invoke, thread_id="other-thread")
    run_as(account_a, invoke, run_id="other-run")

    assert calls["memory"] == 4
    assert calls["analysis"] == 4
    assert calls["actions"] == 4
    assert calls["ledger"] == 4
    assert calls["events"] == 12


def test_checkpoint_completed_receipt_survives_reload_and_lost_response(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    calls = _patch_successful_checkpoint(monkeypatch, module)
    first = _run(
        module,
        checkpoint_event_seq=9001,
        run_id="generation-run-lost-response",
        resume_round=5,
    )
    assert calls["memory"] == 1

    # Simulate a full module/runtime generation after the response was lost. A
    # completed durable receipt is replayed before any learning or event side effect.
    monkeypatch.setattr(module, "_PROCESS_INSTANCE_ID", "reloaded-process-instance")
    monkeypatch.setattr(
        "core.memory.mem0_store.add_experience",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Mem0 repeated")),
    )
    monkeypatch.setattr(
        module,
        "run_adaptation_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysis repeated")),
    )
    monkeypatch.setattr(
        module,
        "apply_engine_actions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("actions repeated")),
    )
    monkeypatch.setattr(
        module,
        "append_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ledger repeated")),
    )
    monkeypatch.setattr(
        module,
        "next_event_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("event clock repeated")),
    )

    replay = _run(
        module,
        checkpoint_event_seq=9001,
        run_id="generation-run-lost-response",
        resume_round=6,
    )
    assert replay == first


def test_checkpoint_orphaned_claim_recovers_without_repeating_side_effects(monkeypatch):
    from core.helix_engine import adaptive_checkpoint as module

    _patch_checkpoint_basics(monkeypatch, module)
    observed = {
        "adaptive_resume_round": 7,
        "adaptive_checkpoint": {"reason": "context_ratio", "occupancy_tokens": 800},
    }
    identity = module._identity(
        session_id="session",
        thread_id="thread",
        turn_id="turn",
        run_id="crashed-run",
        checkpoint_id=None,
        checkpoint_event_seq=9101,
        resume_round=None,
        telemetry=observed,
        partial_result="partial",
        subject="acct",
    )
    owns, _ = module._claim_checkpoint(identity)
    assert owns is True
    module._phase_update(
        identity,
        "mem0_update_pending",
        boundaries={"start": {"sequence": 51, "created_at_ms": 1_234}},
    )
    with module._ACTIVE_LOCK:
        module._ACTIVE_KEYS.discard(identity["dedupe_key"])
    monkeypatch.setattr(module, "_PROCESS_INSTANCE_ID", "after-crash")
    monkeypatch.setattr(
        "core.memory.mem0_store.add_experience",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Mem0 repeated")),
    )
    monkeypatch.setattr(
        module,
        "run_adaptation_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("analysis repeated")),
    )
    monkeypatch.setattr(
        module,
        "append_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ledger repeated")),
    )
    monkeypatch.setattr(
        module,
        "next_event_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("event clock repeated")),
    )

    recovered = _run(
        module,
        telemetry=observed,
        partial_result="partial",
        checkpoint_event_seq=9101,
        run_id="crashed-run",
        resume_round=8,
    )

    assert recovered["available"] is False
    assert recovered["checkpoint_event_seq"] == 9101
    assert recovered["resume_round"] == 7
    assert recovered["start_sequence"] == 51
    assert recovered["recovery"]["status"] == "interrupted_checkpoint_recovered"
    assert recovered["recovery"]["phase"] == "mem0_update_pending"


def test_adaptive_checkpoint_route_schema_accepts_server_owned_replay_identity():
    from routes.helix_engine import AdaptiveCheckpointIn

    payload = AdaptiveCheckpointIn(
        checkpoint_id="fallback-id",
        checkpoint_event_seq=123,
        run_id="run-123",
        resume_round=4,
        session_id="session",
        thread_id="thread",
        turn_id="turn",
    )
    assert payload.checkpoint_event_seq == 123
    assert payload.run_id == "run-123"
    assert payload.resume_round == 4
