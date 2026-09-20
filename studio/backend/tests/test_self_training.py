# SPDX-License-Identifier: AGPL-3.0-only

import asyncio

from fastapi import BackgroundTasks, HTTPException

from routes import self_training


def test_fixed_holdout_scoring_is_external_to_the_model():
    arithmetic = self_training._HOLDOUT_BENCHMARK[0]
    json_contract = self_training._HOLDOUT_BENCHMARK[1]
    ordered = self_training._HOLDOUT_BENCHMARK[2]
    assert self_training._score_holdout(arithmetic, "391") == 1.0
    assert self_training._score_holdout(arithmetic, "The answer is 391") == 1.0
    assert self_training._score_holdout(json_contract, '{"count":64,"name":"mlx"}') == 1.0
    assert self_training._score_holdout(json_contract, '{"name":"mlx","count":63}') == 0.0
    assert self_training._score_holdout(ordered, "alpha, beta, gamma") == 1.0
    assert self_training._score_holdout(ordered, "alpha, gamma, beta") == 0.0


class _TimedBackend:
    def __init__(self, measured: bool = True):
        self.measured = measured

    def generate_with_adapter_control(self, **kwargs):
        holder = kwargs["stats_holder"]
        holder["stats"] = {
            "timings": {"predicted_per_second": 20.0} if self.measured else {}
        }
        prompt = kwargs["messages"][0]["content"]
        if "17 * 23" in prompt:
            yield "391"
        elif "JSON object" in prompt:
            yield '{"name":"mlx","count":64}'
        elif "three tokens" in prompt:
            yield "alpha,beta,gamma"
        else:
            yield "revert and retrain"


def test_holdout_requires_real_backend_timing():
    score, speed, measured, receipts = self_training._run_holdout(_TimedBackend(), False)
    assert score == 1.0
    assert speed == 20.0
    assert measured is True
    assert len(receipts) == len(self_training._HOLDOUT_BENCHMARK)

    score, speed, measured, _ = self_training._run_holdout(_TimedBackend(measured=False), False)
    assert score == 1.0
    assert speed == 0.0
    assert measured is False


def test_state_wal_snapshot_recovers_from_torn_json_view(tmp_path, monkeypatch):
    state_path = tmp_path / "self_qlora" / "state.json"
    monkeypatch.setattr(self_training, "_state_path", lambda: state_path)
    state = self_training._empty_state()
    state.update(
        {
            "status": "queued",
            "baseModelId": "org/model",
            "lastJobId": "job-durable",
        }
    )

    self_training._write_state(state)
    state_path.write_text("{torn", encoding="utf-8")

    recovered = self_training._read_state()
    assert recovered["status"] == "queued"
    assert recovered["baseModelId"] == "org/model"
    assert recovered["lastJobId"] == "job-durable"
    assert self_training._state_db_path().is_file()


def test_reconcile_stale_training_unwedges_start_with_provenance(monkeypatch):
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-dead",
        "trainingQualifiedOnly": True,
    })
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))
    monkeypatch.setattr(self_training, "_training_terminal_snapshot", lambda _job=None: None)

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "error"
    assert state["trainingQualifiedOnly"] is False
    assert state["lastRecovery"]["kind"] == "stale-training"
    assert state["lastRecovery"]["persistedJobId"] == "job-dead"


def test_reconcile_stale_manual_queue_never_replays_silently(monkeypatch):
    state = self_training._empty_state()
    state.update({"status": "queued", "trainingQualifiedOnly": False})
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "idle"
    assert state["lastRecovery"]["kind"] == "stale-queue-cleared"


def test_reconcile_stale_autonomous_queue_requires_current_policy(monkeypatch):
    state = self_training._empty_state()
    state.update(
        {
            "status": "queued",
            "trainingQualifiedOnly": True,
            "examples": [
                {
                    "eligibleForTraining": True,
                    "sourceTrajectoryId": "turn-final",
                }
            ],
            "autonomousQueueReceipt": {
                "provenance": "final_helix_ingest",
                "sourceTrajectoryId": "turn-final",
                "sourceThreadId": "thread-1",
                "modelId": "",
                "qualifiedExampleCount": 1,
                "createdAt": 1,
            },
        }
    )
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))
    monkeypatch.setattr(self_training, "_autonomous_qlora_policy_allows", lambda current: True)

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is True
    assert state["status"] == "queued"
    assert state["trainingQualifiedOnly"] is True
    assert state["lastRecovery"]["kind"] == "stale-autonomous-queue-rescheduled"


def test_reconcile_legacy_autonomous_queue_without_final_ingest_receipt_never_replays(monkeypatch):
    state = self_training._empty_state()
    state.update(
        {
            "status": "queued",
            "trainingQualifiedOnly": True,
            "examples": [
                {
                    "eligibleForTraining": True,
                    "sourceTrajectoryId": "legacy-turn",
                }
            ],
        }
    )
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))
    monkeypatch.setattr(self_training, "_autonomous_qlora_policy_allows", lambda current: True)

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "idle"
    assert state["trainingQualifiedOnly"] is False
    assert state["autonomousQueueReceipt"] is None
    assert state["lastRecovery"]["kind"] == "stale-queue-cleared"
    assert "provenance" in state["lastRecovery"]["reason"]


def test_hermes_queue_provenance_reports_actual_admission_gate(monkeypatch):
    from routes import learning

    state = self_training._empty_state()
    state.update(
        {
            "autoTrain": True,
            "minExamples": 2,
            "examples": [{"eligibleForTraining": True}],
        }
    )
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(
        learning,
        "_read_state",
        lambda: {
            "decisionMode": "autonomous",
            "allowQloraTraining": True,
        },
    )

    deferred = self_training.queue_hermes_training_with_provenance()
    assert deferred == {
        "outcome": "deferred",
        "reason": "needs_more_verified_examples:1/2",
    }
    assert state["status"] == "idle"

    state["examples"].append(
        {
            "eligibleForTraining": True,
            "sourceTrajectoryId": "turn-final",
        }
    )
    queued = self_training.queue_hermes_training_with_provenance(
        source_trajectory_id="turn-final",
        source_thread_id="thread-1",
    )
    assert queued == {
        "outcome": "queued",
        "reason": "qualified_candidate_passed_final_autonomous_queue_gate",
    }
    assert state["status"] == "queued"
    assert state["trainingQualifiedOnly"] is True
    assert state["autonomousQueueReceipt"]["provenance"] == "final_helix_ingest"
    assert state["autonomousQueueReceipt"]["sourceTrajectoryId"] == "turn-final"
    assert state["autonomousQueueReceipt"]["sourceThreadId"] == "thread-1"
    assert state["lastAutonomousQueueReceipt"] == state["autonomousQueueReceipt"]

    retry = self_training.queue_hermes_training_with_provenance(
        source_trajectory_id="turn-final",
        source_thread_id="thread-1",
    )
    assert retry == {
        "outcome": "queued",
        "reason": "qualified_candidate_queue_already_persisted",
    }

    state["status"] = "idle"
    state["autoTrain"] = False
    denied = self_training.queue_hermes_training_with_provenance()
    assert denied == {
        "outcome": "denied",
        "reason": "automatic_qlora_training_disabled",
    }


def test_model_qlora_recommendation_is_advisory_and_cannot_queue_training(monkeypatch):
    state = self_training._empty_state()
    state.update(
        {
            "status": "idle",
            "autoTrain": True,
            "minExamples": 1,
            "examples": [{"eligibleForTraining": True}],
        }
    )
    writes = []
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda current: writes.append(dict(current)))
    background = BackgroundTasks()

    result = asyncio.run(
        self_training.set_self_training_recommendation(
            self_training.SelfTrainingRecommendationRequest(
                action="qlora",
                reason="model thinks weights may help",
            ),
            background,
            current_subject="unsloth",
        )
    )

    assert result["status"] == "idle"
    assert state["trainingQualifiedOnly"] is False
    assert state["lastRecommendation"]["advisoryOnly"] is True
    assert state["lastRecommendation"]["action"] == "qlora"
    assert background.tasks == []
    assert writes


def test_verified_hermes_candidate_staging_is_idempotent_for_final_ingest_retry(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)

    first = self_training.record_hermes_qualified_candidate_with_provenance(
        model_id="qwen/base",
        prompt="inspect once",
        completion="use the verified answer",
        source_trajectory_id="turn-retry",
        evidence_ids=["tool:0:verification"],
        source_thread_id="thread-1",
    )
    retry = self_training.record_hermes_qualified_candidate_with_provenance(
        model_id="qwen/base",
        prompt="inspect once",
        completion="use the verified answer",
        source_trajectory_id="turn-retry",
        evidence_ids=["tool:0:verification"],
        source_thread_id="thread-1",
    )
    conflict = self_training.record_hermes_qualified_candidate_with_provenance(
        model_id="qwen/base",
        prompt="inspect once",
        completion="different target must not reuse the prior receipt",
        source_trajectory_id="turn-retry",
        evidence_ids=["tool:0:verification"],
        source_thread_id="thread-1",
    )

    assert first["stored"] is True
    assert first["reason"] == "verified_hermes_candidate_staged"
    assert retry["stored"] is True
    assert retry["reason"] == "verified_hermes_candidate_already_staged"
    assert retry["idempotent"] is True
    assert retry["example_id"] == first["example_id"]
    assert len(state["examples"]) == 1
    assert conflict == {
        "stored": False,
        "reason": "source_trajectory_candidate_conflict",
    }


def test_reconcile_unknown_runtime_state_is_fail_open(monkeypatch):
    state = self_training._empty_state()
    state.update({"status": "training", "lastJobId": "job-unknown"})
    before = dict(state)
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: (None, None))

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is False
    assert reschedule is False
    assert state == before


def test_gguf_repo_maps_to_trainable_source_without_guessing_arbitrary_local_file(tmp_path):
    assert (
        self_training._trainable_baseline_for_serving_model("unsloth/Qwen3.8-27B-GGUF")
        == "unsloth/Qwen3.8-27B"
    )
    standalone = tmp_path / "custom.gguf"
    standalone.write_bytes(b"not-a-real-gguf")
    assert self_training._trainable_baseline_for_serving_model(str(standalone)) == str(standalone)


def test_hf_cache_gguf_path_maps_to_trainable_source(tmp_path):
    model_root = tmp_path / "hub" / "models--unsloth--Qwen3.8-27B-GGUF"
    blob = model_root / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"not-a-real-gguf")
    target = model_root / "snapshots" / "abc" / "Qwen3.8-27B-Q4.gguf"
    target.parent.mkdir(parents=True)
    target.symlink_to(blob)
    assert (
        self_training._trainable_baseline_for_serving_model(str(target))
        == "unsloth/Qwen3.8-27B"
    )


def test_collected_gguf_turn_separates_serving_and_trainable_identity(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                modelId="unsloth/Qwen3.8-27B-GGUF",
                prompt="task",
                completion="answer",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    assert result["recorded"] is True
    assert state["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert state["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["examples"][0]["modelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["examples"][0]["trainableBaseModelId"] == "unsloth/Qwen3.8-27B"


def test_completed_raw_example_replay_is_idempotent(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)
    payload = self_training.SelfTrainingExampleRequest(
        modelId="unsloth/Qwen3.8-27B",
        prompt="same completed task",
        completion="same final answer",
        sourceThreadId="thread-replay",
    )

    first = asyncio.run(
        self_training.record_self_training_example(
            payload,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    retry = asyncio.run(
        self_training.record_self_training_example(
            payload,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    assert first["recorded"] is True
    assert retry["recorded"] is True
    assert retry["idempotent"] is True
    assert retry["reason"] == "duplicate_completed_turn"
    assert retry["exampleId"] == state["examples"][0]["id"]
    assert len(state["examples"]) == 1

    distinct_thread = asyncio.run(
        self_training.record_self_training_example(
            payload.model_copy(update={"sourceThreadId": "thread-distinct"}),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    assert distinct_thread["recorded"] is True
    assert "idempotent" not in distinct_thread
    assert len(state["examples"]) == 2


def test_completed_turn_key_survives_lost_response_and_fresh_disk_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    payload = self_training.SelfTrainingExampleRequest(
        modelId="unsloth/Qwen3.8-27B",
        prompt="durable completed task",
        completion="durable final answer",
        sourceThreadId="thread-durable",
        idempotencyKey="turn-run-001",
    )

    first = asyncio.run(
        self_training.record_self_training_example(
            payload,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    first_receipt = dict(first["idempotencyReceipt"])
    persisted = self_training._read_state()
    assert len(persisted["examples"]) == 1
    assert persisted["examples"][0]["idempotencyKey"] == "turn-run-001"
    assert persisted["examples"][0]["idempotencyReceipt"] == first_receipt

    # A process restart has no in-memory receipt cache to recover. Also change a
    # mutable policy bit to prove replay resolves from the committed disk identity
    # before normal admission checks.
    persisted["enabled"] = False
    self_training._write_state(persisted)
    retry = asyncio.run(
        self_training.record_self_training_example(
            payload,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    assert retry["recorded"] is True
    assert retry["idempotent"] is True
    assert retry["reason"] == "idempotency_key_replay"
    assert retry["exampleId"] == first["exampleId"]
    assert retry["idempotencyReceipt"] == first_receipt
    assert len(self_training._read_state()["examples"]) == 1


def test_completed_turn_key_conflict_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    payload = self_training.SelfTrainingExampleRequest(
        modelId="unsloth/Qwen3.8-27B",
        prompt="stable logical turn",
        completion="first committed answer",
        sourceThreadId="thread-conflict",
        idempotencyKey="turn-run-conflict",
    )
    asyncio.run(
        self_training.record_self_training_example(
            payload,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    try:
        asyncio.run(
            self_training.record_self_training_example(
                payload.model_copy(update={"completion": "changed answer"}),
                BackgroundTasks(),
                current_subject="unsloth",
            )
        )
        assert False, "conflicting reuse of a logical-turn key must fail closed"
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "idempotency key" in str(exc.detail)

    state = self_training._read_state()
    assert len(state["examples"]) == 1
    assert state["examples"][0]["completion"] == "first committed answer"


def test_completed_turn_key_scope_separates_threads_and_accounts(tmp_path, monkeypatch):
    from utils.account_context import AccountContext, arun_as, run_as

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    shared = {
        "modelId": "unsloth/Qwen3.8-27B",
        "prompt": "same request key in a distinct scope",
        "idempotencyKey": "scope-key",
    }
    first = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                **shared,
                completion="thread one answer",
                sourceThreadId="thread-1",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    second = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                **shared,
                completion="thread two answer",
                sourceThreadId="thread-2",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    assert first["exampleId"] != second["exampleId"]
    assert len(self_training._read_state()["examples"]) == 2

    alice = AccountContext("11111111111111111111111111111111", "alice")
    bob = AccountContext("22222222222222222222222222222222", "bob")
    alice_payload = self_training.SelfTrainingExampleRequest(
        **shared,
        completion="alice answer",
        sourceThreadId="thread-shared",
    )
    bob_payload = alice_payload.model_copy(update={"completion": "bob answer"})

    alice_result = asyncio.run(
        arun_as(
            alice,
            self_training.record_self_training_example(
                alice_payload,
                BackgroundTasks(),
                current_subject="alice",
            ),
        )
    )
    bob_result = asyncio.run(
        arun_as(
            bob,
            self_training.record_self_training_example(
                bob_payload,
                BackgroundTasks(),
                current_subject="bob",
            ),
        )
    )
    assert alice_result["recorded"] is True
    assert bob_result["recorded"] is True
    assert alice_result["exampleId"] != bob_result["exampleId"]
    assert len(run_as(alice, self_training._read_state)["examples"]) == 1
    assert len(run_as(bob, self_training._read_state)["examples"]) == 1


def test_completed_turn_concurrent_duplicate_key_commits_once(tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def submit() -> None:
        try:
            barrier.wait()
            results.append(
                asyncio.run(
                    self_training.record_self_training_example(
                        self_training.SelfTrainingExampleRequest(
                            modelId="unsloth/Qwen3.8-27B",
                            prompt="concurrent completed task",
                            completion="one committed answer",
                            sourceThreadId="thread-concurrent",
                            idempotencyKey="turn-run-concurrent",
                        ),
                        BackgroundTasks(),
                        current_subject="unsloth",
                    )
                )
            )
        except BaseException as exc:  # test thread: surface every failure in the parent
            errors.append(exc)

    threads = [threading.Thread(target=submit), threading.Thread(target=submit)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert len(self_training._read_state()["examples"]) == 1
    assert {item["exampleId"] for item in results} == {results[0]["exampleId"]}
    assert sum(item.get("idempotent") is True for item in results) == 1
    assert results[0]["idempotencyReceipt"] == results[1]["idempotencyReceipt"]


def test_distinct_completed_turn_keys_keep_identical_text_as_distinct_examples(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    base = self_training.SelfTrainingExampleRequest(
        modelId="unsloth/Qwen3.8-27B",
        prompt="legitimately repeated task",
        completion="same answer",
        sourceThreadId="thread-repeat",
        idempotencyKey="logical-turn-a",
    )
    first = asyncio.run(
        self_training.record_self_training_example(
            base,
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    second = asyncio.run(
        self_training.record_self_training_example(
            base.model_copy(update={"idempotencyKey": "logical-turn-b"}),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    assert first["exampleId"] != second["exampleId"]
    assert len(self_training._read_state()["examples"]) == 2


def test_explicit_trainable_baseline_keeps_matching_gguf_examples(monkeypatch):
    state = self_training._empty_state()
    state["examples"] = [
        {
            "modelId": "unsloth/Qwen3.8-27B-GGUF",
            "trainableBaseModelId": "unsloth/Qwen3.8-27B",
            "prompt": "task",
            "completion": "answer",
        },
        {
            "modelId": "other/Model-GGUF",
            "trainableBaseModelId": "other/Model",
            "prompt": "wrong",
            "completion": "wrong",
        },
    ]
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = self_training.set_self_training_baseline(
        self_training.BaselineRequest(modelId="unsloth/Qwen3.8-27B"),
        current_subject="unsloth",
    )

    assert result["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert len(state["examples"]) == 1
    assert state["examples"][0]["modelId"] == "unsloth/Qwen3.8-27B-GGUF"


def test_different_serving_family_is_not_mixed_into_existing_baseline(monkeypatch):
    state = self_training._empty_state()
    state["baseModelId"] = "unsloth/Qwen3.8-27B"
    state["baseServingModelId"] = "unsloth/Qwen3.8-27B-GGUF"
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                modelId="other/Model-GGUF",
                prompt="task",
                completion="answer",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    assert result["recorded"] is False
    assert result["reason"] == "model differs from baseline"


def test_legacy_gguf_state_migrates_to_trainable_base_and_preserves_serving_snapshot():
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "unsloth/Qwen3.8-27B-GGUF",
            "baseSnapshotPath": "/cache/qwen3.8-gguf/snapshot",
            "examples": [
                {
                    "modelId": "unsloth/Qwen3.8-27B-GGUF",
                    "prompt": "task",
                    "completion": "answer",
                }
            ],
        }
    )
    assert self_training._migrate_trainable_baseline(state) is True
    assert state["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert state["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["baseSnapshotPath"] is None
    assert state["baseServingSnapshotPath"] == "/cache/qwen3.8-gguf/snapshot"
    assert state["examples"][0]["trainableBaseModelId"] == "unsloth/Qwen3.8-27B"
    assert self_training._migrate_trainable_baseline(state) is False


def test_setting_gguf_baseline_never_marks_gguf_snapshot_as_trainable(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = self_training.set_self_training_baseline(
        self_training.BaselineRequest(
            modelId="unsloth/Qwen3.8-27B-GGUF",
            snapshotPath="/cache/gguf-snapshot",
        ),
        current_subject="unsloth",
    )
    assert result["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert result["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert result["baseSnapshotPath"] is None
    assert result["baseServingSnapshotPath"] == "/cache/gguf-snapshot"


def test_reconcile_completed_training_becomes_candidate_ready(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-done",
        "trainingQualifiedOnly": True,
    })
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("job-done", False))
    monkeypatch.setattr(
        self_training,
        "_training_terminal_snapshot",
        lambda _job=None: {
            "job_id": "job-done",
            "active": False,
            "completed": True,
            "error": None,
            "output_dir": str(adapter),
            "message": "Training completed",
        },
    )

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "candidate-ready"
    assert state["lastCandidateAdapterPath"] == str(adapter.resolve())
    assert state["candidateQualifiedOnly"] is True
    assert state["trainingQualifiedOnly"] is False
    assert state["lastRecovery"]["kind"] == "training-completed"


def test_completed_candidate_autobenchmark_requires_current_autonomous_policy(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "candidate-ready",
        "candidateQualifiedOnly": True,
        "lastCandidateAdapterPath": str(adapter),
    })
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: False)
    assert self_training._queue_candidate_benchmark_if_allowed(state) is None
    assert state["status"] == "candidate-ready"

    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)
    assert self_training._queue_candidate_benchmark_if_allowed(state) == str(adapter.resolve())
    assert state["status"] == "benchmark-queued"


def test_live_training_watcher_runs_autonomous_benchmark_after_valid_artifact(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-live",
        "trainingQualifiedOnly": True,
    })
    writes = []
    benchmarks = []
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(
        self_training,
        "_training_terminal_snapshot",
        lambda _job=None: {
            "job_id": "job-live",
            "active": False,
            "completed": True,
            "error": None,
            "output_dir": str(adapter),
            "message": "Training completed",
        },
    )
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)

    async def _no_sleep(_seconds):
        return None

    async def _benchmark(subject, path):
        benchmarks.append((subject, path))

    monkeypatch.setattr(self_training.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(self_training, "_benchmark_completed_candidate", _benchmark)

    asyncio.run(self_training._watch_self_training_job("unsloth", "job-live", True))

    assert state["status"] == "benchmark-queued"
    assert state["lastCandidateAdapterPath"] == str(adapter.resolve())
    assert benchmarks == [("unsloth", str(adapter.resolve()))]
    assert writes


def test_interrupted_benchmark_recovers_to_policy_gated_candidate_boundary(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "benchmarking",
        "candidateQualifiedOnly": True,
        "lastCandidateAdapterPath": str(adapter),
    })
    changed, reschedule = self_training._reconcile_persisted_training_state(state)
    assert changed is True
    assert reschedule is False
    assert state["status"] == "candidate-ready"
    assert state["lastRecovery"]["kind"] == "interrupted-benchmark-recovered"


def test_restart_reconciles_stale_active_adapter_to_inactive_candidate(tmp_path, monkeypatch):
    from core.inference import orchestrator

    adapter = tmp_path / "persisted-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update(
        {
            "status": "promoted",
            "activeAdapterPath": str(adapter),
            "activeServingModelId": "base/model",
        }
    )
    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: None)

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["activeAdapterPath"] is None
    assert state["activeServingModelId"] is None
    assert state["lastCandidateAdapterPath"] == str(adapter.resolve())
    assert state["status"] == "candidate-ready"
    assert state["lastRecovery"]["kind"] == "inactive-persisted-adapter-reconciled"


def test_candidate_recovery_accepts_standard_peft_artifact_name(tmp_path):
    adapter = tmp_path / "peft-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    assert self_training._candidate_artifact_path(str(adapter)) == str(adapter.resolve())


def test_objective_benchmark_promotes_only_after_score_and_speed_pass(tmp_path, monkeypatch):
    import core.inference as inference_core

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})
    writes = []

    class _Backend:
        def __init__(self):
            self.active = False
            self.hot_swaps = 0
            self.reverts = 0

        def hot_swap_adapter(self, path, name, base_model):
            assert path == str(adapter.resolve())
            assert base_model == "base/model"
            self.hot_swaps += 1
            self.active = True
            return True

        def revert_to_base_model(self, base_model):
            assert base_model == "base/model"
            self.reverts += 1
            self.active = False
            return True

        def generate_with_adapter_control(self, **kwargs):
            holder = kwargs["stats_holder"]
            using_candidate = bool(kwargs.get("use_adapter"))
            holder["stats"] = {
                "timings": {"predicted_per_second": 30.0 if using_candidate else 20.0}
            }
            prompt = kwargs["messages"][0]["content"]
            if using_candidate:
                if "17 * 23" in prompt:
                    yield "391"
                elif "JSON object" in prompt:
                    yield '{"name":"mlx","count":64}'
                elif "three tokens" in prompt:
                    yield "alpha,beta,gamma"
                else:
                    yield "revert and retrain"
            else:
                yield "wrong"

    backend = _Backend()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)

    result = asyncio.run(
        self_training.benchmark_self_training_candidate(
            self_training.SelfTrainingBenchmarkRequest(
                candidateAdapterPath=str(adapter), adapterName="candidate"
            ),
            current_subject="unsloth",
        )
    )
    ev = result["lastEvaluation"]
    assert ev["baseScore"] == 0.0
    assert ev["candidateScore"] == 1.0
    assert ev["baseTokPerSec"] == 20.0
    assert ev["candidateTokPerSec"] == 30.0
    assert ev["speedMeasured"] is True
    assert ev["promoted"] is True
    # Candidate attach for measurement, revert for base measurement, then attach
    # again only after the objective gates passed.
    assert backend.hot_swaps == 2
    assert backend.reverts == 1
    assert backend.active is True
    assert result["status"] == "promoted"
    assert result["activeAdapterPath"] == str(adapter.resolve())
    assert result["activeServingModelId"] == "base/model"
    assert writes


def test_reported_evaluation_metrics_cannot_promote_or_activate_adapter(tmp_path, monkeypatch):
    adapter = tmp_path / "reported-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    state = self_training._empty_state()
    state.update({"status": "candidate-ready", "activeAdapterPath": None})
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)

    result = self_training.evaluate_self_training_candidate(
        self_training.SelfTrainingEvaluationRequest(
            candidateAdapterPath=str(adapter),
            baseScore=0.1,
            candidateScore=1.0,
            baseTokPerSec=10.0,
            candidateTokPerSec=100.0,
            evaluator="holdout",
        ),
        current_subject="unsloth",
    )

    assert result["lastEvaluation"]["reportedMetricsPass"] is True
    assert result["lastEvaluation"]["promoted"] is False
    assert result["lastEvaluation"]["promotionAuthority"] == "backend_holdout_or_explicit_hot_swap"
    assert state["activeAdapterPath"] is None
    assert state["status"] == "candidate-ready"


def test_benchmark_cancellation_waits_for_worker_and_reverts_candidate(tmp_path, monkeypatch):
    import threading
    from contextlib import asynccontextmanager

    import core.inference as inference_core
    from core.training import lifecycle

    adapter = tmp_path / "candidate"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})
    entered = threading.Event()
    release = threading.Event()

    class _Backend:
        def __init__(self):
            self.active = False
            self.reverts = 0

        def hot_swap_adapter(self, *_args):
            self.active = True
            return True

        def revert_to_base_model(self, _base):
            self.reverts += 1
            self.active = False
            return True

    backend = _Backend()

    def blocking_holdout(_backend, _adapter):
        entered.set()
        release.wait(timeout=2.0)
        return 1.0, 20.0, True, []

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_run_holdout", blocking_holdout)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    async def run():
        task = asyncio.create_task(
            self_training.benchmark_self_training_candidate(
                self_training.SelfTrainingBenchmarkRequest(candidateAdapterPath=str(adapter)),
                current_subject="unsloth",
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("benchmark cancellation must propagate")

    asyncio.run(run())

    assert backend.active is False
    assert backend.reverts >= 1
    assert state["status"] == "candidate-ready"
    assert state["activeAdapterPath"] is None
    assert state["lastRecovery"]["kind"] == "benchmark-cancelled-recovered"


def test_rejected_benchmark_restores_previously_active_adapter(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    import core.inference as inference_core
    from core.training import lifecycle

    previous = tmp_path / "previous"
    candidate = tmp_path / "candidate"
    for path in (previous, candidate):
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "base/model",
            "status": "promoted",
            "activeAdapterPath": str(previous),
            "activeServingModelId": "base/model",
        }
    )

    class _Backend:
        def __init__(self):
            self.active_path = str(previous)
            self.swaps = []

        def hot_swap_adapter(self, path, _name, _base):
            self.active_path = path
            self.swaps.append(path)
            return True

        def revert_to_base_model(self, _base):
            self.active_path = None
            return True

    backend = _Backend()

    def holdout(_backend, use_adapter):
        if use_adapter:
            return 0.0, 20.0, True, []
        return 1.0, 20.0, True, []

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_run_holdout", holdout)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    result = asyncio.run(
        self_training.benchmark_self_training_candidate(
            self_training.SelfTrainingBenchmarkRequest(candidateAdapterPath=str(candidate)),
            current_subject="unsloth",
        )
    )

    assert result["lastEvaluation"]["promoted"] is False
    assert backend.active_path == str(previous.resolve())
    assert state["activeAdapterPath"] == str(previous)
    assert state["activeServingModelId"] == "base/model"
    assert state["status"] == "promoted"
    assert backend.swaps[-1] == str(previous.resolve())


def test_autonomous_wrapper_preserves_restored_previous_adapter_state(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    import core.inference as inference_core
    from core.training import lifecycle

    previous = tmp_path / "previous-auto"
    candidate = tmp_path / "candidate-auto"
    for path in (previous, candidate):
        path.mkdir()
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "base/model",
            "status": "benchmark-queued",
            "candidateQualifiedOnly": True,
            "lastCandidateAdapterPath": str(candidate),
            "activeAdapterPath": str(previous),
            "activeServingModelId": "base/model",
        }
    )

    class _Backend:
        def __init__(self):
            self.active_path = str(previous)

        def hot_swap_adapter(self, path, _name, _base):
            self.active_path = path
            return True

        def revert_to_base_model(self, _base):
            self.active_path = None
            return True

    backend = _Backend()

    def holdout(_backend, use_adapter):
        return (0.0, 20.0, True, []) if use_adapter else (1.0, 20.0, True, [])

    async def ensure(_subject):
        return backend, False

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)
    monkeypatch.setattr(self_training, "_ensure_trainable_base_for_benchmark", ensure)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_run_holdout", holdout)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    asyncio.run(self_training._benchmark_completed_candidate("unsloth", str(candidate)))

    assert backend.active_path == str(previous.resolve())
    assert state["activeAdapterPath"] == str(previous)
    assert state["activeServingModelId"] == "base/model"
    assert state["status"] == "hot-swapped"
    assert state["candidateQualifiedOnly"] is False


def test_candidate_benchmark_holds_inference_lifecycle_gate(tmp_path, monkeypatch):
    import threading

    import core.inference as inference_core
    from core.inference import llama_keepwarm

    adapter = tmp_path / "candidate-gated"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})
    entered = threading.Event()
    release = threading.Event()
    holdout_calls = 0

    class _Backend:
        def hot_swap_adapter(self, *_args):
            return True

        def revert_to_base_model(self, _base):
            return True

    def holdout(_backend, use_adapter):
        nonlocal holdout_calls
        holdout_calls += 1
        if holdout_calls == 1:
            entered.set()
            release.wait(timeout=2.0)
        return (0.0 if use_adapter else 1.0), 20.0, True, []

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_run_holdout", holdout)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: _Backend())

    async def run():
        benchmark = asyncio.create_task(
            self_training.benchmark_self_training_candidate(
                self_training.SelfTrainingBenchmarkRequest(candidateAdapterPath=str(adapter)),
                current_subject="unsloth",
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)

        reservation = llama_keepwarm.InferenceActivityReservation()
        reservation.reserve()
        pending_chat = asyncio.create_task(reservation.start())
        await asyncio.sleep(0)
        assert pending_chat.done() is False

        release.set()
        await benchmark
        await asyncio.wait_for(pending_chat, timeout=1.0)
        reservation.finish()

    asyncio.run(run())

    assert llama_keepwarm.other_inference_request_count(current_request_counted=False) == 0


def test_autonomous_benchmark_rechecks_policy_before_final_adapter_activation(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    import core.inference as inference_core
    from core.training import lifecycle

    adapter = tmp_path / "policy-race-candidate"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "base/model",
            "status": "benchmarking",
            "candidateQualifiedOnly": True,
        }
    )

    class _Backend:
        def __init__(self):
            self.hot_swaps = 0
            self.reverts = 0
            self.active = False

        def hot_swap_adapter(self, *_args):
            self.hot_swaps += 1
            self.active = True
            return True

        def revert_to_base_model(self, _base):
            self.reverts += 1
            self.active = False
            return True

    backend = _Backend()

    def holdout(_backend, use_adapter):
        return (1.0, 30.0, True, []) if use_adapter else (0.0, 20.0, True, [])

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_run_holdout", holdout)
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: False)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    result = asyncio.run(
        self_training._benchmark_self_training_candidate(
            self_training.SelfTrainingBenchmarkRequest(candidateAdapterPath=str(adapter)),
            current_subject="unsloth",
            autonomous_policy_required=True,
        )
    )

    evaluation = result["lastEvaluation"]
    assert evaluation["intelligenceImproved"] is True
    assert evaluation["speedPreserved"] is True
    assert evaluation["promoted"] is False
    assert evaluation["autonomousPolicyRevoked"] is True
    assert state["status"] == "candidate-ready"
    assert state["activeAdapterPath"] is None
    assert backend.active is False
    assert backend.hot_swaps == 1
    assert backend.reverts == 1


def test_cancelled_automatic_benchmark_unloads_temporarily_loaded_base(tmp_path, monkeypatch):
    adapter = tmp_path / "auto-cancel-candidate"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "base/model",
            "status": "benchmark-queued",
            "candidateQualifiedOnly": True,
            "lastCandidateAdapterPath": str(adapter),
        }
    )

    class _Backend:
        def __init__(self):
            self.unloaded = []

        def unload_model(self, model):
            self.unloaded.append(model)
            return True

    backend = _Backend()

    async def ensure(_subject):
        return backend, True

    async def cancelled_benchmark(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)
    monkeypatch.setattr(self_training, "_ensure_trainable_base_for_benchmark", ensure)
    monkeypatch.setattr(self_training, "_benchmark_self_training_candidate", cancelled_benchmark)

    try:
        asyncio.run(self_training._benchmark_completed_candidate("unsloth", str(adapter)))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("automatic benchmark cancellation must propagate")

    assert backend.unloaded == ["base/model"]
    assert state["status"] == "candidate-ready"
    assert state["candidateQualifiedOnly"] is True
    assert state["activeServingModelId"] is None
    assert state["lastRecovery"]["kind"] == "automatic-benchmark-cancelled-recovered"


def test_manual_hotswap_and_revert_keep_serving_identity_in_sync(tmp_path, monkeypatch):
    import core.inference as inference_core

    adapter = tmp_path / "manual-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})

    class _Backend:
        def hot_swap_adapter(self, *_args):
            return True

        def revert_to_base_model(self, _base):
            return True

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: _Backend())

    swapped = asyncio.run(
        self_training.hot_swap_self_training_adapter(
            self_training.AdapterHotSwapRequest(adapterPath=str(adapter)),
            current_subject="unsloth",
        )
    )
    assert swapped["activeAdapterPath"] == str(adapter.resolve())
    assert swapped["activeServingModelId"] == "base/model"

    reverted = asyncio.run(
        self_training.revert_self_training_adapter(current_subject="unsloth")
    )
    assert reverted["activeAdapterPath"] is None
    assert reverted["activeServingModelId"] is None


def test_cancelled_manual_hotswap_records_settled_runtime_mutation(tmp_path, monkeypatch):
    import threading

    import core.inference as inference_core

    adapter = tmp_path / "cancelled-manual-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})
    entered = threading.Event()
    release = threading.Event()

    class _Backend:
        def hot_swap_adapter(self, *_args):
            entered.set()
            release.wait(timeout=2.0)
            return True

    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: _Backend())

    async def run():
        task = asyncio.create_task(
            self_training.hot_swap_self_training_adapter(
                self_training.AdapterHotSwapRequest(adapterPath=str(adapter)),
                current_subject="unsloth",
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("request cancellation must propagate after state catches up")

    asyncio.run(run())

    assert state["activeAdapterPath"] == str(adapter.resolve())
    assert state["activeServingModelId"] == "base/model"
    assert state["status"] == "hot-swapped"


def test_self_training_waits_for_a_stable_inference_idle_window(monkeypatch):
    from routes import self_training

    # Busy twice, then idle long enough to cross the 500 ms stability fence.
    counts = iter([1, 1, 0, 0, 0, 0])
    monkeypatch.setattr(
        "core.inference.llama_keepwarm.other_inference_request_count",
        lambda current_request_counted=False: next(counts, 0),
    )
    monkeypatch.setattr("routes.training._background_video_generation_active", lambda: False)
    assert asyncio.run(self_training._wait_for_foreground_inference_idle(5.0)) is True


def test_autonomous_training_treats_background_video_as_foreground_ownership(monkeypatch):
    monkeypatch.setattr(
        "core.inference.llama_keepwarm.other_inference_request_count",
        lambda current_request_counted=False: 0,
    )
    monkeypatch.setattr("routes.training._background_video_generation_active", lambda: True)

    assert self_training._foreground_inference_busy() is True


def test_autonomous_training_treats_active_image_generation_as_foreground_ownership(monkeypatch):
    from core.inference import diffusion_engine_router

    class _Diffusion:
        def generate_progress(self):
            return {"active": True}

    monkeypatch.setattr(
        "core.inference.llama_keepwarm.other_inference_request_count",
        lambda current_request_counted=False: 0,
    )
    monkeypatch.setattr("routes.training._background_video_generation_active", lambda: False)
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: _Diffusion())

    assert self_training._foreground_inference_busy() is True


def test_self_training_atomic_recheck_defers_a_chat_that_won_the_idle_race(monkeypatch):
    from contextlib import asynccontextmanager

    from core.training import lifecycle
    from routes import self_training

    state = self_training._empty_state()
    state.update({"status": "queued", "baseModelId": "unsloth/test"})
    writes = []

    async def idle(_timeout=120.0):
        return True

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_wait_for_foreground_inference_idle", idle)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: True)
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(
        self_training,
        "_write_state",
        lambda current: writes.append(dict(current)),
    )
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    asyncio.run(self_training._start_training_for_state("unsloth"))

    assert state["status"] == "queued"
    assert state["lastRecovery"]["kind"] == "training-deferred-for-active-inference"
    assert writes


def test_autonomous_training_execution_rejects_missing_final_ingest_queue_receipt(monkeypatch):
    from contextlib import asynccontextmanager

    from core.training import lifecycle
    from routes import training

    state = self_training._empty_state()
    state.update(
        {
            "status": "queued",
            "baseModelId": "unsloth/test",
            "minExamples": 1,
            "trainingQualifiedOnly": True,
            "examples": [
                {
                    "prompt": "p",
                    "completion": "c",
                    "eligibleForTraining": True,
                    "sourceTrajectoryId": "legacy-advisory-turn",
                }
            ],
            # Intentionally no autonomousQueueReceipt: this models a persisted
            # queue created by the historical advisory recommendation bug.
        }
    )
    started = []

    async def idle(_timeout=120.0):
        return True

    @asynccontextmanager
    async def gate():
        yield

    async def forbidden_start(*_args, **_kwargs):
        started.append(True)
        raise AssertionError("invalid autonomous queue must not reach training start")

    monkeypatch.setattr(self_training, "_wait_for_foreground_inference_idle", idle)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_autonomous_qlora_policy_allows", lambda _state: True)
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)
    monkeypatch.setattr(training, "start_training", forbidden_start)

    asyncio.run(self_training._start_training_for_state("unsloth"))

    assert started == []
    assert state["status"] == "idle"
    assert state["trainingQualifiedOnly"] is False
    assert state["autonomousQueueReceipt"] is None
    assert state["lastRecovery"]["kind"] == "invalid-autonomous-queue-cleared"


def test_self_training_holds_inference_gate_until_training_claim_returns(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from core.inference import llama_keepwarm
    from routes import self_training, training

    state = self_training._empty_state()
    state.update({
        "status": "queued",
        "baseModelId": "unsloth/test",
        "baseContextLength": 4_096,
        "maxSeqLength": 8_192,
        "minExamples": 1,
        "examples": [{"prompt": "p", "completion": "c"}],
    })
    snapshot = tmp_path / "training.jsonl"
    snapshot.write_text('{"text":"x"}\n')
    waiting = {}
    observed = []

    async def idle(_timeout=120.0):
        return True

    async def no_watch(*_args, **_kwargs):
        return None

    async def fake_start(request, **_kwargs):
        # The autonomous request must be clamped to the known serving/baseline context.
        assert request.max_seq_length == 4_096
        reservation = llama_keepwarm.InferenceActivityReservation()
        reservation.reserve()
        task = asyncio.create_task(reservation.start())
        waiting["reservation"] = reservation
        waiting["task"] = task
        # New inference is pending but cannot cross the lifecycle gate while the
        # training start claim is still inside it.
        await asyncio.sleep(0)
        observed.append(task.done())
        return SimpleNamespace(job_id=None, status="error", message="synthetic stop")

    monkeypatch.setattr(self_training, "_wait_for_foreground_inference_idle", idle)
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(
        self_training,
        "_snapshot_training_dataset",
        lambda *_args, **_kwargs: (snapshot, "sha", 1),
    )
    monkeypatch.setattr(self_training, "_watch_self_training_job", no_watch)
    monkeypatch.setattr(training, "start_training", fake_start)

    async def run():
        before = llama_keepwarm.other_inference_request_count(current_request_counted=False)
        assert before == 0
        await self_training._start_training_for_state("unsloth")
        await asyncio.wait_for(waiting["task"], timeout=1.0)
        waiting["reservation"].finish()
        assert llama_keepwarm.other_inference_request_count(current_request_counted=False) == 0

    asyncio.run(run())

    assert observed == [False]


def test_cancelled_self_training_start_preserves_spawned_job_identity(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from core.training import lifecycle
    from routes import training

    state = self_training._empty_state()
    state.update(
        {
            "status": "queued",
            "baseModelId": "unsloth/test",
            "minExamples": 1,
            "examples": [{"prompt": "p", "completion": "c"}],
        }
    )
    snapshot = tmp_path / "cancel-training.jsonl"
    snapshot.write_text('{"text":"x"}\n', encoding="utf-8")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def idle(_timeout=120.0):
        return True

    async def fake_start(_request, **_kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(job_id="job-cancelled-request", status="queued", message="queued")

    @asynccontextmanager
    async def gate():
        yield

    monkeypatch.setattr(self_training, "_wait_for_foreground_inference_idle", idle)
    monkeypatch.setattr(self_training, "_foreground_inference_busy", lambda: False)
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _current: None)
    monkeypatch.setattr(
        self_training,
        "_snapshot_training_dataset",
        lambda *_args, **_kwargs: (snapshot, "sha", 1),
    )
    monkeypatch.setattr(training, "start_training", fake_start)
    monkeypatch.setattr(lifecycle, "training_inference_admission_guard", gate)

    async def run():
        task = asyncio.create_task(self_training._start_training_for_state("unsloth"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled start must propagate after preserving its job receipt")

    asyncio.run(run())

    assert state["status"] == "training"
    assert state["lastJobId"] == "job-cancelled-request"
    assert state["lastRecovery"]["kind"] == "cancelled-start-job-preserved"


def test_self_training_max_sequence_never_exceeds_known_baseline_context():
    from routes import self_training

    state = self_training._empty_state()
    state["maxSeqLength"] = 65_536
    state["baseContextLength"] = 8_192
    assert self_training._safe_training_max_seq_length(state) == 8_192

    state["maxSeqLength"] = 4_096
    assert self_training._safe_training_max_seq_length(state) == 4_096

    state["baseContextLength"] = None
    state["maxSeqLength"] = 131_072  # persisted legacy/corrupt state still gets bounded
    assert self_training._safe_training_max_seq_length(state) == 65_536

    state["maxSeqLength"] = "not-a-number"
    assert self_training._safe_training_max_seq_length(state) == 8_192
