# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import replace

import pytest

from core.helix_engine.optimization_benchmark import (
    LATENCY_METRICS,
    MEMORY_AND_RUNTIME_METRICS,
    QUALITY_AND_WORK_METRICS,
    RECOVERY_METRICS,
    BenchmarkIdentity,
    BenchmarkMetrics,
    CorrectnessOutcome,
    CorrectnessResult,
    EvidenceReference,
    ExecutionScope,
    InvocationCountMatrix,
    InvocationCounts,
    InvocationOutcome,
    MetricMeasurement,
    MetricProvenance,
    ModelInvocationEvent,
    ObjectiveIdentity,
    OptimizationBenchmarkTrial,
    ProviderTier,
    SCHEMA_VERSION,
    SemanticContribution,
    artifact_identity,
    canonical_json,
    canonical_sha256,
    derive_invocation_counts,
    unavailable_metric_group,
    unavailable_metrics,
)


_CONFIG_HASH = "a" * 64
_CRITERIA_HASH = "b" * 64
_CONTENT_HASH = "c" * 64


def _identity(**overrides) -> BenchmarkIdentity:
    values = {
        "benchmark_id": "phase-1-fixture",
        "scenario_id": "semantic-tool-task",
        "trial_id": "trial-0001",
        "replicate": 0,
        "runner_version": "runner-v1",
        "frozen_control_identity": "snapshot:control",
        "candidate_source_snapshot_identity": "snapshot:candidate",
        "hardware_identity": "mac-studio-m2-ultra-192gb",
        "os_identity": "macos-15.7",
        "runtime_identity": "python-3.14",
        "model_identity": "configured-model-set-v1",
        "provider_identity": "configured-provider-set-v1",
        "checkpoint_identity": "checkpoint-set-v1",
        "quantization_identity": "quantization-set-v1",
        "context_identity": "configured-context-32768",
        "kv_identity": "kv-f16",
        "speculation_identity": "disabled",
        "benchmark_configuration_sha256": _CONFIG_HASH,
        "acceptance_criteria_sha256": _CRITERIA_HASH,
        "backend_overlay_identity": None,
        "packaged_app_identity": None,
    }
    values.update(overrides)
    return BenchmarkIdentity(**values)


def _objective(**overrides) -> ObjectiveIdentity:
    values = {
        "owner_subject": "account-1",
        "thread_id": "thread-1",
        "user_message_id": "message-1",
    }
    values.update(overrides)
    return ObjectiveIdentity(**values)


def _evidence(objective: ObjectiveIdentity, **overrides) -> EvidenceReference:
    values = {
        "ref_id": "verifier:pytest:1",
        "kind": "backend_test_receipt",
        "subject": "artifact:candidate-tree",
        "objective_key": objective.key,
        "acceptance_criteria_sha256": _CRITERIA_HASH,
        "content_sha256": _CONTENT_HASH,
        "backend_owned": True,
    }
    values.update(overrides)
    return EvidenceReference(**values)


def _measurement(value, *, provenance=MetricProvenance.MEASURED, exposure=("run:1",), derivation=None):
    return MetricMeasurement(
        value=value,
        provenance=provenance,
        exposure=tuple(exposure),
        derivation=derivation,
    )


def _metrics() -> BenchmarkMetrics:
    quality = unavailable_metric_group(QUALITY_AND_WORK_METRICS)
    latency = unavailable_metric_group(LATENCY_METRICS)
    memory = unavailable_metric_group(MEMORY_AND_RUNTIME_METRICS)
    recovery = unavailable_metric_group(RECOVERY_METRICS)
    quality.update(
        {
            "tool_calls": _measurement(2),
            "deterministic_suboperations": _measurement(7),
            "failed_tool_calls": _measurement(0),
            "prevented_tool_calls": _measurement(1),
            "redundant_tool_calls": _measurement(0),
            "input_tokens": _measurement(100),
            "cached_tokens": _measurement(60),
            "newly_evaluated_tokens": _measurement(40),
        }
    )
    latency["end_to_end_task_wall_ms"] = _measurement(10.5)
    memory["actual_peak_occupied_context_tokens"] = _measurement(100)
    recovery["ambiguous_outcome"] = _measurement(False)
    recovery["duplicate_side_effects"] = _measurement(0)
    return BenchmarkMetrics(
        quality_and_work=quality,
        latency=latency,
        memory_and_runtime=memory,
        recovery=recovery,
    )


def _events() -> tuple[ModelInvocationEvent, ...]:
    return (
        ModelInvocationEvent(
            invocation_id="invoke-local",
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.LOCAL,
            model_identity="local-8b",
            provider_identity="mlx",
            outcome=InvocationOutcome.COMPLETED,
            semantic_contribution=SemanticContribution.ACTION_PLAN,
        ),
        # A completed model request that contributes no accepted semantic output
        # still consumes compute, but it is not a semantic turn.
        ModelInvocationEvent(
            invocation_id="invoke-tiny-rejected",
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.TINY,
            model_identity="tiny-specialist",
            provider_identity="local",
            outcome=InvocationOutcome.COMPLETED,
            semantic_contribution=SemanticContribution.NONE,
            wasted=True,
            wasted_reason="proposal rejected by deterministic validator",
        ),
        ModelInvocationEvent(
            invocation_id="invoke-frontier-failed",
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.FRONTIER,
            model_identity="frontier-model",
            provider_identity="remote",
            outcome=InvocationOutcome.FAILED,
            wasted=True,
            wasted_reason="transport failed after compute began",
            escalation_from_tier=ProviderTier.LOCAL,
        ),
        ModelInvocationEvent(
            invocation_id="invoke-background-senior",
            scope=ExecutionScope.BACKGROUND,
            provider_tier=ProviderTier.SENIOR,
            model_identity="senior-local-27b",
            provider_identity="mlx",
            outcome=InvocationOutcome.COMPLETED,
            semantic_contribution=SemanticContribution.EVIDENCE_INTERPRETATION,
        ),
    )


def _trial(
    *,
    correctness: CorrectnessResult | None = None,
    evidence: tuple[EvidenceReference, ...] | None = None,
    metrics: BenchmarkMetrics | None = None,
    events: tuple[ModelInvocationEvent, ...] | None = None,
) -> OptimizationBenchmarkTrial:
    objective = _objective()
    evidence = evidence if evidence is not None else (_evidence(objective),)
    correctness = correctness or CorrectnessResult(
        outcome=CorrectnessOutcome.PASS,
        evidence_refs=(evidence[0].ref_id,),
    )
    return OptimizationBenchmarkTrial.from_events(
        identity=_identity(),
        objective=objective,
        correctness=correctness,
        invocation_events=events or _events(),
        evidence=evidence,
        metrics=metrics or _metrics(),
    )


def test_trial_is_versioned_json_safe_and_has_stable_artifact_identity():
    trial = _trial()
    payload = trial.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["identity"]["schema_version"] == SCHEMA_VERSION
    assert payload["metrics"]["latency"]["model_wall_ms"] == {
        "derivation": None,
        "exposure": [],
        "provenance": "unavailable",
        "value": None,
    }
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_sha256(payload) == canonical_sha256(trial)
    assert trial.digest == artifact_identity(payload)
    assert trial.digest.startswith("sha256:") and len(trial.digest) == 71

    with pytest.raises(ValueError, match="signed 64-bit"):
        canonical_json({"unbounded": 1 << 80})


def test_semantic_turns_derive_only_from_accepted_model_outputs_by_scope_and_tier():
    counts = derive_invocation_counts(_events())

    assert counts.model_invocations.total == 4
    assert counts.semantic_turns.total == 2
    assert counts.semantic_turns.counts["foreground"]["local"] == 1
    assert counts.semantic_turns.counts["foreground"]["tiny"] == 0
    assert counts.semantic_turns.counts["background"]["senior"] == 1
    assert counts.failed_invocations.total == 1
    assert counts.wasted_invocations.total == 2
    assert counts.escalations == 1
    assert counts.expensive_interventions == 1
    # Seven deterministic sub-operations are work, but add no semantic turns.
    assert _trial().metrics.quality_and_work["deterministic_suboperations"].value == 7
    assert _trial().invocation_counts.semantic_turns.total == 2


def test_failed_or_wasted_invocation_cannot_claim_an_accepted_semantic_turn():
    with pytest.raises(ValueError, match="completed invocation"):
        ModelInvocationEvent(
            invocation_id="bad",
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.LOCAL,
            model_identity="model",
            provider_identity="provider",
            outcome=InvocationOutcome.FAILED,
            semantic_contribution=SemanticContribution.DECISION,
        )

    with pytest.raises(ValueError, match="cannot be marked wasted"):
        replace(_events()[0], wasted=True, wasted_reason="bad accounting")


def test_correctness_pass_requires_resolving_backend_owned_exact_objective_evidence():
    with pytest.raises(ValueError, match="requires objective evidence"):
        CorrectnessResult(outcome=CorrectnessOutcome.PASS)

    objective = _objective()
    untrusted = (_evidence(objective, backend_owned=False),)
    with pytest.raises(ValueError, match="backend-owned"):
        _trial(evidence=untrusted)

    wrong_objective = (_evidence(_objective(user_message_id="other-message")),)
    with pytest.raises(ValueError, match="exact objective"):
        _trial(evidence=wrong_objective)

    wrong_criteria = (_evidence(objective, acceptance_criteria_sha256="d" * 64),)
    with pytest.raises(ValueError, match="acceptance criteria"):
        _trial(evidence=wrong_criteria)


def test_unverified_or_failed_outcomes_do_not_require_objective_evidence():
    for outcome in (CorrectnessOutcome.UNVERIFIED, CorrectnessOutcome.FAIL):
        trial = _trial(
            correctness=CorrectnessResult(outcome=outcome),
            evidence=(),
        )
        assert trial.correctness.outcome is outcome


def test_metric_provenance_preserves_null_and_rejects_ambiguous_zero():
    unavailable = MetricMeasurement(value=None, provenance=MetricProvenance.UNAVAILABLE)
    not_applicable = MetricMeasurement(value=None, provenance=MetricProvenance.NOT_APPLICABLE)
    assert unavailable.to_dict()["value"] is None
    assert not_applicable.to_dict()["value"] is None

    with pytest.raises(ValueError, match="without exposure"):
        MetricMeasurement(value=0, provenance=MetricProvenance.MEASURED)
    with pytest.raises(ValueError, match="preserve value=null"):
        MetricMeasurement(value=0, provenance=MetricProvenance.UNAVAILABLE)
    with pytest.raises(ValueError, match="require a value"):
        MetricMeasurement(
            value=None,
            provenance=MetricProvenance.MEASURED,
            exposure=("run:1",),
        )
    with pytest.raises(ValueError, match="derivation"):
        MetricMeasurement(
            value=1,
            provenance=MetricProvenance.DERIVED,
            exposure=("event:1",),
        )

    derived = MetricMeasurement(
        value=2,
        provenance=MetricProvenance.DERIVED,
        exposure=("event:1", "event:2"),
        derivation="count accepted model invocation events",
    )
    synthetic = MetricMeasurement(
        value=3,
        provenance=MetricProvenance.SYNTHETIC,
        exposure=("fixture:test",),
    )
    assert derived.to_dict()["provenance"] == "derived"
    assert synthetic.to_dict()["provenance"] == "synthetic"


@pytest.mark.parametrize("value", [-1, -0.5, float("nan"), float("inf"), float("-inf")])
def test_negative_nan_and_infinite_metrics_are_rejected(value):
    with pytest.raises(ValueError, match="negative|NaN|infinite"):
        _measurement(value)


def test_required_metric_groups_reject_omissions_and_count_inconsistencies():
    quality = unavailable_metric_group(QUALITY_AND_WORK_METRICS)
    quality.pop("tool_calls")
    with pytest.raises(ValueError, match="omits required metrics: tool_calls"):
        BenchmarkMetrics(
            quality_and_work=quality,
            latency=unavailable_metric_group(LATENCY_METRICS),
            memory_and_runtime=unavailable_metric_group(MEMORY_AND_RUNTIME_METRICS),
            recovery=unavailable_metric_group(RECOVERY_METRICS),
        )

    quality = unavailable_metric_group(QUALITY_AND_WORK_METRICS)
    quality["tool_calls"] = _measurement(1)
    quality["failed_tool_calls"] = _measurement(2)
    with pytest.raises(ValueError, match="failed_tool_calls cannot exceed tool_calls"):
        BenchmarkMetrics(
            quality_and_work=quality,
            latency=unavailable_metric_group(LATENCY_METRICS),
            memory_and_runtime=unavailable_metric_group(MEMORY_AND_RUNTIME_METRICS),
            recovery=unavailable_metric_group(RECOVERY_METRICS),
        )

    quality["failed_tool_calls"] = _measurement(0)
    quality["input_tokens"] = _measurement(10)
    quality["cached_tokens"] = _measurement(8)
    quality["newly_evaluated_tokens"] = _measurement(4)
    with pytest.raises(ValueError, match="cannot exceed input_tokens"):
        BenchmarkMetrics(
            quality_and_work=quality,
            latency=unavailable_metric_group(LATENCY_METRICS),
            memory_and_runtime=unavailable_metric_group(MEMORY_AND_RUNTIME_METRICS),
            recovery=unavailable_metric_group(RECOVERY_METRICS),
        )


def test_reported_invocation_counts_must_match_explicit_events():
    trial = _trial()
    counts = trial.invocation_counts
    altered_model_matrix = {
        scope: dict(tiers) for scope, tiers in counts.model_invocations.counts.items()
    }
    altered_model_matrix["foreground"]["tiny"] += 1
    tampered = InvocationCounts(
        model_invocations=InvocationCountMatrix(altered_model_matrix),
        semantic_turns=counts.semantic_turns,
        failed_invocations=counts.failed_invocations,
        wasted_invocations=counts.wasted_invocations,
        escalations=counts.escalations,
        expensive_interventions=counts.expensive_interventions,
    )

    with pytest.raises(ValueError, match="do not match explicit invocation events"):
        OptimizationBenchmarkTrial(
            identity=trial.identity,
            objective=trial.objective,
            correctness=trial.correctness,
            invocation_events=trial.invocation_events,
            invocation_counts=tampered,
            evidence=trial.evidence,
            metrics=trial.metrics,
        )


def test_retries_must_reference_an_earlier_counted_invocation():
    orphan = ModelInvocationEvent(
        invocation_id="retry",
        scope=ExecutionScope.FOREGROUND,
        provider_tier=ProviderTier.LOCAL,
        model_identity="model",
        provider_identity="provider",
        outcome=InvocationOutcome.FAILED,
        retry_of="missing",
    )
    with pytest.raises(ValueError, match="earlier invocation"):
        derive_invocation_counts((orphan,))


def test_identity_omissions_and_malformed_hashes_are_rejected():
    with pytest.raises(ValueError, match="hardware_identity"):
        _identity(hardware_identity="")
    with pytest.raises(ValueError, match="benchmark_configuration_sha256"):
        _identity(benchmark_configuration_sha256="not-a-hash")
    with pytest.raises(ValueError, match="replicate"):
        _identity(replicate=-1)


def test_correctness_pass_rejects_ambiguous_recovery_outcome():
    metrics = _metrics()
    recovery = dict(metrics.recovery)
    recovery["ambiguous_outcome"] = _measurement(True)
    ambiguous = BenchmarkMetrics(
        quality_and_work=metrics.quality_and_work,
        latency=metrics.latency,
        memory_and_runtime=metrics.memory_and_runtime,
        recovery=recovery,
    )
    with pytest.raises(ValueError, match="ambiguous outcome"):
        _trial(metrics=ambiguous)


def test_complete_unavailable_vector_does_not_fabricate_zero_exposure():
    metrics = unavailable_metrics()
    assert all(
        item.value is None and item.provenance is MetricProvenance.UNAVAILABLE
        for group in (
            metrics.quality_and_work,
            metrics.latency,
            metrics.memory_and_runtime,
            metrics.recovery,
        )
        for item in group.values()
    )
