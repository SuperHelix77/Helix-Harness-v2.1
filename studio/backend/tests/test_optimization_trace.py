# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for explicit model-invocation trace lifecycle primitives."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from core.helix_engine.optimization_benchmark import (
    ExecutionScope,
    InvocationOutcome,
    ProviderTier,
    SemanticContribution,
    derive_invocation_counts,
)
from core.helix_engine.optimization_trace import (
    DurableOptimizationTraceRequest,
    OptimizationTraceConfig,
    OptimizationTraceStatus,
    OptimizationTraceRecorder,
    optimization_trace_envelope,
    parse_optimization_trace_envelope,
)


def _recorder(model: str = "mlx/test-model") -> OptimizationTraceRecorder:
    return OptimizationTraceRecorder(
        OptimizationTraceConfig(
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.LOCAL,
            model_identity=model,
            provider_identity="local-mlx",
        )
    )


def test_recorder_lifecycle_retry_order_and_validation():
    recorder = _recorder()
    first = recorder.begin_invocation()
    first_event = first.complete_wasted("empty_model_output")
    retry = recorder.begin_invocation(retry_of=first.invocation_id)
    retry_event = retry.complete_final_answer()

    events = recorder.snapshot()
    assert events == (first_event, retry_event)
    assert retry_event.retry_of == first_event.invocation_id
    assert retry_event.semantic_contribution is SemanticContribution.FINAL_ANSWER
    assert derive_invocation_counts(events).model_invocations.total == 2

    with pytest.raises(RuntimeError, match="already terminal"):
        retry.fail()
    with pytest.raises(ValueError, match="earlier recorded"):
        recorder.begin_invocation(retry_of="missing")


def test_recorder_rejects_invalid_configuration_and_event_combinations():
    with pytest.raises(TypeError, match="ExecutionScope"):
        OptimizationTraceConfig(  # type: ignore[arg-type]
            scope="foreground",
            provider_tier=ProviderTier.LOCAL,
            model_identity="model",
            provider_identity="provider",
        )

    handle = _recorder().begin_invocation()
    with pytest.raises(ValueError, match="accepted semantic turn"):
        handle.complete(
            SemanticContribution.ACTION_PLAN,
            wasted=True,
            wasted_reason="contradiction",
        )
    # Failed validation must leave the handle open so the caller can classify it.
    event = handle.discard("validation_recovered")
    assert event.outcome is InvocationOutcome.DISCARDED


def test_recorders_are_thread_safe_and_isolated():
    left = _recorder("left")
    right = _recorder("right")

    def record_many(recorder: OptimizationTraceRecorder, count: int) -> None:
        for _ in range(count):
            recorder.begin_invocation().complete_action_plan()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(record_many, left if index % 2 == 0 else right, 40)
            for index in range(8)
        ]
        for future in futures:
            future.result()

    left_events = left.snapshot()
    right_events = right.snapshot()
    assert len(left_events) == len(right_events) == 160
    assert not ({event.invocation_id for event in left_events} & {
        event.invocation_id for event in right_events
    })
    assert {event.model_identity for event in left_events} == {"left"}
    assert {event.model_identity for event in right_events} == {"right"}


def test_trace_wire_round_trip_rederives_counts_and_rejects_tampering():
    recorder = _recorder()
    recorder.begin_invocation().complete_action_plan()
    recorder.begin_invocation().complete_final_answer()

    wire = optimization_trace_envelope(recorder).to_dict()
    parsed = parse_optimization_trace_envelope(wire)

    assert parsed.status is OptimizationTraceStatus.AVAILABLE
    assert parsed.all_handles_settled is True
    assert parsed.invocation_counts.model_invocations.total == 2
    assert parsed.invocation_counts.semantic_turns.total == 2

    wire["invocation_counts"]["model_invocations"]["counts"]["foreground"][
        "local"
    ] = 0
    with pytest.raises(ValueError):
        parse_optimization_trace_envelope(wire)


def test_empty_and_open_handle_traces_are_never_available_zero_success():
    empty = optimization_trace_envelope(_recorder())
    assert empty.status is OptimizationTraceStatus.UNAVAILABLE
    assert empty.invocation_counts is None
    assert empty.all_handles_settled is True

    recorder = _recorder()
    recorder.begin_invocation()
    incomplete = parse_optimization_trace_envelope(
        optimization_trace_envelope(recorder).to_dict()
    )
    assert incomplete.status is OptimizationTraceStatus.INCOMPLETE
    assert incomplete.all_handles_settled is False
    assert incomplete.error == "open_invocation_handles"

    failed = _recorder()
    failed.begin_invocation().fail("backend_error")
    failed_trace = optimization_trace_envelope(failed)
    assert failed_trace.invocation_counts.model_invocations.total == 1
    assert failed_trace.invocation_counts.failed_invocations.total == 1
    assert failed_trace.invocation_counts.wasted_invocations.total == 1

    cancelled = _recorder()
    cancelled.begin_invocation().cancel("stopped")
    cancelled_trace = optimization_trace_envelope(cancelled)
    assert cancelled_trace.invocation_counts.model_invocations.total == 1
    assert cancelled_trace.invocation_counts.wasted_invocations.total == 1


def test_invalidated_recorder_never_exposes_partial_available_evidence():
    recorder = _recorder()
    recorder.begin_invocation().complete_action_plan()
    recorder.invalidate("trace_finalize_failed")

    trace = optimization_trace_envelope(recorder)

    assert trace.status is OptimizationTraceStatus.UNAVAILABLE
    assert trace.invocation_events == ()
    assert trace.invocation_counts is None
    assert trace.error == "trace_finalize_failed"


def test_trace_invalidation_is_first_failure_wins_and_thread_safe():
    recorder = _recorder()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(recorder.invalidate, ["first", "second", "third", "fourth"]))

    assert recorder.invalid_reason in {"first", "second", "third", "fourth"}
    first = recorder.invalid_reason
    recorder.invalidate("later")
    assert recorder.invalid_reason == first


def test_durable_trace_environment_configuration_is_strict_and_fail_closed():
    assert (
        DurableOptimizationTraceRequest.from_environment(
            enabled=None,
            provider_tier="senior",
        )
        is None
    )
    assert (
        DurableOptimizationTraceRequest.from_environment(
            enabled="true",
            provider_tier="senior",
        )
        is None
    )
    enabled = DurableOptimizationTraceRequest.from_environment(
        enabled="1",
        provider_tier="senior",
    )
    assert enabled.provider_tier is ProviderTier.SENIOR
    invalid = DurableOptimizationTraceRequest.from_environment(
        enabled="1",
        provider_tier="not-a-tier",
    )
    assert invalid.provider_tier is None
    assert invalid.configuration_error == "invalid_provider_tier"
