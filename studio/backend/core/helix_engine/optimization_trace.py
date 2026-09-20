# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit, thread-safe model-invocation tracing primitives.

The recorder is deliberately request-scoped and opt-in.  Callers pass it down
to the model boundary they want to observe; this module does not install a
process-global collector or depend on implicit context propagation.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core.helix_engine.optimization_benchmark import (
    MAX_INVOCATION_EVENTS,
    ExecutionScope,
    InvocationCountMatrix,
    InvocationCounts,
    InvocationOutcome,
    ModelInvocationEvent,
    ProviderTier,
    SemanticContribution,
    canonical_json,
    derive_invocation_counts,
)


OPTIMIZATION_TRACE_SCHEMA_VERSION = "helix.optimization-trace.v1"
OPTIMIZATION_TRACE_SCOPE_STATE_KEY = "_helix_optimization_trace_v1"
OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY = "_helix_optimization_trace_v1"
MAX_OPTIMIZATION_TRACE_WIRE_BYTES = 4 * 1024 * 1024


class OptimizationTraceStatus(str, Enum):
    AVAILABLE = "available"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DurableOptimizationTraceRequest:
    """Server-created authorization/configuration carried only in ASGI scope state.

    Public headers and request payload fields never construct this value.  An
    invalid explicitly-enabled benchmark configuration is represented as an
    unavailable trace request instead of being replaced with a default tier.
    """

    provider_tier: ProviderTier | None = None
    configuration_error: str | None = None

    def __post_init__(self) -> None:
        if self.provider_tier is not None and not isinstance(self.provider_tier, ProviderTier):
            raise TypeError("trace request provider_tier must be a ProviderTier")
        if self.configuration_error is not None:
            _required_text(
                self.configuration_error,
                "trace request configuration_error",
                maximum=500,
            )
        if (self.provider_tier is None) == (self.configuration_error is None):
            raise ValueError(
                "trace request must contain exactly one of provider_tier or configuration_error"
            )

    @classmethod
    def from_environment(
        cls,
        *,
        enabled: str | None,
        provider_tier: str | None,
    ) -> "DurableOptimizationTraceRequest | None":
        if enabled != "1":
            return None
        try:
            tier = ProviderTier(provider_tier or "")
        except ValueError:
            return cls(configuration_error="invalid_provider_tier")
        return cls(provider_tier=tier)


def _required_text(value: str, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return value


@dataclass(frozen=True)
class OptimizationTraceConfig:
    """Immutable identity applied to every event emitted by one recorder."""

    scope: ExecutionScope
    provider_tier: ProviderTier
    model_identity: str
    provider_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ExecutionScope):
            raise TypeError("trace scope must be an ExecutionScope")
        if not isinstance(self.provider_tier, ProviderTier):
            raise TypeError("trace provider_tier must be a ProviderTier")
        _required_text(self.model_identity, "trace model_identity", maximum=2_000)
        _required_text(self.provider_identity, "trace provider_identity", maximum=2_000)


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} has invalid fields (missing={missing}, extra={extra})")


def _parse_count_matrix(value: Any, label: str) -> InvocationCountMatrix:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    _strict_keys(value, {"counts"}, label)
    counts = value["counts"]
    if not isinstance(counts, Mapping):
        raise TypeError(f"{label}.counts must be an object")
    copied: dict[str, dict[str, int]] = {}
    for scope, tiers in counts.items():
        if not isinstance(scope, str) or not isinstance(tiers, Mapping):
            raise TypeError(f"{label}.counts must map strings to objects")
        copied[scope] = dict(tiers)
    return InvocationCountMatrix(counts=copied)


def _parse_invocation_counts(value: Any) -> InvocationCounts:
    if not isinstance(value, Mapping):
        raise TypeError("trace counts must be an object")
    expected = {
        "model_invocations",
        "semantic_turns",
        "failed_invocations",
        "wasted_invocations",
        "escalations",
        "expensive_interventions",
    }
    _strict_keys(value, expected, "trace counts")
    return InvocationCounts(
        model_invocations=_parse_count_matrix(
            value["model_invocations"], "trace counts.model_invocations"
        ),
        semantic_turns=_parse_count_matrix(
            value["semantic_turns"], "trace counts.semantic_turns"
        ),
        failed_invocations=_parse_count_matrix(
            value["failed_invocations"], "trace counts.failed_invocations"
        ),
        wasted_invocations=_parse_count_matrix(
            value["wasted_invocations"], "trace counts.wasted_invocations"
        ),
        escalations=value["escalations"],
        expensive_interventions=value["expensive_interventions"],
    )


def _parse_invocation_event(value: Any) -> ModelInvocationEvent:
    if not isinstance(value, Mapping):
        raise TypeError("trace invocation event must be an object")
    expected = {
        "invocation_id",
        "scope",
        "provider_tier",
        "model_identity",
        "provider_identity",
        "outcome",
        "semantic_contribution",
        "wasted",
        "wasted_reason",
        "retry_of",
        "escalation_from_tier",
    }
    _strict_keys(value, expected, "trace invocation event")
    escalation = value["escalation_from_tier"]
    return ModelInvocationEvent(
        invocation_id=value["invocation_id"],
        scope=ExecutionScope(value["scope"]),
        provider_tier=ProviderTier(value["provider_tier"]),
        model_identity=value["model_identity"],
        provider_identity=value["provider_identity"],
        outcome=InvocationOutcome(value["outcome"]),
        semantic_contribution=SemanticContribution(value["semantic_contribution"]),
        wasted=value["wasted"],
        wasted_reason=value["wasted_reason"],
        retry_of=value["retry_of"],
        escalation_from_tier=(
            ProviderTier(escalation) if escalation is not None else None
        ),
    )


@dataclass(frozen=True)
class OptimizationTraceEnvelope:
    """Bounded private wire record persisted as benchmark evidence.

    Counts are carried for convenience but are never trusted: construction and
    parsing both recompute them from the validated event records.  An empty or
    open-handle trace cannot claim availability, so it cannot be interpreted as
    a successful zero-invocation run.
    """

    status: OptimizationTraceStatus
    invocation_events: tuple[ModelInvocationEvent, ...]
    invocation_counts: InvocationCounts | None
    all_handles_settled: bool
    error: str | None = None
    schema_version: str = OPTIMIZATION_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPTIMIZATION_TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"trace schema_version must be {OPTIMIZATION_TRACE_SCHEMA_VERSION}"
            )
        if not isinstance(self.status, OptimizationTraceStatus):
            raise TypeError("trace status must be an OptimizationTraceStatus")
        if not isinstance(self.invocation_events, tuple):
            raise TypeError("trace invocation_events must be a tuple")
        if not isinstance(self.all_handles_settled, bool):
            raise TypeError("trace all_handles_settled must be boolean")
        derived = derive_invocation_counts(self.invocation_events)
        if self.status is OptimizationTraceStatus.AVAILABLE:
            if not self.all_handles_settled or not self.invocation_events:
                raise ValueError("an available trace must be non-empty and fully settled")
            if self.error is not None:
                raise ValueError("an available trace cannot carry an error")
        elif self.status is OptimizationTraceStatus.INCOMPLETE:
            if self.all_handles_settled:
                raise ValueError("an incomplete trace must have an open invocation handle")
            _required_text(self.error or "", "trace error", maximum=500)
        else:
            if self.invocation_events or self.invocation_counts is not None:
                raise ValueError("an unavailable trace cannot carry invocation evidence")
            _required_text(self.error or "", "trace error", maximum=500)
            return
        if not isinstance(self.invocation_counts, InvocationCounts):
            raise TypeError("available/incomplete trace invocation_counts are required")
        if self.invocation_counts.to_dict() != derived.to_dict():
            raise ValueError("trace invocation_counts do not match invocation_events")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "invocation_events": [event.to_dict() for event in self.invocation_events],
            "invocation_counts": (
                self.invocation_counts.to_dict()
                if self.invocation_counts is not None
                else None
            ),
            "all_handles_settled": self.all_handles_settled,
            "error": self.error,
        }
        encoded = canonical_json(value).encode("utf-8")
        if len(encoded) > MAX_OPTIMIZATION_TRACE_WIRE_BYTES:
            raise ValueError(
                f"optimization trace exceeds {MAX_OPTIMIZATION_TRACE_WIRE_BYTES} wire bytes"
            )
        return value


def parse_optimization_trace_envelope(value: Any) -> OptimizationTraceEnvelope:
    """Strictly parse and re-derive one private trace envelope."""

    if not isinstance(value, Mapping):
        raise TypeError("optimization trace envelope must be an object")
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_OPTIMIZATION_TRACE_WIRE_BYTES:
        raise ValueError(
            f"optimization trace exceeds {MAX_OPTIMIZATION_TRACE_WIRE_BYTES} wire bytes"
        )
    expected = {
        "schema_version",
        "status",
        "invocation_events",
        "invocation_counts",
        "all_handles_settled",
        "error",
    }
    _strict_keys(value, expected, "optimization trace envelope")
    raw_events = value["invocation_events"]
    if not isinstance(raw_events, list):
        raise TypeError("trace invocation_events must be an array")
    if len(raw_events) > MAX_INVOCATION_EVENTS:
        raise ValueError(
            f"a trace may contain at most {MAX_INVOCATION_EVENTS} invocations"
        )
    status = OptimizationTraceStatus(value["status"])
    events = tuple(_parse_invocation_event(event) for event in raw_events)
    counts = (
        None
        if value["invocation_counts"] is None
        else _parse_invocation_counts(value["invocation_counts"])
    )
    return OptimizationTraceEnvelope(
        schema_version=value["schema_version"],
        status=status,
        invocation_events=events,
        invocation_counts=counts,
        all_handles_settled=value["all_handles_settled"],
        error=value["error"],
    )


class ModelInvocationHandle:
    """A single-use lifecycle handle returned by :meth:`begin_invocation`."""

    __slots__ = ("_event", "_invocation_id", "_recorder", "_retry_of")

    def __init__(
        self,
        recorder: "OptimizationTraceRecorder",
        invocation_id: str,
        retry_of: str | None,
    ) -> None:
        self._recorder = recorder
        self._invocation_id = invocation_id
        self._retry_of = retry_of
        self._event: ModelInvocationEvent | None = None

    @property
    def invocation_id(self) -> str:
        return self._invocation_id

    @property
    def retry_of(self) -> str | None:
        return self._retry_of

    @property
    def is_terminal(self) -> bool:
        return self._recorder._is_terminal(self)

    @property
    def event(self) -> ModelInvocationEvent | None:
        return self._recorder._terminal_event(self)

    def complete(
        self,
        semantic_contribution: SemanticContribution = SemanticContribution.NONE,
        *,
        wasted: bool = False,
        wasted_reason: str | None = None,
    ) -> ModelInvocationEvent:
        return self._recorder._finalize(
            self,
            outcome=InvocationOutcome.COMPLETED,
            semantic_contribution=semantic_contribution,
            wasted=wasted,
            wasted_reason=wasted_reason,
        )

    def complete_final_answer(self) -> ModelInvocationEvent:
        return self.complete(SemanticContribution.FINAL_ANSWER)

    def complete_action_plan(self) -> ModelInvocationEvent:
        return self.complete(SemanticContribution.ACTION_PLAN)

    def complete_wasted(self, reason: str) -> ModelInvocationEvent:
        return self.complete(wasted=True, wasted_reason=reason)

    def complete_continuation(self) -> ModelInvocationEvent:
        """Record a retained length/adaptive segment without claiming a turn."""
        return self.complete()

    def fail(self, reason: str = "model_generator_exception") -> ModelInvocationEvent:
        return self._recorder._finalize(
            self,
            outcome=InvocationOutcome.FAILED,
            wasted=True,
            wasted_reason=reason,
        )

    def cancel(self, reason: str = "cooperative_cancellation") -> ModelInvocationEvent:
        return self._recorder._finalize(
            self,
            outcome=InvocationOutcome.CANCELLED,
            wasted=True,
            wasted_reason=reason,
        )

    def discard(self, reason: str = "invocation_abandoned") -> ModelInvocationEvent:
        return self._recorder._finalize(
            self,
            outcome=InvocationOutcome.DISCARDED,
            wasted=True,
            wasted_reason=reason,
        )


class OptimizationTraceRecorder:
    """Request-owned event collector safe for use by concurrent producers.

    Retry links are accepted only after their target has been appended, which
    makes every snapshot directly valid for ``derive_invocation_counts``.
    """

    def __init__(self, config: OptimizationTraceConfig) -> None:
        if not isinstance(config, OptimizationTraceConfig):
            raise TypeError("config must be an OptimizationTraceConfig")
        self.config = config
        self._lock = threading.RLock()
        self._events: list[ModelInvocationEvent] = []
        self._event_ids: set[str] = set()
        self._open: dict[str, ModelInvocationHandle] = {}
        self._recorder_id = uuid.uuid4().hex
        self._next_id = 0
        self._invalid_reason: str | None = None

    def invalidate(self, reason: str) -> None:
        """Permanently reject this recorder as benchmark evidence.

        Instrumentation is not allowed to affect model execution.  Callers use
        this latch after any recorder-side failure, then continue the ordinary
        generation path.  Keeping the first reason makes concurrent failures
        deterministic without exposing any partial events as available data.
        """
        _required_text(reason, "trace invalidation reason", maximum=500)
        with self._lock:
            if self._invalid_reason is None:
                self._invalid_reason = reason

    @property
    def invalid_reason(self) -> str | None:
        with self._lock:
            return self._invalid_reason

    def begin_invocation(self, *, retry_of: str | None = None) -> ModelInvocationHandle:
        with self._lock:
            if len(self._events) + len(self._open) >= MAX_INVOCATION_EVENTS:
                raise ValueError(
                    f"a trace may contain at most {MAX_INVOCATION_EVENTS} invocations"
                )
            if retry_of is not None:
                _required_text(retry_of, "retry_of", maximum=1_000)
                if retry_of not in self._event_ids:
                    raise ValueError("retry_of must identify an earlier recorded invocation")
            invocation_id = f"{self._recorder_id}:{self._next_id}"
            self._next_id += 1
            handle = ModelInvocationHandle(self, invocation_id, retry_of)
            self._open[invocation_id] = handle
            return handle

    def snapshot(self) -> tuple[ModelInvocationEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def snapshot_state(self) -> tuple[tuple[ModelInvocationEvent, ...], bool]:
        """Atomically return terminal events and whether any handle remains open."""
        with self._lock:
            return tuple(self._events), not self._open

    def snapshot_trace_state(
        self,
    ) -> tuple[tuple[ModelInvocationEvent, ...], bool, str | None]:
        """Atomically return events, settlement, and the evidence-invalid latch."""
        with self._lock:
            return tuple(self._events), not self._open, self._invalid_reason

    @property
    def all_handles_settled(self) -> bool:
        with self._lock:
            return not self._open

    @property
    def open_handle_count(self) -> int:
        with self._lock:
            return len(self._open)

    def _is_terminal(self, handle: ModelInvocationHandle) -> bool:
        with self._lock:
            self._require_owner(handle)
            return handle._event is not None

    def _terminal_event(self, handle: ModelInvocationHandle) -> ModelInvocationEvent | None:
        with self._lock:
            self._require_owner(handle)
            return handle._event

    def _require_owner(self, handle: ModelInvocationHandle) -> None:
        if not isinstance(handle, ModelInvocationHandle) or handle._recorder is not self:
            raise ValueError("invocation handle belongs to a different recorder")

    def _finalize(
        self,
        handle: ModelInvocationHandle,
        *,
        outcome: InvocationOutcome,
        semantic_contribution: SemanticContribution = SemanticContribution.NONE,
        wasted: bool,
        wasted_reason: str | None,
    ) -> ModelInvocationEvent:
        with self._lock:
            self._require_owner(handle)
            if handle._event is not None or self._open.get(handle.invocation_id) is not handle:
                raise RuntimeError("invocation handle is already terminal")
            event = ModelInvocationEvent(
                invocation_id=handle.invocation_id,
                scope=self.config.scope,
                provider_tier=self.config.provider_tier,
                model_identity=self.config.model_identity,
                provider_identity=self.config.provider_identity,
                outcome=outcome,
                semantic_contribution=semantic_contribution,
                wasted=wasted,
                wasted_reason=wasted_reason,
                retry_of=handle.retry_of,
            )
            del self._open[handle.invocation_id]
            handle._event = event
            self._events.append(event)
            self._event_ids.add(event.invocation_id)
            return event


def optimization_trace_envelope(
    recorder: OptimizationTraceRecorder,
) -> OptimizationTraceEnvelope:
    """Snapshot one recorder without converting missing evidence into zero."""

    if not isinstance(recorder, OptimizationTraceRecorder):
        raise TypeError("recorder must be an OptimizationTraceRecorder")
    events, settled, invalid_reason = recorder.snapshot_trace_state()
    if invalid_reason is not None:
        return unavailable_optimization_trace(
            invalid_reason,
            all_handles_settled=settled,
        )
    if not settled:
        return OptimizationTraceEnvelope(
            status=OptimizationTraceStatus.INCOMPLETE,
            invocation_events=events,
            invocation_counts=derive_invocation_counts(events),
            all_handles_settled=False,
            error="open_invocation_handles",
        )
    if not events:
        return unavailable_optimization_trace(
            "no_model_invocations_recorded",
            all_handles_settled=True,
        )
    return OptimizationTraceEnvelope(
        status=OptimizationTraceStatus.AVAILABLE,
        invocation_events=events,
        invocation_counts=derive_invocation_counts(events),
        all_handles_settled=True,
    )


def unavailable_optimization_trace(
    reason: str,
    *,
    all_handles_settled: bool = False,
) -> OptimizationTraceEnvelope:
    return OptimizationTraceEnvelope(
        status=OptimizationTraceStatus.UNAVAILABLE,
        invocation_events=(),
        invocation_counts=None,
        all_handles_settled=all_handles_settled,
        error=reason,
    )
