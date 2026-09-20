# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, evidence-aware wire records for Helix optimization benchmarks.

This module is deliberately independent of production routing.  It gives the
research phases one fail-closed envelope for recording trials without turning a
missing observation into a zero or a model-authored assertion into correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "helix.optimization-benchmark.v1"
CANONICALIZATION_VERSION = "helix.canonical-json.v1"

MAX_STRING_LENGTH = 32_768
MAX_COLLECTION_ITEMS = 10_000
MAX_JSON_DEPTH = 16
MAX_JSON_INTEGER = (1 << 63) - 1
MAX_INVOCATION_EVENTS = 10_000
MAX_EVIDENCE_REFS = 512
MAX_METRICS_PER_GROUP = 256

_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class MetricProvenance(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    SYNTHETIC = "synthetic"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ExecutionScope(str, Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class ProviderTier(str, Enum):
    TINY = "tiny"
    LOCAL = "local"
    SENIOR = "senior"
    FRONTIER = "frontier"


class InvocationOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCARDED = "discarded"


class SemanticContribution(str, Enum):
    NONE = "none"
    INTENT = "intent"
    ACTION_PLAN = "action_plan"
    DECISION = "decision"
    EVIDENCE_INTERPRETATION = "evidence_interpretation"
    FINAL_ANSWER = "final_answer"


class CorrectnessOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"


def _text(value: str, label: str, *, maximum: int = MAX_STRING_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return value


def _sha256(value: str, label: str) -> str:
    _text(value, label, maximum=71)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a 64-character sha256 identity")
    return value


def _json_safe(value: Any, *, depth: int = 0, path: str = "$") -> Any:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"{path} exceeds maximum JSON depth {MAX_JSON_DEPTH}")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value), depth=depth, path=path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_JSON_INTEGER:
            raise ValueError(f"{path} exceeds the signed 64-bit JSON integer bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValueError(f"{path} exceeds {MAX_STRING_LENGTH} characters")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError(f"{path} exceeds {MAX_COLLECTION_ITEMS} members")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string JSON object key")
            if len(key) > MAX_STRING_LENGTH:
                raise ValueError(f"{path} contains an overlong key")
            result[key] = _json_safe(item, depth=depth + 1, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError(f"{path} exceeds {MAX_COLLECTION_ITEMS} items")
        return [
            _json_safe(item, depth=depth + 1, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON-safe type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return bounded deterministic UTF-8 JSON; NaN and infinity are rejected."""

    safe = _json_safe(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def artifact_identity(value: Any) -> str:
    """Return a self-describing digest suitable for artifact identity fields."""

    return f"sha256:{canonical_sha256(value)}"


class WireRecord:
    def to_dict(self) -> dict[str, Any]:
        safe = _json_safe(asdict(self))
        if not isinstance(safe, dict):  # pragma: no cover - dataclass invariant
            raise TypeError("wire record did not serialize to an object")
        return safe


@dataclass(frozen=True)
class ObjectiveIdentity(WireRecord):
    owner_subject: str
    thread_id: str
    user_message_id: str

    def __post_init__(self) -> None:
        _text(self.owner_subject, "owner_subject", maximum=1_000)
        _text(self.thread_id, "thread_id", maximum=1_000)
        _text(self.user_message_id, "user_message_id", maximum=1_000)

    @property
    def key(self) -> str:
        return artifact_identity(self.to_dict())


@dataclass(frozen=True)
class BenchmarkIdentity(WireRecord):
    benchmark_id: str
    scenario_id: str
    trial_id: str
    replicate: int
    runner_version: str
    frozen_control_identity: str
    candidate_source_snapshot_identity: str
    hardware_identity: str
    os_identity: str
    runtime_identity: str
    model_identity: str
    provider_identity: str
    checkpoint_identity: str
    quantization_identity: str
    context_identity: str
    kv_identity: str
    speculation_identity: str
    benchmark_configuration_sha256: str
    acceptance_criteria_sha256: str
    backend_overlay_identity: str | None = None
    packaged_app_identity: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for name in (
            "benchmark_id",
            "scenario_id",
            "trial_id",
            "runner_version",
            "frozen_control_identity",
            "candidate_source_snapshot_identity",
            "hardware_identity",
            "os_identity",
            "runtime_identity",
            "model_identity",
            "provider_identity",
            "checkpoint_identity",
            "quantization_identity",
            "context_identity",
            "kv_identity",
            "speculation_identity",
        ):
            _text(getattr(self, name), name, maximum=2_000)
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int) or self.replicate < 0:
            raise ValueError("replicate must be a non-negative integer")
        _sha256(self.benchmark_configuration_sha256, "benchmark_configuration_sha256")
        _sha256(self.acceptance_criteria_sha256, "acceptance_criteria_sha256")
        for name in ("backend_overlay_identity", "packaged_app_identity"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name, maximum=2_000)


@dataclass(frozen=True)
class EvidenceReference(WireRecord):
    ref_id: str
    kind: str
    subject: str
    objective_key: str
    acceptance_criteria_sha256: str
    content_sha256: str
    backend_owned: bool = True

    def __post_init__(self) -> None:
        _text(self.ref_id, "evidence ref_id", maximum=1_000)
        _text(self.kind, "evidence kind", maximum=200)
        _text(self.subject, "evidence subject", maximum=2_000)
        _sha256(self.objective_key, "evidence objective_key")
        _sha256(self.acceptance_criteria_sha256, "evidence acceptance_criteria_sha256")
        _sha256(self.content_sha256, "evidence content_sha256")
        if not isinstance(self.backend_owned, bool):
            raise TypeError("evidence backend_owned must be boolean")


@dataclass(frozen=True)
class CorrectnessResult(WireRecord):
    outcome: CorrectnessOutcome
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, CorrectnessOutcome):
            raise TypeError("correctness outcome must be a CorrectnessOutcome")
        if len(self.evidence_refs) > MAX_EVIDENCE_REFS:
            raise ValueError("too many correctness evidence references")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("correctness evidence references must be unique")
        for ref_id in self.evidence_refs:
            _text(ref_id, "correctness evidence ref", maximum=1_000)
        if self.outcome is CorrectnessOutcome.PASS and not self.evidence_refs:
            raise ValueError("correctness=pass requires objective evidence references")


@dataclass(frozen=True)
class MetricMeasurement(WireRecord):
    value: Any
    provenance: MetricProvenance
    exposure: tuple[str, ...] = ()
    derivation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, MetricProvenance):
            raise TypeError("metric provenance must be a MetricProvenance")
        if len(self.exposure) > MAX_EVIDENCE_REFS:
            raise ValueError("metric has too many exposure references")
        for ref in self.exposure:
            _text(ref, "metric exposure ref", maximum=2_000)

        absent = self.provenance in {
            MetricProvenance.UNAVAILABLE,
            MetricProvenance.NOT_APPLICABLE,
        }
        if absent:
            if self.value is not None:
                raise ValueError(f"{self.provenance.value} metrics must preserve value=null")
            if self.exposure:
                raise ValueError(f"{self.provenance.value} metrics cannot claim exposure")
            if self.derivation is not None:
                raise ValueError(f"{self.provenance.value} metrics cannot claim a derivation")
            return

        if self.value is None:
            raise ValueError("observed, derived, and synthetic metrics require a value")
        if not self.exposure:
            raise ValueError("a metric without exposure is unavailable, not zero")
        if self.provenance is MetricProvenance.DERIVED:
            _text(self.derivation or "", "derived metric derivation", maximum=4_000)
        elif self.derivation is not None:
            raise ValueError("only derived metrics may specify a derivation")

        if isinstance(self.value, bool):
            return
        if isinstance(self.value, int):
            if self.value < 0:
                raise ValueError("numeric metrics cannot be negative")
            return
        if isinstance(self.value, float):
            if not math.isfinite(self.value):
                raise ValueError("numeric metrics cannot be NaN or infinite")
            if self.value < 0:
                raise ValueError("numeric metrics cannot be negative")
            return
        if isinstance(self.value, str):
            _text(self.value, "metric string value")
            return
        raise TypeError("metric values must be null or a JSON scalar")


@dataclass(frozen=True)
class ModelInvocationEvent(WireRecord):
    invocation_id: str
    scope: ExecutionScope
    provider_tier: ProviderTier
    model_identity: str
    provider_identity: str
    outcome: InvocationOutcome
    semantic_contribution: SemanticContribution = SemanticContribution.NONE
    wasted: bool = False
    wasted_reason: str | None = None
    retry_of: str | None = None
    escalation_from_tier: ProviderTier | None = None

    def __post_init__(self) -> None:
        _text(self.invocation_id, "invocation_id", maximum=1_000)
        if not isinstance(self.scope, ExecutionScope):
            raise TypeError("invocation scope must be an ExecutionScope")
        if not isinstance(self.provider_tier, ProviderTier):
            raise TypeError("provider_tier must be a ProviderTier")
        _text(self.model_identity, "invocation model_identity", maximum=2_000)
        _text(self.provider_identity, "invocation provider_identity", maximum=2_000)
        if not isinstance(self.outcome, InvocationOutcome):
            raise TypeError("invocation outcome must be an InvocationOutcome")
        if not isinstance(self.semantic_contribution, SemanticContribution):
            raise TypeError("semantic_contribution must be a SemanticContribution")
        if not isinstance(self.wasted, bool):
            raise TypeError("wasted must be boolean")
        if self.semantic_contribution is not SemanticContribution.NONE:
            if self.outcome is not InvocationOutcome.COMPLETED:
                raise ValueError("only a completed invocation can contribute a semantic turn")
            if self.wasted:
                raise ValueError("an accepted semantic turn cannot be marked wasted")
        if self.wasted:
            _text(self.wasted_reason or "", "wasted_reason", maximum=2_000)
        elif self.wasted_reason is not None:
            raise ValueError("wasted_reason requires wasted=true")
        if self.retry_of is not None:
            _text(self.retry_of, "retry_of", maximum=1_000)
            if self.retry_of == self.invocation_id:
                raise ValueError("an invocation cannot retry itself")
        if self.escalation_from_tier is not None:
            if not isinstance(self.escalation_from_tier, ProviderTier):
                raise TypeError("escalation_from_tier must be a ProviderTier")
            rank = {tier: index for index, tier in enumerate(ProviderTier)}
            if rank[self.escalation_from_tier] >= rank[self.provider_tier]:
                raise ValueError("an escalation must move to a higher provider tier")

    @property
    def contributes_semantic_turn(self) -> bool:
        return self.semantic_contribution is not SemanticContribution.NONE


def _empty_count_matrix() -> dict[str, dict[str, int]]:
    return {
        scope.value: {tier.value: 0 for tier in ProviderTier}
        for scope in ExecutionScope
    }


@dataclass(frozen=True)
class InvocationCountMatrix(WireRecord):
    counts: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if set(self.counts) != {scope.value for scope in ExecutionScope}:
            raise ValueError("count matrix must contain foreground and background")
        expected_tiers = {tier.value for tier in ProviderTier}
        for scope, tiers in self.counts.items():
            if set(tiers) != expected_tiers:
                raise ValueError(f"count matrix {scope} must contain all provider tiers")
            for count in tiers.values():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("count matrix values must be non-negative integers")

    @property
    def total(self) -> int:
        return sum(count for tiers in self.counts.values() for count in tiers.values())


@dataclass(frozen=True)
class InvocationCounts(WireRecord):
    model_invocations: InvocationCountMatrix
    semantic_turns: InvocationCountMatrix
    failed_invocations: InvocationCountMatrix
    wasted_invocations: InvocationCountMatrix
    escalations: int
    expensive_interventions: int

    def __post_init__(self) -> None:
        for name in ("escalations", "expensive_interventions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.semantic_turns.total > self.model_invocations.total:
            raise ValueError("semantic turns cannot exceed model invocations")
        if self.failed_invocations.total > self.model_invocations.total:
            raise ValueError("failed invocations cannot exceed model invocations")
        if self.wasted_invocations.total > self.model_invocations.total:
            raise ValueError("wasted invocations cannot exceed model invocations")
        if self.escalations > self.model_invocations.total:
            raise ValueError("escalations cannot exceed model invocations")
        expensive = sum(
            self.semantic_turns.counts[scope.value][tier.value]
            for scope in ExecutionScope
            for tier in (ProviderTier.SENIOR, ProviderTier.FRONTIER)
        )
        if self.expensive_interventions != expensive:
            raise ValueError("expensive interventions must equal senior/frontier semantic turns")


def derive_invocation_counts(events: Sequence[ModelInvocationEvent]) -> InvocationCounts:
    """Derive semantic work only from explicit model-invocation events.

    Deterministic operations have no representation in this input and therefore
    cannot inflate semantic turns.  Failed and wasted counts intentionally
    overlap when both statements are true for one attempted request.
    """

    if len(events) > MAX_INVOCATION_EVENTS:
        raise ValueError(f"a trial may contain at most {MAX_INVOCATION_EVENTS} invocations")
    matrices = [_empty_count_matrix() for _ in range(4)]
    invocations, semantic, failed, wasted = matrices
    escalations = 0
    seen: set[str] = set()
    prior: set[str] = set()
    for event in events:
        if not isinstance(event, ModelInvocationEvent):
            raise TypeError("invocation events must be ModelInvocationEvent records")
        if event.invocation_id in seen:
            raise ValueError(f"duplicate invocation_id: {event.invocation_id}")
        seen.add(event.invocation_id)
        if event.retry_of is not None and event.retry_of not in prior:
            raise ValueError("retry_of must identify an earlier invocation in the trial")
        scope = event.scope.value
        tier = event.provider_tier.value
        invocations[scope][tier] += 1
        if event.contributes_semantic_turn:
            semantic[scope][tier] += 1
        if event.outcome is InvocationOutcome.FAILED:
            failed[scope][tier] += 1
        if event.wasted:
            wasted[scope][tier] += 1
        if event.escalation_from_tier is not None:
            escalations += 1
        prior.add(event.invocation_id)

    semantic_matrix = InvocationCountMatrix(semantic)
    return InvocationCounts(
        model_invocations=InvocationCountMatrix(invocations),
        semantic_turns=semantic_matrix,
        failed_invocations=InvocationCountMatrix(failed),
        wasted_invocations=InvocationCountMatrix(wasted),
        escalations=escalations,
        expensive_interventions=sum(
            semantic[scope.value][tier.value]
            for scope in ExecutionScope
            for tier in (ProviderTier.SENIOR, ProviderTier.FRONTIER)
        ),
    )


QUALITY_AND_WORK_METRICS = (
    "tool_calls",
    "deterministic_suboperations",
    "failed_tool_calls",
    "prevented_tool_calls",
    "redundant_tool_calls",
    "retrieval_calls",
    "redundant_retrieval_calls",
    "retrieval_recall",
    "irrelevant_hit_rate",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "newly_evaluated_tokens",
    "prefill_tokens",
    "context_occupancy_high_water",
    "context_reconstructions",
    "context_compactions",
)

LATENCY_METRICS = (
    "end_to_end_task_wall_ms",
    "model_wall_ms",
    "ttft_first_protocol_event_ms",
    "ttft_first_user_visible_token_ms",
    "prefill_ms",
    "decode_ms",
    "decode_tokens_per_second",
    "tool_ms",
    "retrieval_ms",
    "context_reconstruction_ms",
    "controller_ms",
    "finalization_ms",
)

MEMORY_AND_RUNTIME_METRICS = (
    "peak_process_or_unified_memory_bytes",
    "host_available_memory_bytes",
    "memory_pressure_class",
    "swap_start_bytes",
    "swap_end_bytes",
    "swap_delta_bytes",
    "compression_start_bytes",
    "compression_end_bytes",
    "compression_delta_bytes",
    "resident_model_bytes",
    "kv_bytes",
    "speculative_companion_bytes",
    "small_model_bytes",
    "actual_peak_occupied_context_tokens",
)

RECOVERY_METRICS = (
    "crash_boundary",
    "restart_count",
    "recovery_attempts",
    "recovery_result",
    "ambiguous_outcome",
    "duplicate_side_effects",
    "approval_identity_preserved",
    "capability_identity_preserved",
    "call_identity_preserved",
    "checkpoint_identity_preserved",
    "payload_identity_preserved",
)

_COUNT_METRICS = {
    "tool_calls",
    "deterministic_suboperations",
    "failed_tool_calls",
    "prevented_tool_calls",
    "redundant_tool_calls",
    "retrieval_calls",
    "redundant_retrieval_calls",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "newly_evaluated_tokens",
    "prefill_tokens",
    "context_occupancy_high_water",
    "context_reconstructions",
    "context_compactions",
    "restart_count",
    "recovery_attempts",
    "duplicate_side_effects",
}
_RATIO_METRICS = {"retrieval_recall", "irrelevant_hit_rate"}
_BOOLEAN_METRICS = {
    "ambiguous_outcome",
    "approval_identity_preserved",
    "capability_identity_preserved",
    "call_identity_preserved",
    "checkpoint_identity_preserved",
    "payload_identity_preserved",
}
_STRING_METRICS = {"memory_pressure_class", "crash_boundary", "recovery_result"}


def unavailable_measurement() -> MetricMeasurement:
    return MetricMeasurement(value=None, provenance=MetricProvenance.UNAVAILABLE)


def unavailable_metric_group(names: Sequence[str]) -> dict[str, MetricMeasurement]:
    return {name: unavailable_measurement() for name in names}


@dataclass(frozen=True)
class BenchmarkMetrics(WireRecord):
    quality_and_work: Mapping[str, MetricMeasurement]
    latency: Mapping[str, MetricMeasurement]
    memory_and_runtime: Mapping[str, MetricMeasurement]
    recovery: Mapping[str, MetricMeasurement]

    def __post_init__(self) -> None:
        groups = (
            ("quality_and_work", self.quality_and_work, QUALITY_AND_WORK_METRICS),
            ("latency", self.latency, LATENCY_METRICS),
            ("memory_and_runtime", self.memory_and_runtime, MEMORY_AND_RUNTIME_METRICS),
            ("recovery", self.recovery, RECOVERY_METRICS),
        )
        for group_name, metrics, required in groups:
            if not isinstance(metrics, Mapping):
                raise TypeError(f"{group_name} must be a metric mapping")
            if len(metrics) > MAX_METRICS_PER_GROUP:
                raise ValueError(f"{group_name} exceeds {MAX_METRICS_PER_GROUP} metrics")
            missing = sorted(set(required) - set(metrics))
            if missing:
                raise ValueError(f"{group_name} omits required metrics: {', '.join(missing)}")
            for name, measurement in metrics.items():
                _text(name, f"{group_name} metric name", maximum=200)
                if not isinstance(measurement, MetricMeasurement):
                    raise TypeError(f"{group_name}.{name} must be a MetricMeasurement")
                value = measurement.value
                if value is None:
                    continue
                if name in _COUNT_METRICS and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    raise TypeError(f"{name} must be an integer count")
                if name in _RATIO_METRICS and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value > 1
                ):
                    raise ValueError(f"{name} must be a numeric ratio in [0, 1]")
                if name in _BOOLEAN_METRICS and not isinstance(value, bool):
                    raise TypeError(f"{name} must be boolean")
                if name in _STRING_METRICS and not isinstance(value, str):
                    raise TypeError(f"{name} must be a string")
        self._validate_count_relationships()

    def _numeric(self, group: Mapping[str, MetricMeasurement], name: str) -> int | float | None:
        value = group[name].value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    def _validate_count_relationships(self) -> None:
        work = self.quality_and_work
        for subset, total in (
            ("failed_tool_calls", "tool_calls"),
            ("redundant_tool_calls", "tool_calls"),
            ("redundant_retrieval_calls", "retrieval_calls"),
        ):
            subset_value = self._numeric(work, subset)
            total_value = self._numeric(work, total)
            if subset_value is not None and total_value is not None and subset_value > total_value:
                raise ValueError(f"{subset} cannot exceed {total}")
        input_tokens = self._numeric(work, "input_tokens")
        cached_tokens = self._numeric(work, "cached_tokens")
        new_tokens = self._numeric(work, "newly_evaluated_tokens")
        if (
            input_tokens is not None
            and cached_tokens is not None
            and new_tokens is not None
            and cached_tokens + new_tokens > input_tokens
        ):
            raise ValueError("cached_tokens + newly_evaluated_tokens cannot exceed input_tokens")


def unavailable_metrics() -> BenchmarkMetrics:
    """Create a complete metric vector whose values are explicitly unavailable."""

    return BenchmarkMetrics(
        quality_and_work=unavailable_metric_group(QUALITY_AND_WORK_METRICS),
        latency=unavailable_metric_group(LATENCY_METRICS),
        memory_and_runtime=unavailable_metric_group(MEMORY_AND_RUNTIME_METRICS),
        recovery=unavailable_metric_group(RECOVERY_METRICS),
    )


@dataclass(frozen=True)
class OptimizationBenchmarkTrial(WireRecord):
    identity: BenchmarkIdentity
    objective: ObjectiveIdentity
    correctness: CorrectnessResult
    invocation_events: tuple[ModelInvocationEvent, ...]
    invocation_counts: InvocationCounts
    evidence: tuple[EvidenceReference, ...]
    metrics: BenchmarkMetrics
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BenchmarkIdentity):
            raise TypeError("identity must be a BenchmarkIdentity")
        if not isinstance(self.objective, ObjectiveIdentity):
            raise TypeError("objective must be an ObjectiveIdentity")
        if not isinstance(self.correctness, CorrectnessResult):
            raise TypeError("correctness must be a CorrectnessResult")
        if not isinstance(self.metrics, BenchmarkMetrics):
            raise TypeError("metrics must be BenchmarkMetrics")
        if len(self.evidence) > MAX_EVIDENCE_REFS:
            raise ValueError("too many evidence records")
        evidence_by_id: dict[str, EvidenceReference] = {}
        for item in self.evidence:
            if not isinstance(item, EvidenceReference):
                raise TypeError("evidence must contain EvidenceReference records")
            if item.ref_id in evidence_by_id:
                raise ValueError(f"duplicate evidence ref_id: {item.ref_id}")
            evidence_by_id[item.ref_id] = item

        derived = derive_invocation_counts(self.invocation_events)
        if self.invocation_counts.to_dict() != derived.to_dict():
            raise ValueError("reported invocation counts do not match explicit invocation events")

        for ref_id in self.correctness.evidence_refs:
            if ref_id not in evidence_by_id:
                raise ValueError(f"correctness evidence reference does not resolve: {ref_id}")

        if self.correctness.outcome is CorrectnessOutcome.PASS:
            for ref_id in self.correctness.evidence_refs:
                item = evidence_by_id[ref_id]
                if not item.backend_owned:
                    raise ValueError("correctness=pass requires backend-owned evidence")
                if item.objective_key != self.objective.key:
                    raise ValueError("correctness evidence is not bound to the exact objective")
                if item.acceptance_criteria_sha256 != self.identity.acceptance_criteria_sha256:
                    raise ValueError("correctness evidence is not bound to the acceptance criteria")
            ambiguous = self.metrics.recovery["ambiguous_outcome"].value
            if ambiguous is True:
                raise ValueError("correctness=pass is incompatible with an ambiguous outcome")

        # The full envelope receives the same bounded JSON-safety check used by
        # canonicalization, so arbitrary extension metrics cannot smuggle bytes,
        # NaN, huge collections, or hidden object types into an artifact.
        _json_safe(asdict(self))

    @classmethod
    def from_events(
        cls,
        *,
        identity: BenchmarkIdentity,
        objective: ObjectiveIdentity,
        correctness: CorrectnessResult,
        invocation_events: Sequence[ModelInvocationEvent],
        evidence: Sequence[EvidenceReference],
        metrics: BenchmarkMetrics,
    ) -> "OptimizationBenchmarkTrial":
        events = tuple(invocation_events)
        return cls(
            identity=identity,
            objective=objective,
            correctness=correctness,
            invocation_events=events,
            invocation_counts=derive_invocation_counts(events),
            evidence=tuple(evidence),
            metrics=metrics,
        )

    @property
    def digest(self) -> str:
        return artifact_identity(self.to_dict())
