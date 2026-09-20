# SPDX-License-Identifier: AGPL-3.0-only
"""Versioned wire schemas for the Helix adaptive-intelligence control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "helix.adaptive.v1"
TRAJECTORY_SCHEMA_VERSION = "helix.trajectory.v1"
COUNTERFACTUAL_SCHEMA_VERSION = "helix.counterfactual.v1"
TOOL_CONTROL_SCHEMA_VERSION = "helix.tool-control.v1"


class CacheCause(str, Enum):
    NECESSARY = "NECESSARY"
    USER_CAUSED = "USER_CAUSED"
    TOOL_CAUSED = "TOOL_CAUSED"
    HARNESS_CAUSED = "HARNESS_CAUSED"
    MODEL_CAUSED = "MODEL_CAUSED"
    UNKNOWN = "UNKNOWN"


class EvidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AdaptationKind(str, Enum):
    IGNORE = "IGNORE"
    RUNTIME_POLICY = "RUNTIME_POLICY"
    MEMORY = "MEMORY"
    SKILL = "SKILL"
    QLORA_CANDIDATE = "QLORA_CANDIDATE"
    CAPABILITY_GAP = "CAPABILITY_GAP"


class DecisionKind(str, Enum):
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    SEARCH_CONVERSATION = "SEARCH_CONVERSATION"
    RETRIEVE_SKILL = "RETRIEVE_SKILL"
    CREATE_TEMPORARY_SKILL = "CREATE_TEMPORARY_SKILL"
    USE_EXISTING_SKILL = "USE_EXISTING_SKILL"
    COMPACT_CONTEXT = "COMPACT_CONTEXT"
    PRESERVE_CONTEXT = "PRESERVE_CONTEXT"
    SEARCH_REPOSITORY = "SEARCH_REPOSITORY"
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    RUN_ANOTHER_TEST = "RUN_ANOTHER_TEST"
    SPAWN_VERIFICATION = "SPAWN_VERIFICATION"
    RETRY_FAILED_TOOL = "RETRY_FAILED_TOOL"
    ESCALATE_MODEL = "ESCALATE_MODEL"
    USE_LOCAL_MODEL = "USE_LOCAL_MODEL"
    USE_ASTRA = "USE_ASTRA"
    USE_SPECULATIVE_DECODING = "USE_SPECULATIVE_DECODING"
    DEEP_SELF_AUDIT = "DEEP_SELF_AUDIT"
    REUSABLE_PROCEDURE = "REUSABLE_PROCEDURE"
    QLORA_CANDIDATE = "QLORA_CANDIDATE"
    IGNORE_NOISE = "IGNORE_NOISE"


class DecisionChoice(str, Enum):
    TAKE = "TAKE"
    SKIP = "SKIP"


def _wire(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


class WireRecord:
    def to_dict(self) -> dict[str, Any]:
        return _wire(asdict(self))


@dataclass
class CacheDisruption(WireRecord):
    cause: CacheCause
    action: str
    necessary: bool
    estimated_cost_tokens: int = 0
    evidence: list[str] = field(default_factory=list)
    provenance: str = ""


@dataclass
class CacheIntegrityReport(WireRecord):
    schema_version: str = SCHEMA_VERSION
    prompt_tokens: int = 0
    stable_prefix_tokens: int = 0
    newly_evaluated_tokens: int = 0
    cached_tokens: int = 0
    cache_reuse_ratio: float = 0.0
    kv_cache_resets: int = 0
    context_compactions: int = 0
    prompt_reconstructions: int = 0
    system_prompt_changes: int = 0
    tool_schema_changes: int = 0
    context_insertions: list[str] = field(default_factory=list)
    repeated_context_insertions: int = 0
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    context_reorders: int = 0
    speculative_requested: str = ""
    speculative_engaged: str = ""
    accepted_drafts: int = 0
    rejected_drafts: int = 0
    runtime_config: dict[str, Any] = field(default_factory=dict)
    disruptions: list[CacheDisruption] = field(default_factory=list)
    observable_fields: list[str] = field(default_factory=list)
    telemetry_provenance: dict[str, str] = field(default_factory=dict)
    unavailable_fields: list[str] = field(default_factory=list)


@dataclass
class ToolControlEvent(WireRecord):
    """A tool-loop action prevented before execution; never a fake ToolStep."""

    schema_version: str = TOOL_CONTROL_SCHEMA_VERSION
    action: str = ""
    tool_name: str = ""
    arguments: str = ""
    reason: str = ""
    equivalent_to: str = ""
    failed_attempts: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    provenance: str = "runtime_tool_loop"
    # Zero keeps old constructors/wire fixtures valid while captured events carry
    # exact live ordering metadata.
    sequence: int = 0
    created_at_ms: int = 0


@dataclass
class TrajectoryRecord(WireRecord):
    """Versioned observable trajectory envelope for persistence and audit I/O."""

    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    trajectory_id: str = ""
    objective: str = ""
    presented_context: str = ""
    tool_steps: list[dict[str, Any]] = field(default_factory=list)
    control_events: list[dict[str, Any]] = field(default_factory=list)
    final_result: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    # Legacy pipeline completion/critic bit. This is not objective proof.
    verified: bool = False
    # Backend-resolved objective outcome verification, kept separate so the wire
    # record cannot imply that a completed tool loop proved its own claims.
    objective_verified: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_id: str = ""


@dataclass
class CounterfactualTrajectoryCandidate(WireRecord):
    """A proposed shorter trajectory, never ground truth by construction."""

    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION
    candidate_id: str = ""
    source_trajectory_id: str = ""
    provenance: str = "deterministic_credit_compression"
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    kept_indices: list[int] = field(default_factory=list)
    same_result_target: str = ""
    actual_tool_calls: int = 0
    proposed_tool_calls: int = 0
    estimated_tool_calls_saved: int = 0
    estimated_tokens_saved: int = 0
    confidence: float = 0.0
    equivalence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    equivalence_verified: bool = False
    evidence_ids: list[str] = field(default_factory=list)
    source_quality: dict[str, Any] = field(default_factory=dict)
    training_pair_eligible: bool = False


@dataclass
class DecisionRecord(WireRecord):
    schema_version: str = SCHEMA_VERSION
    decision_id: str = ""
    trajectory_id: str = ""
    decision: DecisionKind = DecisionKind.IGNORE_NOISE
    probability: float = 0.5
    confidence: float = 0.0
    choice: DecisionChoice = DecisionChoice.SKIP
    advisory_only: bool = True
    policy_version: str = "helix-shadow-v1"
    evidence_features: dict[str, Any] = field(default_factory=dict)
    eventual_outcome: Optional[bool] = None
    retrospective_usefulness: Optional[bool] = None


@dataclass
class EvidenceClaim(WireRecord):
    schema_version: str = SCHEMA_VERSION
    claim_id: str = ""
    claim: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    reported_supporting_evidence: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED


@dataclass
class SelfAuditReport(WireRecord):
    schema_version: str = SCHEMA_VERSION
    objective: str = ""
    achieved: Optional[bool] = None
    contributing_actions: list[str] = field(default_factory=list)
    unnecessary_actions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    retries: list[str] = field(default_factory=list)
    rediscovered_information: list[str] = field(default_factory=list)
    excess_retrieval: list[str] = field(default_factory=list)
    avoidable_cache_disruption: list[str] = field(default_factory=list)
    tool_selection_correct: Optional[bool] = None
    expensive_resource_misuse: list[str] = field(default_factory=list)
    overclaimed_claims: list[str] = field(default_factory=list)
    stopped_too_early: bool = False
    continued_too_long: bool = False
    better_trajectory: list[str] = field(default_factory=list)
    reusable_lessons: list[str] = field(default_factory=list)
    likely_behavioral_pattern: bool = False
    recommendation: AdaptationKind = AdaptationKind.IGNORE
    recommendation_reason: str = ""
    self_assessment_confidence: float = 0.5
    model_id: str = ""
    source: str = "model"


@dataclass
class QualityVector(WireRecord):
    schema_version: str = SCHEMA_VERSION
    task_quality: float = 0.0
    evidentiary_completeness: float = 0.0
    computational_efficiency: float = 0.0
    self_assessment_calibration: float = 0.0
    raw_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationDecision(WireRecord):
    schema_version: str = SCHEMA_VERSION
    action: AdaptationKind = AdaptationKind.IGNORE
    reason: str = ""
    qlora_eligible: bool = False
    recurrence_count: int = 1
    pattern_fingerprint: str = ""
    source_trajectory_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    rejected_claim_ids: list[str] = field(default_factory=list)
    self_assessment_disagreement: bool = False
    advisory_only: bool = True
    counterfactual_candidate_id: str = ""
    counterfactual_equivalence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    efficiency_training_eligible: bool = False
