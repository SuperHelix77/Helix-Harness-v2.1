# SPDX-License-Identifier: AGPL-3.0-only
"""Hermes adjudicates self-report against objective evidence before adaptation."""

from __future__ import annotations

from .schemas import (
    AdaptationDecision,
    AdaptationKind,
    CacheIntegrityReport,
    CounterfactualTrajectoryCandidate,
    EvidenceClaim,
    EvidenceStatus,
    QualityVector,
    SelfAuditReport,
)
from .training_targets import validated_training_target_receipt
from .trajectory import Trajectory

_MIN_QLORA_RECURRENCE = 3


def adjudicate(
    traj: Trajectory,
    audit: SelfAuditReport,
    evidence: list[EvidenceClaim],
    cache: CacheIntegrityReport,
    quality: QualityVector,
    *,
    recurrence: int = 1,
    counterfactual: CounterfactualTrajectoryCandidate | None = None,
) -> AdaptationDecision:
    unsupported = [
        item for item in evidence if item.status in {EvidenceStatus.CONTRADICTED, EvidenceStatus.UNVERIFIED}
    ]
    disagreement = bool(audit.achieved is True and any(item.claim_id == "task-outcome" for item in unsupported))
    if audit.overclaimed_claims and unsupported:
        disagreement = True

    requested = audit.recommendation
    action = AdaptationKind.IGNORE
    reason = "no corroborated reusable adaptation signal"
    qlora_eligible = False

    avoidable_cache = [item for item in cache.disruptions if not item.necessary]
    if requested == AdaptationKind.CAPABILITY_GAP:
        action = AdaptationKind.CAPABILITY_GAP
        reason = "model reported a capability gap; retained as a hypothesis, not a weight update"
    elif requested == AdaptationKind.MEMORY:
        action = AdaptationKind.MEMORY
        reason = "audit identified reusable factual experience"
    elif requested == AdaptationKind.SKILL:
        action = AdaptationKind.SKILL
        reason = "audit identified a reusable procedure"
    elif requested == AdaptationKind.RUNTIME_POLICY:
        action = AdaptationKind.RUNTIME_POLICY
        reason = "mechanical behavior is better addressed deterministically"
    elif requested == AdaptationKind.QLORA_CANDIDATE:
        evidence_good = bool(evidence) and not unsupported and quality.evidentiary_completeness >= 0.75
        if recurrence >= _MIN_QLORA_RECURRENCE and audit.likely_behavioral_pattern and evidence_good:
            action = AdaptationKind.QLORA_CANDIDATE
            extras = traj.extras if isinstance(traj.extras, dict) else {}
            trajectory_id = str(extras.get("trajectory_id") or "")
            target_receipt = validated_training_target_receipt(
                extras.get("verified_training_target_receipt"),
                trajectory_id=trajectory_id,
            )
            qlora_eligible = target_receipt is not None
            reason = (
                "repeated corroborated behavior is a QLoRA candidate with a backend-verified corrected-target receipt"
                if qlora_eligible
                else "repeated corroborated behavior is a QLoRA candidate, but training is blocked until a corrected target is independently verified"
            )
        else:
            action = AdaptationKind.SKILL if audit.reusable_lessons else AdaptationKind.IGNORE
            reason = "QLoRA recommendation rejected: requires repeated corroborated evidence"
    elif avoidable_cache:
        action = AdaptationKind.RUNTIME_POLICY
        reason = "avoidable cache/context disruption is mechanical and should be fixed before training"
    elif audit.reusable_lessons:
        action = AdaptationKind.SKILL
        reason = "reusable procedure is better represented as a skill before weights"

    # A self-audit recommendation is advisory. When Hermes selects a different
    # adaptation because objective cache/evidence/safety gates say otherwise,
    # preserve that mismatch explicitly as self-assessment disagreement.
    if action != requested:
        disagreement = True

    # Objective contradiction always wins over a model's claim of success. It does not erase an
    # independently useful runtime/skill adaptation; it only records the disagreement.
    rejected = [item.claim_id for item in unsupported]
    decision = AdaptationDecision(
        action=action,
        reason=reason,
        qlora_eligible=qlora_eligible,
        recurrence_count=max(1, recurrence),
        source_trajectory_ids=[str(traj.extras.get("trajectory_id") or "")]
        if isinstance(traj.extras, dict) and traj.extras.get("trajectory_id")
        else [],
        evidence_ids=[item.claim_id for item in evidence],
        rejected_claim_ids=rejected,
        self_assessment_disagreement=disagreement,
        advisory_only=True,
    )
    if counterfactual is not None:
        decision.counterfactual_candidate_id = counterfactual.candidate_id
        decision.counterfactual_equivalence_status = counterfactual.equivalence_status
        # Hermes may admit an efficiency pair only after an independent
        # equivalence verifier established the same outcome and the source run
        # retained high task/evidence quality. Shortness/confidence alone cannot.
        decision.efficiency_training_eligible = bool(
            counterfactual.training_pair_eligible
            and counterfactual.equivalence_verified
            and quality.task_quality >= 0.75
            and quality.evidentiary_completeness >= 0.75
        )
    return decision
