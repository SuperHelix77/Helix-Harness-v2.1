# SPDX-License-Identifier: AGPL-3.0-only
"""Independent trajectory quality dimensions. Never collapsed into one opaque scalar."""

from __future__ import annotations

from .schemas import CacheIntegrityReport, EvidenceClaim, EvidenceStatus, QualityVector, SelfAuditReport
from .trajectory import Trajectory


def build_quality_vector(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    evidence: list[EvidenceClaim],
    audit: SelfAuditReport,
) -> QualityVector:
    extras = traj.extras if isinstance(traj.extras, dict) else {}
    control_events = (
        extras.get("tool_control_events")
        if isinstance(extras.get("tool_control_events"), list)
        else []
    )
    prevented_exact = sum(
        isinstance(item, dict) and item.get("action") == "duplicate" for item in control_events
    )
    prevented_equivalent = sum(
        isinstance(item, dict) and item.get("action") == "equivalent_duplicate"
        for item in control_events
    )
    prevented_repeated_failure = sum(
        isinstance(item, dict) and item.get("action") == "repeated_failure"
        for item in control_events
    )
    objective_verified = extras.get("objective_verified") is True
    task_claim = next((item for item in evidence if item.claim_id == "task-outcome"), None)
    task_contradicted = bool(task_claim and task_claim.status == EvidenceStatus.CONTRADICTED)
    # 0.5 means completed-but-unverified, not half-correct. Only objective evidence
    # earns 1.0; a model's own achieved=true can never promote task quality.
    q = 1.0 if objective_verified else (0.0 if task_contradicted else (0.5 if traj.final_result else 0.0))
    applicable = [item for item in evidence if item.status != EvidenceStatus.NOT_APPLICABLE]
    supported = sum(1 for item in applicable if item.status == EvidenceStatus.SUPPORTED)
    partial = sum(1 for item in applicable if item.status == EvidenceStatus.PARTIALLY_SUPPORTED)
    e = ((supported + 0.5 * partial) / len(applicable)) if applicable else 1.0
    hinted_redundant = sum(1 for step in traj.steps if step.useful_hint == "redundant")
    observed_redundant = sum(
        1
        for item in cache.disruptions
        if not item.necessary and item.action.startswith("repeat_tool:")
    )
    # Backend cache capture can objectively detect alias-equivalent retrievals
    # even when the original tool step arrived with the generic "useful" hint.
    redundant = max(hinted_redundant, observed_redundant)
    failed = sum(1 for step in traj.steps if step.error)
    tool_efficiency = 1.0 if not traj.steps else max(0.0, 1.0 - ((redundant + failed) / len(traj.steps)))
    cache_efficiency = cache.cache_reuse_ratio if cache.prompt_tokens else 1.0
    c = max(0.0, min(1.0, 0.7 * tool_efficiency + 0.3 * cache_efficiency))
    if not objective_verified and not task_contradicted:
        # Calibration cannot be scored without an objective outcome label.
        s = 0.5
    elif audit.achieved is None:
        s = 0.5
    else:
        outcome = 1.0 if objective_verified else 0.0
        claimed = audit.self_assessment_confidence if audit.achieved else 1.0 - audit.self_assessment_confidence
        s = max(0.0, 1.0 - abs(claimed - outcome))
    return QualityVector(
        task_quality=q,
        evidentiary_completeness=e,
        computational_efficiency=c,
        self_assessment_calibration=s,
        raw_metrics={
            "tool_calls": len(traj.steps),
            "redundant_tool_calls": redundant,
            "failed_tool_calls": failed,
            # Prevented calls are behavioral observations, not computational cost:
            # they intentionally do not reduce C because the runtime avoided them.
            "prevented_tool_calls": len(control_events),
            "prevented_exact_duplicates": prevented_exact,
            "prevented_equivalent_duplicates": prevented_equivalent,
            "prevented_repeated_failures": prevented_repeated_failure,
            "prompt_tokens": traj.prompt_tokens or cache.prompt_tokens,
            "completion_tokens": traj.completion_tokens,
            "cached_tokens": cache.cached_tokens,
            "newly_evaluated_tokens": cache.newly_evaluated_tokens,
            "cache_reuse_ratio": cache.cache_reuse_ratio,
            "prefill_ms": cache.prefill_ms,
            "decode_ms": cache.decode_ms,
            "ttft_ms": cache.ttft_ms,
            "context_compactions": cache.context_compactions,
            "accepted_drafts": cache.accepted_drafts,
            "rejected_drafts": cache.rejected_drafts,
            "latency_ms": traj.latency_ms,
        },
    )
