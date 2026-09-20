# SPDX-License-Identifier: AGPL-3.0-only
"""Typed model self-audit over observable artifacts only. No hidden reasoning capture."""

from __future__ import annotations

from typing import Any

from .schemas import AdaptationKind, CacheIntegrityReport, DecisionChoice, DecisionKind, EvidenceClaim, SelfAuditReport
from .trajectory import Trajectory, trajectory_record


def _strings(value: Any, limit: int = 32, width: int = 1_000) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:width] for item in value[:limit] if str(item).strip()]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def parse_self_audit(raw: dict[str, Any] | None, *, model_id: str = "") -> SelfAuditReport | None:
    if not isinstance(raw, dict):
        return None
    rec_raw = str(raw.get("recommendation") or "IGNORE").strip().upper().replace("HERMES_SKILL", "SKILL")
    try:
        recommendation = AdaptationKind(rec_raw)
    except ValueError:
        return None
    achieved = _optional_bool(raw.get("achieved"))
    try:
        confidence = max(0.0, min(1.0, float(raw.get("self_assessment_confidence", raw.get("confidence", 0.5)))))
    except (TypeError, ValueError):
        confidence = 0.5
    return SelfAuditReport(
        objective=str(raw.get("objective") or "")[:4_000],
        achieved=achieved,
        contributing_actions=_strings(raw.get("contributing_actions")),
        unnecessary_actions=_strings(raw.get("unnecessary_actions")),
        failures=_strings(raw.get("failures")),
        retries=_strings(raw.get("retries")),
        rediscovered_information=_strings(raw.get("rediscovered_information")),
        excess_retrieval=_strings(raw.get("excess_retrieval")),
        avoidable_cache_disruption=_strings(raw.get("avoidable_cache_disruption")),
        tool_selection_correct=_optional_bool(raw.get("tool_selection_correct")),
        expensive_resource_misuse=_strings(raw.get("expensive_resource_misuse")),
        overclaimed_claims=_strings(raw.get("overclaimed_claims")),
        stopped_too_early=bool(raw.get("stopped_too_early", False)),
        continued_too_long=bool(raw.get("continued_too_long", False)),
        better_trajectory=_strings(raw.get("better_trajectory")),
        reusable_lessons=_strings(raw.get("reusable_lessons")),
        likely_behavioral_pattern=bool(raw.get("likely_behavioral_pattern", False)),
        recommendation=recommendation,
        recommendation_reason=str(raw.get("recommendation_reason") or raw.get("reason") or "")[:2_000],
        self_assessment_confidence=confidence,
        model_id=model_id or str(raw.get("model_id") or "")[:500],
        source="model",
    )


def observable_audit_payload(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    evidence: list[EvidenceClaim],
) -> dict[str, Any]:
    """Artifacts the model may audit. Intentionally excludes ``Trajectory.reasoning``."""
    record = trajectory_record(traj)
    edits = [
        step
        for step in record.tool_steps
        if any(token in str(step.get("name") or "").lower() for token in ("edit", "write", "patch", "replace"))
    ]
    tests = [
        step
        for step in record.tool_steps
        if isinstance(step.get("verification"), dict)
        and step["verification"].get("kind") == "test"
    ]
    benchmarks = [
        step
        for step in record.tool_steps
        if isinstance(step.get("verification"), dict)
        and step["verification"].get("kind") == "benchmark"
    ]
    return {
        "schema_version": "helix.audit-input.v1",
        "trajectory": record.to_dict(),
        "objective": record.objective[:4_000],
        "presented_context": record.presented_context[:8_000],
        "tool_steps": record.tool_steps[-100:],
        "control_events": record.control_events[-100:],
        "edits": edits[-32:],
        "tests": tests[-32:],
        "benchmarks": benchmarks[-32:],
        "final_result": record.final_result[:8_000],
        "acceptance_criteria": record.acceptance_criteria,
        "cache_integrity": cache.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "objective_outcome_evidence": [
            item.to_dict() for item in evidence if item.claim_id == "task-outcome"
        ],
        "verified": bool(traj.verified),
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "latency_ms": traj.latency_ms,
    }


def prepare_observable_self_audit(
    traj: Trajectory,
    *,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare the backend-resolved audit bundle and a cheap pre-audit gate.

    Any failure in this optional path must preserve the historical behavior of
    running the deep audit, so the fail-open choice is TAKE.
    """
    from .cache_integrity import build_cache_integrity_report
    from .decision_controller import safe_advisory_decision
    from .evidence import build_evidence_claims, discover_important_claims
    from .ledger import append_record

    extras = traj.extras if isinstance(traj.extras, dict) else {}
    telemetry = extras.get("telemetry") if isinstance(extras.get("telemetry"), dict) else {}
    control_events = (
        extras.get("tool_control_events")
        if isinstance(extras.get("tool_control_events"), list)
        else []
    )
    trajectory_id = str(extras.get("trajectory_id") or "")
    cache = build_cache_integrity_report(telemetry, traj.steps)
    discovered_claims = discover_important_claims(traj, claims)
    evidence = build_evidence_claims(traj, cache, discovered_claims)
    missing = sum(len(item.missing_evidence) for item in evidence)
    contradictions = sum(bool(item.contradicting_evidence) for item in evidence)
    high_impact = any(
        token in item.claim.lower()
        for item in evidence
        for token in ("speed", "throughput", "5x", "security", "correctness")
    )
    criteria = extras.get("acceptance_criteria") if isinstance(extras.get("acceptance_criteria"), list) else []
    meaningful = bool(traj.steps or control_events or criteria or high_impact)
    features = {
        "missing_evidence_count": missing,
        "contradiction_count": contradictions,
        "has_evidence": any(item.supporting_evidence for item in evidence),
        "meaningful_task": meaningful,
        "high_impact_claim": high_impact,
        "tool_calls": len(traj.steps),
        "tool_control_suppressions": len(control_events),
        "failed_tool": any(step.error for step in traj.steps),
        "acceptance_criteria_count": len(criteria),
    }
    decision = safe_advisory_decision(
        DecisionKind.DEEP_SELF_AUDIT,
        features,
        decision_id=f"{trajectory_id}:{DecisionKind.DEEP_SELF_AUDIT.value}:pre",
        trajectory_id=trajectory_id,
        fallback_choice=DecisionChoice.TAKE,
    )
    # Deterministic reliability conditions can force a deep audit; the advisory
    # controller can save cost only in the low-risk remainder.
    forced_reasons = []
    if any(step.error for step in traj.steps):
        forced_reasons.append("failed_tool")
    if criteria:
        forced_reasons.append("acceptance_criteria")
    if high_impact:
        forced_reasons.append("high_impact_claim")
    if any(step.verification is not None for step in traj.steps):
        forced_reasons.append("typed_verification")
    perform_deep = bool(forced_reasons or decision.choice == DecisionChoice.TAKE)
    append_record("decisions", decision.to_dict())
    payload = observable_audit_payload(traj, cache, evidence)
    payload["pre_audit_decision"] = decision.to_dict()
    payload["deep_audit_forced_reasons"] = forced_reasons
    return {
        "schema_version": "helix.audit-preparation.v1",
        "available": True,
        "perform_deep_audit": perform_deep,
        "decision": decision.to_dict(),
        "artifacts": payload,
    }


def fallback_self_audit(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    evidence: list[EvidenceClaim],
    *,
    model_id: str = "",
) -> SelfAuditReport:
    extras = traj.extras if isinstance(traj.extras, dict) else {}
    control_events = (
        extras.get("tool_control_events")
        if isinstance(extras.get("tool_control_events"), list)
        else []
    )
    redundant = [f"{step.name}: repeated/redundant call" for step in traj.steps if step.useful_hint == "redundant"]
    prevented = [
        f"{str(item.get('tool_name') or 'tool')}: prevented {str(item.get('action') or 'controller no-op')}"
        for item in control_events
        if isinstance(item, dict)
        and item.get("action") in {"duplicate", "equivalent_duplicate", "repeated_failure"}
    ]
    unnecessary = redundant + prevented
    failures = [f"{step.name}: {step.error}" for step in traj.steps if step.error]
    unsupported = [item.claim for item in evidence if item.status.value in {"UNVERIFIED", "CONTRADICTED"}]
    return SelfAuditReport(
        objective=traj.prompt_state[:4_000],
        achieved=(
            True
            if isinstance(traj.extras, dict) and traj.extras.get("objective_verified") is True
            else None
        ),
        contributing_actions=[step.name for step in traj.steps if step.useful_hint in {"useful", "evidence"}][:32],
        unnecessary_actions=unnecessary[:32],
        failures=failures[:32],
        excess_retrieval=[item for item in unnecessary if "read" in item.lower() or "search" in item.lower()][:32],
        avoidable_cache_disruption=[item.action for item in cache.disruptions if not item.necessary][:32],
        tool_selection_correct=False if failures else None,
        overclaimed_claims=unsupported[:32],
        continued_too_long=bool(unnecessary),
        better_trajectory=["retain only credited non-redundant actions"] if unnecessary else [],
        recommendation=AdaptationKind.RUNTIME_POLICY if unnecessary else AdaptationKind.IGNORE,
        recommendation_reason="deterministic fallback derived from observable trajectory; model audit unavailable",
        self_assessment_confidence=0.0,
        model_id=model_id,
        source="deterministic_fallback",
    )
