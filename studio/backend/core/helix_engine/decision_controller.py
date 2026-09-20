# SPDX-License-Identifier: AGPL-3.0-only
"""Jev-inspired typed advisory decisions with measurable calibration."""

from __future__ import annotations

from typing import Any, Iterable

from .schemas import DecisionChoice, DecisionKind, DecisionRecord

POLICY_VERSION = "helix-shadow-v1"
FALLBACK_POLICY_VERSION = "helix-fallback-v1"
_MIN_CALIBRATION_LABELS = 50


def parse_decision_kind(value: str | DecisionKind) -> DecisionKind:
    if isinstance(value, DecisionKind):
        return value
    try:
        return DecisionKind(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError(f"unknown Helix decision kind: {value!r}") from exc


def _probability(kind: DecisionKind, f: dict[str, Any]) -> float:
    missing = int(f.get("missing_evidence_count") or 0)
    contradictions = int(f.get("contradiction_count") or 0)
    recurrence = int(f.get("recurrence_count") or 0)
    redundant = int(f.get("redundant_tool_calls") or 0)
    failed = bool(f.get("failed_tool"))
    if kind in {DecisionKind.RETRIEVE_MEMORY, DecisionKind.SEARCH_CONVERSATION}:
        return 0.9 if f.get("missing_prior_context") else 0.15
    if kind in {DecisionKind.RETRIEVE_SKILL, DecisionKind.USE_EXISTING_SKILL}:
        return 0.88 if f.get("matching_skill") else 0.2
    if kind == DecisionKind.CREATE_TEMPORARY_SKILL:
        return 0.72 if f.get("reusable_procedure") and not f.get("matching_skill") else 0.08
    if kind == DecisionKind.COMPACT_CONTEXT:
        return 0.9 if f.get("context_pressure") else 0.12
    if kind == DecisionKind.PRESERVE_CONTEXT:
        return 0.85 if not f.get("context_pressure") and f.get("cache_reuse_ratio", 0) > 0.25 else 0.45
    if kind == DecisionKind.SEARCH_REPOSITORY:
        return 0.82 if f.get("missing_repository_evidence") else max(0.05, 0.25 - redundant * 0.05)
    if kind == DecisionKind.EVIDENCE_SUFFICIENT:
        return 0.92 if missing == 0 and contradictions == 0 and f.get("has_evidence") else 0.1
    if kind == DecisionKind.RUN_ANOTHER_TEST:
        return 0.9 if missing > 0 and f.get("test_can_resolve") else 0.18
    if kind == DecisionKind.SPAWN_VERIFICATION:
        return 0.82 if f.get("high_impact_claim") and missing > 0 else 0.12
    if kind == DecisionKind.RETRY_FAILED_TOOL:
        return 0.78 if failed and f.get("retryable", True) else 0.08
    if kind == DecisionKind.ESCALATE_MODEL:
        return 0.8 if f.get("capability_failure") and f.get("verification_failed") else 0.1
    if kind == DecisionKind.USE_LOCAL_MODEL:
        return 0.8 if f.get("local_sufficient") else 0.3
    if kind == DecisionKind.USE_ASTRA:
        return 0.75 if f.get("strong_model_needed") and not f.get("local_sufficient") else 0.1
    if kind == DecisionKind.USE_SPECULATIVE_DECODING:
        return 0.85 if f.get("accepted_drafts", 0) > 0 and f.get("sidecar_ok") else 0.1
    if kind == DecisionKind.DEEP_SELF_AUDIT:
        return 0.85 if f.get("meaningful_task") and (missing > 0 or f.get("high_impact_claim")) else 0.35
    if kind == DecisionKind.REUSABLE_PROCEDURE:
        return 0.78 if f.get("reusable_procedure") and recurrence >= 2 else 0.2
    if kind == DecisionKind.QLORA_CANDIDATE:
        return 0.78 if recurrence >= 3 and f.get("behavior_gap") and f.get("evidence_supported") else 0.03
    if kind == DecisionKind.IGNORE_NOISE:
        return 0.85 if f.get("one_off") or f.get("noise") else 0.25
    return 0.5


def shadow_decision(
    kind: str | DecisionKind,
    features: dict[str, Any] | None = None,
    *,
    decision_id: str = "",
    trajectory_id: str = "",
) -> DecisionRecord:
    parsed = parse_decision_kind(kind)
    evidence = dict(features or {})
    probability = max(0.0, min(1.0, _probability(parsed, evidence)))
    # Confidence is distance from the decision boundary, not a claim of empirical calibration.
    confidence = min(1.0, abs(probability - 0.5) * 2.0)
    return DecisionRecord(
        decision_id=decision_id,
        trajectory_id=trajectory_id,
        decision=parsed,
        probability=probability,
        confidence=confidence,
        choice=DecisionChoice.TAKE if probability >= 0.5 else DecisionChoice.SKIP,
        advisory_only=True,
        policy_version=POLICY_VERSION,
        evidence_features=evidence,
    )


def safe_advisory_decision(
    kind: str | DecisionKind,
    features: dict[str, Any] | None = None,
    *,
    decision_id: str = "",
    trajectory_id: str = "",
    fallback_choice: DecisionChoice = DecisionChoice.TAKE,
) -> DecisionRecord:
    """Fail-open advisory wrapper for production allocation points.

    The fallback preserves the caller's established behavior. A controller bug is
    recorded as zero-confidence metadata; it never becomes authority to suppress a
    previously executed safety/reliability path.
    """
    try:
        return shadow_decision(
            kind,
            features,
            decision_id=decision_id,
            trajectory_id=trajectory_id,
        )
    except BaseException as exc:  # optional controller must also survive cancellation-like faults
        try:
            parsed = parse_decision_kind(kind)
        except Exception:
            parsed = DecisionKind.IGNORE_NOISE
        evidence = dict(features or {})
        evidence["controller_fallback"] = f"{type(exc).__name__}: {exc}"[:500]
        return DecisionRecord(
            decision_id=decision_id,
            trajectory_id=trajectory_id,
            decision=parsed,
            probability=1.0 if fallback_choice == DecisionChoice.TAKE else 0.0,
            confidence=0.0,
            choice=fallback_choice,
            advisory_only=True,
            policy_version=FALLBACK_POLICY_VERSION,
            evidence_features=evidence,
        )


def brier_score(records: Iterable[DecisionRecord]) -> float | None:
    pairs = [record for record in records if record.eventual_outcome is not None]
    if not pairs:
        return None
    return sum((record.probability - (1.0 if record.eventual_outcome else 0.0)) ** 2 for record in pairs) / len(pairs)


def record_decision_outcome(
    decision_id: str,
    *,
    eventual_outcome: bool,
    retrospective_usefulness: bool | None = None,
) -> bool:
    """Append an outcome observation for an existing decision receipt."""
    from .ledger import append_record, recent_records

    if not decision_id.strip():
        return False
    original = None
    for row in reversed(recent_records("decisions", limit=2_000)):
        if row.get("decision_id") == decision_id and row.get("eventual_outcome") is None:
            original = row
            break
    if original is None:
        return False
    merged = dict(original)
    merged["eventual_outcome"] = bool(eventual_outcome)
    merged["retrospective_usefulness"] = (
        bool(retrospective_usefulness) if retrospective_usefulness is not None else None
    )
    merged["record_type"] = "outcome"
    return append_record("decisions", merged)


def calibration_summary() -> dict[str, Any]:
    """Empirical Brier score over decisions that actually have outcome labels."""
    from .ledger import recent_records

    latest: dict[str, dict[str, Any]] = {}
    for row in recent_records("decisions", limit=2_000):
        decision_id = str(row.get("decision_id") or "")
        if decision_id:
            latest[decision_id] = row
    labelled = [row for row in latest.values() if isinstance(row.get("eventual_outcome"), bool)]
    if not labelled:
        return {
            "schema_version": "helix.decision-calibration.v1",
            "labelled_decisions": 0,
            "brier_score": None,
            "calibrated": False,
            "calibration_measured": False,
            "measurement_mature": False,
            "minimum_labels_for_maturity": _MIN_CALIBRATION_LABELS,
            "bins": [],
        }
    squared = []
    for row in labelled:
        try:
            probability = max(0.0, min(1.0, float(row.get("probability", 0.5))))
        except (TypeError, ValueError):
            probability = 0.5
        outcome = 1.0 if row.get("eventual_outcome") else 0.0
        squared.append((probability - outcome) ** 2)
    bins = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = []
        for row in labelled:
            try:
                probability = max(0.0, min(1.0, float(row.get("probability", 0.5))))
            except (TypeError, ValueError):
                probability = 0.5
            if lower <= probability < upper or (upper >= 1.0 and probability == 1.0):
                members.append((probability, 1.0 if row.get("eventual_outcome") else 0.0))
        if members:
            bins.append(
                {
                    "lower": round(lower, 2),
                    "upper": round(min(1.0, upper), 2),
                    "count": len(members),
                    "mean_probability": sum(item[0] for item in members) / len(members),
                    "observed_frequency": sum(item[1] for item in members) / len(members),
                }
            )
    return {
        "schema_version": "helix.decision-calibration.v1",
        "labelled_decisions": len(squared),
        "brier_score": sum(squared) / len(squared),
        "calibration_measured": True,
        "measurement_mature": len(squared) >= _MIN_CALIBRATION_LABELS,
        "minimum_labels_for_maturity": _MIN_CALIBRATION_LABELS,
        "bins": bins,
        # Measurement is not equivalent to a calibration guarantee. Keep this
        # conservative until a dedicated held-out calibration study exists.
        "calibrated": False,
    }
