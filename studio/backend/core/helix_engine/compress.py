# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib

from .credit import assign_credit
from .schemas import CounterfactualTrajectoryCandidate, EvidenceStatus, QualityVector
from .trajectory import Trajectory


def _tokens(text: str, cap: int) -> int:
    return min(cap, max(1, len(text) // 4))


def compress_counterfactual(traj: Trajectory) -> dict:
    kept = assign_credit(traj)
    steps = [traj.steps[i] for i in kept]
    state = traj.prompt_state.strip()
    evidence = "\n".join(step.result for step in steps if step.useful_hint == "evidence")
    decision = traj.final_result.strip()
    return {
        "tool_calls": len(steps),
        "state_summary_tokens": _tokens(state, 1_800),
        "necessary_evidence_tokens": _tokens(evidence or traj.retrieved_context, 3_200),
        "final_decision": decision[:700],
        "kept_indices": kept,
    }


def build_counterfactual_candidate(
    traj: Trajectory,
    *,
    quality: QualityVector | None = None,
    evidence_ids: list[str] | None = None,
) -> CounterfactualTrajectoryCandidate:
    """Build a shorter-trajectory hypothesis without treating it as equivalent proof."""
    compressed = compress_counterfactual(traj)
    kept = [int(index) for index in compressed.get("kept_indices", [])]
    proposed = [
        {
            "index": index,
            "name": traj.steps[index].name,
            "arguments": traj.steps[index].arguments[:1_000],
            "expected_evidence": traj.steps[index].result[:1_000],
        }
        for index in kept
        if 0 <= index < len(traj.steps)
    ]
    extras = traj.extras if isinstance(traj.extras, dict) else {}
    trajectory_id = str(extras.get("trajectory_id") or "")
    # No client/model-authored telemetry field may establish equivalence. A future
    # replay/verification service can issue a backend-owned typed receipt and this
    # function can consume that receipt. Until then every compression is a
    # hypothesis only, regardless of any similarly named client field.
    equivalence_verified = False
    status = EvidenceStatus.UNVERIFIED
    actual_tokens = max(0, int(traj.prompt_tokens or 0) + int(traj.completion_tokens or 0))
    proposed_tokens = int(compressed.get("state_summary_tokens", 0)) + int(
        compressed.get("necessary_evidence_tokens", 0)
    )
    digest_source = f"{trajectory_id}|{kept}|{traj.final_result[:500]}"
    candidate_id = "cf:" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:20]
    removed = max(0, len(traj.steps) - len(kept))
    confidence = min(0.95, 0.35 + (removed / max(1, len(traj.steps))) * 0.4)
    source_quality = quality.to_dict() if quality is not None else {}
    eligible = False
    return CounterfactualTrajectoryCandidate(
        candidate_id=candidate_id,
        source_trajectory_id=trajectory_id,
        proposed_actions=proposed,
        kept_indices=kept,
        same_result_target=traj.final_result[:4_000],
        actual_tool_calls=len(traj.steps),
        proposed_tool_calls=len(kept),
        estimated_tool_calls_saved=max(0, len(traj.steps) - len(kept)),
        estimated_tokens_saved=max(0, actual_tokens - proposed_tokens),
        confidence=confidence,
        equivalence_status=status,
        equivalence_verified=equivalence_verified,
        evidence_ids=[str(item)[:200] for item in (evidence_ids or [])[:64]],
        source_quality=source_quality,
        training_pair_eligible=eligible,
    )
