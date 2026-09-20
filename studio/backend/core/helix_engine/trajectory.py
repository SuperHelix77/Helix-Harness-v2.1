# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class VerificationKind(str, Enum):
    TEST = "test"
    BENCHMARK = "benchmark"
    VERIFIER = "verifier"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolVerificationReceipt:
    kind: VerificationKind
    status: VerificationStatus
    provenance: str = "backend_tool_capture"
    detail: str = ""
    # Backend-owned semantic binding. A passed verifier is not evidence for an
    # arbitrary model-authored claim merely because the model cites its tool id.
    # ``claim`` names the exact assertion the verifier established; ``subject``
    # binds the run to the artifact/object it verified (for example a corrected
    # training target digest). Model tool arguments never populate either field.
    claim: str = ""
    subject: str = ""


@dataclass
class ToolStep:
    name: str
    arguments: str
    result: str
    useful_hint: str = ""
    error: Optional[str] = None
    retry: int = 0
    verification: Optional[ToolVerificationReceipt] = None
    # Observable execution ordering. Zero preserves constructors and historical
    # records that predate live provenance sequencing.
    sequence: int = 0
    created_at_ms: int = 0


@dataclass
class Correction:
    state: str
    bad_action: str
    failure_evidence: str
    correct_action: str
    why: str


@dataclass
class Trajectory:
    prompt_state: str
    retrieved_context: str
    reasoning: str
    steps: list[ToolStep]
    user_corrections: list[Correction]
    final_result: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    verified: bool
    semantic_turns: int = 0
    holdout_passed: bool = False
    allow_qlora: bool = False
    frequent_behavior: bool = False
    one_off_fact: bool = False
    regression_passed: bool = False
    dataset_hash: str = ""
    adapter_version: str = ""
    extras: dict = field(default_factory=dict)


def trajectory_record(traj: Trajectory):
    """Return a versioned observable wire record; hidden reasoning is excluded."""
    from .schemas import TrajectoryRecord

    extras = traj.extras if isinstance(traj.extras, dict) else {}
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(traj.steps[-200:]):
        verification = None
        if step.verification is not None:
            verification = {
                "kind": step.verification.kind.value,
                "status": step.verification.status.value,
                "provenance": step.verification.provenance,
                "detail": step.verification.detail,
                "claim": step.verification.claim,
                "subject": step.verification.subject,
            }
        steps.append({
            "index": index,
            "name": step.name,
            "arguments": step.arguments,
            "result": step.result,
            "useful_hint": step.useful_hint,
            "error": step.error,
            "retry": step.retry,
            "sequence": step.sequence,
            "created_at_ms": step.created_at_ms,
            "verification": verification,
        })
    telemetry = extras.get("telemetry") if isinstance(extras.get("telemetry"), dict) else {}
    criteria = extras.get("acceptance_criteria") if isinstance(extras.get("acceptance_criteria"), list) else []
    control_events = (
        extras.get("tool_control_events")
        if isinstance(extras.get("tool_control_events"), list)
        else []
    )
    return TrajectoryRecord(
        trajectory_id=str(extras.get("trajectory_id") or ""),
        objective=traj.prompt_state[:8_000],
        presented_context=traj.retrieved_context[:16_000],
        tool_steps=steps,
        control_events=[dict(item) for item in control_events[:200] if isinstance(item, dict)],
        final_result=traj.final_result[:16_000],
        acceptance_criteria=[str(item)[:1_000] for item in criteria[:64]],
        telemetry=dict(telemetry),
        verified=bool(traj.verified),
        objective_verified=extras.get("objective_verified") is True,
        latency_ms=max(0.0, float(traj.latency_ms or 0.0)),
        prompt_tokens=max(0, int(traj.prompt_tokens or 0)),
        completion_tokens=max(0, int(traj.completion_tokens or 0)),
        model_id=str(extras.get("model_id") or "")[:500],
    )
