# SPDX-License-Identifier: AGPL-3.0-only
"""In-app Helix Engine API — not a separate webapp."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject

from core.helix_engine import (
    Correction,
    ToolStep,
    Trajectory,
    cluster_deficits,
    ingest_turn,
    monitor_semantic_turns,
    run_adaptation_pipeline,
)
from core.inference.computer_browse import feed_session_id, list_live_feed
from core.hub.simple_hub import download_local_model, list_local_gguf, load_local_model, select_local_model

router = APIRouter()


class ToolStepIn(BaseModel):
    name: str
    arguments: str = ""
    result: str = ""
    useful_hint: str = ""
    error: str | None = None


class CorrectionIn(BaseModel):
    state: str
    bad_action: str
    failure_evidence: str
    correct_action: str
    why: str


class TrajectoryIn(BaseModel):
    analysis_id: str | None = None
    prompt_state: str
    retrieved_context: str = ""
    reasoning: str = ""
    steps: list[ToolStepIn] = Field(default_factory=list)
    user_corrections: list[CorrectionIn] = Field(default_factory=list)
    final_result: str = ""
    latency_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verified: bool = False
    semantic_turns: int = 0
    holdout_passed: bool = False
    allow_qlora: bool = False
    frequent_behavior: bool = False
    one_off_fact: bool = False
    regression_passed: bool = False
    dataset_hash: str = ""
    telemetry: dict[str, Any] = Field(default_factory=dict)
    self_audit: dict[str, Any] | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    model_id: str = ""


def _traj(payload: TrajectoryIn) -> Trajectory:
    return Trajectory(
        prompt_state=payload.prompt_state,
        retrieved_context=payload.retrieved_context,
        reasoning=payload.reasoning,
        steps=[ToolStep(**item.model_dump()) for item in payload.steps],
        user_corrections=[Correction(**item.model_dump()) for item in payload.user_corrections],
        final_result=payload.final_result,
        latency_ms=payload.latency_ms,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
        verified=payload.verified,
        semantic_turns=payload.semantic_turns,
        holdout_passed=payload.holdout_passed,
        allow_qlora=payload.allow_qlora,
        frequent_behavior=payload.frequent_behavior,
        one_off_fact=payload.one_off_fact,
        regression_passed=payload.regression_passed,
        dataset_hash=payload.dataset_hash,
        extras={
            "telemetry": payload.telemetry,
            "acceptance_criteria": payload.acceptance_criteria,
            "model_id": payload.model_id,
            # /analyze is an exploratory/manual control-plane surface. A client
            # checkbox or UI default is not an objective verifier receipt.
            "objective_verified": False,
            "trajectory_id": payload.analysis_id or "",
            "manual_analysis": True,
        },
    )


class VerifiedTrainingTargetIn(BaseModel):
    target: str = Field(min_length=1, max_length=24_000)
    source: Literal["human_correction", "objective_correction"]
    source_ref: str = Field(default="", max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=64)


class IngestTurnIn(BaseModel):
    idempotency_key: str | None = Field(default=None, max_length=240)
    session_id: str = ""
    thread_id: str | None = None
    turn_id: str | None = None
    prompt: str = ""
    final_result: str = ""
    critic: dict[str, Any] | None = None
    self_audit: dict[str, Any] | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    acceptance_criteria: list[str] = Field(default_factory=list)
    # `model_id` remains the loadable/base checkpoint used by training.  The
    # effective identity differentiates base and adapter behavior for recurrence
    # and self-audit provenance without inventing an unloadable training model id.
    model_id: str = ""
    effective_model_id: str = ""
    adapter_state: bool | None = None
    verified_training_target: VerifiedTrainingTargetIn | None = None


class AdaptiveCheckpointIn(BaseModel):
    """One safe-boundary checkpoint inside a still-running logical answer."""

    checkpoint_id: str | None = Field(default=None, max_length=240)
    checkpoint_event_seq: int | None = Field(default=None, ge=1)
    run_id: str | None = Field(default=None, max_length=240)
    resume_round: int | None = Field(default=None, ge=0)
    session_id: str = ""
    thread_id: str | None = None
    turn_id: str | None = None
    prompt: str = ""
    partial_result: str = ""
    telemetry: dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""
    effective_model_id: str = ""
    adapter_state: bool | None = None


class DecisionIn(BaseModel):
    decision: str
    features: dict[str, Any] = Field(default_factory=dict)


class DecisionOutcomeIn(BaseModel):
    decision_id: str
    eventual_outcome: bool
    retrospective_usefulness: bool | None = None


@router.post("/analyze")
def analyze_trajectory(payload: TrajectoryIn) -> dict[str, Any]:
    from uuid import uuid4

    if not payload.analysis_id:
        payload = payload.model_copy(update={"analysis_id": f"manual:{uuid4().hex}"})
    return run_adaptation_pipeline(
        _traj(payload), self_audit=payload.self_audit, claims=payload.claims
    )


@router.post("/decision")
def advisory_decision(payload: DecisionIn) -> dict[str, Any]:
    from uuid import uuid4
    from core.helix_engine.decision_controller import shadow_decision
    from core.helix_engine.ledger import append_record

    decision = shadow_decision(
        payload.decision,
        payload.features,
        decision_id=f"manual:{uuid4().hex}",
    )
    append_record("decisions", decision.to_dict())
    return decision.to_dict()


@router.post("/decision-outcome")
def decision_outcome(payload: DecisionOutcomeIn) -> dict[str, Any]:
    from core.helix_engine.decision_controller import record_decision_outcome

    recorded = record_decision_outcome(
        payload.decision_id,
        eventual_outcome=payload.eventual_outcome,
        retrospective_usefulness=payload.retrospective_usefulness,
    )
    return {"recorded": recorded}


@router.get("/decision-calibration")
def decision_calibration() -> dict[str, Any]:
    from core.helix_engine.decision_controller import calibration_summary

    return calibration_summary()


@router.post("/prepare-audit")
def prepare_completed_turn_audit(payload: IngestTurnIn) -> dict[str, Any]:
    """Build backend-resolved observable artifacts before generative self-audit.

    This endpoint is deliberately fail-open. The caller's pre-existing behavior
    was to run the deep audit, so any optional Helix failure returns that choice.
    """
    try:
        from core.helix_engine.audit import prepare_observable_self_audit
        from core.helix_engine.capture import capture_session_key, trajectory_from_session

        key = capture_session_key(payload.session_id, payload.thread_id, payload.turn_id)
        telemetry = dict(payload.telemetry) if isinstance(payload.telemetry, dict) else {}
        telemetry.pop("objective_verified", None)
        trajectory_id = str(payload.turn_id or telemetry.get("trajectory_id") or "")
        behavioral_model_id = str(payload.effective_model_id or payload.model_id or "").strip()
        traj = trajectory_from_session(
            key,
            prompt_state=payload.prompt,
            final_result=payload.final_result,
            latency_ms=float(telemetry.get("latency_ms") or 0),
            prompt_tokens=int(telemetry.get("prompt_tokens") or telemetry.get("promptTokens") or 0),
            completion_tokens=int(telemetry.get("completion_tokens") or telemetry.get("completionTokens") or 0),
            verified=False,
            extras={
                "telemetry": telemetry,
                "acceptance_criteria": list(payload.acceptance_criteria or []),
                "model_id": behavioral_model_id,
                "base_model_id": payload.model_id,
                "adapter_state": payload.adapter_state,
                "objective_verified": False,
                "trajectory_id": trajectory_id,
                "thread_id": payload.thread_id,
                "capture_session_id": key,
            },
        )
        return prepare_observable_self_audit(traj, claims=payload.claims)
    except BaseException as exc:
        return {
            "schema_version": "helix.audit-preparation.v1",
            "available": False,
            "perform_deep_audit": True,
            "fail_open": True,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def _ingest_completed_turn_once(
    payload: IngestTurnIn,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    from core.helix_engine.capture import capture_session_key, session_steps
    from core.helix_engine.training_targets import issue_verified_training_target_receipt

    key = capture_session_key(payload.session_id, payload.thread_id, payload.turn_id)
    target_receipt = None
    target_receipt_error = ""
    if payload.verified_training_target is not None:
        if not payload.turn_id:
            target_receipt_error = "turn_id is required for a verified training target"
        else:
            correction = payload.verified_training_target
            try:
                target_receipt = issue_verified_training_target_receipt(
                    trajectory_id=payload.turn_id,
                    target=correction.target,
                    source=correction.source,
                    verifier_subject=current_subject,
                    source_ref=correction.source_ref,
                    evidence_refs=correction.evidence_refs,
                    steps=session_steps(key),
                    source_thread_id=payload.thread_id,
                )
                if target_receipt is None:
                    target_receipt_error = "verified training target provenance could not be persisted"
            except ValueError as exc:
                target_receipt_error = str(exc)[:500]

    result = ingest_turn(
        session_id=key,
        prompt=payload.prompt,
        final_result=payload.final_result,
        critic=payload.critic,
        self_audit=payload.self_audit,
        claims=payload.claims,
        telemetry=payload.telemetry,
        acceptance_criteria=payload.acceptance_criteria,
        model_id=payload.model_id,
        effective_model_id=payload.effective_model_id,
        adapter_state=payload.adapter_state,
        training_target_receipt=target_receipt,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        subject=current_subject,
        archive=False,
    )
    adaptive = result.get("adaptive_cycle") if isinstance(result.get("adaptive_cycle"), dict) else {}
    adaptation = adaptive.get("adaptation") if isinstance(adaptive.get("adaptation"), dict) else {}
    qlora_outcome: dict[str, str] = {
        "outcome": "not_applicable",
        "reason": "no_final_hermes_qlora_candidate_was_staged",
    }
    if "stage_qlora_candidate" in result.get("actions", []):
        try:
            from routes.self_training import (
                _start_training_for_state,
                queue_hermes_training_with_provenance,
            )

            queue_receipt = queue_hermes_training_with_provenance(
                source_trajectory_id=str(result.get("trajectory_id") or payload.turn_id or ""),
                source_thread_id=payload.thread_id,
            )
            outcome = str(queue_receipt.get("outcome") or "")
            reason = str(queue_receipt.get("reason") or "")[:500]
            if outcome not in {"queued", "deferred", "denied"}:
                outcome = "deferred"
                reason = reason or "final_queue_gate_returned_no_supported_outcome"
            qlora_outcome = {
                "outcome": outcome,
                "reason": reason or "final_queue_gate_returned_no_reason",
            }
            if outcome == "queued":
                result["actions"].append("queue_qlora_training")
                background_tasks.add_task(_start_training_for_state, current_subject)
        except Exception as exc:
            qlora_outcome = {
                "outcome": "deferred",
                "reason": f"final_queue_admission_check_failed:{type(exc).__name__}"[:500],
            }
    elif str(adaptation.get("action") or "") == "QLORA_CANDIDATE":
        qlora_stage = result.get("qlora_stage") if isinstance(result.get("qlora_stage"), dict) else {}
        qlora_outcome = {
            "outcome": "denied",
            "reason": str(
                qlora_stage.get("reason")
                or "final_hermes_qlora_candidate_failed_verified_target_or_candidate_staging_gate"
            )[:500],
        }
    result["qlora_outcome"] = qlora_outcome
    result["actions"] = list(dict.fromkeys(result.get("actions", [])))
    if payload.verified_training_target is not None:
        result["training_target_receipt"] = (
            {"accepted": True, **target_receipt.metadata()}
            if target_receipt is not None
            else {"accepted": False, "reason": target_receipt_error or "verified training target rejected"}
        )
    try:
        from core.helix_engine.provenance import append_turn_receipt

        append_turn_receipt(
            session_id=payload.session_id,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            trajectory_id=str(result.get("trajectory_id") or payload.turn_id or ""),
            model=str(
                result.get("model_id")
                or payload.effective_model_id
                or payload.model_id
                or ""
            ),
            base_model_id=str(result.get("base_model_id") or payload.model_id or ""),
            actions=list(result.get("actions") or []),
            skill_retention=list(result.get("skill_retention") or []),
            qlora_outcome=result.get("qlora_outcome"),
        )
    except Exception:
        # Provenance indexing must never turn completed learning into a failed turn.
        pass
    return result


@router.post("/ingest-turn")
def ingest_completed_turn(
    payload: IngestTurnIn,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    idempotency_key = str(payload.idempotency_key or "").strip()
    if not idempotency_key:
        return _ingest_completed_turn_once(payload, background_tasks, current_subject)

    from core.helix_engine.final_ingest_idempotency import (
        FinalIngestIdempotencyConflict,
        FinalIngestReplayUnavailable,
        claim_final_ingest,
        complete_final_ingest,
        fail_final_ingest,
    )
    from core.helix_engine.capture import (
        capture_session_key,
        capture_snapshot,
        restore_capture_snapshot,
    )

    request_payload = payload.model_dump(mode="json", exclude={"idempotency_key"})
    capture_key = capture_session_key(payload.session_id, payload.thread_id, payload.turn_id)
    durable_capture = capture_snapshot(capture_key)
    try:
        claim, replay = claim_final_ingest(
            idempotency_key=idempotency_key,
            session_id=payload.session_id,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            subject=current_subject,
            payload=request_payload,
            replay_safe=True,
            capture_snapshot=durable_capture,
        )
    except FinalIngestIdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FinalIngestReplayUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if replay is not None:
        try:
            from core.helix_engine.capture import archive_session, capture_session_key

            archive_session(
                capture_session_key(payload.session_id, payload.thread_id, payload.turn_id)
            )
        except Exception:
            pass
        return replay
    if claim is None:  # pragma: no cover - defensive invariant
        raise HTTPException(status_code=500, detail="final ingest idempotency claim missing")
    restore_capture_snapshot(capture_key, claim.get("capture_snapshot"))

    try:
        result = _ingest_completed_turn_once(payload, background_tasks, current_subject)
    except BaseException as exc:
        fail_final_ingest(claim, exc)
        raise

    try:
        completed = complete_final_ingest(claim, result)
    except FinalIngestReplayUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        from core.helix_engine.capture import archive_session, capture_session_key

        archive_session(
            capture_session_key(payload.session_id, payload.thread_id, payload.turn_id)
        )
    except Exception:
        # Receipt completion is authoritative. Capture cleanup is bounded storage
        # hygiene and can be retried on a later replay/startup.
        pass
    return completed


@router.post("/adaptive-checkpoint")
def adaptive_checkpoint(
    payload: AdaptiveCheckpointIn,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    """Persist learning and run Helix/Hermes without consuming the live turn.

    This endpoint deliberately has no ``BackgroundTasks`` argument: a checkpoint
    may nominate a QLoRA candidate, but training must wait until the logical answer
    has completed and released inference ownership.
    """

    from core.helix_engine.adaptive_checkpoint import run_adaptive_checkpoint

    return run_adaptive_checkpoint(
        session_id=payload.session_id,
        thread_id=payload.thread_id,
        turn_id=payload.turn_id,
        prompt=payload.prompt,
        partial_result=payload.partial_result,
        telemetry=payload.telemetry,
        model_id=payload.model_id,
        effective_model_id=payload.effective_model_id,
        adapter_state=payload.adapter_state,
        subject=current_subject,
        checkpoint_id=payload.checkpoint_id,
        checkpoint_event_seq=payload.checkpoint_event_seq,
        run_id=payload.run_id,
        resume_round=payload.resume_round,
    )


@router.post("/semantic-turns")
def semantic_turns(payload: dict[str, list[str]]) -> dict[str, Any]:
    return monitor_semantic_turns(payload.get("turns") or [])


@router.get("/provenance")
def provenance(
    session_id: str | None = Query(default=None, max_length=500),
    thread_id: str | None = Query(default=None, max_length=200),
    turn_id: str | None = Query(default=None, max_length=200),
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    """Return bounded account-scoped receipts for one requested/latest logical turn."""

    _ = current_subject
    from core.helix_engine.provenance import read_provenance

    return read_provenance(
        session_id=session_id,
        thread_id=thread_id,
        turn_id=turn_id,
    )


@router.get("/live-feed")
def live_feed(
    session_id: str | None = None,
    thread_id: str | None = None,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    _ = current_subject
    from core.helix_event_sequence import public_event_scope_id

    key = feed_session_id(session_id, thread_id)
    public_key = public_event_scope_id(session_id, thread_id)
    events = [
        {**event, "session_id": public_key}
        for event in list_live_feed(key)
    ]
    return {"session_id": public_key, "events": events}


@router.get("/session/{session_id}")
def session_trace(
    session_id: str,
    thread_id: str | None = None,
    turn_id: str | None = None,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    _ = current_subject
    from core.helix_engine.capture import (
        capture_session_key,
        latest_session_snapshot,
        session_control_events,
        session_steps,
    )

    key = capture_session_key(session_id, thread_id, turn_id)
    from core.helix_engine.evidence import tool_verification_receipt

    if turn_id:
        resolved_turn_id = str(turn_id)
        steps = session_steps(key)
        control_events = session_control_events(key)
    else:
        resolved_turn_id, steps, control_events = latest_session_snapshot(session_id, thread_id)
    if not steps and not control_events and not resolved_turn_id:
        raise HTTPException(status_code=404, detail="Helix session not found.")
    payload_steps = []
    for index, step in enumerate(steps):
        evidence_ids = [f"tool:{index}:error" if step.error else f"tool:{index}:result"]
        if tool_verification_receipt(step):
            evidence_ids.append(f"tool:{index}:verification")
        payload_steps.append(
            {
                "index": index,
                "evidence_ids": evidence_ids,
                "name": step.name,
                "arguments": step.arguments[:2_000],
                "result": step.result[:1_200],
                "useful_hint": step.useful_hint,
                "error": step.error,
                "retry": step.retry,
                "sequence": step.sequence,
                "created_at_ms": step.created_at_ms,
            }
        )
    return {
        "session_id": key,
        "turn_id": resolved_turn_id or None,
        "steps": payload_steps,
        "control_events": [event.to_dict() for event in control_events[-200:]],
    }


@router.get("/hub/local")
def hub_local(root: str = "") -> dict[str, Any]:
    from pathlib import Path

    models = list_local_gguf(Path(root).expanduser()) if root else []
    return {"models": models}


@router.post("/hub/select")
def hub_select(payload: dict[str, str]) -> dict[str, Any]:
    return select_local_model(payload.get("path") or "")


@router.post("/hub/load")
async def hub_load(
    payload: dict[str, str],
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    from models.inference import LoadRequest
    from routes.inference import load_model_gated

    selected = select_local_model(payload.get("path") or "")
    if not selected.get("ok"):
        return selected

    async def _load(path: str):
        return await load_model_gated(
            LoadRequest(model_path=path),
            fastapi_request,
            current_subject,
            user_initiated=True,
        )

    try:
        loaded = await _load(selected["path"])
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:800], "path": selected["path"], "loaded": False}
    return load_local_model(selected["path"], load_fn=lambda path: loaded)


@router.post("/hub/download")
def hub_download(payload: dict[str, str]) -> dict[str, Any]:
    from pathlib import Path

    return download_local_model(payload.get("source") or "", Path(payload.get("dest") or "").expanduser())


@router.post("/deficits")
def deficits(payload: list[TrajectoryIn]) -> dict[str, Any]:
    return {"clusters": cluster_deficits(_traj(item) for item in payload)}
