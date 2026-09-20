# SPDX-License-Identifier: AGPL-3.0-only
"""Durable backend-owned post-answer finalization for Studio chat runs.

The generation row is the transaction coordinator.  Every effect below already
has its own stable logical-turn idempotency key; this module only sequences those
effects and records their receipts.  Re-entering after a process crash therefore
replays receipts rather than duplicating memory, examples, or Helix ingest.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import BackgroundTasks

from storage import chat_generation_runs_db as db

_LEARNING_PROTOCOL_PREFIXES = (
    "[UNSLOTH_HERMES_LEARNING_REVIEW]",
    "[HELIX_SELF_CRITIC]",
    "[HELIX_SELF_AUDIT]",
)
_HIDDEN_LEARNING_TAGS = ("unsloth-learning", "unsloth-skill-draft")


class FinalizationClaimLost(RuntimeError):
    """The durable claim was expired or fenced by another finalizer."""


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") in {None, "text", "input_text", "output_text"}:
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part for part in parts if part)


def _last_user_text(request_payload: dict[str, Any]) -> str:
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        value = _text_content(message.get("content"))
        if value.strip():
            return value
    return ""


def _all_events(run_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    after = 0
    while True:
        page = db.list_events(run_id, after=after, limit=1000)
        if not page:
            break
        events.extend(page)
        after = int(page[-1]["seq"])
        if len(page) < 1000:
            break
    return events


def _assistant_text(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") != "chunk":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        text = _text_content(delta.get("content"))
        if text:
            parts.append(text)
    return "".join(parts)


def _usage_telemetry(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    tool_events = 0
    adaptive_events = 0
    for event in events:
        if event.get("type") != "chunk":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("usage")
        if isinstance(candidate, dict):
            usage = dict(candidate)
        event_type = str(payload.get("type") or "")
        if event_type.startswith("tool_"):
            tool_events += 1
        elif event_type == "adaptive_checkpoint":
            adaptive_events += 1
    telemetry: dict[str, Any] = {
        "finalization_owner": "backend_turn_finalizer",
        "durable_event_count": len(events),
        "tool_control_frame_count": tool_events,
        "adaptive_checkpoint_frame_count": adaptive_events,
    }
    for source, target in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, (int, float)):
            telemetry[target] = value
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        telemetry["completion_tokens_details"] = details
    return telemetry


def _is_learning_protocol(prompt: str) -> bool:
    return any(prefix in prompt for prefix in _LEARNING_PROTOCOL_PREFIXES)


def _strip_hidden_learning(text: str) -> str:
    value = text
    for tag in _HIDDEN_LEARNING_TAGS:
        value = re.sub(
            rf"<{re.escape(tag)}>\s*[\s\S]*?\s*</{re.escape(tag)}>",
            "",
            value,
            flags=re.IGNORECASE,
        )
    return value.strip()


def _learning_excerpt(text: str, limit: int) -> str:
    """Bound text without discarding the completed turn's final evidence.

    Prefix-only truncation loses the exact correction, last tool result, or final
    acceptance statement on long turns. Preserve both ends with an explicit gap
    marker so downstream learning never mistakes the splice for contiguous text.
    """
    value = str(text or "").strip()
    cap = max(128, int(limit))
    if len(value) <= cap:
        return value
    marker = "\n\n[...middle omitted by durable finalizer...]\n\n"
    room = max(1, cap - len(marker))
    head = max(1, int(room * 0.55))
    tail = max(1, room - head)
    return value[:head].rstrip() + marker + value[-tail:].lstrip()


async def _phase(
    run_id: str,
    token: str,
    name: str,
    payload: dict[str, Any],
) -> None:
    seq = await asyncio.to_thread(
        db.append_finalization_event,
        run_id,
        token,
        f"turn_finalization.{name}",
        payload,
    )
    if seq is None:
        raise FinalizationClaimLost(f"Lost finalization claim before {name}")


async def _require_claim(run_id: str, token: str) -> None:
    if not await asyncio.to_thread(db.renew_finalization_claim, run_id, token):
        raise FinalizationClaimLost("Lost finalization claim")


async def _claim_heartbeat(
    run_id: str,
    token: str,
    owner_task: asyncio.Task[Any],
) -> None:
    interval = max(0.25, db.FINALIZATION_LEASE_MS / 3000.0)
    while True:
        await asyncio.sleep(interval)
        try:
            owned = await asyncio.to_thread(db.renew_finalization_claim, run_id, token)
        except Exception:
            # Transient database contention is not proof that ownership was lost;
            # the next renewal or phase append remains the authoritative fence.
            continue
        if not owned:
            owner_task.cancel()
            return


def _committed_phase_receipts(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = {
        "memory": "memory",
        "self_training": "selfTraining",
        "audit": "audit",
        "helix_ingest": "helix",
        "qlora_dispatch": "qloraDispatch",
    }
    receipts: dict[str, Any] = {}
    for event in events:
        event_type = str(event.get("type") or "")
        prefix = "turn_finalization."
        if not event_type.startswith(prefix):
            continue
        key = names.get(event_type[len(prefix) :])
        payload = event.get("payload")
        if key is not None and isinstance(payload, dict):
            receipts[key] = dict(payload)
    return receipts


def _small_memory_receipt(value: dict[str, Any]) -> dict[str, Any]:
    node = value.get("node") if isinstance(value.get("node"), dict) else {}
    return {
        "stored": value.get("stored") is True,
        "reason": str(value.get("reason") or "")[:500] or None,
        "nodeId": str(node.get("id") or "")[:200] or None,
    }


def _small_self_training_receipt(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded": value.get("recorded") is True,
        "idempotent": value.get("idempotent") is True,
        "reason": str(value.get("reason") or "")[:500] or None,
        "exampleId": str(value.get("exampleId") or "")[:200] or None,
    }


def _small_helix_receipt(value: dict[str, Any]) -> dict[str, Any]:
    qlora = value.get("qlora_outcome") if isinstance(value.get("qlora_outcome"), dict) else {}
    return {
        "trajectoryId": str(value.get("trajectory_id") or "")[:200] or None,
        "actions": [str(item)[:200] for item in list(value.get("actions") or [])[:32]],
        "qloraOutcome": {
            "outcome": str(qlora.get("outcome") or "")[:80] or None,
            "reason": str(qlora.get("reason") or "")[:500] or None,
        },
    }


async def finalize_run(run_id: str, *, app: Any = None) -> dict[str, Any] | None:
    """Claim and finish one logical turn; safe to call repeatedly."""

    claim = await asyncio.to_thread(db.claim_finalization, run_id)
    if claim is None:
        return await asyncio.to_thread(db.get_run, run_id)
    run, token, owner = claim
    receipts: dict[str, Any] = {}
    owner_task = asyncio.current_task()
    heartbeat = (
        asyncio.create_task(
            _claim_heartbeat(run_id, token, owner_task),
            name=f"turn-finalization-lease:{run_id}",
        )
        if owner_task is not None
        else None
    )
    try:
        request_payload = run.get("requestPayload")
        if not isinstance(request_payload, dict):
            settled = await asyncio.to_thread(
                db.finish_finalization,
                run_id,
                token,
                status="failed",
                receipt={"reason": "request_payload_unavailable"},
                error="Durable finalization request payload is unavailable",
            )
            return settled or await asyncio.to_thread(db.get_run, run_id)
        if str(request_payload.get("finalization_idempotency_key") or "").strip() != run_id:
            return await asyncio.to_thread(
                db.finish_finalization,
                run_id,
                token,
                status="skipped",
                receipt={"reason": "missing_or_mismatched_idempotency_key"},
            )

        events = await asyncio.to_thread(_all_events, run_id)
        receipts.update(_committed_phase_receipts(events))
        prompt = _last_user_text(request_payload).strip()
        raw_completion = _assistant_text(events).strip()
        training_completion = _strip_hidden_learning(raw_completion)
        if not prompt or not raw_completion or not training_completion:
            reason = "missing_prompt" if not prompt else "missing_completion"
            return await asyncio.to_thread(
                db.finish_finalization,
                run_id,
                token,
                status="skipped",
                receipt={"reason": reason},
            )
        if _is_learning_protocol(prompt):
            return await asyncio.to_thread(
                db.finish_finalization,
                run_id,
                token,
                status="skipped",
                receipt={"reason": "learning_protocol_turn"},
            )

        thread_id = str(run.get("threadId") or "")
        model_id = str(request_payload.get("model") or "").strip()
        adapter_state = request_payload.get("use_adapter")
        if not isinstance(adapter_state, bool):
            adapter_state = None
        effective_model_id = f"{model_id}::adapter={('enabled' if adapter_state else 'disabled') if adapter_state is not None else 'unspecified'}"
        session_id = str(request_payload.get("session_id") or thread_id or "default")
        telemetry = _usage_telemetry(events)
        telemetry["finalization_idempotency_key"] = run_id

        if "memory" not in receipts:
            from routes.learning import _read_state as _read_learning_state
            from routes.memory import persist_memory_experience

            await _require_claim(run_id, token)
            learning_state = await asyncio.to_thread(_read_learning_state)
            memory_result = await asyncio.to_thread(
                persist_memory_experience,
                owner,
                (
                    f"User: {_learning_excerpt(prompt, 1500)}\n"
                    f"Assistant: {_learning_excerpt(raw_completion, 1500)}"
                ),
                thread_id=thread_id or None,
                kind="experience",
                title=prompt[:80] or "Completed turn",
                enabled=learning_state.get("mem0Enabled") is not False,
                idempotency_key=run_id,
            )
            receipts["memory"] = _small_memory_receipt(memory_result)
            await _phase(run_id, token, "memory", receipts["memory"])

        from routes.self_training import SelfTrainingExampleRequest, record_self_training_example

        if "selfTraining" not in receipts:
            await _require_claim(run_id, token)
            example_result = await record_self_training_example(
                SelfTrainingExampleRequest(
                    modelId=model_id,
                    prompt=prompt,
                    completion=training_completion,
                    sourceThreadId=thread_id or None,
                    idempotencyKey=run_id,
                ),
                BackgroundTasks(),
                owner,
            )
            receipts["selfTraining"] = _small_self_training_receipt(example_result)
            await _phase(run_id, token, "self_training", receipts["selfTraining"])

        from routes.helix_engine import (
            IngestTurnIn,
            ingest_completed_turn,
            prepare_completed_turn_audit,
        )

        ingest_payload = IngestTurnIn(
            idempotency_key=run_id,
            session_id=session_id,
            thread_id=thread_id or None,
            turn_id=run_id,
            prompt=_learning_excerpt(prompt, 4_000),
            final_result=_learning_excerpt(raw_completion, 4_000),
            telemetry=telemetry,
            model_id=model_id,
            effective_model_id=effective_model_id,
            adapter_state=adapter_state,
        )
        audit_receipt = receipts.get("audit")
        if isinstance(audit_receipt, dict):
            audit_requested = audit_receipt.get("requested") is True
            ingest_payload.telemetry["self_audit_requested"] = audit_requested
            stored_audit = audit_receipt.get("selfAudit")
            stored_claims = audit_receipt.get("claims")
            if isinstance(stored_audit, dict):
                ingest_payload.self_audit = stored_audit
                ingest_payload.claims = stored_claims if isinstance(stored_claims, list) else []
                ingest_payload.telemetry["self_audit_performed"] = True
                ingest_payload.telemetry.pop("self_audit_skipped_reason", None)
            else:
                ingest_payload.telemetry["self_audit_performed"] = False
                ingest_payload.telemetry["self_audit_skipped_reason"] = str(
                    audit_receipt.get("reason")
                    or ("backend_deterministic_fallback" if audit_requested else "pre_audit_decision_skip")
                )[:200]
        else:
            await _require_claim(run_id, token)
            audit_preparation = await asyncio.to_thread(prepare_completed_turn_audit, ingest_payload)
            audit_requested = bool(
                isinstance(audit_preparation, dict)
                and audit_preparation.get("perform_deep_audit") is True
            )
            ingest_payload.telemetry["self_audit_requested"] = audit_requested
            audit_receipt = {
                "requested": audit_requested,
                "performed": False,
                "fallback": "backend_deterministic_observable_audit",
            }
            if audit_requested:
                from core.inference.turn_finalizer_audit import run_optional_model_audit

                artifacts = (
                    audit_preparation.get("artifacts")
                    if isinstance(audit_preparation, dict)
                    and isinstance(audit_preparation.get("artifacts"), dict)
                    else {
                        "objective": _learning_excerpt(prompt, 4_000),
                        "final_result": _learning_excerpt(raw_completion, 4_000),
                        "telemetry": telemetry,
                    }
                )
                model_audit, claims, model_receipt = await run_optional_model_audit(
                    app=app,
                    owner=owner,
                    run_id=run_id,
                    original_model_id=model_id,
                    adapter_state=adapter_state,
                    focus=(
                        f"User: {_learning_excerpt(prompt, 1500)}\n"
                        f"Assistant: {_learning_excerpt(raw_completion, 1500)}"
                    ),
                    artifacts=artifacts,
                )
                audit_receipt.update(model_receipt)
                if model_audit is not None:
                    ingest_payload.self_audit = model_audit
                    ingest_payload.claims = claims
                    ingest_payload.telemetry["self_audit_performed"] = True
                    ingest_payload.telemetry.pop("self_audit_skipped_reason", None)
                    # The next retry must reconstruct exactly the audit committed
                    # by this phase rather than ask the model a second time.
                    audit_receipt["selfAudit"] = model_audit
                    audit_receipt["claims"] = claims
                else:
                    ingest_payload.telemetry["self_audit_performed"] = False
                    ingest_payload.telemetry["self_audit_skipped_reason"] = str(
                        model_receipt.get("reason") or "backend_deterministic_fallback"
                    )[:200]
            else:
                ingest_payload.telemetry["self_audit_performed"] = False
                ingest_payload.telemetry["self_audit_skipped_reason"] = "pre_audit_decision_skip"
            receipts["audit"] = audit_receipt
            await _phase(run_id, token, "audit", receipts["audit"])

        background: BackgroundTasks | None = None
        if "helix" not in receipts:
            await _require_claim(run_id, token)
            background = BackgroundTasks()
            helix_result = await asyncio.to_thread(
                ingest_completed_turn,
                ingest_payload,
                background,
                owner,
            )
            receipts["helix"] = _small_helix_receipt(helix_result)
            await _phase(run_id, token, "helix_ingest", receipts["helix"])

        if "qloraDispatch" not in receipts:
            qlora = (
                receipts.get("helix", {}).get("qloraOutcome", {})
                if isinstance(receipts.get("helix"), dict)
                else {}
            )
            qlora_outcome = str(qlora.get("outcome") or "")
            if qlora_outcome == "queued":
                await _require_claim(run_id, token)
                if background is not None:
                    await background()
                else:
                    # Ingest replay returns its durable receipt with an empty
                    # BackgroundTasks object. Dispatch directly; the callee
                    # rechecks foreground/training gates and no-ops unless the
                    # durable self-training state is still queued.
                    from routes.self_training import _start_training_for_state

                    await _start_training_for_state(owner)
                dispatch_receipt = {"outcome": "dispatched", "qloraOutcome": "queued"}
            else:
                dispatch_receipt = {
                    "outcome": "not_required",
                    "qloraOutcome": qlora_outcome or "not_applicable",
                }
            receipts["qloraDispatch"] = dispatch_receipt
            await _phase(run_id, token, "qlora_dispatch", dispatch_receipt)

        settled = await asyncio.to_thread(
            db.finish_finalization,
            run_id,
            token,
            status="completed",
            receipt=receipts,
        )
        return settled or await asyncio.to_thread(db.get_run, run_id)
    except FinalizationClaimLost:
        return await asyncio.to_thread(db.get_run, run_id)
    except asyncio.CancelledError:
        # Leave the durable claim in running state. Startup reconciliation turns
        # it back into pending, after which the same idempotency keys safely replay.
        raise
    except Exception as exc:
        # Every effect above is individually keyed by run_id, so a later explicit
        # retry or process-recovery re-entry can safely continue from receipts.
        await asyncio.to_thread(
            db.retry_finalization,
            run_id,
            token,
            receipt=receipts or None,
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )
        return await asyncio.to_thread(db.get_run, run_id)
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
