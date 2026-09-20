# SPDX-License-Identifier: AGPL-3.0-only
"""Durable approval checkpoints and effect admission for local tool loops."""

from __future__ import annotations

import contextvars
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator
from uuid import uuid4

from storage import chat_generation_runs_db as runs_db
from utils.account_context import current_account_id
from core.inference.tool_producer_receipt import ObservationSeed

CHECKPOINT_VERSION = 1
APPROVAL_TTL_MS = 3_600_000


class DurableToolJournalError(RuntimeError):
    pass


@dataclass(frozen=True)
class DurableExecutionHandle:
    execution_id: str
    run_id: str = ""
    worker_token: str = ""
    approval_id: str = ""
    claim_token: str = ""
    pre_tool_checkpoint_digest: str = ""
    authority_kind: str = "ungated"
    replay_completion: dict[str, Any] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class PreparedDurableResult:
    value: Any
    execution: DurableExecutionHandle
    producer_receipt: dict[str, Any] | None = None
    observation_seed: ObservationSeed | None = None
    replay_completion: dict[str, Any] | None = None


_CURRENT: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "helix_durable_tool_run", default=None
)
_NEXT_APPROVAL: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "helix_durable_tool_approval", default=None
)
_RESUME_CHECKPOINT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "helix_durable_tool_resume_checkpoint", default=None
)
_CLAIMED_EXECUTION: contextvars.ContextVar[DurableExecutionHandle | None] = (
    contextvars.ContextVar("helix_durable_tool_claim", default=None)
)


@contextmanager
def durable_tool_run(
    run_id: str,
    worker_token: str,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
) -> Iterator[None]:
    current_token = _CURRENT.set((str(run_id), str(worker_token)))
    approval_token = _NEXT_APPROVAL.set(None)
    resume_token = _RESUME_CHECKPOINT.set(resume_checkpoint)
    claim_token = _CLAIMED_EXECUTION.set(None)
    try:
        yield
    finally:
        _CLAIMED_EXECUTION.reset(claim_token)
        _RESUME_CHECKPOINT.reset(resume_token)
        _NEXT_APPROVAL.reset(approval_token)
        _CURRENT.reset(current_token)


def current_durable_tool_run() -> tuple[str, str] | None:
    return _CURRENT.get()


def current_resume_checkpoint(*, backend: str | None = None) -> dict[str, Any] | None:
    checkpoint = _RESUME_CHECKPOINT.get()
    if not isinstance(checkpoint, dict):
        return None
    if int(checkpoint.get("version") or 0) != CHECKPOINT_VERSION:
        raise DurableToolJournalError("durable tool checkpoint version is incompatible")
    if backend is not None and str(checkpoint.get("backend") or "") != backend:
        raise DurableToolJournalError("durable tool checkpoint backend does not match")
    return checkpoint


def bind_next_tool_approval(run_id: str, approval_id: str) -> None:
    bound = _CURRENT.get()
    if bound is None or bound[0] != str(run_id):
        raise DurableToolJournalError("approval does not belong to the active durable run")
    _NEXT_APPROVAL.set((str(run_id), str(approval_id)))


def _canonical_arguments(arguments: dict[str, Any]) -> tuple[str, str]:
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_pre_tool_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Detach mutable loop state and return its canonical identity."""

    encoded = json.dumps(
        checkpoint,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    frozen = json.loads(encoded)
    if not isinstance(frozen, dict):
        raise DurableToolJournalError("durable tool checkpoint must be an object")
    if int(frozen.get("version") or 0) != CHECKPOINT_VERSION:
        raise DurableToolJournalError("unsupported durable checkpoint version")
    return frozen, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def proposal_envelope(
    *,
    approval_id: str,
    session_id: str | None,
    tool_name: str,
    tool_call_id: str,
    card_call_id: str,
    arguments: dict[str, Any],
    resume_checkpoint: dict[str, Any],
    expires_at: int | None = None,
) -> dict[str, Any] | None:
    """Private envelope consumed by the durable supervisor at ``tool_start``."""

    if _CURRENT.get() is None or not approval_id:
        return None
    checkpoint, _checkpoint_digest = freeze_pre_tool_checkpoint(resume_checkpoint)
    checkpoint.setdefault("version", CHECKPOINT_VERSION)
    if int(checkpoint.get("version") or 0) != CHECKPOINT_VERSION:
        raise DurableToolJournalError("unsupported durable checkpoint version")
    _encoded, fingerprint = _canonical_arguments(arguments)
    proposed_at = runs_db.now_ms()
    return {
        "schema_version": "helix.tool-approval.v2",
        "approval_id": str(approval_id),
        "session_id": str(session_id or ""),
        "tool_name": str(tool_name or "unknown")[:240],
        "tool_call_id": str(tool_call_id or "")[:500],
        "card_call_id": str(card_call_id or tool_call_id or "")[:500],
        "execution_id": f"tool-exec-{uuid4().hex}",
        "arguments": arguments,
        "arguments_fingerprint": fingerprint,
        "checkpoint_version": CHECKPOINT_VERSION,
        "resume_checkpoint": checkpoint,
        "proposed_at": proposed_at,
        "expires_at": int(expires_at or (proposed_at + APPROVAL_TTL_MS)),
    }


def _append(event_type: str, payload: dict[str, Any]) -> int | None:
    bound = _CURRENT.get()
    if bound is None:
        return None
    run_id, worker_token = bound
    try:
        sequences = runs_db.append_events(
            run_id,
            worker_token,
            [(event_type, payload, runs_db.now_ms())],
        )
    except BaseException as exc:
        raise DurableToolJournalError(
            f"Could not persist {event_type} before continuing the durable agent run"
        ) from exc
    if not sequences:
        raise DurableToolJournalError(
            f"Durable run no longer accepts {event_type}; tool execution was fenced"
        )
    return int(sequences[0])


def _ungated_execution_id(run_id: str, tool_call_id: str, card_call_id: str) -> str:
    stable = json.dumps(
        [str(run_id), str(tool_call_id or ""), str(card_call_id or "")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "tool-exec-" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]


def claim_execution(
    tool_name: str,
    tool_call_id: str,
    *,
    card_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    session_id: str | None = None,
    thread_id: str | None = None,
    pre_tool_checkpoint: dict[str, Any] | None = None,
) -> DurableExecutionHandle | None:
    bound = _CURRENT.get()
    if bound is None:
        return None
    run_id, worker_token = bound
    actual_arguments = arguments if isinstance(arguments, dict) else {}
    if not isinstance(pre_tool_checkpoint, dict):
        raise DurableToolJournalError("durable tool pre-execution checkpoint is missing")
    frozen_checkpoint, checkpoint_digest = freeze_pre_tool_checkpoint(
        pre_tool_checkpoint
    )
    card_call_id = str(card_call_id or tool_call_id or "")
    worker_run = runs_db.get_worker_run(run_id, worker_token)
    if worker_run is None:
        raise DurableToolJournalError("durable tool worker ownership was fenced")
    run, _owner, _token = worker_run
    bound_thread_id = str(run.get("threadId") or "")
    if thread_id is not None and str(thread_id) != bound_thread_id:
        raise DurableToolJournalError("tool invocation thread does not match durable run")
    next_approval = _NEXT_APPROVAL.get()
    if next_approval is not None:
        _NEXT_APPROVAL.set(None)
        if next_approval[0] != run_id:
            raise DurableToolJournalError("tool approval/run identity mismatch")
        approval_id = next_approval[1]
        approval = runs_db.get_tool_approval(run_id, approval_id, include_checkpoint=True)
        if approval is None:
            raise DurableToolJournalError("approved tool call no longer exists")
        if (
            str(approval.get("toolName") or "") != str(tool_name or "unknown")
            or str(approval.get("toolCallId") or "") != str(tool_call_id or "")
            or str(approval.get("cardCallId") or approval.get("toolCallId") or "")
            != card_call_id
        ):
            raise DurableToolJournalError("approved tool identity does not match execution")
        approved_checkpoint, approved_digest = freeze_pre_tool_checkpoint(
            approval.get("resumeCheckpoint") or {}
        )
        if approved_digest != checkpoint_digest or approved_checkpoint != frozen_checkpoint:
            raise DurableToolJournalError(
                "approved tool checkpoint does not match execution"
            )
        claimed = runs_db.claim_tool_execution(
            run_id,
            approval_id,
            worker_token=worker_token,
            backend_account_id=current_account_id(),
            session_id=session_id,
            thread_id=bound_thread_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            card_call_id=card_call_id,
            arguments=actual_arguments,
        )
        if claimed is None:
            raise DurableToolJournalError("approved tool execution could not be claimed")
        handle = DurableExecutionHandle(
            execution_id=str(claimed["executionId"]),
            run_id=run_id,
            worker_token=worker_token,
            approval_id=approval_id,
            claim_token=str(claimed.get("claimToken") or ""),
            pre_tool_checkpoint_digest=checkpoint_digest,
            authority_kind="approved",
            replay_completion=(
                claimed.get("completion")
                if claimed.get("executionState") in {"finished", "ambiguous"}
                else None
            ),
        )
        _CLAIMED_EXECUTION.set(handle)
        return handle

    execution_id = _ungated_execution_id(run_id, tool_call_id, card_call_id)
    claimed = runs_db.claim_ungated_tool_execution(
        run_id,
        execution_id,
        worker_token=worker_token,
        backend_account_id=current_account_id(),
        session_id=session_id,
        thread_id=bound_thread_id,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        card_call_id=card_call_id,
        arguments=actual_arguments,
        pre_tool_checkpoint=frozen_checkpoint,
        pre_tool_checkpoint_digest=checkpoint_digest,
    )
    if claimed is None:
        raise DurableToolJournalError("ungated tool execution could not be claimed")
    handle = DurableExecutionHandle(
        execution_id=execution_id,
        run_id=run_id,
        worker_token=worker_token,
        claim_token=str(claimed.get("claimToken") or ""),
        pre_tool_checkpoint_digest=checkpoint_digest,
        authority_kind="ungated",
        replay_completion=(
            claimed.get("completion")
            if claimed.get("executionState") in {"finished", "ambiguous"}
            else None
        ),
    )
    _CLAIMED_EXECUTION.set(handle)
    return handle


def record_execution_started(
    tool_name: str,
    tool_call_id: str,
) -> DurableExecutionHandle | None:
    execution = _CLAIMED_EXECUTION.get()
    _CLAIMED_EXECUTION.set(None)
    if execution is None:
        return None
    if execution.replay_completion is not None:
        return execution
    if execution.authority_kind == "approved":
        started = runs_db.mark_tool_execution_started(
            execution.run_id,
            execution.approval_id,
            worker_token=execution.worker_token,
            claim_token=execution.claim_token,
        )
    else:
        started = runs_db.mark_ungated_tool_execution_started(
            execution.run_id,
            execution.execution_id,
            worker_token=execution.worker_token,
            claim_token=execution.claim_token,
            pre_tool_checkpoint_digest=execution.pre_tool_checkpoint_digest,
        )
    if started is None:
        raise DurableToolJournalError("tool execution was fenced before worker start")
    return execution


def record_approval_decision(
    *, tool_name: str, tool_call_id: str, approval_id: str, decision: str
) -> int | None:
    # v2 decisions are committed by the authenticated endpoint/expiry path in
    # the same transaction as their authoritative row.  Avoid a second event.
    binding = _CURRENT.get()
    if not approval_id or binding is None:
        return None
    if runs_db.get_tool_approval(
        binding[0], approval_id, include_checkpoint=False
    ):
        return None
    return _append(
        "approval.decided",
        {
            "schema_version": "helix.tool-approval.v1",
            "tool_name": str(tool_name or "unknown")[:240],
            "tool_call_id": str(tool_call_id or "")[:500],
            "approval_id": str(approval_id)[:500],
            "decision": "allow" if decision == "allow" else "deny",
        },
    )


def record_execution_finished(
    execution: DurableExecutionHandle | None,
    *,
    tool_name: str,
    tool_call_id: str,
    result: Any = None,
    error: BaseException | None = None,
    ambiguous: bool = False,
    producer_receipt: dict[str, Any] | None = None,
    controller_is_error: bool | None = None,
    completion_annotations: dict[str, Any] | None = None,
    post_controller_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if execution is None or _CURRENT.get() is None:
        return None
    if execution.authority_kind == "approved":
        finished = runs_db.finish_tool_execution(
            execution.run_id,
            execution.approval_id,
            worker_token=execution.worker_token,
            claim_token=execution.claim_token,
            result=result,
            error=error,
            ambiguous=ambiguous,
            producer_receipt=producer_receipt,
            controller_is_error=controller_is_error,
            completion_annotations=completion_annotations,
            post_controller_checkpoint=post_controller_checkpoint,
        )
        if finished is None:
            raise DurableToolJournalError("tool completion receipt was fenced")
        return finished
    finished = runs_db.finish_ungated_tool_execution(
        execution.run_id,
        execution.execution_id,
        worker_token=execution.worker_token,
        claim_token=execution.claim_token,
        result=result,
        error=error,
        ambiguous=ambiguous,
        producer_receipt=producer_receipt,
        controller_is_error=controller_is_error,
        completion_annotations=completion_annotations,
        post_controller_checkpoint=post_controller_checkpoint,
        pre_tool_checkpoint_digest=execution.pre_tool_checkpoint_digest,
    )
    if finished is None:
        raise DurableToolJournalError("tool completion receipt was fenced")
    return finished


def prepared_result(
    value: Any, execution: DurableExecutionHandle | None
) -> Any:
    if execution is None:
        return value
    from core.inference.tool_producer_receipt import receipt_dict

    seed = getattr(value, "observation_seed", None)
    if not isinstance(seed, ObservationSeed):
        seed = None

    return PreparedDurableResult(
        value=value,
        execution=execution,
        producer_receipt=receipt_dict(value),
        observation_seed=seed,
        replay_completion=execution.replay_completion,
    )


def durable_result_value(result: Any) -> Any:
    """Return the selected producer value without discarding durable metadata."""

    return result.value if isinstance(result, PreparedDurableResult) else result


def replace_durable_result_value(result: Any, value: Any) -> Any:
    """Apply a pre-controller transformation while preserving its execution handle."""

    if not isinstance(result, PreparedDurableResult):
        return value
    return PreparedDurableResult(
        value=value,
        execution=result.execution,
        producer_receipt=result.producer_receipt,
        observation_seed=result.observation_seed,
        replay_completion=result.replay_completion,
    )


def _restore_completion(decision: Any, controller: Any, stored: dict[str, Any]) -> Any:
    from core.inference.tool_loop_controller import ToolCallCompletion

    checkpoint = stored.get("post_controller_checkpoint")
    if not isinstance(checkpoint, dict):
        raise DurableToolJournalError(
            "committed tool completion has no post-controller checkpoint"
        )
    controller.restore_state(checkpoint)
    annotations = stored.get("completion_annotations")
    if isinstance(annotations, dict):
        decision.provenance.clear()
        decision.provenance.update(annotations)
    return ToolCallCompletion(
        decision=decision,
        result=str(stored.get("selected_result") or ""),
        is_error=bool(stored.get("controller_is_error")),
        executed=True,
    )


def settle_controller_completion(result: Any, decision: Any, controller: Any) -> Any:
    """Select once, commit once, and replay without re-running controller logic."""

    if not isinstance(result, PreparedDurableResult):
        return controller.record_result(decision, result)
    if isinstance(result.replay_completion, dict):
        return _restore_completion(decision, controller, result.replay_completion)
    completion = controller.record_result(decision, result.value)
    finished = record_execution_finished(
        result.execution,
        tool_name=decision.tool_name,
        tool_call_id=decision.card_id,
        result=str(completion.result),
        producer_receipt=result.producer_receipt,
        controller_is_error=bool(completion.is_error),
        completion_annotations=dict(decision.provenance),
        post_controller_checkpoint=controller.export_state(),
    )
    # The finisher returns the row it committed (including terminal replay).
    # Do not issue a second lookup: a lost response must be settled from the
    # same transaction's authoritative completion, and a concurrent recovery
    # worker must not change which receipt this turn restores.
    committed = finished.get("completion") if isinstance(finished, dict) else None
    if not isinstance(committed, dict):
        raise DurableToolJournalError("committed tool completion could not be reloaded")
    return _restore_completion(decision, controller, committed)


def resume_checkpoint_calls(
    checkpoint: dict[str, Any],
    *,
    backend: str,
    controller: Any,
    conversation: list[dict[str, Any]],
    execute_tool: Callable[..., Any],
    stream_tool_execution: Callable[..., Generator[dict, None, Any]],
    cancel_event: Any,
    tool_call_timeout: int,
    session_id: str | None,
    thread_id: str | None,
    rag_scope: dict[str, Any] | None,
    bypass_permissions: bool,
    permission_mode: str | None,
    confirm_tool_calls: bool,
    invocation_kwargs: Callable[[Any, Callable[[str], None], Any], dict[str, Any]] | None = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Resume the prepared call and siblings before the next model sample.

    Both local backends enter this helper from inside their ordinary tool-loop
    generator. It uses the same account/capability stream wrapper and execution
    callable, preserving deadlines, cancellation and bounded live output.
    """

    if int(checkpoint.get("version") or 0) != CHECKPOINT_VERSION:
        raise DurableToolJournalError("durable tool checkpoint version is incompatible")
    if str(checkpoint.get("backend") or "") != backend:
        raise DurableToolJournalError("durable tool checkpoint backend does not match")
    controller.restore_state(checkpoint.get("controller") or {})
    current = checkpoint.get("current_call")
    remaining = checkpoint.get("remaining_calls") or []
    if not isinstance(current, dict) or not isinstance(remaining, list):
        raise DurableToolJournalError("durable tool checkpoint call batch is invalid")
    first = current.get("tool_call")
    if not isinstance(first, dict):
        raise DurableToolJournalError("durable tool checkpoint current call is invalid")
    first = dict(first)
    first["card_id"] = str(current.get("card_call_id") or first.get("id") or "")
    calls = [first, *remaining]
    recovered_approval_id = str(checkpoint.get("recovery_approval_id") or "")
    recovered_execution_id = str(checkpoint.get("recovery_execution_id") or "")
    current_public_end_persisted = bool(
        checkpoint.get("current_public_end_persisted", False)
    )
    receipt = checkpoint.get("recovered_receipt")
    executed = 0
    denied_any = False

    from core.inference.tool_loop_controller import awaiting_approval_status
    from core.inference.tool_stream_exec import accepts_output_callback, search_images_kwargs
    from core.inference.tools import is_high_risk_tool_call
    from state.tool_approvals import (
        TOOL_REJECTED_MESSAGE,
        abort_tool_decision,
        begin_tool_decision,
        new_approval_id,
        wait_tool_decision,
    )

    for index, raw_call in enumerate(calls):
        if cancel_event is not None and cancel_event.is_set():
            break
        decision = controller.prepare_call(raw_call)
        if not decision.should_execute:
            raise DurableToolJournalError(
                "durable checkpoint no longer classifies the prepared call as executable"
            )
        if index:
            assistant = next(
                (item for item in reversed(conversation) if item.get("role") == "assistant"),
                None,
            )
            if assistant is None:
                assistant = {"role": "assistant", "content": "", "tool_calls": []}
                conversation.append(assistant)
            assistant.setdefault("tool_calls", []).append(decision.as_assistant_tool_call())

        checkpoint_base = {
            key: value
            for key, value in checkpoint.items()
            if key
            not in {
                "recovered_receipt",
                "recovery_approval_id",
                "recovery_execution_id",
                "current_public_end_persisted",
            }
        }
        if index:
            checkpoint_base.update(
                {
                    "conversation": conversation,
                    "current_call": {
                        "tool_call": decision.as_assistant_tool_call(),
                        "card_call_id": decision.card_id,
                        "provenance": decision.provenance,
                    },
                    "remaining_calls": calls[index + 1 :],
                    "controller": controller.export_state(),
                }
            )
        next_checkpoint, _next_checkpoint_digest = freeze_pre_tool_checkpoint(
            checkpoint_base
        )

        needs_confirmation = (
            bool(confirm_tool_calls) and not bypass_permissions and permission_mode != "off"
        )
        if needs_confirmation and permission_mode == "auto":
            needs_confirmation = is_high_risk_tool_call(
                decision.tool_name, decision.arguments
            )
        approval_id = recovered_approval_id if index == 0 else (
            new_approval_id() if needs_confirmation else ""
        )
        verdict: str | None = None

        if index == 0 and isinstance(receipt, dict):
            state = str(receipt.get("state") or "")
            if state == "denied":
                verdict = "deny"
            elif state != "finished":
                raise DurableToolJournalError("durable tool receipt state is invalid")
        elif index == 0 and approval_id:
            approval = runs_db.get_tool_approval(
                (_CURRENT.get() or ("", ""))[0], approval_id, include_checkpoint=True
            )
            if approval is None:
                raise DurableToolJournalError("recovered durable approval is missing")
            verdict = str(approval.get("decision") or approval.get("status") or "")
            if verdict == "allow":
                bind_next_tool_approval(str(approval["runId"]), approval_id)
            elif verdict not in {"deny"}:
                slot = begin_tool_decision(session_id, approval_id)
                verdict = wait_tool_decision(
                    slot, approval_id, cancel_event=cancel_event
                )
        elif index == 0 and recovered_execution_id:
            # The original public tool_start is already in the durable event
            # log. Claim/start the same execution identity without emitting a
            # duplicate public start.
            verdict = "allow"
        else:
            slot = begin_tool_decision(session_id, approval_id) if needs_confirmation else None
            start_event = decision.tool_start_event()
            start_event["approval_id"] = approval_id
            start_event["awaiting_confirmation"] = needs_confirmation
            if needs_confirmation:
                private = proposal_envelope(
                    approval_id=approval_id,
                    session_id=session_id,
                    tool_name=decision.tool_name,
                    tool_call_id=decision.tool_call_id,
                    card_call_id=decision.card_id,
                    arguments=decision.arguments,
                    resume_checkpoint=next_checkpoint,
                )
                if private is not None:
                    start_event["_durable_tool_approval"] = private
            yield {
                "type": "status",
                "text": (
                    awaiting_approval_status(decision.tool_name)
                    if needs_confirmation
                    else decision.status_text
                ),
            }
            yield start_event
            try:
                if slot is not None:
                    verdict = wait_tool_decision(
                        slot, approval_id, cancel_event=cancel_event
                    )
                    slot = None
            finally:
                if slot is not None:
                    abort_tool_decision(slot, approval_id)

        if verdict == "deny":
            denied_any = True
            yield {
                "type": "tool_end",
                "tool_name": decision.tool_name,
                "tool_call_id": decision.card_id,
                "result": TOOL_REJECTED_MESSAGE,
                "provenance": decision.provenance,
            }
            message: dict[str, Any] = {
                "role": "tool",
                "name": decision.tool_name,
                "content": TOOL_REJECTED_MESSAGE,
            }
            if decision.tool_call_id:
                message["tool_call_id"] = decision.tool_call_id
            conversation.append(message)
            continue

        committed_completion = None
        if index == 0 and isinstance(receipt, dict) and receipt.get("state") == "finished":
            if receipt.get("error"):
                result = f"Error: {receipt['error']}"
            else:
                result = receipt.get("result")
            if isinstance(receipt.get("completion"), dict):
                committed_completion = receipt["completion"]
        else:
            if verdict == "allow":
                yield {"type": "status", "text": decision.status_text}

            def invoke(output_callback, tool_cancel_event=None, _decision=decision):
                if invocation_kwargs is not None:
                    kwargs = invocation_kwargs(
                        _decision, output_callback, tool_cancel_event or cancel_event
                    )
                else:
                    kwargs = {
                        "cancel_event": tool_cancel_event or cancel_event,
                        "timeout": None if tool_call_timeout >= 9999 else tool_call_timeout,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "rag_scope": rag_scope,
                        "disable_sandbox": bypass_permissions,
                    }
                    if accepts_output_callback(execute_tool):
                        kwargs["output_callback"] = output_callback
                    kwargs.update(search_images_kwargs(execute_tool, _decision.tool_name))
                return execute_tool(_decision.tool_name, _decision.arguments, **kwargs)

            result = yield from stream_tool_execution(
                invoke,
                tool_name=decision.tool_name,
                tool_call_id=decision.card_id,
                journal_tool_call_id=decision.tool_call_id,
                arguments=decision.arguments,
                session_id=session_id,
                thread_id=thread_id,
                pre_tool_checkpoint=next_checkpoint,
                cancel_event=cancel_event,
                timeout_s=None if tool_call_timeout >= 9999 else tool_call_timeout,
            )
        completion = (
            _restore_completion(decision, controller, committed_completion)
            if isinstance(committed_completion, dict)
            else settle_controller_completion(result, decision, controller)
        )
        executed += 1
        if not (index == 0 and current_public_end_persisted):
            yield completion.tool_end_event()
        conversation.append(completion.tool_message())

    return {"executed": executed, "denied": denied_any}
