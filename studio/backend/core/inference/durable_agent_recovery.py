# SPDX-License-Identifier: AGPL-3.0-only
"""Crash/restart recovery planner for server-owned durable chat runs.

Recovery is intentionally receipt-driven. The original API request remains
immutable; this module derives a server-only continuation request from the
ordered run log. Completed tool effects are replayed into conversation history,
never executed again. A tool that has ``tool_execution.started`` without a
matching durable finish/denial is an ambiguous side effect and is not resumed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from state.tool_approvals import TOOL_REJECTED_MESSAGE
from storage import chat_generation_runs_db as runs_db


@dataclass(frozen=True)
class RecoveryPlan:
    run_id: str
    safe: bool
    reason: str
    request_payload: dict[str, Any] | None = None
    action: str = "fail_closed"
    approval_id: str | None = None
    resume_checkpoint: dict[str, Any] | None = None
    expected_worker_token: str | None = None
    expected_progress_at: int | None = None
    expected_last_event_seq: int | None = None


_RECOVERY_EVENT_PAGE_SIZE = 10_000
_RECOVERY_MAX_EVENT_COUNT = 110_000
_RECOVERY_MAX_SERIALIZED_BYTES = 16 * 1024 * 1024
# Optimization traces are private benchmark evidence, not operational replay
# input.  Keep a separate, deliberately small allowance so one bounded trace
# cannot consume the frozen public/tool ceilings, while a corrupt store full of
# private rows still fails closed instead of making startup scan without bound.
_RECOVERY_MAX_PRIVATE_EVENT_COUNT = 1_024
_RECOVERY_MAX_PRIVATE_SERIALIZED_BYTES = 8 * 1024 * 1024
_RECOVERY_EVENT_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    default=str,
)


def _load_exact_event_snapshot(
    run_id: str, expected_last_seq: int
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Read the complete contiguous event prefix through the fenced run cursor."""

    if expected_last_seq < 0:
        return None, "recovery_event_log_incomplete"
    if expected_last_seq > (
        _RECOVERY_MAX_EVENT_COUNT + _RECOVERY_MAX_PRIVATE_EVENT_COUNT
    ):
        return None, "recovery_event_count_limit_exceeded"
    events: list[dict[str, Any]] = []
    cursor = 0
    event_count = 0
    serialized_bytes = 0
    private_event_count = 0
    private_serialized_bytes = 0
    while cursor < expected_last_seq:
        page = runs_db.list_events(
            run_id,
            after=cursor,
            limit=min(_RECOVERY_EVENT_PAGE_SIZE, expected_last_seq - cursor),
        )
        if not page:
            return None, "recovery_event_log_incomplete"
        for event in page:
            try:
                sequence = int(event.get("seq"))
            except (TypeError, ValueError):
                return None, "recovery_event_log_incomplete"
            if sequence != cursor + 1 or sequence > expected_last_seq:
                return None, "recovery_event_log_incomplete"
            if event.get("type") == "optimization.trace":
                private_event_count += 1
                if private_event_count > _RECOVERY_MAX_PRIVATE_EVENT_COUNT:
                    return None, "recovery_private_event_count_limit_exceeded"
                for fragment in _RECOVERY_EVENT_ENCODER.iterencode(event):
                    private_serialized_bytes += len(fragment.encode("utf-8"))
                    if (
                        private_serialized_bytes
                        > _RECOVERY_MAX_PRIVATE_SERIALIZED_BYTES
                    ):
                        return None, "recovery_private_event_byte_limit_exceeded"
                cursor = sequence
                continue
            event_count += 1
            if event_count > _RECOVERY_MAX_EVENT_COUNT:
                return None, "recovery_event_count_limit_exceeded"
            for fragment in _RECOVERY_EVENT_ENCODER.iterencode(event):
                serialized_bytes += len(fragment.encode("utf-8"))
                if serialized_bytes > _RECOVERY_MAX_SERIALIZED_BYTES:
                    return None, "recovery_event_byte_limit_exceeded"
            events.append(event)
            cursor = sequence
    return events, None


def _validated_checkpoint(approval: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = approval.get("resumeCheckpoint")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("version") or 0) != 1:
        return None
    if str(checkpoint.get("backend") or "") not in {"gguf", "safetensors"}:
        return None
    arguments = approval.get("arguments")
    if not isinstance(arguments, dict):
        return None
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    import hashlib

    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if fingerprint != str(approval.get("argumentsFingerprint") or ""):
        return None
    current = checkpoint.get("current_call")
    if not isinstance(current, dict) or not isinstance(current.get("tool_call"), dict):
        return None
    call = current["tool_call"]
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    checkpoint_arguments = function.get("arguments")
    if isinstance(checkpoint_arguments, str):
        try:
            checkpoint_arguments = json.loads(checkpoint_arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(checkpoint_arguments, dict):
        return None
    checkpoint_encoded = json.dumps(
        checkpoint_arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    if (
        checkpoint_encoded != encoded
        or str(function.get("name") or "") != str(approval.get("toolName") or "")
        or str(call.get("id") or "") != str(approval.get("toolCallId") or "")
        or str(current.get("card_call_id") or call.get("id") or "")
        != str(approval.get("cardCallId") or approval.get("toolCallId") or "")
        or not isinstance(checkpoint.get("conversation"), list)
        or not isinstance(checkpoint.get("remaining_calls"), list)
        or not isinstance(checkpoint.get("controller"), dict)
    ):
        return None
    return checkpoint


def _validated_execution_checkpoint(
    execution: dict[str, Any],
) -> dict[str, Any] | None:
    checkpoint = execution.get("preToolCheckpoint")
    if not isinstance(checkpoint, dict):
        return None
    candidate = {
        "resumeCheckpoint": checkpoint,
        "arguments": execution.get("arguments"),
        "argumentsFingerprint": execution.get("argumentsFingerprint"),
        "toolName": execution.get("toolName"),
        "toolCallId": execution.get("toolCallId"),
        "cardCallId": execution.get("cardCallId"),
    }
    validated = _validated_checkpoint(candidate)
    if validated is None or int(execution.get("preToolCheckpointVersion") or 0) != 1:
        return None
    encoded = json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if digest != str(execution.get("preToolCheckpointDigest") or ""):
        return None
    return validated


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _public_tool_spans(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Correlate the one globally ordered public tool frontier.

    Authority-specific planners must not independently reinterpret the public
    stream: a later approved call supersedes an older ungated checkpoint (and
    vice versa).  A public close is exact and single-use.
    """

    spans: list[dict[str, Any]] = []
    for event in events:
        payload = _chunk(event)
        if payload is None:
            continue
        frame_type = str(payload.get("type") or "")
        if frame_type == "tool_start":
            spans.append(
                {
                    "start_seq": int(event.get("seq") or -1),
                    "tool_name": str(payload.get("tool_name") or ""),
                    "tool_call_id": str(payload.get("tool_call_id") or ""),
                    "approval_id": str(payload.get("approval_id") or ""),
                    "arguments": payload.get("arguments"),
                    "arguments_text": payload.get("arguments_text"),
                    "state": "open",
                }
            )
            continue
        if frame_type != "tool_end":
            continue
        tool_name = str(payload.get("tool_name") or "")
        tool_call_id = str(payload.get("tool_call_id") or "")
        matching = [
            span
            for span in spans
            if span["state"] == "open"
            and span["tool_name"] == tool_name
            and span["tool_call_id"] == tool_call_id
        ]
        if len(matching) != 1:
            if any(
                span["state"] == "open" and span["approval_id"] for span in spans
            ):
                return None, "tool_approval_public_end_mismatch"
            return None, "tool_public_frontier_is_ambiguous"
        span = matching[0]
        span["state"] = "closed"
        span["end_seq"] = int(event.get("seq") or -1)
        span["end_result"] = payload.get("result")
        if span["end_seq"] <= span["start_seq"]:
            return None, "tool_public_frontier_is_reordered"
    return spans, None


def _exact_event(
    events: list[dict[str, Any]], event_type: str, execution_id: str
) -> dict[str, Any] | None:
    matches = [
        event
        for event in events
        if event.get("type") == event_type
        and str((event.get("payload") or {}).get("execution_id") or "")
        == execution_id
    ]
    return matches[0] if len(matches) == 1 else None


def _terminal_events(
    events: list[dict[str, Any]], execution_id: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("type")
        in {"tool_execution.finished", "tool_execution.ambiguous"}
        and str((event.get("payload") or {}).get("execution_id") or "")
        == execution_id
    ]


def _exact_finished_terminal(
    events: list[dict[str, Any]], execution_id: str
) -> dict[str, Any] | None:
    matches = _terminal_events(events, execution_id)
    if len(matches) != 1 or matches[0].get("type") != "tool_execution.finished":
        return None
    return matches[0]


def _approval_receipt_shape(approval: dict[str, Any]) -> str:
    """Classify persisted approval completion columns without downgrade ambiguity."""

    core = (
        approval.get("receiptRef"),
        approval.get("receiptDigest"),
        approval.get("terminalSeq"),
        approval.get("completion"),
    )
    supplemental = (
        approval.get("producerReceipt"),
        approval.get("controllerIsError"),
        approval.get("completionAnnotations"),
        approval.get("postControllerCheckpoint"),
    )
    if all(value is None for value in (*core, *supplemental)):
        return "absent"
    if all(value is not None for value in core):
        return "modern"
    return "corrupt"


def _approval_row_event_frontier_coherent(
    approval: dict[str, Any], events: list[dict[str, Any]]
) -> bool:
    """Require the persisted approval state to agree with direct effect events."""

    execution_id = str(approval.get("executionId") or "")
    claimed = [
        event
        for event in events
        if event.get("type") == "tool_execution.claimed"
        and str((event.get("payload") or {}).get("execution_id") or "")
        == execution_id
    ]
    started = [
        event
        for event in events
        if event.get("type") == "tool_execution.started"
        and str((event.get("payload") or {}).get("execution_id") or "")
        == execution_id
    ]
    terminals = _terminal_events(events, execution_id)
    state = str(approval.get("executionState") or "")
    if state == "unclaimed":
        return not claimed and not started and not terminals
    if state == "claimed":
        return len(claimed) == 1 and not started and not terminals
    if state == "started":
        return len(claimed) == 1 and len(started) == 1 and not terminals
    if state == "finished":
        return (
            len(claimed) == 1
            and len(started) == 1
            and len(terminals) == 1
            and terminals[0].get("type") == "tool_execution.finished"
        )
    if state == "ambiguous":
        return (
            len(claimed) == 1
            and len(started) == 1
            and len(terminals) == 1
            and terminals[0].get("type") == "tool_execution.ambiguous"
        )
    if state == "cancelled":
        return len(claimed) <= 1 and not started and not terminals
    return False


def _validated_legacy_approval_terminal(
    approval: dict[str, Any],
    span: dict[str, Any],
    events: list[dict[str, Any]],
) -> bool:
    """Recognize a pre-receipt approval without accepting erased modern data."""

    execution_id = str(approval.get("executionId") or "")
    terminal = _exact_finished_terminal(events, execution_id)
    claimed = _exact_event(events, "tool_execution.claimed", execution_id)
    started = _exact_event(events, "tool_execution.started", execution_id)
    if terminal is None or claimed is None or started is None:
        return False
    payload = terminal.get("payload")
    if (
        not isinstance(payload, dict)
        or "receipt_ref" in payload
        or "receipt_digest" in payload
        or payload.get("effect_state") != "finished"
        or str(payload.get("tool_name") or "")
        != str(approval.get("toolName") or "")
        or str(payload.get("tool_call_id") or "")
        != str(approval.get("toolCallId") or "")
    ):
        return False
    if approval.get("error") is not None:
        if payload.get("error") != approval.get("error"):
            return False
    elif payload.get("result") != approval.get("result"):
        return False
    ordered = (
        int(span["start_seq"])
        < int(claimed.get("seq") or -1)
        < int(started.get("seq") or -1)
        < int(terminal.get("seq") or -1)
    )
    if span.get("state") == "closed":
        ordered = ordered and int(terminal.get("seq") or -1) < int(
            span.get("end_seq") or -1
        )
    return ordered


def _has_later_semantic_progress(
    events: list[dict[str, Any]], end_seq: int
) -> bool:
    """Return whether replaying a closed checkpoint would roll back newer work."""

    return any(
        int(event.get("seq") or -1) > end_seq
        and event.get("type") != "optimization.trace"
        for event in events
    )


def _validated_ungated_causality(
    execution: dict[str, Any],
    checkpoint: dict[str, Any],
    span: dict[str, Any],
    events: list[dict[str, Any]],
) -> bool:
    """Bind the public proposal to the exact durable execution event chain."""

    arguments = execution.get("arguments")
    if not isinstance(arguments, dict) or span.get("arguments") != arguments:
        return False
    arguments_text = span.get("arguments_text")
    if arguments_text is not None:
        try:
            if not isinstance(arguments_text, str) or json.loads(arguments_text) != arguments:
                return False
        except (TypeError, ValueError):
            return False
    execution_id = str(execution.get("executionId") or "")
    claimed = _exact_event(events, "tool_execution.claimed", execution_id)
    if claimed is None:
        return False
    claimed_payload = claimed.get("payload")
    if claimed_payload != {
        "schema_version": "helix.tool-execution.v3",
        "execution_id": execution_id,
        "authority_kind": "ungated",
        "arguments_fingerprint": str(execution.get("argumentsFingerprint") or ""),
        "pre_tool_checkpoint_digest": str(
            execution.get("preToolCheckpointDigest") or ""
        ),
    }:
        return False
    state = str(execution.get("executionState") or "")
    started = _exact_event(events, "tool_execution.started", execution_id)
    terminal = _exact_finished_terminal(events, execution_id)
    if state == "claimed":
        return (
            started is None
            and not _terminal_events(events, execution_id)
            and int(span["start_seq"]) < int(claimed.get("seq") or -1)
        )
    if state != "finished" or started is None or terminal is None:
        return False
    started_payload = started.get("payload")
    if started_payload != {
        "schema_version": "helix.tool-execution.v3",
        "execution_id": execution_id,
        "tool_name": str(execution.get("toolName") or ""),
        "tool_call_id": str(
            execution.get("cardCallId") or execution.get("toolCallId") or ""
        ),
        "effect_state": "started",
        "authority_kind": "ungated",
    }:
        return False
    ordered = (
        int(span["start_seq"])
        < int(claimed.get("seq") or -1)
        < int(started.get("seq") or -1)
        < int(terminal.get("seq") or -1)
    )
    if span["state"] == "closed":
        ordered = ordered and int(terminal.get("seq") or -1) < int(
            span.get("end_seq") or -1
        )
    return ordered


def _validated_finished_execution(
    execution: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Verify one finished row against its completion digest and terminal event."""

    completion = execution.get("completion")
    execution_id = str(execution.get("executionId") or "")
    receipt_ref = str(execution.get("receiptRef") or "")
    receipt_digest = str(execution.get("receiptDigest") or "")
    checkpoint_digest = str(execution.get("preToolCheckpointDigest") or "")
    if (
        not execution_id
        or not isinstance(completion, dict)
        or receipt_ref != f"tool-receipt:{execution_id}"
        or len(receipt_digest) != 64
        or len(checkpoint_digest) != 64
    ):
        return None
    encoded = _canonical_json(completion)
    if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != receipt_digest:
        return None
    binding = completion.get("binding")
    expected_binding = {
        "backend_account_id": str(execution.get("backendAccountId") or ""),
        "owner_subject": str(execution.get("ownerSubject") or ""),
        "run_id": str(execution.get("runId") or ""),
        "session_id": str(execution.get("sessionId") or ""),
        "thread_id": str(execution.get("threadId") or ""),
        "execution_id": execution_id,
        "tool_name": str(execution.get("toolName") or ""),
        "tool_call_id": str(execution.get("toolCallId") or ""),
        "card_call_id": str(execution.get("cardCallId") or ""),
        "arguments_fingerprint": str(execution.get("argumentsFingerprint") or ""),
        "pre_tool_checkpoint_digest": checkpoint_digest,
        "authority_kind": "ungated",
        "approval_id": None,
    }
    if (
        completion.get("schema_version") != "helix.tool-completion.v1"
        or str(completion.get("execution_id") or "") != execution_id
        or completion.get("authority_kind") != "ungated"
        or completion.get("terminal_state") != "finished"
        or completion.get("receipt_ref") != receipt_ref
        or binding != expected_binding
        or completion.get("selected_result") != execution.get("result")
        or completion.get("error") != execution.get("error")
        or completion.get("producer_receipt") != execution.get("producerReceipt")
        or completion.get("controller_is_error")
        != execution.get("controllerIsError")
        or completion.get("completion_annotations")
        != execution.get("completionAnnotations")
        or completion.get("post_controller_checkpoint")
        != execution.get("postControllerCheckpoint")
    ):
        return None
    terminal_seq = execution.get("terminalSeq")
    terminal = _exact_finished_terminal(events, execution_id)
    if terminal is None or terminal_seq is None:
        return None
    payload = terminal.get("payload")
    expected_payload: dict[str, Any] = {
        "schema_version": "helix.tool-execution.v3",
        "execution_id": execution_id,
        "tool_name": str(execution.get("toolName") or ""),
        "tool_call_id": str(
            execution.get("cardCallId") or execution.get("toolCallId") or ""
        ),
        "effect_state": "finished",
        "authority_kind": "ungated",
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
    }
    if execution.get("error") is not None:
        expected_payload["error"] = execution.get("error")
    else:
        expected_payload["result"] = execution.get("result")
    if int(terminal.get("seq") or -1) != int(terminal_seq) or payload != expected_payload:
        return None
    return completion


def _validated_finished_approval(
    approval: dict[str, Any],
    span: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate a modern approved completion before replaying its span."""

    execution_id = str(approval.get("executionId") or "")
    approval_id = str(approval.get("approvalId") or "")
    completion = approval.get("completion")
    receipt_ref = str(approval.get("receiptRef") or "")
    receipt_digest = str(approval.get("receiptDigest") or "")
    arguments = approval.get("arguments")
    if (
        not execution_id
        or not approval_id
        or str(approval.get("executionState") or "") != "finished"
        or not isinstance(completion, dict)
        or not isinstance(approval.get("postControllerCheckpoint"), dict)
        or receipt_ref != f"tool-receipt:{execution_id}"
        or len(receipt_digest) != 64
        or not isinstance(arguments, dict)
        or span.get("arguments") != arguments
        or (
            span.get("state") == "closed"
            and span.get("end_result") != approval.get("result")
        )
    ):
        return None
    if (
        hashlib.sha256(_canonical_json(completion).encode("utf-8")).hexdigest()
        != receipt_digest
    ):
        return None
    expected_binding = {
        "backend_account_id": str(approval.get("backendAccountId") or ""),
        "owner_subject": str(approval.get("ownerSubject") or ""),
        "run_id": str(approval.get("runId") or ""),
        "session_id": str(approval.get("sessionId") or ""),
        "thread_id": str(approval.get("threadId") or ""),
        "execution_id": execution_id,
        "tool_name": str(approval.get("toolName") or ""),
        "tool_call_id": str(approval.get("toolCallId") or ""),
        "card_call_id": str(approval.get("cardCallId") or ""),
        "arguments_fingerprint": str(approval.get("argumentsFingerprint") or ""),
        "authority_kind": "approved",
        "approval_id": approval_id,
    }
    if (
        completion.get("schema_version") != "helix.tool-completion.v1"
        or str(completion.get("execution_id") or "") != execution_id
        or completion.get("terminal_state") != "finished"
        or completion.get("authority_kind") != "approved"
        or completion.get("receipt_ref") != receipt_ref
        or completion.get("binding") != expected_binding
        or completion.get("selected_result") != approval.get("result")
        or completion.get("error") != approval.get("error")
        or completion.get("producer_receipt") != approval.get("producerReceipt")
        or completion.get("controller_is_error")
        != approval.get("controllerIsError")
        or completion.get("completion_annotations")
        != approval.get("completionAnnotations")
        or completion.get("post_controller_checkpoint")
        != approval.get("postControllerCheckpoint")
    ):
        return None
    proposed = _exact_event(events, "approval.proposed", execution_id)
    decided = _exact_event(events, "approval.decided", execution_id)
    claimed = _exact_event(events, "tool_execution.claimed", execution_id)
    started = _exact_event(events, "tool_execution.started", execution_id)
    terminal = _exact_finished_terminal(events, execution_id)
    if any(
        item is None for item in (proposed, decided, claimed, started, terminal)
    ):
        return None
    if proposed.get("payload") != {
        "schema_version": "helix.tool-approval.v2",
        "approval_id": approval_id,
        "tool_name": str(approval.get("toolName") or ""),
        "tool_call_id": str(approval.get("toolCallId") or ""),
        "card_call_id": str(approval.get("cardCallId") or ""),
        "execution_id": execution_id,
        "arguments_fingerprint": str(approval.get("argumentsFingerprint") or ""),
        "expires_at": int(approval.get("expiresAt") or 0),
        "status": "pending",
    }:
        return None
    if decided.get("payload") != {
        "schema_version": "helix.tool-approval.v2",
        "approval_id": approval_id,
        "tool_name": str(approval.get("toolName") or ""),
        "tool_call_id": str(approval.get("toolCallId") or ""),
        "execution_id": execution_id,
        "decision": "allow",
        "source": str(approval.get("decisionSource") or ""),
    }:
        return None
    if claimed.get("payload") != {
        "schema_version": "helix.tool-execution.v2",
        "approval_id": approval_id,
        "execution_id": execution_id,
    }:
        return None
    if started.get("payload") != {
        "schema_version": "helix.tool-execution.v2",
        "approval_id": approval_id,
        "execution_id": execution_id,
        "tool_name": str(approval.get("toolName") or ""),
        "tool_call_id": str(approval.get("toolCallId") or ""),
        "effect_state": "started",
    }:
        return None
    expected_terminal: dict[str, Any] = {
        "schema_version": "helix.tool-execution.v2",
        "approval_id": approval_id,
        "execution_id": execution_id,
        "tool_name": str(approval.get("toolName") or ""),
        "tool_call_id": str(approval.get("toolCallId") or ""),
        "effect_state": "finished",
        "receipt_ref": receipt_ref,
        "receipt_digest": receipt_digest,
    }
    if approval.get("error") is not None:
        expected_terminal["error"] = approval.get("error")
    else:
        expected_terminal["result"] = approval.get("result")
    if terminal.get("payload") != expected_terminal:
        return None
    if int(terminal.get("seq") or -1) != int(approval.get("terminalSeq") or -1):
        return None
    ordered = (
        int(span["start_seq"])
        < int(proposed.get("seq") or -1)
        < int(decided.get("seq") or -1)
        < int(claimed.get("seq") or -1)
        < int(started.get("seq") or -1)
        < int(terminal.get("seq") or -1)
    )
    if span.get("state") == "closed":
        ordered = ordered and int(terminal.get("seq") or -1) < int(
            span.get("end_seq") or -1
        )
    return completion if ordered else None


def _ungated_recovery_plan(
    run_id: str,
    request_payload: dict[str, Any],
    executions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    public_spans: list[dict[str, Any]],
) -> RecoveryPlan | None:
    """Recover exactly one ungated public span from its authoritative row."""

    ungated_spans = [span for span in public_spans if not span["approval_id"]]

    validated_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for execution in executions:
        matching_spans = [
            span
            for span in ungated_spans
            if span["tool_name"] == str(execution.get("toolName") or "")
            and span["tool_call_id"]
            == str(execution.get("cardCallId") or execution.get("toolCallId") or "")
        ]
        if len(matching_spans) != 1:
            return RecoveryPlan(
                run_id, False, "ungated_row_public_span_identity_mismatch"
            )
        checkpoint = _validated_execution_checkpoint(execution)
        if checkpoint is None:
            return RecoveryPlan(
                run_id, False, "ungated_tool_checkpoint_missing_or_corrupt"
            )
        state = str(execution.get("executionState") or "")
        span = matching_spans[0]
        if not _validated_ungated_causality(execution, checkpoint, span, events):
            return RecoveryPlan(run_id, False, "ungated_tool_event_causality_mismatch")
        if span["state"] == "closed" and state != "finished":
            return RecoveryPlan(
                run_id, False, "ungated_public_end_without_finished_row"
            )
        if state == "finished":
            completion = _validated_finished_execution(execution, events)
            if not isinstance(completion, dict):
                return RecoveryPlan(
                    run_id, False, "finished_ungated_tool_completion_missing_or_corrupt"
                )
            if span["state"] == "closed" and span.get("end_result") != execution.get(
                "result"
            ):
                return RecoveryPlan(
                    run_id, False, "ungated_public_end_result_mismatch"
                )
        validated_rows.append((execution, checkpoint))

    open_frontier = [span for span in ungated_spans if span["state"] == "open"]
    if not open_frontier:
        latest = public_spans[-1] if public_spans else None
        if (
            latest is not None
            and not latest["approval_id"]
            and latest["state"] == "closed"
        ):
            if _has_later_semantic_progress(events, int(latest["end_seq"])):
                return RecoveryPlan(
                    run_id, False, "progress_after_published_tool_completion"
                )
            candidates = [
                item
                for item in validated_rows
                if item[0].get("executionState") == "finished"
                and item[0].get("toolName") == latest["tool_name"]
                and str(item[0].get("cardCallId") or item[0].get("toolCallId") or "")
                == latest["tool_call_id"]
            ]
            if len(candidates) != 1:
                return RecoveryPlan(
                    run_id, False, "ungated_tool_suffix_frontier_is_ambiguous"
                )
            execution, checkpoint = candidates[0]
            completion = execution.get("completion")
            if (
                not isinstance(completion, dict)
                or not isinstance(execution.get("postControllerCheckpoint"), dict)
            ):
                return RecoveryPlan(
                    run_id,
                    False,
                    "finished_ungated_tool_completion_missing_or_corrupt",
                )
            resumed = copy.deepcopy(checkpoint)
            resumed["recovery_execution_id"] = str(
                execution.get("executionId") or ""
            )
            resumed["current_public_end_persisted"] = True
            resumed["recovered_receipt"] = {
                "state": "finished",
                "result": execution.get("result"),
                "error": execution.get("error"),
                "completion": completion,
            }
            return RecoveryPlan(
                run_id,
                True,
                "resume_after_published_ungated_completion",
                copy.deepcopy(request_payload),
                "resume_tool",
                None,
                resumed,
            )
        return None
    if len(open_frontier) != 1:
        return RecoveryPlan(run_id, False, "ungated_tool_frontier_is_ambiguous")
    span = open_frontier[0]
    if not public_spans or span is not public_spans[-1]:
        return RecoveryPlan(run_id, False, "global_tool_frontier_is_ambiguous")
    matching_rows = [
        row
        for row in executions
        if str(row.get("authorityKind") or "") == "ungated"
        and str(row.get("toolName") or "") == span["tool_name"]
        and str(row.get("cardCallId") or row.get("toolCallId") or "")
        == span["tool_call_id"]
    ]
    if not matching_rows:
        return RecoveryPlan(run_id, False, "ungated_public_start_has_no_durable_row")
    if len(matching_rows) != 1:
        return RecoveryPlan(run_id, False, "ungated_tool_row_is_ambiguous")
    execution = matching_rows[0]
    checkpoint = _validated_execution_checkpoint(execution)
    if checkpoint is None:
        return RecoveryPlan(run_id, False, "ungated_tool_checkpoint_missing_or_corrupt")
    state = str(execution.get("executionState") or "")
    if state in {"started", "ambiguous"}:
        return RecoveryPlan(run_id, False, "tool_effect_started_without_durable_finish")
    if state == "cancelled":
        return RecoveryPlan(run_id, False, "ungated_tool_cancelled_while_public_span_open")
    if state not in {"claimed", "finished"}:
        return RecoveryPlan(run_id, False, "ungated_tool_state_is_not_recoverable")
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["recovery_execution_id"] = str(execution.get("executionId") or "")
    reason = "resume_claimed_ungated_tool"
    if state == "finished":
        completion = _validated_finished_execution(execution, events)
        if (
            not isinstance(completion, dict)
            or not isinstance(execution.get("postControllerCheckpoint"), dict)
        ):
            return RecoveryPlan(
                run_id, False, "finished_ungated_tool_completion_missing_or_corrupt"
            )
        checkpoint["recovered_receipt"] = {
            "state": "finished",
            "result": execution.get("result"),
            "error": execution.get("error"),
            "completion": completion,
        }
        reason = "resume_finished_ungated_tool_receipt"
    return RecoveryPlan(
        run_id,
        True,
        reason,
        copy.deepcopy(request_payload),
        "resume_tool",
        None,
        checkpoint,
    )


def _approval_recovery_plan(
    run_id: str,
    request_payload: dict[str, Any],
    approvals: list[dict[str, Any]],
    events: list[dict[str, Any]],
    global_spans: list[dict[str, Any]],
) -> RecoveryPlan | None:
    if not approvals:
        return None

    by_id = {
        str(item.get("approvalId") or ""): item
        for item in approvals
        if str(item.get("approvalId") or "")
    }
    if len(by_id) != len(approvals):
        return RecoveryPlan(run_id, False, "tool_approval_public_span_is_ambiguous")
    span_state = {approval_id: "unseen" for approval_id in by_id}
    public_spans: list[dict[str, str]] = []

    def matches_approval(approval: dict[str, Any], payload: dict[str, Any]) -> bool:
        public_call_id = str(payload.get("tool_call_id") or "")
        stable_call_id = str(
            approval.get("cardCallId") or approval.get("toolCallId") or ""
        )
        arguments = payload.get("arguments")
        expected_arguments = approval.get("arguments")
        return bool(
            public_call_id
            and stable_call_id == public_call_id
            and str(approval.get("toolName") or "")
            == str(payload.get("tool_name") or "")
            and isinstance(arguments, dict)
            and arguments == expected_arguments
        )

    for event in events:
        payload = _chunk(event)
        if payload is None:
            continue
        frame_type = str(payload.get("type") or "")
        if frame_type == "tool_start":
            approval_id = str(payload.get("approval_id") or "")
            if approval_id:
                if approval_id not in by_id or span_state[approval_id] != "unseen":
                    return RecoveryPlan(
                        run_id, False, "tool_approval_public_span_is_ambiguous"
                    )
                approval = by_id[approval_id]
                if not matches_approval(approval, payload):
                    return RecoveryPlan(
                        run_id, False, "tool_approval_public_span_identity_mismatch"
                    )
                span_state[approval_id] = "open"
            public_spans.append(
                {
                    "tool_name": str(payload.get("tool_name") or ""),
                    "tool_call_id": str(payload.get("tool_call_id") or ""),
                    "approval_id": approval_id,
                    "state": "open",
                }
            )
            continue
        if frame_type != "tool_end":
            continue
        matching = [
            span
            for span in public_spans
            if span["state"] == "open"
            and span["tool_name"] == str(payload.get("tool_name") or "")
            and span["tool_call_id"] == str(payload.get("tool_call_id") or "")
        ]
        if len(matching) == 1:
            span = matching[0]
            span["state"] = "closed"
            if span["approval_id"]:
                span_state[span["approval_id"]] = "closed"
            continue
        if len(matching) > 1:
            return RecoveryPlan(run_id, False, "tool_approval_public_span_is_ambiguous")
        open_approval = any(state == "open" for state in span_state.values())
        if open_approval:
            return RecoveryPlan(run_id, False, "tool_approval_public_end_mismatch")
        # A second matching end for an already closed durable span is corrupt;
        # an unrelated ungated tool_end is outside this correlation.
        if any(
            span["state"] == "closed"
            and span["approval_id"]
            and span["tool_name"] == str(payload.get("tool_name") or "")
            and span["tool_call_id"] == str(payload.get("tool_call_id") or "")
            for span in public_spans
        ):
            return RecoveryPlan(run_id, False, "tool_approval_public_span_is_ambiguous")

    if any(state == "unseen" for state in span_state.values()):
        return RecoveryPlan(run_id, False, "tool_approval_public_start_missing")
    open_ids = [approval_id for approval_id, state in span_state.items() if state == "open"]
    if not open_ids:
        latest = global_spans[-1] if global_spans else None
        if latest is None or not latest["approval_id"] or latest["state"] != "closed":
            return None
        if _has_later_semantic_progress(events, int(latest["end_seq"])):
            return RecoveryPlan(
                run_id, False, "progress_after_published_tool_completion"
            )
        approval = by_id.get(str(latest["approval_id"]))
        if approval is None:
            return RecoveryPlan(
                run_id, False, "tool_approval_public_span_identity_mismatch"
            )
        checkpoint = _validated_checkpoint(approval)
        state = str(approval.get("executionState") or "")
        decision = str(approval.get("decision") or approval.get("status") or "")
        receipt_shape = _approval_receipt_shape(approval)
        if not _approval_row_event_frontier_coherent(approval, events):
            return RecoveryPlan(
                run_id, False, "approved_row_event_frontier_is_inconsistent"
            )
        if receipt_shape != "absent" and state != "finished":
            return RecoveryPlan(
                run_id, False, "approved_receipt_state_is_inconsistent"
            )
        if state == "finished" and receipt_shape == "corrupt":
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        if (
            state == "finished"
            and receipt_shape == "absent"
            and not _validated_legacy_approval_terminal(approval, latest, events)
        ):
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        completion = (
            _validated_finished_approval(approval, latest, events)
            if state == "finished" and receipt_shape == "modern"
            else None
        )
        if state == "finished" and receipt_shape == "modern" and completion is None:
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        if state == "finished" and receipt_shape == "modern" and checkpoint is None:
            return RecoveryPlan(
                run_id, False, "tool_checkpoint_missing_or_incompatible"
            )
        if checkpoint is not None and completion is not None:
            resumed = copy.deepcopy(checkpoint)
            resumed["recovery_approval_id"] = str(approval["approvalId"])
            resumed["current_public_end_persisted"] = True
            resumed["recovered_receipt"] = {
                "state": "finished",
                "result": approval.get("result"),
                "error": approval.get("error"),
                "completion": completion,
            }
            return RecoveryPlan(
                run_id,
                True,
                "resume_after_published_approved_completion",
                copy.deepcopy(request_payload),
                "resume_tool",
                str(approval["approvalId"]),
                resumed,
            )
        if state != "finished" and decision != "deny":
            return RecoveryPlan(
                run_id, False, "approved_public_end_without_finished_row"
            )
        if checkpoint is not None and checkpoint.get("remaining_calls"):
            return RecoveryPlan(
                run_id, False, "published_approval_suffix_is_not_replayable"
            )
        # Legacy approvals lack the committed controller state needed for exact
        # replay.  Empty-suffix rows retain the compatible event-log path.
        return None
    if len(open_ids) != 1:
        return RecoveryPlan(run_id, False, "tool_approval_public_span_is_ambiguous")
    if any(
        span["state"] == "open" and span["approval_id"] != open_ids[0]
        for span in public_spans
    ):
        return RecoveryPlan(run_id, False, "tool_approval_public_span_is_ambiguous")
    approval = by_id[open_ids[0]]
    approval_span = next(
        span for span in global_spans if span.get("approval_id") == open_ids[0]
    )
    if not global_spans or approval_span is not global_spans[-1]:
        return RecoveryPlan(run_id, False, "global_tool_frontier_is_ambiguous")
    if str(approval.get("executionState") or "") == "cancelled":
        return RecoveryPlan(run_id, False, "tool_approval_cancelled_while_public_span_open")
    checkpoint = _validated_checkpoint(approval)
    if checkpoint is None:
        return RecoveryPlan(run_id, False, "tool_checkpoint_missing_or_incompatible")
    state = str(approval.get("executionState") or "")
    decision = str(approval.get("decision") or approval.get("status") or "pending")
    receipt_shape = _approval_receipt_shape(approval)
    if not _approval_row_event_frontier_coherent(approval, events):
        return RecoveryPlan(
            run_id, False, "approved_row_event_frontier_is_inconsistent"
        )
    if receipt_shape != "absent" and state != "finished":
        return RecoveryPlan(run_id, False, "approved_receipt_state_is_inconsistent")
    if state in {"started", "ambiguous"}:
        return RecoveryPlan(run_id, False, "tool_effect_started_without_durable_finish")
    if state == "finished":
        # The local loop replays the durable result into its conversation and
        # never invokes the effect again, even if tool_end was not appended.
        if receipt_shape == "corrupt":
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        if receipt_shape == "absent" and not _validated_legacy_approval_terminal(
            approval, approval_span, events
        ):
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        completion = (
            _validated_finished_approval(approval, approval_span, events)
            if receipt_shape == "modern"
            else None
        )
        if receipt_shape == "modern" and completion is None:
            return RecoveryPlan(
                run_id, False, "finished_approved_tool_completion_missing_or_corrupt"
            )
        checkpoint = copy.deepcopy(checkpoint)
        recovered_receipt = {
            "state": "finished",
            "result": approval.get("result"),
            "error": approval.get("error"),
        }
        if completion is not None:
            recovered_receipt["completion"] = completion
        checkpoint["recovered_receipt"] = recovered_receipt
        return RecoveryPlan(
            run_id,
            True,
            "resume_finished_tool_receipt",
            copy.deepcopy(request_payload),
            "resume_tool",
            str(approval.get("approvalId") or ""),
            checkpoint,
        )
    if decision == "deny":
        checkpoint = copy.deepcopy(checkpoint)
        checkpoint["recovered_receipt"] = {"state": "denied"}
        return RecoveryPlan(
            run_id,
            True,
            "resume_denied_tool",
            copy.deepcopy(request_payload),
            "resume_tool",
            str(approval.get("approvalId") or ""),
            checkpoint,
        )
    if decision == "allow" and state in {"unclaimed", "claimed"}:
        return RecoveryPlan(
            run_id,
            True,
            "resume_allowed_tool",
            copy.deepcopy(request_payload),
            "resume_tool",
            str(approval.get("approvalId") or ""),
            copy.deepcopy(checkpoint),
        )
    if decision == "pending" and state == "unclaimed":
        return RecoveryPlan(
            run_id,
            True,
            "wait_for_durable_tool_approval",
            copy.deepcopy(request_payload),
            "wait_approval",
            str(approval.get("approvalId") or ""),
            copy.deepcopy(checkpoint),
        )
    return RecoveryPlan(run_id, False, "tool_approval_state_is_not_recoverable")


def _chunk(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "chunk":
        return None
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else None


def _delta_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {None, "text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _event_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _matching_open_call(
    open_calls: list[dict[str, Any]],
    *,
    tool_call_id: str,
    tool_name: str,
    execution_id: str = "",
) -> dict[str, Any] | None:
    if execution_id:
        for call in open_calls:
            if call.get("execution_id") == execution_id:
                return call
    if tool_call_id:
        for call in open_calls:
            if call.get("tool_call_id") == tool_call_id:
                return call
    same_name = [
        call
        for call in open_calls
        if str(call.get("tool_name") or "") == tool_name and not call.get("execution_id")
    ]
    return same_name[0] if len(same_name) == 1 else None


def _assistant_tool_message(call: dict[str, Any]) -> dict[str, Any]:
    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": str(call.get("tool_name") or "unknown"),
            "arguments": str(call.get("arguments_text") or "{}"),
        },
    }
    tool_call_id = str(call.get("tool_call_id") or "")
    if tool_call_id:
        tool_call["id"] = tool_call_id
    message: dict[str, Any] = {
        "role": "assistant",
        "content": str(call.get("assistant_prefix") or ""),
        "tool_calls": [tool_call],
    }
    return message


def _tool_result_message(call: dict[str, Any], result: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "name": str(call.get("tool_name") or "unknown"),
        "content": _event_text(result),
    }
    call_id = str(call.get("tool_call_id") or "")
    if call_id:
        message["tool_call_id"] = call_id
    return message


def build_restart_request(
    request_payload: dict[str, Any],
    events: list[dict[str, Any]],
) -> RecoveryPlan:
    run_id = str(request_payload.get("generation_run_id") or request_payload.get("cancel_id") or "")
    if not run_id:
        return RecoveryPlan("", False, "missing_durable_run_identity")
    if str(request_payload.get("finalization_idempotency_key") or "") != run_id:
        return RecoveryPlan(run_id, False, "not_a_v3_durable_turn")

    base_messages = request_payload.get("messages")
    if not isinstance(base_messages, list):
        return RecoveryPlan(run_id, False, "original_messages_unavailable")

    messages = copy.deepcopy(base_messages)
    assistant_text = ""
    open_calls: list[dict[str, Any]] = []
    saw_generated_output = False
    saw_tool_activity = False
    unsafe_reason = ""

    def finalize_call(call: dict[str, Any], result: Any) -> None:
        nonlocal assistant_text, saw_generated_output
        messages.append(_assistant_tool_message(call))
        messages.append(_tool_result_message(call, result))
        assistant_text = ""
        saw_generated_output = True
        if call in open_calls:
            open_calls.remove(call)

    for event in events:
        direct_type = str(event.get("type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if direct_type == "tool_execution.started":
            saw_tool_activity = True
            call = _matching_open_call(
                open_calls,
                tool_call_id=str(payload.get("tool_call_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
            )
            if call is None:
                unsafe_reason = "tool_execution_started_without_proposal"
                break
            call["execution_id"] = str(payload.get("execution_id") or "")
            call["execution_started"] = True
            continue

        if direct_type in {"tool_execution.finished", "tool_execution.ambiguous"}:
            saw_tool_activity = True
            call = _matching_open_call(
                open_calls,
                tool_call_id=str(payload.get("tool_call_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
                execution_id=str(payload.get("execution_id") or ""),
            )
            if call is None:
                unsafe_reason = "tool_execution_receipt_without_proposal"
                break
            if direct_type == "tool_execution.ambiguous" or payload.get("effect_state") == "ambiguous":
                unsafe_reason = "tool_effect_outcome_ambiguous"
                break
            call["execution_finished"] = True
            if "error" in payload:
                call["execution_result"] = f"Error: {str(payload.get('error') or '')}"
            else:
                call["execution_result"] = payload.get("result")
            continue

        if direct_type == "approval.decided":
            call = _matching_open_call(
                open_calls,
                tool_call_id=str(payload.get("tool_call_id") or ""),
                tool_name=str(payload.get("tool_name") or ""),
            )
            if call is not None:
                call["approval_decision"] = str(payload.get("decision") or "deny")
            continue

        chunk = _chunk(event)
        if chunk is None:
            continue
        text = _delta_text(chunk)
        if text:
            assistant_text += text
            saw_generated_output = True

        frame_type = str(chunk.get("type") or "")
        if frame_type == "tool_start":
            saw_tool_activity = True
            arguments_text = chunk.get("arguments_text")
            if not isinstance(arguments_text, str):
                arguments_text = json.dumps(
                    chunk.get("arguments") if isinstance(chunk.get("arguments"), dict) else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            open_calls.append(
                {
                    "tool_name": str(chunk.get("tool_name") or "unknown"),
                    "tool_call_id": str(chunk.get("tool_call_id") or ""),
                    "arguments_text": arguments_text,
                    "assistant_prefix": assistant_text,
                    "approval_id": str(chunk.get("approval_id") or ""),
                    "awaiting_confirmation": chunk.get("awaiting_confirmation") is True,
                }
            )
            assistant_text = ""
            continue

        if frame_type == "tool_end":
            saw_tool_activity = True
            call = _matching_open_call(
                open_calls,
                tool_call_id=str(chunk.get("tool_call_id") or ""),
                tool_name=str(chunk.get("tool_name") or ""),
            )
            if call is None:
                unsafe_reason = "tool_end_without_durable_proposal"
                break
            result = chunk.get("result")
            # A real execution on v3 must have a direct finished receipt. A
            # denial is the only legitimate tool_end without execution.
            if result != TOOL_REJECTED_MESSAGE and not call.get("execution_finished"):
                unsafe_reason = "tool_end_missing_execution_receipt"
                break
            finalize_call(call, result)

    if unsafe_reason:
        return RecoveryPlan(run_id, False, unsafe_reason)

    for call in list(open_calls):
        if call.get("execution_finished"):
            finalize_call(call, call.get("execution_result"))
            continue
        if call.get("approval_decision") == "deny":
            finalize_call(call, TOOL_REJECTED_MESSAGE)
            continue
        if call.get("execution_started"):
            return RecoveryPlan(run_id, False, "tool_effect_started_without_durable_finish")
        if call.get("awaiting_confirmation"):
            return RecoveryPlan(run_id, False, "tool_approval_interrupted_before_decision")
        return RecoveryPlan(run_id, False, "tool_proposal_not_proven_unexecuted")

    resume = copy.deepcopy(request_payload)
    resume["messages"] = messages
    resume["stream"] = True
    resume["cancel_id"] = run_id
    resume["generation_run_id"] = run_id
    resume["finalization_idempotency_key"] = run_id
    if assistant_text:
        resume["messages"].append({"role": "assistant", "content": assistant_text})
        resume["continue_final_message"] = True
        reason = "resume_trailing_assistant_partial"
    else:
        resume.pop("continue_final_message", None)
        reason = "resume_after_durable_tool_result" if saw_tool_activity else "restart_before_output"

    if not saw_generated_output and not saw_tool_activity:
        reason = "restart_before_output"
    return RecoveryPlan(run_id, True, reason, resume)


def plan_orphaned_runs() -> list[RecoveryPlan]:
    plans: list[RecoveryPlan] = []
    for run in runs_db.list_all_active():
        run_id = str(run.get("id") or "")
        if not run_id:
            continue
        if run.get("status") == "cancelling" or run.get("cancelRequested") is True:
            plans.append(RecoveryPlan(run_id, False, "explicit_stop_was_pending"))
            continue
        snapshot = runs_db.get_recovery_snapshot(run_id)
        if snapshot is None:
            plans.append(RecoveryPlan(run_id, False, "ownership_changed_during_planning"))
            continue
        run, worker_token, progress_at = snapshot
        last_event_seq = int(run.get("lastEventSeq") or 0)
        request_payload = run.get("requestPayload")
        if not isinstance(request_payload, dict):
            plans.append(RecoveryPlan(run_id, False, "request_payload_unavailable"))
            continue
        events, event_error = _load_exact_event_snapshot(run_id, last_event_seq)
        if event_error is not None or events is None:
            plan = RecoveryPlan(run_id, False, event_error or "recovery_event_log_incomplete")
        else:
            public_spans, span_error = _public_tool_spans(events)
            approvals = runs_db.list_tool_approvals(run_id, include_checkpoint=True)
            executions = runs_db.list_ungated_tool_executions(run_id)
            refreshed = runs_db.get_recovery_snapshot(run_id)
            if refreshed is None:
                plan = RecoveryPlan(run_id, False, "recovery_event_snapshot_changed")
            else:
                refreshed_run, refreshed_token, refreshed_progress = refreshed
                if (
                    refreshed_token != worker_token
                    or refreshed_progress != progress_at
                    or int(refreshed_run.get("lastEventSeq") or 0) != last_event_seq
                ):
                    plan = RecoveryPlan(run_id, False, "recovery_event_snapshot_changed")
                elif span_error is not None or public_spans is None:
                    plan = RecoveryPlan(
                        run_id,
                        False,
                        span_error or "tool_public_frontier_is_ambiguous",
                    )
                else:
                    approval_plan = _approval_recovery_plan(
                        run_id, request_payload, approvals, events, public_spans
                    )
                    ungated_plan = _ungated_recovery_plan(
                        run_id, request_payload, executions, events, public_spans
                    )
                    authority_plans = [
                        candidate
                        for candidate in (approval_plan, ungated_plan)
                        if candidate is not None
                    ]
                    if len(authority_plans) > 1:
                        plan = RecoveryPlan(
                            run_id,
                            False,
                            "mixed_tool_authority_frontier_is_ambiguous",
                        )
                    elif authority_plans:
                        plan = authority_plans[0]
                    else:
                        plan = None
        if plan is None:
            # With no proposed tool call, the existing receipt planner remains
            # useful for restart-before-output and trailing plain text. It may
            # not authorize a historical tool proposal without a v2 checkpoint.
            plan = build_restart_request(
                request_payload,
                events,
            )
            if plan.safe:
                plan = RecoveryPlan(
                    plan.run_id,
                    True,
                    plan.reason,
                    plan.request_payload,
                    "resume_model",
                )
        plans.append(
            RecoveryPlan(
                plan.run_id,
                plan.safe,
                plan.reason,
                plan.request_payload,
                plan.action,
                plan.approval_id,
                plan.resume_checkpoint,
                worker_token,
                progress_at,
                last_event_seq,
            )
        )
    return plans


def requeue_planned_runs(
    plans: list[RecoveryPlan],
    *,
    stale_before_ms: int | None = None,
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for plan in plans:
        if not plan.safe or plan.request_payload is None:
            continue
        run = runs_db.requeue_run_for_restart(
            plan.run_id,
            plan.request_payload,
            reason=plan.reason,
            expected_worker_token=plan.expected_worker_token,
            expected_progress_at=plan.expected_progress_at,
            expected_last_event_seq=plan.expected_last_event_seq,
            stale_before_ms=stale_before_ms,
        )
        if run is not None:
            run["_recoveryAction"] = plan.action
            run["_approvalId"] = plan.approval_id
            run["_resumeCheckpoint"] = plan.resume_checkpoint
            recovered.append(run)
    return recovered
