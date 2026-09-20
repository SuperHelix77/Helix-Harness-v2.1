# SPDX-License-Identifier: AGPL-3.0-only
"""Live tool-loop capture. Pure in-process store; llama I/O stays out."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from core.helix_event_sequence import next_event_metadata, public_event_scope_id
from state import active_generations
from utils.account_context import current_account_id

from .pipeline import run_adaptation_pipeline
from .schemas import ToolControlEvent
from .trajectory import (
    ToolStep,
    ToolVerificationReceipt,
    Trajectory,
    VerificationKind,
    VerificationStatus,
)

_LOCK = threading.Lock()
_StoreKey = tuple[str, str]
_STEPS: dict[_StoreKey, list[ToolStep]] = {}
_RECENT_COMPLETED: dict[_StoreKey, list[ToolStep]] = {}
_CONTROL_EVENTS: dict[_StoreKey, list[ToolControlEvent]] = {}
_RECENT_CONTROL_EVENTS: dict[_StoreKey, list[ToolControlEvent]] = {}
_RECENT_TURN_IDS: dict[_StoreKey, str] = {}
_MAX_CAPTURE_SESSIONS = 512
_MAX_CAPTURE_GLOBAL_SCOPES = 4096


def _can_reserve_entry(store: dict[_StoreKey, Any], key: _StoreKey) -> bool:
    """Return whether ``key`` may be inserted without evicting another account.

    The process-local stores have both an account quota and a global bound.  A
    full global store is allowed to make room only from the current account;
    otherwise a new account is deliberately dropped rather than displacing a
    live scope belonging to somebody else.
    """
    if key in store:
        return True
    owned = any(existing[0] == key[0] for existing in store)
    owned_count = sum(1 for existing in store if existing[0] == key[0])
    return (
        owned_count > 0
        if len(store) >= _MAX_CAPTURE_GLOBAL_SCOPES
        else owned_count < _MAX_CAPTURE_SESSIONS or owned
    )


def _reserve_entry(store: dict[_StoreKey, Any], key: _StoreKey) -> bool:
    if not _can_reserve_entry(store, key):
        return False
    if key in store:
        return True
    # A per-account quota is enforced first.  Since the account owns at least
    # one entry in this branch, both quota and global eviction remain private.
    owned_count = sum(1 for existing in store if existing[0] == key[0])
    if owned_count >= _MAX_CAPTURE_SESSIONS:
        for existing in store:
            if existing[0] == key[0]:
                store.pop(existing, None)
                break
    if len(store) >= _MAX_CAPTURE_GLOBAL_SCOPES:
        for existing in store:
            if existing[0] == key[0]:
                store.pop(existing, None)
                break
        else:
            return False
    return True


def _evict_account_entry(store: dict[_StoreKey, Any], account_id: str) -> None:
    """Evict one oldest scope owned by ``account_id`` when its quota is full.

    These stores are process-local but account isolation still applies to their
    bounded-memory policy: an account must never be able to evict another
    account's live or archived capture.  The insertion-ordered dicts provide a
    deterministic FIFO choice without a second recency index.
    """
    owned = sum(1 for key in store if key[0] == account_id)
    if owned < _MAX_CAPTURE_SESSIONS:
        return
    for key in store:
        if key[0] == account_id:
            store.pop(key, None)
            return


def purge_account(account_id: str) -> int:
    """Drop all process-local capture state owned by a retired account."""
    account = str(account_id)
    removed = 0
    with _LOCK:
        for store in (
            _STEPS,
            _RECENT_COMPLETED,
            _CONTROL_EVENTS,
            _RECENT_CONTROL_EVENTS,
            _RECENT_TURN_IDS,
        ):
            for key in tuple(store):
                if key[0] == account:
                    store.pop(key, None)
                    removed += 1
    return removed


def capture_session_key(
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Return the capture scope without changing the filesystem sandbox id.

    Project chats intentionally share a filesystem sandbox, but their behavioral
    trajectories must not share tool history.  Thread scope is therefore appended
    only for Helix capture.  Non-project callers that already use the thread id as
    session id retain the historical key.
    """
    turn = str(turn_id or "").strip()
    base = public_event_scope_id(session_id, thread_id)
    return f"{base}::turn::{turn}" if turn else base


def _store_key(session_id: str | None) -> _StoreKey:
    """Namespace every process-local capture access by immutable account id."""
    return current_account_id(), str(session_id or "default")


def _args_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


_EXIT_RE = re.compile(r"^exit code\s+(-?\d+)\s*:?", re.IGNORECASE)


def _receipt_error(text: str) -> str | None:
    stripped = text.lstrip()
    lowered = stripped.lower()
    if lowered.startswith("error:"):
        return stripped[:400]
    match = _EXIT_RE.match(stripped)
    if match and int(match.group(1)) != 0:
        return stripped[:400]
    if lowered.startswith((
        "cancelled",
        "canceled",
        "timed out",
        "timeout:",
        "execution cancelled",
        "execution canceled",
        "execution timed out",
    )):
        return stripped[:400]
    return None


def _verification_kind(value: VerificationKind | str | None) -> VerificationKind | None:
    if isinstance(value, VerificationKind):
        return value
    if value is None:
        return None
    try:
        return VerificationKind(str(value).strip().lower())
    except Exception:  # noqa: BLE001 -- optional capture metadata must fail closed
        return None


def record_tool_execution(
    session_id: str,
    name: str,
    arguments: Any,
    result: str,
    *,
    verification_kind: VerificationKind | str | None = None,
    verification_claim: str | None = None,
    verification_subject: str | None = None,
    lifecycle_epoch: int | None = None,
) -> ToolStep:
    text = str(result or "")
    args = _args_text(arguments)
    error = _receipt_error(text)
    hint = "wrong" if error else "useful"
    typed_kind = _verification_kind(verification_kind)
    bound_claim = str(verification_claim or "").strip()[:2_000]
    bound_subject = str(verification_subject or "").strip()[:500]
    verification = (
        ToolVerificationReceipt(
            kind=typed_kind,
            status=VerificationStatus.FAILED if error else VerificationStatus.PASSED,
            detail=(error or text)[:500],
            claim=bound_claim,
            subject=bound_subject,
        )
        if typed_kind is not None
        else None
    )
    logical_key = session_id or "default"
    key = _store_key(logical_key)
    account = current_account_id()
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    ordering = next_event_metadata(logical_key, lifecycle_epoch=bound_epoch)
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        with _LOCK:
            retained = _reserve_entry(_STEPS, key) if allowed else False
            prior = _STEPS.setdefault(key, []) if retained else []
            # Retry is objective execution history, not a model-authored label.  The
            # first execution is 0; each later execution of the same normalized tool
            # + arguments increments it regardless of whether the earlier attempt
            # succeeded.  Successful duplicates should normally be stopped by the
            # live ToolLoopController, while failed calls are allowed a bounded retry.
            retry = sum(1 for step in prior if step.name == name and step.arguments == args)
            if any(step.name == name and step.arguments == args and step.result[:200] == text[:200] for step in prior):
                hint = "redundant"
            step = ToolStep(
                name=str(name),
                arguments=args,
                result=text[:8_000],
                useful_hint=hint,
                error=error,
                retry=retry,
                verification=verification,
                sequence=ordering["sequence"],
                created_at_ms=ordering["created_at_ms"],
            )
            if retained:
                prior.append(step)
                _STEPS[key] = prior[-200:]
            return step


def record_tool_control_event(
    session_id: str,
    *,
    action: str,
    tool_name: str,
    arguments: Any,
    reason: str = "",
    equivalent_to: str = "",
    failed_attempts: int = 0,
    progress: dict[str, Any] | None = None,
    lifecycle_epoch: int | None = None,
) -> ToolControlEvent:
    """Record a bounded pre-execution controller no-op separately from real tools."""
    args = _args_text(arguments)[:4_000]
    bounded_progress: dict[str, Any] = {}
    for key, value in dict(progress or {}).items():
        if isinstance(value, (bool, int, float)) or value is None:
            bounded_progress[str(key)[:120]] = value
    logical_key = session_id or "default"
    key = _store_key(logical_key)
    account = current_account_id()
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    ordering = next_event_metadata(logical_key, lifecycle_epoch=bound_epoch)
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        with _LOCK:
            event = ToolControlEvent(
                action=str(action)[:120],
                tool_name=str(tool_name)[:200],
                arguments=args,
                reason=str(reason or "")[:2_000],
                equivalent_to=str(equivalent_to or "")[:200],
                failed_attempts=max(0, int(failed_attempts or 0)),
                progress=bounded_progress,
                sequence=ordering["sequence"],
                created_at_ms=ordering["created_at_ms"],
            )
            if allowed and _reserve_entry(_CONTROL_EVENTS, key):
                prior = _CONTROL_EVENTS.setdefault(key, [])
                prior.append(event)
                _CONTROL_EVENTS[key] = prior[-200:]
            return event


def make_tool_control_observer(
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
):
    """Return a fail-open observer for ToolLoopController no-op decisions."""
    if not str(turn_id or "").strip():
        return None
    key = capture_session_key(session_id, thread_id, turn_id)
    bound_epoch = active_generations.bound_lifecycle_epoch()

    def _observe(decision, progress) -> None:
        try:
            record_tool_control_event(
                key,
                action=getattr(decision, "action", ""),
                tool_name=getattr(decision, "tool_name", ""),
                arguments=getattr(decision, "arguments", {}),
                reason=getattr(decision, "noop_result", ""),
                equivalent_to=getattr(decision, "equivalent_to", ""),
                failed_attempts=getattr(decision, "failed_attempts", 0),
                progress=dict(progress or {}),
                lifecycle_epoch=bound_epoch,
            )
        except BaseException:
            # Optional Helix observation must never alter the foreground loop.
            pass

    return _observe


def session_steps(session_id: str) -> list[ToolStep]:
    with _LOCK:
        return list(_STEPS.get(_store_key(session_id), []))


def session_control_events(session_id: str) -> list[ToolControlEvent]:
    with _LOCK:
        return list(_CONTROL_EVENTS.get(_store_key(session_id), []))


def capture_snapshot(session_id: str) -> dict[str, Any]:
    """Return a bounded JSON-safe snapshot of one exact live capture scope.

    Final-ingest durability uses this before any post-answer learning effect. The
    in-process capture store intentionally stays hot-path-only, while the keyed
    final-ingest claim persists this snapshot so a backend restart can replay the
    same immutable tool trajectory rather than analyzing an empty process-local
    store.
    """
    key = _store_key(session_id)
    with _LOCK:
        steps = list(_STEPS.get(key, []))[-200:]
        controls = list(_CONTROL_EVENTS.get(key, []))[-200:]

    encoded_steps: list[dict[str, Any]] = []
    for step in steps:
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
        encoded_steps.append(
            {
                "name": step.name,
                "arguments": step.arguments,
                "result": step.result,
                "useful_hint": step.useful_hint,
                "error": step.error,
                "retry": step.retry,
                "verification": verification,
                "sequence": step.sequence,
                "created_at_ms": step.created_at_ms,
            }
        )
    return {
        "schema_version": "helix.capture-snapshot.v1",
        "steps": encoded_steps,
        "control_events": [event.to_dict() for event in controls],
    }


def restore_capture_snapshot(
    session_id: str,
    snapshot: dict[str, Any] | None,
    *,
    lifecycle_epoch: int | None = None,
) -> bool:
    """Restore a trusted durable capture snapshot when the process-local copy is gone."""
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "helix.capture-snapshot.v1":
        return False
    raw_steps = snapshot.get("steps")
    raw_controls = snapshot.get("control_events")
    if not isinstance(raw_steps, list) or not isinstance(raw_controls, list):
        return False

    restored_steps: list[ToolStep] = []
    for raw in raw_steps[:200]:
        if not isinstance(raw, dict):
            continue
        verification = None
        raw_verification = raw.get("verification")
        if isinstance(raw_verification, dict):
            try:
                verification = ToolVerificationReceipt(
                    kind=VerificationKind(str(raw_verification.get("kind") or "")),
                    status=VerificationStatus(str(raw_verification.get("status") or "")),
                    provenance=str(raw_verification.get("provenance") or "backend_tool_capture")[:120],
                    detail=str(raw_verification.get("detail") or "")[:500],
                    claim=str(raw_verification.get("claim") or "")[:2_000],
                    subject=str(raw_verification.get("subject") or "")[:500],
                )
            except (TypeError, ValueError):
                verification = None
        try:
            retry = max(0, int(raw.get("retry") or 0))
            sequence = max(0, int(raw.get("sequence") or 0))
            created_at_ms = max(0, int(raw.get("created_at_ms") or 0))
        except (TypeError, ValueError, OverflowError):
            continue
        restored_steps.append(
            ToolStep(
                name=str(raw.get("name") or "")[:240],
                arguments=str(raw.get("arguments") or "")[:8_000],
                result=str(raw.get("result") or "")[:8_000],
                useful_hint=str(raw.get("useful_hint") or "")[:120],
                error=(str(raw.get("error"))[:400] if raw.get("error") is not None else None),
                retry=retry,
                verification=verification,
                sequence=sequence,
                created_at_ms=created_at_ms,
            )
        )

    restored_controls: list[ToolControlEvent] = []
    for raw in raw_controls[:200]:
        if not isinstance(raw, dict):
            continue
        try:
            restored_controls.append(
                ToolControlEvent(
                    action=str(raw.get("action") or "")[:120],
                    tool_name=str(raw.get("tool_name") or "")[:200],
                    arguments=str(raw.get("arguments") or "")[:4_000],
                    reason=str(raw.get("reason") or "")[:2_000],
                    equivalent_to=str(raw.get("equivalent_to") or "")[:200],
                    failed_attempts=max(0, int(raw.get("failed_attempts") or 0)),
                    progress=(
                        dict(raw.get("progress"))
                        if isinstance(raw.get("progress"), dict)
                        else {}
                    ),
                    provenance=str(raw.get("provenance") or "runtime_tool_loop")[:120],
                    sequence=max(0, int(raw.get("sequence") or 0)),
                    created_at_ms=max(0, int(raw.get("created_at_ms") or 0)),
                )
            )
        except (TypeError, ValueError, OverflowError):
            continue

    key = _store_key(session_id)
    account = current_account_id()
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        if not allowed:
            return False
        with _LOCK:
            # A still-live same-process capture is newer authority. Durable restore is
            # only for the process-restart hole where both process-local stores are empty.
            if _STEPS.get(key) or _CONTROL_EVENTS.get(key):
                return False
            # Preflight both stores before evicting either one.  A brand-new account
            # must not partially restore into a globally full store.
            if not _can_reserve_entry(_STEPS, key) or not _can_reserve_entry(_CONTROL_EVENTS, key):
                return False
            if not _reserve_entry(_STEPS, key) or not _reserve_entry(_CONTROL_EVENTS, key):
                return False
            _STEPS[key] = restored_steps
            _CONTROL_EVENTS[key] = restored_controls
        return True


def _capture_base(key: str) -> str:
    return key.split("::turn::", 1)[0]


def archive_session(session_id: str, *, lifecycle_epoch: int | None = None) -> None:
    """Consume one exact turn into a bounded latest-completed inspector snapshot."""
    logical_key = session_id or "default"
    key = _store_key(logical_key)
    account = current_account_id()
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        if not allowed:
            return
        with _LOCK:
            steps = _STEPS.pop(key, [])
            control_events = _CONTROL_EVENTS.pop(key, [])
        # An exact completed turn with no tool/controller activity is still a
        # real turn boundary. Persist its empty snapshot so the inspector cannot
        # keep presenting the previous tool-using turn after a text-only answer.
            if not steps and not control_events and "::turn::" not in logical_key:
                return
            base = _store_key(_capture_base(logical_key))
            turn_id = logical_key.split("::turn::", 1)[1] if "::turn::" in logical_key else ""
            archive_stores = [_RECENT_COMPLETED, _RECENT_CONTROL_EVENTS]
            if turn_id:
                archive_stores.insert(0, _RECENT_TURN_IDS)
            # Keep the archived snapshot coherent when a global bound rejects a new
            # account: do not retain a turn id without its matching records (or vice
            # versa).
            if any(not _can_reserve_entry(store, base) for store in archive_stores):
                return
            if any(not _reserve_entry(store, base) for store in archive_stores):
                return
            if turn_id:
                _RECENT_TURN_IDS.pop(base, None)
                _RECENT_TURN_IDS[base] = turn_id
            # Keep one coherent archived snapshot. If the new turn has only steps or
            # only controller events, the empty half must replace the previous turn's
            # data instead of leaking stale events into the new turn.
            _RECENT_COMPLETED.pop(base, None)
            _RECENT_COMPLETED[base] = list(steps)
            _RECENT_CONTROL_EVENTS.pop(base, None)
            _RECENT_CONTROL_EVENTS[base] = list(control_events)


def _latest_live_turn_key_locked(base: _StoreKey) -> _StoreKey | None:
    """Return the newest live exact-turn key by backend event sequence."""
    account_id, logical_base = base
    prefix = logical_base + "::turn::"
    candidates: dict[_StoreKey, tuple[int, int]] = {}
    ordinal = 0
    for store in (_STEPS, _CONTROL_EVENTS):
        for key, records in store.items():
            if key[0] != account_id or not key[1].startswith(prefix):
                continue
            ordinal += 1
            sequence = max((int(getattr(item, "sequence", 0) or 0) for item in records), default=0)
            previous = candidates.get(key, (0, 0))
            candidates[key] = (max(previous[0], sequence), max(previous[1], ordinal))
    if not candidates:
        return None
    return max(candidates, key=lambda key: candidates[key])


def latest_session_snapshot(
    session_id: str | None = None, thread_id: str | None = None
) -> tuple[str, list[ToolStep], list[ToolControlEvent]]:
    """Return one internally consistent live-or-archived turn snapshot.

    A live turn takes precedence over the previous archived turn. The turn id,
    tool steps and controller events are selected under one lock so callers can
    never label records from one logical turn with another turn's id.
    """
    base = _store_key(capture_session_key(session_id, thread_id))
    with _LOCK:
        live_key = _latest_live_turn_key_locked(base)
        if live_key:
            return (
                live_key[1].split("::turn::", 1)[1],
                list(_STEPS.get(live_key, [])),
                list(_CONTROL_EVENTS.get(live_key, [])),
            )
        return (
            str(_RECENT_TURN_IDS.get(base) or ""),
            list(_RECENT_COMPLETED.get(base, _STEPS.get(base, []))),
            list(_RECENT_CONTROL_EVENTS.get(base, _CONTROL_EVENTS.get(base, []))),
        )


def latest_session_steps(
    session_id: str | None = None, thread_id: str | None = None
) -> list[ToolStep]:
    return latest_session_snapshot(session_id, thread_id)[1]


def latest_session_control_events(
    session_id: str | None = None, thread_id: str | None = None
) -> list[ToolControlEvent]:
    return latest_session_snapshot(session_id, thread_id)[2]


def latest_session_turn_id(
    session_id: str | None = None, thread_id: str | None = None
) -> str:
    """Exact logical turn id for the latest archived/live capture in this scope."""
    return latest_session_snapshot(session_id, thread_id)[0]


def clear_session(session_id: str) -> None:
    with _LOCK:
        logical_key = session_id or "default"
        key = _store_key(logical_key)
        _STEPS.pop(key, None)
        _CONTROL_EVENTS.pop(key, None)
        base = _store_key(_capture_base(logical_key))
        if "::turn::" not in logical_key:
            _RECENT_COMPLETED.pop(base, None)
            _RECENT_CONTROL_EVENTS.pop(base, None)
            _RECENT_TURN_IDS.pop(base, None)
        else:
            turn_id = logical_key.split("::turn::", 1)[1]
            if _RECENT_TURN_IDS.get(base) == turn_id:
                _RECENT_COMPLETED.pop(base, None)
                _RECENT_CONTROL_EVENTS.pop(base, None)
                _RECENT_TURN_IDS.pop(base, None)


def trajectory_from_session(
    session_id: str,
    *,
    prompt_state: str = "",
    retrieved_context: str = "",
    reasoning: str = "",
    final_result: str = "",
    user_corrections: list | None = None,
    latency_ms: float = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    verified: bool = False,
    **kwargs: Any,
) -> Trajectory:
    extras = kwargs.pop("extras", {})
    extras = dict(extras) if isinstance(extras, dict) else {}
    control_events = session_control_events(session_id)
    if control_events:
        extras["tool_control_events"] = [event.to_dict() for event in control_events]
    return Trajectory(
        prompt_state=prompt_state,
        retrieved_context=retrieved_context,
        reasoning=reasoning,
        steps=session_steps(session_id),
        user_corrections=list(user_corrections or []),
        final_result=final_result,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        verified=verified,
        extras=extras,
        **kwargs,
    )


def finalize_session(session_id: str, **kwargs: Any) -> dict[str, Any]:
    traj = trajectory_from_session(session_id, **kwargs)
    return run_adaptation_pipeline(traj)
