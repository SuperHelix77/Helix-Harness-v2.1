# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Account-scoped wakeups for durable and legacy tool approvals.

SQLite is authoritative for durable chat runs. The process-local event is only
an optimization: every waiter reads before sleeping and after every wake/poll,
so a decision committed before a crash or with a lost notification still wins.
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any

from storage import chat_generation_runs_db as runs_db
from utils.account_context import current_account_id

_DECISION_TIMEOUT = 3600.0
TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."


class ToolApprovalDetached(RuntimeError):
    """The worker detached without a user Stop; leave the durable row pending."""


_lock = threading.Lock()
# (account, run, approval) for durable; run is "" for legacy.
class _PendingSlots(dict):
    """Tuple-keyed map with legacy current-account string lookup compatibility."""

    @staticmethod
    def _scope(key):
        if isinstance(key, str):
            return current_account_id(), "", key
        return key

    def __contains__(self, key):
        return super().__contains__(self._scope(key))

    def __getitem__(self, key):
        return super().__getitem__(self._scope(key))

    def get(self, key, default=None):
        return super().get(self._scope(key), default)

    def pop(self, key, default=None):
        return super().pop(self._scope(key), default)


_pending: _PendingSlots = _PendingSlots()


def _durable_binding() -> tuple[str, str] | None:
    try:
        from core.inference.durable_tool_journal import current_durable_tool_run

        return current_durable_tool_run()
    except Exception:
        return None


def _key(approval_id: str, run_id: str = "") -> tuple[str, str, str]:
    return current_account_id(), str(run_id or ""), str(approval_id or "")


def new_approval_id() -> str:
    return secrets.token_urlsafe(16)


def begin_tool_decision(session_id, approval_id) -> dict[str, Any]:
    binding = _durable_binding()
    run_id = binding[0] if binding is not None else ""
    slot: dict[str, Any] = {
        "event": threading.Event(),
        "decision": None,
        "session": session_id or "",
        "account": current_account_id(),
        "run_id": run_id,
        "worker_token": binding[1] if binding is not None else "",
    }
    with _lock:
        _pending[(slot["account"], run_id, approval_id)] = slot
    return slot


def _read_durable(slot: dict[str, Any], approval_id: str) -> dict[str, Any] | None:
    run_id = str(slot.get("run_id") or "")
    if not run_id:
        return None
    return runs_db.get_tool_approval(run_id, approval_id, include_checkpoint=True)


def _bind_allowed_approval(slot: dict[str, Any], approval_id: str) -> None:
    from core.inference.durable_tool_journal import bind_next_tool_approval

    bind_next_tool_approval(str(slot["run_id"]), approval_id)


def wait_tool_decision(
    slot,
    approval_id,
    cancel_event=None,
    timeout=_DECISION_TIMEOUT,
):
    """Return allow/deny from durable state, or detach on non-Stop shutdown."""

    run_id = str(slot.get("run_id") or "")
    deadline = time.monotonic() + max(0.0, float(timeout))
    try:
        while True:
            if run_id:
                approval = _read_durable(slot, approval_id)
                if approval is not None:
                    decision = approval.get("decision") or approval.get("status")
                    if decision in {"allow", "deny"}:
                        if decision == "allow":
                            _bind_allowed_approval(slot, approval_id)
                        return decision
                    expires_at = int(approval.get("expiresAt") or 0)
                    if expires_at and expires_at <= runs_db.now_ms():
                        expired = runs_db.expire_tool_approval(run_id, approval_id)
                        if expired is not None:
                            return "deny"
                if cancel_event is not None and cancel_event.is_set():
                    run = runs_db.get_run(run_id)
                    if run is not None and (
                        run.get("cancelRequested") is True
                        or run.get("status") in {"cancelling", "cancelled"}
                    ):
                        # request_cancel() committed the sourced denial in the
                        # same transaction as cancel_requested.
                        return "deny"
                    raise ToolApprovalDetached(
                        "durable tool approval worker detached before a decision"
                    )
            else:
                if slot.get("decision") is not None or slot["event"].is_set():
                    return slot.get("decision") or "deny"
                if cancel_event is not None and cancel_event.is_set():
                    return "deny"

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if run_id:
                    runs_db.expire_tool_approval(run_id, approval_id)
                return "deny"
            slot["event"].wait(timeout=min(0.5, remaining))
            slot["event"].clear()
    finally:
        with _lock:
            key = (str(slot.get("account") or ""), run_id, approval_id)
            if _pending.get(key) is slot:
                _pending.pop(key, None)


def abort_tool_decision(slot, approval_id) -> None:
    """Drop only the local wake slot; durable pending state is untouched."""

    with _lock:
        key = (
            str(slot.get("account") or ""),
            str(slot.get("run_id") or ""),
            approval_id,
        )
        if _pending.get(key) is slot:
            _pending.pop(key, None)


def request_tool_decision(
    session_id,
    approval_id,
    cancel_event=None,
    timeout=_DECISION_TIMEOUT,
):
    slot = begin_tool_decision(session_id, approval_id)
    return wait_tool_decision(slot, approval_id, cancel_event=cancel_event, timeout=timeout)


def notify_tool_decision(run_id: str, approval_id: str) -> None:
    """Best-effort local wake after the authoritative transaction commits."""

    key = _key(approval_id, run_id)
    with _lock:
        slot = _pending.get(key)
        if slot is not None:
            slot["event"].set()


def resolve_tool_decision(
    approval_id,
    decision,
    session_id=None,
    *,
    run_id: str | None = None,
    owner_subject: str | None = None,
) -> bool:
    """Resolve a durable decision transactionally, or a scoped legacy slot."""

    if not approval_id:
        return False
    if run_id:
        if owner_subject is None:
            raise ValueError("owner_subject is required for durable tool approval")
        runs_db.decide_tool_approval(
            run_id,
            approval_id,
            owner_subject=owner_subject,
            decision=decision,
            session_id=session_id,
            source="user",
        )
        notify_tool_decision(run_id, approval_id)
        return True

    key = _key(approval_id)
    with _lock:
        slot = _pending.get(key)
        if not slot:
            return False
        if session_id is not None and slot["session"] != (session_id or ""):
            return False
        if slot["decision"] is not None:
            return slot["decision"] == decision
        slot["decision"] = decision
        slot["event"].set()
    return True
