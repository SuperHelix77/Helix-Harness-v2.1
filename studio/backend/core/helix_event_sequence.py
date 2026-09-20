# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded process-local ordering clock for Helix observable execution events."""

from __future__ import annotations

import hashlib
import threading
import time

from utils.account_context import OWNER_ACCOUNT_ID, current_account_id
from state import active_generations

_LOCK = threading.Lock()
_CLOCKS: dict[tuple[str, str], tuple[int, int]] = {}
_MAX_EVENT_SCOPES = 1024
_MAX_EVENT_GLOBAL_SCOPES = 8192


def _evict_account_scope(account_id: str) -> None:
    """Evict only the oldest clock owned by the account at its quota."""
    owned = sum(1 for key in _CLOCKS if key[0] == account_id)
    if owned < _MAX_EVENT_SCOPES:
        return
    for key in _CLOCKS:
        if key[0] == account_id:
            _CLOCKS.pop(key, None)
            return


def _reserve_event_scope(scope: tuple[str, str]) -> bool:
    """Reserve a clock scope without evicting another account's clock."""
    if scope in _CLOCKS:
        return True
    owned = [existing for existing in _CLOCKS if existing[0] == scope[0]]
    if len(owned) >= _MAX_EVENT_SCOPES:
        _CLOCKS.pop(owned[0], None)
    elif len(_CLOCKS) >= _MAX_EVENT_GLOBAL_SCOPES:
        if not owned:
            return False
        _CLOCKS.pop(owned[0], None)
    return True


def purge_account(account_id: str) -> int:
    """Drop all process-local event clocks owned by a retired account."""
    account = str(account_id)
    removed = 0
    with _LOCK:
        for scope in tuple(_CLOCKS):
            if scope[0] == account:
                _CLOCKS.pop(scope, None)
                removed += 1
    return removed


def event_scope_id(
    session_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Return the account-private base Helix execution scope.

    Project workspaces may share one session/sandbox id across multiple chat threads.
    Those threads need separate execution traces, so differing non-empty ids use the
    same ``session::thread::<thread>`` convention as Helix capture. Callers whose
    session already equals the thread (or only have one id) keep their historical key.

    Owner keys intentionally keep their historical shape. Managed-account keys use an
    opaque account digest so process-global consumers such as the computer live-feed
    store cannot collide when two accounts choose the same caller-controlled ids.
    """
    logical = public_event_scope_id(session_id, thread_id)
    account_id = current_account_id()
    if account_id == OWNER_ACCOUNT_ID:
        # A managed key is opaque storage state, not a caller-controlled public
        # session id. Escape the reserved namespace so an owner request cannot
        # replay a managed account's internal feed key if it learns that value.
        return f"owner::{logical}" if logical.startswith("account::") else logical
    digest = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:24]
    return f"account::{digest}::{logical}"


def public_event_scope_id(
    session_id: str | None = None,
    thread_id: str | None = None,
) -> str:
    """Return the caller-facing logical scope without its storage namespace."""
    session = str(session_id or "").strip()
    thread = str(thread_id or "").strip()
    if session and thread and session != thread:
        suffix = f"::thread::{thread}"
        return session if session.endswith(suffix) else f"{session}{suffix}"
    return session or thread or "default"


def _base_scope(scope_id: str | None) -> str:
    """Tool capture is turn-specific; ordering is shared by the thread/session base."""
    return str(scope_id or "default").split("::turn::", 1)[0] or "default"


def next_event_metadata(
    scope_id: str | None,
    *,
    lifecycle_epoch: int | None = None,
) -> dict[str, int]:
    """Return a monotonic per-scope sequence and non-decreasing wall-clock timestamp.

    The clock is intentionally process-local: it orders live observations for the
    execution graph and is not a durable global counter. The map is FIFO bounded so
    abandoned sessions cannot accumulate unbounded state.
    """
    scope = (current_account_id(), _base_scope(scope_id))
    account = scope[0]
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    now_ms = time.time_ns() // 1_000_000
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        with _LOCK:
            previous_sequence, previous_ms = _CLOCKS.get(scope, (0, 0))
            sequence = previous_sequence + 1
            created_at_ms = max(now_ms, previous_ms)
            retained = _reserve_event_scope(scope) if allowed else False
            # Refresh insertion order for active scopes so the bound preferentially
            # evicts genuinely stale sessions.
            if retained:
                _CLOCKS.pop(scope, None)
                _CLOCKS[scope] = (sequence, created_at_ms)
        return {"sequence": sequence, "created_at_ms": created_at_ms}
