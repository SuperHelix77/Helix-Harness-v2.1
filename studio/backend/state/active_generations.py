# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registry of in-flight chat generations, keyed by conversation.

New Chat leaves the previous conversation streaming, so /load and /unload need
to know which chats a reload would interrupt: they refuse with 409 unless the
caller opts in to cancelling them, and GET /inference/active-generations lets
the UI name them. A frontend guard alone would miss a second tab or a REST call.

Entries hold the same threading.Event as the per-run cancel registry in
routes/inference.py, so cancel_all() closes each generation's own upstream
stream and never signals llama-server itself.

A plain dict plus a threading.Lock: no signals, no process groups, no event loop
affinity, so it behaves identically on Linux, macOS, Windows and WSL.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from utils.account_context import current_account_id

# Keyed by handle, not thread_id: a tool continuation can register before the previous leg
# unregisters, and one key would drop the other.
_ACTIVE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
# Disabled or retired accounts: a generation registering late for one starts cancelled.
_FENCED: set[str] = set()
# Every fence transition advances the account epoch.  Telemetry callbacks bind
# to the epoch in which their generation was registered, so a stale callback
# cannot repopulate state after a deactivate/reactivate ABA cycle.
_EPOCHS: dict[str, int] = {}
_CURRENT_EPOCH: ContextVar[tuple[str, int] | None] = ContextVar(
    "active_generation_epoch", default = None
)


class ActiveGeneration:
    """Registers one in-flight generation for the duration of the block.

    Each __enter__ mints its own handle, so overlapping uses never clobber.
    """

    __slots__ = (
        "thread_id",
        "run_id",
        "cancel_event",
        "model",
        "kind",
        "account_id",
        "_handle",
        "_borrowed",
        "_epoch",
        "_epoch_token",
    )

    def __init__(
        self,
        cancel_event: threading.Event,
        *,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        kind: str = "chat",
        account_id: Optional[str] = None,
    ):
        self.account_id = account_id or current_account_id()
        self.thread_id = thread_id or None
        self.run_id = run_id or None
        self.cancel_event = cancel_event
        self.model = model or None
        self.kind = kind
        self._handle: Optional[str] = None
        self._borrowed = False
        self._epoch: Optional[int] = None
        self._epoch_token = None

    def __enter__(self) -> "ActiveGeneration":
        with _LOCK:
            self._epoch = _EPOCHS.setdefault(self.account_id, 0)
            # A durable supervisor registers before model loading starts and the route later enters its normal
            # tracker with that same event and run, so borrow the outer registration.
            if self.run_id:
                for entry in _ACTIVE.values():
                    if (
                        entry["run_id"] != self.run_id
                        or entry["event"] is not self.cancel_event
                        or entry["account_id"] != self.account_id
                    ):
                        continue
                    if self.thread_id:
                        entry["thread_id"] = self.thread_id
                    if self.model:
                        entry["model"] = self.model
                    if self.kind:
                        entry["kind"] = self.kind
                    self._borrowed = True
                    # Borrow the owner's immutable lifecycle, not the account's
                    # current epoch. An old generation that reaches this nested
                    # tracker after deactivate/reactivate must remain stale.
                    self._epoch = int(entry["lifecycle_epoch"])
                    self.bind()
                    return self
            self._handle = uuid.uuid4().hex
            _ACTIVE[self._handle] = {
                "handle": self._handle,
                "thread_id": self.thread_id,
                "run_id": self.run_id,
                "model": self.model,
                "kind": self.kind,
                "account_id": self.account_id,
                "lifecycle_epoch": self._epoch,
                "started_at": time.time(),
                "event": self.cancel_event,
            }
            if self.account_id in _FENCED:
                self.cancel_event.set()
            self.bind()
        return self

    @property
    def lifecycle_binding(self) -> tuple[str, int] | None:
        if self._epoch is None:
            return None
        return (self.account_id, int(self._epoch))

    def bind(self) -> None:
        """Bind this registration's immutable epoch to the current Context."""
        if self._epoch_token is not None:
            return
        if self._epoch is None:
            raise RuntimeError("active generation has not been registered")
        self._epoch_token = _CURRENT_EPOCH.set((self.account_id, int(self._epoch)))

    def unbind(self) -> None:
        """Restore the current Context; tokens may not cross asyncio Contexts."""
        token = self._epoch_token
        if token is not None:
            _CURRENT_EPOCH.reset(token)
            self._epoch_token = None

    def unregister(self) -> None:
        """Release process-global ownership without touching Context-local state."""
        if self._borrowed:
            self._borrowed = False
            return
        handle, self._handle = self._handle, None
        if handle is not None:
            with _LOCK:
                _ACTIVE.pop(handle, None)

    def __exit__(self, *exc) -> bool:
        try:
            self.unbind()
        finally:
            self.unregister()
        return False


def snapshot(account_id: Optional[str] = None) -> list[dict[str, Any]]:
    """In-flight generations, newest last; ``account_id`` None (all) is shutdown/arbiter only."""
    with _LOCK:
        entries = [
            e for e in _ACTIVE.values() if account_id is None or e["account_id"] == account_id
        ]
    entries.sort(key = lambda e: e["started_at"])
    return [
        {
            "handle": e["handle"],
            "thread_id": e["thread_id"],
            "run_id": e["run_id"],
            "model": e["model"],
            "kind": e["kind"],
            "account_id": e["account_id"],
            "started_at": e["started_at"],
        }
        for e in entries
    ]


def active_thread_ids(account_id: Optional[str] = None) -> list[str]:
    """Distinct conversation ids with a generation in flight, in start order.

    A first turn that races persistence has no thread id yet: count() sees it,
    this cannot name it.
    """
    seen: list[str] = []
    for e in snapshot(account_id):
        tid = e["thread_id"]
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def count(account_id: Optional[str] = None) -> int:
    with _LOCK:
        if account_id is None:
            return len(_ACTIVE)
        return sum(1 for e in _ACTIVE.values() if e["account_id"] == account_id)


def foreign_count(account_id: str) -> int:
    """Generations in flight for OTHER accounts, which a load or unload must not interrupt."""
    with _LOCK:
        return sum(1 for e in _ACTIVE.values() if e["account_id"] != account_id)


def cancel_all(account_id: Optional[str] = None) -> int:
    """Signal in-flight generations to stop, returning the count. Request-driven callers must pass
    ``account_id``; None means everyone and is for shutdown only."""
    with _LOCK:
        events = [
            e["event"]
            for e in _ACTIVE.values()
            if account_id is None or e["account_id"] == account_id
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_thread(thread_id: str, account_id: Optional[str] = None) -> int:
    """Signal ``thread_id``'s generations; thread ids are client-chosen, so scope by account."""
    if not thread_id:
        return 0
    scope = account_id or current_account_id()
    with _LOCK:
        events = [
            e["event"]
            for e in _ACTIVE.values()
            if e["thread_id"] == thread_id and e["account_id"] == scope
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def cancel_run(run_id: str, account_id: Optional[str] = None) -> int:
    if not run_id:
        return 0
    scope = account_id or current_account_id()
    with _LOCK:
        events = [
            e["event"]
            for e in _ACTIVE.values()
            if e["run_id"] == run_id and e["account_id"] == scope
        ]
    for ev in events:
        try:
            ev.set()
        except Exception:
            pass
    return len(events)


def fence(account_id: str) -> None:
    """Deactivation or retirement: cancel and invalidate prior callbacks."""
    with _LOCK:
        account = str(account_id)
        _EPOCHS[account] = _EPOCHS.get(account, 0) + 1
        _FENCED.add(account)


def lift_fence(account_id: str) -> None:
    with _LOCK:
        account = str(account_id)
        # Advance on both edges so callbacks born while the account was fenced
        # are also stale once the account is active again.
        _EPOCHS[account] = _EPOCHS.get(account, 0) + 1
        _FENCED.discard(account)


def fenced(account_id: str) -> bool:
    with _LOCK:
        return account_id in _FENCED


def lifecycle_epoch(account_id: Optional[str] = None) -> int:
    """Return the current account epoch, or the bound generation epoch in context."""
    account = str(account_id or current_account_id())
    bound = _CURRENT_EPOCH.get()
    if bound is not None and bound[0] == account:
        return int(bound[1])
    with _LOCK:
        return int(_EPOCHS.get(account, 0))


def bound_lifecycle_epoch(account_id: Optional[str] = None) -> int:
    """Return the epoch a telemetry callback should use for this execution context."""
    return lifecycle_epoch(account_id)


@contextmanager
def bind_lifecycle_epoch(account_id: str, epoch: int):
    """Bind an immutable owner epoch using a fresh token in this Context only."""
    token = _CURRENT_EPOCH.set((str(account_id), int(epoch)))
    try:
        yield
    finally:
        _CURRENT_EPOCH.reset(token)


@contextmanager
def telemetry_guard(account_id: str, epoch: Optional[int] = None):
    """Hold the lifecycle lock while a telemetry store mutation is performed.

    The lock ordering is lifecycle -> telemetry store.  Fencing therefore cannot
    race a final writer check and an insertion, and the epoch comparison closes
    the deactivate/reactivate ABA hole.
    """
    account = str(account_id)
    bound = lifecycle_epoch(account) if epoch is None else int(epoch)
    with _LOCK:
        current = int(_EPOCHS.get(account, 0))
        allowed = account not in _FENCED and bound == current
        yield allowed


def reset_for_tests() -> None:
    """Drop every entry. Test-only; never called from request paths."""
    with _LOCK:
        _ACTIVE.clear()
        _FENCED.clear()
        _EPOCHS.clear()
        _CURRENT_EPOCH.set(None)
