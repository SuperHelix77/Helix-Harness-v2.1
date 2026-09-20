# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background producer for durable local Studio chat generations."""

from __future__ import annotations

import asyncio
import copy
import contextlib
import json
import math
import os
import threading
import time
from typing import Any, AsyncIterator

from starlette.requests import Request

from core.inference.llama_keepwarm import InferenceActivityReservation
from core.training.account_jobs import sweepable_job_accounts
from utils.account_context import AccountContext, current_account, current_account_id, run_as
from loggers import get_logger
from models.inference import ChatCompletionRequest
from state import active_generations
from storage import chat_generation_runs_db as db

logger = get_logger(__name__)
_EVENT_BATCH_SIZE = 16
_EVENT_BATCH_MIN_SIZE = 2
_EVENT_BATCH_SECONDS = 0.1
_EVENT_SINGLE_FLUSH_SECONDS = 1.0
_SHUTDOWN_GRACE_SECONDS = 10.0
_FINALIZATION_WATCH_RETRY_BASE_SECONDS = 1.0
_FINALIZATION_WATCH_RETRY_MAX_SECONDS = 30.0
RunKey = tuple[str, str]
# Second budget, after task.cancel(). Shorter than the grace period: by this point the run is already being abandoned,
# and the only question is whether shutdown returns.
_SHUTDOWN_CANCEL_SECONDS = 5.0
# The sweeper's own shutdown budget, far below the producers'. Its work is redundant at shutdown and Desktop
# force-kills the backend after five seconds.
_SWEEP_SHUTDOWN_SECONDS = 0.5
# A durable run sets cancel_on_disconnect=False, so reaping is keyed on progress rather than on connectedness. The
# default matches llama_cpp._DEFAULT_FIRST_TOKEN_TIMEOUT_S, the request path's own first-token budget: a lease older
# than that cannot be legitimate prefill, and slow decode is safe at any speed. A century: clear of any real lease, far
# below where integer milliseconds overflow.
_MAX_ENV_SECONDS = 100.0 * 365.0 * 24.0 * 60.0 * 60.0
# The longest admission keep-alive cadence worth deriving a lease from. A day already means the queue never reports,
# and tripling it stays far inside _MAX_ENV_SECONDS.
_MAX_ADMISSION_INTERVAL_SECONDS = 24.0 * 60.0 * 60.0
_LEASE_TIMEOUT_SECONDS = 1200.0
_LEASE_SWEEP_INTERVAL_SECONDS = 60.0
_LEASE_ERROR = "Generation stopped making progress"


class _SSEDecoder:
    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text.replace("\r\n", "\n")
        values: list[str] = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            data = "\n".join(
                line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
            )
            if data:
                values.append(data)
        return values


def _background_request(app: Any, run_id: str, cancel_event: threading.Event) -> Request:
    from core.helix_engine.optimization_trace import (
        OPTIMIZATION_TRACE_SCOPE_STATE_KEY,
        DurableOptimizationTraceRequest,
    )

    request_state: dict[str, Any] = {"generation_cancel_event": cancel_event}
    trace_request = DurableOptimizationTraceRequest.from_environment(
        enabled=os.environ.get("HELIX_OPTIMIZATION_TRACE_V1"),
        provider_tier=os.environ.get("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER"),
    )
    if trace_request is not None:
        request_state[OPTIMIZATION_TRACE_SCOPE_STATE_KEY] = trace_request
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/chat-runs/producer",
        "raw_path": b"/api/inference/chat-runs/producer",
        "query_string": b"",
        "headers": [
            (b"x-unsloth-generation-run", run_id.encode("ascii", "ignore")),
            # Durable runs replay their event log to the UI, which needs the Unsloth control frames (see routes.inference).
            (b"x-unsloth-events", b"1"),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": request_state,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _bound_optimization_trace_event(
    *,
    run: dict[str, Any],
    owner_subject: str,
    account_id: str,
    envelope,
    error: str | None,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Bind private evidence to the producer-owned row, never to wire claims."""

    if unavailable_reason is not None:
        status = "unavailable"
        trace = None
        error = unavailable_reason
    elif error is not None:
        status = "error"
        trace = None
    elif envelope is None:
        status = "unavailable"
        trace = None
        error = "missing_trace_envelope"
    else:
        status = envelope.status.value
        trace = envelope.to_dict()
        error = envelope.error
    aggregation_reason = None if status == "available" else error or f"trace_status_{status}"
    return {
        "schema_version": "helix.optimization-trace-event.v1",
        "status": status,
        "aggregation": {
            "admissible": status == "available",
            "reason": aggregation_reason,
        },
        "binding": {
            "run_id": str(run["id"]),
            "thread_id": str(run["threadId"]),
            "user_message_id": str(run["userMessageId"]),
            "owner_subject": str(owner_subject),
            "account_id": str(account_id),
        },
        "trace": trace,
        "error": error,
    }


def _chunk_finish_reason(chunk: dict[str, Any]) -> str | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
            return str(choice["finish_reason"])
    return None


def _chunk_error(chunk: dict[str, Any]) -> str | None:
    error = chunk.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("detail") or "Generation failed")[:1000]
    if error:
        return str(error)[:1000]
    return None


async def _close_iterator(iterator: AsyncIterator[Any] | None) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass


def _env_seconds(name: str, default: float) -> float:
    """Seconds from the environment, falling back on anything unusable.

    Non-finite values parse cleanly and fail silently downstream: `inf` raises on
    `int(timeout * 1000)` every sweep, and `max(0.0, nan)` returns 0.0, disabling reaping.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        logger.warning(
            "chat_generation_lease_env_ignored",
            variable = name,
            value = raw,
            reason = "not a finite number",
        )
        return default
    if value > _MAX_ENV_SECONDS:
        # Finite is not usable: past ~1.8e305 the multiply to milliseconds overflows and every sweep raises. Clamped,
        # not rejected, since this already meant "never reap".
        logger.warning(
            "chat_generation_lease_env_clamped",
            variable = name,
            value = raw,
            applied_s = _MAX_ENV_SECONDS,
            reason = "larger than can be converted to integer milliseconds",
        )
        return _MAX_ENV_SECONDS
    return value


async def _sweep_in_daemon_thread(fn, /, *args, **kwargs):
    """Run one blocking sweep on a daemon thread.

    Not asyncio.to_thread: its executor threads are non-daemon and joined by an atexit
    hook, so a sweep parked on SQLite's writer lock keeps the whole process alive long
    after shutdown gave up waiting for it. Studio Desktop allows five seconds for a
    graceful backend exit before force-killing, and stops the backend this way before it
    updates, so an unbounded exit is a user-visible hang. A daemon thread abandoned here
    cannot hold the interpreter open.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _settle(setter, value):
        # The loop can be closed already: this thread outlived the shutdown that abandoned it, which is exactly the case
        # the daemon thread exists to make safe.
        try:
            loop.call_soon_threadsafe(lambda: future.done() or setter(value))
        except RuntimeError:
            pass

    def _runner():
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - relayed to the awaiting caller
            _settle(future.set_exception, exc)
        else:
            _settle(future.set_result, result)

    threading.Thread(target = _runner, name = "chat-lease-sweep", daemon = True).start()
    return await future


def _run_key(run_id: str, account: AccountContext | None = None) -> RunKey:
    """Process-local identity for an account-scoped durable run id."""
    return ((account.account_id if account is not None else current_account_id()), run_id)


class ChatGenerationLeaseSweeper:
    """Periodically settle durable runs whose progress lease has expired.

    reconcile_orphaned_runs used to run exactly once, at process boot, so a run that
    wedged while Studio kept running was never repaired and browser reloads could not
    clear it. This runs the same reconciliation on an interval, bounded to runs that
    have made no progress for the lease timeout so a live generation is never reaped.
    """

    def __init__(
        self,
        app: Any,
        *,
        interval_s: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.app = app
        self._interval = max(
            1.0,
            interval_s
            if interval_s is not None
            else _env_seconds(
                "UNSLOTH_STUDIO_CHAT_RUN_LEASE_SWEEP_INTERVAL_S", _LEASE_SWEEP_INTERVAL_SECONDS
            ),
        )
        # 0 disables the sweep entirely, matching UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S.
        configured = max(
            0.0,
            timeout_s
            if timeout_s is not None
            else _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS),
        )
        self._timeout = _clamped_lease_timeout(configured)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # Grace for a settled producer to notice the cooperative cancel. Generous: unwinding cleanly beats being cancelled
    # mid-teardown, and the run is already declared dead.
    _FORCE_CANCEL_GRACE_S = 30.0

    @property
    def enabled(self) -> bool:
        return self._timeout > 0.0

    def start(self) -> None:
        if self._task is not None or not self.enabled:
            return
        # A second lifespan reuses the instance parked on app.state, and stop() left the event set. Recreated rather
        # than cleared, because the second lifespan can also be a different event loop (repeated TestClient contexts, an
        # embedded server restart), and an asyncio.Event stays bound to the loop it was made on: clearing it would leave
        # the new task failing its first wait with "bound to a different event loop", silently disabling reaping for
        # that whole lifespan.
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name = "chat-generation-lease-sweeper")

    async def _run(self) -> None:
        while True:
            waiter = asyncio.ensure_future(self._stop_event.wait())
            done, _pending = await asyncio.wait({waiter}, timeout = self._interval)
            if done:
                return
            waiter.cancel()
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # one failed sweep (a locked database, a torn-down home in tests) must not retire the watchdog for the
                # life of the process
                logger.warning("chat_generation_lease_sweep_failed", error = repr(exc))

    async def sweep_once(self) -> list[str]:
        if not self.enabled:
            return []
        settled: list[tuple[Any, str]] = []
        recovered: list[tuple[Any, dict[str, Any]]] = []
        supervisor = getattr(getattr(self.app, "state", None), "chat_generation_supervisor", None)
        stale_before = recovery_stale_before_ms(self._timeout)
        # Deactivated accounts too: their wedged producer never sees the cancel event.
        for account in sweepable_job_accounts():
            if stale_before is not None:
                from core.inference.durable_agent_recovery import (
                    plan_orphaned_runs,
                    requeue_planned_runs,
                )

                plans = await _sweep_in_daemon_thread(run_as, account, plan_orphaned_runs)
                if supervisor is not None:
                    # A task owned by this supervisor is handled by the ordinary
                    # stale-run cancellation path below. Restart replay is for an
                    # orphaned owner (for example another backend process), not a
                    # second task beside a wedged local engine call.
                    plans = [
                        plan
                        for plan in plans
                        if _run_key(plan.run_id, account)
                        not in getattr(supervisor, "_tasks", {})
                    ]
                recovered.extend(
                    (account, run)
                    for run in await _sweep_in_daemon_thread(
                        run_as,
                        account,
                        requeue_planned_runs,
                        plans,
                        stale_before_ms=stale_before,
                    )
                )
            settled.extend(
                (account, run_id)
                for run_id in await _sweep_in_daemon_thread(
                    run_as,
                    account,
                    db.reconcile_runs,
                    error = _LEASE_ERROR,
                    stale_after_ms = int(self._timeout * 1000),
                )
            )
        if not settled and not recovered:
            return []
        if supervisor is not None:
            for account, run in recovered:
                run_id = str(run.get("id") or "")
                if not run_id:
                    continue
                run_as(
                    account,
                    supervisor.start_recovered,
                    run_id,
                    thread_id=str(run.get("threadId") or "") or None,
                    model=str(
                        (run.get("resumeRequestPayload") or run.get("requestPayload") or {}).get(
                            "model"
                        )
                        or ""
                    )
                    or None,
                )
        for account, run_id in settled:
            logger.warning(
                "chat_generation_run_lease_expired",
                run_id = run_id,
                idle_s = round(self._timeout, 1),
            )
            if supervisor is None:
                continue
            # The row is settled, but a producer wedged inside the engine is still holding its slot and activity
            # reservation; cancel unwinds it, bound to the owning account's namespace.
            try:
                run_as(account, supervisor.cancel, run_id)
            except Exception as exc:
                logger.warning(
                    "chat_generation_lease_cancel_failed", run_id = run_id, error = repr(exc)
                )
                continue
            asyncio.create_task(
                self._force_cancel_after_grace(supervisor, run_id, account),
                name = f"chat-generation-lease-force-cancel:{run_id}",
            )
        return [str(run.get("id") or "") for _account, run in recovered] + [
            run_id for _account, run_id in settled
        ]

    async def _force_cancel_after_grace(
        self,
        supervisor: Any,
        run_id: str,
        account: Any = None,
    ) -> None:
        """Escalate from the cooperative cancel to cancelling the producer task.

        supervisor.cancel() only sets a threading.Event, which a producer blocked inside
        next(gen) never looks at, so it goes on holding its activity reservation and engine
        slot. Cancelling the task releases that bookkeeping. It cannot unblock a thread
        already inside the engine, and the warning says so rather than implying otherwise.
        """
        await asyncio.sleep(self._FORCE_CANCEL_GRACE_S)
        task = getattr(supervisor, "_tasks", {}).get(_run_key(run_id, account))
        if task is None or task.done():
            return
        logger.warning(
            "chat_generation_run_force_cancelled",
            run_id = run_id,
            grace_s = self._FORCE_CANCEL_GRACE_S,
            note = "producer ignored the cooperative cancel; any engine thread it left "
            "blocked cannot be reclaimed from here",
        )
        task.cancel()

    async def stop(self) -> None:
        self._stop_event.set()
        task, self._task = self._task, None
        if task is None:
            return
        # A short wait, not the producer grace. Letting a sweep finish at shutdown buys nothing: the boot reconcile
        # settles every active run anyway, while a sweep parked on the writer lock would otherwise spend Studio
        # Desktop's whole graceful-exit budget before producers are even signalled. asyncio.wait, never
        # wait_for(gather(...)); see ChatGenerationSupervisor.stop.
        _done, pending = await asyncio.wait({task}, timeout = _SWEEP_SHUTDOWN_SECONDS)
        if not pending:
            return
        task.cancel()
        _done, pending = await asyncio.wait({task}, timeout = _SHUTDOWN_CANCEL_SECONDS)
        if pending:
            logger.warning(
                "The chat generation lease sweeper did not stop within the shutdown budget"
            )


def start_lease_sweeper(app: Any) -> ChatGenerationLeaseSweeper | None:
    """Attach one lease sweeper to the app and start it. Idempotent per app."""
    state = getattr(app, "state", None)
    sweeper = getattr(state, "chat_generation_lease_sweeper", None)
    if sweeper is None:
        sweeper = ChatGenerationLeaseSweeper(app)
        if state is not None:
            state.chat_generation_lease_sweeper = sweeper
    sweeper.start()
    return sweeper


# The admission stream's own comment, matched rather than imported to keep this module free of a routes import at
# module scope. Pinned by a test against the constant there.
_ADMISSION_WAIT_MARKER = ": admission-wait"
# Leaving the queue. Renewed unconditionally: wait renewals are rate limited, and the lease equals the first-token
# timeout, so any age carried in is negative margin.
_ADMISSION_DONE_MARKER = ": admission-done"


def _requires_durable_barrier(chunk: dict[str, Any]) -> bool:
    """Whether upstream must stay suspended until this control frame is committed.

    ``tool_start`` is the critical effect boundary: advancing the underlying
    generator may begin approval/execution. ``tool_end`` keeps the result durable
    before the model can consume it, and ``adaptive_checkpoint`` preserves its
    established safe-boundary ordering.
    """

    return str(chunk.get("type") or "") in {
        "tool_start",
        "tool_end",
        "adaptive_checkpoint",
    }


def _restart_recoverable_run(run: dict[str, Any] | None) -> bool:
    """Whether process shutdown may leave this run for receipt-driven restart.

    The v3 finalization key is the version marker: older durable rows never had a
    tool-effect journal, so preserving them across process death would make their
    side effects unknowable. Explicit Stop is checked by the caller/run row and
    remains terminal.
    """
    if not isinstance(run, dict):
        return False
    run_id = str(run.get("id") or "")
    payload = run.get("requestPayload")
    return bool(
        run_id
        and isinstance(payload, dict)
        and str(payload.get("finalization_idempotency_key") or "").strip() == run_id
        and run.get("cancelRequested") is not True
        and run.get("status") != "cancelling"
    )


def _record_context_governor_checkpoint(model_id: str, checkpoint: dict[str, Any]) -> None:
    """Feed one exact adaptive-boundary observation back into the active machine profile."""
    try:
        context_length = int(checkpoint.get("context_length") or 0)
        occupancy = int(checkpoint.get("occupancy_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        return
    if context_length <= 0:
        return
    try:
        from core.inference.runtime_context_governor import record_active_observation

        record_active_observation(
            model_id=model_id,
            context_length=context_length,
            occupancy_tokens=occupancy or None,
            qualified_boundary=True,
        )
    except Exception:
        # Empirical profiling must never affect the generation/checkpoint path.
        return


def _minimum_lease_seconds() -> float:
    """The shortest lease every renewal source can keep up with.

    The admission keep-alive interval is upstream and not ours to speed up, so a queued run
    only produces a renewable marker that often. A shorter lease expires between markers
    however fast we poll, reaping a healthy queued run.
    """
    from core.inference.llama_admission import DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S

    try:
        from core.inference.llama_admission import llama_admission_config_from_env
        interval = float(llama_admission_config_from_env().keepalive_interval_s)
    except Exception:
        interval = float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)
    # That parser is not ours and only checks the value is positive, so `inf` arrives intact and makes the applied lease
    # infinite, which the sweeper cannot convert to milliseconds. An oversized finite cadence stretches it past any
    # horizon instead.
    if not math.isfinite(interval) or interval > _MAX_ADMISSION_INTERVAL_SECONDS:
        logger.warning(
            "chat_generation_admission_cadence_ignored",
            value = interval,
            applied_s = DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S,
            reason = "not a usable keep-alive cadence",
        )
        interval = float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)
    return max(1.0, interval) * 3.0


def _applied_lease_timeout(configured: float) -> float:
    """The lease actually in force, without the warning. Zero still means disabled.

    Separate from the logging wrapper because the renewal cadence consults this on every
    keep-alive, and warning once per keep-alive would bury the one line that matters.
    """
    if configured <= 0.0:
        return 0.0
    return max(configured, _minimum_lease_seconds())


def _clamped_lease_timeout(configured: float) -> float:
    """Raise a lease that no renewal source could satisfy, and say so.

    Silently honouring it would reap healthy queued runs, and silently ignoring it would
    hide that the setting did nothing. Zero still means disabled.
    """
    applied = _applied_lease_timeout(configured)
    if applied == configured:
        return applied
    logger.warning(
        "chat_generation_lease_timeout_clamped",
        configured_s = round(configured, 2),
        applied_s = round(applied, 2),
        reason = "shorter than the admission keep-alive cadence could renew",
    )
    return applied


def _renew_interval_seconds() -> float:
    """How often a lease may be renewed by something other than streamed output.

    Derived from the lease rather than fixed: a fixed 30s against a shorter lease would
    first renew after the sweep had already settled the run. A quarter gives three renewals
    per window, and the floor keeps a short lease from becoming a write per second.
    """
    lease = _applied_lease_timeout(
        _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS)
    )
    if lease <= 0.0:  # sweeping disabled, so cadence only controls write volume
        return 30.0
    # The floor must stay UNDER the lease: a one second floor against a one second lease first renews no earlier than
    # expiry. A quarter keeps three renewals per window.
    return min(30.0, max(0.25, lease / 4.0))


def recovery_stale_before_ms(timeout_s: float | None = None) -> int | None:
    """Lease cutoff that constitutes positive orphan proof for restart replay."""
    configured = (
        timeout_s
        if timeout_s is not None
        else _env_seconds("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", _LEASE_TIMEOUT_SECONDS)
    )
    applied = _clamped_lease_timeout(max(0.0, float(configured)))
    if applied <= 0.0:
        return None
    return db.now_ms() - int(applied * 1000)


class ChatGenerationSupervisor:
    def __init__(self, app: Any) -> None:
        self.app = app
        self._tasks: dict[RunKey, asyncio.Task] = {}
        self._finalization_tasks: dict[RunKey, asyncio.Task] = {}
        self._recovery_tasks: dict[RunKey, asyncio.Task] = {}
        self._cancel_events: dict[RunKey, threading.Event] = {}
        self._active_registrations: dict[RunKey, active_generations.ActiveGeneration] = {}
        self._activities: dict[RunKey, InferenceActivityReservation] = {}
        self._account_contexts: dict[RunKey, AccountContext] = {}
        self._shutdown_runs: set[RunKey] = set()
        self._stopping = False

    def _remember_account(self, key: RunKey) -> None:
        self._account_contexts.setdefault(key, current_account())

    def _forget_account_if_idle(self, key: RunKey) -> None:
        if any(
            key in collection
            for collection in (
                self._tasks,
                self._finalization_tasks,
                self._recovery_tasks,
                self._cancel_events,
                self._active_registrations,
                self._activities,
                self._shutdown_runs,
            )
        ):
            return
        self._account_contexts.pop(key, None)

    def _ensure_reservation(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model: str | None = None,
    ) -> bool:
        key = _run_key(run_id)
        self._remember_account(key)
        if self._stopping:
            return False
        cancel_event = self._cancel_events.get(key)
        if cancel_event is not None:
            with active_generations.ActiveGeneration(
                cancel_event,
                run_id = run_id,
                thread_id = thread_id,
                model = model,
            ):
                pass
            return True
        cancel_event = threading.Event()
        activity = InferenceActivityReservation()
        activity.reserve()
        registration = active_generations.ActiveGeneration(
            cancel_event,
            run_id = run_id,
            thread_id = thread_id,
            model = model,
        )
        registration.__enter__()
        self._cancel_events[key] = cancel_event
        self._activities[key] = activity
        self._active_registrations[key] = registration
        return True

    def start(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model: str | None = None,
    ) -> None:
        key = _run_key(run_id)
        self._remember_account(key)
        if (
            self._stopping
            or key in self._tasks
            or not self._ensure_reservation(run_id, thread_id = thread_id, model = model)
        ):
            return
        cancel_event = self._cancel_events[key]
        activity = self._activities[key]
        registration = self._active_registrations[key]
        registration.bind()
        producer = self._produce(run_id, cancel_event, activity)
        task: asyncio.Task | None = None
        try:
            task = asyncio.create_task(
                producer,
                name = f"chat-generation-{run_id}",
            )
            self._tasks[key] = task
            task.add_done_callback(
                lambda completed, run_key=key, rid=run_id: self._task_done(
                    run_key, rid, completed
                )
            )
        except BaseException:
            if task is None:
                producer.close()
            else:
                self._tasks.pop(key, None)
                task.cancel()
            self._cleanup_registration(run_id, key=key)
            raise
        finally:
            # create_task and add_done_callback both copy the bound generation
            # epoch. Restore the request Context before returning to the route.
            registration.unbind()

    def start_recovered(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        model: str | None = None,
        expected_worker_token: str | None = None,
        recovery_action: str = "resume_model",
        approval_id: str | None = None,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        """Resume one crash-recovered run without making its model load self-deadlock.

        A normal run arrives only after the frontend has made its model resident, so
        :meth:`start` can register generation ownership immediately.  After a full
        Desktop/backend restart no model is resident. Registering the recovered run
        first would make the ordinary load gate see an active generation and refuse
        the very load recovery needs. Keep the row queued, rehydrate the exact local
        model through the normal load gate, and only then acquire generation
        ownership and continue the durable log.
        """

        key = _run_key(run_id)
        self._remember_account(key)
        if self._stopping or key in self._tasks:
            return
        if not expected_worker_token:
            return
        try:
            task = asyncio.create_task(
                self._rehydrate_and_produce(
                    run_id,
                    thread_id=thread_id,
                    model=model,
                    expected_worker_token=expected_worker_token,
                    recovery_action=recovery_action,
                    approval_id=approval_id,
                    resume_checkpoint=resume_checkpoint,
                ),
                name=f"chat-generation-recovery-{run_id}",
            )
        except BaseException:
            raise
        self._tasks[key] = task
        task.add_done_callback(
            lambda completed, run_key=key, rid=run_id: self._task_done(
                run_key, rid, completed
            )
        )

    def schedule_recovery_plans(
        self,
        plans: list[Any],
        *,
        timeout_s: float = _LEASE_TIMEOUT_SECONDS,
    ) -> None:
        """Recover fresh crash orphans after a lease even when sweeping is disabled."""
        timeout = _applied_lease_timeout(max(0.001, float(timeout_s)))
        for plan in plans:
            run_id = str(getattr(plan, "run_id", "") or "")
            key = _run_key(run_id)
            if (
                not run_id
                or getattr(plan, "safe", False) is not True
                or getattr(plan, "request_payload", None) is None
                or key in self._recovery_tasks
                or key in self._tasks
            ):
                continue
            self._remember_account(key)
            task = asyncio.create_task(
                self._recover_generation_after_lease(plan, timeout),
                name=f"chat-generation-deferred-recovery:{run_id}",
            )
            self._recovery_tasks[key] = task
            task.add_done_callback(
                lambda completed, run_key=key, rid=run_id: self._recovery_done(
                    run_key, rid, completed
                )
            )

    def _recovery_done(self, key: RunKey, run_id: str, task: asyncio.Task) -> None:
        self._recovery_tasks.pop(key, None)
        if task.cancelled():
            self._forget_account_if_idle(key)
            return
        try:
            task.result()
        except Exception as exc:
            logger.error("Durable chat recovery watcher %s crashed: %s", run_id, exc)
        self._forget_account_if_idle(key)

    async def _recover_generation_after_lease(self, plan: Any, timeout_s: float) -> None:
        from core.inference.durable_agent_recovery import (
            plan_orphaned_runs,
            requeue_planned_runs,
        )

        timeout_ms = max(1, int(timeout_s * 1000))
        current = plan
        while not self._stopping:
            progress_at = getattr(current, "expected_progress_at", None)
            if progress_at is None:
                return
            delay = max(0.0, (int(progress_at) + timeout_ms - db.now_ms()) / 1000.0)
            if delay:
                await asyncio.sleep(delay + 0.01)
            stale_before = db.now_ms() - timeout_ms
            recovered = await asyncio.to_thread(
                requeue_planned_runs,
                [current],
                stale_before_ms=stale_before,
            )
            if recovered:
                run = recovered[0]
                self.start_recovered(
                    str(run.get("id") or ""),
                    thread_id=str(run.get("threadId") or "") or None,
                    model=str(
                        (run.get("resumeRequestPayload") or run.get("requestPayload") or {}).get(
                            "model"
                        )
                        or ""
                    )
                    or None,
                    expected_worker_token=str(run.get("_workerToken") or ""),
                    recovery_action=str(run.get("_recoveryAction") or "resume_model"),
                    approval_id=str(run.get("_approvalId") or "") or None,
                    resume_checkpoint=run.get("_resumeCheckpoint"),
                )
                return
            refreshed = await asyncio.to_thread(plan_orphaned_runs)
            current = next(
                (candidate for candidate in refreshed if candidate.run_id == current.run_id),
                None,
            )
            if current is None:
                return
            if not current.safe or current.request_payload is None:
                await asyncio.to_thread(db.reconcile_runs, stale_after_ms=timeout_ms)
                return

    async def _rehydrate_recovery_model(
        self,
        run_id: str,
        *,
        model: str,
        owner: str,
    ) -> None:
        """Restore a recovered run's prior local model without generic auto-switch.

        Recovery is server-owned and must work with the user-facing OpenAI
        auto-switch preference disabled (its default). It may load only when no
        different foreground model has become resident in the meantime; background
        recovery never evicts a user's newer choice.
        """

        from routes import inference as inference_routes

        loaded = await asyncio.to_thread(inference_routes._loaded_slot_ident)
        if loaded is not None:
            if inference_routes._same_loaded_identifier(loaded, model):
                return
            raise RuntimeError(
                "A different foreground model became resident before durable recovery could resume"
            )

        from core.inference.openai_auto_download import looks_like_quant, split_model_ref
        from models.inference import LoadRequest
        from utils.openai_auto_switch_settings import (
            model_override_load_kwargs,
            resolve_override_for_load,
        )

        base_model, maybe_variant = split_model_ref(model)
        variant = maybe_variant if looks_like_quant(maybe_variant) else None
        load_id = base_model if variant else model
        is_gguf_hint = bool(
            variant or load_id.lower().endswith(".gguf") or "gguf" in load_id.lower()
        )
        _override_key, override = resolve_override_for_load(load_id, model, variant)
        load_kwargs = model_override_load_kwargs(override, is_gguf=is_gguf_hint)
        load_kwargs.update(
            {
                "model_path": load_id,
                "gguf_variant": variant,
                "force_reload": False,
                "force_cancel_active": False,
            }
        )
        load_request = LoadRequest.model_validate(load_kwargs)
        await inference_routes.load_model_gated(
            load_request,
            _background_request(self.app, f"{run_id}-model-recovery", threading.Event()),
            owner,
            user_initiated=False,
        )
        loaded = await asyncio.to_thread(inference_routes._loaded_slot_ident)
        if loaded is None or not inference_routes._same_loaded_identifier(loaded, model):
            raise RuntimeError("Durable recovery model load completed without restoring the target")

    async def _apply_recovery_output_budget(
        self,
        run_id: str,
        worker_token: str,
        *,
        original_payload: dict[str, Any],
        resume_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Preserve one interrupted assistant segment's original output budget.

        A crash restart uses ``continue_final_message`` only when it resumes the same
        trailing assistant segment. Reusing the original max token limit there would
        grant a second full allowance after every process restart. Once the model is
        resident we can price the persisted assistant prefill with its own tokenizer:
        count the same one-message prefill with and without its text and subtract the
        delta from the immutable request's original limit.

        ``None`` means the original budget was already exhausted, so the caller should
        settle the run as a normal length stop without asking the model for another
        token. Tool-result recovery deliberately skips this: it begins a new model
        round and keeps the tool loop's historical per-round max-token semantics.
        """

        if resume_payload.get("continue_final_message") is not True:
            return resume_payload
        messages = resume_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return resume_payload
        trailing = messages[-1]
        if not isinstance(trailing, dict) or trailing.get("role") != "assistant":
            return resume_payload
        content = trailing.get("content")
        if not isinstance(content, str) or not content:
            return resume_payload

        limit_field = None
        original_limit = original_payload.get("max_completion_tokens")
        if isinstance(original_limit, int) and original_limit > 0:
            limit_field = "max_completion_tokens"
        else:
            original_limit = original_payload.get("max_tokens")
            if isinstance(original_limit, int) and original_limit > 0:
                limit_field = "max_tokens"
        if limit_field is None:
            return resume_payload

        full_messages = copy.deepcopy(messages)
        empty_messages = copy.deepcopy(messages)
        empty_messages[-1] = {**empty_messages[-1], "content": ""}
        tools = resume_payload.get("tools")
        if not isinstance(tools, list):
            tools = None
        enable_thinking = resume_payload.get("enable_thinking")
        reasoning_effort = resume_payload.get("reasoning_effort")
        preserve_thinking = resume_payload.get("preserve_thinking")

        # Backend singletons are owned by the inference route module: llama.cpp
        # keeps a route-level lazy singleton, while the non-GGUF orchestrator is
        # reached through the same accessor the public endpoints use.
        from routes import inference as inference_routes

        llama = inference_routes.get_llama_cpp_backend()
        if llama.is_loaded:
            def _count(value: list[dict[str, Any]]) -> int:
                return int(
                    llama.count_chat_tokens(
                        value,
                        None,
                        tools,
                        strict=True,
                        chat_template_kwargs={
                            "enable_thinking": enable_thinking,
                            "reasoning_effort": reasoning_effort,
                        },
                        continue_final_message=True,
                    )
                )
        else:
            backend = await asyncio.to_thread(inference_routes.get_inference_backend)

            def _count(value: list[dict[str, Any]]) -> int:
                counted, _model = backend.count_chat_tokens(
                    value,
                    "",
                    tools=tools,
                    enable_thinking=enable_thinking if isinstance(enable_thinking, bool) else None,
                    reasoning_effort=(
                        reasoning_effort if isinstance(reasoning_effort, str) else None
                    ),
                    preserve_thinking=(
                        preserve_thinking if isinstance(preserve_thinking, bool) else None
                    ),
                )
                return int(counted)

        try:
            # Both inference backends serialize token counting with the same
            # single-model generation/count lock. Run these sequentially: two
            # concurrent exact counts would make the second fail with "Cannot
            # count tokens while a generation is in progress" and silently
            # disable the budget correction on every real recovery.
            full_count = await asyncio.to_thread(_count, full_messages)
            empty_count = await asyncio.to_thread(_count, empty_messages)
        except BaseException as exc:
            # Correctness beats an approximate subtraction. If exact tokenizer
            # accounting is unavailable, leave the old behavior but make the loss
            # of budget precision observable in the durable log.
            await asyncio.to_thread(
                db.append_events,
                run_id,
                worker_token,
                [
                    (
                        "run.recovery_budget_unavailable",
                        {"reason": f"{type(exc).__name__}: {exc}"[:500]},
                    )
                ],
            )
            return resume_payload
        partial_tokens = max(0, int(full_count) - int(empty_count))
        remaining = int(original_limit) - partial_tokens
        await asyncio.to_thread(
            db.append_events,
            run_id,
            worker_token,
            [
                (
                    "run.recovery_budget",
                    {
                        "field": limit_field,
                        "original": int(original_limit),
                        "already_emitted": partial_tokens,
                        "remaining": max(0, remaining),
                    },
                )
            ],
        )
        if remaining <= 0:
            return None
        adjusted = copy.deepcopy(resume_payload)
        adjusted[limit_field] = remaining
        # ChatCompletionRequest treats max_completion_tokens as authoritative when
        # present; do not leave a larger deprecated max_tokens alongside it.
        if limit_field == "max_completion_tokens" and "max_tokens" in adjusted:
            adjusted["max_tokens"] = min(int(adjusted["max_tokens"]), remaining)
        return adjusted

    async def _rehydrate_and_produce(
        self,
        run_id: str,
        *,
        thread_id: str | None,
        model: str | None,
        expected_worker_token: str | None = None,
        recovery_action: str = "resume_model",
        approval_id: str | None = None,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        worker_run = await asyncio.to_thread(
            db.get_worker_run,
            run_id,
            expected_worker_token,
        )
        if worker_run is None:
            return
        run, owner, worker_token = worker_run
        if resume_checkpoint is not None:
            resume_checkpoint = copy.deepcopy(resume_checkpoint)
            resume_checkpoint["recovery_approval_id"] = approval_id
        if recovery_action == "wait_approval":
            if not approval_id or not isinstance(resume_checkpoint, dict):
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token=worker_token,
                    status="failed",
                    finish_reason="interrupted",
                    error="Durable approval checkpoint is incomplete",
                )
                return
            # Approval state is lightweight; do not load model weights until the
            # original absolute-expiry decision has settled.
            while not self._stopping:
                approval = await asyncio.to_thread(
                    db.get_tool_approval,
                    run_id,
                    approval_id,
                    include_checkpoint=True,
                )
                if approval is None:
                    return
                decision = approval.get("decision") or approval.get("status")
                if decision in {"allow", "deny"}:
                    if decision == "deny":
                        resume_checkpoint["recovered_receipt"] = {"state": "denied"}
                    break
                if int(approval.get("expiresAt") or 0) <= db.now_ms():
                    await asyncio.to_thread(db.expire_tool_approval, run_id, approval_id)
                    resume_checkpoint["recovered_receipt"] = {"state": "denied"}
                    break
                current_run = await asyncio.to_thread(db.get_run, run_id)
                if current_run is None or current_run.get("cancelRequested") is True:
                    return
                await asyncio.sleep(0.5)
            if self._stopping:
                return
        target_model = str(
            model
            or (run.get("resumeRequestPayload") or run.get("requestPayload") or {}).get("model")
            or ""
        ).strip()
        if not target_model:
            await asyncio.to_thread(
                db.finish_run,
                run_id,
                worker_token=worker_token,
                status="failed",
                finish_reason="interrupted",
                error="Durable recovery has no model identity to restore",
            )
            return
        try:
            await self._rehydrate_recovery_model(
                run_id,
                model=target_model,
                owner=owner,
            )
        except asyncio.CancelledError:
            # request_cancel() settles a queued Stop before supervisor.cancel()
            # cancels this task. A shutdown that merely interrupts model recovery
            # leaves the row active for the next process to try again.
            if not self._stopping:
                current = await asyncio.to_thread(db.get_run, run_id)
                if current is not None and current.get("status") in db.ACTIVE_STATUSES:
                    await asyncio.to_thread(
                        db.finish_run,
                        run_id,
                        worker_token=worker_token,
                        status="failed",
                        finish_reason="interrupted",
                        error="Durable recovery model load was interrupted",
                    )
            raise
        except BaseException as exc:
            await asyncio.to_thread(
                db.finish_run,
                run_id,
                worker_token=worker_token,
                status="failed",
                finish_reason="interrupted",
                error=f"Durable recovery could not restore the model: {exc}"[:1000],
            )
            return

        resume_payload = run.get("resumeRequestPayload")
        original_payload = run.get("requestPayload")
        if isinstance(resume_payload, dict) and isinstance(original_payload, dict):
            budgeted = await self._apply_recovery_output_budget(
                run_id,
                worker_token,
                original_payload=original_payload,
                resume_payload=resume_payload,
            )
            if budgeted is None:
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token=worker_token,
                    status="completed",
                    finish_reason="length",
                )
                return
            resume_payload = budgeted

        current_worker = await asyncio.to_thread(db.get_worker_run, run_id, worker_token)
        current = current_worker[0] if current_worker is not None else None
        if (
            current is None
            or current.get("status") not in db.ACTIVE_STATUSES
            or current.get("cancelRequested") is True
            or self._stopping
        ):
            return
        if not self._ensure_reservation(run_id, thread_id=thread_id, model=target_model):
            return
        key = _run_key(run_id)
        registration = self._active_registrations[key]
        registration.bind()
        try:
            await self._produce(
                run_id,
                self._cancel_events[key],
                self._activities[key],
                worker_payload_override=(
                    resume_payload if isinstance(resume_payload, dict) else None
                ),
                expected_worker_token=worker_token,
                resume_checkpoint=resume_checkpoint,
            )
        finally:
            registration.unbind()

    def _cleanup_registration(self, run_id: str, *, key: RunKey | None = None) -> None:
        key = key or _run_key(run_id)
        self._cancel_events.pop(key, None)
        activity = self._activities.pop(key, None)
        if activity is not None:
            activity.finish()
        registration = self._active_registrations.pop(key, None)
        if registration is not None:
            registration.unregister()
        self._forget_account_if_idle(key)

    def _task_done(self, key: RunKey, run_id: str, task: asyncio.Task) -> None:
        registration = self._active_registrations.get(key)
        lifecycle_binding = registration.lifecycle_binding if registration is not None else None
        self._tasks.pop(key, None)
        self._cleanup_registration(run_id, key=key)
        self._shutdown_runs.discard(key)
        if not task.cancelled():
            try:
                task.result()
            except Exception as exc:
                logger.error("Durable chat generation %s crashed: %s", run_id, exc)
        # Generation ownership is fully released before post-answer learning may
        # touch the model or training lifecycle. A non-finalizable run claims no
        # finalizer state, so this is a cheap no-op for errors/Stop/length cuts.
        if not self._stopping:
            if lifecycle_binding is None:
                self._schedule_finalization(run_id, key=key)
            else:
                with active_generations.bind_lifecycle_epoch(*lifecycle_binding):
                    self._schedule_finalization(run_id, key=key)
        self._forget_account_if_idle(key)

    def _schedule_finalization(
        self,
        run_id: str,
        *,
        due_at: int | None = None,
        key: RunKey | None = None,
    ) -> None:
        key = key or _run_key(run_id)
        self._remember_account(key)
        if self._stopping or key in self._finalization_tasks:
            self._forget_account_if_idle(key)
            return
        try:
            task = asyncio.create_task(
                self._finalize_after_due(run_id, due_at),
                name=f"chat-turn-finalization:{run_id}",
            )
        except RuntimeError:
            self._forget_account_if_idle(key)
            return
        self._finalization_tasks[key] = task
        task.add_done_callback(
            lambda completed, run_key=key, rid=run_id: self._finalization_done(
                run_key, rid, completed
            )
        )

    async def _finalize_after_due(
        self,
        run_id: str,
        due_at: int | None,
    ) -> dict[str, Any] | None:
        if due_at is not None:
            delay = max(0.0, (int(due_at) - db.now_ms()) / 1000.0)
            if delay:
                await asyncio.sleep(delay)
        if self._stopping:
            return None
        return await self._finalize(run_id)

    async def _finalize(self, run_id: str) -> dict[str, Any] | None:
        from core.inference.turn_finalizer import finalize_run

        return await finalize_run(run_id, app=self.app)

    def _finalization_done(self, key: RunKey, run_id: str, task: asyncio.Task) -> None:
        self._finalization_tasks.pop(key, None)
        if task.cancelled():
            self._forget_account_if_idle(key)
            return
        try:
            result = task.result()
        except Exception as exc:
            logger.error("Durable turn finalization %s crashed: %s", run_id, exc)
            if not self._stopping:
                self._schedule_finalization_watcher(run_id, key=key)
            self._forget_account_if_idle(key)
            return
        if not self._stopping and isinstance(result, dict):
            status = result.get("finalizationStatus")
            if status == "pending":
                self._schedule_finalization(
                    run_id,
                    due_at=result.get("finalizationNextAttemptAt"),
                    key=key,
                )
            elif status == "running":
                self._schedule_finalization_watcher(
                    run_id,
                    lease_expires_at=result.get("finalizationLeaseExpiresAt"),
                    key=key,
                )
        self._forget_account_if_idle(key)

    def _schedule_finalization_watcher(
        self,
        run_id: str,
        *,
        lease_expires_at: int | None = None,
        key: RunKey | None = None,
    ) -> None:
        key = key or _run_key(run_id)
        self._remember_account(key)
        if self._stopping or key in self._finalization_tasks:
            self._forget_account_if_idle(key)
            return
        try:
            task = asyncio.create_task(
                self._recover_finalization_after_lease(run_id, lease_expires_at),
                name=f"chat-turn-finalization-recovery:{key[0]}:{run_id}",
            )
        except RuntimeError:
            self._forget_account_if_idle(key)
            return
        self._finalization_tasks[key] = task
        task.add_done_callback(
            lambda completed, run_key=key, rid=run_id: self._finalization_done(
                run_key, rid, completed
            )
        )

    def resume_pending_finalizations(self) -> list[str]:
        """Resume pending work and wait out still-live persisted claim leases."""
        recovered = db.requeue_interrupted_finalizations()
        pending = db.list_pending_finalizations()
        for run in pending:
            run_id = str(run.get("id") or "")
            if run_id:
                self._schedule_finalization(
                    run_id,
                    due_at=run.get("finalizationNextAttemptAt"),
                )
        for run_id, lease_expires_at in db.list_running_finalizations():
            self._schedule_finalization_watcher(
                run_id,
                lease_expires_at=lease_expires_at,
            )
        return recovered

    async def _recover_finalization_after_lease(
        self,
        run_id: str,
        lease_expires_at: int | None,
    ) -> dict[str, Any] | None:
        current_expiry = int(lease_expires_at) if lease_expires_at is not None else None
        error_attempts = 0
        while not self._stopping:
            try:
                if current_expiry is not None:
                    delay = max(0.0, (current_expiry - db.now_ms()) / 1000.0)
                    if delay:
                        await asyncio.sleep(delay + 0.01)
                current = await asyncio.to_thread(db.get_run, run_id)
                if not isinstance(current, dict):
                    return current
                status = current.get("finalizationStatus")
                if status == "pending":
                    return await self._finalize_after_due(
                        run_id,
                        current.get("finalizationNextAttemptAt"),
                    )
                if status != "running":
                    return current
                renewed_expiry = current.get("finalizationLeaseExpiresAt")
                if renewed_expiry is None:
                    raise RuntimeError("running finalization has no ownership lease")
                current_expiry = int(renewed_expiry)
                if current_expiry > db.now_ms():
                    error_attempts = 0
                    continue
                recovered = await asyncio.to_thread(
                    db.requeue_interrupted_finalizations,
                    run_id,
                )
                if run_id not in recovered:
                    # A remote owner may have renewed between our read and CAS.
                    current_expiry = None
                    error_attempts = 0
                    continue
                result = await self._finalize(run_id)
                if not isinstance(result, dict) or result.get("finalizationStatus") != "running":
                    return result
                renewed_expiry = result.get("finalizationLeaseExpiresAt")
                current_expiry = int(renewed_expiry) if renewed_expiry is not None else None
                error_attempts = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_attempts += 1
                delay = min(
                    _FINALIZATION_WATCH_RETRY_MAX_SECONDS,
                    _FINALIZATION_WATCH_RETRY_BASE_SECONDS
                    * (2 ** min(error_attempts - 1, 10)),
                )
                logger.warning(
                    "Durable turn finalization watcher %s retrying after error: %s",
                    run_id,
                    exc,
                )
                current_expiry = None
                await asyncio.sleep(max(0.001, delay))
        return None

    def cancel(self, run_id: str) -> None:
        key = _run_key(run_id)
        cancel_event = self._cancel_events.get(key)
        if cancel_event is not None:
            cancel_event.set()
        else:
            task = self._tasks.get(key)
            if task is not None and not task.done():
                task.cancel()
        active_generations.cancel_run(run_id)
        # The inference cancel registry closes the narrow gap where registration is imminent but this supervisor has
        # not yet observed it.
        from routes.inference import _cancel_by_cancel_id_or_stash

        _cancel_by_cancel_id_or_stash(run_id)

    async def stop(self) -> None:
        self._stopping = True
        # before the runs, so the sweeper cannot settle a run as stalled while shutdown is settling it as interrupted
        sweeper = getattr(getattr(self.app, "state", None), "chat_generation_lease_sweeper", None)
        if sweeper is not None:
            await sweeper.stop()
        tasks = list(self._tasks.items())
        self._shutdown_runs.update(key for key, _task in tasks)
        for key, task in tasks:
            account = self._account_contexts.get(key)
            if account is None:
                task.cancel()
            else:
                run_as(account, self.cancel, key[1])
        if not tasks:
            generation_pending: set[asyncio.Task] = set()
        else:
            generation_pending = {task for _run_id, task in tasks}
        # asyncio.wait, not wait_for(gather(...)): on timeout wait_for cancels the inner future and then awaits it, so a
        # producer that does not unwind on cancellation -- an engine draining its subprocess inside the generator's
        # aclose -- makes the wait itself unbounded, and takes the whole uvicorn shutdown down with it. wait returns the
        # pending set instead and leaves those tasks alone.
        if generation_pending:
            _done, generation_pending = await asyncio.wait(
                generation_pending, timeout = _SHUTDOWN_GRACE_SECONDS
            )
            if generation_pending:
                for task in generation_pending:
                    task.cancel()
                _done, generation_pending = await asyncio.wait(
                    generation_pending, timeout = _SHUTDOWN_CANCEL_SECONDS
                )
                if generation_pending:
                    stuck = [key[1] for key, task in tasks if task in generation_pending]
                    # Abandoned, not leaked: the run is already fenced and reconcile_orphaned_runs settles it on the next boot.
                    # Process exit reclaims the rest.
                    logger.warning(
                        "Durable chat generations did not stop within the shutdown budget: %s",
                        ", ".join(stuck),
                    )

        # Finalizers are restartable transactions.  Do not spend the generation
        # shutdown budget waiting for optional learning; cancel them and leave
        # their durable running claims for startup reconciliation.
        finalizers = list(self._finalization_tasks.values())
        for task in finalizers:
            task.cancel()
        if finalizers:
            await asyncio.wait(finalizers, timeout=_SHUTDOWN_CANCEL_SECONDS)
        recovery_tasks = list(self._recovery_tasks.values())
        for task in recovery_tasks:
            task.cancel()
        if recovery_tasks:
            await asyncio.gather(*recovery_tasks, return_exceptions=True)

    # Total time, not a count: the interval derives from the lease, so a count would mean very different durations.
    # Bounded because an unbounded heartbeat would keep a preparation that never returns alive forever, the failure this
    # file exists to end.
    _PREPARE_RENEW_MAX_SECONDS = 2 * 60 * 60

    @contextlib.asynccontextmanager
    async def _lease_heartbeat(self, run_id: str, worker_token: str):
        """Hold the progress lease open across work that produces no output.

        Covers the lifecycle gate as well as model preparation: a run waiting on the gate is
        still queued, so its lease ages from created_at with nothing renewing it. Cancelled
        on exit, including an early return, so it never overlaps the streaming phase.
        """
        task = asyncio.create_task(
            self._renew_lease_while_preparing(run_id, worker_token),
            name = f"chat-generation-prepare-lease:{run_id}",
        )
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _try_touch_progress(
        self,
        run_id: str,
        worker_token: str | None = None,
    ) -> None:
        """Renew the lease, treating contention as a missed stamp rather than a failure.

        Every renewal that is not streamed output goes through here. A history transaction
        holding SQLite's writer lock past the busy timeout would otherwise abort a healthy
        generation; a missed stamp costs one interval instead.
        """
        try:
            args = (run_id,) if worker_token is None else (run_id, worker_token)
            await asyncio.to_thread(db.touch_progress, *args)
        except Exception:
            return

    async def _renew_lease_while_preparing(
        self,
        run_id: str,
        worker_token: str | None = None,
    ) -> None:
        interval = _renew_interval_seconds()
        for _ in range(max(1, int(self._PREPARE_RENEW_MAX_SECONDS / interval))):
            await asyncio.sleep(interval)
            # skip a contended stamp rather than abandon the rest, else a healthy long load is reaped once the last
            # stamp ages out
            await self._try_touch_progress(run_id, worker_token)

    async def _produce(
        self,
        run_id: str,
        cancel_event: threading.Event | None = None,
        activity: InferenceActivityReservation | None = None,
        worker_payload_override: dict[str, Any] | None = None,
        expected_worker_token: str | None = None,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        key = _run_key(run_id)
        cancel_event = cancel_event or threading.Event()
        if activity is None:
            activity = InferenceActivityReservation()
            activity.reserve()
        pending: list[tuple[str, dict[str, Any], int]] = []
        iterator: AsyncIterator[Any] | None = None
        last_flush = time.monotonic()
        finish_reason: str | None = None
        error: str | None = None
        saw_done = False
        worker_token: str | None = None
        next_raw_task: asyncio.Task | None = None
        durable_tool_scope = None
        trace_expected = False
        trace_configuration_error: str | None = None
        trace_provider_tier = None
        trace_envelope = None
        trace_wire_error: str | None = None
        trace_frame_seen = False
        trace_persist_attempted = False
        trace_run: dict[str, Any] | None = None
        trace_owner: str | None = None

        def _is_recovered_trace_run() -> bool:
            return bool(
                expected_worker_token is not None
                or (
                    trace_run is not None
                    and isinstance(trace_run.get("resumeRequestPayload"), dict)
                )
            )

        def _optimization_trace_event(
            terminal_status: str,
        ) -> tuple[str, dict[str, Any], int] | None:
            nonlocal trace_persist_attempted
            if (
                not trace_expected
                or trace_persist_attempted
                or worker_token is None
                or trace_run is None
                or trace_owner is None
            ):
                return None
            trace_persist_attempted = True
            unavailable_reason = None
            if _is_recovered_trace_run():
                # A replacement recorder sees only the latest physical attempt.
                # The durable run log proves recovery occurred, so never expose
                # that suffix as an available whole-run benchmark trace.
                unavailable_reason = "recovered_run_trace_incomplete"
            elif terminal_status != "completed":
                unavailable_reason = "generation_attempt_incomplete"
            try:
                payload = _bound_optimization_trace_event(
                    run=trace_run,
                    owner_subject=trace_owner,
                    account_id=current_account_id(),
                    envelope=trace_envelope,
                    error=trace_configuration_error or trace_wire_error,
                    unavailable_reason=unavailable_reason,
                )
            except Exception:
                # Parsed evidence can still fail while being converted to its
                # bounded stored form.  It is private, best-effort evidence and
                # must never reclassify an otherwise successful generation.
                logger.exception(
                    "optimization_trace.serialization_failed",
                    run_id=run_id,
                )
                return None
            return ("optimization.trace", payload, db.now_ms())

        async def _persist_recovery_trace_exclusion() -> None:
            nonlocal trace_persist_attempted
            if not trace_expected or not _is_recovered_trace_run():
                return
            trace_event = _optimization_trace_event("completed")
            if trace_event is None:
                return
            try:
                await asyncio.to_thread(
                    db.append_events,
                    run_id,
                    worker_token,
                    [trace_event],
                )
            except Exception:
                # Keep generation semantics unchanged and let the terminal
                # transaction retry the same conservative exclusion.
                trace_persist_attempted = False
                logger.exception(
                    "optimization_trace.recovery_exclusion_persistence_failed",
                    run_id=run_id,
                )

        async def _finish_with_trace(
            *,
            status: str,
            finish_reason_value: str | None,
            error_value: str | None,
        ) -> None:
            nonlocal pending
            ordinary_events = pending
            trace_event = _optimization_trace_event(status)
            terminal_events = ordinary_events + ([trace_event] if trace_event else [])
            try:
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token=worker_token,
                    status=status,
                    finish_reason=finish_reason_value,
                    error=error_value,
                    pending_events=terminal_events,
                )
            except Exception:
                if trace_event is None:
                    raise
                # Trace publication is best effort.  Retrying the fenced,
                # idempotent terminal transaction without it preserves the
                # user-visible result if private evidence cannot be stored.
                logger.exception(
                    "optimization_trace.persistence_failed",
                    run_id=run_id,
                )
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token=worker_token,
                    status=status,
                    finish_reason=finish_reason_value,
                    error=error_value,
                    pending_events=ordinary_events,
                )
            pending = []
        try:
            worker_run = await asyncio.to_thread(
                db.get_worker_run,
                run_id,
                expected_worker_token,
            )
            if worker_run is None:
                return
            run, owner, worker_token = worker_run
            if cancel_event.is_set():
                shutting_down = key in self._shutdown_runs
                if shutting_down and _restart_recoverable_run(run):
                    # No side effect has been admitted after the last durable
                    # receipt. Keep the row active so startup can build a
                    # receipt-driven continuation instead of turning Quit into
                    # an artificial failed answer.
                    return
                await asyncio.to_thread(
                    db.finish_run,
                    run_id,
                    worker_token = worker_token,
                    status = "failed" if shutting_down else "cancelled",
                    finish_reason = "interrupted" if shutting_down else "cancelled",
                    error = "Studio shut down during generation" if shutting_down else None,
                )
                return
            # Spans the lifecycle gate as well as preparation: a run waiting on the gate is still queued, so its lease
            # ages from created_at with nothing renewing it.
            async with self._lease_heartbeat(run_id, worker_token):
                await activity.start(cancel_event)
                if cancel_event.is_set():
                    shutting_down = key in self._shutdown_runs
                    if shutting_down and _restart_recoverable_run(run):
                        return
                    await asyncio.to_thread(
                        db.finish_run,
                        run_id,
                        worker_token = worker_token,
                        status = "failed" if shutting_down else "cancelled",
                        finish_reason = "interrupted" if shutting_down else "cancelled",
                        error = "Studio shut down during generation" if shutting_down else None,
                    )
                    return
                if not await asyncio.to_thread(db.mark_running, run_id, worker_token):
                    return
                worker_run = await asyncio.to_thread(db.get_worker_run, run_id, worker_token)
                if worker_run is None:
                    return
                run, owner, worker_token = worker_run
                trace_run, trace_owner = run, owner
                if run["status"] != "running" or run["cancelRequested"]:
                    await asyncio.to_thread(
                        db.finish_run,
                        run_id,
                        worker_token = worker_token,
                        status = "cancelled",
                    )
                    return

                from core.inference.durable_tool_journal import durable_tool_run

                durable_tool_scope = durable_tool_run(
                    run_id,
                    worker_token,
                    resume_checkpoint=resume_checkpoint,
                )
                durable_tool_scope.__enter__()

                from routes.inference import produce_openai_chat_completions

                worker_payload = (
                    worker_payload_override
                    or run.get("resumeRequestPayload")
                    or run["requestPayload"]
                )
                payload = ChatCompletionRequest.model_validate(worker_payload)
                # Switching, idle reload and auto-download all happen in the call below, and llama.cpp's first-token
                # budget only starts after it. One touch afterwards cannot cover a preparation longer than the lease
                # itself.
                background_request = _background_request(self.app, run_id, cancel_event)
                from core.helix_engine.optimization_trace import (
                    OPTIMIZATION_TRACE_SCOPE_STATE_KEY,
                    DurableOptimizationTraceRequest,
                )

                trace_request = getattr(
                    background_request.state,
                    OPTIMIZATION_TRACE_SCOPE_STATE_KEY,
                    None,
                )
                trace_expected = isinstance(trace_request, DurableOptimizationTraceRequest)
                if trace_expected:
                    trace_configuration_error = trace_request.configuration_error
                    trace_provider_tier = trace_request.provider_tier
                await _persist_recovery_trace_exclusion()
                response = await produce_openai_chat_completions(
                    payload,
                    background_request,
                    owner,
                    cancel_on_disconnect = False,
                )
            await self._try_touch_progress(run_id, worker_token)
            if int(getattr(response, "status_code", 200)) >= 400:
                raise RuntimeError(f"Local generation returned HTTP {response.status_code}")
            iterator = getattr(response, "body_iterator", None)
            if iterator is None:
                raise RuntimeError("Local generation did not return an event stream")
            decoder = _SSEDecoder()
            from core.helix_engine.optimization_trace import (
                OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY,
                parse_optimization_trace_envelope,
            )

            private_trace_sentinel = object()
            last_keepalive = time.monotonic()
            next_raw_task = asyncio.create_task(iterator.__anext__())
            while True:
                timeout = (
                    max(
                        0.0,
                        (
                            _EVENT_BATCH_SECONDS
                            if len(pending) >= _EVENT_BATCH_MIN_SIZE
                            else _EVENT_SINGLE_FLUSH_SECONDS
                        )
                        - (time.monotonic() - last_flush),
                    )
                    if pending
                    else None
                )
                ready, _waiting = await asyncio.wait({next_raw_task}, timeout = timeout)
                if not ready:
                    await asyncio.to_thread(
                        db.append_events,
                        run_id,
                        worker_token,
                        pending,
                    )
                    pending = []
                    last_flush = time.monotonic()
                    continue
                try:
                    raw = next_raw_task.result()
                except StopAsyncIteration:
                    next_raw_task = None
                    break
                text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
                # Admission comments are progress; plain keep-alives are not. A queued run only emits `:
                # admission-wait`, which _SSEDecoder drops, so nothing renewed the lease and a healthy queue reaped its
                # own runs. `: keep-alive` is the opposite signal, emitted when the generator has produced NOTHING, so
                # renewing on any byte would keep a wedged run alive forever. Rate limited because chunk traffic already
                # renews through append_events.
                if _ADMISSION_DONE_MARKER in text:
                    last_keepalive = time.monotonic()
                    await self._try_touch_progress(run_id, worker_token)
                elif _ADMISSION_WAIT_MARKER in text:
                    now_s = time.monotonic()
                    if now_s - last_keepalive >= _renew_interval_seconds():
                        last_keepalive = now_s
                        await self._try_touch_progress(run_id, worker_token)
                durable_barrier = False
                tool_proposal: dict[str, Any] | None = None
                governor_checkpoints: list[dict[str, Any]] = []
                for encoded in decoder.feed(text):
                    if encoded == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(encoded)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    private_proposal = chunk.pop("_durable_tool_approval", None)
                    if private_proposal is not None:
                        if not isinstance(private_proposal, dict) or tool_proposal is not None:
                            raise RuntimeError("invalid durable tool proposal envelope")
                        tool_proposal = private_proposal
                    private_trace = chunk.pop(
                        OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY,
                        private_trace_sentinel,
                    )
                    if private_trace is not private_trace_sentinel:
                        if trace_expected:
                            if trace_frame_seen:
                                trace_envelope = None
                                trace_wire_error = "duplicate_trace_envelope"
                            else:
                                trace_frame_seen = True
                                if trace_configuration_error is not None:
                                    trace_wire_error = trace_configuration_error
                                else:
                                    try:
                                        trace_envelope = parse_optimization_trace_envelope(
                                            private_trace
                                        )
                                        from core.helix_engine.optimization_benchmark import (
                                            ExecutionScope,
                                        )

                                        if any(
                                            event.scope is not ExecutionScope.FOREGROUND
                                            or event.provider_identity != "local:mlx"
                                            or event.provider_tier is not trace_provider_tier
                                            for event in trace_envelope.invocation_events
                                        ):
                                            trace_envelope = None
                                            trace_wire_error = "wrong_trace_configuration"
                                    except Exception:
                                        trace_wire_error = "malformed_trace_envelope"
                        # The transport frame is never a public durable chunk.
                        if not chunk:
                            continue
                    pending.append(("chunk", chunk, db.now_ms()))
                    durable_barrier = durable_barrier or _requires_durable_barrier(chunk)
                    if chunk.get("type") == "adaptive_checkpoint":
                        governor_checkpoints.append(chunk)
                    finish_reason = _chunk_finish_reason(chunk) or finish_reason
                    error = _chunk_error(chunk) or error
                    now = time.monotonic()
                    if len(pending) >= _EVENT_BATCH_SIZE or (
                        len(pending) >= _EVENT_BATCH_MIN_SIZE
                        and now - last_flush >= _EVENT_BATCH_SECONDS
                    ):
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                        pending = []
                        last_flush = now
                if saw_done:
                    break
                if durable_barrier and pending:
                    # Keep the source generator suspended at this yield until
                    # the effect boundary is durable. Advancing a tool loop past
                    # tool_start can begin approval or filesystem/process work.
                    if tool_proposal is not None:
                        await asyncio.to_thread(
                            db.append_events_with_tool_proposal,
                            run_id,
                            worker_token,
                            pending,
                            tool_proposal,
                        )
                    else:
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                    pending = []
                    last_flush = time.monotonic()
                for checkpoint in governor_checkpoints:
                    await asyncio.to_thread(
                        _record_context_governor_checkpoint,
                        str(payload.model or ""),
                        checkpoint,
                    )
                next_raw_task = asyncio.create_task(iterator.__anext__())

            current = await asyncio.to_thread(db.get_run, run_id)
            if current is None:
                return
            if key in self._shutdown_runs:
                if _restart_recoverable_run(current):
                    if pending:
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                        pending = []
                    return
                status = "failed"
                finish_reason = "interrupted"
                error = "Studio shut down during generation"
            elif current["cancelRequested"] or (cancel_event.is_set() and error is None):
                # A bare event is not proof of a user stop: the streaming paths set this same event from their cleanup
                # after emitting an in-band error, so a parsed failure outranks it. An explicit cancelRequested still
                # wins, and a real Stop carries no error chunk, so neither loses its identity.
                status = "cancelled"
                finish_reason = "cancelled"
            elif error is not None:
                status = "failed"
            elif not saw_done and finish_reason is None:
                status = "failed"
                finish_reason = "interrupted"
                error = "Generation stream ended before completion"
            else:
                status = "completed"
            # Public chunks, private trace evidence and the terminal event share
            # one fenced transaction.  A process death can therefore leave
            # either the whole terminal record or no trace, never an apparently
            # available trace for an incomplete physical attempt.
            await _finish_with_trace(
                status=status,
                finish_reason_value=finish_reason,
                error_value=error,
            )
        except asyncio.CancelledError:
            if worker_token is not None:
                shutting_down = key in self._shutdown_runs
                current = await asyncio.to_thread(db.get_run, run_id)
                if shutting_down and _restart_recoverable_run(current):
                    if pending:
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                        pending = []
                    raise
                cancelled = cancel_event.is_set() and not shutting_down
                await _finish_with_trace(
                    status="cancelled" if cancelled else "failed",
                    finish_reason_value="cancelled" if cancelled else "interrupted",
                    error_value=(
                        None if cancelled else "Generation worker stopped unexpectedly"
                    ),
                )
            pending = []
            raise
        except Exception as exc:
            if worker_token is not None:
                shutting_down = key in self._shutdown_runs
                current = await asyncio.to_thread(db.get_run, run_id)
                if shutting_down and _restart_recoverable_run(current):
                    if pending:
                        await asyncio.to_thread(
                            db.append_events,
                            run_id,
                            worker_token,
                            pending,
                        )
                        pending = []
                    return
                cancelled = cancel_event.is_set() and not shutting_down
                await _finish_with_trace(
                    status="cancelled" if cancelled else "failed",
                    finish_reason_value=(
                        "cancelled" if cancelled else "interrupted" if shutting_down else "error"
                    ),
                    error_value=(
                        None
                        if cancelled
                        else "Studio shut down during generation"
                        if shutting_down
                        else str(exc)[:1000]
                    ),
                )
            pending = []
        finally:
            # Keep the durable tool scope active while the response iterator
            # closes: an async-generator teardown can still finish a tool
            # receipt. But never let cancellation or iterator cleanup skip the
            # ContextVar reset. If the generator-backed context manager is
            # instead left for GC, Python may finalize it in a different
            # context and ContextVar.reset(token) raises.
            try:
                if next_raw_task is not None:
                    if not next_raw_task.done():
                        next_raw_task.cancel()
                    await asyncio.gather(next_raw_task, return_exceptions = True)
                await _close_iterator(iterator)
            finally:
                try:
                    if durable_tool_scope is not None:
                        durable_tool_scope.__exit__(None, None, None)
                finally:
                    activity.finish()
