# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming wrapper around blocking server-side tool execution.

``stream_tool_execution`` runs a blocking tool call in a worker thread and
turns it into a generator that yields:

* ``{"type": "tool_output", "tool_name", "tool_call_id", "text"}`` -- an
  incremental stdout/stderr chunk (python/terminal tools) for live UI output;
* ``{"type": "heartbeat"}`` -- emitted whenever nothing else has been yielded
  for ``heartbeat_interval_s`` seconds, so the SSE route can write a
  keepalive and reverse proxies (Cloudflare tunnels cap idle streams at
  ~100 s) never see a silent connection while a tool runs;

and *returns* the tool's final result string via ``StopIteration.value``
(``result = yield from stream_tool_execution(...)``). A tool that finishes
within its deadline returns the same result as a direct call. A tool that
outlives the wrapper gets a deterministic timeout result instead, so the agent
can continue without waiting for a cancellation-ignoring callable.
"""

from __future__ import annotations

import contextvars
import inspect
import os
import queue
import threading
import time
from time import monotonic as _deadline_monotonic
from typing import Any, Callable, Generator

from loggers import get_logger

logger = get_logger(__name__)


def accepts_kwarg(func: Callable[..., str], name: str) -> bool:
    """Whether an injectable ``execute_tool`` supports the keyword ``name``.

    ``execute_tool`` is replaceable (tests inject fakes / the pre-PR signature),
    so forward a kwarg only when the callable declares it or takes ``**kwargs``
    (passing it unconditionally would ``TypeError`` on an old signature).
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def accepts_output_callback(func: Callable[..., str]) -> bool:
    return accepts_kwarg(func, "output_callback")


def search_images_kwargs(func: Callable[..., str], tool_name: str) -> dict[str, bool]:
    """``{"search_images": True}`` when web_search should also return images, else ``{}``.

    Read per call rather than per request so the Settings toggle applies to the
    next search without a reload, and only for web_search so other tools never
    pay the settings read.
    """
    if tool_name != "web_search" or not accepts_kwarg(func, "search_images"):
        return {}
    from .search_images import search_images_enabled

    return {"search_images": True} if search_images_enabled() else {}


# Cadence of heartbeat events while a tool blocks with no output. Well under common proxy idle caps (Cloudflare ~100 s,
# nginx default 60 s).
TOOL_HEARTBEAT_INTERVAL_S = 10.0

# delay before the first approval keepalive so it cannot coalesce with the gated card.
TOOL_APPROVAL_FLUSH_DELAY_S = 0.05

# How often the wrapper wakes to poll for output / completion / cancellation.
_POLL_INTERVAL_S = 0.25

# A request may deliberately choose the UI's "Max" timeout sentinel, which the
# tool implementations receive as ``None``.  That may remove a *normal* tool
# deadline, but it must not let a buggy built-in or third-party callable pin the
# agent forever.  This outer watchdog is intentionally generous and configurable;
# finite per-request timeouts below it still win.
try:
    TOOL_HARD_TIMEOUT_S = max(
        1.0, float(os.environ.get("UNSLOTH_TOOL_HARD_TIMEOUT_SECONDS", "1800") or 1800)
    )
except (TypeError, ValueError):
    TOOL_HARD_TIMEOUT_S = 1800.0

# Upper bound on how long teardown waits for the worker once the stream is closed or errors. A cancel-observing tool
# returns within this after ``cancel_event`` is set; a cancel-ignoring one is a daemon left to finish on its own rather
# than blocking teardown for the tool's full timeout.
_WORKER_JOIN_TIMEOUT_S = 5.0
# Give a cancellation-observing callable one scheduler slice to unwind before it
# is classified as an orphan. This is deliberately tiny: disconnect/timeout
# teardown stays bounded even when the callable ignores cancellation entirely.
_CANCEL_SETTLE_TIMEOUT_S = 0.05

# Cap on total streamed live-output characters per tool call, bounding the transient UI stream so a tight print loop
# cannot flood the SSE channel. Much higher than the model-visible result cap (tools._MAX_OUTPUT_CHARS) since the UI
# keeps the live stream as the displayed output when the result is truncated.
TOOL_OUTPUT_STREAM_MAX_CHARS = 400_000

_STREAM_CAPPED_NOTICE = "\n... (further live output not streamed)\n"

_STALLED_TOOLS_LOCK = threading.Lock()
_STALLED_TOOLS: dict[tuple[str, str], object] = {}


def _tool_scope_key(tool_name: str) -> tuple[str, str]:
    try:
        from utils.account_context import current_account_id

        account_id = current_account_id()
    except Exception:
        account_id = "owner"
    return str(account_id), str(tool_name or "unknown")


def _stalled_tool_result(tool_name: str) -> str:
    return (
        f"Error: an earlier timed-out call to tool '{tool_name or 'unknown'}' is still running "
        "in the background because it did not stop when cancelled. Unsloth will not start "
        "another copy until that call exits. Continue with another strategy or the final answer."
    )


def _clear_stalled_tool(key: tuple[str, str], token: object) -> None:
    with _STALLED_TOOLS_LOCK:
        if _STALLED_TOOLS.get(key) is token:
            _STALLED_TOOLS.pop(key, None)


class LinkedToolCancelEvent:
    """Per-tool cancellation linked to the request cancellation event.

    ``set()`` cancels only this tool.  ``is_set()``/``wait()`` also observe the
    parent request event, so Stop/disconnect still reaches the tool.  A hard tool
    deadline can therefore cancel one execution without poisoning the shared
    request event and aborting every later tool/final-answer pass.
    """

    def __init__(self, parent: Any = None) -> None:
        self._local = threading.Event()
        self._parent = parent

    def set(self) -> None:
        self._local.set()

    def is_set(self) -> bool:
        if self._local.is_set():
            return True
        parent = self._parent
        if parent is None:
            return False
        try:
            return bool(parent.is_set())
        except Exception:
            return False

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        if self._parent is None:
            return self._local.wait(timeout)
        deadline = None if timeout is None else _deadline_monotonic() + max(0.0, timeout)
        while not self.is_set():
            if deadline is None:
                wait = 0.05
            else:
                remaining = deadline - _deadline_monotonic()
                if remaining <= 0:
                    return self.is_set()
                wait = min(0.05, remaining)
            self._local.wait(wait)
        return True


def tool_timeout_result(tool_name: str, timeout_s: float) -> str:
    label = tool_name or "tool"
    return (
        f"Error: tool '{label}' exceeded the {timeout_s:g}s wall-clock limit. "
        "Its execution was cancelled. If the tool can change state, its final state is unknown; "
        "inspect that state before deciding whether to retry."
    )


def _drain_queue(q: "queue.Queue", sentinel: object, max_chars: int | None) -> tuple[str, bool]:
    """Pull every currently-queued item, joining chunks in FIFO order.

    With ``max_chars`` set, stop concatenating at the budget and discard the
    remaining chunks in place, bounding peak allocation when a chatty tool queues
    far more than the cap before the consumer wakes. The crossing chunk is sliced
    to one char past the budget, enough for the caller's truncation to stay
    byte-identical. Returns ``(joined_text, hit_sentinel)``; the surplus is still
    scanned so completion is detected promptly.
    """
    parts: list[str] = []
    total = 0
    dropping = False
    hit_sentinel = False
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is sentinel:
            hit_sentinel = True
            break
        if dropping:
            continue
        if max_chars is not None and total + len(item) > max_chars:
            # Keep one char past the budget as the overflow signal; drop the rest.
            parts.append(item[: max(0, max_chars - total) + 1])
            dropping = True
            continue
        parts.append(item)
        total += len(item)
    return "".join(parts), hit_sentinel


def stream_tool_execution(
    invoke: Callable[[Callable[[str], None]], str],
    *,
    tool_name: str,
    tool_call_id: str = "",
    journal_tool_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    session_id: str | None = None,
    thread_id: str | None = None,
    pre_tool_checkpoint: dict[str, Any] | None = None,
    cancel_event: Any = None,
    timeout_s: float | None = None,
    heartbeat_interval_s: float = TOOL_HEARTBEAT_INTERVAL_S,
    poll_interval_s: float = _POLL_INTERVAL_S,
) -> Generator[dict, None, str]:
    """Run ``invoke(output_callback)`` in a thread; yield live events; return the result.

    ``invoke`` receives a thread-safe ``callable(str)`` it may call with
    incremental output chunks (or ignore entirely). If it accepts the optional
    ``tool_cancel_event`` keyword, that event is linked to the request cancel
    signal and is also set by this wrapper's hard deadline. Exceptions raised by
    the tool propagate to the caller unchanged after the worker thread finishes.

    ``cancel_event`` is the request-level cancellation signal. A private linked
    event is set when the consumer closes this generator or when the tool exceeds
    its deadline, so one abandoned tool never sets the shared request event.

    ``timeout_s`` is the caller's requested wall limit. ``None`` still receives
    the process safety ceiling ``TOOL_HARD_TIMEOUT_S``; otherwise the smaller of
    the requested limit and that ceiling wins.
    """
    output_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()
    outcome: dict[str, Any] = {}
    stall_key = _tool_scope_key(tool_name)
    with _STALLED_TOOLS_LOCK:
        if stall_key in _STALLED_TOOLS:
            return _stalled_tool_result(tool_name)
    stall_token = object()
    execution_cancel = LinkedToolCancelEvent(cancel_event)
    requested_timeout = None
    if timeout_s is not None:
        try:
            requested_timeout = max(0.001, float(timeout_s))
        except (TypeError, ValueError):
            requested_timeout = None
    hard_timeout_s = (
        TOOL_HARD_TIMEOUT_S
        if requested_timeout is None
        else min(requested_timeout, TOOL_HARD_TIMEOUT_S)
    )
    deadline = _deadline_monotonic() + hard_timeout_s

    # Durable agent runs establish an execution receipt before the worker may
    # begin. Outside a durable run these helpers are strict no-ops, preserving
    # the ordinary OpenAI/legacy tool path.
    from core.inference.durable_tool_journal import (
        DurableExecutionHandle,
        claim_execution,
        prepared_result,
        record_execution_finished,
        record_execution_started,
    )
    from core.inference.tool_producer_receipt import receipt_dict

    claim_execution(
        tool_name,
        journal_tool_call_id or tool_call_id,
        card_call_id=tool_call_id,
        arguments=arguments,
        session_id=session_id,
        thread_id=thread_id,
        pre_tool_checkpoint=pre_tool_checkpoint,
    )

    # Bound accepted output at the PRODUCER boundary: the consumer-side cap alone wouldn't stop a fast worker
    # enqueuing unboundedly while a slow SSE client backpressures. Accept at most one char past the cap (so the
    # consumer still emits the capped notice) and drop the rest. The final result is captured independently, so this
    # never changes the byte-identical result.
    accepted_output_chars = 0
    accepted_output_lock = threading.Lock()

    def _on_output(text: str) -> None:
        nonlocal accepted_output_chars
        if not text:
            return
        with accepted_output_lock:
            remaining = TOOL_OUTPUT_STREAM_MAX_CHARS + 1 - accepted_output_chars
            if remaining <= 0:
                return
            accepted = text[:remaining]
            accepted_output_chars += len(accepted)
        output_queue.put(accepted)

    def _run() -> None:
        try:
            if accepts_kwarg(invoke, "tool_cancel_event"):
                outcome["result"] = invoke(_on_output, tool_cancel_event = execution_cancel)
            else:
                outcome["result"] = invoke(_on_output)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller side
            outcome["error"] = exc
        finally:
            _clear_stalled_tool(stall_key, stall_token)
            # Posted after the result/error is recorded; wakes the consumer immediately so fast tools pay no
            # poll-interval latency.
            output_queue.put(done_sentinel)

    # The worker runs in the caller's context, so a tool started for one account cannot resolve another's roots.
    worker = threading.Thread(
        target = contextvars.copy_context().run,
        args = (_run,),
        daemon = True,
        name = f"tool-exec-{tool_name or 'unknown'}",
    )
    # The final durable CAS is intentionally adjacent to Thread.start(). A
    # concurrent Stop or ownership rotation that wins first prevents the worker
    # from being admitted at all.
    execution = record_execution_started(tool_name, tool_call_id)
    if (
        isinstance(execution, DurableExecutionHandle)
        and execution.replay_completion is not None
    ):
        selected = execution.replay_completion.get("selected_result")
        return prepared_result(selected, execution)
    worker.start()

    # Heartbeats are paced by counting idle queue polls; the wall clock is used
    # only for the hard execution deadline.
    idle_polls_per_heartbeat = max(1, int(round(heartbeat_interval_s / poll_interval_s)))
    idle_polls = 0
    streamed_chars = 0
    stream_capped = False
    finished = False

    def _drain_pending(max_chars: int | None = None) -> str:
        nonlocal finished
        text, hit_sentinel = _drain_queue(output_queue, done_sentinel, max_chars)
        if hit_sentinel:
            finished = True
        return text

    def _drain_and_drop() -> None:
        """Discard the current and every queued chunk without concatenating.

        Past the cap every chunk is dropped, so don't pay to build a combined
        string only to drop it. Still detect completion so the loop can exit.
        """
        nonlocal finished
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                return
            if item is done_sentinel:
                finished = True
                return

    abnormal_exit = False
    timed_out = False
    try:
        while not finished:
            remaining_time = deadline - _deadline_monotonic()
            if remaining_time <= 0:
                timed_out = True
                execution_cancel.set()
                break
            try:
                item = output_queue.get(timeout = min(poll_interval_s, remaining_time))
            except queue.Empty:
                if _deadline_monotonic() >= deadline:
                    timed_out = True
                    execution_cancel.set()
                    break
                # A disconnect sets cancel_event while the worker is silent; surface a heartbeat this poll so the route
                # regains control and tears down at once, not after a full heartbeat interval.
                if cancel_event is not None and cancel_event.is_set():
                    yield {"type": "heartbeat"}
                    continue
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            if item is done_sentinel:
                break

            if stream_capped:
                # Past the cap: drop this chunk and every queued sibling (see _drain_and_drop). Pace with one time.sleep
                # per poll (not time.monotonic -- tests patch the clock), counted as an idle poll so heartbeats keep
                # flowing while the queue stays non-empty.
                _drain_and_drop()
                if finished:
                    break
                time.sleep(min(poll_interval_s, max(0.0, deadline - _deadline_monotonic())))
                idle_polls += 1
                if idle_polls >= idle_polls_per_heartbeat:
                    idle_polls = 0
                    yield {"type": "heartbeat"}
                continue

            # Bound the join to the remaining budget so the crossing batch can't allocate far past the cap (surplus is
            # truncated below anyway); the prefix is long enough that truncation stays byte-identical.
            budget = TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars
            chunk = item + _drain_pending(max_chars = budget - len(item))
            idle_polls = 0
            if streamed_chars + len(chunk) > TOOL_OUTPUT_STREAM_MAX_CHARS:
                chunk = chunk[: max(0, TOOL_OUTPUT_STREAM_MAX_CHARS - streamed_chars)]
                chunk += _STREAM_CAPPED_NOTICE
                stream_capped = True
            streamed_chars += len(chunk)
            if chunk:
                yield {
                    "type": "tool_output",
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "text": chunk,
                }
    except BaseException:
        # The loop only raises when the consumer closes us early: an SSE disconnect calls gen.close() (GeneratorExit at
        # the yield) or the route throws in. Signal cancellation so a cancel-observing tool returns; the daemon worker
        # is then abandoned (see finally). Re-raise so the caller sees the real cause (GeneratorExit must not be
        # swallowed). Runs ONLY on abnormal exit, so the shared cancel_event is never set out from under the next tool
        # in a clean multi-tool turn.
        abnormal_exit = True
        execution_cancel.set()
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception:
                pass
        raise
    finally:
        # Clean finish: the worker already recorded its result and queued the sentinel we consumed, so this join returns
        # at once. Cancellation/timeout gets only a tiny settle window: cooperative callables normally exit inside it,
        # while a cancellation-ignoring daemon is still abandoned quickly and cannot hold response teardown open.
        worker.join(
            timeout = _CANCEL_SETTLE_TIMEOUT_S
            if (abnormal_exit or timed_out)
            else _WORKER_JOIN_TIMEOUT_S
        )
        if abnormal_exit and worker.is_alive():
            with _STALLED_TOOLS_LOCK:
                _STALLED_TOOLS[stall_key] = stall_token
            if not worker.is_alive():
                _clear_stalled_tool(stall_key, stall_token)
        if abnormal_exit:
            # The consumer disappeared after the execution receipt was created.
            # If the worker already settled we can persist that exact outcome;
            # otherwise the only safe durable statement is that the effect may
            # have happened. Recovery must never infer "not executed" from it.
            if worker.is_alive():
                record_execution_finished(
                    execution,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result="tool stream closed while execution was still running",
                    ambiguous=True,
                )
            elif outcome.get("error") is not None:
                record_execution_finished(
                    execution,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    error=outcome["error"],
                )
            else:
                record_execution_finished(
                    execution,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=outcome.get("result"),
                    producer_receipt=receipt_dict(outcome.get("result")),
                )

    if timed_out:
        # A cooperative callable usually exits as soon as the private cancel event
        # is set.  Only mark it stalled if the worker really survived the deadline;
        # this prevents repeated model retries from multiplying orphan threads.
        if worker.is_alive():
            with _STALLED_TOOLS_LOCK:
                _STALLED_TOOLS[stall_key] = stall_token
            # Close the race where the worker exited just before the marker was
            # installed and therefore already ran its finally-block clear.
            if not worker.is_alive():
                _clear_stalled_tool(stall_key, stall_token)
        timeout_result = tool_timeout_result(tool_name, hard_timeout_s)
        record_execution_finished(
            execution,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=timeout_result,
            ambiguous=True,
        )
        return timeout_result

    error = outcome.get("error")
    if error is not None:
        record_execution_finished(
            execution,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            error=error,
        )
        raise error
    # Returned verbatim (the loop's record_result handles non-str), so the final tool result is byte-identical to a
    # direct execute_tool call.
    result = outcome.get("result")
    if execution is not None and not isinstance(execution, DurableExecutionHandle):
        # Compatibility for injected/legacy journal adapters that return only
        # an opaque execution token and cannot participate in prepared commit.
        record_execution_finished(
            execution,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=result,
        )
        return result
    return prepared_result(result, execution)
