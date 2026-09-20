# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from core.helix_engine.optimization_benchmark import ExecutionScope, ProviderTier
from core.helix_engine.optimization_trace import (
    OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY,
    OPTIMIZATION_TRACE_SCOPE_STATE_KEY,
    OptimizationTraceConfig,
    OptimizationTraceRecorder,
    optimization_trace_envelope,
)
from core.inference import llama_keepwarm
from core.inference.chat_generation_runs import (
    _EVENT_BATCH_SECONDS,
    _EVENT_SINGLE_FLUSH_SECONDS,
    _background_request,
    _run_key,
    ChatGenerationSupervisor,
)
from models.inference import ChatCompletionRequest
from routes import chat_generation_runs as run_routes
from routes import inference
from state import active_generations
from storage import chat_generation_runs_db as runs_db
from storage import studio_db


@pytest.fixture
def durable_run(request):
    engine = getattr(request, "param", "gguf")
    model = "local.gguf" if engine == "gguf" else "local.safetensors"
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": model, "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )
    run, _created = runs_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "cancel_id": "run-1",
            "thread_id": "thread-1",
            "generation_run_id": "run-1",
        },
    )
    active_generations.reset_for_tests()
    yield run
    active_generations.reset_for_tests()


def _create_payload(content = "Hello"):
    return run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {"model": "local", "messages": [{"role": "user", "content": content}]},
    )


def _route_request(supervisor):
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor))
    )


def _trace_wire(model="local.safetensors"):
    recorder = OptimizationTraceRecorder(
        OptimizationTraceConfig(
            scope=ExecutionScope.FOREGROUND,
            provider_tier=ProviderTier.SENIOR,
            model_identity=model,
            provider_identity="local:mlx",
        )
    )
    recorder.begin_invocation().complete_final_answer()
    return optimization_trace_envelope(recorder).to_dict()


def test_only_server_created_durable_scope_can_enable_optimization_trace(monkeypatch):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    internal = _background_request(
        SimpleNamespace(state=SimpleNamespace()),
        "run-1",
        threading.Event(),
    )
    assert getattr(internal.state, OPTIMIZATION_TRACE_SCOPE_STATE_KEY).provider_tier is ProviderTier.SENIOR

    public = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"x-helix-optimization-trace-v1", b"1")],
            "state": {},
        }
    )
    assert (
        inference._mlx_optimization_trace_recorder(
            public,
            model_identity="local.safetensors",
            is_mlx=True,
        )
        is None
    )


def test_default_off_background_scope_is_byte_compatible(monkeypatch):
    monkeypatch.delenv("HELIX_OPTIMIZATION_TRACE_V1", raising=False)
    monkeypatch.delenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", raising=False)
    request = _background_request(
        SimpleNamespace(state=SimpleNamespace()),
        "run-1",
        threading.Event(),
    )
    assert OPTIMIZATION_TRACE_SCOPE_STATE_KEY not in request.scope["state"]


@pytest.mark.asyncio
async def test_public_chat_wrapper_keeps_cancel_on_disconnect(monkeypatch):
    observed = []

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        observed.append(cancel_on_disconnect)
        return "response"

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    payload = ChatCompletionRequest(
        model = "local",
        messages = [{"role": "user", "content": "Hello"}],
    )
    assert await inference.openai_chat_completions(payload, object(), "alice") == "response"
    assert observed == [True]


@pytest.mark.parametrize(
    "is_mlx,durable_run,generation_run_id,completion_tokens,expected",
    [
        (True, True, "run-1", 8, "length"),
        (True, True, "run-1", 7, "stop"),
        (True, False, "run-1", 8, "stop"),
        (False, True, "run-1", 8, "stop"),
    ],
)
def test_only_durable_mlx_normalizes_stop_at_token_cap(
    is_mlx, durable_run, generation_run_id, completion_tokens, expected
):
    payload = SimpleNamespace(
        generation_run_id = generation_run_id,
        max_tokens = 8,
        max_completion_tokens = None,
    )
    stats = {"usage": {"completion_tokens": completion_tokens}}
    assert (
        inference._safetensors_finish_reason(stats, payload, is_mlx = is_mlx, durable_run = durable_run)
        == expected
    )


@pytest.mark.asyncio
async def test_create_route_schedules_producer_on_request_loop(monkeypatch):
    started = []
    supervisor = SimpleNamespace(
        start = lambda run_id, **identity: started.append((run_id, identity))
    )
    request = _route_request(supervisor)

    def create_run(**_kwargs):
        assert started == []
        return (
            {
                "id": "run-1",
                "status": "queued",
                "threadId": "thread-1",
                "requestPayload": {"model": "local"},
            },
            True,
        )

    monkeypatch.setattr(run_routes.db, "create_run", create_run)
    response = await run_routes.create_chat_generation_run(_create_payload(), request, "alice")
    assert response["created"] is True
    assert started == [("run-1", {"thread_id": "thread-1", "model": "local"})]


@pytest.mark.asyncio
async def test_terminal_idempotent_create_does_not_reserve_generation(monkeypatch):
    terminal = {
        "id": "run-1",
        "status": "completed",
        "threadId": "thread-1",
        "requestPayload": {"model": "local"},
    }
    supervisor = SimpleNamespace(
        start = lambda *_args, **_kwargs: pytest.fail("terminal run must not start"),
    )
    request = _route_request(supervisor)
    monkeypatch.setattr(run_routes.db, "create_run", lambda **_kwargs: (terminal, False))
    response = await run_routes.create_chat_generation_run(_create_payload("Hi"), request, "alice")
    assert response["created"] is False


@pytest.mark.asyncio
async def test_background_producer_persists_chunks_and_completes(durable_run, monkeypatch):
    observed = []
    leaked = []
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        observed.append(cancel_on_disconnect)
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: leaked.append(context))
    try:
        await supervisor._produce("run-1")
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)
    run = runs_db.get_run("run-1", "alice")
    assert leaked == []
    assert observed == [False]
    assert (run["status"], run["finishReason"]) == ("completed", "stop")
    assert [
        event["payload"] for event in runs_db.list_events("run-1") if event["type"] == "chunk"
    ] == chunks


@pytest.mark.asyncio
async def test_private_trace_frame_is_stripped_bound_and_worker_fenced(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    wire = _trace_wire()
    public_chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}]}

    async def body():
        yield f"data: {json.dumps(public_chunk)}\n\n"
        yield f"data: {json.dumps({OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: wire})}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(_payload, request, _subject, *, cancel_on_disconnect):
        assert cancel_on_disconnect is False
        assert getattr(request.state, OPTIMIZATION_TRACE_SCOPE_STATE_KEY).provider_tier is ProviderTier.SENIOR
        return SimpleNamespace(status_code=200, body_iterator=body())

    original_finish = runs_db.finish_run
    fenced_calls = []

    def finish_run(run_id, *, worker_token, pending_events=(), **kwargs):
        batch = list(pending_events)
        if any(event[0] == "optimization.trace" for event in batch):
            fenced_calls.append((run_id, worker_token))
        return original_finish(
            run_id,
            worker_token=worker_token,
            pending_events=batch,
            **kwargs,
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(runs_db, "finish_run", finish_run)
    expected_worker_token = runs_db.get_worker_token("run-1")

    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    events = runs_db.list_events("run-1")
    assert [event["payload"] for event in events if event["type"] == "chunk"] == [
        public_chunk
    ]
    [stored] = [event["payload"] for event in events if event["type"] == "optimization.trace"]
    assert stored["status"] == "available"
    assert stored["aggregation"] == {"admissible": True, "reason": None}
    assert stored["binding"] == {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message_id": "user-1",
        "owner_subject": "alice",
        "account_id": stored["binding"]["account_id"],
    }
    assert stored["binding"]["account_id"]
    assert stored["trace"]["invocation_counts"]["model_invocations"]["counts"][
        "foreground"
    ]["senior"] == 1
    assert fenced_calls == [("run-1", expected_worker_token)]


@pytest.mark.asyncio
async def test_cancel_winning_terminal_transaction_rewrites_available_trace(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    public = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

    async def body():
        yield f"data: {json.dumps(public)}\n\n"
        private = {OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire()}
        yield f"data: {json.dumps(private)}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    original_finish = runs_db.finish_run
    cancellation_won = False

    def cancel_then_finish(run_id, *, worker_token, pending_events=(), **kwargs):
        nonlocal cancellation_won
        batch = list(pending_events)
        [proposed] = [event[1] for event in batch if event[0] == "optimization.trace"]
        assert proposed["status"] == "available"
        assert proposed["aggregation"]["admissible"] is True
        cancelled = runs_db.request_cancel(run_id, "alice")
        assert cancelled is not None and cancelled["status"] == "cancelling"
        cancellation_won = True
        return original_finish(
            run_id,
            worker_token=worker_token,
            pending_events=batch,
            **kwargs,
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(runs_db, "finish_run", cancel_then_finish)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    [stored] = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    assert cancellation_won is True
    assert (run["status"], run["finishReason"], run["error"]) == (
        "cancelled",
        "cancelled",
        None,
    )
    assert stored["status"] == "unavailable"
    assert stored["trace"] is None
    assert stored["error"] == "generation_attempt_incomplete"
    assert stored["aggregation"] == {
        "admissible": False,
        "reason": "generation_attempt_incomplete",
    }
    assert [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "chunk"
    ] == [public]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_kind",
    ["caller_ids", "wrong_config", "malformed", "duplicate"],
)
async def test_bad_private_trace_fails_evidence_closed_without_binding_crossover(
    durable_run, monkeypatch, bad_kind
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    wire = _trace_wire()
    if bad_kind == "caller_ids":
        wire["run_id"] = "other-run"
        wire["owner_subject"] = "mallory"
        frames = [wire]
    elif bad_kind == "wrong_config":
        wire["invocation_events"][0]["provider_identity"] = "remote:spoofed"
        frames = [wire]
    elif bad_kind == "malformed":
        frames = [{"schema_version": "wrong"}]
    else:
        frames = [wire, wire]

    async def body():
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}\n\n'
        for frame in frames:
            yield f"data: {json.dumps({OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: frame})}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    [stored] = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    assert stored["status"] == "error"
    assert stored["trace"] is None
    assert stored["binding"]["run_id"] == "run-1"
    assert stored["binding"]["thread_id"] == "thread-1"
    assert stored["binding"]["owner_subject"] == "alice"
    assert "mallory" not in json.dumps(stored)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier,expected_status,expected_error",
    [
        ("senior", "unavailable", "missing_trace_envelope"),
        ("not-a-tier", "error", "invalid_provider_tier"),
    ],
)
async def test_missing_or_invalid_trace_configuration_never_becomes_zero_success(
    durable_run, monkeypatch, tier, expected_status, expected_error
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", tier)

    async def body():
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    assert run["status"] == "completed"
    [stored] = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    assert stored["status"] == expected_status
    assert stored["error"] == expected_error
    assert stored["trace"] is None


@pytest.mark.asyncio
async def test_trace_persistence_failure_does_not_reclassify_user_generation(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")

    async def body():
        public = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(public)}\n\n"
        yield f"data: {json.dumps({OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire()})}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    original_finish = runs_db.finish_run

    def fail_trace_only(run_id, *, worker_token, pending_events=(), **kwargs):
        batch = list(pending_events)
        if any(event[0] == "optimization.trace" for event in batch):
            raise RuntimeError("trace store unavailable")
        return original_finish(
            run_id,
            worker_token=worker_token,
            pending_events=batch,
            **kwargs,
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(runs_db, "finish_run", fail_trace_only)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"], run["error"]) == (
        "completed",
        "stop",
        None,
    )
    assert not [
        event for event in runs_db.list_events("run-1") if event["type"] == "optimization.trace"
    ]


@pytest.mark.asyncio
async def test_post_parse_trace_serialization_failure_is_fail_neutral(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")

    async def body():
        public = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(public)}\n\n"
        private = {OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire()}
        yield f"data: {json.dumps(private)}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    from core.helix_engine import optimization_trace as trace_module

    original_parse = trace_module.parse_optimization_trace_envelope
    serialized = False

    class ParsedButUnserializable:
        def __init__(self, parsed):
            self.invocation_events = parsed.invocation_events
            self.status = parsed.status
            self.error = parsed.error

        def to_dict(self):
            nonlocal serialized
            serialized = True
            raise RuntimeError("post-parse serialization failed")

    def parse_then_fail_serialization(value):
        return ParsedButUnserializable(original_parse(value))

    monkeypatch.setattr(
        trace_module,
        "parse_optimization_trace_envelope",
        parse_then_fail_serialization,
    )
    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    assert serialized is True
    assert (run["status"], run["finishReason"], run["error"]) == (
        "completed",
        "stop",
        None,
    )
    assert not [
        event for event in runs_db.list_events("run-1") if event["type"] == "optimization.trace"
    ]


@pytest.mark.asyncio
async def test_trace_insert_and_terminal_event_are_one_rollback_boundary(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")

    async def body():
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})}\n\n'
        yield f"data: {json.dumps({OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire()})}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    original_append_locked = runs_db._append_events_locked
    failed_after_trace = False

    def crash_before_terminal(conn, run_id, events):
        nonlocal failed_after_trace
        batch = list(events)
        if (
            not failed_after_trace
            and any(event[0] == "run.completed" for event in batch)
        ):
            failed_after_trace = True
            raise RuntimeError("crash after trace insert")
        return original_append_locked(conn, run_id, batch)

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(runs_db, "_append_events_locked", crash_before_terminal)

    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    assert failed_after_trace is True
    assert (run["status"], run["finishReason"]) == ("completed", "stop")
    assert not [
        event for event in runs_db.list_events("run-1") if event["type"] == "optimization.trace"
    ], "the failed terminal transaction must roll its earlier trace insert back"


@pytest.mark.asyncio
async def test_interrupted_attempt_trace_is_explicitly_inadmissible(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")

    async def body():
        yield f"data: {json.dumps({OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire()})}\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code=200, body_iterator=body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    [stored] = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
    assert stored["status"] == "unavailable"
    assert stored["trace"] is None
    assert stored["aggregation"] == {
        "admissible": False,
        "reason": "generation_attempt_incomplete",
    }


@pytest.mark.asyncio
async def test_recovered_run_persists_exclusion_before_replacement_attempt(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    old_token = runs_db.get_worker_token("run-1")
    runs_db.append_events(
        "run-1",
        old_token,
        [
            (
                "optimization.trace",
                {
                    "schema_version": "helix.optimization-trace-event.v1",
                    "status": "available",
                    "aggregation": {"admissible": True, "reason": None},
                    "binding": {"run_id": "run-1"},
                    "trace": {"physical_attempt": "old"},
                    "error": None,
                },
            )
        ],
    )
    resume = {
        **durable_run["requestPayload"],
        "finalization_idempotency_key": "run-1",
    }
    requeued = runs_db.requeue_run_for_restart("run-1", resume)
    assert requeued is not None
    new_token = requeued["_workerToken"]

    exclusion_was_durable_before_generation = False

    async def body():
        private = {OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY: _trace_wire("local.gguf")}
        yield f"data: {json.dumps(private)}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        nonlocal exclusion_was_durable_before_generation
        traces = [
            event["payload"]
            for event in runs_db.list_events("run-1")
            if event["type"] == "optimization.trace"
        ]
        exclusion_was_durable_before_generation = (
            traces[-1]["error"] == "recovered_run_trace_incomplete"
        )
        return SimpleNamespace(status_code=200, body_iterator=body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce(
        "run-1",
        worker_payload_override=resume,
        expected_worker_token=new_token,
    )

    traces = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    assert exclusion_was_durable_before_generation is True
    assert len(traces) == 2
    assert traces[-1]["status"] == "unavailable"
    assert traces[-1]["trace"] is None
    assert traces[-1]["aggregation"] == {
        "admissible": False,
        "reason": "recovered_run_trace_incomplete",
    }


@pytest.mark.asyncio
async def test_stale_recovery_worker_cannot_append_trace_exclusion(
    durable_run, monkeypatch
):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    stale_token = runs_db.get_worker_token("run-1")
    resume = {
        **durable_run["requestPayload"],
        "finalization_idempotency_key": "run-1",
    }
    assert runs_db.requeue_run_for_restart("run-1", resume) is not None
    before = runs_db.get_run("run-1")["lastEventSeq"]

    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce(
        "run-1",
        worker_payload_override=resume,
        expected_worker_token=stale_token,
    )

    assert runs_db.get_run("run-1")["lastEventSeq"] == before
    assert not [
        event for event in runs_db.list_events("run-1") if event["type"] == "optimization.trace"
    ]


@pytest.mark.asyncio
async def test_a_prefill_reporting_only_progress_renews_the_lease(durable_run, monkeypatch):
    """A 250K prefill outruns the 1200s lease before its first token, and the
    write is what renews it, so dropping content-less progress chunks would reap a
    healthy prefill. Sampled mid-run: afterwards every chunk has landed."""
    released = asyncio.Event()
    sampled: dict = {}

    def _progress(processed):
        # What llama-server sends under return_progress: a content-less delta.
        return {
            "choices": [{"delta": {"role": "assistant", "content": None}, "finish_reason": None}],
            "prompt_progress": {
                "total": 250000,
                "cache": 0,
                "processed": processed,
                "time_ms": processed,
            },
        }

    async def body():
        for processed in (1024, 8192, 65536):
            yield f"data: {json.dumps(_progress(processed))}\n\n"
        # Polled, not slept: the idle flush is on a 0.1s timer
        # (_EVENT_BATCH_SECONDS) and a sleep sized against it flakes under load.
        _deadline = time.monotonic() + 10.0
        while time.monotonic() < _deadline:
            sampled["events"] = [
                e["payload"] for e in runs_db.list_events("run-1") if e["type"] == "chunk"
            ]
            if len(sampled["events"]) >= 3:
                break
            await asyncio.sleep(0.01)
        sampled["progress"] = runs_db.get_progress("run-1")
        released.set()
        yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Hi'}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    await supervisor._produce("run-1")
    assert released.is_set()
    assert len(sampled["events"]) == 3, sampled["events"]
    assert all("prompt_progress" in e for e in sampled["events"])
    # The lease counts writes, so it renewed three times before the first token.
    assert sampled["progress"][1] == 3, sampled["progress"]
    assert runs_db.get_run("run-1", "alice")["status"] == "completed"


async def _subscriber_sequences(after = 0):
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = AsyncMock(return_value = True)),
        after = after,
        last_event_id = None,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return [int(line[4:]) for line in raw.splitlines() if line.startswith("id: ")]


def _stub_replay(
    monkeypatch,
    events,
    *,
    status,
    last_event_seq,
    fail_on_second_wait = False,
):
    snapshot = {
        "id": "run-1",
        "status": status,
        "lastEventSeq": last_event_seq,
        "updatedAt": 123,
    }
    waits = []

    def get_run(_run_id):
        return snapshot

    def wait_for_events(_run_id, after, _timeout):
        waits.append(after)
        if fail_on_second_wait and len(waits) > 1:
            raise AssertionError("replay did not advance past the private event")
        return [event for event in events if int(event["seq"]) > after]

    monkeypatch.setattr(run_routes.db, "get_run", get_run)
    monkeypatch.setattr(run_routes.db, "wait_for_events", wait_for_events)
    return waits


async def _replay_body(*, after = None, last_event_id = None, disconnected = True):
    disconnected_check = AsyncMock(return_value = disconnected)
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = disconnected_check),
        after = after,
        last_event_id = last_event_id,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return raw, disconnected_check


@pytest.mark.asyncio
async def test_authenticated_replay_omits_private_trace_only_stream(monkeypatch):
    waits = _stub_replay(
        monkeypatch,
        [
            {
                "seq": 2,
                "type": "optimization.trace",
                "payload": {"trace": "private-proof"},
                "createdAt": 2,
            }
        ],
        status = "running",
        last_event_seq = 2,
    )

    raw, disconnected_check = await _replay_body(disconnected = True)

    assert raw == ""
    assert waits == [0]
    assert disconnected_check.await_count == 1


@pytest.mark.asyncio
async def test_authenticated_replay_preserves_public_order_while_hiding_private_trace(monkeypatch):
    waits = _stub_replay(
        monkeypatch,
        [
            {
                "seq": 2,
                "type": "chunk",
                "payload": {"choices": [{"delta": {"content": "Hello"}}]},
                "createdAt": 2,
            },
            {
                "seq": 3,
                "type": "optimization.trace",
                "payload": {"trace": "private-proof"},
                "createdAt": 3,
            },
            {
                "seq": 4,
                "type": "run.completed",
                "payload": {"status": "completed"},
                "createdAt": 4,
            },
        ],
        status = "completed",
        last_event_seq = 4,
    )

    raw, disconnected_check = await _replay_body(disconnected = False)

    assert [line[4:] for line in raw.splitlines() if line.startswith("id: ")] == ["2", "4"]
    assert [line[7:] for line in raw.splitlines() if line.startswith("event: ")] == [
        "chunk",
        "run.completed",
    ]
    assert "optimization.trace" not in raw
    assert "private-proof" not in raw
    assert waits == [0]
    assert disconnected_check.await_count == 0


@pytest.mark.asyncio
async def test_reconnect_cursor_advances_past_private_trace_tail(monkeypatch):
    waits = _stub_replay(
        monkeypatch,
        [
            {
                "seq": 3,
                "type": "optimization.trace",
                "payload": {"trace": "private-proof"},
                "createdAt": 3,
            }
        ],
        status = "completed",
        last_event_seq = 3,
        fail_on_second_wait = True,
    )

    raw, disconnected_check = await _replay_body(
        last_event_id = "2",
        disconnected = False,
    )

    assert raw == ""
    assert waits == [2]
    assert disconnected_check.await_count == 0


@pytest.mark.asyncio
async def test_terminal_replay_completes_when_last_event_is_private_trace(monkeypatch):
    waits = _stub_replay(
        monkeypatch,
        [
            {
                "seq": 2,
                "type": "optimization.trace",
                "payload": {"trace": "private-proof"},
                "createdAt": 2,
            }
        ],
        status = "completed",
        last_event_seq = 2,
        fail_on_second_wait = True,
    )

    raw, disconnected_check = await _replay_body(disconnected = False)

    assert raw == ""
    assert waits == [0]
    assert disconnected_check.await_count == 0


def _route_engine(monkeypatch, model, first, release):
    gguf = model.endswith(".gguf")

    def generate(*, stats_holder = None, **_kwargs):
        first.set()
        yield "A"
        assert release.wait(5)
        yield "AB"
        if stats_holder is None:
            yield {"type": "metadata", "finish_reason": "stop"}
        else:
            stats_holder["stats"] = {"usage": {"completion_tokens": 2}}

    llama = SimpleNamespace(
        is_loaded = gguf,
        model_identifier = model,
        base_url = "http://llama.test",
        effective_parallel_slots = 1,
        supports_tools = False,
        is_vision = False,
        _is_audio = False,
        context_length = None,
        generate_chat_completion = generate,
    )
    mlx = SimpleNamespace(
        active_model_name = model,
        models = {model: {"is_mlx": True, "chat_template_info": {"template": "chatml"}}},
        generate_chat_response = generate,
        reset_generation_state = lambda *_a, **_k: None,
    )
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: mlx)
    monkeypatch.setattr(inference, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(
        inference, "_detect_safetensors_features", lambda *_a, **_k: {"supports_tools": False}
    )
    monkeypatch.setattr(inference, "_effective_enable_tools", lambda _payload: False)

    async def no_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", no_switch)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_run", ["gguf", "mlx"], indirect = True)
async def test_subscribers_detach_then_replay_the_same_engine_run(durable_run, monkeypatch):
    first_chunk, release = threading.Event(), threading.Event()
    _route_engine(monkeypatch, durable_run["requestPayload"]["model"], first_chunk, release)
    task = asyncio.create_task(
        ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    )
    if not await asyncio.to_thread(first_chunk.wait, 5):
        await task
        pytest.fail(str(runs_db.get_run("run-1", "alice")))
    while len(runs_db.list_events("run-1")) < 4:
        await asyncio.sleep(0.01)
    first, second = await asyncio.gather(_subscriber_sequences(), _subscriber_sequences())
    assert first == second == list(range(1, max(first) + 1))
    assert runs_db.get_run("run-1", "alice")["status"] == "running"
    release.set()
    await task
    tail = await _subscriber_sequences(after = max(first))
    run = runs_db.get_run("run-1", "alice")
    assert first + tail == list(range(1, run["lastEventSeq"] + 1))
    deltas = [
        event["payload"].get("choices", [{}])[0].get("delta", {}).get("content")
        for event in runs_db.list_events("run-1")
        if event["type"] == "chunk"
    ]
    assert [text for text in deltas if text] == ["A", "B"]


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_run", ["mlx"], indirect=True)
async def test_real_mlx_plain_stream_exports_one_final_invocation(durable_run, monkeypatch):
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_V1", "1")
    monkeypatch.setenv("HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER", "senior")
    first_chunk, release = threading.Event(), threading.Event()
    release.set()
    _route_engine(monkeypatch, durable_run["requestPayload"]["model"], first_chunk, release)

    await ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))._produce("run-1")

    [stored] = [
        event["payload"]
        for event in runs_db.list_events("run-1")
        if event["type"] == "optimization.trace"
    ]
    trace = stored["trace"]
    assert stored["status"] == "available"
    assert trace["all_handles_settled"] is True
    assert len(trace["invocation_events"]) == 1
    assert trace["invocation_events"][0]["semantic_contribution"] == "final_answer"
    assert trace["invocation_events"][0]["provider_identity"] == "local:mlx"
    assert trace["invocation_counts"]["model_invocations"]["counts"]["foreground"][
        "senior"
    ] == 1


async def _await_chunk_payloads(run_id: str, count: int, deadline_s: float) -> list:
    """Chunk payloads once `count` of them are durable, or whatever arrived by the deadline.

    Returned rather than asserted so the caller owns the comparison and pytest still
    shows the payload diff on failure.
    """
    started = time.monotonic()
    while True:
        stored = [e["payload"] for e in runs_db.list_events(run_id) if e["type"] == "chunk"]
        if len(stored) >= count or time.monotonic() - started >= deadline_s:
            return stored
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_event_batch_flushes_while_upstream_is_idle(durable_run, monkeypatch):
    release = asyncio.Event()
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
    ]

    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        await release.wait()
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    task = asyncio.create_task(supervisor._produce("run-1"))
    # Poll rather than sleep a fixed span. The flush costs the batch timer plus a
    # thread hop and a SQLite write, which measures ~0.11s on an idle machine, so
    # the old bare sleep(0.2) left under 2x headroom and lost the race on a loaded
    # runner. The budget is still bounded well below _EVENT_SINGLE_FLUSH_SECONDS,
    # so a regression that drops these two events onto the single-event timer, or
    # never flushes them at all, still fails here rather than passing slowly.
    deadline = (_EVENT_BATCH_SECONDS + _EVENT_SINGLE_FLUSH_SECONDS) / 2
    stored = await _await_chunk_payloads("run-1", len(chunks), deadline)
    assert stored == chunks
    release.set()
    await task


@pytest.mark.asyncio
async def test_model_lifecycle_cancel_reaches_same_registered_event(durable_run, monkeypatch):
    registered = asyncio.Event()

    async def body(cancel_event):
        with active_generations.ActiveGeneration(
            cancel_event,
            thread_id = "thread-1",
            run_id = "run-1",
        ):
            registered.set()
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
        if False:
            yield ""

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(
            status_code = 200,
            body_iterator = body(request.state.generation_cancel_event),
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    task = asyncio.create_task(supervisor._produce("run-1"))
    await asyncio.wait_for(registered.wait(), timeout = 2)
    assert active_generations.cancel_all() == 1
    await asyncio.wait_for(task, timeout = 2)
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"
    metadata = studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]
    assert metadata["incomplete"] == {"reason": "cancelled"}
    assert active_generations.count() == 0


@pytest.mark.asyncio
async def test_cancel_before_registration_signals_load_event(durable_run, monkeypatch):
    entered = asyncio.Event()
    cancel_ids = []
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))

    async def fake(_payload, request, *_args, **_kwargs):
        event = request.state.generation_cancel_event
        entered.set()
        while not event.is_set():
            await asyncio.sleep(0.01)

        async def body():
            if False:
                yield ""

        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(
        inference,
        "_cancel_by_cancel_id_or_stash",
        lambda run_id: cancel_ids.append(run_id) or 0,
    )
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    task = supervisor._tasks[_run_key("run-1")]
    supervisor.cancel("run-1")
    await asyncio.wait_for(task, timeout = 2)
    await supervisor.stop()
    assert cancel_ids == ["run-1"]
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_start_reserves_slot_and_lifecycle_before_worker_runs(durable_run, monkeypatch):
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    monkeypatch.setattr(llama_keepwarm, "_inflight", 0)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1", thread_id = "thread-1", model = "local")
    assert (llama_keepwarm._pending, active_generations.count()) == (1, 1)
    assert active_generations.snapshot()[0]["thread_id"] == "thread-1"
    task = supervisor._tasks[_run_key("run-1")]
    supervisor.cancel("run-1")
    await asyncio.wait_for(task, timeout = 2)
    await supervisor.stop()
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"
    assert (llama_keepwarm._pending, llama_keepwarm._inflight) == (0, 0)


@pytest.mark.asyncio
async def test_start_restores_caller_binding_and_finalizes_after_owner_cleanup(
    durable_run, monkeypatch
):
    # Use a non-zero epoch so a leaked ContextVar token cannot masquerade as
    # the caller's current binding.
    active_generations.fence("owner")
    active_generations.lift_fence("owner")
    owner_epoch = active_generations.lifecycle_epoch("owner")
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    monkeypatch.setattr(llama_keepwarm, "_inflight", 0)

    observed: dict[str, int] = {}
    finalized = asyncio.Event()

    async def fake_produce(_run_id, _cancel_event, _activity, **_kwargs):
        observed["producer_epoch"] = active_generations.bound_lifecycle_epoch()
        observed["producer_count"] = active_generations.count()
        # Simulate account deactivate/reactivate after admission. The done
        # callback must hand the immutable owner epoch to finalization rather
        # than sampling this now-new current epoch.
        active_generations.fence("owner")
        active_generations.lift_fence("owner")

    async def fake_finalize(_run_id):
        # _task_done must release both process-global claims before it creates
        # the post-answer finalizer, while carrying the producer's owner epoch.
        observed["finalizer_epoch"] = active_generations.bound_lifecycle_epoch()
        observed["finalizer_count"] = active_generations.count()
        observed["finalizer_pending"] = llama_keepwarm._pending
        observed["finalizer_inflight"] = llama_keepwarm._inflight
        finalized.set()
        return None

    monkeypatch.setattr(
        supervisor := ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace())),
        "_produce",
        fake_produce,
    )
    monkeypatch.setattr(supervisor, "_finalize", fake_finalize)
    loop = asyncio.get_running_loop()
    callback_errors: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))
    try:
        supervisor.start("run-1", thread_id="thread-1", model="local")
        # start() temporarily binds the owner epoch to create_task, then must
        # restore this request context before returning.
        assert active_generations.lifecycle_epoch("owner") == owner_epoch
        assert active_generations.count() == 1
        task = supervisor._tasks[_run_key("run-1")]
        await asyncio.wait_for(task, timeout=2)
        await asyncio.wait_for(finalized.wait(), timeout=2)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert observed == {
        "producer_epoch": owner_epoch,
        "producer_count": 1,
        "finalizer_epoch": owner_epoch,
        "finalizer_count": 0,
        "finalizer_pending": 0,
        "finalizer_inflight": 0,
    }
    assert callback_errors == []
    assert active_generations.count() == 0
    assert active_generations.lifecycle_epoch("owner") == owner_epoch + 2


@pytest.mark.asyncio
async def test_start_create_task_failure_closes_producer_and_releases_owner_state(
    durable_run, monkeypatch
):
    class UnscheduledProducer:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    producer = UnscheduledProducer()
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))

    monkeypatch.setattr(supervisor, "_produce", lambda *_args, **_kwargs: producer)

    def fail_create_task(*_args, **_kwargs):
        raise RuntimeError("injected create_task failure")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    monkeypatch.setattr(llama_keepwarm, "_inflight", 0)

    with pytest.raises(RuntimeError, match="injected create_task failure"):
        supervisor.start("run-1", thread_id="thread-1", model="local")

    key = _run_key("run-1")
    assert producer.close_calls == 1
    assert active_generations.count() == 0
    assert (llama_keepwarm._pending, llama_keepwarm._inflight) == (0, 0)
    assert all(
        key not in collection
        for collection in (
            supervisor._tasks,
            supervisor._cancel_events,
            supervisor._activities,
            supervisor._active_registrations,
            supervisor._account_contexts,
        )
    )

    # A failed admission must not leave the caller bound to the old owner
    # epoch after the account advances through another fence/lift cycle.
    active_generations.fence("owner")
    active_generations.lift_fence("owner")
    assert active_generations.lifecycle_epoch("owner") == 2


@pytest.mark.asyncio
async def test_cancelled_producer_error_is_cancelled(durable_run, monkeypatch):
    entered = asyncio.Event()

    async def fake(_payload, request, *_args, **_kwargs):
        entered.set()
        while not request.state.generation_cancel_event.is_set():
            await asyncio.sleep(0.01)
        raise RuntimeError("Generation cancelled")

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    task = supervisor._tasks[_run_key("run-1")]
    assert active_generations.cancel_all() == 1
    await asyncio.wait_for(task, timeout = 2)
    await supervisor.stop()
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_uncancelled_partial_eof_is_interrupted(durable_run, monkeypatch):
    async def body():
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")


@pytest.mark.asyncio
async def test_a_caught_up_reconnect_to_a_settled_run_does_not_block(durable_run, monkeypatch):
    """Nothing left to replay must return at once, not after the 15s event wait.

    Otherwise a finished answer reads as still generating for the whole timeout and one of
    the event-wait workers is held for it.
    """

    async def body():
        yield 'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    settled = runs_db.get_run("run-1", "alice")
    assert settled["status"] == "completed"

    caught_up = await asyncio.wait_for(
        _subscriber_sequences(after = int(settled["lastEventSeq"])),
        timeout = 5,
    )
    assert caught_up == []
    # A client that is behind still gets the whole ledger.
    assert await _subscriber_sequences() == list(range(1, int(settled["lastEventSeq"]) + 1))


@pytest.mark.asyncio
async def test_streamed_error_outranks_cleanup_cancellation(durable_run, monkeypatch):
    """A backend failure must not be recorded as if the user pressed Stop.

    ``gguf_stream_chunks`` emits the error in band, follows it with ``[DONE]`` and then
    sets this same ``cancel_event`` from its ``finally`` because the stream did not
    complete. Nobody asked to cancel, so the run has to settle as ``failed`` carrying
    the diagnostic the user needs to act on.
    """

    async def body(cancel_event):
        try:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield 'data: {"error": {"message": "Out of memory"}}\n\ndata: [DONE]\n\n'
        finally:
            cancel_event.set()

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(
            status_code = 200,
            body_iterator = body(request.state.generation_cancel_event),
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    run = runs_db.get_run("run-1", "alice")
    assert run["status"] == "failed"
    assert run["error"] == "Out of memory"
    assert run["finishReason"] != "cancelled"
    metadata = studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]
    assert metadata["incomplete"] != {"reason": "cancelled"}


@pytest.mark.asyncio
async def test_graceful_supervisor_shutdown_is_interrupted(durable_run, monkeypatch):
    entered = asyncio.Event()

    async def body(request):
        entered.set()
        while not request.state.generation_cancel_event.is_set():
            await asyncio.sleep(0.01)
        if False:
            yield ""

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body(request))

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    await supervisor.stop()
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
    assert run["error"] == "Studio shut down during generation"


def test_thread_delete_captures_durable_run_before_cascade(durable_run):
    research_ids, chat_ids = studio_db.delete_chat_threads_with_active_runs(["thread-1"])
    assert research_ids == []
    assert chat_ids == ["run-1"]
    assert runs_db.get_run("run-1", "alice") is None


def test_v3_graceful_shutdown_leaves_run_for_receipt_driven_restart(monkeypatch):
    """Quit is not Stop: a v3 run remains recoverable across process shutdown."""
    studio_db.upsert_chat_thread(
        {
            "id": "thread-v3",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local.gguf",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-v3",
            "threadId": "thread-v3",
            "role": "user",
            "content": [{"type": "text", "text": "Keep working"}],
            "createdAt": 2,
        }
    )
    runs_db.create_run(
        run_id="run-v3",
        owner_subject="alice",
        thread_id="thread-v3",
        user_message_id="user-v3",
        assistant_message_id="assistant-v3",
        request_payload={
            "model": "local.gguf",
            "messages": [{"role": "user", "content": "Keep working"}],
            "stream": True,
            "cancel_id": "run-v3",
            "thread_id": "thread-v3",
            "generation_run_id": "run-v3",
            "finalization_idempotency_key": "run-v3",
        },
    )

    async def scenario():
        entered = asyncio.Event()

        async def body(request):
            entered.set()
            while not request.state.generation_cancel_event.is_set():
                await asyncio.sleep(0.01)
            if False:
                yield ""

        async def fake(_payload, request, *_args, **_kwargs):
            return SimpleNamespace(status_code=200, body_iterator=body(request))

        monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
        supervisor.start("run-v3", thread_id="thread-v3", model="local.gguf")
        await asyncio.wait_for(entered.wait(), timeout=2)
        await supervisor.stop()

    asyncio.run(scenario())
    run = runs_db.get_run("run-v3", "alice")
    assert run is not None
    assert run["status"] == "running"
    assert run["cancelRequested"] is False
    assert run["finishReason"] is None

    from core.inference.durable_agent_recovery import build_restart_request

    plan = build_restart_request(run["requestPayload"], runs_db.list_events("run-v3"))
    assert plan.safe is True
    assert plan.reason == "restart_before_output"


def test_recovered_run_rehydrates_model_before_registering_generation(durable_run, monkeypatch):
    resume = {
        "model": "local.gguf",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "cancel_id": "run-1",
        "thread_id": "thread-1",
        "generation_run_id": "run-1",
        "finalization_idempotency_key": "run-1",
    }
    assert runs_db.requeue_run_for_restart("run-1", resume) is not None
    order: list[str] = []
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))

    async def fake_rehydrate(run_id, *, model, owner):
        assert active_generations.count() == 0
        assert (run_id, model, owner) == ("run-1", "local.gguf", "alice")
        order.append("rehydrate")

    def fake_reserve(run_id, *, thread_id=None, model=None):
        order.append("reserve")
        key = _run_key(run_id)
        cancel_event = threading.Event()
        registration = active_generations.ActiveGeneration(
            cancel_event,
            run_id=run_id,
            thread_id=thread_id,
            model=model,
        )
        registration.__enter__()
        supervisor._cancel_events[key] = cancel_event
        supervisor._activities[key] = SimpleNamespace(finish=lambda: None)
        supervisor._active_registrations[key] = registration
        return True

    async def fake_produce(run_id, cancel_event, activity, **kwargs):
        assert run_id == "run-1"
        assert isinstance(cancel_event, threading.Event)
        assert activity is supervisor._activities[_run_key(run_id)]
        assert kwargs.get("worker_payload_override") == resume
        order.append("produce")

    monkeypatch.setattr(supervisor, "_rehydrate_recovery_model", fake_rehydrate)
    monkeypatch.setattr(supervisor, "_ensure_reservation", fake_reserve)
    monkeypatch.setattr(supervisor, "_produce", fake_produce)

    asyncio.run(
        supervisor._rehydrate_and_produce(
            "run-1",
            thread_id="thread-1",
            model="local.gguf",
        )
    )
    assert order == ["rehydrate", "reserve", "produce"]
    supervisor._cleanup_registration("run-1")
    assert active_generations.count() == 0


@pytest.mark.asyncio
async def test_recovered_start_binds_owner_epoch_after_rehydrate_and_cleans_up(
    durable_run, monkeypatch
):
    resume = {
        "model": "local.gguf",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "cancel_id": "run-1",
        "thread_id": "thread-1",
        "generation_run_id": "run-1",
        "finalization_idempotency_key": "run-1",
    }
    recovered = runs_db.requeue_run_for_restart("run-1", resume)
    assert recovered is not None
    worker_token = recovered["_workerToken"]

    active_generations.fence("owner")
    active_generations.lift_fence("owner")
    owner_epoch = active_generations.lifecycle_epoch("owner")
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    monkeypatch.setattr(llama_keepwarm, "_inflight", 0)
    observed: dict[str, int] = {}
    finalized = asyncio.Event()

    async def fake_rehydrate(_run_id, *, model, owner):
        assert (model, owner) == ("local.gguf", "alice")
        observed["load_count"] = active_generations.count()
        observed["load_epoch"] = active_generations.lifecycle_epoch("owner")

    async def fake_produce(_run_id, _cancel_event, _activity, **_kwargs):
        observed["producer_count"] = active_generations.count()
        observed["producer_epoch"] = active_generations.bound_lifecycle_epoch()
        active_generations.fence("owner")
        active_generations.lift_fence("owner")

    async def fake_finalize(_run_id):
        observed["finalizer_count"] = active_generations.count()
        observed["finalizer_epoch"] = active_generations.bound_lifecycle_epoch()
        observed["finalizer_pending"] = llama_keepwarm._pending
        observed["finalizer_inflight"] = llama_keepwarm._inflight
        finalized.set()
        return None

    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(supervisor, "_rehydrate_recovery_model", fake_rehydrate)
    monkeypatch.setattr(supervisor, "_produce", fake_produce)
    monkeypatch.setattr(supervisor, "_finalize", fake_finalize)

    supervisor.start_recovered(
        "run-1",
        thread_id="thread-1",
        model="local.gguf",
        expected_worker_token=worker_token,
    )
    task = supervisor._tasks[_run_key("run-1")]
    await asyncio.wait_for(task, timeout=2)
    await asyncio.wait_for(finalized.wait(), timeout=2)
    # Let the finalizer's done callback remove its bookkeeping before checking
    # the complete release, rather than observing its task in its last moment.
    for _ in range(3):
        await asyncio.sleep(0)

    key = _run_key("run-1")
    assert observed == {
        "load_count": 0,
        "load_epoch": owner_epoch,
        "producer_count": 1,
        "producer_epoch": owner_epoch,
        "finalizer_count": 0,
        "finalizer_epoch": owner_epoch,
        "finalizer_pending": 0,
        "finalizer_inflight": 0,
    }
    assert active_generations.count() == 0
    assert active_generations.lifecycle_epoch("owner") == owner_epoch + 2
    assert all(
        key not in collection
        for collection in (
            supervisor._tasks,
            supervisor._cancel_events,
            supervisor._activities,
            supervisor._active_registrations,
        )
    )


def test_recovery_model_restore_never_evicts_new_foreground_model(durable_run, monkeypatch):
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: "newer-model")
    monkeypatch.setattr(
        inference,
        "_same_loaded_identifier",
        lambda loaded, requested: loaded == requested,
    )

    async def should_not_load(*_args, **_kwargs):
        pytest.fail("background recovery tried to evict the newer foreground model")

    monkeypatch.setattr(inference, "load_model_gated", should_not_load)
    with pytest.raises(RuntimeError, match="different foreground model"):
        asyncio.run(
            supervisor._rehydrate_recovery_model(
                "run-1",
                model="local.gguf",
                owner="alice",
            )
        )


def test_recovery_output_budget_uses_sequential_exact_token_counts(durable_run, monkeypatch):
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    token = runs_db.get_worker_token("run-1")
    assert token
    busy = threading.Lock()

    class FakeBackend:
        def count_chat_tokens(self, messages, *_args, **_kwargs):
            # The production backend has the same single count/generation lock.
            # If the two recovery counts overlap, this test turns that race into
            # a deterministic failure instead of letting the helper fail open.
            assert busy.acquire(blocking=False), "recovery token counts overlapped"
            try:
                time.sleep(0.03)
                trailing = messages[-1].get("content") if messages else ""
                return (30 if trailing else 10), "local.safetensors"
            finally:
                busy.release()

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded=False))
    monkeypatch.setattr(inference, "get_inference_backend", lambda: FakeBackend())
    resume = {
        "model": "local.safetensors",
        "messages": [
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "partial output"},
        ],
        "continue_final_message": True,
        "max_tokens": 100,
    }
    adjusted = asyncio.run(
        supervisor._apply_recovery_output_budget(
            "run-1",
            token,
            original_payload={"max_tokens": 100},
            resume_payload=resume,
        )
    )
    assert adjusted is not None
    assert adjusted["max_tokens"] == 80
    assert resume["max_tokens"] == 100, "the persisted recovery envelope was mutated"
    budget = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "run.recovery_budget"
    ]
    assert budget[-1]["payload"] == {
        "field": "max_tokens",
        "original": 100,
        "already_emitted": 20,
        "remaining": 80,
    }


def test_recovery_output_budget_stops_when_original_allowance_is_spent(durable_run, monkeypatch):
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    token = runs_db.get_worker_token("run-1")
    assert token

    class FakeBackend:
        def count_chat_tokens(self, messages, *_args, **_kwargs):
            trailing = messages[-1].get("content") if messages else ""
            return (35 if trailing else 10), "local.safetensors"

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded=False))
    monkeypatch.setattr(inference, "get_inference_backend", lambda: FakeBackend())
    adjusted = asyncio.run(
        supervisor._apply_recovery_output_budget(
            "run-1",
            token,
            original_payload={"max_tokens": 20},
            resume_payload={
                "messages": [
                    {"role": "user", "content": "Continue"},
                    {"role": "assistant", "content": "already spent"},
                ],
                "continue_final_message": True,
                "max_tokens": 20,
            },
        )
    )
    assert adjusted is None
    budget = [
        event
        for event in runs_db.list_events("run-1")
        if event["type"] == "run.recovery_budget"
    ]
    assert budget[-1]["payload"]["remaining"] == 0


def test_tool_result_recovery_keeps_per_round_output_budget(durable_run, monkeypatch):
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state=SimpleNamespace()))
    token = runs_db.get_worker_token("run-1")
    assert token
    resume = {
        "messages": [
            {"role": "user", "content": "Run it"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ],
        "max_tokens": 100,
    }
    adjusted = asyncio.run(
        supervisor._apply_recovery_output_budget(
            "run-1",
            token,
            original_payload={"max_tokens": 100},
            resume_payload=resume,
        )
    )
    assert adjusted is resume
    assert not any(
        event["type"].startswith("run.recovery_budget")
        for event in runs_db.list_events("run-1")
    )


def test_project_delete_captures_durable_run_before_cascade(durable_run):
    studio_db.upsert_chat_project(
        {"id": "project-1", "name": "Project", "createdAt": 1, "updatedAt": 1}
    )
    studio_db.update_chat_thread("thread-1", {"projectId": "project-1"})
    deleted = studio_db.delete_chat_project("project-1")
    assert deleted["activeChatGenerationRunIds"] == ["run-1"]


def test_clear_captures_durable_run_before_cascade(durable_run):
    removed, research_ids, chat_ids = studio_db.clear_chat_history(
        include_chat_generation_runs = True
    )
    assert (removed, research_ids, chat_ids) == (["thread-1"], [], ["run-1"])


def test_startup_reconcile_marks_stored_assistant_interrupted(durable_run):
    worker_token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", worker_token)
    assert runs_db.reconcile_orphaned_runs() == 1
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"]["generationStatus"] == "failed"
    assert message["metadata"]["incomplete"] == {"reason": "interrupted"}


@pytest.mark.asyncio
async def test_shutdown_returns_even_when_a_producer_will_not_unwind(durable_run, monkeypatch):
    """A generator whose teardown blocks must not take uvicorn's shutdown with it.

    The grace period is bounded, but the gather after task.cancel() has to be too:
    an engine draining a subprocess inside aclose never completes its cancellation,
    and stop() would then wait on it forever.
    """
    import core.inference.chat_generation_runs as chat_generation_runs

    monkeypatch.setattr(chat_generation_runs, "_SHUTDOWN_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(chat_generation_runs, "_SHUTDOWN_CANCEL_SECONDS", 0.5)

    wedged = asyncio.Event()
    release = asyncio.Event()

    async def body():
        yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        wedged.set()
        try:
            await release.wait()
        except (asyncio.CancelledError, GeneratorExit):
            await release.wait()
            raise
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1", thread_id = "thread-1", model = "local.gguf")
    await asyncio.wait_for(wedged.wait(), 10)

    try:
        await asyncio.wait_for(supervisor.stop(), timeout = 10)
    finally:
        release.set()
        await asyncio.sleep(0)
