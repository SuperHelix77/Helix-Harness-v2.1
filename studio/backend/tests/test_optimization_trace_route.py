# SPDX-License-Identifier: AGPL-3.0-only
"""Focused route-boundary tests for optional MLX optimization tracing."""

from __future__ import annotations

import json

import pytest

from core.helix_engine import optimization_trace
from routes.inference import (
    _begin_optimization_trace_invocation,
    _classify_plain_optimization_trace_invocation,
    _encode_optimization_trace_private_sse,
    _optimization_trace_private_sse,
    _sf_heal_events_to_sse,
)


class _Handle:
    def __init__(self, *, fail=False):
        self.calls = []
        self._should_fail = fail

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if self._should_fail:
                raise RuntimeError("trace backend failed")

        return call


class _Recorder:
    def __init__(self, handle=None, *, begin_error=None):
        self.handle = handle or _Handle()
        self.begin_error = begin_error
        self.invalid_reason = None

    def begin_invocation(self):
        if self.begin_error is not None:
            raise self.begin_error
        return self.handle

    def invalidate(self, reason):
        if self.invalid_reason is None:
            self.invalid_reason = reason


def test_plain_length_segment_is_continuation_even_when_visible():
    recorder = _Recorder()
    handle = recorder.begin_invocation()

    result = _classify_plain_optimization_trace_invocation(
        recorder,
        handle,
        finish_reason="length",
        accepted_visible_output=True,
        cancelled=False,
    )

    assert result is None
    assert handle.calls == [("complete_continuation", ())]


def test_healed_visible_prose_with_stop_is_final_answer():
    state = {"idx": 0, "accepted_visible_output": False}
    lines = _sf_heal_events_to_sse(
        [("text", "Recovered ordinary prose.")],
        "chatcmpl-test",
        1,
        "mlx-test",
        state,
        True,
    )
    recorder = _Recorder()
    handle = recorder.begin_invocation()

    _classify_plain_optimization_trace_invocation(
        recorder,
        handle,
        finish_reason="stop",
        accepted_visible_output=state["accepted_visible_output"],
        cancelled=False,
    )

    assert lines
    assert handle.calls == [("complete_final_answer", ())]


def test_accepted_healed_tool_plan_wins_over_visible_preamble():
    recorder = _Recorder()
    handle = recorder.begin_invocation()

    _classify_plain_optimization_trace_invocation(
        recorder,
        handle,
        finish_reason="tool_calls",
        accepted_visible_output=True,
        cancelled=False,
    )

    assert handle.calls == [("complete_action_plan", ())]


def test_reasoning_only_stop_is_not_final_answer():
    recorder = _Recorder()
    handle = recorder.begin_invocation()

    _classify_plain_optimization_trace_invocation(
        recorder,
        handle,
        finish_reason="stop",
        accepted_visible_output=False,
        cancelled=False,
    )

    assert handle.calls == [("complete_wasted", ("empty_user_visible_output",))]


@pytest.mark.parametrize(
    ("finish_reason", "expected_call", "expected_invalid_reason"),
    [
        ("error", ("fail", ("error_finish_reason",)), None),
        (
            "content_filter",
            ("complete_wasted", ("content_filtered_output",)),
            None,
        ),
        ("interrupted", ("cancel", ("interrupted_finish_reason",)), None),
        (
            "unknown",
            ("discard", ("unclassified_finish_reason",)),
            "unclassified_finish_reason",
        ),
        (
            None,
            ("discard", ("unclassified_finish_reason",)),
            "unclassified_finish_reason",
        ),
        (
            "future_reason",
            ("discard", ("unclassified_finish_reason",)),
            "unclassified_finish_reason",
        ),
    ],
)
def test_visible_output_with_non_stop_finish_is_never_final_answer(
    finish_reason,
    expected_call,
    expected_invalid_reason,
):
    recorder = _Recorder()
    handle = recorder.begin_invocation()

    _classify_plain_optimization_trace_invocation(
        recorder,
        handle,
        finish_reason=finish_reason,
        accepted_visible_output=True,
        cancelled=False,
    )

    assert handle.calls == [expected_call]
    assert recorder.invalid_reason == expected_invalid_reason
    assert all(method != "complete_final_answer" for method, _args in handle.calls)


def test_plain_trace_begin_and_finalize_failures_are_non_throwing():
    begin_broken = _Recorder(begin_error=RuntimeError("cap reached"))
    assert _begin_optimization_trace_invocation(begin_broken) is None
    assert begin_broken.invalid_reason == "trace_begin_failed"

    finalize_broken = _Recorder(handle=_Handle(fail=True))
    handle = finalize_broken.begin_invocation()
    assert (
        _classify_plain_optimization_trace_invocation(
            finalize_broken,
            handle,
            finish_reason="stop",
            accepted_visible_output=True,
            cancelled=False,
        )
        is None
    )
    assert finalize_broken.invalid_reason == "trace_finalize_failed"


def test_private_sse_cap_counts_actual_utf8_wrapper_bytes_and_not_ascii_escapes():
    small = _encode_optimization_trace_private_sse({"label": "é"})
    assert "é" in small
    assert "\\u00e9" not in small
    assert len(small.encode("utf-8")) <= optimization_trace.MAX_OPTIMIZATION_TRACE_WIRE_BYTES

    # Each character is two UTF-8 bytes. The data/SSE and private-key wrappers
    # make this cross the cap even though the raw character count is below it.
    too_large = {
        "label": "é" * (optimization_trace.MAX_OPTIMIZATION_TRACE_WIRE_BYTES // 2)
    }
    with pytest.raises(ValueError, match="wire byte cap"):
        _encode_optimization_trace_private_sse(too_large)


def test_private_sse_oversize_and_serialization_failures_fall_back_unavailable(
    monkeypatch,
):
    class OversizeEnvelope:
        def to_dict(self):
            return {
                "label": "é"
                * (optimization_trace.MAX_OPTIMIZATION_TRACE_WIRE_BYTES // 2)
            }

    recorder = _Recorder()
    monkeypatch.setattr(
        optimization_trace,
        "optimization_trace_envelope",
        lambda _recorder: OversizeEnvelope(),
    )
    frame = _optimization_trace_private_sse(recorder)

    assert frame is not None
    assert len(frame.encode("utf-8")) <= optimization_trace.MAX_OPTIMIZATION_TRACE_WIRE_BYTES
    payload = json.loads(frame.removeprefix("data: ").strip())
    wire = payload[optimization_trace.OPTIMIZATION_TRACE_PRIVATE_FRAME_KEY]
    assert wire["status"] == "unavailable"
    assert wire["error"] == "trace_serialization_failed"
    assert recorder.invalid_reason == "trace_serialization_failed"

    monkeypatch.setattr(
        optimization_trace,
        "optimization_trace_envelope",
        lambda _recorder: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )
    second = _optimization_trace_private_sse(_Recorder())
    assert second is not None
    assert '"status":"unavailable"' in second


def test_private_sse_default_off_emits_nothing():
    assert _optimization_trace_private_sse(None) is None
