from core.inference.adaptive_context import (
    adaptive_checkpoint_metadata,
    adaptive_segment_plan,
)


def test_adaptive_segment_cap_reaches_first_token_at_or_above_ratio():
    plan = adaptive_segment_plan(
        prompt_tokens = 3000,
        context_length = 4096,
        requested_max_tokens = 2048,
        ratio = 0.80,
    )

    assert plan is not None
    assert plan.trigger_tokens == 3277
    assert plan.adaptive_room == 277
    assert plan.effective_max_tokens == 277
    assert plan.adaptive_limited is True


def test_user_max_tokens_remains_authoritative_below_adaptive_boundary():
    plan = adaptive_segment_plan(
        prompt_tokens = 600,
        context_length = 1000,
        requested_max_tokens = 100,
        ratio = 0.80,
    )

    assert plan is not None
    assert plan.adaptive_room == 200
    assert plan.effective_max_tokens == 100
    assert plan.adaptive_limited is False
    assert (
        adaptive_checkpoint_metadata(
            plan,
            finish_reason = "length",
            completion_tokens = 100,
        )
        is None
    )


def test_checkpoint_metadata_requires_actual_adaptive_length_stop():
    plan = adaptive_segment_plan(
        prompt_tokens = 750,
        context_length = 1000,
        requested_max_tokens = 500,
        ratio = 0.80,
    )

    assert plan is not None
    assert plan.effective_max_tokens == 50
    assert adaptive_checkpoint_metadata(plan, finish_reason = "stop", completion_tokens = 50) is None
    assert (
        adaptive_checkpoint_metadata(plan, finish_reason = "length", completion_tokens = 49)
        is None
    )

    metadata = adaptive_checkpoint_metadata(
        plan,
        finish_reason = "length",
        completion_tokens = 50,
    )
    assert metadata == {
        "reason": "context_ratio",
        "ratio": 0.80,
        "context_length": 1000,
        "trigger_tokens": 800,
        "prompt_tokens": 750,
        "completion_tokens": 50,
        "occupancy_tokens": 800,
        "segment_max_tokens": 50,
    }


def test_prompt_already_at_boundary_stops_before_decode():
    plan = adaptive_segment_plan(
        prompt_tokens = 800,
        context_length = 1000,
        requested_max_tokens = None,
        ratio = 0.80,
    )

    assert plan is not None
    assert plan.effective_max_tokens == 0
    assert plan.adaptive_limited is True
    assert adaptive_checkpoint_metadata(
        plan,
        finish_reason = "length",
        completion_tokens = 0,
    ) == {
        "reason": "context_ratio",
        "ratio": 0.80,
        "context_length": 1000,
        "trigger_tokens": 800,
        "prompt_tokens": 800,
        "completion_tokens": 0,
        "occupancy_tokens": 800,
        "segment_max_tokens": 0,
    }
