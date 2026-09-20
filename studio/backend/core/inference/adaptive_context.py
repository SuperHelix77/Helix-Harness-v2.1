# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exact-token generation caps for Studio's adaptive context checkpoint."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Optional


DEFAULT_ADAPTIVE_CONTEXT_RATIO = 0.80


@dataclass(frozen = True)
class AdaptiveSegmentPlan:
    """One decode segment's context-ratio boundary.

    ``effective_max_tokens`` may be zero when the rendered prompt has already
    reached the boundary. ``adaptive_limited`` is true only when this boundary
    is stricter than the caller's own token limit, which lets telemetry tell an
    adaptive checkpoint apart from an ordinary Max Tokens stop.
    """

    context_length: int
    prompt_tokens: int
    ratio: float
    trigger_tokens: int
    adaptive_room: int
    requested_max_tokens: Optional[int]
    effective_max_tokens: int
    adaptive_limited: bool


def adaptive_segment_plan(
    *,
    prompt_tokens: int,
    context_length: Optional[int],
    requested_max_tokens: Optional[int],
    ratio: Optional[float],
) -> Optional[AdaptiveSegmentPlan]:
    """Cap a single generation segment at the requested context occupancy.

    The boundary uses ``ceil`` so a ratio such as 0.80 stops on the first token
    at or above 80%, never just below it. A caller token cap that is equal to or
    smaller than the remaining adaptive room stays authoritative and therefore
    is not reported as an adaptive checkpoint.
    """

    if ratio is None or context_length is None:
        return None
    try:
        context = int(context_length)
        prompt = max(0, int(prompt_tokens))
        ratio_value = float(ratio)
    except (TypeError, ValueError, OverflowError):
        return None
    if context <= 0 or not math.isfinite(ratio_value) or not 0.0 < ratio_value < 1.0:
        return None

    trigger = min(context, max(1, int(math.ceil(context * ratio_value))))
    room = max(0, trigger - prompt)
    requested = None
    if requested_max_tokens is not None:
        try:
            requested = max(0, int(requested_max_tokens))
        except (TypeError, ValueError, OverflowError):
            requested = None

    adaptive_limited = requested is None or requested > room
    effective = room if adaptive_limited else requested
    return AdaptiveSegmentPlan(
        context_length = context,
        prompt_tokens = prompt,
        ratio = ratio_value,
        trigger_tokens = trigger,
        adaptive_room = room,
        requested_max_tokens = requested,
        effective_max_tokens = int(effective),
        adaptive_limited = adaptive_limited,
    )


def adaptive_checkpoint_metadata(
    plan: Optional[AdaptiveSegmentPlan],
    *,
    finish_reason: Optional[str],
    completion_tokens: Optional[int] = None,
) -> Optional[dict]:
    """Return Studio checkpoint telemetry when the adaptive cap actually stopped decode."""

    if plan is None or not plan.adaptive_limited:
        return None
    if plan.effective_max_tokens > 0 and finish_reason != "length":
        return None
    if completion_tokens is not None:
        try:
            completion = max(0, int(completion_tokens))
        except (TypeError, ValueError, OverflowError):
            completion = 0
        if plan.effective_max_tokens > 0 and completion < plan.effective_max_tokens:
            return None
    else:
        completion = plan.effective_max_tokens

    occupancy = min(plan.context_length, plan.prompt_tokens + completion)
    return {
        "reason": "context_ratio",
        "ratio": plan.ratio,
        "context_length": plan.context_length,
        "trigger_tokens": plan.trigger_tokens,
        "prompt_tokens": plan.prompt_tokens,
        "completion_tokens": completion,
        "occupancy_tokens": occupancy,
        "segment_max_tokens": plan.effective_max_tokens,
    }

