# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only performance facts for the local llama-server runtime."""

from __future__ import annotations

import math
import re
from typing import Any

import httpx


_METRIC_RE = re.compile(
    r"^(?:llamacpp:)?([A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)


def _metrics(backend: Any) -> dict[str, float]:
    base_url = getattr(backend, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return {}
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/metrics",
            headers=getattr(backend, "_auth_headers", {}) or {},
            timeout=3.0,
            trust_env=False,
        )
        if response.status_code != 200:
            return {}
    except Exception:
        return {}
    result: dict[str, float] = {}
    for name, value in _METRIC_RE.findall(response.text):
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            result[name] = number
    return result


def _slots(backend: Any) -> list[dict[str, Any]]:
    base_url = getattr(backend, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return []
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/slots",
            headers=getattr(backend, "_auth_headers", {}) or {},
            timeout=3.0,
            trust_env=False,
        )
        data = response.json() if response.status_code == 200 else []
    except Exception:
        return []
    return data if isinstance(data, list) else []


def read_local_inference_performance() -> str:
    """Return measured runtime state without changing model or generation state."""
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
    except Exception as exc:  # noqa: BLE001 -- tool must remain model-readable
        return f"Local inference status unavailable: {exc}"

    if not getattr(backend, "is_loaded", False):
        return "Local inference status: no llama-server model is loaded."

    metrics = _metrics(backend)
    slots = _slots(backend)
    predicted = metrics.get("tokens_predicted_total")
    predicted_seconds = metrics.get("tokens_predicted_seconds_total")
    measured_rate = (
        predicted / predicted_seconds
        if predicted is not None and predicted_seconds and predicted_seconds > 0
        else metrics.get("predicted_tokens_seconds")
    )
    draft_tokens = metrics.get("spec_decode_num_draft_tokens_total", 0.0)
    accepted_tokens = metrics.get("spec_decode_num_accepted_tokens_total", 0.0)
    drafts = metrics.get("spec_decode_num_drafts_total", 0.0)
    slot_speculative = any(
        bool(slot.get("speculative")) for slot in slots if isinstance(slot, dict)
    )
    engaged = slot_speculative or draft_tokens > 0 or accepted_tokens > 0
    requested = getattr(backend, "requested_spec_mode", None) or "auto"
    resolved = getattr(backend, "speculative_type", None) or "off"
    context = getattr(backend, "context_length", None) or "unknown"
    lines = [
        "Local inference status (read-only)",
        f"Model: {getattr(backend, 'model_identifier', None) or getattr(backend, 'active_model_name', None) or 'unknown'}",
        f"Context: {context} tokens",
        f"Requested speculative mode: {requested}",
        f"Engaged speculative mode: {resolved if engaged else 'off'}",
        f"Speculative server slot: {'on' if slot_speculative else 'off'}",
        f"Draft tokens / accepted tokens / drafts: {int(draft_tokens)} / {int(accepted_tokens)} / {int(drafts)}",
    ]
    if measured_rate is not None and math.isfinite(float(measured_rate)):
        lines.append(f"Measured cumulative decode rate: {float(measured_rate):.2f} tok/s")
    fallback = getattr(backend, "spec_fallback_reason", None)
    if fallback:
        lines.append(f"Speculative fallback reason: {fallback}")
    lines.append(
        "Interpretation: a multiplier is not active unless the slot is on and accepted draft tokens are non-zero."
    )
    return "\n".join(lines)
