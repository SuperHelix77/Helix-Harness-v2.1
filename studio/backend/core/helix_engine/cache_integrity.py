# SPDX-License-Identifier: AGPL-3.0-only
"""Cache/context efficiency observation and attribution. Observation only."""

from __future__ import annotations

from typing import Any

from .schemas import CacheCause, CacheDisruption, CacheIntegrityReport
from .trajectory import ToolStep


def _int(data: dict[str, Any], *names: str) -> int:
    for name in names:
        value = data.get(name)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _float(data: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = data.get(name)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _bool_count(data: dict[str, Any], singular: str, plural: str) -> int:
    if plural in data:
        return _int(data, plural)
    return 1 if data.get(singular) is True else 0


def _repeat_tool_key(step: ToolStep) -> tuple[str, str, str]:
    # These are two public names for the same thread-memory retrieval path in
    # core.inference.tools.  Calling both with the same arguments and receiving
    # the same leading result is objectively duplicate retrieval, even though the
    # model-visible tool names differ.
    normalized_name = (
        "thread_memory_search"
        if step.name in {"search_memory", "search_conversation"}
        else step.name
    )
    return (normalized_name, step.arguments, step.result[:500])


def build_cache_integrity_report(
    telemetry: dict[str, Any] | None,
    steps: list[ToolStep],
) -> CacheIntegrityReport:
    data = telemetry if isinstance(telemetry, dict) else {}
    prompt_tokens = _int(data, "prompt_tokens", "promptTokens", "prompt_n")
    cached_tokens = _int(data, "cached_tokens", "cachedTokens", "cache_n")
    explicit_stable = any(name in data for name in ("stable_prefix_tokens", "stablePrefixTokens"))
    stable_prefix = _int(data, "stable_prefix_tokens", "stablePrefixTokens") or cached_tokens
    explicit_new = any(name in data for name in ("newly_evaluated_tokens", "prefill_tokens"))
    newly_evaluated = _int(data, "newly_evaluated_tokens", "prefill_tokens")
    if not newly_evaluated and prompt_tokens:
        newly_evaluated = max(0, prompt_tokens - cached_tokens)
    ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0

    raw_provenance = data.get("telemetry_provenance") or data.get("telemetryProvenance") or {}
    provenance = (
        {str(key)[:120]: str(value)[:240] for key, value in raw_provenance.items()}
        if isinstance(raw_provenance, dict)
        else {}
    )
    if not explicit_stable and cached_tokens:
        provenance.setdefault("stable_prefix_tokens", "inferred_from_cached_tokens")
    if not explicit_new and prompt_tokens:
        provenance.setdefault("newly_evaluated_tokens", "derived_prompt_minus_cached")
    if prompt_tokens:
        provenance.setdefault("cache_reuse_ratio", "derived_cached_over_prompt")

    disruptions: list[CacheDisruption] = []
    seen: dict[tuple[str, str, str], int] = {}
    repeated = 0
    backend_insertions: list[str] = []
    for step in steps:
        if not step.error and step.result:
            backend_insertions.append(f"tool_output:{step.name}"[:160])
        key = _repeat_tool_key(step)
        if key in seen:
            repeated += 1
            disruptions.append(
                CacheDisruption(
                    cause=CacheCause.MODEL_CAUSED,
                    action=f"repeat_tool:{step.name}",
                    necessary=False,
                    estimated_cost_tokens=max(1, len(step.result) // 4),
                    evidence=["equivalent retrieval path, arguments, and leading result already occurred in this trajectory"],
                    provenance="backend_tool_capture",
                )
            )
        seen[key] = seen.get(key, 0) + 1
    if backend_insertions:
        provenance.setdefault("context_insertions", "backend_tool_capture")
    if repeated:
        provenance.setdefault("repeated_context_insertions", "backend_tool_capture")

    compactions = _bool_count(data, "context_compaction", "context_compactions")
    if compactions:
        truncation = data.get("context_truncation")
        pressure = bool(data.get("context_limit_reached"))
        if isinstance(truncation, dict):
            before = _int(truncation, "prompt_tokens_before")
            target = _int(truncation, "prompt_target")
            pressure = pressure or truncation.get("fits") is False or bool(
                before and target and before > target
            )
        disruptions.append(
            CacheDisruption(
                cause=CacheCause.NECESSARY if pressure else CacheCause.HARNESS_CAUSED,
                action="context_compaction",
                necessary=pressure,
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["context compaction was reported by the chat runtime"],
                provenance=provenance.get("context_compactions", "chat_runtime"),
            )
        )
    reconstructions = _bool_count(data, "prompt_reconstruction", "prompt_reconstructions")
    if reconstructions:
        disruptions.append(
            CacheDisruption(
                cause=CacheCause.HARNESS_CAUSED,
                action="prompt_reconstruction",
                necessary=bool(data.get("prompt_reconstruction_required")),
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["prompt reconstruction was reported"],
                provenance=provenance.get("prompt_reconstructions", "chat_runtime"),
            )
        )
    system_changes = _bool_count(data, "system_prompt_change", "system_prompt_changes")
    if system_changes:
        disruptions.append(
            CacheDisruption(
                cause=CacheCause.USER_CAUSED if data.get("user_changed_system_prompt") else CacheCause.HARNESS_CAUSED,
                action="system_prompt_change",
                necessary=bool(data.get("user_changed_system_prompt")),
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["system prompt changed between cache-compatible requests"],
                provenance=provenance.get("system_prompt_changes", "context_fingerprint"),
            )
        )
    schema_changes = _bool_count(data, "tool_schema_change", "tool_schema_changes")
    if schema_changes:
        try:
            schema_cause = CacheCause(str(data.get("tool_schema_change_cause") or "HARNESS_CAUSED").upper())
        except ValueError:
            schema_cause = CacheCause.UNKNOWN
        disruptions.append(
            CacheDisruption(
                cause=schema_cause,
                action="tool_schema_change",
                necessary=bool(data.get("tool_schema_change_required")),
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["tool schema changed between requests"],
                provenance=provenance.get("tool_schema_changes", "runtime_tool_catalog"),
            )
        )

    kv_resets = _bool_count(data, "kv_cache_reset", "kv_cache_resets")
    if kv_resets:
        raw_cause = str(data.get("kv_cache_reset_cause") or "UNKNOWN").upper()
        try:
            reset_cause = CacheCause(raw_cause)
        except ValueError:
            reset_cause = CacheCause.UNKNOWN
        required = bool(data.get("kv_cache_reset_required"))
        disruptions.append(
            CacheDisruption(
                cause=reset_cause,
                action="kv_cache_reset",
                necessary=required,
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["runtime reported a KV-cache reset/invalidation"],
                provenance=provenance.get("kv_cache_resets", "runtime"),
            )
        )

    raw_insertions = data.get("context_insertions") or data.get("contextInsertions") or []
    if not isinstance(raw_insertions, list):
        raw_insertions = [raw_insertions] if raw_insertions else []
    insertions: list[str] = list(backend_insertions)
    repeated_context = _int(data, "repeated_context_insertions")
    for item in raw_insertions[:32]:
        if isinstance(item, dict):
            kind = str(item.get("kind") or item.get("source") or "context")[:80]
            insertions.append(kind)
            if item.get("repeated") is True:
                repeated_context += 1
                disruptions.append(
                    CacheDisruption(
                        cause=CacheCause.TOOL_CAUSED if kind in {"tool_output", "retrieval"} else CacheCause.HARNESS_CAUSED,
                        action=f"repeat_context_insertion:{kind}",
                        necessary=bool(item.get("required")),
                        estimated_cost_tokens=max(0, int(item.get("estimated_tokens") or 0)),
                        evidence=["context insertion fingerprint was already observed"],
                        provenance=str(item.get("provenance") or "context_collector")[:240],
                    )
                )
        else:
            insertions.append(str(item)[:160])

    reorders = _bool_count(data, "context_reorder", "context_reorders")
    if reorders:
        disruptions.append(
            CacheDisruption(
                cause=CacheCause.HARNESS_CAUSED,
                action="context_reorder",
                necessary=bool(data.get("context_reorder_required")),
                estimated_cost_tokens=max(0, newly_evaluated),
                evidence=["context order changed while the runtime retained ordering provenance"],
                provenance=provenance.get("context_reorders", "context_collector"),
            )
        )

    observable = []
    for field, value in (
        ("prompt_tokens", prompt_tokens),
        ("cached_tokens", cached_tokens),
        ("prefill_ms", _float(data, "prefill_ms", "prompt_ms")),
        ("decode_ms", _float(data, "decode_ms", "predicted_ms")),
        ("ttft_ms", _float(data, "ttft_ms", "firstTokenTime")),
        ("accepted_drafts", _int(data, "accepted_drafts", "acceptedDrafts")),
    ):
        if value not in (None, 0, 0.0, "") or field in data:
            observable.append(field)

    runtime = data.get("runtime_config") or data.get("runtimeConfig") or {}
    speculative_counter_scope = str(
        data.get("speculative_counter_scope")
        or data.get("speculativeCounterScope")
        or ""
    ).strip().lower()
    if speculative_counter_scope == "unavailable":
        provenance["accepted_drafts"] = "unavailable_request_scoped_counter"
        provenance["rejected_drafts"] = "unavailable_request_scoped_counter"
    core_fields = [
        "prompt_tokens", "stable_prefix_tokens", "newly_evaluated_tokens", "cached_tokens",
        "kv_cache_resets", "context_compactions", "prompt_reconstructions", "system_prompt_changes",
        "tool_schema_changes", "context_insertions", "repeated_context_insertions", "prefill_ms",
        "decode_ms", "ttft_ms", "context_reorders", "speculative_requested", "speculative_engaged",
        "accepted_drafts", "rejected_drafts", "runtime_config",
    ]
    for field in core_fields:
        source = provenance.get(field, "")
        explicitly_unavailable = source == "unavailable" or source.startswith("unavailable_")
        if (field in data or field in provenance) and not explicitly_unavailable:
            if field not in observable:
                observable.append(field)
            provenance.setdefault(field, "runtime_or_request_metadata")
    observable = [
        field
        for field in observable
        if not (
            provenance.get(field, "") == "unavailable"
            or provenance.get(field, "").startswith("unavailable_")
        )
    ]
    unavailable = [field for field in core_fields if field not in observable]
    for field in unavailable:
        provenance.setdefault(field, "unavailable")
    return CacheIntegrityReport(
        prompt_tokens=prompt_tokens,
        stable_prefix_tokens=stable_prefix,
        newly_evaluated_tokens=newly_evaluated,
        cached_tokens=cached_tokens,
        cache_reuse_ratio=max(0.0, min(1.0, ratio)),
        kv_cache_resets=kv_resets,
        context_compactions=compactions,
        prompt_reconstructions=reconstructions,
        system_prompt_changes=system_changes,
        tool_schema_changes=schema_changes,
        context_insertions=[str(item)[:160] for item in insertions[:32]],
        repeated_context_insertions=repeated + repeated_context,
        prefill_ms=_float(data, "prefill_ms", "prompt_ms"),
        decode_ms=_float(data, "decode_ms", "predicted_ms"),
        ttft_ms=_float(data, "ttft_ms", "firstTokenTime"),
        context_reorders=reorders,
        speculative_requested=str(data.get("speculative_requested") or ""),
        speculative_engaged=str(data.get("speculative_engaged") or ""),
        accepted_drafts=_int(data, "accepted_drafts", "acceptedDrafts"),
        rejected_drafts=_int(data, "rejected_drafts", "rejectedDrafts"),
        runtime_config=dict(runtime) if isinstance(runtime, dict) else {},
        disruptions=disruptions,
        observable_fields=observable,
        telemetry_provenance=provenance,
        unavailable_fields=unavailable,
    )
