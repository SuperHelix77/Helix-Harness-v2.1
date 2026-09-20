# SPDX-License-Identifier: AGPL-3.0-only
"""Optional low-priority model self-audit for the durable turn finalizer.

The audit is enrichment, never finalization authority. It is allowed to run only
against the model that is already resident, never auto-loads/switches, exposes no
tools, and cancels itself as soon as foreground generation appears. Failure or
preemption returns a receipt explaining why the deterministic Helix audit should
remain authoritative.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Any


_AUDIT_PREFIX = "[HELIX_SELF_AUDIT]"
_AUDIT_TAG = "helix-self-audit"
_MAX_ARTIFACT_JSON_CHARS = 28_000
_MAX_AUDIT_SECONDS = 20.0
_FOREGROUND_POLL_SECONDS = 0.05


def _bounded(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:800]
    if depth >= 4:
        return "[depth-limited]"
    if isinstance(value, (list, tuple)):
        items = [_bounded(item, depth + 1) for item in list(value)[:32]]
        if len(value) > len(items):
            items.append(f"[{len(value) - len(items)} more items]")
        return items
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        entries = list(value.items())
        for key, item in entries[:64]:
            if str(key).strip().casefold().replace("-", "_") in {
                "reasoning",
                "chain_of_thought",
                "hidden_reasoning",
            }:
                continue
            out[str(key)[:120]] = _bounded(item, depth + 1)
        if len(entries) > 64:
            out["helix_omitted_keys"] = len(entries) - 64
        return out
    return str(value)[:800]


def build_audit_prompt(focus: str, artifacts: dict[str, Any]) -> str:
    bounded = _bounded(artifacts)
    encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _MAX_ARTIFACT_JSON_CHARS:
        compact = {
            "objective": str(artifacts.get("objective") or "")[:2_000],
            "final_result": str(artifacts.get("final_result") or "")[:2_000],
            "acceptance_criteria": _bounded(artifacts.get("acceptance_criteria") or []),
            "cache_integrity": _bounded(artifacts.get("cache_integrity") or {}),
            "evidence": _bounded(list(artifacts.get("evidence") or [])[:8]),
            "tests": _bounded(list(artifacts.get("tests") or [])[-6:]),
            "benchmarks": _bounded(list(artifacts.get("benchmarks") or [])[-6:]),
            "tool_steps": _bounded(list(artifacts.get("tool_steps") or [])[-8:]),
            "pre_audit_decision": _bounded(artifacts.get("pre_audit_decision") or {}),
            "helix_artifact_compaction": "backend_minimal",
        }
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    schema = (
        '{"objective":"","achieved":true,"contributing_actions":[],"unnecessary_actions":[],'
        '"failures":[],"retries":[],"rediscovered_information":[],"excess_retrieval":[],'
        '"avoidable_cache_disruption":[],"tool_selection_correct":true,'
        '"expensive_resource_misuse":[],"overclaimed_claims":[],"stopped_too_early":false,'
        '"continued_too_long":false,"better_trajectory":[],"reusable_lessons":[],'
        '"likely_behavioral_pattern":false,"recommendation":"IGNORE",'
        '"recommendation_reason":"","self_assessment_confidence":0.5,'
        '"claims":[{"claim_id":"","claim":"","supporting_evidence":[],'
        '"evidence_refs":[],"contradicting_evidence":[],"missing_evidence":[],"confidence":0.5}]}'
    )
    return "\n".join(
        (
            _AUDIT_PREFIX,
            "Audit the completed task using ONLY the observable artifacts below. Do not reveal or reconstruct hidden chain-of-thought.",
            "Judge objective completion, useful/unnecessary actions, failures, retries, retrieval/cache waste, tool choice, unsupported claims, stopping behavior, reusable lessons, and whether any issue is a repeated behavioral pattern.",
            "Your self-report is advisory and cannot authorize training. Backend evidence and held-out verification remain authoritative.",
            "Recommendation must be exactly one of IGNORE, RUNTIME_POLICY, MEMORY, SKILL, QLORA_CANDIDATE, CAPABILITY_GAP.",
            "Return strict valid JSON inside exactly one helix-self-audit tag and no prose outside it.",
            f"Return exactly: <{_AUDIT_TAG}>{schema}</{_AUDIT_TAG}> with the values filled in.",
            f"Task summary: {focus.strip()[:3_000] or 'completed task.'}",
            f"Observable artifacts JSON: {encoded}",
        )
    )


def _claim_strings(value: Any, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:1_000] for item in value[:limit] if isinstance(item, str) and item.strip()]


def parse_audit_text(text: str, *, model_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    match = re.search(
        rf"<{re.escape(_AUDIT_TAG)}>\s*([\s\S]*?)\s*</{re.escape(_AUDIT_TAG)}>",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.match(
            rf"^\s*<{re.escape(_AUDIT_TAG)}>\s*([\s\S]*?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return None
    try:
        raw = json.loads(match.group(1))
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(raw, dict):
        return None

    from core.helix_engine.audit import parse_self_audit

    parsed = parse_self_audit(raw, model_id=model_id)
    if parsed is None:
        return None
    claims: list[dict[str, Any]] = []
    for value in list(raw.get("claims") or [])[:8]:
        if not isinstance(value, dict) or not isinstance(value.get("claim"), str):
            continue
        try:
            confidence = max(0.0, min(1.0, float(value.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        claims.append(
            {
                "claim_id": str(value.get("claim_id") or "")[:200],
                "claim": str(value.get("claim") or "")[:2_000],
                "supporting_evidence": _claim_strings(value.get("supporting_evidence")),
                "evidence_refs": [item[:200] for item in _claim_strings(value.get("evidence_refs"))],
                "contradicting_evidence": _claim_strings(value.get("contradicting_evidence")),
                "missing_evidence": _claim_strings(value.get("missing_evidence")),
                "confidence": confidence,
            }
        )
    return parsed.to_dict(), claims


def _audit_request(app: Any, cancel_event: threading.Event):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/helix-background-audit",
        "raw_path": b"/api/inference/helix-background-audit",
        "query_string": b"",
        "headers": [(b"x-helix-background-audit", b"1")],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "app": app,
        "state": {"generation_cancel_event": cancel_event},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _response_json(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if isinstance(body, memoryview):
        body = body.tobytes()
    if isinstance(body, bytes):
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    return None


async def run_optional_model_audit(
    *,
    app: Any,
    owner: str,
    run_id: str,
    original_model_id: str,
    adapter_state: bool | None,
    focus: str,
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Run one preemptible resident-model audit or return a deterministic-fallback receipt."""
    if app is None:
        return None, [], {"performed": False, "reason": "app_context_unavailable"}

    from routes import inference
    from state import active_generations

    resident = inference._loaded_slot_ident()
    if not resident or not inference._same_loaded_identifier(resident, original_model_id):
        return None, [], {"performed": False, "reason": "original_model_not_resident"}
    if active_generations.count() > 0:
        return None, [], {"performed": False, "reason": "foreground_runtime_busy"}

    cancel_event = threading.Event()
    audit_run_id = f"helix-audit-{run_id}"
    request = _audit_request(app, cancel_event)
    # The resident check above is advisory unless the request itself is fenced:
    # the idle-unload loop could free the model in the narrow gap before inference
    # admission, and reload-only mode would otherwise restore it for a background
    # audit. Audits may borrow a resident model, never cause a load or swap.
    inference.disable_openai_auto_switch_for_request(request.scope)
    from models.inference import ChatCompletionRequest

    prompt = build_audit_prompt(focus, artifacts)
    payload = ChatCompletionRequest.model_validate(
        {
            # Deliberately omit model: _switch_model_for_payload then resolves the
            # reload-only sentinel, which can use the resident weights but cannot
            # auto-switch or reload an older checkpoint.
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 900,
            "enable_tools": False,
            "enable_thinking": False,
            "context_policy": "rolling",
            "use_adapter": adapter_state,
            "cancel_id": audit_run_id,
            "generation_run_id": audit_run_id,
        }
    )

    async def preempt_on_foreground() -> None:
        while not cancel_event.is_set():
            await asyncio.sleep(_FOREGROUND_POLL_SECONDS)
            if any(
                entry.get("run_id") != audit_run_id
                for entry in active_generations.snapshot()
            ):
                cancel_event.set()
                return

    started = time.perf_counter()
    watcher = asyncio.create_task(preempt_on_foreground(), name=f"helix-audit-preempt:{run_id}")
    try:
        try:
            response = await asyncio.wait_for(
                inference.produce_openai_chat_completions(
                    payload,
                    request,
                    owner,
                    cancel_on_disconnect=False,
                ),
                timeout=_MAX_AUDIT_SECONDS,
            )
        except asyncio.TimeoutError:
            cancel_event.set()
            return None, [], {"performed": False, "reason": "audit_timeout"}
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except Exception as exc:
            return None, [], {
                "performed": False,
                "reason": f"audit_generation_failed:{type(exc).__name__}",
            }
        if cancel_event.is_set():
            return None, [], {"performed": False, "reason": "foreground_preempted_audit"}
        resident_after = inference._loaded_slot_ident()
        if not resident_after or not inference._same_loaded_identifier(
            resident_after, original_model_id
        ):
            return None, [], {"performed": False, "reason": "resident_model_changed"}
        response_payload = _response_json(response)
        if not response_payload:
            return None, [], {"performed": False, "reason": "audit_response_unreadable"}
        choices = response_payload.get("choices")
        text = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text = str(message.get("content") or "")
        parsed = parse_audit_text(text, model_id=original_model_id)
        if parsed is None:
            return None, [], {"performed": False, "reason": "audit_parse_failed"}
        audit, claims = parsed
        usage = response_payload.get("usage")
        receipt = {
            "performed": True,
            "duration_ms": max(0, round((time.perf_counter() - started) * 1000)),
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "completion_tokens": (
                usage.get("completion_tokens") if isinstance(usage, dict) else None
            ),
        }
        return audit, claims, receipt
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
