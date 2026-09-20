# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

from starlette.responses import JSONResponse

from core.inference import turn_finalizer_audit as audit
from state import active_generations


def _valid_audit_text() -> str:
    return (
        '<helix-self-audit>{"objective":"finish","achieved":true,'
        '"recommendation":"IGNORE","recommendation_reason":"done",'
        '"self_assessment_confidence":0.7,"claims":[{"claim_id":"c1",'
        '"claim":"tests passed","supporting_evidence":["pytest"],'
        '"evidence_refs":["tool:1:verification"],"confidence":0.8}]}'
        '</helix-self-audit>'
    )


def test_audit_prompt_excludes_hidden_reasoning_and_bounds_artifacts():
    prompt = audit.build_audit_prompt(
        "fix bug",
        {
            "objective": "fix bug",
            "trajectory": {
                "reasoning": "SECRET_CHAIN_OF_THOUGHT",
                "chain-of-thought": "ANOTHER_SECRET",
                "tool_steps": [{"name": "test"}],
            },
            "final_result": "done",
            "oversized": "x" * 50_000,
        },
    )
    assert "SECRET_CHAIN_OF_THOUGHT" not in prompt
    assert "ANOTHER_SECRET" not in prompt
    assert "Do not reveal or reconstruct hidden chain-of-thought" in prompt
    assert len(prompt) < 40_000


def test_audit_parser_validates_report_and_bounds_claims():
    parsed = audit.parse_audit_text(_valid_audit_text(), model_id="local")
    assert parsed is not None
    report, claims = parsed
    assert report["source"] == "model"
    assert report["recommendation"] == "IGNORE"
    assert claims == [
        {
            "claim_id": "c1",
            "claim": "tests passed",
            "supporting_evidence": ["pytest"],
            "evidence_refs": ["tool:1:verification"],
            "contradicting_evidence": [],
            "missing_evidence": [],
            "confidence": 0.8,
        }
    ]
    assert audit.parse_audit_text("not json", model_id="local") is None


def test_optional_audit_requires_app_context():
    result = asyncio.run(
        audit.run_optional_model_audit(
            app=None,
            owner="alice",
            run_id="run-1",
            original_model_id="local",
            adapter_state=False,
            focus="done",
            artifacts={"objective": "done"},
        )
    )
    assert result[0] is None
    assert result[2]["reason"] == "app_context_unavailable"


def test_optional_audit_uses_resident_model_without_auto_switch(monkeypatch):
    from routes import inference

    active_generations.reset_for_tests()
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: "local-model")
    monkeypatch.setattr(
        inference,
        "_same_loaded_identifier",
        lambda loaded, requested: loaded == requested,
    )
    observed = {}

    async def fake_produce(payload, request, subject, *, cancel_on_disconnect):
        observed["model_fields_set"] = set(payload.model_fields_set)
        observed["header"] = request.headers.get("X-Helix-Background-Audit")
        observed["auto_switch_disabled"] = bool(
            request.scope.get("_unsloth_disable_openai_auto_switch")
        )
        observed["subject"] = subject
        observed["cancel_on_disconnect"] = cancel_on_disconnect
        return JSONResponse(
            {
                "choices": [{"message": {"content": _valid_audit_text()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake_produce)
    report, claims, receipt = asyncio.run(
        audit.run_optional_model_audit(
            app=SimpleNamespace(state=SimpleNamespace()),
            owner="alice",
            run_id="run-1",
            original_model_id="local-model",
            adapter_state=False,
            focus="done",
            artifacts={"objective": "done"},
        )
    )
    assert report is not None and claims
    assert receipt["performed"] is True
    assert "model" not in observed["model_fields_set"], "background audit must be reload-only"
    assert observed["auto_switch_disabled"] is True, "background audit must never auto-load"
    assert observed["header"] == "1"
    assert observed["subject"] == "alice"
    assert observed["cancel_on_disconnect"] is False


def test_optional_audit_stands_down_when_foreground_is_active(monkeypatch):
    from routes import inference

    active_generations.reset_for_tests()
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: "local-model")
    monkeypatch.setattr(inference, "_same_loaded_identifier", lambda *_args: True)
    called = False

    async def fake_produce(*_args, **_kwargs):
        nonlocal called
        called = True
        return JSONResponse({})

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake_produce)
    event = threading.Event()
    with active_generations.ActiveGeneration(event, run_id="foreground", model="local-model"):
        report, claims, receipt = asyncio.run(
            audit.run_optional_model_audit(
                app=SimpleNamespace(state=SimpleNamespace()),
                owner="alice",
                run_id="run-1",
                original_model_id="local-model",
                adapter_state=False,
                focus="done",
                artifacts={"objective": "done"},
            )
        )
    assert report is None and claims == []
    assert receipt["reason"] == "foreground_runtime_busy"
    assert called is False
    active_generations.reset_for_tests()
