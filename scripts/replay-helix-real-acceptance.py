#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Replay captured real-agent observables through the current Helix controller.

The foreground and self-audit JSON files are produced by
``scripts/collect-helix-acceptance.py``. This replay does not pretend to rerun the
model: it reconstructs only the observable trajectory/tool receipts and executes
the *current* post-task cache/evidence/audit/Hermes controller in an isolated
ledger. That makes post-run controller changes measurable without fabricating a
new agent execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "studio" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.helix_engine.audit import prepare_observable_self_audit  # noqa: E402
from core.helix_engine.pipeline import run_adaptation_pipeline  # noqa: E402
from core.helix_engine.trajectory import ToolStep, Trajectory  # noqa: E402


_AUDIT_RE = re.compile(r"<helix-self-audit>\s*([\s\S]*?)\s*</helix-self-audit>", re.I)
_AUDIT_OPEN_RE = re.compile(r"^\s*<helix-self-audit>\s*([\s\S]*?)\s*$", re.I)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _audit_payload(captured: dict[str, Any]) -> dict[str, Any]:
    text = str(captured.get("assistant_text") or "")
    match = _AUDIT_RE.search(text)
    payload_text = match.group(1) if match else ""
    if not payload_text:
        # Mirror the production parser's bounded recovery: a local model may omit
        # only the closing tag, but no trailing prose or partial JSON is accepted.
        unterminated = _AUDIT_OPEN_RE.fullmatch(text)
        payload_text = unterminated.group(1) if unterminated else ""
    if not payload_text:
        raise ValueError("captured audit does not contain a bounded <helix-self-audit> JSON payload")
    value = json.loads(payload_text)
    if not isinstance(value, dict):
        raise ValueError("self-audit tag did not contain a JSON object")
    return value


def _tool_steps(captured: dict[str, Any]) -> list[ToolStep]:
    starts: dict[str, dict[str, Any]] = {}
    steps: list[ToolStep] = []
    seen: set[tuple[str, str, str]] = set()
    for event in captured.get("tool_events") or []:
        if not isinstance(event, dict):
            continue
        call_id = str(event.get("tool_call_id") or "")
        if event.get("type") == "tool_start":
            starts[call_id] = event
            continue
        if event.get("type") != "tool_end":
            continue
        start = starts.get(call_id, {})
        name = str(event.get("tool_name") or start.get("tool_name") or "")
        arguments = start.get("arguments")
        arguments_text = (
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if isinstance(arguments, dict)
            else str(start.get("arguments_text") or arguments or "")
        )
        result = str(event.get("result") or "")
        key = (name, arguments_text, result[:500])
        hint = "redundant" if key in seen else "evidence"
        # search_memory/search_conversation are equivalent retrieval aliases for
        # duplicate detection, matching cache_integrity.py.
        alias_key = (
            "thread_memory_search" if name in {"search_memory", "search_conversation"} else name,
            arguments_text,
            result[:500],
        )
        if alias_key in seen:
            hint = "redundant"
        seen.add(alias_key)
        steps.append(
            ToolStep(
                name=name,
                arguments=arguments_text,
                result=result,
                useful_hint=hint,
            )
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--parsed-audit",
        type=Path,
        help="Optional production-parser-normalized self-audit JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-id",
        default="helix-final-acceptance-turn2-20260917",
    )
    parser.add_argument(
        "--objective",
        default=(
            "Call search_memory then search_conversation for HELIX-FINAL-1709, then return the codename "
            "and the controlled unsupported 5x claim without a benchmark."
        ),
    )
    parser.add_argument("--model-id", default="Qwythos-9B-v2-MTP-Q4_K_M")
    parser.add_argument(
        "--effective-model-id",
        default="Qwythos-9B-v2-MTP-Q4_K_M::adapter=disabled",
    )
    args = parser.parse_args()

    foreground = _load(args.foreground)
    audit_capture = _load(args.audit)
    audit = _load(args.parsed_audit) if args.parsed_audit else _audit_payload(audit_capture)
    steps = _tool_steps(foreground)
    usage = foreground.get("usage") if isinstance(foreground.get("usage"), dict) else {}
    timings = foreground.get("timings") if isinstance(foreground.get("timings"), dict) else {}
    speculative = (
        foreground.get("speculative") if isinstance(foreground.get("speculative"), dict) else {}
    )
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(
        timings.get("cache_n")
        or ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(usage.get("prompt_tokens_details"), dict) else 0)
        or 0
    )
    telemetry = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "stable_prefix_tokens": cached_tokens,
        "newly_evaluated_tokens": int(timings.get("prompt_n") or max(0, prompt_tokens - cached_tokens)),
        "prefill_ms": timings.get("prompt_ms"),
        "decode_ms": timings.get("predicted_ms"),
        "ttft_ms": foreground.get("first_event_ms"),
        "latency_ms": foreground.get("wall_ms"),
        "speculative_requested": speculative.get("requested") or "",
        "speculative_engaged": speculative.get("engaged") or "",
        "speculative_counter_scope": speculative.get("counter_scope") or "",
        "speculative_counter_reason": speculative.get("counter_reason") or "",
        "accepted_drafts": 0,
        "rejected_drafts": 0,
        "runtime_config": {
            "model": args.model_id,
            "backend": "gguf/llama.cpp",
            "speculative": speculative,
        },
        "telemetry_provenance": {
            "prompt_tokens": "real_openai_usage_chunk",
            "completion_tokens": "real_openai_usage_chunk",
            "cached_tokens": "real_openai_prompt_tokens_details",
            "stable_prefix_tokens": "real_openai_cached_tokens_reuse_observation",
            "newly_evaluated_tokens": "real_llama_timings_prompt_n",
            "prefill_ms": "real_llama_timings_prompt_ms",
            "decode_ms": "real_llama_timings_predicted_ms",
            "ttft_ms": "acceptance_client_first_ui_event",
            "accepted_drafts": "unavailable",
            "rejected_drafts": "unavailable",
        },
    }
    trajectory_id = args.trajectory_id
    traj = Trajectory(
        prompt_state=args.objective,
        retrieved_context="",
        reasoning="",
        steps=steps,
        user_corrections=[],
        final_result=str(foreground.get("assistant_text") or ""),
        latency_ms=float(foreground.get("wall_ms") or 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        verified=True,
        extras={
            "trajectory_id": trajectory_id,
            "model_id": args.effective_model_id,
            "base_model_id": args.model_id,
            "objective_verified": False,
            "telemetry": telemetry,
            "acceptance_criteria": [
                "retrieve HELIX-FINAL-1709",
                "exercise both equivalent retrieval aliases",
                "emit the controlled unsupported 5x claim",
                "do not run a performance benchmark",
            ],
        },
    )

    with tempfile.TemporaryDirectory(prefix="helix-real-replay-") as root:
        os.environ["HELIX_ENGINE_LEDGER_ROOT"] = root
        before = time.perf_counter()
        prepared = prepare_observable_self_audit(traj)
        prep_ms = (time.perf_counter() - before) * 1_000.0
        before = time.perf_counter()
        result = run_adaptation_pipeline(traj, self_audit=audit)
        replay_ms = (time.perf_counter() - before) * 1_000.0
        ledger_bytes = sum(path.stat().st_size for path in Path(root).glob("*.jsonl"))

    payload = {
        "schema_version": "helix.real-acceptance-replay.v1",
        "source": {
            "foreground_schema": foreground.get("schema_version"),
            "audit_schema": audit_capture.get("schema_version"),
            "foreground_wall_ms": foreground.get("wall_ms"),
            "audit_wall_ms": audit_capture.get("wall_ms"),
            "foreground_usage": usage,
            "audit_usage": audit_capture.get("usage"),
            "tool_calls": [step.name for step in steps],
        },
        "pre_audit": prepared,
        "current_controller": result.get("adaptive_cycle"),
        "measurement": {
            "prepare_audit_replay_ms": prep_ms,
            "post_task_replay_ms": replay_ms,
            "isolated_ledger_bytes": ledger_bytes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
