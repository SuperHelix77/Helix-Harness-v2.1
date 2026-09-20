#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Matched post-task control-plane benchmark with an isolated temporary ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "studio" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.helix_engine.ledger import storage_bytes  # noqa: E402
from core.helix_engine.pipeline import run_adaptation_pipeline  # noqa: E402
from core.helix_engine.trajectory import ToolStep, Trajectory  # noqa: E402


WARMUP = 10
ITERATIONS = 100


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def trajectory(index: int) -> Trajectory:
    return Trajectory(
        prompt_state="Inspect a helper, verify it, and do not overclaim performance.",
        retrieved_context="helper.py\ntest_helper.py",
        reasoning="",
        steps=[
            ToolStep(name="read_file", arguments="helper.py", result="def helper(): return 1", useful_hint="evidence"),
            ToolStep(name="read_file", arguments="helper.py", result="def helper(): return 1", useful_hint="redundant"),
            ToolStep(name="grep", arguments="helper", result="helper.py:1", useful_hint="useful"),
            ToolStep(name="run_tests", arguments="pytest test_helper.py", result="1 passed", useful_hint="useful"),
        ],
        user_corrections=[],
        final_result="The helper is present. No performance benchmark was run.",
        latency_ms=100.0,
        prompt_tokens=8_192,
        completion_tokens=256,
        verified=True,
        extras={
            "trajectory_id": f"benchmark-{index}",
            "objective_verified": True,
            "model_id": "benchmark-model",
            "telemetry": {
                "prompt_tokens": 8_192,
                "completion_tokens": 256,
                "cached_tokens": 6_144,
                "prefill_ms": 18.0,
                "decode_ms": 42.0,
                "ttft_ms": 21.0,
                "telemetry_provenance": {
                    "prompt_tokens": "benchmark_fixture",
                    "cached_tokens": "benchmark_fixture",
                    "prefill_ms": "benchmark_fixture",
                    "decode_ms": "benchmark_fixture",
                    "ttft_ms": "benchmark_fixture",
                },
            },
            "acceptance_criteria": ["helper verified"],
        },
    )


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="helix-benchmark-") as ledger_root:
        old_root = os.environ.get("HELIX_ENGINE_LEDGER_ROOT")
        os.environ["HELIX_ENGINE_LEDGER_ROOT"] = ledger_root
        try:
            for index in range(WARMUP):
                run_adaptation_pipeline(trajectory(-index - 1))

            before = storage_bytes()
            wall_ms: list[float] = []
            controller_ms: list[float] = []
            for index in range(ITERATIONS):
                started = time.perf_counter()
                result = run_adaptation_pipeline(trajectory(index))
                wall_ms.append((time.perf_counter() - started) * 1_000.0)
                adaptive = result.get("adaptive_cycle") or {}
                controller_ms.append(float(adaptive.get("controller_overhead_ms") or 0.0))
            after = storage_bytes()
        finally:
            if old_root is None:
                os.environ.pop("HELIX_ENGINE_LEDGER_ROOT", None)
            else:
                os.environ["HELIX_ENGINE_LEDGER_ROOT"] = old_root

    return {
        "schema_version": "helix.control-plane-benchmark.v1",
        "warmup_iterations": WARMUP,
        "iterations": ITERATIONS,
        "post_task_pipeline_wall_ms": {
            "median": statistics.median(wall_ms),
            "p95": percentile(wall_ms, 0.95),
            "mean": statistics.fmean(wall_ms),
        },
        "closed_loop_reported_ms": {
            "median": statistics.median(controller_ms),
            "p95": percentile(controller_ms, 0.95),
            "mean": statistics.fmean(controller_ms),
        },
        "ledger_growth_bytes": after - before,
        "ledger_growth_bytes_per_trajectory": (after - before) / ITERATIONS,
        "notes": [
            "This benchmark excludes generative self-audit latency.",
            "Each iteration uses a unique trajectory id in an isolated temporary ledger.",
            "The workload intentionally contains one duplicate read to exercise cache attribution.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
