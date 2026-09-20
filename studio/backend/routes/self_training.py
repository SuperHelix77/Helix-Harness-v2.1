# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Safe self-QLoRA orchestration.

Completed local chat turns can be collected as ChatML examples. The original
model identity is retained as the baseline and adapter promotion is gated by an
explicit held-out score/throughput comparison. This route never overwrites the
base model. Autonomous promotion is allowed only after a Hermes-qualified run,
under the user's explicit autonomous + QLoRA policy gates.
"""

from __future__ import annotations

import json
import hashlib
import asyncio
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from models.training import TrainingStartRequest
from utils.paths import account_path, datasets_root, ensure_dir

router = APIRouter()

_STATE_LOCK = threading.RLock()
_MAX_EXAMPLES = 256
_MIN_EXAMPLES = 4
_MAX_MIN_EXAMPLES = 64
_PROMOTION_MARGIN = 0.01
_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class _BenchmarkTask(BaseModel):
    id: str
    prompt: str
    expected: str
    kind: Literal["contains", "json", "ordered"]

# Fixed held-out checks are intentionally outside the model. The candidate cannot
# award itself a score or write the rubric; the harness scores both variants using
# the same deterministic tasks.
_HOLDOUT_BENCHMARK: tuple[_BenchmarkTask, ...] = (
    _BenchmarkTask(
        id = "arithmetic",
        prompt = "Compute 17 * 23. Return only the integer.",
        expected = "391",
        kind = "contains",
    ),
    _BenchmarkTask(
        id = "json-contract",
        prompt = "Return exactly one JSON object with keys name and count. Use name=mlx and count=64.",
        expected = '{"name":"mlx","count":64}',
        kind = "json",
    ),
    _BenchmarkTask(
        id = "ordered-instructions",
        prompt = "Write these three tokens in this exact order, separated by commas: alpha, beta, gamma. Return nothing else.",
        expected = "alpha,beta,gamma",
        kind = "ordered",
    ),
    _BenchmarkTask(
        id = "context-policy",
        prompt = "What is the safe default for a candidate adapter when its benchmark score falls? Answer in one short sentence and include the words revert and retrain.",
        expected = "revert retrain",
        kind = "contains",
    ),
)


class SelfTrainingConfigRequest(BaseModel):
    enabled: bool | None = None
    autoTrain: bool | None = None
    minExamples: int | None = Field(default = None, ge = _MIN_EXAMPLES, le = _MAX_MIN_EXAMPLES)
    maxSeqLength: int | None = Field(default = None, ge = 512, le = 65_536)


class BaselineRequest(BaseModel):
    modelId: str = Field(min_length = 1, max_length = 500)
    snapshotPath: str | None = Field(default = None, max_length = 4_096)
    contextLength: int | None = Field(default = None, ge = 512, le = 65_536)


class SelfTrainingExampleRequest(BaseModel):
    modelId: str = Field(min_length = 1, max_length = 500)
    prompt: str = Field(min_length = 1, max_length = 20_000)
    completion: str = Field(min_length = 1, max_length = 24_000)
    sourceThreadId: str | None = Field(default = None, max_length = 200)
    idempotencyKey: str | None = Field(default = None, max_length = 200)
    score: float | None = Field(default = None, ge = 0, le = 1)
    critique: str | None = Field(default = None, max_length = 4_000)
    eligibleForTraining: bool = False
    sourceTrajectoryId: str | None = Field(default = None, max_length = 200)
    evidenceIds: list[str] = Field(default_factory = list, max_length = 64)


class SelfTrainingEvaluationRequest(BaseModel):
    candidateAdapterPath: str = Field(min_length = 1, max_length = 4_096)
    baseScore: float = Field(ge = 0, le = 1)
    candidateScore: float = Field(ge = 0, le = 1)
    baseTokPerSec: float = Field(ge = 0)
    candidateTokPerSec: float = Field(ge = 0)
    evaluator: Literal["human", "holdout"] = "human"
    rubric: str = Field(default = "", max_length = 2_000)


class SelfTrainingBenchmarkRequest(BaseModel):
    candidateAdapterPath: str = Field(min_length = 1, max_length = 4_096)
    adapterName: str | None = Field(default = None, max_length = 64)


class AdapterHotSwapRequest(BaseModel):
    adapterPath: str = Field(min_length = 1, max_length = 4_096)
    adapterName: str | None = Field(default = None, max_length = 64)


class SelfTrainingRecommendationRequest(BaseModel):
    action: Literal["skill", "qlora", "runtime-fix", "none"]
    reason: str = Field(default = "", max_length = 2_000)


def _state_path() -> Path:
    return account_path("learning/self_qlora/state.json")


def _state_db_path() -> Path:
    return _state_path().with_suffix(".sqlite3")


def _open_state_db() -> sqlite3.Connection:
    path = _state_db_path()
    ensure_dir(path.parent)
    conn = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS self_training_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            payload_json TEXT NOT NULL,
            updated_at_ms INTEGER NOT NULL
        )
        """
    )
    return conn


def _read_state_db() -> dict[str, Any] | None:
    try:
        with _open_state_db() as conn:
            row = conn.execute(
                "SELECT payload_json FROM self_training_state WHERE singleton=1"
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row["payload_json"]))
        return value if isinstance(value, dict) else None
    except (OSError, sqlite3.Error, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_state_db(state: dict[str, Any]) -> None:
    payload = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    updated_at = int(state.get("updatedAt") or int(time.time() * 1_000))
    with _open_state_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO self_training_state(singleton, payload_json, updated_at_ms)
            VALUES (1, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                payload_json=excluded.payload_json,
                updated_at_ms=excluded.updated_at_ms
            """,
            (payload, updated_at),
        )
        conn.commit()


def _dataset_path() -> Path:
    return account_path("learning/self_qlora/task_examples.jsonl")


def _empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        # Collection is local and bounded; costly training is separately opt-in.
        "enabled": True,
        "autoTrain": False,
        "minExamples": 8,
        "maxSeqLength": 8_192,
        "baseModelId": None,
        # The checkpoint serving chat can be a GGUF quant while the trainable
        # source remains safetensors/MLX. Keep both identities explicit so an
        # automatically collected turn never feeds a .gguf path to PEFT.
        "baseServingModelId": None,
        "baseSnapshotPath": None,
        "baseServingSnapshotPath": None,
        "baseContextLength": None,
        "activeAdapterPath": None,
        "examples": [],
        "status": "idle",
        "lastJobId": None,
        "lastCandidateAdapterPath": None,
        "lastEvaluation": None,
        "lastRecommendation": None,
        "trainingQualifiedOnly": False,
        "autonomousQueueReceipt": None,
        "lastAutonomousQueueReceipt": None,
        "candidateQualifiedOnly": False,
        "activeServingModelId": None,
        "activeDatasetSnapshot": None,
        "activeDatasetSha256": None,
        "activeDatasetExampleCount": 0,
        "lastError": None,
        "lastRecovery": None,
        "updatedAt": int(time.time() * 1_000),
    }


def _strip_gguf_repo_suffix(repo_id: str) -> str:
    """Map a canonical Hub GGUF distribution repo to its trainable source id."""
    value = str(repo_id or "").strip()
    if value.casefold().endswith("-gguf"):
        return value[:-5]
    return value


def _hf_repo_from_cache_path(path: Path) -> str | None:
    """Recover ``owner/repo`` from a standard Hugging Face cache path."""
    for part in path.parts:
        if not part.startswith("models--"):
            continue
        encoded = part[len("models--") :]
        if "--" not in encoded:
            continue
        owner, repo = encoded.split("--", 1)
        if owner and repo:
            return f"{owner}/{repo}"
    return None


def _trainable_baseline_for_serving_model(model_id: str) -> str:
    """Resolve a serving checkpoint to the safest trainable baseline identity.

    Normal model ids are returned unchanged. A ``*-GGUF`` Hub id, or a local
    GGUF living in Hugging Face's cache, maps narrowly to the corresponding
    source repo by removing only the terminal ``-GGUF`` suffix. Unknown local
    GGUF files remain unchanged and the training preflight rejects them rather
    than guessing a source model.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return raw

    path = Path(raw).expanduser()
    if "/" in raw and not path.exists():
        return _strip_gguf_repo_suffix(raw)
    original_is_gguf = path.suffix.casefold() == ".gguf"
    try:
        resolved = path.resolve(strict=True)
    except (OSError, FileNotFoundError):
        return _strip_gguf_repo_suffix(raw)
    if not original_is_gguf and resolved.suffix.casefold() != ".gguf":
        return raw
    # Snapshot entries are normally symlinks into an extensionless blobs/ file,
    # so inspect both the caller-visible cache path and its resolved target.
    repo = _hf_repo_from_cache_path(path) or _hf_repo_from_cache_path(resolved)
    if repo and repo.casefold().endswith("-gguf"):
        return _strip_gguf_repo_suffix(repo)
    return raw


def _serving_model_matches_baseline(state: dict[str, Any], serving_model_id: str) -> bool:
    baseline = str(state.get("baseModelId") or "").strip()
    serving = str(serving_model_id or "").strip()
    if not baseline:
        return True
    if serving == baseline:
        return True
    if serving == str(state.get("baseServingModelId") or "").strip():
        return True
    return _trainable_baseline_for_serving_model(serving) == baseline


def _example_matches_baseline(example: dict[str, Any], baseline: str) -> bool:
    serving = str(example.get("modelId") or "").strip()
    trainable = str(example.get("trainableBaseModelId") or "").strip()
    return bool(
        baseline
        and (
            trainable == baseline
            or serving == baseline
            or _trainable_baseline_for_serving_model(serving) == baseline
        )
    )


def _migrate_trainable_baseline(state: dict[str, Any]) -> bool:
    """Upgrade legacy state that stored a GGUF serving id as the trainable base."""
    legacy = str(state.get("baseModelId") or "").strip()
    if not legacy:
        return False
    trainable = _trainable_baseline_for_serving_model(legacy)
    if not trainable or trainable == legacy:
        return False
    state["baseModelId"] = trainable
    if not state.get("baseServingModelId"):
        state["baseServingModelId"] = legacy
    if state.get("baseSnapshotPath") and not state.get("baseServingSnapshotPath"):
        state["baseServingSnapshotPath"] = state.get("baseSnapshotPath")
    # A GGUF snapshot cannot be reused by the PEFT/MLX trainer. Let the normal
    # model resolver locate/cache the trainable source instead.
    state["baseSnapshotPath"] = None
    for item in state.get("examples", []):
        if not isinstance(item, dict) or item.get("trainableBaseModelId"):
            continue
        serving = str(item.get("modelId") or "").strip()
        if _trainable_baseline_for_serving_model(serving) == trainable:
            item["trainableBaseModelId"] = trainable
    return True


def _read_state() -> dict[str, Any]:
    file_raw: dict[str, Any] | None = None
    file_error: BaseException | None = None
    try:
        candidate = json.loads(_state_path().read_text(encoding = "utf-8"))
        if isinstance(candidate, dict):
            file_raw = candidate
        else:
            file_error = ValueError("state root is not an object")
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        file_error = error
    db_raw = _read_state_db()
    raw = file_raw
    if db_raw is not None and (
        raw is None
        or int(db_raw.get("updatedAt") or 0) > int(raw.get("updatedAt") or 0)
    ):
        raw = db_raw
    if raw is None:
        if file_error is not None:
            raise HTTPException(
                status_code = 500,
                detail = "The self-QLoRA state could not be read.",
            ) from file_error
        return _empty_state()
    state = _empty_state()
    for key in state:
        if key in raw:
            state[key] = raw[key]
    state["examples"] = [item for item in state.get("examples", []) if isinstance(item, dict)][-_MAX_EXAMPLES:]
    _migrate_trainable_baseline(state)
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    ensure_dir(path.parent)
    state["updatedAt"] = int(time.time() * 1_000)
    # The WAL snapshot is the crash-recovery authority. The JSON file remains a
    # human-readable compatibility view. Readers choose the newest valid copy,
    # so power loss between these two writes converges on restart.
    _write_state_db(state)
    with tempfile.NamedTemporaryFile(
        mode = "w", encoding = "utf-8", dir = path.parent,
        prefix = ".self-qlora-", suffix = ".tmp", delete = False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii = False, indent = 2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _qualified_examples(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state.get("examples", []) if item.get("eligibleForTraining") is True]


def _valid_autonomous_queue_receipt(state: dict[str, Any]) -> bool:
    receipt = state.get("autonomousQueueReceipt")
    if not isinstance(receipt, dict):
        return False
    if receipt.get("provenance") != "final_helix_ingest":
        return False
    trajectory_id = str(receipt.get("sourceTrajectoryId") or "").strip()
    if not trajectory_id:
        return False
    baseline = str(state.get("baseModelId") or "").strip()
    receipt_model = str(receipt.get("modelId") or "").strip()
    if baseline and receipt_model and receipt_model != baseline:
        return False
    return any(
        item.get("eligibleForTraining") is True
        and str(item.get("sourceTrajectoryId") or "").strip() == trajectory_id
        for item in state.get("examples", [])
        if isinstance(item, dict)
    )


def _autonomous_qlora_policy_block_reason(state: dict[str, Any]) -> str | None:
    if not state.get("enabled", True):
        return "self_training_disabled"
    if not state.get("autoTrain"):
        return "automatic_qlora_training_disabled"
    qualified_count = len(_qualified_examples(state))
    minimum = int(state.get("minExamples", 8))
    if qualified_count < minimum:
        return f"needs_more_verified_examples:{qualified_count}/{minimum}"
    try:
        from routes.learning import _read_state as _read_learning_state

        learning = _read_learning_state()
    except Exception:
        return "learning_policy_unavailable"
    if learning.get("decisionMode") != "autonomous":
        return "learning_decision_mode_not_autonomous"
    if learning.get("allowQloraTraining") is not True:
        return "qlora_training_not_allowed_by_learning_policy"
    return None


def _autonomous_qlora_policy_allows(state: dict[str, Any]) -> bool:
    return _autonomous_qlora_policy_block_reason(state) is None


def _autonomous_qlora_admission(state: dict[str, Any]) -> bool:
    """Single admission check for any autonomous training queue transition."""
    return (
        state.get("status") in {"idle", "error", "rejected", "needs-more-data"}
        and _autonomous_qlora_policy_allows(state)
    )


def queue_hermes_training_with_provenance(
    *,
    source_trajectory_id: str | None = None,
    source_thread_id: str | None = None,
) -> dict[str, str]:
    """Atomically evaluate and, when allowed, queue the final Hermes candidate set."""
    with _STATE_LOCK:
        state = _read_state()
        status = str(state.get("status") or "idle")
        trajectory_id = str(source_trajectory_id or "").strip()
        if status == "queued" and trajectory_id:
            receipt = state.get("autonomousQueueReceipt")
            same_queue = bool(
                state.get("trainingQualifiedOnly") is True
                and _valid_autonomous_queue_receipt(state)
                and isinstance(receipt, dict)
                and str(receipt.get("sourceTrajectoryId") or "").strip() == trajectory_id
                and (
                    not str(source_thread_id or "").strip()
                    or not str(receipt.get("sourceThreadId") or "").strip()
                    or str(receipt.get("sourceThreadId") or "").strip()
                    == str(source_thread_id or "").strip()
                )
            )
            if same_queue:
                block_reason = _autonomous_qlora_policy_block_reason(state)
                if block_reason:
                    outcome = (
                        "deferred"
                        if block_reason.startswith("needs_more_verified_examples:")
                        else "denied"
                    )
                    return {"outcome": outcome, "reason": block_reason[:500]}
                return {
                    "outcome": "queued",
                    "reason": "qualified_candidate_queue_already_persisted",
                }
        if status not in {"idle", "error", "rejected", "needs-more-data"}:
            return {
                "outcome": "deferred",
                "reason": f"self_training_status_not_queueable:{status[:80]}",
            }
        block_reason = _autonomous_qlora_policy_block_reason(state)
        if block_reason:
            outcome = (
                "deferred"
                if block_reason.startswith("needs_more_verified_examples:")
                else "denied"
            )
            return {"outcome": outcome, "reason": block_reason[:500]}
        if not trajectory_id:
            return {
                "outcome": "denied",
                "reason": "missing_final_ingest_provenance",
            }
        if not any(
            item.get("eligibleForTraining") is True
            and str(item.get("sourceTrajectoryId") or "").strip() == trajectory_id
            for item in state.get("examples", [])
            if isinstance(item, dict)
        ):
            return {
                "outcome": "denied",
                "reason": "final_ingest_candidate_not_present",
            }
        queue_receipt = {
            "provenance": "final_helix_ingest",
            "sourceTrajectoryId": trajectory_id[:200],
            "sourceThreadId": str(source_thread_id or "").strip()[:200] or None,
            "modelId": str(state.get("baseModelId") or "").strip()[:500],
            "qualifiedExampleCount": len(_qualified_examples(state)),
            "createdAt": int(time.time() * 1_000),
        }
        state["status"] = "queued"
        state["trainingQualifiedOnly"] = True
        state["autonomousQueueReceipt"] = queue_receipt
        state["lastAutonomousQueueReceipt"] = queue_receipt
        _write_state(state)
        return {
            "outcome": "queued",
            "reason": "qualified_candidate_passed_final_autonomous_queue_gate",
        }


def queue_hermes_training_if_allowed(
    *,
    source_trajectory_id: str | None = None,
    source_thread_id: str | None = None,
) -> bool:
    """Backward-compatible boolean wrapper around the provenance-aware queue gate."""
    return queue_hermes_training_with_provenance(
        source_trajectory_id=source_trajectory_id,
        source_thread_id=source_thread_id,
    )["outcome"] == "queued"


def _training_runtime_snapshot() -> tuple[str | None, bool | None]:
    """Return the process-local training job identity without guessing on failure."""
    try:
        from core.training import get_training_backend

        backend = get_training_backend()
        job_id = str(getattr(backend, "current_job_id", "") or "")
        return job_id, bool(backend.is_training_active())
    except Exception:
        # Reconciliation is safety logic, not a reason to break the learning UI.
        # Unknown runtime state means "do not mutate persisted state".
        return None, None


def _training_terminal_snapshot(expected_job_id: str | None = None) -> dict[str, Any] | None:
    """Read terminal training evidence without attaching a foreign job."""
    try:
        from core.training import get_training_backend

        backend = get_training_backend()
        job_id = str(getattr(backend, "current_job_id", "") or "")
        if expected_job_id and job_id != expected_job_id:
            return None
        progress = backend.trainer.training_progress
        output_dir = getattr(progress, "output_dir", None) or getattr(backend, "_output_dir", None)
        return {
            "job_id": job_id or None,
            "active": bool(backend.is_training_active()),
            "completed": bool(getattr(progress, "is_completed", False)),
            "error": str(getattr(progress, "error", "") or "") or None,
            "output_dir": str(output_dir) if output_dir else None,
            "message": str(getattr(progress, "status_message", "") or ""),
        }
    except Exception:
        return None


def _candidate_artifact_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (OSError, FileNotFoundError):
        return None
    if not resolved.is_dir():
        return None
    if not (resolved / "adapter_config.json").is_file():
        return None
    # MLX saves ``adapters.safetensors``; PEFT/Transformers conventionally saves
    # ``adapter_model.safetensors``. The later backend-specific hot-swap performs
    # the strict format check; completion recovery only needs to know a real
    # adapter artifact exists.
    if not any(
        (resolved / name).is_file()
        for name in ("adapters.safetensors", "adapter_model.safetensors", "adapter_model.bin")
    ):
        return None
    return str(resolved)


def _autonomous_candidate_benchmark_allowed(state: dict[str, Any]) -> bool:
    if state.get("candidateQualifiedOnly") is not True or not state.get("enabled", True):
        return False
    try:
        from routes.learning import _read_state as _read_learning_state

        learning = _read_learning_state()
    except Exception:
        return False
    return (
        learning.get("decisionMode") == "autonomous"
        and learning.get("allowQloraTraining") is True
    )


def _record_completed_candidate(
    state: dict[str, Any],
    *,
    job_id: str,
    output_dir: str,
    qualified_only: bool,
) -> bool:
    candidate = _candidate_artifact_path(output_dir)
    if candidate is None:
        state["status"] = "error"
        state["lastError"] = "Self-QLoRA completed but no valid adapter artifact was produced."
        state["trainingQualifiedOnly"] = False
        state["autonomousQueueReceipt"] = None
        state["candidateQualifiedOnly"] = False
        return False
    state["status"] = "candidate-ready"
    state["lastCandidateAdapterPath"] = candidate
    state["lastError"] = None
    state["trainingQualifiedOnly"] = False
    state["autonomousQueueReceipt"] = None
    state["candidateQualifiedOnly"] = bool(qualified_only)
    state["lastRecovery"] = {
        "kind": "training-completed",
        "persistedJobId": job_id,
        "candidateAdapterPath": candidate,
        "recoveredAt": int(time.time() * 1_000),
    }
    return True


def _queue_candidate_benchmark_if_allowed(state: dict[str, Any]) -> str | None:
    if state.get("status") != "candidate-ready":
        return None
    candidate = _candidate_artifact_path(state.get("lastCandidateAdapterPath"))
    if candidate is None or not _autonomous_candidate_benchmark_allowed(state):
        return None
    state["status"] = "benchmark-queued"
    return candidate


def _reconcile_persisted_training_state(state: dict[str, Any]) -> tuple[bool, bool]:
    """Reconcile crash-persisted queued/training states with the live worker.

    Returns ``(changed, reschedule_autonomous)``.  A stale autonomous queue may
    be re-run only when the CURRENT policy still allows it.  A manual queue is
    never silently replayed after restart.  A stale ``training`` marker becomes
    an explicit error with provenance instead of permanently blocking /start.
    """
    status = str(state.get("status") or "")
    if status in {"promoted", "hot-swapped"} and state.get("activeAdapterPath"):
        try:
            from core.inference.orchestrator import peek_inference_backend

            backend = peek_inference_backend()
        except Exception:
            # Runtime ownership is unknown; do not rewrite durable state from an
            # unavailable observer.
            backend = "unknown"
        if backend != "unknown" and (
            backend is None or not str(getattr(backend, "active_model_name", "") or "").strip()
        ):
            previous = str(state.get("activeAdapterPath") or "")
            candidate = _candidate_artifact_path(previous)
            state["activeAdapterPath"] = None
            state["activeServingModelId"] = None
            if candidate is not None:
                state["lastCandidateAdapterPath"] = candidate
                state["status"] = "candidate-ready"
                state["lastError"] = None
            else:
                state["status"] = "error"
                state["lastError"] = "The persisted active adapter is no longer resident and its artifact is missing."
            state["lastRecovery"] = {
                "kind": "inactive-persisted-adapter-reconciled",
                "previousStatus": status,
                "candidateAdapterPath": candidate,
                "recoveredAt": int(time.time() * 1_000),
            }
            return True, False
    if status in {"benchmark-queued", "benchmark-loading-base", "benchmarking"}:
        candidate = _candidate_artifact_path(state.get("lastCandidateAdapterPath"))
        if candidate is None:
            state["status"] = "error"
            state["lastError"] = "Recovered an interrupted benchmark but its candidate adapter is missing."
            state["candidateQualifiedOnly"] = False
        else:
            # Benchmarking is deterministic and reversible. Return to the durable
            # candidate-ready boundary; get_self_training will requeue it only if
            # the CURRENT autonomous policy still permits QLoRA.
            state["status"] = "candidate-ready"
            state["lastError"] = None
            state["lastRecovery"] = {
                "kind": "interrupted-benchmark-recovered",
                "candidateAdapterPath": candidate,
                "recoveredAt": int(time.time() * 1_000),
            }
        return True, False
    if status not in {"queued", "training"}:
        return False, False
    live_job_id, live_active = _training_runtime_snapshot()
    if live_active is None:
        return False, False

    persisted_job_id = str(state.get("lastJobId") or "")
    now = int(time.time() * 1_000)
    if status == "training":
        if live_active and persisted_job_id and live_job_id == persisted_job_id:
            return False, False
        terminal = _training_terminal_snapshot(persisted_job_id) if persisted_job_id else None
        if terminal is not None and terminal.get("completed") is True:
            output_dir = str(terminal.get("output_dir") or "")
            _record_completed_candidate(
                state,
                job_id=persisted_job_id,
                output_dir=output_dir,
                qualified_only=state.get("trainingQualifiedOnly") is True,
            )
            return True, False
        if terminal is not None and terminal.get("error"):
            state["status"] = "error"
            state["lastError"] = str(terminal["error"])[:2_000]
            state["lastRecovery"] = {
                "kind": "training-error-observed",
                "persistedJobId": persisted_job_id or None,
                "runtimeJobId": terminal.get("job_id"),
                "runtimeActive": False,
                "recoveredAt": now,
            }
            state["trainingQualifiedOnly"] = False
            state["autonomousQueueReceipt"] = None
            return True, False
        state["status"] = "error"
        state["lastError"] = (
            "Recovered stale self-QLoRA training state: the persisted job is not active "
            "and no matching terminal completion receipt is available in this process."
        )
        state["lastRecovery"] = {
            "kind": "stale-training",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": live_job_id or None,
            "runtimeActive": bool(live_active),
            "recoveredAt": now,
        }
        state["trainingQualifiedOnly"] = False
        state["autonomousQueueReceipt"] = None
        return True, False

    # queued: a live job belonging to this persisted run means the state write
    # lagged the worker start; promote it to training rather than spawning twice.
    if live_active:
        if persisted_job_id and live_job_id == persisted_job_id:
            state["status"] = "training"
            state["lastRecovery"] = {
                "kind": "queued-job-observed-running",
                "persistedJobId": persisted_job_id,
                "runtimeJobId": live_job_id,
                "runtimeActive": True,
                "recoveredAt": now,
            }
            return True, False
        # Some other training owns the backend.  Preserve the queue as an error
        # instead of attaching Helix provenance to a foreign job.
        state["status"] = "error"
        state["lastError"] = "Recovered queued self-QLoRA state while another training job is active."
        state["lastRecovery"] = {
            "kind": "queued-runtime-conflict",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": live_job_id or None,
            "runtimeActive": True,
            "recoveredAt": now,
        }
        return True, False

    if (
        state.get("trainingQualifiedOnly") is True
        and _autonomous_qlora_policy_allows(state)
        and _valid_autonomous_queue_receipt(state)
    ):
        state["lastRecovery"] = {
            "kind": "stale-autonomous-queue-rescheduled",
            "persistedJobId": persisted_job_id or None,
            "runtimeJobId": None,
            "runtimeActive": False,
            "recoveredAt": now,
        }
        # Keep status=queued; the caller schedules exactly one new background task.
        return True, True

    state["status"] = "idle"
    state["trainingQualifiedOnly"] = False
    state["autonomousQueueReceipt"] = None
    state["lastError"] = None
    state["lastRecovery"] = {
        "kind": "stale-queue-cleared",
        "persistedJobId": persisted_job_id or None,
        "runtimeJobId": None,
        "runtimeActive": False,
        "recoveredAt": now,
        "reason": (
            "manual queue is not silently replayed, autonomous policy is no longer allowed, "
            "or final-ingest queue provenance is missing/invalid"
        ),
    }
    return True, False


def _training_row(example: dict[str, Any]) -> str:
    # Evaluator metadata and self-critique never become target text.
    row = {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _write_dataset(state: dict[str, Any], *, qualified_only: bool = False) -> None:
    path = _dataset_path()
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode = "w", encoding = "utf-8", dir = path.parent,
        prefix = ".task-examples-", suffix = ".jsonl", delete = False,
    ) as handle:
        temporary = Path(handle.name)
        examples = _qualified_examples(state) if qualified_only else state.get("examples", [])
        for example in examples:
            handle.write(_training_row(example))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _snapshot_training_dataset(
    state: dict[str, Any], *, qualified_only: bool
) -> tuple[Path, str, int]:
    """Write an immutable-by-convention per-run dataset and return path/hash/count.

    The training worker gets this unique path, never the mutable collection view.
    Later chat turns therefore cannot change what an already admitted run trains on.
    """
    examples = _qualified_examples(state) if qualified_only else list(state.get("examples", []))
    payload = "".join(_training_row(example) for example in examples).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    run_dir = datasets_root() / "helix-self-qlora-runs"
    ensure_dir(run_dir)
    path = run_dir / f"dataset-{int(time.time() * 1_000)}-{uuid.uuid4().hex[:12]}-{digest[:12]}.jsonl"
    # Exclusive create: no later collection path ever opens this filename for writing.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path, digest, len(examples)


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "datasetPath": str(_dataset_path()),
        "exampleCount": len(state.get("examples", [])),
        "eligibleExampleCount": len(_qualified_examples(state)),
        "promotionMargin": _PROMOTION_MARGIN,
        "acceptanceCriteria": [
            "candidate intelligence score >= base score + 0.01",
            "candidate throughput >= base throughput (no speed regression)",
            "original base model remains resident/available for rollback",
        ],
    }


def _validated_adapter_path(raw_path: str) -> Path:
    """Validate a local PEFT adapter directory before it reaches the worker."""
    path = Path(raw_path).expanduser()
    try:
        if path.is_symlink():
            raise HTTPException(status_code = 400, detail = "Adapter symlinks are not accepted.")
        resolved = path.resolve(strict = True)
    except FileNotFoundError as error:
        raise HTTPException(status_code = 404, detail = "The adapter directory does not exist.") from error
    except OSError as error:
        raise HTTPException(status_code = 400, detail = "The adapter path could not be inspected.") from error
    if not resolved.is_dir():
        raise HTTPException(status_code = 400, detail = "The adapter path must be a directory.")
    config = resolved / "adapter_config.json"
    if not config.is_file() or config.is_symlink():
        raise HTTPException(status_code = 400, detail = "The adapter must contain a local adapter_config.json.")
    try:
        json.loads(config.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code = 400, detail = "The adapter_config.json is invalid.") from error
    return resolved


def _adapter_name(raw_name: str | None, path: Path) -> str:
    name = (raw_name or path.name).strip().replace(".", "_")
    if not _ADAPTER_NAME_RE.fullmatch(name):
        raise HTTPException(status_code = 400, detail = "Adapter name must contain only letters, numbers, '_' or '-'.")
    return name


def _record_mem0_experience(subject: str, payload: SelfTrainingExampleRequest) -> None:
    """Persist a bounded task experience for recurrence search.

    This is deliberately separate from QLoRA: every experience may inform a
    future skill, but it never becomes a training target or a promotion score by
    itself. Mem0 is supplementary; the local JSON ledger remains authoritative.
    """
    try:
        from routes.learning import _read_state

        if _read_state().get("mem0Enabled") is False:
            return
        from core.memory.mem0_store import add_experience

        text = (
            "User task:\n"
            f"{payload.prompt.strip()}\n\n"
            "Assistant result:\n"
            f"{payload.completion.strip()}\n\n"
            f"Critique: {(payload.critique or '').strip()}"
        )
        add_experience(subject, text, thread_id=payload.sourceThreadId, kind="task-experience")
    except Exception:
        # Memory must never make a successful chat or dataset write fail.
        return


def _score_holdout(task: _BenchmarkTask, output: str) -> float:
    """Score a held-out answer without asking either model to judge itself."""
    answer = output.strip()
    if task.kind == "contains":
        wanted = task.expected.lower().split()
        return 1.0 if all(token in answer.lower() for token in wanted) else 0.0
    if task.kind == "ordered":
        compact = re.sub(r"\s+", "", answer).lower()
        return 1.0 if compact == task.expected.lower() else 0.0
    # JSON checks compare parsed values, so whitespace/key order cannot affect the score.
    try:
        start = answer.find("{")
        end = answer.rfind("}")
        parsed = json.loads(answer[start : end + 1]) if start >= 0 and end >= start else None
        expected = json.loads(task.expected)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    return 1.0 if parsed == expected else 0.0


def _run_holdout(backend: Any, use_adapter: str | bool) -> tuple[float, float, bool, list[dict[str, Any]]]:
    """Run the same fixed benchmark and return score plus genuinely measured speed.

    A missing backend timing is not estimated from wall-clock words. Estimates can
    make a candidate look like it preserved throughput when the backend did not
    actually report token accounting, so the promotion gate treats them as an
    unmeasured failure.
    """
    receipts: list[dict[str, Any]] = []
    rates: list[float] = []
    speed_measured = True
    for task in _HOLDOUT_BENCHMARK:
        stats_holder: dict[str, Any] = {}
        chunks: list[str] = []
        for chunk in backend.generate_with_adapter_control(
            use_adapter = use_adapter,
            messages = [{"role": "user", "content": task.prompt}],
            system_prompt = "Answer the held-out task directly. Do not describe your confidence or score.",
            temperature = 0.0,
            top_p = 1.0,
            max_new_tokens = 128,
            stats_holder = stats_holder,
        ):
            chunks.append(str(chunk))
        output = "".join(chunks)
        score = _score_holdout(task, output)
        stats = stats_holder.get("stats")
        timings = stats.get("timings") if isinstance(stats, dict) else None
        tok_per_sec = timings.get("predicted_per_second") if isinstance(timings, dict) else None
        if not isinstance(tok_per_sec, (int, float)) or tok_per_sec <= 0:
            speed_measured = False
        else:
            rates.append(float(tok_per_sec))
        receipts.append({"id": task.id, "score": score, "output": output[:2_000], "tokPerSec": tok_per_sec})
    return (
        sum(item["score"] for item in receipts) / len(receipts),
        sum(rates) / len(rates) if speed_measured and rates else 0.0,
        speed_measured and len(rates) == len(receipts),
        receipts,
    )


async def _settled_thread_call(func: Any, /, *args: Any) -> Any:
    """Do not abandon a model mutation/generation thread on coroutine cancellation.

    ``asyncio.to_thread`` keeps running after its awaiter is cancelled. Adapter
    attach/revert and held-out generation therefore must settle before cleanup can
    safely make another model mutation. Cancellation is still propagated after the
    worker call reaches a terminal state.
    """

    task = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise


async def _thread_call_outcome(
    func: Any, /, *args: Any
) -> tuple[Any | None, BaseException | None, bool]:
    """Return a blocking call's settled outcome even if the caller was cancelled."""

    task = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task), None, False
    except asyncio.CancelledError as cancel_error:
        current = asyncio.current_task()
        caller_cancelled = bool(current is not None and current.cancelling())
        if not caller_cancelled:
            return None, cancel_error, False
        try:
            return await asyncio.shield(task), None, True
        except BaseException as error:
            return None, error, True
    except BaseException as error:
        return None, error, False


async def _restore_serving_adapter(
    backend: Any,
    *,
    base_model: str,
    previous_adapter_path: str | None,
) -> bool:
    """Return to base, then restore the previously active adapter when one existed."""

    reverted = await _settled_thread_call(backend.revert_to_base_model, base_model)
    if reverted is False:
        raise RuntimeError("Could not revert the benchmark model to its base weights.")
    if not previous_adapter_path:
        return False
    previous = _validated_adapter_path(previous_adapter_path)
    previous_name = _adapter_name(None, previous)
    restored = await _settled_thread_call(
        backend.hot_swap_adapter,
        str(previous),
        previous_name,
        base_model,
    )
    if restored is False:
        raise RuntimeError("Could not restore the previously active adapter after benchmarking.")
    return True


async def _ensure_trainable_base_for_benchmark(subject: str) -> tuple[Any, bool]:
    """Return the inference backend with the trainable base resident.

    Training may evict inference to free unified memory. Autonomous evaluation
    therefore reloads the trainable source through the same production load gate
    as ordinary chat. The boolean says whether this helper performed the load so
    a rejected candidate can leave no hidden serving-model change behind.
    """
    from core.inference import get_inference_backend

    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
        context_length = int(state.get("baseContextLength") or state.get("maxSeqLength") or 8_192)
    if not base_model:
        raise RuntimeError("Self-QLoRA has no trainable base model for benchmarking.")
    backend = get_inference_backend()
    if str(getattr(backend, "active_model_name", "") or "") == base_model:
        return backend, False

    from models.inference import LoadRequest
    from routes.inference import load_model_gated

    await load_model_gated(
        LoadRequest(
            model_path=base_model,
            max_seq_length=max(512, min(context_length, 65_536)),
            load_in_4bit=True,
        ),
        None,
        subject,
        user_initiated=False,
    )
    backend = get_inference_backend()
    if str(getattr(backend, "active_model_name", "") or "") != base_model:
        raise RuntimeError("The trainable base did not become the active inference model.")
    return backend, True


async def _benchmark_completed_candidate(subject: str, candidate_path: str) -> None:
    """Run objective post-training acceptance under the current policy.

    A rejected candidate is reverted and, when this helper had to load the source
    model solely for evaluation, that source model is unloaded again. This lets
    the next normal chat reload the user's selected serving GGUF instead of
    silently replacing it with the training source. A promoted candidate remains
    on the trainable source + adapter and the state names that serving model
    explicitly.
    """
    with _STATE_LOCK:
        state = _read_state()
        if state.get("status") not in {"benchmark-queued", "candidate-ready"}:
            return
        if not _autonomous_candidate_benchmark_allowed(state):
            if state.get("status") == "benchmark-queued":
                state["status"] = "candidate-ready"
                _write_state(state)
            return
        candidate = _candidate_artifact_path(candidate_path)
        if candidate is None:
            state["status"] = "error"
            state["lastError"] = "The completed self-QLoRA adapter disappeared before benchmarking."
            _write_state(state)
            return
        state["status"] = "benchmark-loading-base"
        state["lastError"] = None
        _write_state(state)

    backend = None
    loaded_for_benchmark = False
    try:
        backend, loaded_for_benchmark = await _ensure_trainable_base_for_benchmark(subject)
        result = await _benchmark_self_training_candidate(
            SelfTrainingBenchmarkRequest(candidateAdapterPath=candidate),
            current_subject=subject,
            autonomous_policy_required=True,
        )
        evaluation = result.get("lastEvaluation") if isinstance(result, dict) else None
        promoted = bool(isinstance(evaluation, dict) and evaluation.get("promoted"))
        with _STATE_LOCK:
            state = _read_state()
            state["candidateQualifiedOnly"] = False
            _write_state(state)
        if not promoted and loaded_for_benchmark and backend is not None:
            base_model = str(result.get("baseModelId") or "") if isinstance(result, dict) else ""
            if not base_model:
                with _STATE_LOCK:
                    base_model = str(_read_state().get("baseModelId") or "")
            if base_model:
                await _settled_thread_call(backend.unload_model, base_model)
    except asyncio.CancelledError:
        if loaded_for_benchmark and backend is not None:
            try:
                with _STATE_LOCK:
                    base_model = str(_read_state().get("baseModelId") or "")
                if base_model:
                    await _settled_thread_call(backend.unload_model, base_model)
            except BaseException:
                pass
        with _STATE_LOCK:
            state = _read_state()
            candidate = _candidate_artifact_path(candidate_path)
            if candidate is not None:
                state["status"] = "candidate-ready"
                state["lastCandidateAdapterPath"] = candidate
            else:
                state["status"] = "error"
                state["lastError"] = "Automatic benchmark was cancelled and its candidate artifact is missing."
                state["candidateQualifiedOnly"] = False
            state["activeServingModelId"] = None
            state["lastRecovery"] = {
                "kind": "automatic-benchmark-cancelled-recovered",
                "candidateAdapterPath": candidate,
                "loadedBaseUnloaded": bool(loaded_for_benchmark and backend is not None),
                "recoveredAt": int(time.time() * 1_000),
            }
            _write_state(state)
        raise
    except Exception as error:  # noqa: BLE001 -- autonomous learning must fail closed, chat fail open
        if loaded_for_benchmark and backend is not None:
            try:
                with _STATE_LOCK:
                    base_model = str(_read_state().get("baseModelId") or "")
                if base_model:
                    await _settled_thread_call(backend.unload_model, base_model)
            except Exception:
                pass
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "error"
            state["lastError"] = f"Automatic candidate benchmark failed: {error}"[:2_000]
            state["activeServingModelId"] = None
            state["candidateQualifiedOnly"] = False
            _write_state(state)


async def _watch_self_training_job(subject: str, job_id: str, qualified_only: bool) -> None:
    """Follow one accepted self-QLoRA job through artifact and optional benchmark."""
    empty_terminal_polls = 0
    while True:
        await asyncio.sleep(1.0)
        terminal = await asyncio.to_thread(_training_terminal_snapshot, job_id)
        if terminal is None:
            empty_terminal_polls += 1
            if empty_terminal_polls < 30:
                continue
            with _STATE_LOCK:
                state = _read_state()
                if state.get("lastJobId") == job_id and state.get("status") == "training":
                    state["status"] = "error"
                    state["lastError"] = "The self-QLoRA worker state became unavailable before a terminal receipt."
                    state["trainingQualifiedOnly"] = False
                    state["autonomousQueueReceipt"] = None
                    _write_state(state)
            return
        empty_terminal_polls = 0
        if terminal.get("active") is True:
            continue
        if terminal.get("completed") is True:
            candidate_to_benchmark = None
            with _STATE_LOCK:
                state = _read_state()
                if state.get("lastJobId") != job_id:
                    return
                _record_completed_candidate(
                    state,
                    job_id=job_id,
                    output_dir=str(terminal.get("output_dir") or ""),
                    qualified_only=qualified_only,
                )
                candidate_to_benchmark = _queue_candidate_benchmark_if_allowed(state)
                _write_state(state)
            if candidate_to_benchmark:
                await _benchmark_completed_candidate(subject, candidate_to_benchmark)
            return
        error = str(terminal.get("error") or "").strip()
        if error:
            with _STATE_LOCK:
                state = _read_state()
                if state.get("lastJobId") == job_id:
                    state["status"] = "error"
                    state["lastError"] = error[:2_000]
                    state["trainingQualifiedOnly"] = False
                    state["autonomousQueueReceipt"] = None
                    state["candidateQualifiedOnly"] = False
                    _write_state(state)
            return


def _foreground_inference_busy() -> bool:
    """Whether foreground generation is active or queued around autonomous training.

    Unknown ownership is busy: autonomous QLoRA may evict resident models, so failing
    open here risks interrupting a foreground chat or an asynchronous video job.
    """
    try:
        from core.inference.llama_keepwarm import other_inference_request_count

        if other_inference_request_count(current_request_counted=False) > 0:
            return True
    except Exception:
        return True
    try:
        from routes.training import _background_video_generation_active

        if _background_video_generation_active():
            return True
    except Exception:
        return True
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine

        return bool(get_active_diffusion_engine().generate_progress().get("active"))
    except Exception:
        return True


def _mark_training_deferred_for_active_inference() -> None:
    with _STATE_LOCK:
        state = _read_state()
        if state.get("status") == "queued":
            state["lastRecovery"] = {
                "kind": "training-deferred-for-active-inference",
                "recoveredAt": int(time.time() * 1_000),
            }
            _write_state(state)


def _safe_training_max_seq_length(state: dict[str, Any]) -> int:
    """Bound autonomous training to the known baseline context when available."""
    try:
        configured = int(state.get("maxSeqLength") or 8_192)
    except (TypeError, ValueError):
        configured = 8_192
    configured = max(512, min(configured, 65_536))
    known_context = state.get("baseContextLength")
    try:
        context = int(known_context) if known_context is not None else 0
    except (TypeError, ValueError):
        context = 0
    if context >= 512:
        configured = min(configured, context)
    return configured


async def _wait_for_foreground_inference_idle(max_wait_s: float = 120.0) -> bool:
    """Require a stable idle inference interval before self-QLoRA can evict memory."""
    deadline = time.monotonic() + max(0.0, max_wait_s)
    idle_since: float | None = None
    while time.monotonic() < deadline:
        busy = _foreground_inference_busy()
        if not busy:
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= 0.5:
                return True
        else:
            idle_since = None
        await asyncio.sleep(0.2)
    return False


async def _start_training_for_state(subject: str) -> None:
    """Start one bounded LoRA/QLoRA run after foreground inference releases ownership."""
    with _STATE_LOCK:
        initial = _read_state()
        if initial.get("status") != "queued":
            return
    if not await _wait_for_foreground_inference_idle():
        # Keep the durable queue intact. get_self_training already reconciles and
        # reschedules autonomous queued work, so foreground activity delays rather
        # than discards training.
        _mark_training_deferred_for_active_inference()
        return
    start_cancelled = False
    try:
        from core.training.lifecycle import training_inference_admission_guard

        # The stable-idle poll above is intentionally outside the gate so a foreground
        # chat is never stalled for 500 ms. Once it succeeds, take the same process-wide
        # lifecycle gate every local chat request uses, recheck under it, and keep it until
        # start_training has published its spawn/worker ownership. Requests arriving after
        # the recheck become pending but cannot become inflight or allocate beside QLoRA.
        async with training_inference_admission_guard():
            if _foreground_inference_busy():
                _mark_training_deferred_for_active_inference()
                return

            with _STATE_LOCK:
                state = _read_state()
                if state.get("status") != "queued":
                    return
                model_id = str(state.get("baseModelId") or "").strip()
                qualified_only = state.get("trainingQualifiedOnly") is True
                candidate_examples = (
                    _qualified_examples(state) if qualified_only else state.get("examples", [])
                )
                if qualified_only and not _autonomous_qlora_policy_allows(state):
                    # Re-check at execution time: a user can revoke autonomous/QLoRA
                    # permission after queueing but before the background worker runs.
                    state["status"] = "idle"
                    state["trainingQualifiedOnly"] = False
                    state["autonomousQueueReceipt"] = None
                    _write_state(state)
                    return
                if qualified_only and not _valid_autonomous_queue_receipt(state):
                    state["status"] = "idle"
                    state["trainingQualifiedOnly"] = False
                    state["autonomousQueueReceipt"] = None
                    state["lastRecovery"] = {
                        "kind": "invalid-autonomous-queue-cleared",
                        "recoveredAt": int(time.time() * 1_000),
                        "reason": "missing_or_invalid_final_ingest_queue_provenance",
                    }
                    _write_state(state)
                    return
                if not model_id or len(candidate_examples) < int(state.get("minExamples", 8)):
                    state["status"] = "idle"
                    state["trainingQualifiedOnly"] = False
                    state["autonomousQueueReceipt"] = None
                    _write_state(state)
                    return
                dataset_snapshot, dataset_sha256, dataset_count = _snapshot_training_dataset(
                    state, qualified_only=qualified_only
                )
                state["activeDatasetSnapshot"] = str(dataset_snapshot)
                state["activeDatasetSha256"] = dataset_sha256
                state["activeDatasetExampleCount"] = dataset_count
                _write_state(state)
                snapshot = state.get("baseSnapshotPath")
                request_data: dict[str, Any] = {
                    "model_name": model_id,
                    "training_type": "LoRA/QLoRA",
                    "load_in_4bit": True,
                    "max_seq_length": _safe_training_max_seq_length(state),
                    "local_datasets": [str(dataset_snapshot)],
                    "format_type": "chatml",
                    "num_epochs": 1,
                    "max_steps": 32,
                    "save_steps": 16,
                    "project_name": "hermes-self-qlora",
                }
                if isinstance(snapshot, str) and snapshot.strip():
                    request_data.update({
                        "model_local_path": snapshot,
                        "model_snapshot_path": snapshot,
                        "model_known_cached": True,
                    })
                state["status"] = "training"
                state["lastError"] = None
                _write_state(state)

            # Reuse the existing training admission, VRAM coordination, MLX/CUDA
            # selection, and provenance checks instead of creating a second trainer.
            from routes.training import start_training

            start_task = asyncio.create_task(
                start_training(
                    TrainingStartRequest(**request_data),
                    current_subject = subject,
                    via_api_key = False,
                )
            )
            try:
                result = await asyncio.shield(start_task)
            except asyncio.CancelledError:
                # start_training itself shields the backend spawn. Do not release
                # inference ownership or lose the job id while that spawn is still
                # settling; reconcile its concrete result before propagating cancel.
                start_cancelled = True
                try:
                    result = await asyncio.shield(start_task)
                except BaseException as settle_error:
                    with _STATE_LOCK:
                        state = _read_state()
                        state["status"] = "error"
                        state["lastError"] = (
                            f"Cancelled self-QLoRA start failed while settling: "
                            f"{type(settle_error).__name__}: {settle_error}"
                        )[:2_000]
                        _write_state(state)
                    raise asyncio.CancelledError from settle_error
        job_id = getattr(result, "job_id", None) or (
            result.get("job_id") if isinstance(result, dict) else None
        )
        start_failed = getattr(result, "status", None) == "error" or (
            isinstance(result, dict) and result.get("status") == "error"
        )
        with _STATE_LOCK:
            state = _read_state()
            state["lastJobId"] = job_id
            if start_failed:
                state["status"] = "error"
                state["trainingQualifiedOnly"] = False
                state["autonomousQueueReceipt"] = None
                state["lastError"] = getattr(result, "message", None) or (
                    result.get("message") if isinstance(result, dict) else "Self-QLoRA could not start."
                )
            if start_cancelled and not start_failed and job_id:
                state["lastRecovery"] = {
                    "kind": "cancelled-start-job-preserved",
                    "persistedJobId": str(job_id),
                    "recoveredAt": int(time.time() * 1_000),
                }
            _write_state(state)
        if start_cancelled:
            raise asyncio.CancelledError
        if not start_failed and job_id:
            await _watch_self_training_job(subject, str(job_id), qualified_only)
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)


async def _apply_runtime_fix_for_state() -> None:
    """Safest automatic runtime repair: return to the retained base adapter.

    This does not delete an adapter from disk. It only disables the active
    candidate so a later benchmark or hotswap can restore it, which gives the
    autonomous runtime-fix permission a bounded, reversible meaning.
    """
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "runtime-fix-needs-baseline"
            _write_state(state)
        return
    try:
        from core.inference import get_inference_backend

        result, error, cancelled = await _thread_call_outcome(
            get_inference_backend().revert_to_base_model,
            base_model,
        )
        if error is not None:
            raise RuntimeError(f"Runtime-fix revert failed: {error}") from error
        if result is False:
            raise RuntimeError("Runtime-fix revert was rejected by the inference backend.")
        with _STATE_LOCK:
            state = _read_state()
            state["activeAdapterPath"] = None
            state["activeServingModelId"] = None
            state["status"] = "runtime-fixed"
            state["lastError"] = None
            _write_state(state)
        if cancelled:
            raise asyncio.CancelledError
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 -- keep the recommendation visible
        with _STATE_LOCK:
            state = _read_state()
            state["status"] = "runtime-fix-error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)


@router.get("")
def get_self_training(
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    reschedule = False
    candidate_to_benchmark = None
    with _STATE_LOCK:
        state = _read_state()
        changed, reschedule = _reconcile_persisted_training_state(state)
        candidate_to_benchmark = _queue_candidate_benchmark_if_allowed(state)
        if changed or candidate_to_benchmark:
            _write_state(state)
        result = _public_state(state)
    if reschedule:
        background_tasks.add_task(_start_training_for_state, current_subject)
    if candidate_to_benchmark:
        background_tasks.add_task(
            _benchmark_completed_candidate, current_subject, candidate_to_benchmark
        )
    return result


@router.post("/config")
def set_self_training_config(
    payload: SelfTrainingConfigRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        for field, key in (("enabled", "enabled"), ("autoTrain", "autoTrain"), ("minExamples", "minExamples"), ("maxSeqLength", "maxSeqLength")):
            value = getattr(payload, field)
            if value is not None:
                state[key] = value
        if state.get("autoTrain") and not state.get("enabled"):
            state["autoTrain"] = False
        _write_state(state)
        return _public_state(state)


@router.post("/recommendation")
async def set_self_training_recommendation(
    payload: SelfTrainingRecommendationRequest,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    """Store a model's training-vs-skill suggestion as advisory metadata only."""
    schedule_runtime_fix = False
    with _STATE_LOCK:
        state = _read_state()
        state["lastRecommendation"] = {
            "action": payload.action,
            "reason": payload.reason.strip(),
            "createdAt": int(time.time() * 1_000),
            "advisoryOnly": True,
        }
        # A model-authored recommendation is never queue authority. Autonomous
        # QLoRA can transition to queued only from final Helix ingest after the
        # verified-target/evidence gates stage a qualified candidate.
        if payload.action == "runtime-fix":
            state["status"] = "runtime-fix-recommended"
            from routes.learning import _read_state as _read_learning_state

            learning_state = _read_learning_state()
            schedule_runtime_fix = (
                learning_state.get("decisionMode") == "autonomous"
                and learning_state.get("allowRuntimeFix") is True
                and bool(state.get("baseModelId"))
            )
            if schedule_runtime_fix:
                state["status"] = "runtime-fix-queued"
        _write_state(state)
        result = _public_state(state)
    if schedule_runtime_fix:
        background_tasks.add_task(_apply_runtime_fix_for_state)
    return result


@router.post("/baseline")
def set_self_training_baseline(
    payload: BaselineRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        requested = payload.modelId.strip()
        baseline = _trainable_baseline_for_serving_model(requested)
        state["baseModelId"] = baseline
        state["baseServingModelId"] = requested if requested != baseline else None
        requested_snapshot = payload.snapshotPath.strip() if payload.snapshotPath else None
        if requested != baseline:
            state["baseSnapshotPath"] = None
            state["baseServingSnapshotPath"] = requested_snapshot
        else:
            state["baseSnapshotPath"] = requested_snapshot
            state["baseServingSnapshotPath"] = None
        state["baseContextLength"] = payload.contextLength
        # A new baseline may intentionally be the trainable source of a GGUF-serving
        # checkpoint. Keep only examples that map to that same source model.
        state["examples"] = [
            item for item in state.get("examples", [])
            if _example_matches_baseline(item, baseline)
        ]
        _write_dataset(state)
        _write_state(state)
        return _public_state(state)


def record_hermes_qualified_candidate_with_provenance(
    *,
    model_id: str,
    prompt: str,
    completion: str,
    source_trajectory_id: str,
    evidence_ids: list[str],
    source_trajectory_ids: list[str] | None = None,
    evidence_sha256: str | None = None,
    source_thread_id: str | None = None,
) -> dict[str, Any]:
    """Stage an evidence-linked candidate and report the observed storage decision."""
    if not model_id.strip():
        return {"stored": False, "reason": "missing_model_id"}
    if not prompt.strip():
        return {"stored": False, "reason": "missing_prompt"}
    if not completion.strip():
        return {"stored": False, "reason": "missing_verified_training_target"}
    if not source_trajectory_id.strip():
        return {"stored": False, "reason": "missing_source_trajectory_id"}
    with _STATE_LOCK:
        state = _read_state()
        if not state.get("enabled", True):
            return {"stored": False, "reason": "self_training_disabled"}
        baseline = str(state.get("baseModelId") or "").strip()
        serving_model = model_id.strip()
        if baseline and not _serving_model_matches_baseline(state, serving_model):
            return {
                "stored": False,
                "reason": "serving_model_does_not_match_trainable_baseline",
            }
        if not baseline:
            baseline = _trainable_baseline_for_serving_model(serving_model)
            state["baseModelId"] = baseline
            state["baseServingModelId"] = serving_model if serving_model != baseline else None
        elif serving_model != baseline and not state.get("baseServingModelId"):
            state["baseServingModelId"] = serving_model
        existing = next(
            (
                item
                for item in state.get("examples", [])
                if isinstance(item, dict)
                and str(item.get("sourceTrajectoryId") or "") == source_trajectory_id
            ),
            None,
        )
        if existing is not None:
            same_candidate = bool(
                existing.get("eligibleForTraining") is True
                and str(existing.get("trainableBaseModelId") or "").strip() == baseline
                and str(existing.get("prompt") or "") == prompt.strip()[:20_000]
                and str(existing.get("completion") or "") == completion.strip()[:24_000]
            )
            if same_candidate:
                return {
                    "stored": True,
                    "reason": "verified_hermes_candidate_already_staged",
                    "example_id": existing.get("id"),
                    "idempotent": True,
                }
            return {"stored": False, "reason": "source_trajectory_candidate_conflict"}
        example = {
            "id": uuid.uuid4().hex,
            "modelId": serving_model,
            "trainableBaseModelId": baseline,
            "prompt": prompt.strip()[:20_000],
            "completion": completion.strip()[:24_000],
            "sourceThreadId": source_thread_id,
            "score": None,
            "critique": "Hermes-qualified repeated behavioral candidate",
            "eligibleForTraining": True,
            "sourceTrajectoryId": source_trajectory_id.strip()[:200],
            "sourceTrajectoryIds": [str(item)[:200] for item in (source_trajectory_ids or [source_trajectory_id])[:64]],
            "evidenceIds": [str(item)[:200] for item in evidence_ids[:64]],
            "evidenceSha256": str(evidence_sha256 or "")[:128] or None,
            "createdAt": int(time.time() * 1_000),
        }
        state["examples"] = [*state.get("examples", []), example][-_MAX_EXAMPLES:]
        _write_state(state)
        return {
            "stored": True,
            "reason": "verified_hermes_candidate_staged",
            "example_id": example["id"],
        }


def record_hermes_qualified_candidate(
    *,
    model_id: str,
    prompt: str,
    completion: str,
    source_trajectory_id: str,
    evidence_ids: list[str],
    source_trajectory_ids: list[str] | None = None,
    evidence_sha256: str | None = None,
    source_thread_id: str | None = None,
) -> bool:
    """Backward-compatible boolean wrapper for existing Hermes candidate callers."""
    return bool(
        record_hermes_qualified_candidate_with_provenance(
            model_id=model_id,
            prompt=prompt,
            completion=completion,
            source_trajectory_id=source_trajectory_id,
            evidence_ids=evidence_ids,
            source_trajectory_ids=source_trajectory_ids,
            evidence_sha256=evidence_sha256,
            source_thread_id=source_thread_id,
        ).get("stored")
    )


def _completed_turn_request_fingerprint(
    payload: SelfTrainingExampleRequest,
    *,
    serving_model: str,
    source_thread: str | None,
    normalized_prompt: str,
    normalized_completion: str,
) -> str:
    """Hash the semantic request so a logical-turn key cannot be reused with new content."""
    canonical = {
        "modelId": serving_model,
        "prompt": normalized_prompt,
        "completion": normalized_completion,
        "sourceThreadId": source_thread,
        "score": payload.score,
        "critique": payload.critique,
        # These client fields never grant training eligibility, but changing them on
        # a retry is still a conflicting request and must not replay an old receipt.
        "eligibleForTraining": payload.eligibleForTraining,
        "sourceTrajectoryId": str(payload.sourceTrajectoryId or "").strip() or None,
        "evidenceIds": [str(item) for item in payload.evidenceIds],
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _completed_turn_idempotency_receipt(
    *,
    idempotency_key: str,
    example_id: str,
    serving_model: str,
    source_thread: str | None,
    created_at: int,
) -> dict[str, Any]:
    return {
        "kind": "self_training_completed_turn",
        "idempotencyKey": idempotency_key,
        "exampleId": example_id,
        "modelId": serving_model,
        "sourceThreadId": source_thread,
        "createdAt": created_at,
    }


@router.post("/examples")
async def record_self_training_example(
    payload: SelfTrainingExampleRequest,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        serving_model = payload.modelId.strip()
        normalized_prompt = payload.prompt.strip()
        normalized_completion = payload.completion.strip()
        source_thread = str(payload.sourceThreadId or "").strip() or None
        idempotency_key = str(payload.idempotencyKey or "").strip() or None
        request_fingerprint = (
            _completed_turn_request_fingerprint(
                payload,
                serving_model=serving_model,
                source_thread=source_thread,
                normalized_prompt=normalized_prompt,
                normalized_completion=normalized_completion,
            )
            if idempotency_key
            else None
        )

        # A keyed retry is resolved before mutable policy/baseline checks. Once the
        # original request committed its example+receipt in the atomic state file,
        # later config changes cannot turn a lost-response retry into a second write.
        if idempotency_key:
            keyed_existing = next(
                (
                    item
                    for item in state.get("examples", [])
                    if isinstance(item, dict)
                    and str(item.get("idempotencyKey") or "").strip() == idempotency_key
                    and str(item.get("modelId") or "").strip() == serving_model
                    and (str(item.get("sourceThreadId") or "").strip() or None) == source_thread
                ),
                None,
            )
            if keyed_existing is not None:
                persisted_fingerprint = str(
                    keyed_existing.get("idempotencyPayloadSha256") or ""
                ).strip()
                if not persisted_fingerprint or persisted_fingerprint != request_fingerprint:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "The self-training idempotency key is already bound to a different "
                            "completed-turn payload in this model/thread scope."
                        ),
                    )
                receipt = keyed_existing.get("idempotencyReceipt")
                if not isinstance(receipt, dict):
                    raise HTTPException(
                        status_code=409,
                        detail="The persisted self-training idempotency receipt is invalid.",
                    )
                return {
                    "recorded": True,
                    "idempotent": True,
                    "reason": "idempotency_key_replay",
                    "exampleId": keyed_existing.get("id"),
                    "idempotencyReceipt": receipt,
                    **_public_state(state),
                }

        if not state.get("enabled", True):
            return {"recorded": False, "reason": "disabled", **_public_state(state)}
        baseline = str(state.get("baseModelId") or "").strip()
        if baseline and not _serving_model_matches_baseline(state, serving_model):
            return {"recorded": False, "reason": "model differs from baseline", **_public_state(state)}
        if not baseline:
            baseline = _trainable_baseline_for_serving_model(serving_model)
            state["baseModelId"] = baseline
            state["baseServingModelId"] = serving_model if serving_model != baseline else None
        elif serving_model != baseline and not state.get("baseServingModelId"):
            state["baseServingModelId"] = serving_model

        # Legacy callers without a logical request id retain the content replay
        # guard. Keyed callers intentionally bypass it: two distinct logical turns
        # may have identical text and must stay distinct when their keys differ.
        existing = next(
            (
                item
                for item in state.get("examples", [])
                if isinstance(item, dict)
                and idempotency_key is None
                and item.get("eligibleForTraining") is not True
                and str(item.get("modelId") or "").strip() == serving_model
                and (str(item.get("sourceThreadId") or "").strip() or None) == source_thread
                and str(item.get("prompt") or "").strip() == normalized_prompt
                and str(item.get("completion") or "").strip() == normalized_completion
            ),
            None,
        )
        if existing is not None:
            return {
                "recorded": True,
                "idempotent": True,
                "reason": "duplicate_completed_turn",
                "exampleId": existing.get("id"),
                **_public_state(state),
            }
        created_at = int(time.time() * 1_000)
        example_id = uuid.uuid4().hex
        idempotency_receipt = (
            _completed_turn_idempotency_receipt(
                idempotency_key=idempotency_key,
                example_id=example_id,
                serving_model=serving_model,
                source_thread=source_thread,
                created_at=created_at,
            )
            if idempotency_key
            else None
        )
        example = {
            "id": example_id,
            "modelId": serving_model,
            "trainableBaseModelId": baseline,
            "prompt": normalized_prompt,
            "completion": normalized_completion,
            "sourceThreadId": source_thread,
            "score": payload.score,
            "critique": payload.critique,
            # Qualification is an internal Hermes capability, never a client assertion.
            "eligibleForTraining": False,
            "sourceTrajectoryId": None,
            "evidenceIds": [],
            "createdAt": created_at,
        }
        if idempotency_key:
            example["idempotencyKey"] = idempotency_key
            example["idempotencyPayloadSha256"] = request_fingerprint
            example["idempotencyReceipt"] = idempotency_receipt
        state["examples"] = [*state.get("examples", []), example][-_MAX_EXAMPLES:]
        _write_dataset(state)
        _write_state(state)
        result = {
            "recorded": True,
            "exampleId": example_id,
            **_public_state(state),
        }
        if idempotency_receipt is not None:
            result["idempotencyReceipt"] = idempotency_receipt
    return result


@router.post("/start")
async def start_self_training(
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
):
    reschedule = False
    with _STATE_LOCK:
        state = _read_state()
        changed, reschedule = _reconcile_persisted_training_state(state)
        if changed:
            _write_state(state)
        if reschedule:
            result = {"queued": True, "recovered": True, **_public_state(state)}
        else:
            result = None
        if result is not None:
            pass
        elif not state.get("enabled", True):
            raise HTTPException(status_code = 409, detail = "Self-QLoRA collection is disabled in the sidebar.")
        elif not state.get("baseModelId"):
            raise HTTPException(status_code = 400, detail = "Load a model and record a task before starting self-QLoRA.")
        elif len(state.get("examples", [])) < int(state.get("minExamples", 8)):
            raise HTTPException(status_code = 400, detail = f"Self-QLoRA needs at least {state.get('minExamples', 8)} bounded task examples.")
        elif state.get("status") in {"queued", "training"}:
            raise HTTPException(status_code = 409, detail = "Self-QLoRA is already queued or training.")
        elif result is None:
            state["status"] = "queued"
            state["trainingQualifiedOnly"] = False
            state["autonomousQueueReceipt"] = None
            _write_dataset(state, qualified_only=False)
            _write_state(state)
            result = {"queued": True, **_public_state(state)}
    background_tasks.add_task(_start_training_for_state, current_subject)
    return result


@router.post("/evaluate")
def evaluate_self_training_candidate(
    payload: SelfTrainingEvaluationRequest,
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        try:
            _validated_adapter_path(payload.candidateAdapterPath)
            adapter_exists = True
        except HTTPException:
            adapter_exists = False
        intelligence_improved = payload.candidateScore >= payload.baseScore + _PROMOTION_MARGIN
        speed_preserved = payload.candidateTokPerSec >= payload.baseTokPerSec
        speed_measured = payload.baseTokPerSec > 0 and payload.candidateTokPerSec > 0
        reported_metrics_pass = (
            intelligence_improved and speed_preserved and speed_measured and adapter_exists
        )
        evaluation = {
            "candidateAdapterPath": payload.candidateAdapterPath,
            "baseScore": payload.baseScore,
            "candidateScore": payload.candidateScore,
            "baseTokPerSec": payload.baseTokPerSec,
            "candidateTokPerSec": payload.candidateTokPerSec,
            "evaluator": payload.evaluator,
            "rubric": payload.rubric.strip(),
            "adapterExists": adapter_exists,
            "intelligenceImproved": intelligence_improved,
            "speedPreserved": speed_preserved,
            "speedMeasured": speed_measured,
            # These metrics arrive from the caller and are advisory. Only the
            # backend-held-out benchmark can set promoted=True automatically;
            # explicit /hot-swap remains the separate user activation path.
            "reportedMetricsPass": reported_metrics_pass,
            "promoted": False,
            "promotionAuthority": "backend_holdout_or_explicit_hot_swap",
            "createdAt": int(time.time() * 1_000),
        }
        state["lastEvaluation"] = evaluation
        _write_state(state)
        return _public_state(state)


@router.post("/benchmark")
async def benchmark_self_training_candidate(
    payload: SelfTrainingBenchmarkRequest,
    current_subject: str = Depends(get_current_subject),
):
    return await _benchmark_self_training_candidate(
        payload,
        current_subject=current_subject,
        autonomous_policy_required=False,
    )


async def _benchmark_self_training_candidate(
    payload: SelfTrainingBenchmarkRequest,
    *,
    current_subject: str,
    autonomous_policy_required: bool,
):
    """Benchmark base and candidate with fixed held-out checks.

    The model outputs are scored by this process, not by an assistant-generated
    critique. The candidate is evaluated first, then the worker is reverted to
    the untouched base before the baseline run.
    """
    path = _validated_adapter_path(payload.candidateAdapterPath)
    name = _adapter_name(payload.adapterName, path)
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
        if not base_model:
            raise HTTPException(status_code = 400, detail = "Set a self-QLoRA base model before benchmarking.")
        previous_status = str(state.get("status") or "candidate-ready")
        previous_active_adapter = str(state.get("activeAdapterPath") or "").strip() or None
        previous_active_serving = str(state.get("activeServingModelId") or "").strip() or None
        previous_active_status = (
            previous_status
            if previous_status in {"promoted", "hot-swapped"}
            else "hot-swapped"
        )
        state["status"] = "benchmarking"
        state["lastError"] = None
        _write_state(state)

    backend = None
    candidate_loaded = False
    previous_restored = False
    try:
        from core.training.lifecycle import training_inference_admission_guard

        # Benchmarking mutates the resident adapter between candidate and base
        # passes. Own the same lifecycle gate used by autonomous training so no
        # foreground request can observe the temporary candidate state.
        async with training_inference_admission_guard():
            if _foreground_inference_busy():
                raise HTTPException(
                    status_code=409,
                    detail="Cannot benchmark a self-QLoRA candidate while foreground inference is pending or active.",
                )
            from core.inference import get_inference_backend

            backend = get_inference_backend()
            try:
                # Mark before awaiting: a cancellation can arrive while the worker
                # thread is attaching the candidate. _settled_thread_call waits for
                # that mutation to finish before the cleanup below can revert it.
                candidate_loaded = True
                swapped = await _settled_thread_call(
                    backend.hot_swap_adapter,
                    str(path),
                    name,
                    base_model,
                )
                if swapped is False:
                    raise RuntimeError("The candidate adapter could not be activated for benchmarking.")
                candidate_score, candidate_speed, candidate_speed_measured, candidate_tasks = await _settled_thread_call(
                    _run_holdout,
                    backend,
                    name,
                )
                reverted = await _settled_thread_call(backend.revert_to_base_model, base_model)
                if reverted is False:
                    raise RuntimeError("The candidate adapter could not be reverted before the base benchmark.")
                candidate_loaded = False
                base_score, base_speed, base_speed_measured, base_tasks = await _settled_thread_call(
                    _run_holdout,
                    backend,
                    False,
                )

                intelligence_improved = candidate_score >= base_score + _PROMOTION_MARGIN
                speed_preserved = candidate_speed >= base_speed
                speed_measured = candidate_speed_measured and base_speed_measured
                promoted = intelligence_improved and speed_preserved and speed_measured
                promotion_error = None
                autonomous_policy_revoked = False
                if promoted and autonomous_policy_required:
                    with _STATE_LOCK:
                        current_policy_state = _read_state()
                        autonomous_policy_revoked = not _autonomous_candidate_benchmark_allowed(
                            current_policy_state
                        )
                    if autonomous_policy_revoked:
                        promoted = False
                        promotion_error = (
                            "Autonomous QLoRA promotion was deferred because the current learning policy "
                            "no longer permits autonomous adapter activation."
                        )
                if promoted:
                    candidate_loaded = True
                    activated = await _settled_thread_call(
                        backend.hot_swap_adapter,
                        str(path),
                        name,
                        base_model,
                    )
                    if activated is False:
                        candidate_loaded = False
                        promoted = False
                        promotion_error = "The accepted candidate could not be activated."
                if not promoted and previous_active_adapter:
                    previous_restored = await _restore_serving_adapter(
                        backend,
                        base_model=base_model,
                        previous_adapter_path=previous_active_adapter,
                    )
            except BaseException:
                # Always leave a deterministic serving state before releasing the
                # lifecycle gate, including CancelledError/SystemExit-like aborts.
                if backend is not None:
                    try:
                        previous_restored = await _restore_serving_adapter(
                            backend,
                            base_model=base_model,
                            previous_adapter_path=previous_active_adapter,
                        )
                        candidate_loaded = False
                    except BaseException:
                        previous_restored = False
                raise
    except asyncio.CancelledError:
        with _STATE_LOCK:
            state = _read_state()
            state["activeAdapterPath"] = previous_active_adapter if previous_restored else None
            state["activeServingModelId"] = previous_active_serving if previous_restored else None
            state["status"] = (
                previous_active_status
                if previous_restored and previous_active_adapter
                else previous_status if not previous_active_adapter else "error"
            )
            state["lastError"] = None if state["status"] != "error" else "Benchmark cancellation could not restore the previous adapter."
            state["lastRecovery"] = {
                "kind": "benchmark-cancelled-recovered",
                "candidateAdapterPath": str(path),
                "previousAdapterRestored": previous_restored,
                "recoveredAt": int(time.time() * 1_000),
            }
            _write_state(state)
        raise
    except HTTPException:
        with _STATE_LOCK:
            state = _read_state()
            state["activeAdapterPath"] = previous_active_adapter if previous_restored else state.get("activeAdapterPath")
            if previous_restored:
                state["activeServingModelId"] = previous_active_serving
            state["status"] = (
                previous_active_status
                if previous_restored and previous_active_adapter
                else previous_status
            )
            _write_state(state)
        raise
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["activeAdapterPath"] = previous_active_adapter if previous_restored else None
            state["activeServingModelId"] = previous_active_serving if previous_restored else None
            state["status"] = "error"
            state["lastError"] = str(error)[:2_000]
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error

    evaluation = {
        "candidateAdapterPath": str(path),
        "baseScore": base_score,
        "candidateScore": candidate_score,
        "baseTokPerSec": base_speed,
        "candidateTokPerSec": candidate_speed,
        "evaluator": "holdout",
        "rubric": "fixed held-out harness; identical prompts and deterministic checks",
        "adapterExists": True,
        "intelligenceImproved": intelligence_improved,
        "speedPreserved": speed_preserved,
        "speedMeasured": speed_measured,
        "baseTasks": base_tasks,
        "candidateTasks": candidate_tasks,
        "promoted": promoted,
        "promotionError": promotion_error,
        "autonomousPolicyRequired": autonomous_policy_required,
        "autonomousPolicyRevoked": autonomous_policy_revoked,
        "createdAt": int(time.time() * 1_000),
    }
    with _STATE_LOCK:
        state = _read_state()
        state["lastEvaluation"] = evaluation
        if evaluation["promoted"]:
            state["activeAdapterPath"] = str(path)
            state["activeServingModelId"] = base_model
            state["status"] = "promoted"
        elif previous_restored and previous_active_adapter:
            state["activeAdapterPath"] = previous_active_adapter
            state["activeServingModelId"] = previous_active_serving
            state["status"] = previous_active_status
        elif autonomous_policy_revoked:
            state["activeAdapterPath"] = None
            state["activeServingModelId"] = None
            state["lastCandidateAdapterPath"] = str(path)
            state["status"] = "candidate-ready"
        else:
            state["activeAdapterPath"] = None
            state["activeServingModelId"] = None
            state["status"] = "needs-more-data"
        _write_state(state)
        return _public_state(state)


@router.post("/hot-swap")
async def hot_swap_self_training_adapter(
    payload: AdapterHotSwapRequest,
    current_subject: str = Depends(get_current_subject),
):
    path = _validated_adapter_path(payload.adapterPath)
    name = _adapter_name(payload.adapterName, path)
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        raise HTTPException(status_code = 400, detail = "Set a self-QLoRA base model before hotswapping an adapter.")
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        result, error, cancelled = await _thread_call_outcome(
            backend.hot_swap_adapter,
            str(path),
            name,
            base_model,
        )
        if error is not None:
            raise RuntimeError(f"Adapter hot-swap failed: {error}") from error
        if result is False:
            raise RuntimeError("The inference backend rejected the adapter hot-swap.")
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["lastError"] = str(error)[:2_000]
            state["status"] = "error"
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error
    with _STATE_LOCK:
        state = _read_state()
        state["activeAdapterPath"] = str(path)
        state["activeServingModelId"] = base_model
        state["status"] = "hot-swapped"
        state["lastError"] = None
        _write_state(state)
        result_state = _public_state(state)
    if cancelled:
        raise asyncio.CancelledError
    return result_state


@router.post("/revert")
async def revert_self_training_adapter(
    current_subject: str = Depends(get_current_subject),
):
    with _STATE_LOCK:
        state = _read_state()
        base_model = str(state.get("baseModelId") or "").strip()
    if not base_model:
        raise HTTPException(status_code = 400, detail = "There is no self-QLoRA base model to revert.")
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        result, error, cancelled = await _thread_call_outcome(
            backend.revert_to_base_model,
            base_model,
        )
        if error is not None:
            raise RuntimeError(f"Adapter revert failed: {error}") from error
        if result is False:
            raise RuntimeError("The inference backend rejected the adapter revert.")
    except Exception as error:  # noqa: BLE001
        with _STATE_LOCK:
            state = _read_state()
            state["lastError"] = str(error)[:2_000]
            state["status"] = "error"
            _write_state(state)
        raise HTTPException(status_code = 409, detail = str(error)) from error
    with _STATE_LOCK:
        state = _read_state()
        state["activeAdapterPath"] = None
        state["activeServingModelId"] = None
        state["status"] = "reverted"
        state["lastError"] = None
        _write_state(state)
        result_state = _public_state(state)
    if cancelled:
        raise asyncio.CancelledError
    return result_state
