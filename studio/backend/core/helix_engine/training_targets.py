# SPDX-License-Identifier: AGPL-3.0-only
"""Backend-issued corrected-target receipts for autonomous QLoRA admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Iterable
from uuid import uuid4

from .ledger import append_record
from .trajectory import ToolStep

TRAINING_TARGET_RECEIPT_VERSION = "helix.training-target-receipt.v1"
_MAX_TARGET_CHARS = 24_000
_ISSUER_TOKEN = object()


class TrainingTargetSource(str, Enum):
    HUMAN_CORRECTION = "human_correction"
    OBJECTIVE_CORRECTION = "objective_correction"


@dataclass(frozen=True)
class VerifiedTrainingTargetReceipt:
    receipt_id: str
    trajectory_id: str
    target: str
    source: TrainingTargetSource
    source_ref: str
    evidence_refs: tuple[str, ...]
    verifier_ref: str
    target_sha256: str
    receipt_sha256: str
    created_at_ms: int
    source_thread_id: str | None = None
    schema_version: str = TRAINING_TARGET_RECEIPT_VERSION
    provenance: str = "backend_verified_target_ingress"
    _issuer_token: object = field(default=None, repr=False, compare=False)

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "trajectory_id": self.trajectory_id,
            "source": self.source.value,
            "source_ref": self.source_ref,
            "evidence_refs": list(self.evidence_refs),
            "verifier_ref": self.verifier_ref,
            "target_sha256": self.target_sha256,
            "receipt_sha256": self.receipt_sha256,
            "created_at_ms": self.created_at_ms,
            "source_thread_id": self.source_thread_id,
            "provenance": self.provenance,
            "target_chars": len(self.target),
        }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_refs(values: Iterable[str] | None) -> tuple[str, ...]:
    refs: list[str] = []
    for value in values or ():
        ref = str(value or "").strip()[:200]
        if ref and ref not in refs:
            refs.append(ref)
        if len(refs) >= 64:
            break
    return tuple(refs)


def _objective_verification_refs(
    steps: Iterable[ToolStep] | None,
    *,
    target: str,
) -> set[str]:
    from .evidence import tool_verification_receipt, validated_tool_verification

    target_subject = f"target:sha256:{_sha256(target)}"

    return {
        f"tool:{index}:verification"
        for index, step in enumerate(steps or ())
        if tool_verification_receipt(step)
        and (receipt := validated_tool_verification(step)) is not None
        and receipt.subject == target_subject
    }


def _receipt_digest_payload(receipt: VerifiedTrainingTargetReceipt) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "trajectory_id": receipt.trajectory_id,
        "source": receipt.source.value,
        "source_ref": receipt.source_ref,
        "evidence_refs": list(receipt.evidence_refs),
        "verifier_ref": receipt.verifier_ref,
        "target_sha256": receipt.target_sha256,
        "created_at_ms": receipt.created_at_ms,
        "source_thread_id": receipt.source_thread_id,
        "provenance": receipt.provenance,
    }


def _receipt_digest(receipt: VerifiedTrainingTargetReceipt) -> str:
    payload = json.dumps(
        _receipt_digest_payload(receipt),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(payload)


def _existing_receipt(
    *,
    trajectory_id: str,
    target: str,
    source: TrainingTargetSource,
    source_ref: str,
    evidence_refs: tuple[str, ...],
    verifier_ref: str,
    source_thread_id: str | None,
) -> VerifiedTrainingTargetReceipt | None:
    """Recover the exact prior backend-issued receipt for a crash replay."""
    rows: list[dict[str, object]] = []
    try:
        from .provenance_index import query_records

        rows = list(
            query_records(
                "training-target-receipts",
                trajectory_id=trajectory_id,
                limit=64,
            )
        )
    except Exception:
        rows = []
    if not rows:
        try:
            from .ledger import recent_records

            rows = [
                row
                for row in recent_records("training-target-receipts", limit=512)
                if str(row.get("trajectory_id") or "") == trajectory_id
            ]
        except Exception:
            rows = []

    target_sha256 = _sha256(target)
    wanted_thread = str(source_thread_id or "").strip()[:200] or None
    for row in reversed(rows):
        if (
            str(row.get("schema_version") or "") != TRAINING_TARGET_RECEIPT_VERSION
            or str(row.get("provenance") or "") != "backend_verified_target_ingress"
            or str(row.get("target_sha256") or "") != target_sha256
            or str(row.get("source") or "") != source.value
            or str(row.get("source_ref") or "") != source_ref
            or tuple(str(item) for item in (row.get("evidence_refs") or [])) != evidence_refs
            or str(row.get("verifier_ref") or "") != verifier_ref
            or (str(row.get("source_thread_id") or "").strip() or None) != wanted_thread
        ):
            continue
        try:
            receipt = VerifiedTrainingTargetReceipt(
                receipt_id=str(row.get("receipt_id") or ""),
                trajectory_id=trajectory_id,
                target=target,
                source=source,
                source_ref=source_ref,
                evidence_refs=evidence_refs,
                verifier_ref=verifier_ref,
                target_sha256=target_sha256,
                receipt_sha256=str(row.get("receipt_sha256") or ""),
                created_at_ms=int(row.get("created_at_ms") or 0),
                source_thread_id=wanted_thread,
                _issuer_token=_ISSUER_TOKEN,
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if receipt.receipt_id and receipt.created_at_ms > 0 and _receipt_digest(receipt) == receipt.receipt_sha256:
            return receipt
    return None


def issue_verified_training_target_receipt(
    *,
    trajectory_id: str,
    target: str,
    source: TrainingTargetSource | str,
    verifier_subject: str,
    source_ref: str = "",
    evidence_refs: Iterable[str] | None = None,
    steps: Iterable[ToolStep] | None = None,
    source_thread_id: str | None = None,
) -> VerifiedTrainingTargetReceipt | None:
    """Create and persist a backend-owned receipt or reject unsafe provenance.

    Human corrections derive verifier identity from the authenticated subject.
    Objective corrections additionally require a backend-captured typed verifier
    receipt from this trajectory. Model self-audit prose is never an issuer.
    """

    trajectory = str(trajectory_id or "").strip()[:200]
    corrected = str(target or "").strip()
    subject = str(verifier_subject or "").strip()
    if not trajectory:
        raise ValueError("trajectory_id is required for a verified training target")
    if not corrected:
        raise ValueError("corrected training target is empty")
    if len(corrected) > _MAX_TARGET_CHARS:
        raise ValueError("corrected training target exceeds the 24000 character limit")
    if not subject:
        raise ValueError("authenticated verifier identity is required")
    try:
        typed_source = source if isinstance(source, TrainingTargetSource) else TrainingTargetSource(str(source))
    except (TypeError, ValueError) as exc:
        raise ValueError("training target source must be human_correction or objective_correction") from exc

    refs = _normalized_refs(evidence_refs)
    normalized_source_ref = str(source_ref or "").strip()[:500]
    if typed_source == TrainingTargetSource.OBJECTIVE_CORRECTION:
        available_refs = _objective_verification_refs(steps, target=corrected)
        if not refs:
            raise ValueError("objective correction requires backend verification evidence")
        if any(ref not in available_refs for ref in refs):
            raise ValueError(
                "objective correction references unverified evidence or a verifier not bound to the corrected target"
            )
        if not normalized_source_ref:
            normalized_source_ref = refs[0]
    elif not normalized_source_ref:
        normalized_source_ref = "authenticated_human_correction"

    verifier_ref = _sha256(f"helix-training-target-verifier:{subject}")
    source_thread = str(source_thread_id or "").strip()[:200] or None
    existing = _existing_receipt(
        trajectory_id=trajectory,
        target=corrected,
        source=typed_source,
        source_ref=normalized_source_ref,
        evidence_refs=refs,
        verifier_ref=verifier_ref,
        source_thread_id=source_thread,
    )
    if existing is not None:
        return existing

    provisional = VerifiedTrainingTargetReceipt(
        receipt_id=uuid4().hex,
        trajectory_id=trajectory,
        target=corrected,
        source=typed_source,
        source_ref=normalized_source_ref,
        evidence_refs=refs,
        verifier_ref=verifier_ref,
        target_sha256=_sha256(corrected),
        receipt_sha256="",
        created_at_ms=int(time.time() * 1_000),
        source_thread_id=source_thread,
        _issuer_token=_ISSUER_TOKEN,
    )
    receipt = VerifiedTrainingTargetReceipt(
        **{
            **provisional.__dict__,
            "receipt_sha256": _receipt_digest(provisional),
        }
    )
    if not append_record("training-target-receipts", receipt.metadata()):
        return None
    return receipt


def validated_training_target_receipt(
    value: object,
    *,
    trajectory_id: str,
) -> VerifiedTrainingTargetReceipt | None:
    """Return a target only when it is the intact backend-issued typed receipt."""

    if not isinstance(value, VerifiedTrainingTargetReceipt):
        return None
    if value._issuer_token is not _ISSUER_TOKEN:
        return None
    if value.schema_version != TRAINING_TARGET_RECEIPT_VERSION:
        return None
    if value.provenance != "backend_verified_target_ingress":
        return None
    if value.trajectory_id != str(trajectory_id or "").strip():
        return None
    if value.source not in {
        TrainingTargetSource.HUMAN_CORRECTION,
        TrainingTargetSource.OBJECTIVE_CORRECTION,
    }:
        return None
    if not value.target.strip() or _sha256(value.target) != value.target_sha256:
        return None
    if _receipt_digest(value) != value.receipt_sha256:
        return None
    return value
