# SPDX-License-Identifier: AGPL-3.0-only
"""Helix Harness 5x path: DFlash on Auto for Qwen3.8 Darwin, fail-open to v1.1 decode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core.inference.q38_v11_optimization import q38_v11_default_load_updates, q38_v11_is_qwen38

HELIX_5X_SPECULATIVE_TYPE = "dflash"
HELIX_5X_TARGET_RATIO = 5.0


def helix_gdn_host_path(model_identifier: Optional[str], platform_name: str) -> bool:
    """Qwen3.8 Gated DeltaNet is the measured Mac host path. DFlash overlays it."""
    return platform_name == "darwin" and q38_v11_is_qwen38(model_identifier)


def helix_5x_load_updates(
    model_identifier: Optional[str],
    *,
    platform_name: str,
    max_seq_length: int,
    cache_type_kv: Optional[str],
    speculative_type: Optional[str],
    n_batch: Optional[int],
    n_ubatch: Optional[int],
) -> dict[str, object]:
    updates = q38_v11_default_load_updates(
        model_identifier,
        platform_name=platform_name,
        max_seq_length=max_seq_length,
        cache_type_kv=cache_type_kv,
        speculative_type=speculative_type,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
    )
    if platform_name != "darwin":
        return updates
    compact = "".join(ch for ch in str(model_identifier or "").lower() if ch.isalnum())
    if "qwen38" not in compact or "gguf" not in compact:
        return updates
    requested = None if speculative_type is None else str(speculative_type).strip().lower()
    if requested in {None, "", "auto", "default"}:
        updates["speculative_type"] = HELIX_5X_SPECULATIVE_TYPE
    return updates


def helix_speculation_fail_open(
    requested: str,
    engaged: str,
    *,
    accepted_drafts: int,
    sidecar_ok: bool,
) -> str:
    """Ordinary v1.1 decode when DFlash cannot prove accepted drafts."""
    if not sidecar_ok:
        return "off"
    if str(engaged).strip().lower() != "dflash":
        return "off"
    if int(accepted_drafts or 0) <= 0:
        return "off"
    if str(requested).strip().lower() in {"off", "none"}:
        return "off"
    return "dflash"


def helix_speed_ratio(baseline_tok_s: float, candidate_tok_s: float) -> float:
    if baseline_tok_s <= 0:
        return 0.0
    return float(candidate_tok_s) / float(baseline_tok_s)


def helix_may_claim_5x(ratio: float, accepted_drafts: int) -> bool:
    return float(ratio) >= HELIX_5X_TARGET_RATIO and int(accepted_drafts or 0) > 0


def write_speed_json(
    path: Path,
    *,
    baseline_tok_s: Optional[float],
    candidate_tok_s: Optional[float],
    accepted_drafts: int,
    error: Optional[str] = None,
) -> dict[str, object]:
    ratio = (
        helix_speed_ratio(baseline_tok_s, candidate_tok_s)
        if baseline_tok_s and candidate_tok_s
        else None
    )
    payload = {
        "baseline_tok_s": baseline_tok_s,
        "candidate_tok_s": candidate_tok_s,
        "accepted_drafts": int(accepted_drafts or 0),
        "ratio": ratio,
        "claim_5x": bool(ratio is not None and helix_may_claim_5x(ratio, accepted_drafts)),
        "error": error,
        "notes": "Never claim 5x with zero accepted drafts.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
