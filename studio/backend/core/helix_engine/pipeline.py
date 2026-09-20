# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptation gate chain. Success never auto-authorizes weight updates."""

from __future__ import annotations

from typing import Any

from .adapters import adapter_record
from .compress import compress_counterfactual
from .credit import assign_credit
from .routing import mine_corrections, route_adaptation, success_authorizes_weight_update
from .trajectory import Trajectory

GATES = (
    "observe",
    "extract",
    "deduplicate",
    "attribute",
    "critic",
    "counterfactual",
    "holdout",
    "route",
    "regression",
    "promote",
)


def run_adaptation_pipeline(
    traj: Trajectory,
    *,
    self_audit: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    credit = assign_credit(traj)
    compressed = compress_counterfactual(traj)
    corrections = mine_corrections(traj)
    critic_ok = bool(traj.verified) and bool(credit)
    route = route_adaptation(traj)
    authorized = success_authorizes_weight_update(traj)
    holdout_ok = bool(traj.holdout_passed)
    regression_ok = bool(traj.regression_passed)

    decision = "discard"
    adapter = None
    if authorized:
        decision = "reject"
    elif route == "hermes":
        decision = "promote_hermes"
    elif (
        route == "qlora"
        and critic_ok
        and holdout_ok
        and regression_ok
        and traj.dataset_hash
        and traj.allow_qlora
    ):
        decision = "promote_qlora_candidate"
        adapter = adapter_record(
            version=traj.adapter_version or "qlora-candidate",
            dataset_hash=traj.dataset_hash,
            provenance="helix-engine-pipeline",
            eval_delta=0.0,
            parent_version="",
        )
    elif route == "discard":
        decision = "discard"
    else:
        decision = "reject"

    critic = traj.extras.get("critic") if isinstance(traj.extras, dict) else None
    result = {
        "gates": list(GATES),
        "credit": credit,
        "compressed": compressed,
        "corrections": corrections,
        "route": route,
        "success_authorizes_weight_update": authorized,
        "critic_ok": critic_ok,
        "holdout_ok": holdout_ok,
        "regression_ok": regression_ok,
        "decision": decision,
        "adapter": adapter,
        "product": "Helix Harness",
        "critic": critic if isinstance(critic, dict) else None,
    }
    # The adaptive cycle is post-task intelligence. It must never become a
    # reliability dependency of the established trajectory pipeline.
    try:
        from .controller import run_closed_loop

        result["adaptive_cycle"] = run_closed_loop(
            traj, self_audit=self_audit, claims=claims
        )
    except Exception as exc:
        result["adaptive_cycle"] = {
            "schema_version": "helix.closed-loop.v1",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}"[:800],
            "fail_open": True,
        }
    return result
