# SPDX-License-Identifier: AGPL-3.0-only
"""Helix Engine: trajectory control plane between execution and adaptation."""

from .adapters import adapter_record, rollback_adapter
from .capture import finalize_session, record_tool_execution, session_steps
from .compress import build_counterfactual_candidate, compress_counterfactual
from .credit import assign_credit
from .critic import SelfCritic, critic_from_steps, parse_self_critic
from .deficits import classify_step_deficits, cluster_deficits
from .ingest import ingest_turn
from .pipeline import GATES, run_adaptation_pipeline
from .routing import (
    mine_corrections,
    monitor_semantic_turns,
    route_adaptation,
    success_authorizes_weight_update,
)
from .trajectory import Correction, ToolStep, Trajectory, trajectory_record

__all__ = [
    "Correction",
    "SelfCritic",
    "ToolStep",
    "Trajectory",
    "trajectory_record",
    "GATES",
    "adapter_record",
    "assign_credit",
    "classify_step_deficits",
    "cluster_deficits",
    "compress_counterfactual",
    "build_counterfactual_candidate",
    "critic_from_steps",
    "finalize_session",
    "ingest_turn",
    "mine_corrections",
    "monitor_semantic_turns",
    "parse_self_critic",
    "record_tool_execution",
    "rollback_adapter",
    "route_adaptation",
    "run_adaptation_pipeline",
    "session_steps",
    "success_authorizes_weight_update",
]
