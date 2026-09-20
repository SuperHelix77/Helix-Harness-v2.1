# SPDX-License-Identifier: AGPL-3.0-only
"""Q38 capability-deficit clusters from trajectories. Not a Q38 rebuild."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .trajectory import Trajectory


def classify_step_deficits(traj: Trajectory) -> list[str]:
    labels: list[str] = []
    reads: list[str] = []
    for step in traj.steps:
        name = step.name.lower()
        if "read" in name:
            if step.arguments in reads:
                labels.append("rereads_files")
            reads.append(step.arguments)
        if step.useful_hint == "redundant" and ("search" in name or "grep" in name):
            labels.append("unnecessary_repository_searches")
        if step.useful_hint == "wrong":
            labels.append("wrong_tool_under_ambiguity")
        if step.error and "invariant" in (step.error or "").lower():
            labels.append("fails_to_preserve_tool_state_invariants")
        if "compress" in name or "handoff" in name:
            if "lost" in (step.result or "").lower() or "missing" in (step.result or "").lower():
                labels.append("loses_evidence_after_context_compression")
    return labels


def cluster_deficits(trajectories: Iterable[Trajectory]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for traj in trajectories:
        counts.update(classify_step_deficits(traj))
    return dict(counts)
