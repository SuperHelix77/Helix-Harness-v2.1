# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from .trajectory import Trajectory


def _credit_key(name: str, arguments: str, result: str) -> tuple[str, str, str]:
    # Keep credit assignment consistent with cache_integrity: these are two
    # model-visible aliases for the same thread-memory retrieval path.  Only the
    # same arguments + same leading result collapse, so different retrieval
    # outcomes remain separately creditable.
    normalized_name = (
        "thread_memory_search"
        if name in {"search_memory", "search_conversation"}
        else name
    )
    return (normalized_name, arguments, result[:500])


def assign_credit(traj: Trajectory) -> list[int]:
    """Keep the useful subsequence. Success of the whole trajectory is not enough."""
    seen: set[tuple[str, str, str]] = set()
    kept: list[int] = []
    for index, step in enumerate(traj.steps):
        hint = (step.useful_hint or "").lower()
        key = _credit_key(step.name, step.arguments, step.result)
        if hint == "wrong" or step.error:
            continue
        if hint == "redundant" or key in seen:
            continue
        seen.add(key)
        if hint in {"evidence", "useful", ""}:
            kept.append(index)
    return kept
