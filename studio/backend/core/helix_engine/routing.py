# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Literal

from .trajectory import Trajectory

Route = Literal["discard", "hermes", "qlora"]

_SEMANTIC_FILLER = {
    "ok",
    "sure",
    "yeah",
    "let me think",
    "as i said",
    "as i mentioned",
    "going to go ahead",
}


def success_authorizes_weight_update(traj: Trajectory) -> bool:
    """Absolute: verified success never by itself authorizes QLoRA."""
    del traj
    return False


def _critic(traj: Trajectory) -> dict:
    extras = traj.extras if isinstance(traj.extras, dict) else {}
    critic = extras.get("critic")
    return critic if isinstance(critic, dict) else {}


def route_adaptation(traj: Trajectory) -> Route:
    if traj.one_off_fact:
        return "discard"
    critic = _critic(traj)
    qlora_ready = (
        traj.allow_qlora
        and traj.frequent_behavior
        and traj.holdout_passed
        and traj.regression_passed
        and bool(traj.dataset_hash)
    )
    if qlora_ready and not critic:
        return "qlora"
    if (
        qlora_ready
        and critic.get("recommendation") == "qlora"
        and critic.get("finished") == "full"
        and critic.get("right_tools") is not False
        and not critic.get("too_many_tools")
    ):
        return "qlora"
    if critic:
        finished = critic.get("finished")
        if (
            critic.get("too_many_tools")
            or critic.get("right_tools") is False
            or critic.get("right_skills") is False
            or finished in {"partial", "none"}
            or critic.get("recommendation") in {"skill", "runtime-fix", "qlora"}
        ):
            return "hermes"
        if (
            finished == "full"
            and critic.get("recommendation") in {None, "none", ""}
            and critic.get("right_tools") is not False
            and critic.get("right_skills") is not False
            and not critic.get("too_many_tools")
            and not traj.user_corrections
        ):
            return "discard"
    if traj.steps or traj.user_corrections:
        return "hermes"
    return "discard"


def mine_corrections(traj: Trajectory) -> list[dict[str, str]]:
    return [
        {
            "state": item.state,
            "bad_action": item.bad_action,
            "failure_evidence": item.failure_evidence,
            "correct_action": item.correct_action,
            "why": item.why,
        }
        for item in traj.user_corrections
    ]


def monitor_semantic_turns(turns: list[str]) -> dict:
    unnecessary = 0
    for turn in turns:
        compact = " ".join(turn.lower().split())
        if any(token in compact for token in _SEMANTIC_FILLER) and len(compact) < 80:
            unnecessary += 1
    return {
        "unnecessary_turns": unnecessary,
        "feed_self_improvement": unnecessary > 0,
    }
