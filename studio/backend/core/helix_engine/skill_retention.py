# SPDX-License-Identifier: AGPL-3.0-only
"""Retrospective retention for thread-scoped skills used by a completed turn."""

from __future__ import annotations

import json
from typing import Any

from .critic import SelfCritic
from .trajectory import ToolStep


def _arguments(step: ToolStep) -> dict[str, Any]:
    raw = step.arguments
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or ""))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def reconcile_temporary_skills(
    steps: list[ToolStep],
    critic: SelfCritic,
    *,
    thread_id: str | None,
    receipt: list[dict[str, Any]] | None = None,
    outcome_status: str | None = None,
) -> list[str]:
    """Promote a temporary skill only after observed successful use.

    The model is allowed to explore by creating a thread-local skill.  Durability
    is a separate post-task decision. A skill created in this turn must have been
    read back after creation and the task must complete cleanly. A surviving skill
    from an interrupted older turn can also prove itself by being read and used in
    a later clean turn. Older skills that are not used are left untouched; a later
    failure is not enough evidence to delete a previously existing skill.
    """

    if not thread_id:
        return []
    created_at: dict[str, int] = {}
    read_after_creation: set[str] = set()
    successful_reads: set[str] = set()
    for index, step in enumerate(steps):
        args = _arguments(step)
        name = str(args.get("name") or "").strip().lower()
        if not name:
            continue
        created_here = (
            step.name == "learn_skill"
            or (step.name == "create_skill" and args.get("temporary") is True)
        )
        if created_here and not step.error:
            created_at.setdefault(name, index)
        elif step.name == "read_skill" and not step.error:
            successful_reads.add(name)
            if name in created_at and index > created_at[name]:
                read_after_creation.add(name)

    created = list(created_at)

    tool_error_count = sum(bool(step.error) for step in steps)
    clean_finish = bool(
        critic.finished == "full"
        and critic.right_skills
        and critic.right_tools
        and tool_error_count == 0
        and str(outcome_status or "").upper() != "CONTRADICTED"
    )
    actions: list[str] = []

    def record(
        name: str,
        disposition: str,
        reason: str,
        *,
        present: bool | None,
        created_in_turn: bool = True,
        read_successfully: bool | None = None,
    ) -> None:
        if receipt is None or len(receipt) >= 32:
            return
        receipt.append(
            {
                "skill_name": str(name)[:120],
                "disposition": str(disposition)[:80],
                "reason": str(reason)[:800],
                "evidence": {
                    "created_in_turn": created_in_turn,
                    "read_successfully": (
                        name in read_after_creation
                        if read_successfully is None
                        else bool(read_successfully)
                    ),
                    "present_at_reconciliation": present,
                    "task_finished_full": critic.finished == "full",
                    "critic_right_skills": bool(critic.right_skills),
                    "critic_right_tools": bool(critic.right_tools),
                    "tool_error_count": tool_error_count,
                    "objective_outcome_status": str(outcome_status or "UNKNOWN")[:80],
                },
            }
        )

    def nonpromotion_reason(name: str) -> str:
        reasons: list[str] = []
        if critic.finished != "full":
            reasons.append(f"task_finished={critic.finished or 'unknown'}")
        if not critic.right_skills:
            reasons.append("critic_right_skills=false")
        if not critic.right_tools:
            reasons.append("critic_right_tools=false")
        if tool_error_count:
            reasons.append(f"tool_error_count={tool_error_count}")
        if str(outcome_status or "").upper() == "CONTRADICTED":
            reasons.append("objective_task_outcome=CONTRADICTED")
        if name not in read_after_creation:
            reasons.append("skill_not_read_successfully_after_creation")
        return "; ".join(reasons) or "final retention predicates were not satisfied"

    try:
        from core.inference.skills import (
            discard_temp_skill,
            list_skills,
            list_temp_skills,
            promote_temp_skill,
        )

        present = {str(item.get("name") or "") for item in list_temp_skills(thread_id)}
        durable = {
            str(item.get("name") or "")
            for item in list_skills()
            if isinstance(item, dict)
            and item.get("valid") is True
            and item.get("shadowed") is not True
        }
        reused_from_prior_turn = sorted((successful_reads & present) - set(created))
        for name in created:
            if name not in present:
                # A successful learn_skill call in this immutable trajectory proves
                # the temporary skill existed. On crash replay, absence can therefore
                # be the witness that the prior finalizer already consumed it.
                if clean_finish and name in read_after_creation and name in durable:
                    actions.append(f"promote_temp_skill:{name}")
                    record(
                        name,
                        "promoted",
                        "created and read successfully in this turn; task finished fully; critic accepted skill/tool choice; no tool errors",
                        present=True,
                    )
                    continue
                if not (clean_finish and name in read_after_creation):
                    actions.append(f"discard_temp_skill:{name}")
                    record(
                        name,
                        "discarded",
                        nonpromotion_reason(name),
                        present=True,
                    )
                    continue
                record(
                    name,
                    "not_present",
                    "temporary skill was created in this turn but was absent from the thread skill store at final reconciliation",
                    present=False,
                )
                continue
            if clean_finish and name in read_after_creation:
                try:
                    promote_temp_skill(name, thread_id=thread_id)
                    actions.append(f"promote_temp_skill:{name}")
                    record(
                        name,
                        "promoted",
                        "created and read successfully in this turn; task finished fully; critic accepted skill/tool choice; no tool errors",
                        present=True,
                    )
                    continue
                except Exception as exc:
                    # Promotion failure must not leave a skill silently classified as
                    # durable. Keep the temporary copy for a later repair/retry.
                    actions.append(f"retain_temp_skill_after_promotion_error:{name}")
                    record(
                        name,
                        "retained",
                        f"promotion failed with {type(exc).__name__}; temporary copy retained for retry",
                        present=True,
                    )
                    continue
            try:
                discard_temp_skill(name, thread_id=thread_id)
                actions.append(f"discard_temp_skill:{name}")
                record(
                    name,
                    "discarded",
                    nonpromotion_reason(name),
                    present=True,
                )
            except Exception as exc:
                actions.append(f"retain_temp_skill_after_discard_error:{name}")
                record(
                    name,
                    "retained",
                    f"discard failed with {type(exc).__name__}; temporary copy retained despite unmet retention predicates: {nonpromotion_reason(name)}",
                    present=True,
                )

        # A prior-turn temporary skill that was read successfully can disappear
        # only through promotion/discard or an external mutation. When this clean
        # turn deterministically promotes it, a durable skill of the same name is
        # sufficient replay proof and avoids a second promotion attempt.
        replayed_prior_promotions = sorted(
            (successful_reads & durable) - set(created) - set(reused_from_prior_turn)
        )
        for name in replayed_prior_promotions:
            if clean_finish:
                actions.append(f"promote_temp_skill:{name}")
                record(
                    name,
                    "promoted",
                    "temporary skill survived an earlier turn and was read successfully in this clean completed turn",
                    present=True,
                    created_in_turn=False,
                    read_successfully=True,
                )

        for name in reused_from_prior_turn:
            if clean_finish:
                try:
                    promote_temp_skill(name, thread_id=thread_id)
                    actions.append(f"promote_temp_skill:{name}")
                    record(
                        name,
                        "promoted",
                        "temporary skill survived an earlier turn and was read successfully in this clean completed turn",
                        present=True,
                        created_in_turn=False,
                        read_successfully=True,
                    )
                except Exception as exc:
                    actions.append(f"retain_temp_skill_after_promotion_error:{name}")
                    record(
                        name,
                        "retained",
                        f"promotion failed with {type(exc).__name__}; previously existing temporary copy retained for retry",
                        present=True,
                        created_in_turn=False,
                        read_successfully=True,
                    )
            else:
                record(
                    name,
                    "retained",
                    "previously existing temporary skill was read, but this turn did not provide a clean enough outcome to promote or delete it",
                    present=True,
                    created_in_turn=False,
                    read_successfully=True,
                )
    except Exception as exc:
        # Skills are optional; never break answer ingestion over their store.
        for name in list(dict.fromkeys([*created, *sorted(successful_reads)])):
            record(
                name,
                "unknown",
                f"temporary skill reconciliation unavailable due to {type(exc).__name__}",
                present=None,
                created_in_turn=name in created_at,
                read_successfully=name in successful_reads,
            )
        return []
    return actions
