# SPDX-License-Identifier: AGPL-3.0-only
"""Per-turn self-critic. The model judges the turn; the engine decides what to do."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .trajectory import ToolStep

Finished = Literal["full", "partial", "none"]
Recommendation = Literal["skill", "qlora", "runtime-fix", "none"]
_TOO_MANY_TOOLS = 8
_TOO_MANY_REDUNDANT = 3


@dataclass
class SelfCritic:
    finished: Finished = "partial"
    right_tools: bool = True
    right_skills: bool = True
    too_many_tools: bool = False
    tool_count: int = 0
    used_skills: list[str] = field(default_factory=list)
    notes: str = ""
    recommendation: Recommendation = "none"
    skill_title: str = ""
    skill_content: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def parse_self_critic(raw: Any) -> SelfCritic:
    data = raw if isinstance(raw, dict) else {}
    finished_raw = str(data.get("finished") or data.get("completion") or "partial").strip().lower()
    finished: Finished = finished_raw if finished_raw in {"full", "partial", "none"} else "partial"
    rec_raw = str(data.get("recommendation") or data.get("recommendationAction") or "none").strip().lower()
    recommendation: Recommendation = rec_raw if rec_raw in {"skill", "qlora", "runtime-fix", "none"} else "none"
    used = data.get("usedSkills") or data.get("used_skills") or []
    if not isinstance(used, list):
        used = []
    tool_count = data.get("toolCount", data.get("tool_count", 0))
    try:
        count = max(0, int(tool_count or 0))
    except (TypeError, ValueError):
        count = 0
    return SelfCritic(
        finished=finished,
        right_tools=_as_bool(data.get("rightTools", data.get("right_tools", True))),
        right_skills=_as_bool(data.get("rightSkills", data.get("right_skills", True))),
        too_many_tools=_as_bool(data.get("tooManyTools", data.get("too_many_tools", False)), default=False)
        or count >= _TOO_MANY_TOOLS,
        tool_count=count,
        used_skills=[str(item).strip() for item in used if str(item).strip()][:16],
        notes=str(data.get("notes") or "")[:2_000],
        recommendation=recommendation,
        skill_title=str(data.get("skillTitle") or data.get("title") or "")[:240],
        skill_content=str(data.get("skillContent") or data.get("content") or "")[:8_000],
        reason=str(data.get("reason") or data.get("recommendationReason") or "")[:2_000],
    )


def critic_from_steps(steps: list[ToolStep], *, finished: Finished = "partial") -> SelfCritic:
    redundant = sum(1 for step in steps if (step.useful_hint or "").lower() == "redundant")
    wrong = any((step.useful_hint or "").lower() == "wrong" or step.error for step in steps)
    used_skills = [
        step.arguments.strip()[:80]
        for step in steps
        if step.name == "read_skill" and str(step.arguments).strip()
    ]
    too_many = len(steps) >= _TOO_MANY_TOOLS or redundant >= _TOO_MANY_REDUNDANT
    recommendation: Recommendation = "none"
    if too_many or wrong or (finished != "full" and not used_skills):
        recommendation = "skill"
    return SelfCritic(
        finished=finished,
        right_tools=not wrong,
        right_skills=bool(used_skills) or finished == "full",
        too_many_tools=too_many,
        tool_count=len(steps),
        used_skills=used_skills[:16],
        notes="deterministic critic from the captured tool loop",
        recommendation=recommendation,
        skill_title="Reuse or create a skill before changing weights" if recommendation == "skill" else "",
        skill_content="Look at enabled skills, call read_skill, and follow it. Create only a temporary skill if none covers this workflow."
        if recommendation == "skill"
        else "",
        reason="Tool-loop waste or an unfinished turn is a skill/runtime gap, not a QLoRA signal.",
    )
