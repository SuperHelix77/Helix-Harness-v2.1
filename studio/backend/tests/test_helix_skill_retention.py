# SPDX-License-Identifier: AGPL-3.0-only

import json

from core.helix_engine.critic import SelfCritic
from core.helix_engine.trajectory import ToolStep


def _step(name: str, args: dict, *, error: str | None = None) -> ToolStep:
    return ToolStep(
        name=name,
        arguments=json.dumps(args),
        result="ok",
        useful_hint="wrong" if error else "useful",
        error=error,
    )


def test_used_successful_temp_skill_is_promoted(monkeypatch):
    from core.helix_engine import skill_retention

    promoted = []
    discarded = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread: [{"name": "repo-repair"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda name, thread_id: promoted.append((name, thread_id)),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda name, thread_id: discarded.append((name, thread_id)),
    )
    steps = [
        _step("learn_skill", {"name": "repo-repair"}),
        _step("read_skill", {"name": "repo-repair"}),
    ]
    critic = SelfCritic(finished="full", right_skills=True, right_tools=True)
    receipt = []
    actions = skill_retention.reconcile_temporary_skills(
        steps,
        critic,
        thread_id="t1",
        receipt=receipt,
    )
    assert promoted == [("repo-repair", "t1")]
    assert discarded == []
    assert actions == ["promote_temp_skill:repo-repair"]
    assert receipt == [
        {
            "skill_name": "repo-repair",
            "disposition": "promoted",
            "reason": (
                "created and read successfully in this turn; task finished fully; "
                "critic accepted skill/tool choice; no tool errors"
            ),
            "evidence": {
                "created_in_turn": True,
                "read_successfully": True,
                "present_at_reconciliation": True,
                "task_finished_full": True,
                "critic_right_skills": True,
                "critic_right_tools": True,
                "tool_error_count": 0,
                "objective_outcome_status": "UNKNOWN",
            },
        }
    ]


def test_unused_or_failed_temp_skill_is_discarded(monkeypatch):
    from core.helix_engine import skill_retention

    discarded = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread: [{"name": "bad-skill"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not promote")),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda name, thread_id: discarded.append((name, thread_id)),
    )
    steps = [_step("learn_skill", {"name": "bad-skill"})]
    critic = SelfCritic(finished="partial", right_skills=False, right_tools=True)
    receipt = []
    actions = skill_retention.reconcile_temporary_skills(
        steps,
        critic,
        thread_id="t1",
        receipt=receipt,
    )
    assert discarded == [("bad-skill", "t1")]
    assert actions == ["discard_temp_skill:bad-skill"]
    assert receipt[0]["skill_name"] == "bad-skill"
    assert receipt[0]["disposition"] == "discarded"
    assert "task_finished=partial" in receipt[0]["reason"]
    assert "critic_right_skills=false" in receipt[0]["reason"]
    assert "skill_not_read_successfully_after_creation" in receipt[0]["reason"]
    assert receipt[0]["evidence"]["read_successfully"] is False


def test_failed_creation_cannot_promote_or_discard_an_older_temp_skill(monkeypatch):
    from core.helix_engine import skill_retention

    promoted = []
    discarded = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread: [{"name": "existing-skill"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda name, thread_id: promoted.append((name, thread_id)),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda name, thread_id: discarded.append((name, thread_id)),
    )
    steps = [
        _step("learn_skill", {"name": "existing-skill"}, error="already exists"),
        _step("read_skill", {"name": "existing-skill"}),
    ]

    actions = skill_retention.reconcile_temporary_skills(
        steps,
        SelfCritic(finished="full", right_skills=True, right_tools=True),
        thread_id="t1",
    )

    assert actions == []
    assert promoted == []
    assert discarded == []


def test_objective_contradiction_blocks_temp_skill_promotion(monkeypatch):
    from core.helix_engine import skill_retention

    promoted = []
    discarded = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread: [{"name": "bad-outcome"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda name, thread_id: promoted.append((name, thread_id)),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda name, thread_id: discarded.append((name, thread_id)),
    )
    receipt = []
    actions = skill_retention.reconcile_temporary_skills(
        [
            _step("learn_skill", {"name": "bad-outcome"}),
            _step("read_skill", {"name": "bad-outcome"}),
        ],
        SelfCritic(finished="full", right_skills=True, right_tools=True),
        thread_id="t1",
        receipt=receipt,
        outcome_status="CONTRADICTED",
    )

    assert promoted == []
    assert discarded == [("bad-outcome", "t1")]
    assert actions == ["discard_temp_skill:bad-outcome"]
    assert "objective_task_outcome=CONTRADICTED" in receipt[0]["reason"]
    assert receipt[0]["evidence"]["objective_outcome_status"] == "CONTRADICTED"


def test_surviving_temp_skill_can_be_promoted_after_successful_reuse_in_later_turn(monkeypatch):
    from core.helix_engine import skill_retention

    promoted = []
    monkeypatch.setattr(
        "core.inference.skills.list_temp_skills",
        lambda thread: [{"name": "survivor"}],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda name, thread_id: promoted.append((name, thread_id)),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not discard")),
    )
    receipt = []

    actions = skill_retention.reconcile_temporary_skills(
        [_step("read_skill", {"name": "survivor"})],
        SelfCritic(finished="full", right_skills=True, right_tools=True),
        thread_id="t1",
        receipt=receipt,
    )

    assert promoted == [("survivor", "t1")]
    assert actions == ["promote_temp_skill:survivor"]
    assert receipt[0]["disposition"] == "promoted"
    assert receipt[0]["evidence"]["created_in_turn"] is False
    assert receipt[0]["evidence"]["read_successfully"] is True


def test_crash_replay_recognizes_already_promoted_created_skill(monkeypatch):
    from core.helix_engine import skill_retention

    monkeypatch.setattr("core.inference.skills.list_temp_skills", lambda _thread: [])
    monkeypatch.setattr(
        "core.inference.skills.list_skills",
        lambda: [
            {
                "name": "repo-repair",
                "valid": True,
                "shadowed": False,
                "enabled": True,
            }
        ],
    )
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed promotion must not replay")
        ),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed promotion must not discard")
        ),
    )
    receipt = []
    actions = skill_retention.reconcile_temporary_skills(
        [
            _step("learn_skill", {"name": "repo-repair"}),
            _step("read_skill", {"name": "repo-repair"}),
        ],
        SelfCritic(finished="full", right_skills=True, right_tools=True),
        thread_id="t1",
        receipt=receipt,
    )

    assert actions == ["promote_temp_skill:repo-repair"]
    assert receipt[0]["disposition"] == "promoted"


def test_crash_replay_recognizes_already_discarded_created_skill(monkeypatch):
    from core.helix_engine import skill_retention

    monkeypatch.setattr("core.inference.skills.list_temp_skills", lambda _thread: [])
    monkeypatch.setattr("core.inference.skills.list_skills", lambda: [])
    monkeypatch.setattr(
        "core.inference.skills.promote_temp_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("discard path must not promote")
        ),
    )
    monkeypatch.setattr(
        "core.inference.skills.discard_temp_skill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed discard must not replay")
        ),
    )
    receipt = []
    actions = skill_retention.reconcile_temporary_skills(
        [_step("learn_skill", {"name": "bad-skill"})],
        SelfCritic(finished="partial", right_skills=False, right_tools=True),
        thread_id="t1",
        receipt=receipt,
    )

    assert actions == ["discard_temp_skill:bad-skill"]
    assert receipt[0]["disposition"] == "discarded"
