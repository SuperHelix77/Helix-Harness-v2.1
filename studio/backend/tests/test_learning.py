# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

import pytest

from routes import learning


def test_autonomous_skill_proposal_receipt_survives_restart_and_replays_once(
    tmp_path, monkeypatch
):
    state_path = tmp_path / "learning" / "state.json"
    monkeypatch.setattr(learning, "_state_path", lambda: state_path)
    state = learning._empty_state()
    state.update(
        {
            "decisionMode": "autonomous",
            "allowSkillCreation": True,
        }
    )
    learning._write_state(state)
    writes: list[str] = []
    monkeypatch.setattr(
        learning,
        "_write_plain_skill",
        lambda proposal: writes.append(str(proposal["name"])),
    )
    payload = learning.LearningProposalRequest(
        kind="skill",
        title="Durable repair",
        content="Use the verified repair sequence.",
        name="durable-repair",
        idempotencyKey="turn-1:hermes:skill",
    )

    first = learning.create_learning_proposal(
        payload,
        current_subject="tester",
        autonomous_authorized=True,
    )
    # The second call re-reads state from disk, matching a process restart. The
    # durable proposal receipt must be enough to avoid recreating the skill.
    replay = learning.create_learning_proposal(
        payload,
        current_subject="tester",
        autonomous_authorized=True,
    )

    assert first["autoApproved"] is True
    assert replay["autoApproved"] is True
    assert replay["idempotent"] is True
    assert replay["proposal"]["id"] == first["proposal"]["id"]
    assert writes == ["durable-repair"]


def test_autonomous_skill_intent_reconciles_after_completion_receipt_write_failure(
    tmp_path, monkeypatch
):
    from core.inference import skills as skills_module

    state_path = tmp_path / "learning" / "state.json"
    monkeypatch.setattr(learning, "_state_path", lambda: state_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(learning.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(skills_module, "_owner_home", lambda: home)
    state = learning._empty_state()
    state.update({"decisionMode": "autonomous", "allowSkillCreation": True})
    learning._write_state(state)
    original_write_state = learning._write_state
    writes = 0

    def fail_once(state):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected completion receipt failure")
        return original_write_state(state)

    monkeypatch.setattr(learning, "_write_state", fail_once)
    payload = learning.LearningProposalRequest(
        kind="skill",
        title="Recoverable repair",
        content="Use the verified repair sequence.",
        name="recoverable-repair",
        idempotencyKey="turn-replay:skill",
    )

    with pytest.raises(OSError, match="injected completion"):
        learning.create_learning_proposal(payload, current_subject="tester")
    intent_state = learning._read_state()
    assert len(intent_state["skillCreationIntents"]) == 1
    manifest = tmp_path / "home" / ".agents" / "skills" / "recoverable-repair" / "SKILL.md"
    before = manifest.read_bytes()

    monkeypatch.setattr(learning, "_write_state", original_write_state)
    replay = learning.create_learning_proposal(payload, current_subject="tester")

    assert replay["idempotent"] is True
    assert replay["autoApproved"] is True
    assert manifest.read_bytes() == before
    assert learning._read_state()["skillCreationIntents"] == []

    with pytest.raises(learning.HTTPException) as same_key_conflict:
        learning.create_learning_proposal(
            payload.model_copy(update={"content": "A different procedure."}),
            current_subject="tester",
        )
    assert same_key_conflict.value.status_code == 409
    with pytest.raises(learning.HTTPException) as different_key_conflict:
        learning.create_learning_proposal(
            payload.model_copy(update={"idempotencyKey": "turn-replay:other"}),
            current_subject="tester",
        )
    assert different_key_conflict.value.status_code == 409


def test_autonomous_skill_intent_retries_when_writer_failed_before_first_file(tmp_path, monkeypatch):
    from core.inference import skills as skills_module
    from utils.account_context import OWNER

    state_path = tmp_path / "learning" / "state.json"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(learning, "_state_path", lambda: state_path)
    monkeypatch.setattr(learning.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(skills_module, "_owner_home", lambda: home)
    state = learning._empty_state()
    state.update({"decisionMode": "autonomous", "allowSkillCreation": True})
    learning._write_state(state)
    original_write = learning._write_plain_skill

    def fail_before_write(_proposal):
        raise skills_module.SkillError("transient writer failure")

    monkeypatch.setattr(learning, "_write_plain_skill", fail_before_write)
    payload = learning.LearningProposalRequest(
        kind="skill",
        title="Retry before first file",
        content="Use the verified retry procedure.",
        name="retry-before-first-file",
        idempotencyKey="retry-before-first-file:key",
    )
    with pytest.raises(skills_module.SkillError, match="transient writer"):
        learning.create_learning_proposal(payload, current_subject="tester")
    assert not list(home.rglob("SKILL.md"))
    assert len(learning._read_state()["skillCreationIntents"]) == 1

    monkeypatch.setattr(learning, "_write_plain_skill", original_write)
    replay = learning.create_learning_proposal(payload, current_subject="tester")
    assert replay["idempotent"] is True
    assert (home / ".agents" / "skills" / "retry-before-first-file" / "SKILL.md").is_file()
    assert learning._read_state()["skillCreationIntents"] == []


def test_autonomous_skill_intent_retries_only_missing_half_of_owner_both(tmp_path, monkeypatch):
    from core.inference import skills as skills_module

    state_path = tmp_path / "learning" / "state.json"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(learning, "_state_path", lambda: state_path)
    monkeypatch.setattr(learning.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(skills_module, "_owner_home", lambda: home)
    state = learning._empty_state()
    state.update({"decisionMode": "autonomous", "allowSkillCreation": True})
    learning._write_state(state)
    original_write = learning._write_plain_skill

    def fail_after_agents(proposal):
        name, manifest = learning._skill_manifest(proposal)
        skills_module._write_new_skill_manifest(home, name, manifest, root=Path(".agents") / "skills")
        raise skills_module.SkillError("transient after first destination")

    monkeypatch.setattr(learning, "_write_plain_skill", fail_after_agents)
    payload = learning.LearningProposalRequest(
        kind="skill",
        title="Retry both roots",
        content="Use the verified both-root retry procedure.",
        name="retry-both-roots",
        target="both",
        idempotencyKey="retry-both-roots:key",
    )
    with pytest.raises(skills_module.SkillError, match="after first"):
        learning.create_learning_proposal(payload, current_subject="tester")
    first = home / ".agents" / "skills" / "retry-both-roots" / "SKILL.md"
    second = home / ".claude" / "skills" / "retry-both-roots" / "SKILL.md"
    before = first.read_bytes()
    assert not second.exists()

    monkeypatch.setattr(learning, "_write_plain_skill", original_write)
    replay = learning.create_learning_proposal(payload, current_subject="tester")
    assert replay["idempotent"] is True
    assert first.read_bytes() == before
    assert second.is_file()


@pytest.mark.parametrize("name", ["with.dot", "with_underscore"])
def test_skill_proposal_uses_core_name_validator_before_staging(tmp_path, monkeypatch, name):
    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning" / "state.json")
    state = learning._empty_state()
    state.update({"decisionMode": "autonomous", "allowSkillCreation": True})
    learning._write_state(state)

    with pytest.raises(learning.HTTPException) as error:
        learning.create_learning_proposal(
            learning.LearningProposalRequest(
                kind="skill",
                title="Invalid skill",
                content="This must never be staged.",
                name=name,
                idempotencyKey=f"invalid:{name}",
            ),
            current_subject="tester",
        )
    assert error.value.status_code == 400
    assert learning._read_state()["skillCreationIntents"] == []


@pytest.mark.parametrize("ancestor", [".agents", ".claude"])
def test_owner_skill_promotion_refuses_symlinked_root_ancestor(tmp_path, monkeypatch, ancestor):
    from core.inference import skills as skills_module

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (home / ancestor).symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    monkeypatch.setattr(learning.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(skills_module, "_owner_home", lambda: home)

    target = "codex" if ancestor == ".agents" else "claude"
    with pytest.raises(learning.HTTPException, match="unsafe"):
        learning._write_plain_skill(
            {
                "name": f"blocked-{ancestor[1:]}",
                "title": "Blocked",
                "content": "Do not follow the link.",
                "target": target,
            }
        )
    assert list(outside.rglob("SKILL.md")) == []
import routes.memory as memory_routes


def test_learning_entries_keep_newest_content_inside_budget():
    entries = [
        {"title": "old", "content": "x" * 100},
        {"title": "new", "content": "y" * 100},
    ]
    kept = learning._fit_entries(entries, 120)
    assert kept[-1]["title"] == "new"
    assert len(learning._entry_text(kept[-1])) <= 120


def test_learning_context_is_bounded_and_marked_as_approved():
    state = learning._empty_state()
    state["memory"] = [{"title": "Loader", "content": "Verify the active backend."}]
    state["user"] = [{"title": "Style", "content": "Prefer concise status updates."}]
    context = learning._context_instruction(state)
    assert context.startswith("<hermes_memory>")
    assert "approved or autonomous-policy-admitted" in context
    assert "Loader" in context
    assert "Mem0" in context
    assert "Verify the active backend" in context
    assert "Prefer concise status updates." in context


def test_approved_memory_body_remains_available_when_mem0_is_disabled():
    state = learning._empty_state()
    state["mem0Enabled"] = False
    state["memory"] = [
        {"title": "Durable fallback", "content": "Never lose an approved lesson because vector recall is off."}
    ]

    context = learning._context_instruction(state)

    assert "Durable fallback" in context
    assert "Never lose an approved lesson because vector recall is off." in context


def test_learning_state_defaults_to_the_guarded_experience_ladder():
    state = learning._empty_state()
    assert state["mem0Enabled"] is True
    assert state["onTheFlySkills"] is True
    assert state["decisionMode"] == "ask"
    assert state["allowQloraTraining"] is False
    context = learning._context_instruction(state)
    assert "record the completed experience first" in context
    assert "use Mem0 to find recurrence" in context
    assert "deterministic held-out benchmark" in context


def test_learned_codex_target_uses_live_agent_skills_root(tmp_path, monkeypatch):
    from core.inference import skills as skills_module

    monkeypatch.setattr(learning.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(skills_module, "_owner_home", lambda: tmp_path)

    assert learning._skill_targets("codex") == [tmp_path / ".agents" / "skills"]
    assert learning._skill_targets("claude") == [tmp_path / ".claude" / "skills"]
    assert learning._skill_targets("both") == [
        tmp_path / ".agents" / "skills",
        tmp_path / ".claude" / "skills",
    ]
    name = learning._write_plain_skill(
        {
            "name": "learned-agent-skill",
            "title": "Learned Agent Skill",
            "content": "Use the verified procedure.",
            "target": "codex",
        }
    )
    assert name == "learned-agent-skill"
    assert (tmp_path / ".agents" / "skills" / name / "SKILL.md").is_file()
    assert not (tmp_path / ".codex" / "skills" / name).exists()


def test_approved_memory_is_mirrored_to_mem0_after_learning_lock_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning.json")
    state = learning._empty_state()
    proposal = {
        "id": "memory-1",
        "kind": "memory",
        "title": "Durable lesson",
        "content": "Reuse the verified backend result.",
        "reason": "repeatable",
        "name": None,
        "target": "codex",
        "sourceThreadId": "thread-1",
        "createdAt": 1,
        "recommendationAction": None,
        "recommendationReason": "",
    }
    state["pending"] = [proposal]
    learning._write_state(state)
    observed = {}

    def persist(subject, text, **kwargs):
        is_owned = getattr(learning._LEARNING_LOCK, "_is_owned", lambda: False)
        observed.update(subject=subject, text=text, kwargs=kwargs, lock_owned=is_owned())
        return {"stored": True}

    monkeypatch.setattr(memory_routes, "persist_memory_experience", persist)
    result = learning.approve_learning_proposal("memory-1", current_subject="alice@example.test")

    saved = learning._read_state()
    assert saved["pending"] == []
    assert saved["memory"][0]["title"] == "Durable lesson"
    assert observed["lock_owned"] is False
    assert observed["subject"] == "alice@example.test"
    assert observed["kwargs"]["kind"] == "hermes-memory"
    assert observed["kwargs"]["thread_id"] == "thread-1"
    assert "Reuse the verified backend result." in observed["text"]
    assert result["memoryPersistence"]["stored"] is True


def test_autonomous_memory_is_approved_and_mirrored_to_mem0_without_pending_review(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "_state_path", lambda: tmp_path / "learning.json")
    state = learning._empty_state()
    state["decisionMode"] = "autonomous"
    learning._write_state(state)
    calls = []

    def persist(subject, text, **kwargs):
        is_owned = getattr(learning._LEARNING_LOCK, "_is_owned", lambda: False)
        calls.append((subject, text, kwargs, is_owned()))
        return {"stored": True, "node": {"id": "n-1"}}

    monkeypatch.setattr(memory_routes, "persist_memory_experience", persist)
    result = learning.create_learning_proposal(
        learning.LearningProposalRequest(
            kind="memory",
            title="Autonomous lesson",
            content="Remember this verified invariant.",
            sourceThreadId="thread-auto",
        ),
        current_subject="alice@example.test",
    )

    saved = learning._read_state()
    assert result["autoApproved"] is True
    assert saved["pending"] == []
    assert saved["memory"][0]["title"] == "Autonomous lesson"
    assert len(calls) == 1
    assert calls[0][3] is False
    assert calls[0][2]["kind"] == "hermes-memory"
    assert calls[0][2]["thread_id"] == "thread-auto"
    assert result["memoryPersistence"]["stored"] is True
