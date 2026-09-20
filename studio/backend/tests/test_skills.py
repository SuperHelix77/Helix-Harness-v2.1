# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import errno
import os
import stat
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import skills
from routes.skills import router


def _write_skill(
    home: Path,
    source: str,
    name: str,
    *,
    description: str = "Use this skill for testing.",
    frontmatter: str = "",
    body: str = "Instructions",
) -> Path:
    root = home / (".agents" if source == "agents" else ".claude") / "skills" / name
    root.mkdir(parents = True)
    extra = f"\n{frontmatter.rstrip()}" if frontmatter else ""
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}{extra}\n---\n{body}",
        encoding = "utf-8",
    )
    return root


@pytest.fixture(autouse = True)
def _reset_tool_request_budget():
    # execute_tool never resets these, so the pagination test's tight budget must not leak.
    from core.inference import tools

    context_token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    budget_token = tools._REQUEST_RESULT_BUDGET.set(None)
    yield
    tools._REQUEST_CONTEXT_TOKENS.reset(context_token)
    tools._REQUEST_RESULT_BUDGET.reset(budget_token)


@pytest.fixture
def isolated_skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    studio = tmp_path / "studio"
    home.mkdir()
    monkeypatch.setattr(skills, "studio_root", lambda: studio)
    return home, studio


def test_enabled_skill_cache_is_fresh_after_slow_discovery(monkeypatch):
    from routes import inference as inference_routes

    scans = 0

    def discover():
        nonlocal scans
        scans += 1
        return [{"name": "cached"}]

    clock = iter((10.0, 12.0, 12.5))
    monkeypatch.setattr(skills, "enabled_skills", discover)
    monkeypatch.setattr(inference_routes.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {})

    assert inference_routes._enabled_agent_skills() == [{"name": "cached"}]
    assert inference_routes._enabled_agent_skills() == [{"name": "cached"}]
    assert scans == 1


def test_discovers_both_roots_with_agents_precedence(isolated_skills):
    home, _ = isolated_skills
    _write_skill(home, "agents", "shared", description = "Agent copy")
    _write_skill(home, "claude", "claude-only")
    _write_skill(home, "claude", "shared", description = "Claude copy")

    records = skills.list_skills(home = home)

    assert [(item["name"], item["source"], item["shadowed"]) for item in records] == [
        ("shared", "agents", False),
        ("claude-only", "claude", False),
        ("shared", "claude", True),
    ]
    assert records[0]["description"] == "Agent copy"
    assert records[0]["enabled"] is True
    assert records[2]["shadowed_by"] == "agents"


def test_bundled_skill_creator_is_enabled_and_user_override_wins(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
        ("bundled", Path(skills.__file__).with_name("bundled_skills")),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    creator = next(record for record in skills.list_skills() if record["name"] == "skill-creator")
    assert creator["source"] == "bundled"
    # skill-creator is on by default so the agent can look at / create skills in preflight.
    assert creator["enabled"] is True
    assert "create_skill" in skills.read_skill_resource("skill-creator")
    assert skills.set_skill_enabled("skill-creator", False)["enabled"] is False
    assert skills._load_overrides() == {"skill-creator": False}
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("skill-creator")
    assert skills.set_skill_enabled("skill-creator", True)["enabled"] is True
    assert skills._load_overrides() == {}

    _write_skill(home, "agents", "skill-creator", description = "User override")
    records = [record for record in skills.list_skills() if record["name"] == "skill-creator"]
    assert [(record["source"], record["shadowed"]) for record in records] == [
        ("agents", False),
        ("bundled", True),
    ]


@pytest.mark.parametrize(
    "directory,manifest",
    [
        ("wrong-dir", "---\nname: other\ndescription: valid\n---\n"),
        ("BadName", "---\nname: BadName\ndescription: valid\n---\n"),
        (
            "bad-metadata",
            "---\nname: bad-metadata\ndescription: valid\nmetadata:\n  version: 1\n---\n",
        ),
        ("no-description", "---\nname: no-description\n---\n"),
    ],
)
def test_invalid_skill_is_reported_without_hiding_valid_skills(
    isolated_skills, directory, manifest
):
    home, _ = isolated_skills
    invalid = home / ".agents" / "skills" / directory
    invalid.mkdir(parents = True)
    (invalid / "SKILL.md").write_text(manifest, encoding = "utf-8")
    _write_skill(home, "agents", "valid")

    records = skills.list_skills(home = home)

    invalid_record = next(item for item in records if item["name"] == directory)
    assert invalid_record["valid"] is False
    assert invalid_record["error"]
    assert next(item for item in records if item["name"] == "valid")["valid"] is True


@pytest.mark.skipif(
    os.name == "nt" or sys.platform == "darwin",
    reason = "Windows has no surrogate-escaped filenames and APFS rejects non-UTF-8 names",
)
def test_non_utf8_directory_name_does_not_hide_valid_skills(isolated_skills):
    home, _ = isolated_skills
    root = home / ".agents" / "skills"
    root.mkdir(parents = True)
    os.mkdir(os.fsencode(root) + b"/bad-\xff")
    _write_skill(home, "agents", "valid")

    records = skills.list_skills(home = home)

    assert [record["name"] for record in records] == ["valid"]
    json.dumps(records, ensure_ascii = False).encode("utf-8")


def test_disable_override_persists_without_touching_or_falling_through(isolated_skills):
    home, studio = isolated_skills
    winner = _write_skill(home, "agents", "shared", body = "winner")
    _write_skill(home, "claude", "shared", body = "shadowed")
    before = (winner / "SKILL.md").read_bytes()

    updated = skills.set_skill_enabled("shared", False, home = home)

    assert updated["enabled"] is False
    assert skills.enabled_skills(home = home) == []
    assert json.loads((studio / "skill-overrides.json").read_text()) == {"shared": False}
    assert (winner / "SKILL.md").read_bytes() == before
    skills.set_skill_enabled("shared", True, home = home)
    assert json.loads((studio / "skill-overrides.json").read_text()) == {}


def test_read_resource_is_contained_utf8_and_paginated(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    resource = root / "references" / "guide.md"
    resource.parent.mkdir()
    resource.write_text("abcdef", encoding = "utf-8")

    page = skills.read_skill_resource("reader", "references/guide.md", 1, page_chars = 3, home = home)

    assert "Characters: 1-4 of 6" in page
    assert "\nbcd\n" in page
    assert "offset=4" in page
    skills.set_skill_enabled("reader", False, home = home)
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("reader", home = home)


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason = "POSIX FIFOs are unavailable on this platform"
)
def test_read_resource_rejects_fifo_without_blocking(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    pipe = root / "pipe"
    os.mkfifo(pipe)
    finished = threading.Event()
    errors = []

    def read_pipe():
        try:
            skills.read_skill_resource("reader", "pipe", home = home)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target = read_pipe, daemon = True)
    worker.start()
    completed_without_writer = finished.wait(1)
    if not completed_without_writer:
        with pipe.open("wb"):
            pass
    worker.join(1)

    assert completed_without_writer, "reading a FIFO waited for a writer"
    assert errors and isinstance(errors[0], skills.SkillError)
    assert "regular file" in str(errors[0])


def test_read_resource_rejects_escaping_symlink(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    outside = home / "secret.txt"
    outside.write_text("secret", encoding = "utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        # Reason: Windows may deny symlink creation without Developer Mode.
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "link.txt", home = home)
    with pytest.raises(skills.SkillError, match = "stay inside"):
        skills.read_skill_resource("reader", "../secret.txt", home = home)


def test_read_resource_rejects_link_swapped_during_open(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    resource = root / "guide.md"
    resource.write_text("safe", encoding = "utf-8")
    outside = home / "secret.txt"
    outside.write_text("secret", encoding = "utf-8")
    original_open = os.open
    swapped = False

    def replacing_open(path, *args, **kwargs):
        nonlocal swapped
        # The descriptor-relative walk opens the bare component name against a dir_fd.
        if path in (resource, resource.name) and not swapped:
            swapped = True
            resource.unlink()
            try:
                resource.symlink_to(outside)
            except (OSError, NotImplementedError):
                # Reason: Windows may deny symlink creation without Developer Mode.
                pytest.skip("symlinks are unavailable on this platform")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "guide.md", home = home)


def test_read_resource_rejects_skill_root_swapped_after_selection(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    (root / "guide.md").write_text("safe", encoding = "utf-8")
    outside = home / "outside"
    outside.mkdir()
    (outside / "guide.md").write_text("secret", encoding = "utf-8")
    original_root = home / "original-reader"
    original_selected_skill = skills._selected_skill

    def replacing_selected_skill(name, *, home = None):
        record, path, identity = original_selected_skill(name, home = home)
        root.rename(original_root)
        try:
            root.symlink_to(outside, target_is_directory = True)
        except (OSError, NotImplementedError):
            # Reason: Windows may deny symlink creation without Developer Mode.
            pytest.skip("symlinks are unavailable on this platform")
        return record, path, identity

    monkeypatch.setattr(skills, "_selected_skill", replacing_selected_skill)
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "guide.md", home = home)


def test_skill_directory_name_must_match_exactly(isolated_skills):
    home, _ = isolated_skills
    root = home / ".agents" / "skills" / "ｓｋｉｌｌ"
    root.mkdir(parents = True)
    (root / "SKILL.md").write_text(
        "---\nname: skill\ndescription: test\n---\n",
        encoding = "utf-8",
    )

    record = skills.list_skills(home = home)[0]

    assert record["valid"] is False
    assert "match its parent directory" in record["error"]


def test_create_skill_writes_valid_manifest_without_overwriting(isolated_skills):
    home, _ = isolated_skills

    record = skills.create_skill(
        "release-notes",
        "Draft concise release notes.",
        "# Workflow\n\n1. Inspect the diff.\n2. Summarize user-visible changes.",
        home = home,
    )

    assert record["name"] == "release-notes"
    assert record["source"] == "agents"
    created = home / ".agents" / "skills" / "release-notes" / "SKILL.md"
    metadata, real_dir = skills._validate_skill_dir(created.parent)
    assert metadata["description"] == "Draft concise release notes."
    assert real_dir == created.parent
    with pytest.raises(skills.SkillError, match = "already exists"):
        skills.create_skill("release-notes", "Different", "Do something else.", home = home)
    assert "Different" not in created.read_text(encoding = "utf-8")


@pytest.mark.parametrize("name", ("../escape", "Bad Name", "con"))
def test_create_skill_rejects_unsafe_names(isolated_skills, name):
    home, _ = isolated_skills
    with pytest.raises(skills.SkillError):
        skills.create_skill(name, "Description", "Instructions", home = home)


def test_create_skill_rejects_a_linked_agents_ancestor(isolated_skills):
    home, _ = isolated_skills
    outside = home.parent / "outside"
    outside.mkdir()
    try:
        (home / ".agents").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(skills.SkillError, match = "unsafe"):
        skills.create_skill("escaped", "Description", "Instructions", home = home)

    assert not (outside / "skills").exists()


def test_create_skill_tool_invalidates_the_inference_cache(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module
    from routes import inference as inference_routes

    home, _ = isolated_skills
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [{"name": "stale"}])}
    )

    result = tools_module.execute_tool(
        "create_skill",
        {"name": "fresh", "description": "Description", "instructions": "Instructions"},
    )

    assert "Created Agent Skill 'fresh'" in result
    assert inference_routes._AGENT_SKILLS_CACHE == {}


def test_create_skill_tool_does_not_commit_when_override_clear_fails(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, studio = isolated_skills
    studio.mkdir()
    (studio / "skill-overrides.json").write_text('{"blocked":false}', encoding = "utf-8")
    monkeypatch.setenv("HOME", str(home))

    def deny_override_write(_overrides):
        raise PermissionError("read-only overrides")

    monkeypatch.setattr(skills, "_save_overrides", deny_override_write)
    result = tools_module.execute_tool(
        "create_skill",
        {"name": "blocked", "description": "Description", "instructions": "Instructions"},
    )

    assert result.startswith("Error:")
    assert not (home / ".agents" / "skills" / "blocked").exists()


def test_learn_skill_tool_bootstraps_a_thread_scoped_temporary_skill(
    isolated_skills, tmp_path, monkeypatch
):
    from core.inference import tools as tools_module

    monkeypatch.setattr(skills, "_temp_skills_root", lambda thread_id: tmp_path / thread_id)

    result = tools_module.execute_tool(
        "learn_skill",
        {
            "name": "inspect-contract",
            "description": "Inspect this kind of contract before editing.",
            "instructions": "Read the contract, trace its callers, then change the smallest shared layer.",
        },
        thread_id = "thread-a",
    )

    assert "Learned temporary thread skill 'inspect-contract'" in result
    assert [item["name"] for item in skills.list_temp_skills("thread-a")] == [
        "inspect-contract"
    ]
    assert skills.list_temp_skills("thread-b") == []
    assert tools_module.is_high_risk_tool_call("learn_skill", {}) is False


def test_learn_skill_tool_requires_a_thread_scope(isolated_skills):
    from core.inference import tools as tools_module

    result = tools_module.execute_tool(
        "learn_skill",
        {
            "name": "inspect-contract",
            "description": "Inspect this kind of contract before editing.",
            "instructions": "Read the contract first.",
        },
    )

    assert result == "Error: learn_skill needs a thread id so the skill can stay thread-scoped."


def test_catalog_is_bounded_at_complete_entries():
    candidates = [{"name": f"skill-{index}", "description": "x" * 300} for index in range(20)]

    catalog = skills.format_skill_catalog(candidates)

    *listed, marker = catalog.splitlines()
    assert len("\n".join(listed).encode("utf-8")) <= skills.MAX_SKILL_CATALOG_BYTES
    assert all(line.startswith("- skill-") for line in listed)
    assert (
        marker
        == f"- {20 - len(listed)} more enabled skills not listed; mention one as @skill-name."
    )
    large = skills.format_skill_catalog(candidates, budget = skills.LARGE_SKILL_CATALOG_BYTES)
    assert len(large.splitlines()) > len(listed)
    assert "more enabled skills" not in skills.format_skill_catalog(candidates[:2])


def test_linked_skill_directory_is_followed_once_and_pinned(isolated_skills, tmp_path):
    home, _ = isolated_skills
    real = tmp_path / "dotfiles" / "skills" / "linked"
    real.mkdir(parents = True)
    (real / "SKILL.md").write_text(
        "---\nname: linked\ndescription: Linked in.\n---\nREAL", encoding = "utf-8"
    )
    root = home / ".agents" / "skills"
    root.mkdir(parents = True)
    try:
        (root / "linked").symlink_to(real, target_is_directory = True)
        (root / "dangling").symlink_to(tmp_path / "gone", target_is_directory = True)
        (root / "to-file").symlink_to(real / "SKILL.md")
        (real / "escape.md").symlink_to(tmp_path / "dotfiles")
    except (OSError, NotImplementedError):
        # Reason: Windows may deny symlink creation without Developer Mode.
        pytest.skip("symlinks are unavailable on this platform")

    records = {record["name"]: record for record in skills.list_skills(home = home)}

    assert records["linked"]["valid"] is True
    assert records["dangling"]["valid"] is False and records["to-file"]["valid"] is False
    assert skills.read_skill_resource("linked", home = home).endswith("REAL")
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("linked", "escape.md", home = home)


def test_catalog_skips_an_oversized_entry_without_hiding_later_skills():
    candidates = [
        {"name": "oversized", "description": "界" * 600},
        {"name": "usable", "description": "Use this skill."},
    ]

    catalog = skills.format_skill_catalog(candidates)

    assert "oversized" not in catalog
    assert catalog.splitlines() == [
        "- usable: Use this skill.",
        "- 1 more enabled skills not listed; mention one as @skill-name.",
    ]


def test_authenticated_list_and_toggle_routes(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    long_body = "Route-private instructions.\n" + ("x" * 9_000)
    _write_skill(home, "agents", "api-skill", body=long_body)
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    app = FastAPI()
    app.include_router(router, prefix = "/api/skills")
    assert TestClient(app).get("/api/skills").status_code in (401, 403)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "api-skill"
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    response = client.get("/api/skills/api-skill")
    assert response.status_code == 200
    assert response.json()["instructions"].endswith(long_body)
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    monkeypatch.setattr(
        inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [{"name": "stale"}])}
    )
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": True})
    assert response.status_code == 200
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    assert client.put("/api/skills/api-skill/enabled", json = {"enabled": "false"}).status_code == 422


def test_local_skill_manager_routes_are_account_private_and_deduplicate_managed_targets(
    tmp_path, monkeypatch
):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, current_account_id, run_as

    home = tmp_path / "home"
    home.mkdir()
    accounts_root = tmp_path / "accounts"
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        skills_routes,
        "workspace_root",
        lambda: accounts_root / current_account_id(),
    )
    alice = AccountContext("route-alice", "alice")
    bob = AccountContext("route-bob", "bob")

    created = run_as(
        alice,
        skills_routes.create_skill,
        skills_routes.CreateSkillRequest(
            name="route-private",
            description="Alice only",
            instructions="Keep this in the account workspace.",
            target="both",
        ),
        "alice",
    )
    assert created["skills"][0]["path"] == "skills/route-private"
    assert len(run_as(alice, skills_routes._targets, "both")) == 1
    assert (accounts_root / "route-alice" / "skills" / "route-private" / "SKILL.md").is_file()
    assert "route-alice" not in str(created)
    assert run_as(bob, skills_routes.get_local_skills, "bob")["skills"] == []

    # Managed local installs are confined to the current account workspace.
    source = accounts_root / "route-bob" / "source"
    source.mkdir(parents=True)
    package = source / "installed"
    package.mkdir()
    (package / "SKILL.md").write_text(
        "---\nname: installed\ndescription: Installed\n---\nBody\n",
        encoding="utf-8",
    )
    installed = run_as(
        bob,
        skills_routes.install_skill,
        skills_routes.InstallSkillRequest(source=str(source), target="claude"),
        "bob",
    )
    assert (accounts_root / "route-bob" / "skills" / "installed" / "SKILL.md").is_file()
    assert all(not str(item.get("path", "")).startswith(str(tmp_path)) for item in installed["skills"])

    owner = run_as(
        AccountContext("owner", "unsloth", "owner"),
        skills_routes.create_skill,
        skills_routes.CreateSkillRequest(
            name="owner-route",
            description="Owner",
            instructions="Keep legacy owner behavior.",
            target="codex",
        ),
        "owner",
    )
    assert (home / ".codex" / "skills" / "owner-route" / "SKILL.md").is_file()
    assert owner["skills"][0]["path"].startswith(str(home))


def test_managed_local_install_cannot_read_owner_or_outside_sources(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, current_account_id, run_as

    home = tmp_path / "home"
    home.mkdir()
    accounts_root = tmp_path / "accounts"
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        skills_routes,
        "workspace_root",
        lambda: accounts_root / current_account_id(),
    )
    owner_secret = home / ".codex" / "skills" / "owner-secret"
    owner_secret.mkdir(parents=True)
    (owner_secret / "SKILL.md").write_text(
        "---\nname: owner-secret\ndescription: private\n---\nOWNER_SECRET",
        encoding="utf-8",
    )
    outside = tmp_path / "outside" / "secret"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text(
        "---\nname: outside\ndescription: private\n---\nOUTSIDE_SECRET",
        encoding="utf-8",
    )
    alice = AccountContext("route-confined-alice", "alice")

    for source in (str(owner_secret.parent), str(outside), "../../home/.codex/skills"):
        with pytest.raises(HTTPException) as error:
            run_as(alice, skills_routes._install, source, "codex")
        assert error.value.status_code == 400
        assert "OWNER_SECRET" not in str(error.value.detail)
        assert "OUTSIDE_SECRET" not in str(error.value.detail)
    assert not (accounts_root / alice.account_id / "skills" / "owner-secret").exists()


def test_local_install_rejects_hardlinked_and_linked_package_files(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, current_account_id, run_as

    home = tmp_path / "home"
    home.mkdir()
    accounts_root = tmp_path / "accounts"
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        skills_routes,
        "workspace_root",
        lambda: accounts_root / current_account_id(),
    )
    alice = AccountContext("route-package-alice", "alice")
    source = accounts_root / alice.account_id / "imports" / "hardlinked"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: hardlinked\ndescription: test\n---\nBody", encoding="utf-8"
    )
    secret = tmp_path / "owner-secret.txt"
    secret.write_text("OWNER_SECRET", encoding="utf-8")
    (source / "resource.txt").hardlink_to(secret)
    with pytest.raises(HTTPException, match="hard-linked"):
        run_as(alice, skills_routes._install, str(source.parent), "codex")

    linked = accounts_root / alice.account_id / "linked-imports" / "linked"
    linked.mkdir(parents=True)
    (linked / "SKILL.md").write_text(
        "---\nname: linked\ndescription: test\n---\nBody", encoding="utf-8"
    )
    (linked / "resource.txt").symlink_to(secret)
    with pytest.raises(HTTPException, match="symlinks"):
        run_as(alice, skills_routes._install, str(linked.parent), "codex")


def test_local_owner_codex_skill_round_trips_through_read_route(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")

    run_as(
        owner,
        skills_routes.create_skill,
        skills_routes.CreateSkillRequest(
            name = "codex-roundtrip",
            description = "Owner Codex copy",
            instructions = "Read this from the legacy Codex root.",
            target = "codex",
        ),
        "owner",
    )

    result = run_as(owner, skills_routes.read_skill, "codex-roundtrip", "owner")
    assert result["description"] == "Owner Codex copy"
    assert result["instructions"].endswith("Read this from the legacy Codex root.\n")


def test_stage_basename_swap_fails_without_publishing_or_deleting_replacement(
    tmp_path, monkeypatch
):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    package = skills_routes._Package(
        b"---\nname: swapped\ndescription: test\n---\nbody\n",
        ((("SKILL.md",), b"---\nname: swapped\ndescription: test\n---\nbody\n", 0o600),),
    )

    def swap_then_abort(parent, source, destination):
        moved = f"{source}.attacker-owned"
        os.rename(source, moved, src_dir_fd = parent, dst_dir_fd = parent)
        try:
            os.symlink(str(outside), source, dir_fd = parent)
        except (OSError, NotImplementedError):
            pytest.skip("symlink replacement is unavailable on this platform")
        raise OSError(errno.EAGAIN, "stage basename replaced")

    monkeypatch.setattr(skills_routes, "_rename_noreplace", swap_then_abort)
    with pytest.raises(HTTPException, match = "publication"):
        run_as(
            owner,
            skills_routes._publish_package,
            package,
            home / ".codex" / "skills",
            "swapped",
        )
    assert not (home / ".codex" / "skills" / "swapped").exists()
    assert not any(outside.iterdir())


def test_stage_symlink_swap_at_rename_is_rejected_before_syscall(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    marker = b"---\nname: swap-at-rename\ndescription: test\n---\nbody\n"
    package = skills_routes._Package(marker, ((("SKILL.md",), marker, 0o600),))
    original_rename = skills_routes._rename_noreplace

    def swap_then_call_original(parent, source, destination):
        os.rename(source, f"{source}.attacker-owned", src_dir_fd = parent, dst_dir_fd = parent)
        try:
            os.symlink(str(outside), source, dir_fd = parent)
        except (OSError, NotImplementedError):
            pytest.skip("symlink replacement is unavailable on this platform")
        return original_rename(parent, source, destination)

    monkeypatch.setattr(skills_routes, "_rename_noreplace", swap_then_call_original)
    with pytest.raises(HTTPException, match = "publication"):
        run_as(
            owner,
            skills_routes._publish_package,
            package,
            home / ".codex" / "skills",
            "swap-at-rename",
        )
    assert not (home / ".codex" / "skills" / "swap-at-rename").exists()
    assert not any(outside.iterdir())


def test_stage_swap_between_fd_check_and_rename_is_recovered_without_unlinking(
    tmp_path, monkeypatch
):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    marker = b"---\nname: swap-between\ndescription: test\n---\nbody\n"
    package = skills_routes._Package(marker, ((("SKILL.md",), marker, 0o600),))
    original_raw = skills_routes._rename_noreplace_raw
    swapped = False

    def swap_inside_rename(parent, source, destination):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(source, f"{source}.attacker-owned", src_dir_fd = parent, dst_dir_fd = parent)
            try:
                os.symlink(str(outside), source, dir_fd = parent)
            except (OSError, NotImplementedError):
                pytest.skip("symlink replacement is unavailable on this platform")
        return original_raw(parent, source, destination)

    monkeypatch.setattr(skills_routes, "_rename_noreplace_raw", swap_inside_rename)
    with pytest.raises(HTTPException, match = "publication"):
        run_as(
            owner,
            skills_routes._publish_package,
            package,
            home / ".codex" / "skills",
            "swap-between",
        )
    assert not (home / ".codex" / "skills" / "swap-between").exists()
    assert not any(outside.iterdir())


def test_catalog_skips_hardlinked_marker_but_source_install_fails_closed(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    root = home / ".codex" / "skills"
    good = root / "good"
    bad = root / "bad"
    good.mkdir(parents = True)
    bad.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: good\ndescription: valid\n---\nbody\n", encoding = "utf-8"
    )
    secret = tmp_path / "secret-marker"
    secret.write_text("---\nname: bad\ndescription: secret\n---\nSECRET\n", encoding = "utf-8")
    try:
        (bad / "SKILL.md").hardlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("hard links are unavailable on this platform")

    listed = run_as(owner, skills_routes._list_skills)
    assert [item["name"] for item in listed] == ["good"]

    source = tmp_path / "source"
    source.mkdir()
    source_bad = source / "bad"
    source_good = source / "good"
    source_bad.mkdir()
    source_good.mkdir()
    (source_good / "SKILL.md").write_text(
        "---\nname: good-source\ndescription: valid\n---\nbody\n", encoding = "utf-8"
    )
    (source_bad / "SKILL.md").hardlink_to(secret)
    with pytest.raises(HTTPException, match = "unsafe"):
        run_as(owner, skills_routes._install, str(source), "codex")
    assert not (root / "good-source").exists()


def test_fallback_catalog_handles_missing_root_marker_and_rejects_linked_ancestor(
    tmp_path, monkeypatch
):
    import routes.skills as skills_routes

    root = tmp_path / "catalog"
    child = root / "child"
    child.mkdir(parents = True)
    (child / "SKILL.md").write_text(
        "---\nname: child\ndescription: valid\n---\nbody\n", encoding = "utf-8"
    )
    monkeypatch.setattr(skills_routes, "_DIR_FD_READS", False)
    entries = skills_routes._safe_skill_entries(root, target_root = True)
    assert [entry.path.name for entry in entries] == ["child"]

    secret_root = tmp_path / "secret-root"
    secret_child = secret_root / "private"
    secret_child.mkdir(parents = True)
    (secret_child / "SKILL.md").write_text(
        "---\nname: private\ndescription: secret\n---\nSECRET\n", encoding = "utf-8"
    )
    linked = tmp_path / "linked-catalog"
    try:
        linked.symlink_to(secret_root, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    assert skills_routes._safe_skill_entries(linked, target_root = True) == []


def test_fallback_marker_aba_does_not_disclose_outside_content(tmp_path, monkeypatch):
    import routes.skills as skills_routes

    root = tmp_path / "catalog"
    child = root / "race"
    child.mkdir(parents = True)
    marker = child / "SKILL.md"
    marker.write_text(
        "---\nname: race\ndescription: safe\n---\nSAFE\n", encoding = "utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "SKILL.md"
    outside_marker.write_text(
        "---\nname: race\ndescription: private\n---\nOWNER_SECRET\n", encoding = "utf-8"
    )
    monkeypatch.setattr(skills_routes, "_DIR_FD_READS", False)
    original_lstat = skills_routes.os.lstat
    child_lstat_calls = 0
    swapped = False
    restored = False
    saved = child.with_name("race-original")

    def swap_after_child_identity(path):
        nonlocal child_lstat_calls, swapped
        result = original_lstat(path)
        if Path(path) == child:
            child_lstat_calls += 1
            # The first lstat is the traversal identity.  Replace the child
            # only after the marker-side component validation has inspected it,
            # but before the pinned directory open.
            if child_lstat_calls == 2 and not swapped:
                swapped = True
                child.rename(saved)
                outside.rename(child)
        return result

    monkeypatch.setattr(skills_routes.os, "lstat", swap_after_child_identity)
    try:
        entries = skills_routes._safe_skill_entries(root, target_root = True)
    finally:
        if swapped:
            child.rename(outside)
            saved.rename(child)
            restored = True
    assert swapped is True
    assert restored is True
    assert entries == []
    assert "OWNER_SECRET" not in str(entries)


def test_fallback_catalog_skips_unreadable_marker_sibling(tmp_path, monkeypatch):
    import routes.skills as skills_routes

    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("permission bits are not enforced for this user")
    root = tmp_path / "catalog"
    good = root / "good"
    bad = root / "bad"
    good.mkdir(parents = True)
    bad.mkdir()
    (good / "SKILL.md").write_text(
        "---\nname: good\ndescription: valid\n---\nGOOD\n", encoding = "utf-8"
    )
    (bad / "SKILL.md").write_text(
        "---\nname: bad\ndescription: hidden\n---\nBAD\n", encoding = "utf-8"
    )
    bad.chmod(0)
    monkeypatch.setattr(skills_routes, "_DIR_FD_READS", False)
    try:
        entries = skills_routes._safe_skill_entries(root, target_root = True)
    finally:
        bad.chmod(0o700)
    assert [entry.path.name for entry in entries] == ["good"]


def test_fallback_marker_path_hook_is_not_used_for_aba(tmp_path, monkeypatch):
    """The fallback must bind a child directory before any marker path read."""
    import routes.skills as skills_routes

    root = tmp_path / "catalog"
    child = root / "race"
    child.mkdir(parents = True)
    marker = child / "SKILL.md"
    marker.write_text(
        "---\nname: race\ndescription: safe\n---\nSAFE\n", encoding = "utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: race\ndescription: private\n---\nOWNER_SECRET\n", encoding = "utf-8"
    )
    monkeypatch.setattr(skills_routes, "_DIR_FD_READS", False)
    original_lstat = skills_routes.os.lstat
    marker_lstat_seen = False

    def reject_marker_lstat(path):
        nonlocal marker_lstat_seen
        if Path(path) == marker:
            marker_lstat_seen = True
        return original_lstat(path)

    monkeypatch.setattr(skills_routes.os, "lstat", reject_marker_lstat)
    entries = skills_routes._safe_skill_entries(root, target_root = True)
    assert marker_lstat_seen is False
    assert [entry.path.name for entry in entries] == ["race"]
    assert "OWNER_SECRET" not in str(entries)


def test_nested_discovery_is_bounded_before_accumulating_markers(tmp_path, monkeypatch):
    import routes.skills as skills_routes

    root = tmp_path / "source"
    nested = root / "skills"
    nested.mkdir(parents = True)
    monkeypatch.setattr(skills_routes, "_MAX_LOCAL_DISCOVERY_ENTRIES", 4)
    for index in range(5):
        item = nested / f"skill-{index}"
        item.mkdir()
        (item / "SKILL.md").write_text(
            f"---\nname: skill-{index}\ndescription: test\n---\nbody\n", encoding = "utf-8"
        )
    with pytest.raises(HTTPException, match = "too many entries"):
        skills_routes._safe_skill_entries(root)


def test_install_preserves_user_executable_mode_without_special_bits(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, current_account_id, run_as

    home = tmp_path / "home"
    home.mkdir()
    accounts_root = tmp_path / "accounts"
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        skills_routes, "workspace_root", lambda: accounts_root / current_account_id()
    )
    alice = AccountContext("mode-alice", "alice")
    source = accounts_root / alice.account_id / "imports" / "source"
    package = source / "mode-skill"
    package.mkdir(parents = True)
    (package / "SKILL.md").write_text(
        "---\nname: mode-skill\ndescription: mode\n---\nbody\n", encoding = "utf-8"
    )
    tool = package / "run.sh"
    tool.write_text("#!/bin/sh\necho safe\n", encoding = "utf-8")
    tool.chmod(0o775)

    run_as(alice, skills_routes._install, str(source), "codex")
    installed = accounts_root / alice.account_id / "skills" / "mode-skill" / "run.sh"
    mode = stat.S_IMODE(installed.stat().st_mode)
    assert mode & stat.S_IXUSR
    assert not mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)


def test_stage_replacement_with_precious_files_is_never_cleaned(tmp_path, monkeypatch):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    marker = b"---\nname: precious-stage\ndescription: test\n---\nbody\n"
    package = skills_routes._Package(marker, ((("SKILL.md",), marker, 0o600),))
    original_open = skills_routes.os.open
    replacement: dict[str, Path] = {}
    swapped = False

    def replace_stage_before_open(path, *args, **kwargs):
        nonlocal swapped
        if (
            isinstance(path, str)
            and path.startswith(".skill-stage-")
            and kwargs.get("dir_fd") is not None
            and not swapped
        ):
            swapped = True
            parent = kwargs["dir_fd"]
            moved = f"{path}.original-stage"
            os.rename(path, moved, src_dir_fd = parent, dst_dir_fd = parent)
            os.mkdir(path, 0o700, dir_fd = parent)
            replacement_fd = original_open(
                path,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd = parent,
            )
            precious_fd = original_open(
                "precious.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd = replacement_fd,
            )
            try:
                os.write(precious_fd, b"DO_NOT_DELETE")
            finally:
                os.close(precious_fd)
                os.close(replacement_fd)
            replacement["path"] = Path(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(skills_routes.os, "open", replace_stage_before_open)
    with pytest.raises(HTTPException):
        run_as(
            owner,
            skills_routes._publish_package,
            package,
            home / ".codex" / "skills",
            "precious-stage",
        )
    assert swapped is True
    replacement_path = home / ".codex" / "skills" / replacement["path"]
    assert (replacement_path / "precious.txt").read_bytes() == b"DO_NOT_DELETE"
    assert not (home / ".codex" / "skills" / "precious-stage").exists()


def test_unreadable_final_replacement_is_quarantined_without_public_destination(
    tmp_path, monkeypatch
):
    import routes.skills as skills_routes
    from utils.account_context import AccountContext, run_as

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(skills_routes.Path, "home", classmethod(lambda cls: home))
    owner = AccountContext("owner", "unsloth", "owner")
    marker = b"---\nname: unreadable-final\ndescription: test\n---\nbody\n"
    package = skills_routes._Package(marker, ((("SKILL.md",), marker, 0o600),))
    original_raw = skills_routes._rename_noreplace_raw
    swapped = False
    replacement: dict[str, str] = {}

    def install_then_replace(parent, source, destination):
        nonlocal swapped
        result = original_raw(parent, source, destination)
        if destination == "unreadable-final" and not swapped:
            swapped = True
            saved = f"{destination}.published-original"
            os.rename(destination, saved, src_dir_fd = parent, dst_dir_fd = parent)
            os.mkdir(destination, 0o700, dir_fd = parent)
            replacement_fd = os.open(
                destination,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd = parent,
            )
            precious_fd = os.open(
                "precious.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd = replacement_fd,
            )
            os.write(precious_fd, b"DO_NOT_DELETE")
            os.close(precious_fd)
            os.close(replacement_fd)
            os.chmod(destination, 0o000, dir_fd = parent, follow_symlinks = False)
            replacement["stage"] = source
            replacement["saved"] = saved
        return result

    monkeypatch.setattr(skills_routes, "_rename_noreplace_raw", install_then_replace)
    with pytest.raises(HTTPException, match = "publication"):
        run_as(
            owner,
            skills_routes._publish_package,
            package,
            home / ".codex" / "skills",
            "unreadable-final",
        )
    assert swapped is True
    root = home / ".codex" / "skills"
    assert not os.path.lexists(root / "unreadable-final")
    quarantined = root / replacement["stage"]
    os.chmod(quarantined, 0o700)
    assert (quarantined / "precious.txt").read_bytes() == b"DO_NOT_DELETE"


@pytest.mark.skipif(os.name == "nt", reason = "RLIMIT_NOFILE is POSIX-only")
def test_package_writer_uses_depth_bounded_descriptors_under_low_nofile(tmp_path):
    import resource
    import routes.skills as skills_routes

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft <= 128 or hard < 128:
        pytest.skip("the process limit is already too low for this check")
    stage = tmp_path / "stage"
    stage.mkdir()
    marker = b"---\nname: fd-bounded\ndescription: test\n---\nbody\n"
    files = [(('SKILL.md',), marker, 0o600)]
    for index in range(150):
        files.append(((f"directory-{index}", "payload.txt"), b"payload", 0o755))
    descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
        skills_routes._write_package_at(
            descriptor, skills_routes._Package(marker, tuple(files))
        )
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        os.close(descriptor)
    assert (stage / "directory-149" / "payload.txt").is_file()


def test_route_skill_read_does_not_mix_manifest_replacement_between_pages(
    isolated_skills, monkeypatch
):
    import routes.skills as skills_routes

    home, _ = isolated_skills
    root = _write_skill(home, "agents", "race", body="A" * 9_000)
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    marker = root / "SKILL.md"
    replacement = root / "SKILL.md.replacement"
    replacement.write_text(
        "---\nname: race\ndescription: Use this skill for testing.\n---\n" + "B" * 9_000,
        encoding="utf-8",
    )
    original_read = skills._read_limited
    replaced = False

    def replacing_read(path, limit, **kwargs):
        nonlocal replaced
        raw = original_read(path, limit, **kwargs)
        if path == marker and not replaced:
            replaced = True
            os.replace(replacement, marker)
        return raw

    monkeypatch.setattr(skills, "_read_limited", replacing_read)
    result = skills_routes.read_skill("race")

    assert result["instructions"].endswith("A" * 9_000)
    assert "B" * 200 not in result["instructions"]
    assert replaced is True


def test_skill_tool_selection_honors_explicit_allowlist(isolated_skills, monkeypatch):
    import asyncio

    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    home, _ = isolated_skills
    _write_skill(home, "agents", "guided")
    _write_skill(home, "agents", "skill-creator")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", skills.enabled_skills)
    read_only = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enable_tools = True,
        enabled_tools = ["read_skill"],
        permission_mode = "auto",
        stream = True,
    )

    selected = asyncio.run(
        inference_routes._select_request_tools(read_only, tools_on = True, mcp_allowed = False)
    )
    names = [tool["function"]["name"] for tool in selected]
    inference_routes._reject_confirm_gate_without_channel(
        read_only, ui_events = False, selected_names = set(names)
    )
    assert names == ["read_skill"]

    local_default = read_only.model_copy(update = {"enabled_tools": ["read_skill", "create_skill"]})
    selected = asyncio.run(
        inference_routes._select_request_tools(local_default, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill", "create_skill"]


def test_learn_skill_is_selectable_before_any_skill_exists(isolated_skills, monkeypatch):
    import asyncio

    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", lambda: [])

    learn_only = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enable_tools = True,
        enabled_tools = ["learn_skill"],
        thread_id = "thread-a",
    )
    selected = asyncio.run(
        inference_routes._select_request_tools(learn_only, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["learn_skill", "read_skill"]

    create_then_read = learn_only.model_copy(
        update = {"enabled_tools": ["learn_skill", "read_skill"]}
    )
    selected = asyncio.run(
        inference_routes._select_request_tools(
            create_then_read, tools_on = True, mcp_allowed = False
        )
    )
    assert [tool["function"]["name"] for tool in selected] == ["learn_skill", "read_skill"]
    nudge = inference_routes._build_tool_action_nudge(
        tools = selected,
        model_name = "test",
        thread_id = "thread-a",
    )
    assert "learn_skill always creates a temporary skill scoped to this thread" in nudge

    legacy_client = learn_only.model_copy(update = {"enabled_tools": ["web_search"]})
    selected = asyncio.run(
        inference_routes._select_request_tools(
            legacy_client, tools_on = True, mcp_allowed = False
        )
    )
    assert [tool["function"]["name"] for tool in selected] == [
        "web_search",
        "learn_skill",
        "read_skill",
    ]


def test_skill_tools_registration_selection_and_prompt(isolated_skills, monkeypatch):
    import asyncio

    from core.inference import tools as tools_module
    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    home, _ = isolated_skills
    _write_skill(home, "agents", "guided", description = "Guide this task")
    _write_skill(home, "agents", "skill-creator")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", skills.enabled_skills)
    payload = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enabled_tools = ["read_skill", "create_skill"],
    )

    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill", "create_skill"]
    assert tools_module.is_always_safe_tool("read_skill") is True
    assert tools_module.is_high_risk_tool_call("create_skill", {}) is True
    result = tools_module.execute_tool("read_skill", {"name": "guided"})
    assert "Skill: guided" in result
    nudge = inference_routes._build_tool_action_nudge(tools = selected, model_name = "test")
    assert "- guided: Guide this task" in nudge
    assert "@skill-name" in nudge
    assert "create_skill" in nudge
    # Codex and external paths skip the general nudge but keep the catalog for @mentions.
    narrow = inference_routes._build_tool_action_nudge(
        tools = [*selected, tools_module.WEB_SEARCH_TOOL],
        model_name = "test",
        full_access_only = True,
    )
    assert "- guided: Guide this task" in narrow
    assert inference_routes._TOOL_BASE_NUDGE not in narrow
    assert "web_search" not in narrow

    skills.set_skill_enabled("skill-creator", False, home = home)
    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill"]
    assert "create_skill" not in inference_routes._build_tool_action_nudge(
        tools = selected, model_name = "test"
    )
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("skill-creator", home = home)

    skills.set_skill_enabled("guided", False, home = home)
    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert selected == []


def test_read_skill_tool_keeps_pagination_consistent_with_tight_room(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    _write_skill(home, "agents", "paged", body = "x" * 12_000)
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    result = tools_module.execute_tool(
        "read_skill",
        {"name": "paged"},
        context_tokens = 4096,
        result_budget_tokens = 300,
    )

    assert "Resource continues. Call read_skill again" in result
    header = next(line for line in result.splitlines() if line.startswith("Characters:"))
    end = int(header.split("-")[1].split()[0])
    assert f"offset={end}." in result
    assert "truncated to" not in result


# Account scoping: the owner keeps the home folders, a managed account gets its own workspace.

_ALICE_ID = "a" * 32
_BOB_ID = "b" * 32


@pytest.fixture
def managed_accounts(isolated_skills, monkeypatch):
    from utils.account_context import AccountContext

    home, studio = isolated_skills
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    monkeypatch.setattr(
        skills, "_BUNDLED_ROOT", ("bundled", Path(skills.__file__).with_name("bundled_skills"))
    )
    monkeypatch.setattr(skills, "workspace_root", lambda: studio / "accounts" / _account_id())
    return home, studio, AccountContext(_ALICE_ID, "alice"), AccountContext(_BOB_ID, "bob")


def _account_id() -> str:
    from utils.account_context import current_account_id
    return current_account_id()


def _account_skill(studio: Path, account_id: str, name: str, description: str) -> None:
    root = studio / "accounts" / account_id / "skills" / name
    root.mkdir(parents = True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody", encoding = "utf-8"
    )


def test_managed_account_never_sees_the_owners_home_skills(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "owner-only", description = "OWNER_PRIVATE")
    _write_skill(home, "claude", "owner-claude", description = "OWNER_PRIVATE")
    _account_skill(studio, _ALICE_ID, "alice-skill", "ALICE_PRIVATE")

    owner_names = [record["name"] for record in skills.list_skills()]
    assert owner_names == ["owner-only", "owner-claude", "skill-creator"]

    alice_records = run_as(alice, skills.list_skills)
    assert [(r["name"], r["source"]) for r in alice_records] == [
        ("alice-skill", "agents"),
        ("skill-creator", "bundled"),
    ]
    assert run_as(bob, skills.list_skills)[0]["name"] == "skill-creator"
    assert "ALICE_PRIVATE" not in str(run_as(bob, skills.list_skills))

    with pytest.raises(skills.SkillNotFoundError):
        run_as(alice, skills.read_skill_resource, "owner-only")
    with pytest.raises(skills.SkillNotFoundError):
        run_as(bob, skills.read_skill_resource, "alice-skill")
    assert "Body" in run_as(alice, skills.read_skill_resource, "alice-skill")


def test_managed_account_overrides_and_creation_stay_in_its_workspace(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "shared-name", description = "owner copy")
    _account_skill(studio, _ALICE_ID, "shared-name", "alice copy")

    # Alice disabling her copy leaves the owner's enabled and writes only her overrides file.
    assert run_as(alice, skills.set_skill_enabled, "shared-name", False)["enabled"] is False
    assert (studio / "accounts" / _ALICE_ID / "skill-overrides.json").is_file()
    assert not (studio / "skill-overrides.json").exists()
    assert next(r for r in skills.list_skills() if r["name"] == "shared-name")["enabled"] is True
    assert run_as(bob, skills.enabled_skills) == []

    # Bob toggling a skill he cannot see is a not-found, not a write into Alice's file.
    with pytest.raises(skills.SkillNotFoundError):
        run_as(bob, skills.set_skill_enabled, "shared-name", False)
    assert not (studio / "accounts" / _BOB_ID / "skill-overrides.json").exists()

    # create_skill lands in the caller's own workspace, never in the owner's home.
    record = run_as(bob, skills.create_skill, "bob-made", "Bob's skill", "Instructions")
    assert record["path"] == "skills/bob-made/SKILL.md"
    assert (studio / "accounts" / _BOB_ID / "skills" / "bob-made" / "SKILL.md").is_file()
    assert not (home / ".agents" / "skills" / "bob-made").exists()
    assert [r["name"] for r in run_as(bob, skills.enabled_skills)] == ["bob-made"]
    assert "bob-made" not in [r["name"] for r in skills.list_skills()]

    owner_record = skills.create_skill("owner-made", "Owner's skill", "Instructions")
    assert owner_record["path"] == "~/.agents/skills/owner-made/SKILL.md"
    assert (home / ".agents" / "skills" / "owner-made" / "SKILL.md").is_file()


def test_inference_catalog_cache_is_per_account(managed_accounts, monkeypatch):
    import asyncio

    from routes import inference as inference_routes
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "owner-only", description = "OWNER_PRIVATE")
    _account_skill(studio, _ALICE_ID, "alice-skill", "ALICE_PRIVATE")
    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {})

    assert [s["name"] for s in inference_routes._enabled_agent_skills() if s["name"] != "skill-creator"] == ["owner-only"]
    assert [s["name"] for s in run_as(alice, inference_routes._enabled_agent_skills)] == [
        "alice-skill"
    ]
    assert run_as(bob, inference_routes._enabled_agent_skills) == []
    assert set(inference_routes._AGENT_SKILLS_CACHE) == {None, _ALICE_ID, _BOB_ID}

    # The catalog a request is built from follows the acting account.
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        model = "test", messages = [{"role": "user", "content": "hi"}], enable_tools = True
    )

    def select():
        return asyncio.run(
            inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
        )

    assert "read_skill" in [t["function"]["name"] for t in run_as(alice, select)]
    assert [
        t["function"]["name"] for t in run_as(bob, select) if "skill" in t["function"]["name"]
    ] == []
    nudge = run_as(
        alice,
        lambda: inference_routes._build_tool_action_nudge(
            tools = run_as(alice, select), model_name = "test"
        ),
    )
    assert "ALICE_PRIVATE" in nudge and "OWNER_PRIVATE" not in nudge

    inference_routes._invalidate_agent_skills_cache()
    assert inference_routes._AGENT_SKILLS_CACHE == {}


def test_read_skill_page_floor_reports_no_room_instead_of_slivers(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    _write_skill(home, "agents", "long", body = "x" * 12_000)
    # Whatever the room, a page smaller than the floor is not worth a round trip.
    monkeypatch.setattr(tools_module, "_fit_result_to_room", lambda result, name: result[:40])
    result = tools_module.execute_tool("read_skill", {"name": "long"})
    assert result.startswith("Error: Not enough context room")


def test_read_resource_rejects_ancestor_swapped_after_selection(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    (root / "guide.md").write_text("safe", encoding = "utf-8")
    skills_root = root.parent
    outside = home / "outside"
    _write_skill(outside, "agents", "reader")
    (outside / ".agents" / "skills" / "reader" / "guide.md").write_text("secret", encoding = "utf-8")
    original_selected_skill = skills._selected_skill

    def replacing_selected_skill(name, *, home = None):
        selection = original_selected_skill(name, home = home)
        # The skill directory itself stays a real directory; only its parent is swapped.
        skills_root.rename(home / "original-skills")
        try:
            skills_root.symlink_to(outside / ".agents" / "skills", target_is_directory = True)
        except (OSError, NotImplementedError):
            # Reason: Windows may deny symlink creation without Developer Mode.
            pytest.skip("symlinks are unavailable on this platform")
        return selection

    monkeypatch.setattr(skills, "_selected_skill", replacing_selected_skill)
    with pytest.raises(skills.SkillError, match = "changed after it was selected"):
        skills.read_skill_resource("reader", "guide.md", home = home)


@pytest.mark.skipif(
    os.name == "nt",
    reason = "Windows sharing semantics refuse to replace a manifest that is still open for writing",
)
def test_failed_create_keeps_a_manifest_another_writer_replaced(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    manifest = home / ".agents" / "skills" / "racer" / "SKILL.md"
    original_fsync = os.fsync

    def replacing_fsync(descriptor):
        original_fsync(descriptor)
        replacement = manifest.with_name("SKILL.md.new")
        replacement.write_text(
            "---\nname: racer\ndescription: Theirs.\n---\nTHEIRS", encoding = "utf-8"
        )
        os.replace(replacement, manifest)

    monkeypatch.setattr(os, "fsync", replacing_fsync)
    with pytest.raises(skills.SkillError, match = "changed while the manifest was being written"):
        skills.create_skill("racer", "Mine.", "MINE", home = home)

    assert manifest.read_text(encoding = "utf-8").endswith("THEIRS")


def test_unreadable_root_reports_itself_without_hiding_other_roots(isolated_skills):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("permission bits are not enforced for this user")
    home, _ = isolated_skills
    _write_skill(home, "claude", "visible")
    unreadable = home / ".agents" / "skills"
    unreadable.mkdir(parents = True)
    unreadable.chmod(0)
    try:
        records = skills.list_skills(home = home)
    finally:
        unreadable.chmod(0o700)

    assert next(item for item in records if item["name"] == "visible")["valid"] is True
    failed = next(item for item in records if item["source"] == "agents")
    assert failed["valid"] is False
    assert "scan" in failed["error"]


def test_root_entry_limit_ignores_hidden_and_regular_files(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "counted").parent
    for index in range(skills.MAX_SKILLS_PER_ROOT):
        (root / f".hidden-{index}").write_text("", encoding = "utf-8")
    (root / "README.md").write_text("about these skills", encoding = "utf-8")

    records = skills.list_skills(home = home)

    assert [item["name"] for item in records] == ["counted"]


def test_corrupt_overrides_are_ignored_and_repaired_by_the_next_toggle(isolated_skills):
    home, studio = isolated_skills
    _write_skill(home, "agents", "sturdy")
    studio.mkdir()
    (studio / "skill-overrides.json").write_text("{not json", encoding = "utf-8")

    assert next(item for item in skills.list_skills(home = home) if item["name"] == "sturdy")[
        "enabled"
    ]
    skills.set_skill_enabled("sturdy", False, home = home)

    assert json.loads((studio / "skill-overrides.json").read_text(encoding = "utf-8")) == {
        "sturdy": False
    }
    (studio / "skill-overrides.json").write_text(
        '{"sturdy": "no", "Bad Name": false, "other": true}', encoding = "utf-8"
    )
    assert next(item for item in skills.list_skills(home = home) if item["name"] == "sturdy")[
        "enabled"
    ]


def test_indented_separator_inside_a_block_scalar_stays_in_the_frontmatter(isolated_skills):
    home, _ = isolated_skills
    _write_skill(
        home,
        "agents",
        "divided",
        description = "|\n  Use for reports.\n  ---\n  Also for summaries.",
    )

    record = next(item for item in skills.list_skills(home = home) if item["name"] == "divided")

    assert record["valid"] is True
    assert record["description"] == "Use for reports.\n---\nAlso for summaries."


def test_skill_preflight_matches_existing_skills_and_tells_the_model_to_use_them():
    skills_list = [
        {
            "name": "json-checks",
            "description": "Verify JSON contracts against fixtures",
            "enabled": True,
            "valid": True,
        },
        {
            "name": "release-notes",
            "description": "Draft GitHub release notes",
            "enabled": True,
            "valid": True,
        },
    ]
    matched = skills.match_skills("please verify the json contract", skills_list)
    assert [item["name"] for item in matched] == ["json-checks"]
    block = skills.skill_preflight_instruction(
        skills_list,
        user_text = "please verify the json contract",
        can_create = True,
        catalog = "- json-checks: Verify JSON contracts against fixtures",
    )
    assert block.startswith("<skill_preflight>")
    assert "json-checks" in block
    assert "read_skill" in block
    assert "create_skill" in block
    assert "Do not re-invent" in block


def test_temporary_skills_are_thread_scoped_and_promotable(isolated_skills, tmp_path, monkeypatch):
    home, _ = isolated_skills
    monkeypatch.setattr(skills, "_temp_skills_root", lambda thread_id: tmp_path / thread_id)
    record = skills.create_skill(
        "session-json",
        "Check JSON this session",
        "Parse then compare.",
        home = home,
        temporary = True,
        thread_id = "thread-a",
    )
    assert record["temporary"] is True
    temps = skills.list_temp_skills("thread-a")
    assert temps[0]["name"] == "session-json"
    assert skills.list_temp_skills("thread-b") == []
    promoted = skills.promote_temp_skill("session-json", thread_id = "thread-a", home = home)
    assert promoted["temporary"] is False
    assert (home / ".agents" / "skills" / "session-json" / "SKILL.md").is_file()


def test_temporary_skill_discard_is_thread_scoped(isolated_skills, tmp_path, monkeypatch):
    home, _ = isolated_skills
    monkeypatch.setattr(skills, "_temp_skills_root", lambda thread_id: tmp_path / thread_id)
    skills.create_skill(
        "session-json",
        "Check JSON this session",
        "Parse then compare.",
        home = home,
        temporary = True,
        thread_id = "thread-a",
    )

    with pytest.raises(skills.SkillNotFoundError, match = "was not found"):
        skills.discard_temp_skill("session-json", thread_id = "thread-b")

    assert [item["name"] for item in skills.list_temp_skills("thread-a")] == ["session-json"]
    discarded = skills.discard_temp_skill("session-json", thread_id = "thread-a")
    assert discarded == {
        "name": "session-json",
        "temporary": True,
        "thread_id": "thread-a",
        "discarded": True,
    }
    assert skills.list_temp_skills("thread-a") == []


def test_temporary_skill_discard_refuses_symlink(isolated_skills, tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "_temp_skills_root", lambda thread_id: tmp_path / thread_id)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "SKILL.md"
    marker.write_text("outside", encoding = "utf-8")
    root = tmp_path / "thread-a"
    root.mkdir()
    (root / "session-json").symlink_to(outside, target_is_directory = True)

    with pytest.raises(skills.SkillError, match = "missing or unsafe"):
        skills.discard_temp_skill("session-json", thread_id = "thread-a")

    assert marker.read_text(encoding = "utf-8") == "outside"


def test_temporary_skill_thread_root_does_not_alias_sanitized_ids(isolated_skills, tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "account_path", lambda relative: tmp_path / relative)

    underscored = skills._temp_skills_root("thread_a")
    slashed = skills._temp_skills_root("thread/a")

    assert underscored != slashed
    assert underscored.name == "thread_a"
    assert slashed.name.startswith("thread_a-")


def test_read_skill_tool_applies_defaults_for_null_arguments(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    _write_skill(home, "agents", "nullable", body = "Body text")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    result = tools_module.execute_tool(
        "read_skill", {"name": "nullable", "resource": None, "offset": None}
    )

    assert "Body text" in result


def test_reserved_device_name_resource_gets_its_own_message(isolated_skills):
    home, _ = isolated_skills
    _write_skill(home, "agents", "reserved")

    with pytest.raises(skills.SkillError, match = "reserved device name"):
        skills.read_skill_resource("reserved", "con.md", home = home)


def test_aliased_metadata_cannot_expand_past_the_manifest_limit(isolated_skills):
    home, _ = isolated_skills
    big = "x" * (4 * 1024)
    aliases = "\n".join(f"  k{i}: *big" for i in range(8))
    _write_skill(
        home,
        "agents",
        "aliased",
        frontmatter = f"big: &big {big}\nmetadata:\n{aliases}",
    )

    record = next(r for r in skills.list_skills(home = home) if r["name"] == "aliased")

    assert record["valid"] is False and "16 KB" in record["error"]


@pytest.mark.parametrize("field", ["allowed-tools", "license"])
def test_oversized_scalar_fields_are_rejected(isolated_skills, field):
    home, _ = isolated_skills
    _write_skill(home, "agents", "wide", frontmatter = f"{field}: {'x' * 2000}")

    record = next(r for r in skills.list_skills(home = home) if r["name"] == "wide")

    assert record["valid"] is False and "1024" in record["error"]
