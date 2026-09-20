# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
from contextlib import closing

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from auth import policy
from storage import studio_db
from utils.account_context import run_as

from .factories import (
    FACTORIES,
    SKIPPED,
    format_path,
    initialize_workspaces,
    seed_resource,
    snapshot_resource,
)
from .inventory import (
    OBJECT_ROUTES,
    ROUTES,
    collect_routes,
    looks_like_object_id,
    render_inventory,
    walk_router,
)
from .support import bearer

ACTORS = ("owner", "right", "wrong", "unauthenticated", "deactivated")


def matrix_parameters():
    for case in OBJECT_ROUTES:
        if case.key in FACTORIES:
            for actor in ACTORS:
                yield pytest.param(case, actor, id = f"{case.key}[{actor}]")


def test_object_route_factory_completeness(capsys):
    uncovered = [
        case.key for case in OBJECT_ROUTES if case.key not in FACTORIES and case.key not in SKIPPED
    ]
    covered = [case.key for case in OBJECT_ROUTES if case.key in FACTORIES]
    skipped = [case.key for case in OBJECT_ROUTES if case.key in SKIPPED]
    with capsys.disabled():
        print(
            f"\nroute isolation matrix: {len(OBJECT_ROUTES)} object routes, "
            f"{len(covered)} covered by factory, {len(skipped)} skipped with reason, "
            f"{len(covered) * len(ACTORS)} actor cases"
        )
    assert not uncovered, uncovered


def test_skipped_routes_are_real_object_routes_with_a_reason():
    keys = {case.key for case in OBJECT_ROUTES}
    assert set(SKIPPED) <= keys, sorted(set(SKIPPED) - keys)
    assert all(reason.strip() for reason in SKIPPED.values())
    assert not set(SKIPPED) & set(FACTORIES)


def test_factories_that_leave_the_default_contract_state_a_reason():
    unexplained = [
        key for key, factory in FACTORIES.items() if factory.deviates and not factory.reason.strip()
    ]
    assert not unexplained, unexplained


@pytest.mark.parametrize("case,actor", list(matrix_parameters()))
def test_object_route_account_matrix(case, actor, request):
    accounts = request.getfixturevalue("accounts")
    auth_db = request.getfixturevalue("isolated_auth")
    factory = FACTORIES[case.key]
    initialize_workspaces(accounts)
    params = seed_resource(factory, accounts["alice"], actor)
    before = snapshot_resource(accounts["alice"])
    username = {"owner": "unsloth", "right": "alice", "wrong": "bob", "deactivated": "alice"}.get(
        actor
    )
    headers = bearer(username) if username else {}
    if actor == "deactivated":
        with closing(sqlite3.connect(auth_db.DB_PATH)) as conn:
            conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
            conn.commit()
        policy.invalidate_account_cache()

    app = FastAPI()
    app.include_router(case.router, prefix = "/matrix")
    with TestClient(app, raise_server_exceptions = False) as client:
        response = client.request(
            case.method,
            "/matrix" + format_path(case.path, params),
            headers = headers,
            params = factory.query,
            json = factory.body,
        )
    assert response.status_code in factory.expected(actor), (
        case.key,
        actor,
        response.status_code,
        response.text,
    )
    if actor == "right":
        if factory.fragment:
            assert factory.fragment in response.text
    else:
        assert snapshot_resource(accounts["alice"]) == before
        if factory.absent:
            assert factory.absent not in response.text, response.text
        if factory.name == "api-key":
            assert len(auth_db.list_api_keys("alice")) == 1


@pytest.mark.parametrize(
    "case", [case for case in OBJECT_ROUTES if case.key in FACTORIES], ids = lambda case: case.key
)
def test_owner_can_still_use_own_resource(case, accounts):
    initialize_workspaces(accounts)
    factory = FACTORIES[case.key]
    params = seed_resource(factory, accounts["unsloth"])
    app = FastAPI()
    app.include_router(case.router, prefix = "/matrix")
    with TestClient(app, raise_server_exceptions = False) as client:
        response = client.request(
            case.method,
            "/matrix" + format_path(case.path, params),
            headers = bearer("unsloth"),
            params = factory.query,
            json = factory.body,
        )
    assert response.status_code in (factory.self_expected or (factory.success,)), response.text


def test_first_database_use_in_each_account_initializes_its_schema(accounts):
    for account in accounts.values():
        assert run_as(account, studio_db.list_chat_threads) == []


def test_inventory_contains_hidden_routes_and_no_duplicate_method_paths():
    assert ROUTES == collect_routes()
    assert len({case.key for case in ROUTES}) == len(ROUTES)
    assert "routes.rag:GET:/jobs/{job_id}/events" in {case.key for case in ROUTES}
    assert set(FACTORIES) <= {
        case.key for case in OBJECT_ROUTES
    }, "A registered route disappeared or changed shape"
    generated = {
        (parameter.values[0].key, parameter.values[1]) for parameter in matrix_parameters()
    }
    assert generated == {
        (case.key, actor) for case in OBJECT_ROUTES if case.key in FACTORIES for actor in ACTORS
    }
    report = render_inventory()
    assert all(f"`{case.path}`" in report for case in OBJECT_ROUTES)


def test_nested_router_prefixes_are_collected():
    nested, parent = APIRouter(), APIRouter()

    @nested.get("/{item_id}", include_in_schema = False)
    def get_item(item_id: str):
        return item_id

    parent.include_router(nested, prefix = "/items")
    assert [path for path, _ in walk_router(parent)] == ["/items/{item_id}"]


@pytest.mark.parametrize(
    "name", ["id", "thread_id", "run_id", "job_id", "server_id", "key_id", "filename", "ref"]
)
def test_object_parameter_detection(name):
    assert looks_like_object_id(name)


def test_same_helix_identifiers_and_live_feed_are_account_private(accounts, isolated_auth):
    from core.helix_engine.capture import clear_session, record_tool_execution
    from core.inference.computer_browse import feed_session_id, record_live_feed
    from routes import helix_engine

    session_id = "shared-hostile-session-id"
    storage_keys = {}
    for account, marker in (
        (accounts["alice"], "alice-private-helix"),
        (accounts["bob"], "bob-private-helix"),
    ):
        run_as(account, clear_session, session_id)
        run_as(
            account,
            record_tool_execution,
            session_id,
            "read_file",
            {"path": f"{account.username}.txt"},
            marker,
        )
        feed_key = run_as(account, feed_session_id, session_id, None)
        storage_keys[account.username] = feed_key
        run_as(
            account,
            record_live_feed,
            feed_key,
            {"action": "snapshot", "kind": "computer", "title": marker},
        )

    app = FastAPI()
    app.include_router(helix_engine.router, prefix="/helix")
    alice_headers = bearer("alice")
    with TestClient(app, raise_server_exceptions=False) as client:
        alice_trace = client.get(
            f"/helix/session/{session_id}", headers=alice_headers
        )
        bob_trace = client.get(
            f"/helix/session/{session_id}", headers=bearer("bob")
        )
        owner_trace = client.get(
            f"/helix/session/{session_id}", headers=bearer("unsloth")
        )
        unauth_trace = client.get(f"/helix/session/{session_id}")
        alice_feed = client.get(
            "/helix/live-feed", params={"session_id": session_id}, headers=alice_headers
        )
        bob_feed = client.get(
            "/helix/live-feed", params={"session_id": session_id}, headers=bearer("bob")
        )
        owner_feed = client.get(
            "/helix/live-feed", params={"session_id": session_id}, headers=bearer("unsloth")
        )
        owner_internal_key_probe = client.get(
            "/helix/live-feed",
            params={"session_id": storage_keys["alice"]},
            headers=bearer("unsloth"),
        )
        unauth_feed = client.get("/helix/live-feed", params={"session_id": session_id})
        with closing(sqlite3.connect(isolated_auth.DB_PATH)) as conn:
            conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
            conn.commit()
        policy.invalidate_account_cache()
        deactivated_feed = client.get(
            "/helix/live-feed", params={"session_id": session_id}, headers=alice_headers
        )

    assert alice_trace.status_code == bob_trace.status_code == 200
    assert "alice-private-helix" in alice_trace.text
    assert "bob-private-helix" not in alice_trace.text
    assert "bob-private-helix" in bob_trace.text
    assert "alice-private-helix" not in bob_trace.text
    assert owner_trace.status_code == 404
    assert unauth_trace.status_code in (401, 403)
    assert alice_feed.status_code == bob_feed.status_code == owner_feed.status_code == 200
    assert alice_feed.json()["session_id"] == session_id
    assert alice_feed.json()["events"][0]["session_id"] == session_id
    assert "alice-private-helix" in alice_feed.text
    assert "bob-private-helix" not in alice_feed.text
    assert "bob-private-helix" in bob_feed.text
    assert "alice-private-helix" not in bob_feed.text
    assert owner_feed.json()["events"] == []
    assert owner_internal_key_probe.status_code == 200
    assert owner_internal_key_probe.json()["events"] == []
    assert unauth_feed.status_code in (401, 403)
    assert deactivated_feed.status_code == 401


def test_managed_skill_approval_and_same_name_reads_are_confined(accounts):
    from routes import learning, skills
    from utils.paths import workspace_root

    initialize_workspaces(accounts)
    skill_name = "matrix-approved-skill"
    bob_root = run_as(accounts["bob"], workspace_root) / "skills" / skill_name
    bob_root.mkdir(parents=True)
    (bob_root / "SKILL.md").write_text(
        "---\nname: matrix-approved-skill\ndescription: Bob private skill\n---\n\nBob only.\n",
        encoding="utf-8",
    )
    proposal_id = "alice-managed-skill-approval"
    state = learning._empty_state()
    state["pending"] = [
        {
            "id": proposal_id,
            "kind": "skill",
            "title": "Alice private skill",
            "content": "Alice only approved instructions.",
            "reason": "verified workflow",
            "name": skill_name,
            "target": "codex",
            "sourceThreadId": "thread-a",
            "createdAt": 1000,
            "recommendationAction": "skill",
            "recommendationReason": "repeatable",
        }
    ]
    run_as(accounts["alice"], learning._write_state, state)

    app = FastAPI()
    app.include_router(learning.router, prefix="/learning")
    app.include_router(skills.router, prefix="/skills")
    with TestClient(app, raise_server_exceptions=False) as client:
        approved = client.post(
            f"/learning/proposals/{proposal_id}/approve",
            headers=bearer("alice"),
        )
        alice_read = client.get(f"/skills/{skill_name}", headers=bearer("alice"))
        bob_read = client.get(f"/skills/{skill_name}", headers=bearer("bob"))
        owner_read = client.get(f"/skills/{skill_name}", headers=bearer("unsloth"))

    alice_root = run_as(accounts["alice"], workspace_root) / "skills" / skill_name
    assert approved.status_code == 200, approved.text
    assert (alice_root / "SKILL.md").is_file()
    assert "Alice only approved instructions." in alice_read.text
    assert "Bob only." not in alice_read.text
    assert "Bob only." in bob_read.text
    assert "Alice only approved instructions." not in bob_read.text
    assert owner_read.status_code == 404
