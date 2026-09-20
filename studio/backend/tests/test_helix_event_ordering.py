# SPDX-License-Identifier: AGPL-3.0-only

from core.helix_engine.capture import (
    capture_session_key,
    clear_session,
    record_tool_control_event,
    record_tool_execution,
    session_steps,
)
from core.helix_engine.schemas import ToolControlEvent
from core.helix_engine.trajectory import ToolStep
from core.inference.computer_browse import feed_session_id, list_live_feed, record_live_feed
from routes.helix_engine import live_feed, session_trace


def test_ordering_fields_preserve_legacy_constructors():
    step = ToolStep(name="read_file", arguments="a.py", result="ok")
    control = ToolControlEvent(action="noop", tool_name="read_file")

    assert step.sequence == 0
    assert step.created_at_ms == 0
    assert control.sequence == 0
    assert control.created_at_ms == 0


def test_feed_scope_matches_capture_and_preserves_historical_ids():
    assert feed_session_id("same", "same") == "same"
    assert feed_session_id("session-only", None) == "session-only"
    assert feed_session_id(None, "thread-only") == "thread-only"
    assert feed_session_id(None, None) == "default"
    assert feed_session_id("project", "thread-a") == "project::thread::thread-a"
    assert capture_session_key("project", "thread-a") == "project::thread::thread-a"
    # Idempotent for callers that already resolved the shared project/thread scope.
    assert (
        feed_session_id("project::thread::thread-a", "thread-a")
        == "project::thread::thread-a"
    )


def test_capture_control_and_live_feed_share_one_monotonic_thread_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_ENGINE_LEDGER_ROOT", str(tmp_path))
    session_id = "ordering-project"
    thread_id = "ordering-thread"
    turn_id = "ordering-turn"
    exact_key = capture_session_key(session_id, thread_id, turn_id)
    feed_key = feed_session_id(session_id, thread_id)
    clear_session(exact_key)

    step = record_tool_execution(exact_key, "search_memory", {"query": "x"}, "memory hit")
    feed_event = record_live_feed(
        feed_key,
        {"kind": "computer", "action": "screenshot"},
        turn_id=turn_id,
    )
    control = record_tool_control_event(
        exact_key,
        action="skip",
        tool_name="search_memory",
        arguments={"query": "x"},
        reason="duplicate",
    )

    assert step.sequence < feed_event["sequence"] < control.sequence
    assert 0 < step.created_at_ms <= feed_event["created_at_ms"] <= control.created_at_ms
    assert feed_event["turn_id"] == turn_id

    trace = session_trace(session_id, thread_id=thread_id, turn_id=turn_id)
    assert trace["steps"][0]["sequence"] == step.sequence
    assert trace["steps"][0]["created_at_ms"] == step.created_at_ms
    assert trace["control_events"][0]["sequence"] == control.sequence
    assert trace["control_events"][0]["created_at_ms"] == control.created_at_ms

    feed = live_feed(session_id=session_id, thread_id=thread_id)
    assert feed["session_id"] == feed_key
    assert feed["events"][-1]["sequence"] == feed_event["sequence"]
    assert feed["events"][-1]["created_at_ms"] == feed_event["created_at_ms"]
    assert feed["events"][-1]["turn_id"] == turn_id


def test_project_live_feed_does_not_mix_neighbor_threads():
    session_id = "shared-project-workspace"
    first_key = feed_session_id(session_id, "thread-one")
    second_key = feed_session_id(session_id, "thread-two")

    first = record_live_feed(first_key, {"action": "click", "kind": "computer", "x": 1})
    second = record_live_feed(second_key, {"action": "navigate", "kind": "browse", "url": "https://example.com"})

    first_feed = live_feed(session_id=session_id, thread_id="thread-one")
    second_feed = live_feed(session_id=session_id, thread_id="thread-two")

    assert first_feed["session_id"] == first_key
    assert second_feed["session_id"] == second_key
    assert first in first_feed["events"]
    assert second not in first_feed["events"]
    assert second in second_feed["events"]
    assert first not in second_feed["events"]


def test_live_feed_bounds_session_scopes_and_refreshes_active_scope_recency(monkeypatch):
    from core.inference import computer_browse as module

    monkeypatch.setattr(module, "_MAX_FEED_SESSIONS", 2)
    with module._LOCK:
        module._FEED.clear()

    first = record_live_feed("feed-a", {"action": "first", "kind": "computer"})
    record_live_feed("feed-b", {"action": "second", "kind": "computer"})
    refreshed = record_live_feed("feed-a", {"action": "refresh", "kind": "computer"})
    record_live_feed("feed-c", {"action": "third", "kind": "computer"})

    assert first in list(module.list_live_feed("feed-a"))
    assert refreshed in list(module.list_live_feed("feed-a"))
    assert module.list_live_feed("feed-b") == []
    assert module.list_live_feed("feed-c")


def test_alice_cap_exhaustion_does_not_evict_bob_capture_feed_or_sequence(monkeypatch):
    from core.helix_engine import capture
    from core import helix_event_sequence
    from core.inference import computer_browse
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("alice-cap-test", "alice", "user")
    bob = AccountContext("bob-cap-test", "bob", "user")
    monkeypatch.setattr(capture, "_MAX_CAPTURE_SESSIONS", 2)
    monkeypatch.setattr(computer_browse, "_MAX_FEED_SESSIONS", 2)
    monkeypatch.setattr(helix_event_sequence, "_MAX_EVENT_SCOPES", 2)
    with capture._LOCK:
        capture._STEPS.clear()
        capture._CONTROL_EVENTS.clear()
        capture._RECENT_COMPLETED.clear()
        capture._RECENT_CONTROL_EVENTS.clear()
        capture._RECENT_TURN_IDS.clear()
    with computer_browse._LOCK:
        computer_browse._FEED.clear()
    with helix_event_sequence._LOCK:
        helix_event_sequence._CLOCKS.clear()

    bob_session = "bob-cap-session"
    bob_feed_key = run_as(bob, feed_session_id, bob_session, None)
    bob_step = run_as(
        bob,
        record_tool_execution,
        bob_session,
        "read_file",
        {"path": "bob.txt"},
        "bob-capture",
    )
    bob_feed_event = run_as(
        bob,
        record_live_feed,
        bob_feed_key,
        {"action": "snapshot", "kind": "computer", "marker": "bob-feed"},
    )

    for index in range(8):
        run_as(
            alice,
            record_tool_execution,
            f"alice-cap-{index}",
            "read_file",
            {"path": f"alice-{index}.txt"},
            "alice-capture",
        )
        alice_feed_key = run_as(alice, feed_session_id, f"alice-feed-{index}", None)
        run_as(
            alice,
            record_live_feed,
            alice_feed_key,
            {"action": "snapshot", "kind": "computer", "marker": f"alice-{index}"},
        )

    assert run_as(bob, session_steps, bob_session) == [bob_step]
    assert bob_feed_event in run_as(bob, list_live_feed, bob_feed_key)
    bob_next = run_as(
        bob,
        record_tool_execution,
        bob_session,
        "read_file",
        {"path": "bob-next.txt"},
        "bob-still-live",
    )
    assert bob_next.sequence > bob_step.sequence
    assert run_as(bob, session_steps, bob_session)[-1].result == "bob-still-live"


def test_global_caps_refuse_new_account_restore_and_purge_only_the_retired_account(monkeypatch):
    from core import helix_event_sequence
    from core.helix_engine import capture
    from core.inference import computer_browse
    from core.helix_engine.capture import capture_snapshot, restore_capture_snapshot
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("alice-global-cap", "alice", "user")
    bob = AccountContext("bob-global-cap", "bob", "user")
    carol = AccountContext("carol-global-cap", "carol", "user")
    monkeypatch.setattr(capture, "_MAX_CAPTURE_SESSIONS", 10)
    monkeypatch.setattr(capture, "_MAX_CAPTURE_GLOBAL_SCOPES", 2)
    monkeypatch.setattr(computer_browse, "_MAX_FEED_SESSIONS", 10)
    monkeypatch.setattr(computer_browse, "_MAX_FEED_GLOBAL_SCOPES", 2)
    monkeypatch.setattr(helix_event_sequence, "_MAX_EVENT_SCOPES", 10)
    monkeypatch.setattr(helix_event_sequence, "_MAX_EVENT_GLOBAL_SCOPES", 2)
    with capture._LOCK:
        for store in (
            capture._STEPS,
            capture._CONTROL_EVENTS,
            capture._RECENT_COMPLETED,
            capture._RECENT_CONTROL_EVENTS,
            capture._RECENT_TURN_IDS,
        ):
            store.clear()
    with computer_browse._LOCK:
        computer_browse._FEED.clear()
    with helix_event_sequence._LOCK:
        helix_event_sequence._CLOCKS.clear()

    bob_step = run_as(
        bob,
        record_tool_execution,
        "bob-global",
        "read_file",
        {"path": "bob.txt"},
        "bob",
    )
    bob_feed = run_as(
        bob,
        record_live_feed,
        "bob-global",
        {"action": "snapshot", "marker": "bob"},
    )
    snapshot = run_as(bob, capture_snapshot, "bob-global")
    run_as(alice, record_tool_execution, "alice-global", "read_file", {}, "alice")
    run_as(alice, record_live_feed, "alice-global", {"action": "snapshot", "marker": "alice"})
    run_as(alice, helix_event_sequence.next_event_metadata, "alice-global")

    # A brand-new account cannot displace either retained account globally.
    run_as(carol, record_tool_execution, "carol-global", "read_file", {}, "carol")
    run_as(carol, record_live_feed, "carol-global", {"action": "snapshot", "marker": "carol"})
    run_as(carol, helix_event_sequence.next_event_metadata, "carol-global")
    assert run_as(bob, session_steps, "bob-global") == [bob_step]
    assert bob_feed in run_as(bob, list_live_feed, "bob-global")
    assert run_as(carol, session_steps, "carol-global") == []
    assert run_as(carol, list_live_feed, "carol-global") == []

    # Restore is quota-aware too: Carol's flood cannot create a new capture key.
    assert run_as(carol, restore_capture_snapshot, "carol-restored", snapshot) is False
    assert run_as(carol, session_steps, "carol-restored") == []

    assert capture.purge_account(alice.account_id) >= 1
    assert computer_browse.purge_account(alice.account_id) >= 1
    assert helix_event_sequence.purge_account(alice.account_id) >= 1
    assert run_as(bob, session_steps, "bob-global") == [bob_step]
    assert bob_feed in run_as(bob, list_live_feed, "bob-global")


def test_deactivate_reactivate_rejects_stale_telemetry_but_accepts_new_epoch():
    from core import helix_event_sequence
    from core.helix_engine import capture
    from core.inference import computer_browse
    from state import active_generations
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("alice-epoch-test", "alice", "user")
    bob = AccountContext("bob-epoch-test", "bob", "user")
    with capture._LOCK:
        for store in (capture._STEPS, capture._CONTROL_EVENTS):
            store.clear()
    with computer_browse._LOCK:
        computer_browse._FEED.clear()
    with helix_event_sequence._LOCK:
        helix_event_sequence._CLOCKS.clear()

    old_epoch = run_as(alice, active_generations.bound_lifecycle_epoch)
    run_as(alice, active_generations.fence, alice.account_id)
    run_as(alice, active_generations.lift_fence, alice.account_id)

    run_as(
        alice,
        record_tool_execution,
        "stale-epoch",
        "read_file",
        {},
        "stale",
        lifecycle_epoch=old_epoch,
    )
    run_as(
        alice,
        record_live_feed,
        "stale-epoch",
        {"action": "stale"},
        lifecycle_epoch=old_epoch,
    )
    run_as(alice, helix_event_sequence.next_event_metadata, "stale-epoch", lifecycle_epoch=old_epoch)
    assert run_as(alice, session_steps, "stale-epoch") == []
    assert run_as(alice, list_live_feed, "stale-epoch") == []

    # A callback in the newly active lifecycle is allowed, and another account
    # remains unaffected by Alice's fence transitions.
    new_step = run_as(
        alice,
        record_tool_execution,
        "new-epoch",
        "read_file",
        {},
        "new",
    )
    bob_step = run_as(bob, record_tool_execution, "bob-epoch", "read_file", {}, "bob")
    assert new_step.result == "new"
    assert run_as(alice, session_steps, "new-epoch") == [new_step]
    assert run_as(bob, session_steps, "bob-epoch") == [bob_step]
