# SPDX-License-Identifier: AGPL-3.0-only

from core.inference import computer_browse
from core.inference import tools
from core.inference.computer_browse import (
    browse_action,
    feed_session_id,
    list_live_feed,
    map_computer_action,
    record_live_feed,
)
from core.inference.computer_use import use_computer
from routes.helix_engine import live_feed


def test_browse_navigate_is_an_explicit_action_and_emits_a_feed_event(monkeypatch):
    monkeypatch.setattr(
        computer_browse,
        "_fetch_page",
        lambda url, timeout=8.0: {
            "ok": True,
            "url": url,
            "title": "Example Domain",
            "snippet": "This domain is for use in documentation examples.",
            "engine": "urllib",
        },
    )
    event = browse_action("navigate", url="https://example.com", session_id="s1")
    assert event["action"] == "navigate"
    assert event["url"] == "https://example.com"
    assert event["title"] == "Example Domain"
    assert event["session_id"] == "s1"
    feed = list_live_feed("s1")
    assert feed[-1]["action"] == "navigate"


def test_computer_action_mapping_includes_browse_and_screenshot_feed():
    mapped = map_computer_action({"action": "browse", "url": "https://example.com"})
    assert mapped["kind"] in {"browse", "computer"}
    assert mapped["action"] == "browse"
    shot = map_computer_action({"action": "screenshot"})
    assert shot["action"] == "screenshot"


def test_live_feed_is_session_scoped():
    record_live_feed("a", {"action": "click", "x": 1})
    record_live_feed("b", {"action": "type", "text": "hi"})
    assert all(item["action"] != "type" for item in list_live_feed("a"))
    assert list_live_feed("b")[-1]["action"] == "type"


def test_use_computer_browse_feed_is_what_the_api_and_gui_read(monkeypatch):
    monkeypatch.setattr(
        computer_browse,
        "_fetch_page",
        lambda url, timeout=8.0: {
            "ok": True,
            "url": url,
            "title": "Example Domain",
            "snippet": "docs",
            "engine": "urllib",
        },
    )
    session = "thread-live-feed"
    raw = use_computer(
        {"action": "browse", "url": "https://example.com/gui"},
        session_id=session,
        thread_id="thread-live",
        turn_id="turn-live",
    )
    assert "example.com/gui" in raw
    scoped_session = feed_session_id(session, "thread-live")
    body = live_feed(session_id=session, thread_id="thread-live")
    assert body["session_id"] == scoped_session
    assert body["events"] == list_live_feed(scoped_session)
    assert body["events"][-1]["url"] == "https://example.com/gui"
    assert body["events"][-1]["action"] == "navigate"
    assert body["events"][-1]["turn_id"] == "turn-live"
    assert all(item.get("url") != "https://example.com/gui" for item in list_live_feed("default"))
    assert feed_session_id(session, "other-thread") == f"{session}::thread::other-thread"


def test_execute_tool_forwards_hidden_helix_turn_id_to_computer_feed(monkeypatch):
    monkeypatch.setattr(
        computer_browse,
        "_fetch_page",
        lambda url, timeout=8.0: {
            "ok": True,
            "url": url,
            "title": "Turn scoped",
            "snippet": "docs",
            "engine": "urllib",
        },
    )
    monkeypatch.setattr("state.tool_policy.require_tool_access", lambda **kwargs: None)

    raw = tools.execute_tool(
        "computer",
        {"action": "browse", "url": "https://example.com/turn"},
        session_id="project-session",
        thread_id="thread-hidden",
        helix_turn_id="turn-hidden",
    )

    assert '"turn_id": "turn-hidden"' in raw
    events = list_live_feed(feed_session_id("project-session", "thread-hidden"))
    assert events[-1]["turn_id"] == "turn-hidden"
