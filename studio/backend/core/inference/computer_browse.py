# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium-acceptable browse actions plus a session-scoped live feed."""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from core.helix_event_sequence import event_scope_id, next_event_metadata
from state import active_generations
from utils.account_context import current_account_id

_LOCK = threading.Lock()
_StoreKey = tuple[str, str]
_FEED: dict[_StoreKey, list[dict[str, Any]]] = {}
_MAX_FEED_SESSIONS = 512
_MAX_FEED_GLOBAL_SCOPES = 4096
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _store_key(session_id: str | None) -> _StoreKey:
    return current_account_id(), str(session_id or "default")


def _evict_account_feed(account_id: str) -> None:
    owned = sum(1 for key in _FEED if key[0] == account_id)
    if owned < _MAX_FEED_SESSIONS:
        return
    for key in _FEED:
        if key[0] == account_id:
            _FEED.pop(key, None)
            return


def _reserve_feed_scope(key: _StoreKey) -> bool:
    """Reserve a feed scope without evicting another account's scope."""
    if key in _FEED:
        return True
    owned = [existing for existing in _FEED if existing[0] == key[0]]
    if len(owned) >= _MAX_FEED_SESSIONS:
        _FEED.pop(owned[0], None)
    elif len(_FEED) >= _MAX_FEED_GLOBAL_SCOPES:
        if not owned:
            return False
        _FEED.pop(owned[0], None)
    return True


def purge_account(account_id: str) -> int:
    """Drop all process-local browser feed state owned by a retired account."""
    account = str(account_id)
    removed = 0
    with _LOCK:
        for key in tuple(_FEED):
            if key[0] == account:
                _FEED.pop(key, None)
                removed += 1
    return removed


def feed_session_id(session_id: str | None = None, thread_id: str | None = None) -> str:
    """Match Helix capture scoping without changing historical equal/missing ids."""
    return event_scope_id(session_id, thread_id)


def record_live_feed(
    session_id: str,
    event: dict[str, Any],
    *,
    turn_id: str | None = None,
    lifecycle_epoch: int | None = None,
) -> dict[str, Any]:
    key = str(session_id or "default")
    store_key = _store_key(key)
    account = current_account_id()
    bound_epoch = active_generations.bound_lifecycle_epoch(account) if lifecycle_epoch is None else lifecycle_epoch
    ordering = next_event_metadata(key, lifecycle_epoch=bound_epoch)
    # Scope and ordering are backend-owned provenance; a caller-supplied event must
    # not be able to relabel itself into another session or forge sequence metadata.
    payload = {**event, "session_id": key, **ordering}
    logical_turn = str(turn_id or "").strip()
    if logical_turn:
        payload["turn_id"] = logical_turn[:200]
    with active_generations.telemetry_guard(account, bound_epoch) as allowed:
        with _LOCK:
            if not allowed or not _reserve_feed_scope(store_key):
                # Optional telemetry is fail-open.  The caller still receives the
                # backend-owned event, but a fenced/stale callback and a new account
                # at the global bound cannot displace retained state.
                return payload
            # Reinsert so dict order tracks recently active scopes; otherwise a very
            # old but currently busy thread can be evicted merely because it was the
            # first scope ever inserted.
            prior = _FEED.pop(store_key, [])
            prior.append(payload)
            _FEED[store_key] = prior[-200:]
    return payload


def list_live_feed(session_id: str) -> list[dict[str, Any]]:
    with _LOCK:
        return list(_FEED.get(_store_key(session_id), []))


def _fetch_page(url: str, timeout: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "HelixHarness/2.1 "
                "(+https://github.com/SuperHelix77/Helix-Harness-v2.1)"
            )
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(120_000)
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            final_url = str(response.geturl())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        return {"ok": False, "error": str(error)[:500], "url": url, "engine": "urllib"}
    title_match = _TITLE_RE.search(html)
    title = _TAG_RE.sub("", title_match.group(1)).strip() if title_match else ""
    snippet = " ".join(_TAG_RE.sub(" ", html).split())[:800]
    return {
        "ok": True,
        "url": final_url,
        "title": title[:200],
        "snippet": snippet,
        "engine": "urllib",
    }


def browse_action(
    action: str,
    *,
    url: str | None = None,
    session_id: str = "default",
    turn_id: str | None = None,
) -> dict[str, Any]:
    name = str(action or "").strip().lower()
    if name not in {"navigate", "browse", "snapshot", "back"}:
        return {"ok": False, "error": f"unsupported browse action '{action}'"}
    parsed = urlparse(str(url or ""))
    if name in {"navigate", "browse"}:
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "error": "browse requires an http(s) URL"}
        page = _fetch_page(str(url))
        event = {
            "action": "navigate",
            "url": page.get("url") or url,
            "kind": "browse",
            "ok": bool(page.get("ok")),
            "title": page.get("title") or "",
            "snippet": page.get("snippet") or page.get("error") or "",
            "engine": page.get("engine") or "urllib",
        }
        return record_live_feed(session_id, event, turn_id=turn_id)
    event = {
        "action": name,
        "url": url,
        "kind": "browse",
        "ok": True,
    }
    return record_live_feed(session_id, event, turn_id=turn_id)


def map_computer_action(arguments: dict[str, Any]) -> dict[str, Any]:
    action = str(arguments.get("action") or "").strip().lower()
    if action in {"browse", "navigate"}:
        return {
            "kind": "browse",
            "action": "browse",
            "url": arguments.get("url"),
        }
    return {
        "kind": "computer",
        "action": action or "screenshot",
        "x": arguments.get("x"),
        "y": arguments.get("y"),
    }


def feed_json(session_id: str) -> str:
    return json.dumps({"session_id": session_id, "events": list_live_feed(session_id)})
