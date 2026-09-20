# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small, explicit macOS computer-use primitive for the local agent.

This intentionally is not a general shell bridge. The model receives a fixed
action vocabulary, coordinates are bounded integers, and every action other
than ``screenshot`` is classified as approval-required by the tool policy.
Accessibility permission is owned by macOS and failures are returned as useful
model-readable errors rather than being hidden.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utils.paths import account_path, ensure_dir

_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_APP_RE = re.compile(r"^[^\x00\r\n]{1,160}$")
_KEYS = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
    "pageup": 116,
    "pagedown": 121,
}
_MODIFIER_CODES = {"command": 55, "cmd": 55, "shift": 56, "option": 58, "alt": 58, "control": 59, "ctrl": 59}


def _session_dir(session_id: str | None) -> Path:
    token = _SESSION_RE.sub("_", str(session_id or "default"))[:100] or "default"
    destination = account_path("computer", token)
    ensure_dir(destination)
    return destination


def _run_osascript(script: str, *argv: str) -> str:
    command = ["/usr/bin/osascript", "-e", script, "--", *argv]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Error: macOS automation could not start: {error}"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if result.returncode:
        detail = error or output or f"osascript exited with status {result.returncode}"
        if "not allowed assistive access" in detail.lower() or "accessibility" in detail.lower():
            return (
                "Error: macOS Accessibility permission is required. Enable Helix Harness in "
                "System Settings → Privacy & Security → Accessibility, then retry."
            )
        return f"Error: macOS automation failed: {detail[:1_500]}"
    return output


def _coordinate(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 20_000 else None


def _screenshot(session_id: str | None) -> str:
    if sys.platform != "darwin":
        return "Error: computer use is currently supported on macOS only."
    destination = _session_dir(session_id) / f"screen-{int(time.time() * 1_000)}.png"
    try:
        result = subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-t", "png", str(destination)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Error: screenshot could not be captured: {error}"
    if result.returncode or not destination.is_file():
        detail = (result.stderr or result.stdout or "screencapture failed").strip()
        return f"Error: screenshot could not be captured: {detail[:1_000]}"
    return json.dumps(
        {
            "action": "screenshot",
            "path": str(destination),
            "note": "The screenshot is available to the local app; do not infer unseen UI state from the path alone.",
        },
        ensure_ascii=False,
    )


def _feed(
    session_id: str | None,
    event: dict[str, Any],
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> None:
    try:
        from .computer_browse import feed_session_id, record_live_feed

        record_live_feed(
            feed_session_id(session_id, thread_id),
            event,
            turn_id=turn_id,
        )
    except Exception:
        pass


def use_computer(
    arguments: dict[str, Any],
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Execute one bounded computer action and return JSON/text for the model."""
    action = str(arguments.get("action") or "").strip().lower()
    if action in {"browse", "navigate"}:
        from .computer_browse import browse_action, feed_session_id

        event = browse_action(
            "navigate",
            url=str(arguments.get("url") or ""),
            session_id=feed_session_id(session_id, thread_id),
            turn_id=turn_id,
        )
        return json.dumps(event, ensure_ascii=False)
    if sys.platform != "darwin":
        return "Error: computer use is currently supported on macOS only (browse works everywhere)."
    if action == "screenshot":
        result = _screenshot(session_id)
        _feed(session_id, {"action": "screenshot", "kind": "computer"}, thread_id, turn_id)
        return result
    if action == "click":
        x, y = _coordinate(arguments, "x"), _coordinate(arguments, "y")
        if x is None or y is None:
            return "Error: click requires integer x and y coordinates between 0 and 20000."
        result = _run_osascript(
            "tell application \"System Events\" to click at {" + str(x) + ", " + str(y) + "}",
        )
        _feed(session_id, {"action": "click", "kind": "computer", "x": x, "y": y}, thread_id, turn_id)
        return result or json.dumps({"action": action, "x": x, "y": y})
    if action == "type":
        text = arguments.get("text")
        if not isinstance(text, str) or not text or len(text) > 8_000:
            return "Error: type requires 1-8000 characters of text."
        result = _run_osascript(
            "on run argv\n"
            "tell application \"System Events\" to keystroke (item 1 of argv)\n"
            "end run",
            text,
        )
        _feed(session_id, {"action": "type", "kind": "computer", "characters": len(text)}, thread_id, turn_id)
        return result or json.dumps({"action": action, "characters": len(text)})
    if action == "key":
        raw_key = str(arguments.get("key") or "").strip().lower()
        if raw_key not in _KEYS:
            return f"Error: unsupported key '{raw_key}'. Use one of: {', '.join(sorted(_KEYS))}."
        modifier_values = arguments.get("modifiers") or []
        if not isinstance(modifier_values, list) or any(str(value).lower() not in _MODIFIER_CODES for value in modifier_values):
            return "Error: modifiers must be a list containing command, shift, option, or control."
        codes = [_MODIFIER_CODES[str(value).lower()] for value in modifier_values]
        key_code = _KEYS[raw_key]
        down = "\n".join(f"key down {code}" for code in codes)
        up = "\n".join(f"key up {code}" for code in reversed(codes))
        script = f"tell application \"System Events\"\n{down}\nkey code {key_code}\n{up}\nend tell"
        result = _run_osascript(script)
        _feed(session_id, {"action": "key", "kind": "computer", "key": raw_key}, thread_id, turn_id)
        return result or json.dumps({"action": action, "key": raw_key, "modifiers": modifier_values})
    if action == "scroll":
        try:
            amount = int(arguments.get("amount"))
        except (TypeError, ValueError):
            return "Error: scroll requires an integer amount between -20 and 20."
        if amount < -20 or amount > 20 or amount == 0:
            return "Error: scroll amount must be a non-zero integer between -20 and 20."
        key_code = 121 if amount > 0 else 116
        script = "tell application \"System Events\" to key code " + str(key_code)
        result = "\n".join(_run_osascript(script) for _ in range(min(abs(amount), 20)))
        _feed(session_id, {"action": "scroll", "kind": "computer", "amount": amount}, thread_id, turn_id)
        return result or json.dumps({"action": action, "amount": amount})
    if action == "open_app":
        app = arguments.get("app")
        if not isinstance(app, str) or not _APP_RE.fullmatch(app.strip()):
            return "Error: open_app requires a non-empty macOS application name."
        try:
            result = subprocess.run(
                ["/usr/bin/open", "-a", app.strip()],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Error: application could not be opened: {error}"
        if result.returncode:
            return f"Error: application could not be opened: {(result.stderr or result.stdout).strip()[:1_000]}"
        _feed(session_id, {"action": "open_app", "kind": "computer", "app": app.strip()}, thread_id, turn_id)
        return json.dumps({"action": action, "app": app.strip()})
    return "Error: computer action must be screenshot, click, type, key, scroll, open_app, or browse."
