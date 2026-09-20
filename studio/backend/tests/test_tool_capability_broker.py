# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import replace

import pytest

from core.inference import tool_capabilities, tools
from core.inference.tool_capabilities import ToolCapability
from core.inference import tool_confinement
from core.inference.tool_confinement import ToolConfinementUnavailable
from utils.account_context import AccountContext, run_as


def test_python_network_grant_is_positive_not_ambient():
    offline = tool_capabilities.issue_local_execution_grant("python", "print('offline')")
    online = tool_capabilities.issue_local_execution_grant(
        "python", "import requests\nprint('network')"
    )
    dynamic = tool_capabilities.issue_local_execution_grant(
        "python", "name = 'requests'\n__import__(name)"
    )

    assert offline.capabilities == frozenset()
    assert online.capabilities == frozenset({ToolCapability.NETWORK_EGRESS})
    assert dynamic.capabilities == frozenset()


def test_terminal_network_grant_uses_existing_positive_classifier():
    offline = tool_capabilities.issue_local_execution_grant("terminal", "pwd && git status")
    online = tool_capabilities.issue_local_execution_grant(
        "terminal", "python -m pip install example-package"
    )

    assert offline.capabilities == frozenset()
    assert online.allows(ToolCapability.NETWORK_EGRESS)


def test_grant_is_bound_to_exact_payload_and_authority():
    payload = "python -m pip install example-package"
    grant = tool_capabilities.issue_local_execution_grant("terminal", payload)
    assert tool_capabilities.valid_local_execution_grant(
        grant,
        tool_name="terminal",
        payload=payload,
    )
    assert not tool_capabilities.valid_local_execution_grant(
        grant,
        tool_name="terminal",
        payload=payload + "-other",
    )
    forged = replace(grant, _authority=object())
    assert not tool_capabilities.valid_local_execution_grant(
        forged,
        tool_name="terminal",
        payload=payload,
    )


def test_grant_cannot_cross_account_boundary():
    payload = "print(1)"
    grant = tool_capabilities.issue_local_execution_grant("python", payload)
    other = AccountContext("other-account", "other")
    assert not run_as(
        other,
        tool_capabilities.valid_local_execution_grant,
        grant,
        tool_name="python",
        payload=payload,
    )


def test_broker_rejects_non_local_execution_tools():
    with pytest.raises(ValueError, match="Unsupported local execution tool"):
        tool_capabilities.issue_local_execution_grant("web_search", "cats")


def test_confinement_fails_closed_on_forged_grant():
    payload = "print(1)"
    grant = tool_capabilities.issue_local_execution_grant("python", payload)
    forged = replace(grant, _authority=object())
    with pytest.raises(ToolConfinementUnavailable, match="capability grant"):
        tools._account_confinement(
            capability_grant=forged,
            tool_name="python",
            payload=payload,
        )


def test_macos_network_profile_is_driven_by_valid_grant(monkeypatch):
    monkeypatch.setattr(tool_confinement.sys, "platform", "darwin")
    monkeypatch.setattr(
        tool_confinement.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    offline_payload = "pwd && git status"
    offline = tool_capabilities.issue_local_execution_grant("terminal", offline_payload)
    offline_profile = tools._account_confinement(
        capability_grant=offline,
        tool_name="terminal",
        payload=offline_payload,
    ).wrapper[2]

    online_payload = "python -m pip install example-package"
    online = tool_capabilities.issue_local_execution_grant("terminal", online_payload)
    online_profile = tools._account_confinement(
        capability_grant=online,
        tool_name="terminal",
        payload=online_payload,
    ).wrapper[2]

    assert "(allow network-outbound)" not in offline_profile
    assert "(allow network-outbound)" in online_profile
    for profile in (offline_profile, online_profile):
        assert "(deny network*)" in profile
        assert "(allow network*)" not in profile
        assert "network-inbound" not in profile


def test_linux_network_confinement_is_driven_only_by_valid_grant(monkeypatch):
    seen = []

    def confinement(_sandbox_site_dir, *, allow_network=False):
        seen.append(allow_network)
        return tool_confinement.Confinement(mechanism="test-landlock")

    monkeypatch.setattr(tool_confinement.sys, "platform", "linux")
    monkeypatch.setattr(tool_confinement, "_linux_confinement", confinement)

    offline_payload = "name = 'socket'\n__import__(name)"
    offline = tool_capabilities.issue_local_execution_grant("python", offline_payload)
    assert tools._account_confinement(
        capability_grant=offline,
        tool_name="python",
        payload=offline_payload,
    ).mechanism == "test-landlock"

    online_payload = "import socket\nprint(socket.AF_INET)"
    online = tool_capabilities.issue_local_execution_grant("python", online_payload)
    assert tools._account_confinement(
        capability_grant=online,
        tool_name="python",
        payload=online_payload,
    ).mechanism == "test-landlock"

    assert seen == [False, True]
