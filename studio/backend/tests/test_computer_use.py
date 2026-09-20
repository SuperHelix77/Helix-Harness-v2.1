# SPDX-License-Identifier: AGPL-3.0-only

from core.inference import tools


def test_computer_tool_is_exposed_with_a_bounded_action_contract():
    assert any(item["function"]["name"] == "computer" for item in tools.ALL_TOOLS)
    actions = tools.COMPUTER_TOOL["function"]["parameters"]["properties"]["action"]["enum"]
    assert actions == ["screenshot", "click", "type", "key", "scroll", "open_app", "browse"]


def test_computer_screenshot_is_read_only_but_mutations_need_approval():
    assert tools.is_high_risk_tool_call("computer", {"action": "screenshot"}) is False
    assert tools.is_high_risk_tool_call("computer", {"action": "click", "x": 1, "y": 1}) is True
    assert tools.is_high_risk_tool_call("computer", {"action": "type", "text": "hello"}) is True
    assert tools.is_high_risk_tool_call("computer", {"action": "browse", "url": "https://example.com"}) is True
