# SPDX-License-Identifier: AGPL-3.0-only
"""Adaptive occupancy plus current preflight/search contracts."""

from core.inference import checkpoint, tools
from routes import learning


def test_occupancy_handoff_ratio_is_eighty_percent():
    assert checkpoint.HANDOFF_RATIO == 0.80
    assert checkpoint.occupancy_triggers_handoff(79, 100) is False
    assert checkpoint.occupancy_triggers_handoff(80, 100) is True
    assert checkpoint.occupancy_triggers_handoff(81, 100) is True


def test_search_memory_is_an_alias_of_search_conversation():
    names = {item["function"]["name"] for item in tools.ALL_TOOLS}
    assert "search_memory" in names
    assert "search_conversation" in names


def test_hermes_context_keeps_checkpoint_retrieval_out_of_the_hot_path():
    context = learning._context_instruction(learning._empty_state())
    assert "search_memory" not in context
    assert "Mem0" in context
