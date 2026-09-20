# SPDX-License-Identifier: AGPL-3.0-only
"""Wait-ready Mem0 handoff so compaction can continue the current task."""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

HANDOFF_CONTINUE = (
    "<context_handoff>Older turns were stored in Mem0 and the conversation archive so this "
    "task can continue without intelligence loss. Standing instructions are in carried_forward. "
    "Call search_conversation if you need a dropped detail. Continue the current work; do not restart "
    "or tell the user the conversation was lost.</context_handoff>"
)


def wait_handoff_ready(*, timeout_s: float = 2.0) -> None:
    """Wait briefly for archive/Mem0 to accept a write. Never block generation for long."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            from core.rag import conversation_archive

            if conversation_archive.enabled():
                return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return
        time.sleep(0.05)


def handoff_evicted_to_mem0(thread_id: Optional[str], gone: list[dict]) -> None:
    """Persist dropped turns into the Mem0 graph. Fail-closed: chat continues either way."""
    if not gone:
        return
    wait_handoff_ready()
    chunks: list[str] = []
    used = 0
    for message in gone:
        role = str(message.get("role") or "")
        if role not in {"user", "assistant", "system", "developer"}:
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        text = text.strip()
        if not text:
            continue
        piece = f"{role}: {text}"[:1_200]
        if used + len(piece) > 6_000:
            break
        chunks.append(piece)
        used += len(piece)
    if not chunks:
        return
    try:
        from core.memory.mem0_store import add_experience

        add_experience(
            None,
            "\n\n".join(chunks),
            thread_id=str(thread_id or ""),
            kind="context-handoff",
            title="Conversation handoff",
        )
    except Exception:
        logger.debug("Mem0 context handoff skipped", exc_info=True)


def append_handoff_continue_nudge(conversation: list[dict]) -> list[dict]:
    out = list(conversation)
    for index, message in enumerate(out):
        if message.get("role") in ("system", "developer"):
            text = str(message.get("content") or "")
            if "<context_handoff>" in text:
                return out
            joined = f"{text.rstrip()}\n\n{HANDOFF_CONTINUE}" if text.strip() else HANDOFF_CONTINUE
            out[index] = {**message, "content": joined}
            return out
    return [{"role": "system", "content": HANDOFF_CONTINUE}, *out]


def apply_eviction_handoff(
    conversation: list[dict],
    gone: list[dict],
    *,
    thread_id: Optional[str],
) -> list[dict]:
    if not gone:
        return conversation
    handoff_evicted_to_mem0(thread_id, gone)
    return append_handoff_continue_nudge(conversation)
