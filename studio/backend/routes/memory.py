# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only Mem0 status and retrieval endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.memory.experience_idempotency import (
    MemoryExperienceIdempotencyConflict,
    MemoryExperienceReplayUnavailable,
    run_idempotent_memory_experience,
)

router = APIRouter()


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)


class MemoryExperienceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    threadId: str | None = Field(default=None, max_length=200)
    kind: str = Field(default="experience", max_length=40)
    title: str | None = Field(default=None, max_length=240)
    idempotencyKey: str | None = Field(default=None, min_length=1, max_length=240)


def persist_memory_experience(
    subject: str | None,
    text: str,
    *,
    thread_id: str | None = None,
    kind: str = "experience",
    title: str | None = None,
    enabled: bool = True,
    idempotency_key: str | None = None,
) -> dict:
    """Persist one bounded experience without making callers depend on Mem0 health."""
    def _persist() -> dict:
        if not enabled:
            return {"stored": False, "reason": "disabled"}
        try:
            from core.memory.mem0_store import add_experience

            return add_experience(
                subject,
                text,
                thread_id=thread_id,
                kind=kind,
                title=title,
                idempotency_key=idempotency_key,
            )
        except Exception as error:  # noqa: BLE001 -- learning stays durable in its local ledger
            return {"stored": False, "reason": str(error)[:1_000] or "memory-unavailable"}

    if not idempotency_key:
        return _persist()
    return run_idempotent_memory_experience(
        idempotency_key=idempotency_key,
        thread_id=thread_id,
        payload={
            "text": text,
            "threadId": thread_id,
            "kind": kind,
            "title": title,
        },
        operation=_persist,
        recovery_probe=(
            lambda: __import__(
                "core.memory.mem0_store",
                fromlist=["experience_receipt_for_idempotency"],
            ).experience_receipt_for_idempotency(
                idempotency_key,
                thread_id=thread_id,
            )
        ),
    )


@router.get("")
def get_memory_status(current_subject: str = Depends(get_current_subject)):
    from core.memory.mem0_store import status

    return status()


@router.post("/search")
def search_memory(payload: MemorySearchRequest, current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"results": [], "available": False, "reason": "disabled"}
    from core.memory.mem0_store import search

    return search(current_subject, payload.query, payload.limit)


@router.post("/experiences")
def add_memory_experience(payload: MemoryExperienceRequest, current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    enabled = _read_state().get("mem0Enabled") is not False
    try:
        return persist_memory_experience(
            current_subject,
            payload.text,
            thread_id=payload.threadId,
            kind=payload.kind,
            title=payload.title,
            enabled=enabled,
            idempotency_key=payload.idempotencyKey,
        )
    except MemoryExperienceIdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MemoryExperienceReplayUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/graph")
def get_memory_graph(current_subject: str = Depends(get_current_subject)):
    from routes.learning import _read_state

    if _read_state().get("mem0Enabled") is False:
        return {"nodes": [], "edges": [], "available": False, "reason": "disabled"}
    from core.memory.mem0_store import graph_snapshot

    return graph_snapshot(current_subject)
