# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded correlation/index helpers for the Helix Execution Graph.

All reads go through the existing account-scoped ledger.  The correlator never
reconstructs missing provenance and never performs foreground retrieval work; it
only joins receipts that were already persisted by the backend.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .capture import capture_session_key
from .ledger import _path as _ledger_path
from .ledger import append_record

_SCHEMA_VERSION = "helix.provenance.v1"
_TURN_RECEIPT_VERSION = "helix.turn-receipt.v1"
_SCAN_LIMIT = 384
_MAX_CHECKPOINTS = 8
_MAX_DECISIONS = 32
_MAX_ACTIONS = 64
_MAX_STRING_CHARS = 2_000
_MAX_LIST_ITEMS = 64
_MAX_DICT_ITEMS = 96
_MAX_RESPONSE_BYTES = 512 * 1024
_MAX_LEDGER_READ_BYTES = 2 * 1024 * 1024


def _text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def append_turn_receipt(
    *,
    session_id: str | None,
    thread_id: str | None,
    turn_id: str | None,
    trajectory_id: str | None,
    model: str | None,
    actions: list[str] | tuple[str, ...] | None,
    base_model_id: str | None = None,
    skill_retention: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    qlora_outcome: dict[str, Any] | None = None,
) -> bool:
    """Append the final correlation index only after the caller finalized actions."""

    normalized_actions = []
    for action in actions or []:
        item = _text(action, 240)
        if item and item not in normalized_actions:
            normalized_actions.append(item)
        if len(normalized_actions) >= _MAX_ACTIONS:
            break
    record = {
        "schema_version": _TURN_RECEIPT_VERSION,
        "session_id": _text(session_id, 500),
        "thread_id": _text(thread_id, 200) or None,
        "turn_id": _text(turn_id, 200) or None,
        "trajectory_id": _text(trajectory_id, 200),
        "model": _text(model, 500),
        "base_model_id": _text(base_model_id, 500),
        "actions": normalized_actions,
        "created_at_ms": int(time.time() * 1_000),
    }
    normalized_retention: list[dict[str, Any]] = []
    for item in skill_retention or []:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("skill_name"), 120)
        disposition = _text(item.get("disposition"), 80)
        reason = _text(item.get("reason"), 800)
        if not name or not disposition:
            continue
        entry: dict[str, Any] = {
            "skill_name": name,
            "disposition": disposition,
            "reason": reason,
        }
        if isinstance(item.get("evidence"), dict):
            entry["evidence"] = _bounded(
                item["evidence"],
                max_string=240,
                max_list=16,
                max_dict=16,
            )
        normalized_retention.append(entry)
        if len(normalized_retention) >= 32:
            break
    if normalized_retention:
        record["skill_retention"] = normalized_retention
    if isinstance(qlora_outcome, dict):
        outcome = _text(qlora_outcome.get("outcome"), 40)
        if outcome in {"queued", "deferred", "denied", "not_applicable"}:
            record["qlora_outcome"] = {
                "outcome": outcome,
                "reason": _text(qlora_outcome.get("reason"), 500),
            }
    return append_record("turn-receipts", record)


def _recent_records(kind: str, limit: int = _SCAN_LIMIT) -> list[dict[str, Any]]:
    """Read only a bounded tail of one account-scoped ledger file."""

    cap = max(1, min(int(limit), _SCAN_LIMIT))
    try:
        from .provenance_index import query_records

        indexed = query_records(kind, limit=cap)
        if len(indexed) >= cap:
            return indexed
    except Exception:
        indexed = []
    try:
        path = _ledger_path(kind)
        if not path.is_file():
            return []
        size = path.stat().st_size
        start = max(0, size - _MAX_LEDGER_READ_BYTES)
        with path.open("rb") as handle:
            preceding = b"\n"
            if start:
                handle.seek(start - 1)
                preceding = handle.read(1)
            handle.seek(start)
            raw = handle.read(_MAX_LEDGER_READ_BYTES)
        if start and preceding != b"\n":
            boundary = raw.find(b"\n")
            if boundary < 0:
                return []
            raw = raw[boundary + 1 :]
        rows: list[dict[str, Any]] = []
        for line in raw.decode("utf-8", errors="replace").splitlines()[-cap:]:
            try:
                value = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        if not indexed:
            return rows
        merged: dict[str, dict[str, Any]] = {}
        for row in [*rows, *indexed]:
            try:
                key = json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
                )
            except Exception:
                continue
            merged[key] = row
        return sorted(
            merged.values(),
            key=_created_at_ms,
        )[-cap:]
    except Exception:
        return indexed


def _indexed_records(
    kind: str,
    *,
    session_id: str = "",
    thread_id: str = "",
    turn_id: str = "",
    trajectory_id: str = "",
    limit: int = _SCAN_LIMIT,
) -> list[dict[str, Any]]:
    try:
        from .provenance_index import query_records

        rows = query_records(
            kind,
            session_id=session_id or None,
            thread_id=thread_id or None,
            turn_id=turn_id or None,
            trajectory_id=trajectory_id or None,
            limit=limit,
        )
        if rows:
            return rows
    except Exception:
        pass
    return _recent_records(kind, limit=limit)


def _created_at_ms(row: dict[str, Any]) -> int:
    try:
        return int(row.get("created_at_ms") or 0)
    except (TypeError, ValueError):
        return 0


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        enumerate(rows),
        key=lambda pair: (_created_at_ms(pair[1]), pair[0]),
    )[1]


def _matches_turn(
    row: dict[str, Any],
    *,
    session_id: str,
    thread_id: str,
    turn_id: str,
) -> bool:
    if session_id and _text(row.get("session_id"), 500) != session_id:
        return False
    if thread_id and _text(row.get("thread_id"), 200) != thread_id:
        return False
    if turn_id and _text(row.get("turn_id"), 200) != turn_id:
        return False
    return True


def _checkpoint_matches_query(
    row: dict[str, Any],
    *,
    session_id: str,
    thread_id: str,
    turn_id: str,
) -> bool:
    if thread_id and _text(row.get("thread_id"), 200) != thread_id:
        return False
    if turn_id and _text(row.get("trajectory_id"), 200) != turn_id:
        return False
    if not session_id:
        return True
    checkpoint_session = _text(row.get("session_id"), 500)
    if checkpoint_session == session_id:
        return True
    base = capture_session_key(session_id, thread_id or None, None)
    if checkpoint_session == base or checkpoint_session.startswith(base + "::turn::"):
        return True
    if not thread_id and (
        checkpoint_session.startswith(session_id + "::thread::")
        or checkpoint_session.startswith(session_id + "::turn::")
    ):
        return True
    return False


def _checkpoint_matches_turn(row: dict[str, Any], turn: dict[str, Any]) -> bool:
    trajectory_id = _text(turn.get("trajectory_id"), 200)
    if trajectory_id and _text(row.get("trajectory_id"), 200) != trajectory_id:
        return False
    thread_id = _text(turn.get("thread_id"), 200)
    if thread_id and _text(row.get("thread_id"), 200) != thread_id:
        return False
    session_id = _text(turn.get("session_id"), 500)
    turn_id = _text(turn.get("turn_id"), 200)
    if not session_id:
        return True
    expected = capture_session_key(session_id, thread_id or None, turn_id or None)
    checkpoint_session = _text(row.get("session_id"), 500)
    return checkpoint_session in {session_id, expected}


def _trajectory_matches(row: dict[str, Any], trajectory_id: str) -> bool:
    if not trajectory_id:
        return False
    return _text(
        row.get("trajectory_id") or row.get("source_trajectory_id"),
        200,
    ) == trajectory_id


def _latest_trajectory_record(kind: str, trajectory_id: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in _indexed_records(kind, trajectory_id=trajectory_id, limit=_SCAN_LIMIT)
        if _trajectory_matches(row, trajectory_id)
    ]
    return rows[-1] if rows else None


def _decision_records(trajectory_id: str) -> list[dict[str, Any]]:
    matching = [
        row
        for row in _indexed_records(
            "decisions", trajectory_id=trajectory_id, limit=_SCAN_LIMIT
        )
        if _trajectory_matches(row, trajectory_id)
    ]
    latest_by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in matching:
        decision_id = _text(row.get("decision_id"), 240)
        if decision_id:
            latest_by_id[decision_id] = row
        else:
            anonymous.append(row)
    rows = [*anonymous[-_MAX_DECISIONS:], *latest_by_id.values()]
    return rows[-_MAX_DECISIONS:]


def _scope_collision(
    *,
    turn_receipts: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    selected_turn: dict[str, Any] | None,
    selected_checkpoint: dict[str, Any] | None,
    trajectory_id: str,
) -> bool:
    """Fail closed when one trajectory id is reused across distinct execution scopes."""

    if not trajectory_id:
        return False

    def _turn_scope(row: dict[str, Any]) -> tuple[str, str]:
        session = _text(row.get("session_id"), 500)
        thread = _text(row.get("thread_id"), 200)
        return capture_session_key(session or None, thread or None, None), thread

    def _checkpoint_scope(row: dict[str, Any]) -> tuple[str, str]:
        session = _text(row.get("session_id"), 500)
        thread = _text(row.get("thread_id"), 200)
        # Adaptive checkpoints persist the exact turn capture key. Strip only the
        # final turn suffix so this canonical base matches capture_session_key()
        # used for the corresponding final turn receipt.
        base = session.split("::turn::", 1)[0] or "default"
        return base, thread

    if selected_turn is not None:
        selected_scope = _turn_scope(selected_turn)
    elif selected_checkpoint is not None:
        selected_scope = _checkpoint_scope(selected_checkpoint)
    else:
        return False

    observed_scopes: set[tuple[str, str]] = set()
    for row in turn_receipts:
        if _text(row.get("trajectory_id"), 200) != trajectory_id:
            continue
        observed_scopes.add(_turn_scope(row))
    for row in checkpoints:
        if _text(row.get("trajectory_id"), 200) != trajectory_id:
            continue
        observed_scopes.add(_checkpoint_scope(row))
    return any(scope != selected_scope for scope in observed_scopes)


def _safe_training_target(
    trajectory_id: str,
    *,
    thread_id: str,
    ambiguous: bool,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in _indexed_records(
            "training-target-receipts", trajectory_id=trajectory_id, limit=_SCAN_LIMIT
        )
        if _trajectory_matches(row, trajectory_id)
    ]
    if thread_id:
        scoped = [
            row
            for row in rows
            if not _text(row.get("source_thread_id"), 200)
            or _text(row.get("source_thread_id"), 200) == thread_id
        ]
        if ambiguous:
            scoped = [
                row
                for row in scoped
                if _text(row.get("source_thread_id"), 200) == thread_id
            ]
        rows = scoped
    return rows[-1] if rows else None


def _safe_qlora_admission(
    trajectory_id: str,
    *,
    thread_id: str,
    ambiguous: bool,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in _indexed_records(
            "qlora-admissions", trajectory_id=trajectory_id, limit=_SCAN_LIMIT
        )
        if _trajectory_matches(row, trajectory_id)
    ]
    if thread_id:
        scoped = []
        for row in rows:
            receipt = row.get("training_target_receipt")
            source_thread = _text(
                receipt.get("source_thread_id") if isinstance(receipt, dict) else None,
                200,
            )
            if source_thread == thread_id or (not source_thread and not ambiguous):
                scoped.append(row)
        rows = scoped
    return rows[-1] if rows else None


def _bounded(
    value: Any,
    *,
    depth: int = 0,
    max_string: int = _MAX_STRING_CHARS,
    max_list: int = _MAX_LIST_ITEMS,
    max_dict: int = _MAX_DICT_ITEMS,
) -> Any:
    if depth >= 8:
        return "<depth-truncated>"
    if isinstance(value, str):
        return value[:max_string]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        items = list(value.items())[:max_dict]
        return {
            str(key)[:160]: _bounded(
                item,
                depth=depth + 1,
                max_string=max_string,
                max_list=max_list,
                max_dict=max_dict,
            )
            for key, item in items
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded(
                item,
                depth=depth + 1,
                max_string=max_string,
                max_list=max_list,
                max_dict=max_dict,
            )
            for item in list(value)[:max_list]
        ]
    return str(value)[:max_string]


def _bound_response(payload: dict[str, Any]) -> dict[str, Any]:
    def _size(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            return _MAX_RESPONSE_BYTES + 1

    def _mark_truncated(value: dict[str, Any], sources: list[str] | None = None) -> None:
        value["response_truncated"] = True
        missing = value.setdefault("missing_sources", [])
        if not isinstance(missing, list):
            missing = []
            value["missing_sources"] = missing
        if "response_size_limit" not in missing:
            missing.append("response_size_limit")
        if sources:
            existing = value.setdefault("truncated_sources", [])
            if not isinstance(existing, list):
                existing = []
                value["truncated_sources"] = existing
            for source in sources:
                if source not in existing:
                    existing.append(source)

    bounded = _bounded(payload)
    if _size(bounded) <= _MAX_RESPONSE_BYTES:
        return bounded
    compact = _bounded(payload, max_string=768, max_list=16, max_dict=64)
    if isinstance(compact, dict):
        _mark_truncated(compact)
    if isinstance(compact, dict) and _size(compact) <= _MAX_RESPONSE_BYTES:
        return compact

    # Nested receipts can still exceed the byte budget after per-value truncation
    # because a bounded tree may contain many bounded branches. Drop the largest
    # optional top-level sources first, recording exactly which provenance was
    # omitted instead of returning a nominally "truncated" multi-megabyte body.
    working = dict(compact) if isinstance(compact, dict) else {}
    protected = {
        "schema_version",
        "available",
        "selection",
        "turn_receipt",
        "skill_retention",
        "qlora_outcome",
        "final_actions",
        "missing_sources",
        "response_truncated",
        "truncated_sources",
    }
    optional = [
        key
        for key in working
        if key not in protected and key not in {"response_truncated", "truncated_sources"}
    ]
    optional.sort(key=lambda key: _size(working.get(key)), reverse=True)
    dropped: list[str] = []
    for key in optional:
        if _size(working) <= _MAX_RESPONSE_BYTES:
            break
        value = working.get(key)
        working[key] = [] if isinstance(value, list) else None
        dropped.append(key)
        _mark_truncated(working, [key])
    if _size(working) <= _MAX_RESPONSE_BYTES:
        return working

    # Hard fail-safe: preserve only the compact correlation/index fields. These
    # are all backend-bounded independently and fit comfortably under the cap.
    # Detailed sources remain discoverably absent via response_size_limit and the
    # explicit truncated_sources list.
    detailed_fields = [
        "adaptive_checkpoints",
        "mechanisms",
        "trajectory",
        "evidence",
        "self_audit",
        "adaptation",
        "quality",
        "decisions",
        "decision_outcomes",
        "counterfactual",
        "qlora_admission",
        "training_target_receipt",
    ]
    minimal = {
        "schema_version": _text(payload.get("schema_version"), 120),
        "available": bool(payload.get("available")),
        "selection": _bounded(
            payload.get("selection"), max_string=240, max_list=8, max_dict=24
        ),
        "turn_receipt": _bounded(
            payload.get("turn_receipt"), max_string=320, max_list=16, max_dict=32
        ),
        "skill_retention": _bounded(
            payload.get("skill_retention") or [], max_string=240, max_list=16, max_dict=16
        ),
        "qlora_outcome": _bounded(
            payload.get("qlora_outcome"), max_string=320, max_list=8, max_dict=16
        ),
        "final_actions": _bounded(
            payload.get("final_actions") or [], max_string=240, max_list=_MAX_ACTIONS, max_dict=8
        ),
        "missing_sources": list(
            dict.fromkeys(
                [
                    *(
                        payload.get("missing_sources")
                        if isinstance(payload.get("missing_sources"), list)
                        else []
                    ),
                    "response_size_limit",
                ]
            )
        )[:_MAX_LIST_ITEMS],
        "response_truncated": True,
        "truncated_sources": list(dict.fromkeys([*dropped, *detailed_fields]))[:_MAX_LIST_ITEMS],
    }
    if _size(minimal) <= _MAX_RESPONSE_BYTES:
        return minimal

    # Defensive last resort against malformed caller-owned identifiers. Keep the
    # selection envelope tiny while preserving the fact that provenance exists.
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    return {
        "schema_version": _SCHEMA_VERSION,
        "available": bool(payload.get("available")),
        "selection": {
            key: _text(selection.get(key), 120) or None
            for key in ("source", "session_id", "thread_id", "turn_id", "trajectory_id")
        },
        "turn_receipt": None,
        "adaptive_checkpoints": [],
        "mechanisms": None,
        "trajectory": None,
        "evidence": None,
        "self_audit": None,
        "adaptation": None,
        "quality": None,
        "decisions": [],
        "decision_outcomes": [],
        "counterfactual": None,
        "qlora_admission": None,
        "training_target_receipt": None,
        "skill_retention": [],
        "qlora_outcome": None,
        "final_actions": [],
        "missing_sources": ["response_size_limit"],
        "response_truncated": True,
        "truncated_sources": detailed_fields,
    }


def read_provenance(
    *,
    session_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Return one bounded, correlated provenance bundle for the requested/latest turn."""

    wanted_session = _text(session_id, 500)
    wanted_thread = _text(thread_id, 200)
    wanted_turn = _text(turn_id, 200)
    turn_receipts = (
        _indexed_records(
            "turn-receipts",
            session_id=wanted_session,
            thread_id=wanted_thread,
            turn_id=wanted_turn,
            limit=_SCAN_LIMIT,
        )
        if wanted_session or wanted_thread or wanted_turn
        else _recent_records("turn-receipts", limit=_SCAN_LIMIT)
    )
    checkpoints_all = (
        _indexed_records(
            "adaptive-checkpoints",
            thread_id=wanted_thread,
            turn_id=wanted_turn,
            limit=_SCAN_LIMIT,
        )
        if wanted_thread or wanted_turn
        else _recent_records("adaptive-checkpoints", limit=_SCAN_LIMIT)
    )

    matching_turns = [
        row
        for row in turn_receipts
        if _matches_turn(
            row,
            session_id=wanted_session,
            thread_id=wanted_thread,
            turn_id=wanted_turn,
        )
    ]
    candidate_turn = _latest(matching_turns)
    query_checkpoints = [
        row
        for row in checkpoints_all
        if _checkpoint_matches_query(
            row,
            session_id=wanted_session,
            thread_id=wanted_thread,
            turn_id=wanted_turn,
        )
    ]
    candidate_checkpoint = _latest(query_checkpoints)
    live_checkpoint_is_newer = bool(
        not wanted_turn
        and candidate_checkpoint is not None
        and (
            candidate_turn is None
            or _created_at_ms(candidate_checkpoint) > _created_at_ms(candidate_turn)
        )
        and _text(candidate_checkpoint.get("trajectory_id"), 200)
        != _text((candidate_turn or {}).get("trajectory_id"), 200)
    )
    selected_turn = None if live_checkpoint_is_newer else candidate_turn

    if selected_turn is not None:
        matching_checkpoints = [
            row for row in checkpoints_all if _checkpoint_matches_turn(row, selected_turn)
        ]
    else:
        anchor = candidate_checkpoint
        matching_checkpoints = [
            row
            for row in query_checkpoints
            if anchor is None
            or (
                _text(row.get("trajectory_id"), 200)
                == _text(anchor.get("trajectory_id"), 200)
                and (
                    not _text(anchor.get("thread_id"), 200)
                    or _text(row.get("thread_id"), 200)
                    == _text(anchor.get("thread_id"), 200)
                )
            )
        ]
    selected_checkpoint = _latest(matching_checkpoints)

    trajectory_id = _text(
        (selected_turn or {}).get("trajectory_id")
        or (selected_checkpoint or {}).get("trajectory_id"),
        200,
    )
    selected_thread = _text(
        (selected_turn or {}).get("thread_id")
        or (selected_checkpoint or {}).get("thread_id")
        or wanted_thread,
        200,
    )
    collision_turn_receipts = (
        _indexed_records(
            "turn-receipts",
            trajectory_id=trajectory_id,
            limit=2_000,
        )
        if trajectory_id
        else turn_receipts
    )
    collision_checkpoints = (
        _indexed_records(
            "adaptive-checkpoints",
            trajectory_id=trajectory_id,
            limit=2_000,
        )
        if trajectory_id
        else checkpoints_all
    )
    ambiguous = _scope_collision(
        turn_receipts=collision_turn_receipts,
        checkpoints=collision_checkpoints,
        selected_turn=selected_turn,
        selected_checkpoint=selected_checkpoint,
        trajectory_id=trajectory_id,
    )

    checkpoints = matching_checkpoints[-_MAX_CHECKPOINTS:]
    mechanisms = (
        checkpoints[-1].get("mechanisms")
        if checkpoints and isinstance(checkpoints[-1].get("mechanisms"), dict)
        else None
    )
    missing: list[str] = []
    if selected_turn is None:
        missing.append("turn_receipt")
    if not checkpoints:
        missing.append("adaptive_checkpoints")
    if mechanisms is None:
        missing.append("mechanisms")

    trajectory = evidence = self_audit = adaptation = quality = counterfactual = None
    decisions: list[dict[str, Any]] = []
    decision_outcomes: list[dict[str, Any]] = []
    qlora_admission = training_target_receipt = None

    if trajectory_id and not ambiguous:
        trajectory = _latest_trajectory_record("trajectories", trajectory_id)
        evidence = _latest_trajectory_record("evidence", trajectory_id)
        self_audit = _latest_trajectory_record("self-audits", trajectory_id)
        adaptation = _latest_trajectory_record("adaptations", trajectory_id)
        quality = _latest_trajectory_record("quality", trajectory_id)
        counterfactual = _latest_trajectory_record("counterfactuals", trajectory_id)
        decisions = _decision_records(trajectory_id)
        decision_outcomes = [
            row
            for row in decisions
            if row.get("eventual_outcome") is not None
            or row.get("retrospective_usefulness") is not None
        ]
    elif ambiguous:
        missing.append("trajectory_scope_ambiguous_across_sessions_or_threads")

    if trajectory_id:
        training_target_receipt = _safe_training_target(
            trajectory_id,
            thread_id=selected_thread,
            ambiguous=ambiguous,
        )
        qlora_admission = _safe_qlora_admission(
            trajectory_id,
            thread_id=selected_thread,
            ambiguous=ambiguous,
        )

    sources = {
        "trajectory": trajectory,
        "evidence": evidence,
        "self_audit": self_audit,
        "adaptation": adaptation,
        "quality": quality,
        "decisions": decisions,
        "decision_outcomes": decision_outcomes,
        "counterfactual": counterfactual,
        "qlora_admission": qlora_admission,
        "training_target_receipt": training_target_receipt,
    }
    for name, value in sources.items():
        if value is None or value == []:
            missing.append(name)

    trajectory_telemetry = trajectory.get("telemetry") if isinstance(trajectory, dict) else None
    trajectory_preflight = (
        trajectory_telemetry.get("preflight")
        if isinstance(trajectory_telemetry, dict)
        and isinstance(trajectory_telemetry.get("preflight"), dict)
        else None
    )
    trajectory_memory_preflight = (
        trajectory_preflight.get("memory")
        if isinstance(trajectory_preflight, dict)
        and isinstance(trajectory_preflight.get("memory"), dict)
        else None
    )
    for checkpoint in checkpoints:
        checkpoint_mechanisms = checkpoint.get("mechanisms")
        if not isinstance(checkpoint_mechanisms, dict):
            continue
        mem0 = checkpoint_mechanisms.get("mem0")
        retrieval = mem0.get("retrieval") if isinstance(mem0, dict) else None
        if not isinstance(retrieval, dict) or retrieval.get("status") != "preflight_owned":
            continue
        checkpoint_receipt = retrieval.get("receipt")
        if not isinstance(checkpoint_receipt, dict) and not isinstance(trajectory_memory_preflight, dict):
            missing.append("mem0_preflight_retrieval_details")
        break

    final_actions = list((selected_turn or {}).get("actions") or [])[:_MAX_ACTIONS]
    skill_retention = (
        list((selected_turn or {}).get("skill_retention") or [])[:32]
        if isinstance((selected_turn or {}).get("skill_retention"), list)
        else []
    )
    qlora_outcome = (
        dict((selected_turn or {}).get("qlora_outcome") or {})
        if isinstance((selected_turn or {}).get("qlora_outcome"), dict)
        else None
    )
    if (
        selected_turn is not None
        and any(
            str(action).startswith(
                (
                    "promote_temp_skill:",
                    "discard_temp_skill:",
                    "retain_temp_skill_",
                )
            )
            for action in final_actions
        )
        and not skill_retention
    ):
        missing.append("skill_retention")
    if (
        selected_turn is not None
        and (
            "stage_qlora_candidate" in final_actions
            or "queue_qlora_training" in final_actions
            or qlora_admission is not None
        )
        and qlora_outcome is None
    ):
        missing.append("qlora_outcome")

    missing = list(dict.fromkeys(missing))
    selected_source = "turn_receipt" if selected_turn is not None else (
        "adaptive_checkpoint" if selected_checkpoint is not None else "none"
    )
    response = {
        "schema_version": _SCHEMA_VERSION,
        "available": bool(selected_turn or selected_checkpoint),
        "selection": {
            "source": selected_source,
            "session_id": (selected_turn or {}).get("session_id") or wanted_session or None,
            "thread_id": selected_thread or None,
            "turn_id": (selected_turn or {}).get("turn_id") or wanted_turn or trajectory_id or None,
            "trajectory_id": trajectory_id or None,
            "scope_ambiguous": ambiguous,
        },
        "turn_receipt": selected_turn,
        "adaptive_checkpoints": checkpoints,
        "mechanisms": mechanisms,
        "trajectory": trajectory,
        "evidence": evidence,
        "self_audit": self_audit,
        "adaptation": adaptation,
        "quality": quality,
        "decisions": decisions,
        "decision_outcomes": decision_outcomes,
        "counterfactual": counterfactual,
        "qlora_admission": qlora_admission,
        "training_target_receipt": training_target_receipt,
        "skill_retention": skill_retention,
        "qlora_outcome": qlora_outcome,
        "final_actions": final_actions,
        "missing_sources": missing,
    }
    return _bound_response(response)
