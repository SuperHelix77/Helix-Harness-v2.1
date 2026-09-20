# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Governed Hermes-style self-improvement for the local agent.

The agent may propose a compact lesson or a plain-text skill, but proposals stay
pending until the user approves them. Approval never executes code: skill writes
contain only ``SKILL.md`` and memory is bounded before it is injected into a
future prompt.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from utils.account_context import current_account_id, is_owner_context
from utils.paths import account_path, ensure_dir, workspace_root

router = APIRouter()

LearningKind = Literal["memory", "user", "skill"]
LearningTarget = Literal["codex", "claude", "both"]
LearningRecommendationAction = Literal["skill", "qlora", "runtime-fix", "none"]
LearningDecisionMode = Literal["ask", "autonomous"]

_STATE_VERSION = 1
_MEMORY_MAX_CHARS = 2_200
_USER_MAX_CHARS = 1_375
_PENDING_MAX = 50
_LEARNING_LOCK = threading.RLock()
_SKILL_INTENTS_MAX = _PENDING_MAX


class LearningProposalRequest(BaseModel):
    kind: LearningKind
    title: str = Field(min_length = 1, max_length = 240)
    content: str = Field(min_length = 1, max_length = 100_000)
    reason: str = Field(default = "", max_length = 2_000)
    name: str | None = Field(default = None, max_length = 64)
    target: LearningTarget = "codex"
    sourceThreadId: str | None = Field(default = None, max_length = 200)
    recommendationAction: LearningRecommendationAction | None = None
    recommendationReason: str = Field(default = "", max_length = 2_000)
    idempotencyKey: str | None = Field(default=None, min_length=1, max_length=240)


class LearningConfigRequest(BaseModel):
    enabled: bool | None = None
    mem0Enabled: bool | None = None
    onTheFlySkills: bool | None = None
    decisionMode: LearningDecisionMode | None = None
    allowSkillCreation: bool | None = None
    allowQloraTraining: bool | None = None
    allowRuntimeFix: bool | None = None


def _state_path() -> Path:
    return account_path("learning/state.json")


def _empty_state() -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "enabled": True,
        "mem0Enabled": True,
        "onTheFlySkills": True,
        "decisionMode": "ask",
        "allowSkillCreation": False,
        "allowQloraTraining": False,
        "allowRuntimeFix": False,
        "memory": [],
        "user": [],
        "pending": [],
        "proposalReceipts": [],
        "skillCreationIntents": [],
        "lastRecommendation": None,
    }


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        raw = json.loads(path.read_text(encoding = "utf-8"))
    except FileNotFoundError:
        return _empty_state()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code = 500, detail = "The local learning store could not be read.") from error
    if not isinstance(raw, dict):
        raise HTTPException(status_code = 500, detail = "The local learning store is invalid.")
    state = _empty_state()
    state["enabled"] = raw.get("enabled") is not False
    state["mem0Enabled"] = raw.get("mem0Enabled") is not False
    state["onTheFlySkills"] = raw.get("onTheFlySkills") is not False
    state["decisionMode"] = raw.get("decisionMode") if raw.get("decisionMode") in {"ask", "autonomous"} else "ask"
    for key in ("allowSkillCreation", "allowQloraTraining", "allowRuntimeFix"):
        state[key] = raw.get(key) is True
    for key in ("memory", "user", "pending", "proposalReceipts", "skillCreationIntents"):
        value = raw.get(key)
        if isinstance(value, list):
            state[key] = [item for item in value if isinstance(item, dict)]
    if isinstance(raw.get("lastRecommendation"), dict):
        state["lastRecommendation"] = raw["lastRecommendation"]
    return state


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    ensure_dir(path.parent)
    # Replace in the same directory so a crash cannot leave a half-written JSON store.
    with tempfile.NamedTemporaryFile(
        mode = "w",
        encoding = "utf-8",
        dir = path.parent,
        prefix = ".learning-",
        suffix = ".tmp",
        delete = False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, ensure_ascii = False, indent = 2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _entry_text(entry: dict[str, Any]) -> str:
    title = str(entry.get("title") or "Lesson").strip()
    content = str(entry.get("content") or "").strip()
    return f"{title}: {content}" if title else content


def _fit_entries(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep newest entries while enforcing the prompt budget in characters."""
    kept: list[dict[str, Any]] = []
    remaining = limit
    for original in reversed(entries):
        entry = dict(original)
        title = str(entry.get("title") or "Lesson").strip()[:240]
        content = str(entry.get("content") or "").strip()
        overhead = len(title) + 2
        available = max(0, remaining - overhead)
        if available <= 0:
            break
        entry["title"] = title
        entry["content"] = content[:available]
        used = len(_entry_text(entry)) + 2
        remaining -= used
        kept.append(entry)
    return list(reversed(kept))


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(state.get("enabled", True)),
        "mem0Enabled": bool(state.get("mem0Enabled", True)),
        "onTheFlySkills": bool(state.get("onTheFlySkills", True)),
        "decisionMode": state.get("decisionMode", "ask"),
        "allowSkillCreation": bool(state.get("allowSkillCreation", False)),
        "allowQloraTraining": bool(state.get("allowQloraTraining", False)),
        "allowRuntimeFix": bool(state.get("allowRuntimeFix", False)),
        "memory": state.get("memory", []),
        "user": state.get("user", []),
        "pending": state.get("pending", []),
        "limits": {"memoryChars": _MEMORY_MAX_CHARS, "userChars": _USER_MAX_CHARS},
        "context": _context_instruction(state),
        "lastRecommendation": state.get("lastRecommendation"),
    }


def _context_instruction(state: dict[str, Any]) -> str:
    if state.get("enabled") is False:
        return ""
    memory = [_entry_text(entry) for entry in state.get("memory", []) if _entry_text(entry)]
    user = [_entry_text(entry) for entry in state.get("user", []) if _entry_text(entry)]
    if not memory and not user and not state.get("onTheFlySkills", True):
        return ""
    lines = [
        "<hermes_memory>",
        "These are locally approved or autonomous-policy-admitted lessons. They are not necessarily explicit user statements; treat them as hints, verify them against current evidence, and do not mention this block unless relevant.",
        "Learning ladder: record the completed experience first; use Mem0 to find recurrence; turn a repeated procedural gap into a Hermes skill; recommend QLoRA only after a deterministic held-out benchmark proves a model behavior deficiency. Never use a self-written score as proof.",
        "Mem0 provides broader recall when available. The bounded approved lessons below are also kept inline so a Mem0 outage or disabled memory backend cannot erase an already-approved lesson.",
    ]
    if user:
        lines.append("User preferences:")
        lines.extend(f"- {item}" for item in user)
    if memory:
        lines.append("Verified lessons (bounded local durability fallback):")
        lines.extend(f"- {item}" for item in memory)
    if state.get("onTheFlySkills", True):
        lines.extend([
            "On-the-fly skill lane: look at existing skills first and reuse them. Prefer a plain-text skill when it can solve the gap without changing model weights. At the end of a completed local task, you may emit one <unsloth-skill-draft> JSON block with name, title, content, and reason. The app stages it only after Mem0 finds a prior similar experience; do not claim it was saved and never include executable code.",
        ])
    lines.append("</hermes_memory>")
    return "\n".join(lines)


def _valid_skill_name(value: str) -> str:
    from core.inference import skills as skills_module

    try:
        return skills_module._normalize_skill_name(value)
    except skills_module.SkillError as error:
        raise HTTPException(
            status_code = 400,
            detail = "Skill proposals need a lowercase name using letters, numbers, or single hyphens.",
        ) from error


def _skill_targets(target: LearningTarget) -> list[Path]:
    if not is_owner_context():
        # Managed accounts intentionally have one private Agent Skills root. The
        # historical target label must never redirect them into the host home.
        return [workspace_root() / "skills"]
    roots = {
        "codex": Path.home() / ".agents" / "skills",
        "claude": Path.home() / ".claude" / "skills",
    }
    selected: list[Path] = []
    seen: set[Path] = set()
    for name in ("codex", "claude"):
        if target != "both" and target != name:
            continue
        root = roots[name]
        key = root.resolve(strict = False)
        if key in seen:
            continue
        seen.add(key)
        selected.append(root)
    return selected


def _skill_manifest(proposal: dict[str, Any]) -> tuple[str, bytes]:
    """Build the exact manifest used by autonomous skill promotion."""
    import yaml

    from core.inference import skills as skills_module

    name = _valid_skill_name(str(proposal.get("name") or proposal.get("title") or ""))
    description = str(proposal.get("title") or name).strip()[:500]
    content = str(proposal.get("content") or "").strip()
    frontmatter = yaml.safe_dump(
        {"name": name, "description": description},
        allow_unicode = True,
        sort_keys = False,
    )
    manifest = f"---\n{frontmatter}---\n\n{content}\n".encode("utf-8")
    # Validate before staging the durable intent so an intent can never promise
    # a manifest that the core Agent Skills resolver would reject.
    skills_module._parse_skill_markdown(manifest, name)
    return name, manifest


def _skill_path_redirected(path: Path) -> bool:
    """Detect a linked component below the current account's trusted base."""
    base = workspace_root() if not is_owner_context() else Path.home()
    try:
        relative = path.relative_to(base)
    except ValueError:
        return True
    current = base
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return True
    except OSError:
        return True
    return False


def _skill_creation_intent(proposal: dict[str, Any]) -> dict[str, Any]:
    name, manifest = _skill_manifest(proposal)
    target: LearningTarget = proposal.get("target", "codex")
    destinations = [str(root / name) for root in _skill_targets(target)]
    return {
        **dict(proposal),
        "accountId": current_account_id(),
        "destinations": destinations,
        "manifestSha256": hashlib.sha256(manifest).hexdigest(),
        "manifestBytes": len(manifest),
    }


def _same_proposal_content(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in (
            "kind",
            "title",
            "content",
            "reason",
            "name",
            "target",
            "sourceThreadId",
            "recommendationAction",
            "recommendationReason",
            "idempotencyKey",
        )
    )


def _intent_manifest_is_identical(
    intent: dict[str, Any], proposal: dict[str, Any]
) -> bool:
    """Accept replay only when every expected destination has the same bytes."""
    state, _, _ = _inspect_skill_intent(intent, proposal)
    return state == "complete"


def _inspect_skill_intent(
    intent: dict[str, Any], proposal: dict[str, Any]
) -> tuple[Literal["complete", "missing", "conflict"], list[Path], bytes]:
    """Validate an intent and classify destinations for an exact retry.

    An exact durable intent authorizes creating only destinations that are still
    absent. Existing destinations must contain exactly the expected manifest,
    with no extra entries and no linked path component.
    """
    from core.inference import skills as skills_module

    try:
        name, manifest = _skill_manifest(proposal)
        target: LearningTarget = proposal.get("target", "codex")
        expected_destinations = [str(root / name) for root in _skill_targets(target)]
        if intent.get("accountId") != current_account_id():
            return "conflict", [], manifest
        if intent.get("destinations") != expected_destinations:
            return "conflict", [], manifest
        if intent.get("manifestBytes") != len(manifest):
            return "conflict", [], manifest
        if intent.get("manifestSha256") != hashlib.sha256(manifest).hexdigest():
            return "conflict", [], manifest

        missing: list[Path] = []
        for raw_destination in expected_destinations:
            destination = Path(raw_destination)
            marker = destination / "SKILL.md"
            # resolve(strict=False) changes when any ancestor is a symlink.  Do
            # not let reconciliation follow one into another account or root.
            if _skill_path_redirected(destination):
                return "conflict", [], manifest
            if not destination.exists() and not destination.is_symlink():
                missing.append(destination)
                continue
            if _skill_path_redirected(marker):
                return "conflict", [], manifest
            if not destination.is_dir() or not marker.is_file():
                return "conflict", [], manifest
            if skills_module._is_linked_path(destination) or skills_module._is_linked_path(marker):
                return "conflict", [], manifest
            entries = list(destination.iterdir())
            if len(entries) != 1 or entries[0].name != "SKILL.md":
                return "conflict", [], manifest
            if marker.stat().st_size != len(manifest):
                return "conflict", [], manifest
            if marker.read_bytes() != manifest:
                return "conflict", [], manifest
    except (OSError, UnicodeError, ValueError, skills_module.SkillError):
        return "conflict", [], b""
    return ("missing" if missing else "complete"), missing, manifest


def _write_missing_skill_destinations(
    proposal: dict[str, Any], missing: list[Path], manifest: bytes
) -> None:
    """Create only absent destinations authorized by an exact durable intent."""
    name, _ = _skill_manifest(proposal)
    expected = {
        root / name
        for root in _skill_targets(proposal.get("target", "codex"))
    }
    if any(destination not in expected for destination in missing):
        raise HTTPException(
            status_code = 409,
            detail = "Autonomous skill promotion could not reconcile its durable intent.",
        )
    from core.inference import skills as skills_module

    for destination in missing:
        if (
            destination.exists()
            or destination.is_symlink()
            or _skill_path_redirected(destination)
        ):
            raise HTTPException(
                status_code = 409,
                detail = "Autonomous skill promotion could not reconcile its durable intent.",
            )
        root = destination.parent
        if root.name != "skills":
            raise HTTPException(
                status_code = 409,
                detail = "Autonomous skill promotion could not reconcile its durable intent.",
            )
        parent = root.parent
        base = parent.parent if parent.name.startswith(".") else parent
        relative_root = Path(parent.name) / "skills" if parent.name.startswith(".") else Path("skills")
        if _skill_path_redirected(base) or skills_module._is_linked_path(base):
            raise HTTPException(
                status_code = 409,
                detail = "Autonomous skill promotion could not reconcile its durable intent.",
            )
        try:
            base.mkdir(mode = 0o700, parents = True, exist_ok = True)
            skills_module._write_new_skill_manifest(
                base,
                name,
                manifest,
                root = relative_root,
            )
        except FileExistsError as error:
            raise HTTPException(
                status_code = 409,
                detail = "Autonomous skill promotion could not reconcile its durable intent.",
            ) from error
        except skills_module.SkillError as error:
            raise HTTPException(status_code = 400, detail = str(error)) from error
        except OSError as error:
            raise HTTPException(status_code = 400, detail = "Could not create approved skill.") from error


def _assert_new_skill_destinations(proposal: dict[str, Any]) -> None:
    """Refuse a new key when any destination is occupied or redirected."""
    name, _ = _skill_manifest(proposal)
    for root in _skill_targets(proposal.get("target", "codex")):
        destination = root / name
        if (
            destination.exists()
            or destination.is_symlink()
            or _skill_path_redirected(destination)
        ):
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")


def _has_conflicting_skill_intent(state: dict[str, Any], proposal: dict[str, Any]) -> bool:
    """Reject a second key targeting an outstanding durable promotion."""
    name, _ = _skill_manifest(proposal)
    destinations = {
        str(root / name)
        for root in _skill_targets(proposal.get("target", "codex"))
    }
    account = current_account_id()
    for item in state.get("skillCreationIntents", []):
        if not isinstance(item, dict) or item.get("accountId") != account:
            continue
        raw_destinations = item.get("destinations")
        if not isinstance(raw_destinations, list):
            continue
        if destinations.intersection(str(path) for path in raw_destinations):
            return True
    return False


def _append_skill_receipt(state: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(proposal)
    receipt["_autoApproved"] = True
    receipts = state.setdefault("proposalReceipts", [])
    receipts.append(receipt)
    state["proposalReceipts"] = receipts[-_PENDING_MAX:]
    state["skillCreationIntents"] = [
        item
        for item in state.get("skillCreationIntents", [])
        if not (
            isinstance(item, dict)
            and (
                item.get("id") == proposal.get("id")
                or (
                    item.get("idempotencyKey")
                    and item.get("idempotencyKey") == proposal.get("idempotencyKey")
                )
            )
            and item.get("accountId") == current_account_id()
        )
    ]
    return receipt


def _write_plain_skill(proposal: dict[str, Any]) -> str:
    name, manifest = _skill_manifest(proposal)
    description = str(proposal.get("title") or name).strip()[:500]
    content = str(proposal.get("content") or "").strip()
    target: LearningTarget = proposal.get("target", "codex")

    from core.inference import skills as skills_module

    destinations = [root / name for root in _skill_targets(target)]
    if any(destination.exists() or destination.is_symlink() for destination in destinations):
        raise HTTPException(status_code=409, detail=f"Skill destination already exists: {name}")
    try:
        # Managed accounts always resolve here to <workspace>/skills. Owners keep
        # the established ~/.agents target for codex/both.
        if not is_owner_context() or target in {"codex", "both"}:
            skills_module.create_skill(name, description, content)

        # The core catalog supports the owner's historical Claude root as a read
        # source. Use its contained, no-follow writer for that second ecosystem;
        # managed accounts never enter this host-home branch.
        if is_owner_context() and target in {"claude", "both"}:
            skills_module._write_new_skill_manifest(
                skills_module._owner_home(),
                name,
                manifest,
                root=Path(".claude") / "skills",
            )
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=f"Skill destination already exists: {name}") from error
    except skills_module.SkillError as error:
        status = 409 if "already exists" in str(error).lower() else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=400, detail="Could not create approved skill.") from error
    return name


def _persist_approved_memory(
    entry: dict[str, Any],
    *,
    subject: str,
    enabled: bool,
) -> dict[str, Any]:
    """Mirror an approved Hermes memory into Mem0 outside the learning lock."""
    from routes.memory import persist_memory_experience

    return persist_memory_experience(
        subject,
        _entry_text(entry),
        thread_id=str(entry.get("sourceThreadId") or "") or None,
        kind="hermes-memory",
        title=str(entry.get("title") or "Hermes memory")[:240],
        enabled=enabled,
        idempotency_key=str(entry.get("idempotencyKey") or "").strip() or None,
    )


@router.get("")
def get_learning(current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        return _public_state(_read_state())


@router.get("/context")
def get_learning_context(current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        return {"enabled": bool(state.get("enabled", True)), "instruction": _context_instruction(state)}


@router.post("/config")
def set_learning_config(payload: LearningConfigRequest, current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        if payload.enabled is not None:
            state["enabled"] = payload.enabled
        if payload.mem0Enabled is not None:
            state["mem0Enabled"] = payload.mem0Enabled
        if payload.onTheFlySkills is not None:
            state["onTheFlySkills"] = payload.onTheFlySkills
        for field in ("decisionMode", "allowSkillCreation", "allowQloraTraining", "allowRuntimeFix"):
            value = getattr(payload, field)
            if value is not None:
                state[field] = value
        _write_state(state)
        return _public_state(state)


@router.post("/proposals")
def create_learning_proposal(
    payload: LearningProposalRequest,
    current_subject: str = Depends(get_current_subject),
    *,
    autonomous_authorized: bool = True,
):
    title = payload.title.strip()
    content = payload.content.strip()
    if not title or not content:
        raise HTTPException(status_code = 400, detail = "A learning proposal needs a title and content.")
    skill_name: str | None = None
    if payload.kind == "skill":
        skill_name = _valid_skill_name(payload.name or title)
    idempotency_key = str(payload.idempotencyKey or "").strip() or None
    proposal = {
        "id": uuid.uuid4().hex,
        "kind": payload.kind,
        "title": title,
        "content": content,
        "reason": payload.reason.strip(),
        "name": skill_name if payload.kind == "skill" else None,
        "target": payload.target,
        "sourceThreadId": payload.sourceThreadId,
        "createdAt": int(time.time() * 1_000),
        "recommendationAction": payload.recommendationAction,
        "recommendationReason": payload.recommendationReason.strip(),
    }
    if idempotency_key:
        proposal["idempotencyKey"] = idempotency_key
    memory_to_persist: dict[str, Any] | None = None
    mem0_enabled = False
    with _LEARNING_LOCK:
        state = _read_state()
        if not state.get("enabled", True):
            raise HTTPException(status_code = 409, detail = "Hermes learning is disabled in the sidebar.")
        if idempotency_key:
            prior_bucket: str | None = None
            prior = None
            for bucket in (
                "memory",
                "user",
                "pending",
                "proposalReceipts",
                "skillCreationIntents",
            ):
                prior = next(
                    (
                        item
                        for item in state.get(bucket, [])
                        if isinstance(item, dict)
                        and str(item.get("idempotencyKey") or "").strip() == idempotency_key
                    ),
                    None,
                )
                if prior is not None:
                    prior_bucket = bucket
                    break
            if prior is not None:
                if not _same_proposal_content(prior, proposal):
                    raise HTTPException(
                        status_code=409,
                        detail="Learning proposal idempotency key is already bound to different content.",
                    )
                if prior_bucket == "skillCreationIntents":
                    intent_state, missing, manifest = _inspect_skill_intent(prior, proposal)
                    if intent_state == "conflict":
                        # The intent is durable, but an existing destination is
                        # changed or redirected. Never overwrite it during crash
                        # recovery.
                        raise HTTPException(
                            status_code=409,
                            detail="Autonomous skill promotion could not reconcile its durable intent.",
                        )
                    if intent_state == "missing":
                        _write_missing_skill_destinations(proposal, missing, manifest)
                        final_state, _, _ = _inspect_skill_intent(prior, proposal)
                        if final_state != "complete":
                            raise HTTPException(
                                status_code=409,
                                detail="Autonomous skill promotion could not reconcile its durable intent.",
                            )
                    reconciled = dict(proposal)
                    # A retry constructs a fresh request object, but the durable
                    # intent owns the original proposal id and receipt identity.
                    reconciled["id"] = prior.get("id") or proposal["id"]
                    receipt = _append_skill_receipt(state, reconciled)
                    _write_state(state)
                    return {
                        "proposal": receipt,
                        "autoApproved": True,
                        "idempotent": True,
                        **_public_state(state),
                    }
                return {
                    "proposal": prior,
                    "autoApproved": prior.get("_autoApproved") is True,
                    "idempotent": True,
                    **_public_state(state),
                }
        autonomous_skill = (
            proposal["kind"] == "skill"
            and state.get("decisionMode") == "autonomous"
            and state.get("allowSkillCreation") is True
            and autonomous_authorized
        )
        autonomous_memory = (
            proposal["kind"] == "memory"
            and state.get("decisionMode") == "autonomous"
            and autonomous_authorized
        )
        if autonomous_skill:
            # Persist the intent before touching the skill root. If the process
            # dies after the filesystem write but before the completion receipt,
            # the next same-key request can only accept byte-identical files.
            if _has_conflicting_skill_intent(state, proposal):
                raise HTTPException(
                    status_code=409,
                    detail="Autonomous skill promotion already has a durable intent.",
                )
            _assert_new_skill_destinations(proposal)
            intent = _skill_creation_intent(proposal)
            state.setdefault("skillCreationIntents", []).append(intent)
            state["skillCreationIntents"] = state["skillCreationIntents"][-_SKILL_INTENTS_MAX:]
            _write_state(state)
            _write_plain_skill(proposal)
            _append_skill_receipt(state, proposal)
        elif autonomous_memory:
            proposal["_autoApproved"] = True
            state["memory"] = _fit_entries(
                [*state.get("memory", []), proposal],
                _MEMORY_MAX_CHARS,
            )
            memory_to_persist = next(
                (dict(item) for item in state["memory"] if item.get("id") == proposal["id"]),
                dict(proposal),
            )
            mem0_enabled = state.get("mem0Enabled") is not False
        else:
            pending = state.setdefault("pending", [])
            pending.append(proposal)
            state["pending"] = pending[-_PENDING_MAX:]
        if payload.recommendationAction:
            state["lastRecommendation"] = {
                "action": payload.recommendationAction,
                "reason": payload.recommendationReason.strip(),
                "createdAt": proposal["createdAt"],
            }
        _write_state(state)
        result = {
            "proposal": proposal,
            "autoApproved": autonomous_skill or autonomous_memory,
            **_public_state(state),
        }
    if memory_to_persist is not None:
        result["memoryPersistence"] = _persist_approved_memory(
            memory_to_persist,
            subject=current_subject,
            enabled=mem0_enabled,
        )
    return result


@router.post("/proposals/{proposal_id}/approve")
def approve_learning_proposal(proposal_id: str, current_subject: str = Depends(get_current_subject)):
    memory_to_persist: dict[str, Any] | None = None
    mem0_enabled = False
    with _LEARNING_LOCK:
        state = _read_state()
        pending = state.get("pending", [])
        proposal = next((item for item in pending if item.get("id") == proposal_id), None)
        if proposal is None:
            raise HTTPException(status_code = 404, detail = "Learning proposal not found.")
        if proposal.get("kind") == "skill":
            _write_plain_skill(proposal)
        else:
            kind = "user" if proposal.get("kind") == "user" else "memory"
            state[kind] = _fit_entries(
                [*state.get(kind, []), proposal],
                _USER_MAX_CHARS if kind == "user" else _MEMORY_MAX_CHARS,
            )
            if kind == "memory":
                memory_to_persist = next(
                    (dict(item) for item in state["memory"] if item.get("id") == proposal_id),
                    dict(proposal),
                )
                mem0_enabled = state.get("mem0Enabled") is not False
        state["pending"] = [item for item in pending if item.get("id") != proposal_id]
        _write_state(state)
        result = _public_state(state)
    if memory_to_persist is not None:
        result["memoryPersistence"] = _persist_approved_memory(
            memory_to_persist,
            subject=current_subject,
            enabled=mem0_enabled,
        )
    return result


@router.post("/proposals/{proposal_id}/reject")
def reject_learning_proposal(proposal_id: str, current_subject: str = Depends(get_current_subject)):
    with _LEARNING_LOCK:
        state = _read_state()
        pending = state.get("pending", [])
        if not any(item.get("id") == proposal_id for item in pending):
            raise HTTPException(status_code = 404, detail = "Learning proposal not found.")
        state["pending"] = [item for item in pending if item.get("id") != proposal_id]
        _write_state(state)
        return _public_state(state)
