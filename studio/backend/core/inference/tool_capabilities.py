# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed, per-call capability grants for local execution tools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from utils.account_context import current_account_id


class ToolCapability(str, Enum):
    NETWORK_EGRESS = "network.egress"


_AUTHORITY = object()
_LOCAL_EXEC_TOOLS = frozenset({"python", "terminal"})


def _payload_sha256(payload: str) -> str:
    return hashlib.sha256(str(payload or "").encode("utf-8", errors="surrogatepass")).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolCapabilityGrant:
    """Opaque authority bound to one local execution request."""

    account_id: str
    tool_name: str
    payload_sha256: str
    capabilities: frozenset[ToolCapability]
    rationale: tuple[str, ...] = ()
    _authority: object = field(default=None, repr=False, compare=False)

    def allows(self, capability: ToolCapability) -> bool:
        return capability in self.capabilities


def issue_local_execution_grant(tool_name: str, payload: str) -> ToolCapabilityGrant:
    """Issue the minimum grant for one exact Python/terminal payload."""

    normalized = str(tool_name or "").strip().lower()
    if normalized not in _LOCAL_EXEC_TOOLS:
        raise ValueError(f"Unsupported local execution tool: {tool_name!r}")

    from core.inference import tools as tool_impl  # noqa: PLC0415

    if normalized == "python":
        requests_network = bool(tool_impl._python_requests_network(payload))
    else:
        requests_network = bool(tool_impl._terminal_requests_network(payload))

    capabilities = (
        frozenset({ToolCapability.NETWORK_EGRESS}) if requests_network else frozenset()
    )
    return ToolCapabilityGrant(
        account_id=current_account_id(),
        tool_name=normalized,
        payload_sha256=_payload_sha256(payload),
        capabilities=capabilities,
        rationale=("positive-network-classifier",) if requests_network else (),
        _authority=_AUTHORITY,
    )


def valid_local_execution_grant(
    grant: object,
    *,
    tool_name: str,
    payload: str,
) -> bool:
    """Whether ``grant`` was issued here for this exact actor/tool/payload."""

    if not isinstance(grant, ToolCapabilityGrant) or grant._authority is not _AUTHORITY:
        return False
    normalized = str(tool_name or "").strip().lower()
    return (
        normalized in _LOCAL_EXEC_TOOLS
        and grant.account_id == current_account_id()
        and grant.tool_name == normalized
        and grant.payload_sha256 == _payload_sha256(payload)
    )
