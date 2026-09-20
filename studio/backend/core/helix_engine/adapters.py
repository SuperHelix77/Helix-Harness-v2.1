# SPDX-License-Identifier: AGPL-3.0-only
"""Reversible QLoRA adapter records. Never auto-applied from success."""

from __future__ import annotations

from typing import Any


def adapter_record(
    *,
    version: str,
    dataset_hash: str,
    provenance: str,
    eval_delta: float,
    parent_version: str = "",
) -> dict[str, Any]:
    return {
        "version": version,
        "dataset_hash": dataset_hash,
        "provenance": provenance,
        "eval_delta": float(eval_delta),
        "parent_version": parent_version,
        "applied": False,
        "reversible": True,
    }


def rollback_adapter(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "rolled_back": current.get("version"),
        "active": None if previous is None else previous.get("version"),
        "reversible": True,
        "applied": False,
    }
