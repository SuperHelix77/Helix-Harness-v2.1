#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in baseline harness for the Phase 1 P1-RO-REPO tool scenario.

This is a control-only harness.  It deliberately does not implement an
ObservationPack, action fusion, candidate behavior, or a production route.  A
normal invocation is a read-only contract preflight.  ``--execute`` is the
only way to clone a runtime or contact its explicitly supplied loopback
server, and all execute-path tests use mocks.

The source-runtime runner is the lifecycle authority for APFS cloning,
process identity fencing, cleanup sampling, and offline model validation.  The
tool scenario adds only the terminal request, content-addressed fixture, and a
standalone SQLite read-only verifier.  The verifier never treats chat/SSE as
execution evidence: proposals, durable claim/start/finish events, result
hashes, trace, and the final structured answer must all be present in SQLite.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import datetime as _datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import urllib.parse


SCRIPT_VERSION = "helix-v3-tool-scenarios.v2"
BENCHMARK_SCHEMA = "helix.optimization-benchmark.v1"
SCENARIO_ID = "P1-RO-REPO"
FIXTURE_ID = "p1-ro-repo-v1"
FIXTURE_SCHEMA = "helix.p1-ro-repo-fixture.v1"
FIXTURE_MANIFEST_NAME = "manifest.json"
TOOL_NAME = "terminal"
TRACE_EVENT_TYPE = "optimization.trace"
TOOL_APPROVAL_SCHEMA = "helix.tool-approval.v2"
TOOL_EXECUTION_SCHEMA = "helix.tool-execution.v3"
CHECKPOINT_VERSION = 1
EXPECTED_BACKEND_CONTRACT = "helix.adaptive.backend.v1"
OWNER_USERNAME = "unsloth"
OWNER_ACCOUNT_ID = "owner"
DEFAULT_PROVIDER_TIER = "local"
DEFAULT_PROVIDER_IDENTITY = "local:mlx"
DEFAULT_MODEL_NAME = "local"
DEFAULT_SEED = 20260920
DEFAULT_MAX_TOKENS = 512
COMMAND_TIMEOUT_S = 5.0
CLONE_COMMAND_TIMEOUT_S = 120.0
MAX_SSE_EVENTS = 10_000
MAX_SSE_WIRE_BYTES = 4 * 1024 * 1024
MAX_SSE_LINE_BYTES = 256 * 1024
MAX_SSE_BLOCK_BYTES = 1 * 1024 * 1024
MAX_SSE_JSON_DEPTH = 32

DEFAULT_PROMPT = (
    "Use only the built-in terminal tool. In this exact order, run exactly these "
    "three read-only commands from the fixture root: (1) grep -n -- \"signal:\" "
    "README.md; (2) sed -n '1p' src/alpha.txt; (3) head -n 1 "
    "diagnostics/metrics.txt. Use only relative fixture paths. Do not write, "
    "mutate, connect to a network, invoke MCP, or use compound shell syntax. "
    "Return only one JSON object without Markdown fences, commentary, or any "
    "text before or after it, with exactly these keys in this exact order and "
    "no others: alpha (the value after alpha= in src/alpha.txt), fixture_id "
    "(the string p1-ro-repo-v1), read_count (the integer 3), records (the "
    "integer after records= in diagnostics/metrics.txt), and signals (the "
    "signal names read from README.md). Do not return hashes or a status field."
)

# These are intentionally duplicated as fixed constants rather than learned from
# the fixture manifest.  A caller cannot replace the manifest with a different,
# self-consistent tree and thereby redefine the benchmark contract.
CANONICAL_FIXTURE_SHA256 = "e9d0c0ba285ee3c925c5a55326d9b612cf6e83c7f7ac0868dd368f218e8cab2d"
CANONICAL_MANIFEST_SHA256 = "e19acbebe03bce183f34e40f99cef93dff47c3b051b9a7a2c89d212821f2790c"
CANONICAL_FILES: tuple[Mapping[str, Any], ...] = (
    {"path": "README.md", "length": 126, "sha256": "3901af1fd01e4595ca01613eb162ce13af49160730ffa45bce6bdb6b02041d8b"},
    {"path": "diagnostics/metrics.txt", "length": 24, "sha256": "5a0a169447b9a5e7c3d0f60bd0b9d5766d4f45de3ce0f2f96ac408da9578df03"},
    {"path": "src/alpha.txt", "length": 23, "sha256": "a6a01fc20ee2cde792226e570f25e3b17dc7e25222f052c6a6305bd89c6bde4e"},
)
CANONICAL_SEQUENCE: tuple[Mapping[str, Any], ...] = (
    {
        "tool_name": "terminal",
        "command": 'grep -n -- "signal:" README.md',
        "arguments_sha256": "acef02f2215d9b03a04de6a0e6cbf6e6020078e808d00cc9b56c74d057f6e7c6",
        "proposal_payload_sha256": "5ca86d7adb2bc53613e390ec0285ca49a4c3d97206c906830ea1fdbf10824804",
        "result_sha256": "b8e4c1f4561adfa988fadf4a948cb3a38da735321ac566dc24470fa09ad0d19e",
        "result_length": 31,
        "expected_paths": ["README.md"],
        "byte_ranges": [{"path": "README.md", "start": 0, "end": 126}],
    },
    {
        "tool_name": "terminal",
        "command": "sed -n '1p' src/alpha.txt",
        "arguments_sha256": "276c7b0a3c80374f236b1de715ffb95fa91ec3636f0da6d4b17aa9265f99faee",
        "proposal_payload_sha256": "ef03a2c21e99514c7ab7196dced60aa4542f93ccb0e97d3c30470c9f0de977bc",
        "result_sha256": "350cf5c40ae045124719b5b5f09cf859a009e058e0e10bd39c512011e5f966f6",
        "result_length": 10,
        "expected_paths": ["src/alpha.txt"],
        "byte_ranges": [{"path": "src/alpha.txt", "start": 0, "end": 10}],
    },
    {
        "tool_name": "terminal",
        "command": "head -n 1 diagnostics/metrics.txt",
        "arguments_sha256": "3478fd5fc869dfe041d87b764af111c6d60d980f48e72601f1f383d3d9d5f6a7",
        "proposal_payload_sha256": "560dfacfc9a18161bc205a5ba8c5d42e3a675f02b24082ce7557487dcd8f8e98",
        "result_sha256": "12623e7c080d711f697912b8bf9f455e2252bb8a1eb292ed0dc8cac7b0974c92",
        "result_length": 10,
        "expected_paths": ["diagnostics/metrics.txt"],
        "byte_ranges": [{"path": "diagnostics/metrics.txt", "start": 0, "end": 10}],
    },
)
CANONICAL_FINAL_ANSWER: Mapping[str, Any] = {
    "alpha": "one",
    "fixture_id": FIXTURE_ID,
    "read_count": 3,
    "records": 3,
    "signals": ["alpha", "beta"],
}
CANONICAL_FINAL_ANSWER_SHA256 = "947382cd26a4e46b94f0c8871559626a6ba3db7c184e32c93efdfcec32079c3e"
CANONICAL_FINAL_ANSWER_LENGTH = 98

ACCEPTANCE_CRITERIA: dict[str, Any] = {
    "scenario_id": SCENARIO_ID,
    "fixture_id": FIXTURE_ID,
    "tool_name": TOOL_NAME,
    "request": {
        "enable_tools": True,
        "enabled_tools": [TOOL_NAME],
        "mcp_enabled": False,
        "bypass_permissions": False,
        "permission_mode": "auto",
        "confirm_tool_calls": True,
    },
    "sequence": "manifest.expected_sequence_exact_order",
    "command_policy": {
        "relative_fixture_paths_only": True,
        "single_read_only_command": True,
        "compound_shell_syntax": False,
        "mutation": False,
        "network": False,
    },
    "durable_evidence": [
        "objective_identity",
        "request_payload_sha256",
        "public_tool_proposal_arguments_sha256",
        "public_tool_proposal_payload_sha256",
        "tool_execution_claimed",
        "tool_execution_started",
        "tool_execution_finished",
        "result_sha256_and_length",
        "optimization_trace",
        "final_structured_answer",
        "zero_fixture_mutation",
    ],
}


class HarnessError(RuntimeError):
    """Base class for fail-closed tool-scenario errors."""


class ContractError(HarnessError):
    """An explicit path, request, fixture, or command contract failed."""


class IdentityMismatchError(ContractError):
    """A source/runtime/server identity was not proven."""


class ProtocolMismatchError(ContractError):
    """The public or durable protocol was incomplete or inconsistent."""


class AccountBindingError(ContractError):
    """Durable rows crossed the owner/run/thread boundary."""


@dataclass(frozen=True)
class ToolScenarioConfig:
    runtime_home_template: Path
    run_dir: Path
    output_dir: Path
    repo_root: Path
    fixture_root: Path
    password_file: Path | None = None
    model_name: str = DEFAULT_MODEL_NAME
    model_path: Path | None = None
    model_config_path: Path | None = None
    prompt: str = DEFAULT_PROMPT
    provider_tier: str = DEFAULT_PROVIDER_TIER
    provider_identity: str = DEFAULT_PROVIDER_IDENTITY
    startup_timeout_s: float = 60.0
    request_timeout_s: float = 300.0
    seed: int = DEFAULT_SEED
    max_tokens: int = DEFAULT_MAX_TOKENS
    execute: bool = False


@dataclass(frozen=True)
class FixtureSpec:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    fixture_sha256: str
    files: tuple[Mapping[str, Any], ...]
    expected_sequence: tuple[Mapping[str, Any], ...]
    final_answer_sha256: str
    final_answer_length: int


def _load_source_runner() -> Any:
    """Load the approved source runner without importing backend modules."""

    path = Path(__file__).resolve().with_name("benchmark-helix-v3-source-runtime.py")
    spec = importlib.util.spec_from_file_location("helix_v3_source_runtime_for_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError("approved source runner could not be loaded")
    module = importlib.util.module_from_spec(spec)
    # Keep this module-local: callers can import both runners without replacing
    # a package or backend module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_runner()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_user_message_content(prompt: str = DEFAULT_PROMPT) -> list[dict[str, str]]:
    return [{"type": "text", "text": prompt}]


def _require_pinned_prompt(config: ToolScenarioConfig) -> None:
    if not isinstance(config.prompt, str) or config.prompt != DEFAULT_PROMPT:
        raise ContractError("tool scenario prompt must equal the pinned DEFAULT_PROMPT")


def _prompt_content_sha256(prompt: str = DEFAULT_PROMPT) -> str:
    return _sha256_json(_canonical_user_message_content(prompt))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _require_abs_directory(path: Path, label: str, *, exists: bool = False) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    if requested.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    resolved = requested.resolve(strict=False)
    if exists and not resolved.is_dir():
        raise ContractError(f"{label} does not exist")
    if resolved.exists() and not resolved.is_dir():
        raise ContractError(f"{label} is not a directory")
    return resolved


def _require_regular_file(path: Path, label: str, *, exists: bool = True) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    if requested.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    resolved = requested.resolve(strict=False)
    if exists and (not resolved.is_file() or resolved.is_symlink()):
        raise ContractError(f"{label} is not a regular file")
    return resolved


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return true when either path contains the other (including equality)."""

    return _within(first, second) or _within(second, first)


def _source_config(config: ToolScenarioConfig, *, execute: bool | None = None) -> Any:
    """Build the approved runner's immutable config for shared preflight helpers."""

    _require_pinned_prompt(config)
    return _SOURCE.RuntimeConfig(
        runtime_home_template=config.runtime_home_template,
        run_dir=config.run_dir,
        output_dir=config.output_dir,
        repo_root=config.repo_root,
        password_file=config.password_file,
        model_name=config.model_name,
        model_path=config.model_path,
        model_config_path=config.model_config_path,
        prompt=config.prompt,
        provider_tier=config.provider_tier,
        provider_identity=config.provider_identity,
        startup_timeout_s=config.startup_timeout_s,
        request_timeout_s=config.request_timeout_s,
        seed=config.seed,
        max_tokens=config.max_tokens,
        execute=config.execute if execute is None else execute,
    )


def _safe_relative_path(raw: Any, *, label: str = "path") -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ProtocolMismatchError(f"{label} was not a relative path")
    if "\\" in raw:
        raise ProtocolMismatchError(f"{label} used a non-POSIX path")
    candidate = Path(raw)
    if candidate.is_absolute() or raw.startswith("~"):
        raise ProtocolMismatchError(f"{label} escaped the fixture")
    parts = PurePathParts(raw)
    if not parts or any(item in {"", ".", ".."} for item in parts):
        raise ProtocolMismatchError(f"{label} escaped the fixture")
    return "/".join(parts)


def PurePathParts(raw: str) -> tuple[str, ...]:
    # ``Path.parts`` on macOS is platform-dependent for separators; the fixture
    # contract is explicitly POSIX and rejects backslashes above.
    return tuple(part for part in raw.split("/") if part)


def _fixture_records(root: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except OSError as exc:
        raise ContractError("fixture could not be inspected") from exc
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        if relative == FIXTURE_MANIFEST_NAME:
            continue
        try:
            if entry.is_symlink():
                raise ContractError("fixture may not contain symlinks")
            if entry.is_dir():
                continue
            if not entry.is_file():
                raise ContractError("fixture contains an unsupported filesystem entry")
            data = entry.read_bytes()
        except OSError as exc:
            raise ContractError("fixture entry could not be read") from exc
        records.append({"path": relative, "length": len(data), "sha256": _sha256_bytes(data)})
    return tuple(records)


def fixture_tree_hash(root: Path) -> str:
    """Hash relative paths plus content hashes; the manifest is excluded."""

    digest = hashlib.sha256()
    for item in _fixture_records(root):
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_answer_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    answer = manifest.get("final_answer")
    if answer != CANONICAL_FINAL_ANSWER:
        raise ContractError("fixture final answer contract was not the pinned answer")
    return dict(CANONICAL_FINAL_ANSWER)


def load_fixture(root: Path) -> FixtureSpec:
    root = _require_abs_directory(root, "fixture root", exists=True)
    manifest_path = _require_regular_file(root / FIXTURE_MANIFEST_NAME, "fixture manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ContractError("fixture manifest was not the pinned manifest")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("fixture manifest could not be parsed") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != FIXTURE_SCHEMA or manifest.get("fixture_id") != FIXTURE_ID:
        raise ContractError("fixture manifest identity was invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("fixture manifest had no files")
    expected_files: list[dict[str, Any]] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ContractError("fixture manifest file record was invalid")
        path = _safe_relative_path(raw.get("path"), label="fixture manifest path")
        digest = raw.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError("fixture manifest file hash was invalid")
        length = raw.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ContractError("fixture manifest file length was invalid")
        expected_files.append({"path": path, "length": length, "sha256": digest})
    if tuple(expected_files) != tuple(dict(item) for item in CANONICAL_FILES):
        raise ContractError("fixture manifest files were not the pinned files")
    if tuple(expected_files) != _fixture_records(root):
        raise ContractError("fixture content did not match its manifest")
    expected_fixture_sha = manifest.get("fixture_sha256")
    if expected_fixture_sha != CANONICAL_FIXTURE_SHA256 or expected_fixture_sha != fixture_tree_hash(root):
        raise ContractError("fixture tree hash did not match its manifest")
    sequence = manifest.get("expected_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ContractError("fixture command sequence was missing")
    if tuple(sequence) != tuple(dict(item) for item in CANONICAL_SEQUENCE):
        raise ContractError("fixture command sequence was not the pinned sequence")
    checked_sequence: list[dict[str, Any]] = []
    declared_by_path = {str(item["path"]): item for item in expected_files}
    for raw in sequence:
        if not isinstance(raw, Mapping) or raw.get("tool_name") != TOOL_NAME:
            raise ContractError("fixture command tool was not terminal")
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ContractError("fixture command was empty")
        args = {"command": command}
        if raw.get("arguments_sha256") != _sha256_json(args):
            raise ContractError("fixture command argument hash was invalid")
        proposal = {"tool_name": TOOL_NAME, "arguments": args}
        if raw.get("proposal_payload_sha256") != _sha256_json(proposal):
            raise ContractError("fixture command proposal hash was invalid")
        for field_name in ("result_sha256",):
            if not isinstance(raw.get(field_name), str) or not re.fullmatch(r"[0-9a-f]{64}", str(raw[field_name])):
                raise ContractError("fixture command result hash was invalid")
        if isinstance(raw.get("result_length"), bool) or not isinstance(raw.get("result_length"), int) or int(raw["result_length"]) < 0:
            raise ContractError("fixture command result length was invalid")
        paths = raw.get("expected_paths")
        ranges = raw.get("byte_ranges")
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            raise ContractError("fixture command expected paths were invalid")
        expected_path = _safe_relative_path(paths[0], label="fixture command expected path")
        declared = declared_by_path.get(expected_path)
        if declared is None:
            raise ContractError("fixture command path was not declared")
        if not isinstance(ranges, list) or len(ranges) != 1 or not isinstance(ranges[0], Mapping):
            raise ContractError("fixture command byte range was invalid")
        byte_range = ranges[0]
        if byte_range.get("path") != expected_path or byte_range.get("start") != 0:
            raise ContractError("fixture command byte range path/start was invalid")
        end = byte_range.get("end")
        if isinstance(end, bool) or not isinstance(end, int) or end <= 0 or end > int(declared["length"]):
            raise ContractError("fixture command byte range exceeded the declared file")
        if int(raw["result_length"]) > end:
            raise ContractError("fixture result length exceeded its declared byte range")
        # The exact command itself is checked against the fixed policy.
        _validate_command_text(command, expected=command)
        checked_sequence.append(dict(raw))
    expected_answer = _expected_answer_from_manifest(manifest)
    answer_bytes = _canonical_json(expected_answer).encode("utf-8")
    if manifest.get("final_answer_sha256") != CANONICAL_FINAL_ANSWER_SHA256 or manifest.get("final_answer_length") != CANONICAL_FINAL_ANSWER_LENGTH or manifest.get("final_answer_sha256") != _sha256_bytes(answer_bytes) or manifest.get("final_answer_length") != len(answer_bytes):
        raise ContractError("fixture final-answer identity was invalid")
    return FixtureSpec(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        fixture_sha256=str(expected_fixture_sha),
        files=tuple(expected_files),
        expected_sequence=tuple(checked_sequence),
        final_answer_sha256=str(manifest["final_answer_sha256"]),
        final_answer_length=int(manifest["final_answer_length"]),
    )


_COMPOUND_CHARS = frozenset(";&|><`\n\r\x00")
_MUTATING_WORDS = frozenset(
    {
        "rm", "mv", "cp", "touch", "mkdir", "rmdir", "truncate", "tee", "dd", "chmod",
        "chown", "sed-i", "git", "curl", "wget", "nc", "ssh", "scp", "rsync", "python",
        "python3", "pip", "npm", "env", "sudo", "launchctl", "open", "xargs", "find",
    }
)
_NETWORK_WORDS = frozenset({"curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "ftp", "telnet", "git"})


def _validate_command_text(command: str, *, expected: str | None = None) -> tuple[str, ...]:
    if not isinstance(command, str) or not command.strip():
        raise ProtocolMismatchError("terminal command was empty")
    if any(char in command for char in _COMPOUND_CHARS) or "$" in command:
        raise ProtocolMismatchError("terminal command used compound shell syntax")
    if ".." in command or "//" in command or "./" in command:
        raise ProtocolMismatchError("terminal command escaped the fixture")
    try:
        tokens = tuple(shlex.split(command, posix=True))
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatchError("terminal command could not be parsed") from exc
    if not tokens:
        raise ProtocolMismatchError("terminal command was empty")
    executable = tokens[0]
    if executable not in {"grep", "sed", "head"}:
        if executable in _NETWORK_WORDS:
            raise ProtocolMismatchError("terminal command requested network access")
        if executable in _MUTATING_WORDS:
            raise ProtocolMismatchError("terminal command requested mutation")
        raise ProtocolMismatchError("terminal command was not allowlisted")
    for token in tokens:
        if token.startswith(("/", "~")) or "\\" in token:
            raise ProtocolMismatchError("terminal command used an absolute or non-POSIX path")
        if token.casefold() in _MUTATING_WORDS or token.casefold() in _NETWORK_WORDS:
            raise ProtocolMismatchError("terminal command requested a forbidden operation")
        if token.startswith(("--pre", "--passthrough", "--type-add", "--type-clear", "--glob", "--iglob", "--sort", "--debug")):
            raise ProtocolMismatchError("rg executable hook or unsafe option was forbidden")
    canonical_commands = {str(item["command"]) for item in CANONICAL_SEQUENCE}
    if command not in canonical_commands:
        raise ProtocolMismatchError("terminal command was not in the fixed allowlist")
    if executable == "grep" and tokens != ("grep", "-n", "--", "signal:", "README.md"):
        raise ProtocolMismatchError("grep command was not the bounded read-only form")
    if executable == "sed" and tokens != ("sed", "-n", "1p", "src/alpha.txt"):
        raise ProtocolMismatchError("sed command was not the bounded read-only form")
    if executable == "head" and tokens != ("head", "-n", "1", "diagnostics/metrics.txt"):
        raise ProtocolMismatchError("head command was not the bounded read-only form")
    if expected is not None:
        try:
            expected_tokens = tuple(shlex.split(expected, posix=True))
        except (TypeError, ValueError) as exc:
            raise ContractError("fixture command policy was invalid") from exc
        if tokens != expected_tokens or command != expected:
            raise ProtocolMismatchError("terminal command was not the expected allowlisted command")
    return tokens


def validate_tool_proposal(
    tool_name: Any,
    arguments: Any,
    fixture: FixtureSpec,
    *,
    sequence_index: int | None = None,
) -> dict[str, str]:
    """Validate one public terminal proposal and return hash-only identities."""

    if tool_name != TOOL_NAME:
        raise ProtocolMismatchError("tool proposal was not the built-in terminal")
    if not isinstance(arguments, Mapping) or set(arguments) != {"command"} or not isinstance(arguments.get("command"), str):
        raise ProtocolMismatchError("terminal proposal arguments were not canonical")
    index = len(fixture.expected_sequence) if sequence_index is None else sequence_index
    if index < 0 or index >= len(fixture.expected_sequence):
        raise ProtocolMismatchError("terminal proposal sequence exceeded the fixture")
    expected = fixture.expected_sequence[index]
    command = str(arguments["command"])
    _validate_command_text(command, expected=str(expected["command"]))
    args = {"command": command}
    proposal = {"tool_name": TOOL_NAME, "arguments": args}
    return {
        "tool_name": TOOL_NAME,
        "arguments_sha256": _sha256_json(args),
        "proposal_payload_sha256": _sha256_json(proposal),
        "command_length": str(len(command)),
    }


def terminal_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Execute one read-only fixture command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def acceptance_criteria_sha256() -> str:
    return _sha256_json(ACCEPTANCE_CRITERIA)


def _path_identity_summary(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=False)
    return {
        "absolute": resolved.is_absolute(),
        "symlink": Path(path).is_symlink(),
        "path_sha256": _sha256_bytes(str(resolved).encode("utf-8")),
    }


def _assert_no_symlink_components(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ContractError(f"{label} was not absolute")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label} contained a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ContractError(f"{label} was not a directory")
    return resolved


def _project_payload(project_id: str, stamp: int) -> dict[str, Any]:
    return {
        "id": project_id,
        "name": "Helix v3 P1 read-only repository benchmark",
        "instructions": "Fixed read-only benchmark fixture; no writes or network access.",
        "archived": False,
        "createdAt": stamp,
        "updatedAt": stamp,
    }


def create_managed_project(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    clone_run_dir: Path,
    *,
    clock: Callable[[], float] = time.time,
    timeout: float,
) -> dict[str, Any]:
    """Create and validate the project workspace owned by this clone."""

    stamp = int(clock() * 1000)
    project_id = f"helix-v3-p1-ro-repo-project-{os.getpid()}-{stamp}"
    response = _json_request(
        transport,
        "POST",
        base_url,
        "/api/chat/projects",
        headers=headers,
        payload=_project_payload(project_id, stamp),
        timeout=timeout,
    )
    if not isinstance(response, Mapping) or response.get("id") != project_id:
        raise AccountBindingError("managed project identity was not persisted")
    clone_root = Path(clone_run_dir).resolve(strict=True)
    root_raw = response.get("rootPath")
    sandbox_raw = response.get("sandboxPath")
    if not isinstance(root_raw, str) or not isinstance(sandbox_raw, str):
        raise ContractError("managed project did not return rootPath and sandboxPath")
    root = _assert_no_symlink_components(Path(root_raw), "project rootPath")
    sandbox = _assert_no_symlink_components(Path(sandbox_raw), "project sandboxPath")
    if not _within(root, clone_root) or not _within(sandbox, clone_root) or not _within(sandbox, root):
        raise ContractError("managed project workspace escaped the runtime clone")
    if any(sandbox.iterdir()):
        raise ContractError("managed project sandbox was not empty")
    return {
        "project_id": project_id,
        "root": root,
        "sandbox": sandbox,
        "identity": {
            "project_id": project_id,
            "root": _path_identity_summary(root),
            "sandbox": _path_identity_summary(sandbox),
        },
    }


def stage_fixture_in_project(project: Mapping[str, Any], fixture: FixtureSpec) -> dict[str, Any]:
    sandbox = Path(project["sandbox"])
    if any(sandbox.iterdir()):
        raise ContractError("managed project sandbox was not empty before staging")
    shutil.copytree(fixture.root, sandbox, dirs_exist_ok=True, symlinks=False)
    staged = load_fixture(sandbox)
    if staged.fixture_sha256 != CANONICAL_FIXTURE_SHA256 or staged.manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ContractError("staged fixture did not match the pinned fixture")
    for entry in sorted(sandbox.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if entry.is_symlink():
            raise ContractError("staged fixture contained a symlink")
        mode = stat.S_IMODE(entry.stat().st_mode)
        if entry.is_dir():
            entry.chmod(mode & ~0o222)
        elif entry.is_file():
            entry.chmod(mode & ~0o222)
        else:
            raise ContractError("staged fixture contained an unsupported entry")
    # The sandbox itself is the tool work directory and must remain writable;
    # every copied child is immutable for the duration of the run.
    sandbox.chmod(stat.S_IMODE(sandbox.stat().st_mode) | 0o700)
    return {
        "fixture_sha256": staged.fixture_sha256,
        "manifest_sha256": staged.manifest_sha256,
        "sandbox": _path_identity_summary(sandbox),
        "files": len(staged.files),
    }


def verify_staged_fixture(project: Mapping[str, Any], fixture: FixtureSpec) -> dict[str, Any]:
    staged = load_fixture(Path(project["sandbox"]))
    if staged.fixture_sha256 != fixture.fixture_sha256 or staged.manifest_sha256 != fixture.manifest_sha256:
        raise ContractError("staged project fixture changed")
    return {
        "fixture_sha256": staged.fixture_sha256,
        "manifest_sha256": staged.manifest_sha256,
        "sandbox": _path_identity_summary(Path(project["sandbox"])),
        "files": len(staged.files),
    }


def prepare_staged_fixture_for_cleanup(project: Mapping[str, Any], fixture: FixtureSpec) -> dict[str, Any]:
    """Restore owner-write permission only after the staged fixture is proven exact.

    The benchmark deliberately removes write permission from copied fixture
    directories. ``shutil.rmtree`` cannot unlink their children on POSIX until
    the directory owner regains write/search permission. Change only the exact
    manifest-derived directories inside the disposable runtime clone; the
    authoritative fixture and its file contents remain untouched.
    """

    sandbox = _assert_no_symlink_components(Path(project["sandbox"]), "project sandboxPath")
    staged = load_fixture(sandbox)
    if staged.fixture_sha256 != fixture.fixture_sha256 or staged.manifest_sha256 != fixture.manifest_sha256:
        raise ContractError("staged project fixture changed before cleanup")
    directories = {sandbox}
    for item in fixture.files:
        parent = Path(str(item["path"])).parent
        while parent != Path("."):
            directories.add(sandbox / parent)
            parent = parent.parent
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if directory.is_symlink() or not directory.is_dir() or not _within(directory.resolve(strict=True), sandbox):
            raise ContractError("staged fixture cleanup directory identity changed")
        directory.chmod(stat.S_IMODE(directory.stat().st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return {"directories_restored": len(directories), "fixture_sha256": staged.fixture_sha256}


def build_tool_request_payload(
    config: ToolScenarioConfig,
    run_id: str,
    thread_id: str,
    fixture: FixtureSpec | None = None,
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Build the exact baseline request; only the built-in terminal is offered."""

    _require_pinned_prompt(config)
    if fixture is None:
        fixture = load_fixture(config.fixture_root)
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": config.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "tools": [terminal_tool_schema()],
        "tool_choice": "auto",
        "enable_tools": True,
        "enabled_tools": [TOOL_NAME],
        "mcp_enabled": False,
        "bypass_permissions": False,
        "permission_mode": "auto",
        "confirm_tool_calls": True,
        "auto_heal_tool_calls": False,
        "nudge_tool_calls": False,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "thread_id": thread_id,
        "cancel_id": run_id,
    }
    if project_id is not None:
        if not isinstance(project_id, str) or not project_id:
            raise ContractError("project id was invalid")
        payload["session_id"] = f"project-{project_id}"
    return payload


def summarize_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    prompt = "".join(
        str(message.get("content"))
        for message in messages or ()
        if isinstance(message, Mapping) and message.get("role") == "user" and isinstance(message.get("content"), str)
    ) if isinstance(messages, list) else ""
    tools = payload.get("tools")
    return {
        "request_sha256": _sha256_json(payload),
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "prompt_content_sha256": _prompt_content_sha256(prompt) if prompt else None,
        "prompt_length": len(prompt),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tool_schema_sha256": _sha256_json(tools),
        "tool_names": [
            item.get("function", {}).get("name")
            for item in tools or ()
            if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
        ],
        "model": payload.get("model") if isinstance(payload.get("model"), str) else None,
        "stream": payload.get("stream") is True,
        "enable_tools": payload.get("enable_tools") is True,
        "enabled_tools": list(payload.get("enabled_tools") or ()),
        "mcp_enabled": payload.get("mcp_enabled") is True,
        "bypass_permissions": payload.get("bypass_permissions") is True,
        "permission_mode": payload.get("permission_mode"),
        "confirm_tool_calls": payload.get("confirm_tool_calls") is True,
        "session_id": payload.get("session_id"),
    }


def _validate_pinned_request_payload(payload: Mapping[str, Any], *, label: str = "request") -> None:
    messages = payload.get("messages")
    expected = _canonical_user_message_content()
    if messages != [{"role": "user", "content": DEFAULT_PROMPT}]:
        raise ProtocolMismatchError(f"{label} did not contain the pinned user prompt")
    if _sha256_bytes(DEFAULT_PROMPT.encode("utf-8")) != _sha256_bytes(
        str(messages[0]["content"]).encode("utf-8")
    ):
        raise ProtocolMismatchError(f"{label} user prompt hash did not match")
    if _prompt_content_sha256() != _sha256_json(expected):
        raise ProtocolMismatchError(f"{label} user message content hash could not be bound")


def _durable_request_hash(objective: Mapping[str, str], request_payload: Mapping[str, Any]) -> str:
    identity = {
        "threadId": objective["thread_id"],
        "userMessageId": objective["user_message_id"],
        "assistantMessageId": objective["assistant_message_id"],
        "requestPayload": dict(request_payload),
    }
    return _sha256_json(identity)


def _json_request(transport: Any, method: str, base_url: str, path: str, *, headers: Mapping[str, str], payload: Any = None, timeout: float) -> Any:
    return _SOURCE._json_request(transport, method, base_url, path, headers=headers, payload=payload, timeout=timeout)


def _auth_headers(access_token: Any) -> dict[str, str]:
    return _SOURCE._auth_headers(access_token)


def login_owner(transport: Any, base_url: str, password: str, timeout: float) -> tuple[dict[str, str], dict[str, Any]]:
    result = _SOURCE.login_owner(transport, base_url, password, timeout)
    return result


def create_tool_objective(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    config: ToolScenarioConfig,
    fixture: FixtureSpec,
    project: Mapping[str, Any],
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    _require_pinned_prompt(config)
    stamp = int(clock() * 1000)
    suffix = f"{os.getpid()}_{stamp}"
    thread_id = f"helix-v3-p1-ro-repo-thread-{suffix}"
    user_message_id = f"helix-v3-p1-ro-repo-user-{suffix}"
    assistant_message_id = f"helix-v3-p1-ro-repo-assistant-{suffix}"
    run_id = f"helix-v3-p1-ro-repo-run-{suffix}"
    thread = _json_request(
        transport, "POST", base_url, "/api/chat/threads", headers=headers,
        payload={"id": thread_id, "title": "Helix v3 P1 read-only repository benchmark", "modelType": "base", "modelId": config.model_name, "projectId": project["project_id"], "createdAt": stamp},
        timeout=config.request_timeout_s,
    )
    if not isinstance(thread, Mapping) or thread.get("id") != thread_id or thread.get("projectId") != project["project_id"]:
        raise AccountBindingError("thread identity was not persisted")
    message_path = f"/api/chat/threads/{urllib.parse.quote(thread_id, safe='')}/messages/{urllib.parse.quote(user_message_id, safe='')}"
    message = _json_request(
        transport, "PUT", base_url, message_path, headers=headers,
        payload={"id": user_message_id, "threadId": thread_id, "parentId": None, "role": "user", "content": _canonical_user_message_content(), "createdAt": stamp},
        timeout=config.request_timeout_s,
    )
    if not isinstance(message, Mapping) or message.get("id") != user_message_id or message.get("threadId") != thread_id or message.get("role") != "user":
        raise AccountBindingError("user message identity was not persisted")
    expected_content = _canonical_user_message_content()
    if message.get("content") != expected_content:
        raise ProtocolMismatchError("stored user message content did not match the pinned prompt")
    for field_name in ("contentSha256", "content_sha256"):
        if field_name in message and message.get(field_name) != _prompt_content_sha256():
            raise ProtocolMismatchError("stored user message content hash did not match")
    request_payload = build_tool_request_payload(config, run_id, thread_id, fixture, project_id=str(project["project_id"]))
    run = _json_request(
        transport, "POST", base_url, "/api/inference/chat-runs", headers=headers,
        payload={"runId": run_id, "threadId": thread_id, "userMessageId": user_message_id, "assistantMessageId": assistant_message_id, "requestPayload": request_payload},
        timeout=config.request_timeout_s,
    )
    if not isinstance(run, Mapping):
        raise AccountBindingError("durable run was not persisted")
    for key, expected in (("id", run_id), ("threadId", thread_id), ("userMessageId", user_message_id), ("assistantMessageId", assistant_message_id)):
        if run.get(key) != expected:
            raise AccountBindingError("durable run identity was not persisted")
    stored_payload = run.get("requestPayload")
    if not isinstance(stored_payload, Mapping):
        raise AccountBindingError("durable create response omitted requestPayload")
    _validate_pinned_request_payload(stored_payload, label="durable create response")
    request_hash = run.get("requestHash")
    expected_hash = _durable_request_hash(
        {"thread_id": thread_id, "user_message_id": user_message_id, "assistant_message_id": assistant_message_id},
        stored_payload,
    )
    if not isinstance(request_hash, str) or request_hash != expected_hash:
        raise AccountBindingError("durable create response requestHash did not bind the stored request")
    if stored_payload.get("session_id") != f"project-{project['project_id']}":
        raise AccountBindingError("durable request session_id did not bind the managed project")
    return {"thread_id": thread_id, "user_message_id": user_message_id, "assistant_message_id": assistant_message_id, "run_id": run_id, "project_id": project["project_id"], "session_id": f"project-{project['project_id']}", "user_prompt_sha256": _sha256_bytes(DEFAULT_PROMPT.encode("utf-8")), "user_message_content_sha256": _prompt_content_sha256(), "request_payload": dict(stored_payload), "request_hash": request_hash, "create_response": dict(run)}


def _json_depth(value: Any, *, level: int = 0) -> None:
    if level > MAX_SSE_JSON_DEPTH:
        raise ProtocolMismatchError("frontend SSE JSON nesting exceeded the bound")
    if isinstance(value, Mapping):
        for item in value.values():
            _json_depth(item, level=level + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _json_depth(item, level=level + 1)


def _contains_private_trace(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).startswith("_helix_optimization_trace") or str(key) in {"optimization.trace", "optimization_trace"}:
                return True
            if _contains_private_trace(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_private_trace(item) for item in value)
    elif isinstance(value, str):
        return TRACE_EVENT_TYPE in value or "_helix_optimization_trace" in value
    return False


def _allow_tool_start(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    run_id: str,
    session_id: str,
    payload: Mapping[str, Any],
    fixture: FixtureSpec,
    sequence_index: int,
    timeout: float,
) -> dict[str, Any]:
    """Validate an auto-safe public proposal without creating an approval."""

    approval_id = payload.get("approval_id")
    call_id = payload.get("tool_call_id")
    if payload.get("awaiting_confirmation") is not False:
        raise ProtocolMismatchError("auto-safe terminal proposal requested confirmation")
    if approval_id != "":
        raise ProtocolMismatchError("auto-safe terminal proposal carried an approval id")
    if not isinstance(call_id, str) or not call_id:
        raise ProtocolMismatchError("terminal proposal call identity was missing")
    summary = validate_tool_proposal(
        payload.get("tool_name"), payload.get("arguments"), fixture, sequence_index=sequence_index
    )
    return {
        "index": sequence_index,
        "tool_name": TOOL_NAME,
        "arguments_sha256": summary["arguments_sha256"],
        "proposal_payload_sha256": summary["proposal_payload_sha256"],
        "tool_call_id_sha256": _sha256_bytes(call_id.encode("utf-8")),
        "approval_id_sha256": _sha256_bytes(b""),
        "decision": "ungated",
    }


def _stream_lines(response: Any) -> Iterator[Any]:
    if hasattr(response, "iter_lines"):
        yield from response.iter_lines()
    elif hasattr(response, "readline"):
        while True:
            line = response.readline()
            if not line:
                return
            yield line
    else:
        yield from response


def consume_tool_sse(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    run_id: str,
    *,
    timeout: float,
    fixture: FixtureSpec | None = None,
    session_id: str | None = None,
    permission_mode: str = "auto",
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Parse bounded public replay while retaining no raw stream in artifacts."""

    if permission_mode != "auto":
        raise ContractError("P1-RO-REPO requires auto-safe permission mode")
    started = clock()
    response = transport.open_stream(
        "POST", base_url.rstrip("/") + f"/api/inference/chat-runs/{urllib.parse.quote(run_id, safe='')}/events?after=0",
        headers={**headers, "Accept": "text/event-stream"}, payload={}, timeout=timeout,
    )
    events: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    block: list[str] = []
    raw_bytes = 0
    last_cursor: int | None = None
    try:
        def consume_block(lines: list[str]) -> None:
            nonlocal last_cursor
            if not lines:
                return
            event_type: str | None = None
            event_id: str | None = None
            data_lines: list[str] = []
            for line in lines:
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("id:"):
                    event_id = line[3:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                if event_id is not None:
                    raise ProtocolMismatchError("frontend SSE cursor frame had no data")
                return
            try:
                envelope = json.loads("\n".join(data_lines))
            except (TypeError, ValueError) as exc:
                raise ProtocolMismatchError("frontend SSE contained malformed JSON") from exc
            _json_depth(envelope)
            if _contains_private_trace(envelope) or event_type == TRACE_EVENT_TYPE:
                raise ProtocolMismatchError("private optimization trace leaked to frontend SSE")
            if not isinstance(envelope, Mapping) or not isinstance(event_type, str) or not event_type:
                raise ProtocolMismatchError("frontend SSE envelope was not canonical")
            if event_id is None or not re.fullmatch(r"[0-9]+", event_id):
                raise ProtocolMismatchError("frontend SSE cursor was invalid")
            cursor = int(event_id)
            if envelope.get("seq") != cursor or envelope.get("type") != event_type:
                raise ProtocolMismatchError("frontend SSE cursor did not match envelope")
            payload = envelope.get("payload")
            if not isinstance(payload, Mapping):
                raise ProtocolMismatchError("frontend SSE payload was not an object")
            if last_cursor is not None and cursor <= last_cursor:
                raise ProtocolMismatchError("frontend SSE cursor was not monotonic")
            last_cursor = cursor
            if len(events) >= MAX_SSE_EVENTS:
                raise ProtocolMismatchError("frontend SSE exceeded the event-count bound")
            received_elapsed_ms = max(0.0, (clock() - started) * 1000.0)
            event_record = {"event": event_type, "cursor": cursor, "payload": dict(payload), "run": dict(envelope["run"]) if isinstance(envelope.get("run"), Mapping) else None, "received_elapsed_ms": received_elapsed_ms}
            if event_type == "chunk" and payload.get("type") == "tool_start":
                if fixture is None or not isinstance(session_id, str) or not session_id:
                    raise ProtocolMismatchError("tool proposal was not bound to a managed project session")
                proposal_index = len(proposals)
                try:
                    proposal = _allow_tool_start(
                        transport,
                        base_url,
                        headers,
                        run_id,
                        session_id,
                        payload,
                        fixture,
                        proposal_index,
                        timeout,
                    )
                except Exception:
                    # Auto-safe mode has no approval endpoint or denial side
                    # effect.  A malformed proposal is simply a failed replay.
                    raise
                proposals.append(proposal)
                event_record["proposal"] = proposal
            events.append(event_record)

        for raw in _stream_lines(response):
            text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else str(raw)
            size = len(text.encode("utf-8"))
            if size > MAX_SSE_LINE_BYTES or raw_bytes + size > MAX_SSE_WIRE_BYTES:
                raise ProtocolMismatchError("frontend SSE exceeded its wire bound")
            raw_bytes += size
            text = text.rstrip("\r\n")
            if not text:
                consume_block(block)
                block = []
            else:
                if sum(len(item.encode("utf-8")) for item in block) + size > MAX_SSE_BLOCK_BYTES:
                    raise ProtocolMismatchError("frontend SSE block exceeded its bound")
                block.append(text)
        consume_block(block)
    except UnicodeError as exc:
        raise ProtocolMismatchError("frontend SSE was not UTF-8") from exc
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    terminal: Mapping[str, Any] | None = None
    for event in events:
        candidate = event.get("run")
        payload = event.get("payload")
        for item in (candidate, payload.get("run") if isinstance(payload, Mapping) else None, payload):
            if isinstance(item, Mapping) and item.get("id") == run_id and item.get("status") in {"completed", "failed", "cancelled"}:
                terminal = item
    if terminal is None or terminal.get("status") != "completed":
        raise ProtocolMismatchError("frontend SSE did not prove a completed durable run")
    if not events or int(terminal.get("lastEventSeq", terminal.get("last_event_seq", -1))) != int(events[-1]["cursor"]):
        raise ProtocolMismatchError("frontend SSE terminal cursor did not prove the final event")
    if fixture is not None and len(proposals) != len(fixture.expected_sequence):
        raise ProtocolMismatchError("frontend SSE tool proposal count did not match the fixture")
    return {"status": "pass", "event_count": len(events), "events": events, "proposals": proposals, "wire_bytes": raw_bytes, "elapsed_ms": max(0.0, (clock() - started) * 1000.0), "last_event_seq": events[-1]["cursor"], "private_trace_suppressed": True}


def _observed_answer_text(sse: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for event in sse.get("events") or ():
        payload = event.get("payload") if isinstance(event, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        for choice in payload.get("choices") or ():
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                pieces.append(content)
    return "".join(pieces)


def _assess_answer_text(text: str, fixture: FixtureSpec) -> dict[str, Any]:
    """Classify model quality without turning a wrong answer into protocol loss."""

    expected_text = _canonical_json(_expected_answer_from_manifest(json.loads(fixture.manifest_path.read_text(encoding="utf-8"))))
    semantic: Any = None
    try:
        semantic = json.loads(text)
    except (TypeError, ValueError):
        fenced = re.fullmatch(r"```(?:json)?\s*\n?(\{.*\})\s*\n?```", text, flags=re.DOTALL)
        if fenced is not None:
            try:
                semantic = json.loads(fenced.group(1))
            except (TypeError, ValueError):
                semantic = None
    expected = _expected_answer_from_manifest(json.loads(fixture.manifest_path.read_text(encoding="utf-8")))
    encoded = text.encode("utf-8")
    return {
        "exact": text == expected_text,
        "semantic_match": isinstance(semantic, Mapping) and dict(semantic) == expected,
        "sha256": _sha256_bytes(encoded),
        "length": len(encoded),
    }


def _extract_structured_answer_text(sse: Mapping[str, Any], fixture: FixtureSpec) -> str:
    text = _observed_answer_text(sse)
    if not _assess_answer_text(text, fixture)["exact"]:
        raise ProtocolMismatchError("frontend SSE did not contain the exact structured answer")
    return text


def _extract_structured_answer(sse: Mapping[str, Any], fixture: FixtureSpec) -> dict[str, Any]:
    text = _extract_structured_answer_text(sse, fixture)
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatchError("frontend SSE did not contain a structured answer") from exc
    if not isinstance(value, Mapping):
        raise ProtocolMismatchError("final answer was not a structured object")
    return dict(value)


def persist_structured_answer(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    *,
    objective: Mapping[str, str],
    sse: Mapping[str, Any],
    answer: Mapping[str, Any],
    timeout: float,
    answer_text: str | None = None,
) -> dict[str, Any]:
    """Persist the public structured answer while returning hash-only metadata."""

    if answer_text is None:
        answer_text = _canonical_json(dict(answer))
    if not isinstance(answer_text, str) or not answer_text or len(answer_text.encode("utf-8")) > MAX_SSE_BLOCK_BYTES:
        raise ProtocolMismatchError("assistant answer text was empty or exceeded its bound")
    thread_id = objective["thread_id"]
    assistant_id = objective["assistant_message_id"]
    path = f"/api/chat/threads/{urllib.parse.quote(thread_id, safe='')}/messages/{urllib.parse.quote(assistant_id, safe='')}"
    current = _json_request(transport, "GET", base_url, path, headers=headers, timeout=timeout)
    if not isinstance(current, Mapping) or current.get("id") != assistant_id or current.get("threadId") != thread_id or current.get("parentId") != objective["user_message_id"] or current.get("role") != "assistant":
        raise AccountBindingError("assistant placeholder identity was invalid")
    if current.get("content") not in ([], None):
        raise ProtocolMismatchError("assistant placeholder was already occupied")
    metadata = current.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("generationRunId") != objective["run_id"] or metadata.get("serverManaged") is not True or metadata.get("generationStatus") != "completed":
        raise AccountBindingError("assistant placeholder generation binding was invalid")
    generation_seq = sse.get("last_event_seq")
    if isinstance(generation_seq, bool) or not isinstance(generation_seq, int):
        raise ProtocolMismatchError("assistant answer did not have a durable sequence")
    next_metadata = dict(metadata)
    next_metadata.update({"generationRunId": objective["run_id"], "generationSeq": generation_seq, "generationStatus": "completed", "generationSettled": True, "serverManaged": True})
    payload = {"id": assistant_id, "threadId": thread_id, "parentId": objective["user_message_id"], "role": "assistant", "content": [{"type": "text", "text": answer_text}], "attachments": current.get("attachments"), "metadata": next_metadata, "createdAt": current.get("createdAt")}
    saved = _json_request(transport, "PUT", base_url, path, headers=headers, payload=payload, timeout=timeout)
    if not isinstance(saved, Mapping) or saved.get("id") != assistant_id or saved.get("threadId") != thread_id or saved.get("parentId") != objective["user_message_id"] or saved.get("role") != "assistant" or saved.get("metadata") != next_metadata:
        raise AccountBindingError("assistant answer persistence binding was invalid")
    saved_content = saved.get("content")
    if not isinstance(saved_content, list) or len(saved_content) != 1 or not isinstance(saved_content[0], Mapping) or saved_content[0].get("type") != "text" or saved_content[0].get("text") != answer_text:
        raise ProtocolMismatchError("persisted assistant content was not exact")
    return {"status": "pass", "answer_sha256": _sha256_bytes(answer_text.encode("utf-8")), "answer_length": len(answer_text.encode("utf-8")), "generation_seq": generation_seq}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _parse_json(value: Any, label: str) -> Any:
    if isinstance(value, (Mapping, list)):
        return value
    if not isinstance(value, str):
        raise ProtocolMismatchError(f"{label} was not JSON")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatchError(f"{label} was not valid JSON") from exc


def _hash_result(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        raw = value.encode("utf-8", "surrogatepass")
    else:
        raw = _canonical_json(value).encode("utf-8")
    return _sha256_bytes(raw), len(raw)


def _event_rows(connection: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    if not _table_exists(connection, "chat_generation_events"):
        raise ProtocolMismatchError("durable generation event table was missing")
    return list(connection.execute("SELECT seq, event_type, payload_json, created_at FROM chat_generation_events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall())


def _row_value(row: sqlite3.Row | Mapping[str, Any], *names: str) -> Any:
    for name in names:
        try:
            keys = row.keys() if hasattr(row, "keys") else row
            if name in keys:
                return row[name]
        except TypeError:
            pass
    return None


def read_tool_durable_evidence(
    studio_db_path: Path,
    *,
    objective: Mapping[str, str],
    fixture: FixtureSpec,
    request_payload: Mapping[str, Any],
    provider_tier: str = DEFAULT_PROVIDER_TIER,
    model_name: str = DEFAULT_MODEL_NAME,
    provider_identity: str = DEFAULT_PROVIDER_IDENTITY,
    account_id: str = OWNER_ACCOUNT_ID,
    acceptance_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the exact ungated v3 durable ledger; never infer execution from chat/SSE."""

    database = _require_regular_file(studio_db_path, "clone studio.db")
    before_db = _sha256_bytes(database.read_bytes())
    uri = f"file:{urllib.parse.quote(str(database.resolve(strict=True)), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ProtocolMismatchError("clone studio.db could not be opened read-only") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise ProtocolMismatchError("SQLite verifier was not query-only")
        required_tables = {
            "chat_generation_runs", "chat_generation_events", "chat_generation_tool_approvals",
            "chat_generation_tool_executions", "chat_threads", "chat_messages",
        }
        if any(not _table_exists(connection, table) for table in required_tables):
            raise ProtocolMismatchError("durable objective tables were missing")

        rows = list(connection.execute("SELECT * FROM chat_generation_runs WHERE id=?", (objective["run_id"],)).fetchall())
        if len(rows) != 1:
            raise AccountBindingError("durable objective row was missing or duplicated")
        run = rows[0]
        for key, expected in (
            ("owner_subject", OWNER_USERNAME),
            ("thread_id", objective["thread_id"]),
            ("user_message_id", objective["user_message_id"]),
            ("assistant_message_id", objective["assistant_message_id"]),
        ):
            if _row_value(run, key) != expected:
                raise AccountBindingError("durable objective identity did not match")
        if _row_value(run, "status") != "completed" or _row_value(run, "finish_reason", "finishReason") != "stop":
            raise ProtocolMismatchError("durable objective did not complete")
        if _row_value(run, "finalization_status", "finalizationStatus") != "none":
            raise ProtocolMismatchError("durable objective finalization was not settled")

        stored_request = _parse_json(_row_value(run, "request_json", "requestJson"), "durable request")
        if not isinstance(stored_request, Mapping) or "requestPayload" in stored_request:
            raise ProtocolMismatchError("durable request was not the sanitized stored payload")
        request = dict(stored_request)
        _validate_pinned_request_payload(request, label="durable stored request")
        if _canonical_json(request) != _canonical_json(dict(request_payload)):
            raise AccountBindingError("durable stored request differed from the create response")
        request_hash = _row_value(run, "request_hash", "requestHash")
        expected_request_hash = _durable_request_hash(objective, request)
        if not isinstance(request_hash, str) or request_hash != expected_request_hash or (objective.get("request_hash") and objective.get("request_hash") != request_hash):
            raise AccountBindingError("durable request_hash did not bind the full stored request")
        if {"acceptance_criteria_sha256", "fixture_id", "fixture_sha256", "network_enabled"}.intersection(request):
            raise ProtocolMismatchError("durable request contained unsupported harness fields")
        summary = summarize_request_payload(request)
        if (
            summary["tool_names"] != [TOOL_NAME]
            or not summary["enable_tools"]
            or summary["enabled_tools"] != [TOOL_NAME]
            or summary["mcp_enabled"]
            or summary["bypass_permissions"]
            or summary["permission_mode"] != "auto"
            or not summary["confirm_tool_calls"]
            or summary.get("session_id") != objective.get("session_id")
        ):
            raise ProtocolMismatchError("durable request tool policy was not the ungated v3 contract")

        thread_rows = list(connection.execute("SELECT * FROM chat_threads WHERE id=?", (objective["thread_id"],)).fetchall())
        if len(thread_rows) != 1:
            raise AccountBindingError("exact project thread row was missing or duplicated")
        if _row_value(thread_rows[0], "project_id", "projectId") != objective.get("project_id"):
            raise AccountBindingError("thread project binding did not match")

        events = _event_rows(connection, objective["run_id"])
        event_records: list[tuple[int, str, Mapping[str, Any]]] = []
        prior_seq = 0
        for row in events:
            seq = row["seq"]
            if isinstance(seq, bool) or not isinstance(seq, int) or seq <= prior_seq:
                raise ProtocolMismatchError("durable event sequence was not strictly increasing")
            prior_seq = seq
            payload = _parse_json(row["payload_json"], "durable event payload")
            if not isinstance(payload, Mapping):
                raise ProtocolMismatchError("durable event payload was not an object")
            event_records.append((seq, str(row["event_type"]), payload))
        if not event_records or _row_value(run, "last_event_seq", "lastEventSeq") != event_records[-1][0]:
            raise ProtocolMismatchError("durable last-event binding was invalid")

        # Auto-safe means the approval table and approval event namespace are
        # intentionally empty.  Do not use row position as an authority.
        approvals = list(connection.execute("SELECT * FROM chat_generation_tool_approvals WHERE run_id=?", (objective["run_id"],)).fetchall())
        if approvals or any(kind.startswith("approval.") for _seq, kind, _payload in event_records):
            raise ProtocolMismatchError("ungated v3 run retained approval rows or events")

        tool_starts = [(seq, payload) for seq, kind, payload in event_records if kind == "chunk" and payload.get("type") == "tool_start"]
        if len(tool_starts) != len(fixture.expected_sequence):
            raise ProtocolMismatchError("durable tool_start proposal count did not match the fixture")
        proposal_summaries: list[dict[str, Any]] = []
        public_by_card: dict[str, tuple[int, Mapping[str, Any], dict[str, str]]] = {}
        for index, (proposal_seq, payload) in enumerate(tool_starts):
            if payload.get("type") != "tool_start" or payload.get("approval_id") != "" or payload.get("awaiting_confirmation") is not False or "_durable_tool_approval" in payload:
                raise ProtocolMismatchError("public auto-safe proposal was confirmation-gated")
            if "arguments_text" in payload and payload.get("arguments_text") != json.dumps(payload.get("arguments"), ensure_ascii=False, sort_keys=False, separators=(",", ":")):
                raise ProtocolMismatchError("public tool proposal arguments_text was not canonical")
            card_id = payload.get("tool_call_id")
            if not isinstance(card_id, str) or not card_id or card_id in public_by_card:
                raise ProtocolMismatchError("public tool proposal card identity was blank or duplicated")
            summary_proposal = validate_tool_proposal(payload.get("tool_name"), payload.get("arguments"), fixture, sequence_index=index)
            public_by_card[card_id] = (proposal_seq, payload, summary_proposal)
            proposal_summaries.append({
                "index": index,
                "tool_name": TOOL_NAME,
                "arguments_sha256": summary_proposal["arguments_sha256"],
                "proposal_payload_sha256": summary_proposal["proposal_payload_sha256"],
                "tool_call_id_sha256": _sha256_bytes(card_id.encode("utf-8")),
                "approval_id_sha256": _sha256_bytes(b""),
            })

        execution_rows = list(connection.execute(
            "SELECT * FROM chat_generation_tool_executions WHERE run_id=? ORDER BY claimed_at, execution_id",
            (objective["run_id"],),
        ).fetchall())
        if len(execution_rows) != len(fixture.expected_sequence):
            raise ProtocolMismatchError("ungated execution row count did not match the fixture")
        public_order = {card_id: index for index, card_id in enumerate(public_by_card)}
        # Row ordering is only a stable presentation aid.  Correlation is by
        # the persisted card call identity, so equal timestamps cannot make a
        # valid run appear to have executed the wrong fixture command.
        execution_rows.sort(key=lambda row: public_order.get(str(_row_value(row, "card_call_id")), len(public_order)))
        execution_ids: set[str] = set()
        execution_cards: set[str] = set()
        execution_summaries: list[dict[str, Any]] = []
        execution_by_id: dict[str, sqlite3.Row] = {}
        for index, row in enumerate(execution_rows):
            execution_id = str(_row_value(row, "execution_id") or "")
            if not execution_id or execution_id in execution_ids:
                raise ProtocolMismatchError("ungated execution ids were blank or duplicated")
            execution_ids.add(execution_id)
            execution_by_id[execution_id] = row
            expected = fixture.expected_sequence[index]
            if (
                _row_value(row, "run_id") != objective["run_id"]
                or _row_value(row, "backend_account_id") != account_id
                or _row_value(row, "owner_subject") != OWNER_USERNAME
                or _row_value(row, "session_id") != objective.get("session_id")
                or _row_value(row, "thread_id") != objective["thread_id"]
                or _row_value(row, "tool_name") != TOOL_NAME
                or _row_value(row, "authority_kind") != "ungated"
                or _row_value(row, "approval_id") is not None
                or _row_value(row, "execution_state") != "finished"
                or _row_value(row, "error_message") is not None
                or not isinstance(_row_value(row, "claim_token"), str)
                or not _row_value(row, "claim_token")
                or not isinstance(_row_value(row, "worker_token"), str)
                or not _row_value(row, "worker_token")
            ):
                raise AccountBindingError("ungated execution identity or state was invalid")
            model_call_id = _row_value(row, "tool_call_id")
            card_call_id = _row_value(row, "card_call_id")
            if not isinstance(model_call_id, str) or not model_call_id or not isinstance(card_call_id, str) or not card_call_id:
                raise ProtocolMismatchError("ungated model/card call identity was missing")
            if card_call_id in execution_cards:
                raise ProtocolMismatchError("ungated execution card identities were duplicated")
            execution_cards.add(card_call_id)
            public = public_by_card.get(card_call_id)
            if public is None:
                raise AccountBindingError("ungated execution card call was not bound to a public proposal")
            args = _parse_json(_row_value(row, "arguments_json"), "ungated execution arguments")
            if args != public[1].get("arguments") or args != {"command": expected["command"]} or _row_value(row, "arguments_json") != _canonical_json(args):
                raise ProtocolMismatchError("ungated execution arguments differed from the fixture proposal")
            fingerprint = _sha256_json(args)
            if _row_value(row, "arguments_fingerprint") != fingerprint or fingerprint != expected["arguments_sha256"]:
                raise ProtocolMismatchError("ungated execution arguments fingerprint was invalid")

            checkpoint_version = _row_value(row, "pre_tool_checkpoint_version")
            if isinstance(checkpoint_version, bool) or not isinstance(checkpoint_version, int) or checkpoint_version != CHECKPOINT_VERSION:
                raise ProtocolMismatchError("ungated pre-tool checkpoint version was invalid")
            checkpoint_raw = _row_value(row, "pre_tool_checkpoint_json")
            checkpoint = _parse_json(checkpoint_raw, "ungated pre-tool checkpoint")
            if not isinstance(checkpoint, Mapping) or checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("backend") not in {"gguf", "safetensors"} or not isinstance(checkpoint.get("conversation"), list) or not isinstance(checkpoint.get("controller"), Mapping) or not isinstance(checkpoint.get("remaining_calls"), list) or not isinstance(checkpoint.get("current_call"), Mapping):
                raise ProtocolMismatchError("ungated pre-tool checkpoint was incomplete")
            if checkpoint_raw != _canonical_json(checkpoint) or _row_value(row, "pre_tool_checkpoint_digest") != _sha256_json(checkpoint):
                raise ProtocolMismatchError("ungated pre-tool checkpoint digest was invalid")
            current_call = checkpoint["current_call"]
            tool_call = current_call.get("tool_call") if isinstance(current_call.get("tool_call"), Mapping) else None
            function = tool_call.get("function") if isinstance(tool_call, Mapping) and isinstance(tool_call.get("function"), Mapping) else None
            checkpoint_args = function.get("arguments") if isinstance(function, Mapping) else None
            if isinstance(checkpoint_args, str):
                checkpoint_args = _parse_json(checkpoint_args, "checkpoint arguments")
            if (
                not isinstance(tool_call, Mapping)
                or not isinstance(function, Mapping)
                or function.get("name") != TOOL_NAME
                or checkpoint_args != args
                or tool_call.get("id") != model_call_id
                or current_call.get("card_call_id") != card_call_id
            ):
                raise ProtocolMismatchError("ungated pre-tool checkpoint identity did not match execution")

            started_at = _row_value(row, "started_at")
            finished_at = _row_value(row, "finished_at")
            claimed_at = _row_value(row, "claimed_at")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (claimed_at, started_at, finished_at)) or not (claimed_at < started_at <= finished_at):
                raise ProtocolMismatchError("ungated execution timestamps were invalid")
            result_raw = _row_value(row, "result_json")
            result = _parse_json(result_raw, "ungated raw tool result")
            if not isinstance(result, str) or result == "":
                raise ProtocolMismatchError("ungated execution did not retain a raw result")
            result_sha, result_length = _hash_result(result)
            if result_raw != _canonical_json(result) or result_sha != expected["result_sha256"] or result_length != int(expected["result_length"]):
                raise ProtocolMismatchError("ungated raw tool result did not match the pinned fixture")

            producer_raw = _row_value(row, "producer_receipt_json")
            producer = _parse_json(producer_raw, "producer receipt")
            producer_keys = {
                "schema_version", "producer_version", "tool", "resolved_launch_cwd_observed_before_spawn",
                "cwd_identity_observed_before_spawn", "confinement", "process_outcome_kind", "return_code",
                "return_code_available", "timed_out", "cancelled", "spawn_error",
                "decoded_output_utf8_surrogatepass_sha256", "decoded_output_utf8_surrogatepass_byte_length",
                "capture_complete", "captured_byte_length", "discarded_byte_length", "capture_budget_chars",
                "read_error", "fallback_result_utf8_surrogatepass_sha256", "fallback_result_utf8_surrogatepass_byte_length",
                "fallback_budget_chars",
            }
            if not isinstance(producer, Mapping) or set(producer) != producer_keys or producer_raw != _canonical_json(producer):
                raise ProtocolMismatchError("producer receipt schema was not exact")
            raw_bytes = result.encode("utf-8", "surrogatepass")
            output_digest = _sha256_bytes(raw_bytes)
            if (
                producer.get("schema_version") != "helix.tool-producer-receipt.v1"
                or producer.get("producer_version") != 1
                or producer.get("tool") != TOOL_NAME
                or not isinstance(producer.get("resolved_launch_cwd_observed_before_spawn"), str)
                or not Path(str(producer["resolved_launch_cwd_observed_before_spawn"])).is_absolute()
                or not isinstance(producer.get("cwd_identity_observed_before_spawn"), Mapping)
                or set(producer["cwd_identity_observed_before_spawn"]) != {"device", "inode"}
                or any(isinstance(producer["cwd_identity_observed_before_spawn"].get(key), bool) or not isinstance(producer["cwd_identity_observed_before_spawn"].get(key), int) or producer["cwd_identity_observed_before_spawn"][key] < 0 for key in ("device", "inode"))
                or producer.get("confinement") not in {"sandboxed-owner-direct", "sandboxed-account-confinement"}
                or producer.get("process_outcome_kind") != "exited"
                or producer.get("return_code") != 0
                or producer.get("return_code_available") is not True
                or producer.get("timed_out") is not False
                or producer.get("cancelled") is not False
                or producer.get("spawn_error") is not False
                or producer.get("decoded_output_utf8_surrogatepass_sha256") != output_digest
                or producer.get("decoded_output_utf8_surrogatepass_byte_length") != len(raw_bytes)
                or producer.get("capture_complete") is not True
                or producer.get("captured_byte_length") != len(raw_bytes)
                or producer.get("discarded_byte_length") != 0
                or isinstance(producer.get("capture_budget_chars"), bool)
                or not isinstance(producer.get("capture_budget_chars"), int)
                or producer.get("capture_budget_chars") < len(result)
                or producer.get("read_error") is not None
                or producer.get("fallback_result_utf8_surrogatepass_sha256") != output_digest
                or producer.get("fallback_result_utf8_surrogatepass_byte_length") != len(raw_bytes)
                or (producer.get("fallback_budget_chars") is not None and (isinstance(producer.get("fallback_budget_chars"), bool) or not isinstance(producer.get("fallback_budget_chars"), int) or producer.get("fallback_budget_chars") < len(result)))
            ):
                raise ProtocolMismatchError("producer receipt did not prove a successful bounded terminal result")

            completion_raw = _row_value(row, "completion_json")
            completion = _parse_json(completion_raw, "tool completion")
            completion_keys = {"schema_version", "execution_id", "authority_kind", "terminal_state", "selected_result", "error", "producer_receipt", "controller_is_error", "completion_annotations", "post_controller_checkpoint", "receipt_ref", "binding"}
            if not isinstance(completion, Mapping) or set(completion) != completion_keys or completion_raw != _canonical_json(completion):
                raise ProtocolMismatchError("tool completion schema was not exact")
            binding = {
                "backend_account_id": str(_row_value(row, "backend_account_id")),
                "owner_subject": str(_row_value(row, "owner_subject")),
                "run_id": str(_row_value(row, "run_id")),
                "session_id": str(_row_value(row, "session_id") or ""),
                "thread_id": str(_row_value(row, "thread_id")),
                "execution_id": execution_id,
                "tool_name": TOOL_NAME,
                "tool_call_id": model_call_id,
                "card_call_id": card_call_id,
                "arguments_fingerprint": fingerprint,
                "pre_tool_checkpoint_digest": str(_row_value(row, "pre_tool_checkpoint_digest")),
                "authority_kind": "ungated",
                "approval_id": None,
            }
            post_checkpoint = completion.get("post_controller_checkpoint")
            annotations = completion.get("completion_annotations")
            if (
                completion.get("schema_version") != "helix.tool-completion.v1"
                or completion.get("execution_id") != execution_id
                or completion.get("authority_kind") != "ungated"
                or completion.get("terminal_state") != "finished"
                or completion.get("selected_result") != result
                or completion.get("error") is not None
                or completion.get("producer_receipt") != dict(producer)
                or completion.get("controller_is_error") is not False
                or not isinstance(annotations, Mapping)
                or not isinstance(post_checkpoint, Mapping)
                or post_checkpoint.get("version") != CHECKPOINT_VERSION
                or completion.get("receipt_ref") != f"tool-receipt:{execution_id}"
                or completion.get("binding") != binding
                or _row_value(row, "receipt_ref") != completion.get("receipt_ref")
                or _row_value(row, "receipt_digest") != _sha256_json(completion)
                or _row_value(row, "controller_is_error") != 0
                or _parse_json(_row_value(row, "completion_annotations_json"), "completion annotations") != dict(annotations)
                or _parse_json(_row_value(row, "post_controller_checkpoint_json"), "post-controller checkpoint") != dict(post_checkpoint)
            ):
                raise ProtocolMismatchError("tool completion binding/result/controller parity was invalid")

            execution_by_id[execution_id] = row
            execution_summaries.append({
                "index": index,
                "execution_id_sha256": _sha256_bytes(execution_id.encode("utf-8")),
                "tool_name": TOOL_NAME,
                "model_call_id_sha256": _sha256_bytes(model_call_id.encode("utf-8")),
                "card_call_id_sha256": _sha256_bytes(card_call_id.encode("utf-8")),
                "claim_seq": None,
                "start_seq": None,
                "finish_seq": None,
                "public_end_seq": None,
                "result_sha256": result_sha,
                "result_length": result_length,
                "started_at": started_at,
                "finished_at": finished_at,
            })

        def matching(kind: str, execution_id: str) -> list[tuple[int, Mapping[str, Any]]]:
            return [(seq, payload) for seq, event_kind, payload in event_records if event_kind == kind and payload.get("execution_id") == execution_id]

        claimed_all = [(seq, payload) for seq, kind, payload in event_records if kind == "tool_execution.claimed"]
        started_all = [(seq, payload) for seq, kind, payload in event_records if kind == "tool_execution.started"]
        finished_all = [(seq, payload) for seq, kind, payload in event_records if kind == "tool_execution.finished"]
        if any(kind == "tool_execution.ambiguous" for _seq, kind, _payload in event_records) or any(len(items) != len(execution_rows) for items in (claimed_all, started_all, finished_all)):
            raise ProtocolMismatchError("ungated v3 execution event count was missing, ambiguous, or duplicated")
        public_ends = [(seq, payload) for seq, kind, payload in event_records if kind == "chunk" and payload.get("type") == "tool_end"]
        if len(public_ends) != len(fixture.expected_sequence):
            raise ProtocolMismatchError("public tool_end count did not match the fixture")
        public_end_by_card: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for seq, payload in public_ends:
            card_id = payload.get("tool_call_id")
            if not isinstance(card_id, str) or card_id in public_end_by_card:
                raise ProtocolMismatchError("public tool_end identity was blank or duplicated")
            public_end_by_card[card_id] = (seq, payload)

        for index, row in enumerate(execution_rows):
            execution_id = str(row["execution_id"])
            card_id = str(row["card_call_id"])
            expected = fixture.expected_sequence[index]
            proposal_seq = public_by_card[card_id][0]
            claimed = matching("tool_execution.claimed", execution_id)
            started = matching("tool_execution.started", execution_id)
            finished = matching("tool_execution.finished", execution_id)
            public_end = public_end_by_card.get(card_id)
            if not (len(claimed) == len(started) == len(finished) == 1) or public_end is None:
                raise ProtocolMismatchError("ungated v3 execution receipt was incomplete or duplicated")
            claim_seq, claim_payload = claimed[0]
            start_seq, start_payload = started[0]
            finish_seq, finish_payload = finished[0]
            end_seq, end_payload = public_end
            if set(claim_payload) != {"schema_version", "execution_id", "authority_kind", "arguments_fingerprint", "pre_tool_checkpoint_digest"} or claim_payload != {
                "schema_version": TOOL_EXECUTION_SCHEMA,
                "execution_id": execution_id,
                "authority_kind": "ungated",
                "arguments_fingerprint": row["arguments_fingerprint"],
                "pre_tool_checkpoint_digest": row["pre_tool_checkpoint_digest"],
            }:
                raise ProtocolMismatchError("ungated v3 claim receipt shape or identity was invalid")
            if set(start_payload) != {"schema_version", "execution_id", "tool_name", "tool_call_id", "effect_state", "authority_kind"} or start_payload != {
                "schema_version": TOOL_EXECUTION_SCHEMA,
                "execution_id": execution_id,
                "tool_name": TOOL_NAME,
                "tool_call_id": card_id,
                "effect_state": "started",
                "authority_kind": "ungated",
            }:
                raise ProtocolMismatchError("ungated v3 start receipt shape or identity was invalid")
            expected_result = _parse_json(row["result_json"], "ungated raw tool result")
            expected_finish = {
                "schema_version": TOOL_EXECUTION_SCHEMA,
                "execution_id": execution_id,
                "tool_name": TOOL_NAME,
                "tool_call_id": card_id,
                "effect_state": "finished",
                "authority_kind": "ungated",
                "receipt_ref": row["receipt_ref"],
                "receipt_digest": row["receipt_digest"],
                "result": expected_result,
            }
            if set(finish_payload) != set(expected_finish) or finish_payload != expected_finish:
                raise ProtocolMismatchError("ungated v3 finish receipt shape or identity was invalid")
            if set(end_payload) != {"type", "tool_name", "tool_call_id", "result", "provenance"} or end_payload.get("type") != "tool_end" or end_payload.get("tool_name") != TOOL_NAME or end_payload.get("tool_call_id") != card_id or end_payload.get("result") != expected_result:
                raise ProtocolMismatchError("public tool_end result or identity was invalid")
            if not (proposal_seq < claim_seq < start_seq < finish_seq < end_seq) or row["terminal_seq"] != finish_seq:
                raise ProtocolMismatchError("tool proposal/claim/start/finish/public-end order was invalid")
            execution_summaries[index].update({"claim_seq": claim_seq, "start_seq": start_seq, "finish_seq": finish_seq, "public_end_seq": end_seq})
            if end_payload.get("provenance") != _parse_json(row["completion_annotations_json"], "completion annotations"):
                raise ProtocolMismatchError("public tool_end provenance did not match completion annotations")

        trace_rows = [(seq, payload) for seq, event_type, payload in event_records if event_type == TRACE_EVENT_TYPE]
        if len(trace_rows) != 1:
            raise ProtocolMismatchError("durable optimization trace was missing or duplicated")
        trace_seq, trace_payload = trace_rows[0]
        try:
            validated_trace = _SOURCE.validate_trace_payload(trace_payload, objective=objective, provider_tier=provider_tier, model_name=model_name, provider_identity=provider_identity, account_id=account_id)
        except getattr(_SOURCE, "AccountBindingError", AccountBindingError) as exc:
            raise AccountBindingError("optimization trace owner/account binding did not match") from exc
        except getattr(_SOURCE, "ProtocolMismatchError", ProtocolMismatchError) as exc:
            raise ProtocolMismatchError("optimization trace did not satisfy the approved schema") from exc
        trace_bytes = _canonical_json(trace_payload).encode("utf-8")
        completed_rows = [(seq, payload) for seq, kind, payload in event_records if kind == "run.completed"]
        terminal_seq = completed_rows[0][0] if len(completed_rows) == 1 else None
        if len(event_records) < 2 or len(completed_rows) != 1 or completed_rows[0][1].get("status") != "completed" or completed_rows[0][1].get("finishReason", completed_rows[0][1].get("finish_reason")) != "stop" or terminal_seq != event_records[-1][0] or event_records[-1][1] != "run.completed" or trace_seq != terminal_seq - 1 or event_records[-2][1] != TRACE_EVENT_TYPE:
            raise ProtocolMismatchError("durable completion/trace binding was invalid")

        message_rows = list(connection.execute("SELECT * FROM chat_messages WHERE id IN (?, ?) ORDER BY id", (objective["user_message_id"], objective["assistant_message_id"])).fetchall())
        by_id = {str(_row_value(row, "id")): row for row in message_rows}
        if set(by_id) != {objective["user_message_id"], objective["assistant_message_id"]}:
            raise AccountBindingError("durable objective messages were missing")
        user = by_id[objective["user_message_id"]]
        assistant = by_id[objective["assistant_message_id"]]
        if _row_value(user, "thread_id", "threadId") != objective["thread_id"] or _row_value(user, "role") != "user" or _row_value(assistant, "thread_id", "threadId") != objective["thread_id"] or _row_value(assistant, "parent_id", "parentId") != objective["user_message_id"] or _row_value(assistant, "role") != "assistant":
            raise AccountBindingError("durable message identity did not match")
        user_content = _parse_json(_row_value(user, "content_json", "contentJson"), "durable user content")
        if user_content != _canonical_user_message_content():
            raise ProtocolMismatchError("durable user message content did not match the pinned prompt")
        user_prompt = user_content[0]["text"]
        if _sha256_bytes(user_prompt.encode("utf-8")) != _sha256_bytes(DEFAULT_PROMPT.encode("utf-8")) or _prompt_content_sha256(user_prompt) != _prompt_content_sha256():
            raise ProtocolMismatchError("durable user message content hash did not match")
        metadata = _parse_json(_row_value(assistant, "metadata_json", "metadataJson") or "{}", "assistant metadata")
        if not isinstance(metadata, Mapping) or metadata.get("generationRunId") != objective["run_id"] or metadata.get("generationStatus") != "completed" or metadata.get("generationSettled") is not True or metadata.get("serverManaged") is not True or metadata.get("generationSeq") != _row_value(run, "last_event_seq", "lastEventSeq"):
            raise AccountBindingError("final assistant terminal binding was invalid")
        content = _parse_json(_row_value(assistant, "content_json", "contentJson"), "assistant content")
        if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], Mapping) or content[0].get("type") != "text" or not isinstance(content[0].get("text"), str):
            raise ProtocolMismatchError("durable final answer content was not exact text")
        answer_text = content[0]["text"]
        answer_assessment = _assess_answer_text(answer_text, fixture)
        answer_sha = str(answer_assessment["sha256"])
        answer_length = int(answer_assessment["length"])
        if bool(answer_assessment["exact"]) != (
            answer_sha == fixture.final_answer_sha256
            and answer_length == fixture.final_answer_length
        ):
            raise ProtocolMismatchError("durable final answer exactness classification was inconsistent")

        coverage = {
            "status": "derived",
            "provenance": "pinned_fixture_allowlisted_command_producer_output",
            "inputs": {
                "fixture_sha256": fixture.fixture_sha256,
                "manifest_sha256": fixture.manifest_sha256,
                "allowlist": "validate_tool_proposal",
                "producer_result_sha256": [item["result_sha256"] for item in execution_summaries],
            },
            "filesystem_access": "unknown",
            "filesystem_access_provenance": "not_measured",
            "operations": [
                {"command": item["command"], "paths": list(item["expected_paths"]), "byte_ranges": [dict(byte_range) for byte_range in item["byte_ranges"]]}
                for item in fixture.expected_sequence
            ],
        }
        return {
            "status": "pass",
            "provenance": "sqlite_ro_durable_ungated_v3_tool_evidence",
            "sqlite": {"mode": "ro", "query_only": True},
            "objective": {"owner_subject": OWNER_USERNAME, "account_id": account_id, "run_id": objective["run_id"], "thread_id": objective["thread_id"], "user_message_id": objective["user_message_id"], "assistant_message_id": objective["assistant_message_id"], "project_id": objective.get("project_id")},
            "user_message": {"content_sha256": _prompt_content_sha256(user_prompt), "prompt_sha256": _sha256_bytes(user_prompt.encode("utf-8"))},
            "request": {"payload_sha256": _sha256_json(request), "request_hash": request_hash, "summary": summary},
            "approvals": {"rows": 0, "events": 0},
            "approval_rows": 0,
            "approval_events": 0,
            "proposals": proposal_summaries,
            "executions": execution_summaries,
            "coverage": coverage,
            "trace": {"seq": trace_seq, "payload_sha256": _sha256_bytes(trace_bytes), "payload_length": len(trace_bytes), "invocation_counts": validated_trace["aggregation"]},
            "terminal_event": {"seq": terminal_seq, "event_type": "run.completed"},
            "final_answer": {
                "sha256": answer_sha,
                "length": answer_length,
                "exact": bool(answer_assessment["exact"]),
                "semantic_match": bool(answer_assessment["semantic_match"]),
            },
            "raw_evidence_retained": True,
            "zero_mutation": True,
        }
    finally:
        connection.close()
        after_db = _sha256_bytes(database.read_bytes())
        if before_db != after_db:
            raise ProtocolMismatchError("SQLite database changed during read-only verification")


def unavailable_metrics() -> dict[str, dict[str, dict[str, Any]]]:
    metrics = _SOURCE.unavailable_metrics()
    quality = metrics.setdefault("quality_and_work", {})
    quality.setdefault("total_tokens", {"value": None, "provenance": "unavailable", "reason": "not_exposed"})
    return metrics


def _metric(value: Any, *, provenance: str, exposure: Sequence[str], derivation: str | None = None) -> dict[str, Any]:
    metric: dict[str, Any] = {"value": value, "provenance": provenance, "exposure": list(exposure)}
    if derivation is not None:
        metric["derivation"] = derivation
    return metric


def collect_metrics(sse: Mapping[str, Any], *, elapsed_ms: float) -> dict[str, Any]:
    """Admit only metrics exposed by the public SSE replay.

    The source runner owns the usage/timing parser.  This wrapper adds the
    harness receive-clock observations and keeps unavailable values explicit.
    """

    collected = _SOURCE.collect_metrics(sse, elapsed_ms=elapsed_ms)
    if not isinstance(collected, Mapping) or not isinstance(collected.get("metrics"), Mapping):
        raise ProtocolMismatchError("source metric collector returned no metric vector")
    metrics = {str(group): dict(values) for group, values in collected["metrics"].items() if isinstance(values, Mapping)}
    quality = metrics.setdefault("quality_and_work", {})
    quality.setdefault("total_tokens", {"value": None, "provenance": "unavailable", "reason": "not_exposed"})
    latency = metrics.setdefault("latency", {})
    public_exposure = ["public_sse_replay", "harness_monotonic_receive_clock"]
    e2e = latency.get("end_to_end_task_wall_ms")
    if isinstance(e2e, Mapping) and e2e.get("value") is not None:
        latency["end_to_end_task_wall_ms"] = {
            **dict(e2e),
            "provenance": "measured",
            "exposure": sorted(set([*e2e.get("exposure", ()), *public_exposure])),
        }

    events = sse.get("events") if isinstance(sse, Mapping) else None
    if isinstance(events, list) and events:
        first_protocol_ms: float | None = None
        first_visible_ms: float | None = None
        for event in events:
            if not isinstance(event, Mapping):
                continue
            received = _safe_float(event.get("received_elapsed_ms"))
            if received is None or received < 0:
                continue
            if first_protocol_ms is None:
                first_protocol_ms = received
            payload = event.get("payload")
            if first_visible_ms is None and isinstance(payload, Mapping):
                for choice in payload.get("choices") or ():
                    if not isinstance(choice, Mapping):
                        continue
                    delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, Mapping) else None
                    if isinstance(content, str) and content:
                        first_visible_ms = received
                        break
        if first_protocol_ms is not None:
            latency["ttft_first_protocol_event_ms"] = _metric(first_protocol_ms, provenance="measured", exposure=public_exposure)
        if first_visible_ms is not None:
            latency["ttft_first_user_visible_token_ms"] = _metric(first_visible_ms, provenance="measured", exposure=public_exposure)

    usage = collected.get("usage")
    if isinstance(usage, Mapping):
        total_tokens = _safe_float(usage.get("total_tokens"))
        if total_tokens is not None and total_tokens >= 0:
            quality["total_tokens"] = _metric(
                int(total_tokens) if total_tokens.is_integer() else total_tokens,
                provenance="measured",
                exposure=["public_sse_usage"],
            )
    return {**dict(collected), "metrics": metrics}


def _admit_durable_tool_metrics(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    expected_calls: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Add tool metrics only after the SQLite receipt verifier has passed."""

    admitted = {str(group): dict(values) for group, values in metrics.items() if isinstance(values, Mapping)}
    quality = admitted.setdefault("quality_and_work", {})
    latency = admitted.setdefault("latency", {})
    receipt_exposure = ["sqlite:chat_generation_events", "sqlite:chat_generation_tool_executions"]
    executions = evidence.get("executions") if isinstance(evidence, Mapping) else None
    if not isinstance(executions, list) or len(executions) != expected_calls:
        return admitted
    quality["tool_calls"] = _metric(expected_calls, provenance="measured", exposure=receipt_exposure)
    # The verifier admits exactly one successful, non-ambiguous receipt for
    # every proposal; these zeros are therefore proven by the durable ledger.
    quality["failed_tool_calls"] = _metric(0, provenance="derived", exposure=receipt_exposure, derivation="all admitted receipts finished without error")
    quality["prevented_tool_calls"] = _metric(0, provenance="derived", exposure=receipt_exposure, derivation="all expected proposals were allowed and completed")
    quality["redundant_tool_calls"] = _metric(0, provenance="derived", exposure=receipt_exposure, derivation="one unique approval/execution receipt per fixture operation")
    durations: list[float] = []
    for execution in executions:
        if not isinstance(execution, Mapping):
            durations = []
            break
        started_at = _safe_float(execution.get("started_at"))
        finished_at = _safe_float(execution.get("finished_at"))
        if started_at is None or finished_at is None or started_at < 0 or finished_at < started_at:
            durations = []
            break
        durations.append(finished_at - started_at)
    if len(durations) == expected_calls:
        latency["tool_ms"] = _metric(
            sum(durations),
            provenance="derived",
            exposure=receipt_exposure,
            derivation="sum(finished_at - started_at) for validated durable receipts",
        )
    return admitted


def _unavailable_invocation_counts() -> dict[str, Any]:
    value = {"value": None, "provenance": "unavailable", "reason": "not_exposed"}
    return {name: value.copy() for name in ("model_invocations", "semantic_turns", "failed_invocations", "wasted_invocations", "escalations", "expensive_interventions")}


def _privacy_summary() -> dict[str, Any]:
    return {"raw_prompt_retained": False, "raw_answer_retained": False, "raw_sse_retained": False, "raw_tool_output_retained": False, "privacy_mode": "summaries_and_sha256_only"}


def _base_record(config: ToolScenarioConfig, fixture: FixtureSpec, *, status: str, source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance_sha = acceptance_criteria_sha256()
    configuration = {
        "scenario_id": SCENARIO_ID,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": fixture.fixture_sha256,
        "fixture_manifest_sha256": fixture.manifest_sha256,
        "prompt_sha256": _sha256_bytes(config.prompt.encode("utf-8")),
        "tool_contract": {**ACCEPTANCE_CRITERIA["request"], "network_disabled": True},
        "command_sequence_sha256": _sha256_json([
            {"arguments_sha256": item["arguments_sha256"], "proposal_payload_sha256": item["proposal_payload_sha256"]}
            for item in fixture.expected_sequence
        ]),
        "acceptance_criteria_sha256": acceptance_sha,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "runner_version": SCRIPT_VERSION,
        "artifact_role": "baseline_control_only",
        "benchmark_id": "helix-v3-tool-scenarios",
        "scenario_id": SCENARIO_ID,
        "status": status,
        "offline": True,
        "candidate_behavior": False,
        "source_identity": {"before": dict(source_identity) if source_identity else None, "after": None},
        "fixture": {"fixture_id": FIXTURE_ID, "manifest_sha256": fixture.manifest_sha256, "before_sha256": fixture.fixture_sha256, "after_sha256": None, "content_addressed": True},
        "configuration": configuration,
        "acceptance": {"criteria_sha256": acceptance_sha, "bound": True},
        "benchmark_metadata": {
            "observation_pack": {"enabled": False, "provenance": "disabled_for_control_benchmark"},
            "action_fusion": {"enabled": False, "provenance": "disabled_for_control_benchmark"},
        },
        "tool_contract": {"name": TOOL_NAME, "enabled_tools": [TOOL_NAME], "enable_tools": True, "mcp_enabled": False, "network_disabled": True, "bypass_permissions": False, "permission_mode": "auto", "confirm_tool_calls": True},
        "metrics": unavailable_metrics(),
        "invocation_counts": _unavailable_invocation_counts(),
        "correctness": {"outcome": "unverified", "evidence_refs": []},
        "mandatory_flags": {"read_only_fixture": True, "backend_owned_receipt": False, "exact_objective_binding": False, "h2_repo_eligible": False, "raw_evidence_retained": False},
        "negative_flags": {"write_or_mutation_seen": False, "network_or_mcp_seen": False, "unbound_evidence": False, "model_only_success": True, "fake_execution": False, "ambiguous_or_missing_read_receipt": True},
        "privacy": _privacy_summary(),
    }


def atomic_artifact(output_dir: Path, record: Mapping[str, Any]) -> Path:
    directory = _require_abs_directory(output_dir, "output directory")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = directory / f"p1-ro-repo-{stamp}-{os.getpid()}.json"
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(directory))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        try:
            # A hard-link publish is atomic and, unlike os.replace, refuses to
            # overwrite a pre-existing artifact target.
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ContractError("artifact target already exists; refusing to overwrite") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _tool_launch_command(clone: Any, config: ToolScenarioConfig, port: int) -> list[str]:
    # Keep the approved source command's API-only/loopback/password shape but
    # intentionally omit --disable-tools for this tool scenario.
    return [
        str(clone.venv_python), str(Path(config.repo_root).resolve() / "studio" / "backend" / "run.py"),
        "--host", "127.0.0.1", "--port", str(int(port)), "--api-only", "--no-cloudflare", "--silent", "--password", "-",
    ]


def _tool_launch_environment(clone: Any, config: ToolScenarioConfig) -> dict[str, str]:
    environment = _SOURCE.build_environment(clone.run_dir, Path(config.repo_root).resolve(), provider_tier=config.provider_tier)
    # The authenticated project API resolves its managed workspace from this
    # clone-local root.  The returned rootPath/sandboxPath are still checked
    # after creation; this variable is not trusted as proof by itself.
    environment["UNSLOTH_STUDIO_PROJECTS_HOME"] = str(Path(clone.run_dir).resolve() / "projects")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _launch_tool_runtime(clone: Any, config: ToolScenarioConfig, port: int, *, password: str, popen_factory: Callable[..., Any] | None = None) -> Any:
    command = _tool_launch_command(clone, config, port)
    process = (subprocess.Popen if popen_factory is None else popen_factory)(
        command,
        cwd=str(Path(config.repo_root).resolve()),
        env=_tool_launch_environment(clone, config),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    stream = getattr(process, "stdin", None)
    if stream is None:
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        raise HarnessError("tool runtime did not expose stdin")
    try:
        stream.write(password + "\n")
        stream.flush()
        stream.close()
    except Exception as exc:
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        raise HarnessError("tool runtime password handoff failed") from exc
    return process


def _failure_code(exc: BaseException) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", type(exc).__name__.casefold()).strip("_") or "harness_error"


def _dry_run(config: ToolScenarioConfig, fixture: FixtureSpec, source_identity: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, Any]:
    result = _base_record(config, fixture, status="dry_run", source_identity=source_identity)
    result.update({"would_launch": False, "actions": ["APFS COW clone", "tool-enabled source run.py launch without --disable-tools", "owner login", "model load", "durable objective and terminal tool run", "owner unload", "identity-fenced shutdown", "offline SQLite read-only verification"]})
    result["runtime"] = {"launch_command_policy": "source API-only command without --disable-tools", "run_dir": _path_identity_summary(Path(paths["run_dir"]))}
    return result


def _validate_config_and_fixture(config: ToolScenarioConfig) -> tuple[dict[str, Any], FixtureSpec, dict[str, Any]]:
    source_paths = _SOURCE.validate_paths(_source_config(config), execute=config.execute)
    if Path(source_paths["run_dir"]).exists():
        raise ContractError("run directory must be absent before a tool scenario")
    fixture = load_fixture(config.fixture_root)
    run_path = Path(source_paths["run_dir"]).resolve(strict=False)
    output_path = Path(source_paths["output_dir"]).resolve(strict=False)
    if _paths_overlap(fixture.root, run_path):
        raise ContractError("run directory and fixture must not overlap")
    if _paths_overlap(fixture.root, output_path):
        raise ContractError("output directory and fixture must not overlap")
    source_identity = _SOURCE._source_identity(Path(source_paths["repo"]))
    return source_paths, fixture, source_identity


def run_benchmark(
    config: ToolScenarioConfig,
    *,
    transport: Any | None = None,
    popen_factory: Callable[..., Any] | None = None,
    command_runner: Callable[..., Any] | None = None,
    copy_runner: Callable[..., Any] | None = None,
    source_identity_runner: Callable[..., Any] | None = None,
    port_allocator: Callable[[], int] | None = None,
    session_id_getter: Callable[[int], int] | None = None,
    waitid_fn: Callable[..., Any] | None = None,
    signal_getter: Callable[[int], Any] | None = None,
    group_signal_sender: Callable[[int, int], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    h2_evidence_gate: bool = False,
) -> dict[str, Any]:
    # Source preflight accepts an injected identity runner only through the
    # source run call, so preserve the exact approved helper boundary here.
    source_paths = _SOURCE.validate_paths(_source_config(config), execute=config.execute)
    if Path(source_paths["run_dir"]).exists():
        raise ContractError("run directory must be absent before a tool scenario")
    fixture = load_fixture(config.fixture_root)
    if _paths_overlap(fixture.root, Path(source_paths["run_dir"]).resolve(strict=False)):
        raise ContractError("run directory and fixture must not overlap")
    if _paths_overlap(fixture.root, Path(source_paths["output_dir"]).resolve(strict=False)):
        raise ContractError("output directory and fixture must not overlap")
    before_source = _SOURCE._source_identity(Path(source_paths["repo"]), command_runner=source_identity_runner)
    if not config.execute:
        return _dry_run(config, fixture, before_source, source_paths)
    if config.model_path is None or config.model_config_path is None:
        raise ContractError("execute requires a local model and config")
    model_spec = _SOURCE.validate_offline_model_spec(config.model_path, config.model_config_path)
    load_payload = _SOURCE.build_load_payload(model_spec, seed=config.seed)
    clone: Any | None = None
    process: Any | None = None
    password: str | None = None
    headers: dict[str, str] | None = None
    base_url: str | None = None
    port: int | None = None
    root_pid: int | None = None
    owned: dict[int, dict[str, Any]] = {}
    owned_pids: set[int] = set()
    identity_conflicts: set[int] = set()
    owned_pgid: int | None = None
    objective: dict[str, Any] | None = None
    project: dict[str, Any] | None = None
    staged_fixture: dict[str, Any] | None = None
    staged_fixture_after: dict[str, Any] | None = None
    admitted_model = str(model_spec["model_path"])
    loaded = False
    unloaded = False
    shutdown_admitted = False
    shutdown_requested = False
    cleanup: dict[str, Any] = {"status": "unavailable", "provenance": "not_started"}
    record = _base_record(config, fixture, status="running", source_identity=before_source)
    fixture_before = fixture_tree_hash(fixture.root)
    if fixture_before != fixture.fixture_sha256:
        raise ContractError("fixture changed after manifest preflight")
    record["fixture"]["before_sha256"] = fixture_before
    record["model"] = {"name": config.model_name, "provider": config.provider_identity, "backend": "mlx", "offline": True, "config_sha256": _sha256_bytes(config.model_config_path.read_bytes())}
    record["configuration"]["load_sha256"] = _sha256_json(load_payload)
    safe_secrets: list[str] = []
    http = transport or _SOURCE.UrllibTransport()

    def fail(exc: BaseException | str) -> None:
        record["status"] = "failed"
        record["correctness"] = {"outcome": "unverified", "evidence_refs": []}
        record["error"] = {"code": _failure_code(exc) if isinstance(exc, BaseException) else str(exc), "classification": "lifecycle_or_protocol_failure"}

    try:
        clone = _SOURCE.clone_runtime_home(Path(source_paths["source"]), Path(source_paths["run_dir"]), copy_runner=copy_runner)
        current_password = _SOURCE.read_password(_SOURCE._password_path(clone.run_dir))
        safe_secrets.append(current_password)
        password = _SOURCE.ephemeral_launch_password(current_password)
        safe_secrets.append(password)
        port = int((port_allocator or _SOURCE.allocate_ephemeral_port)())
        if not 1 <= port <= 65535:
            raise ContractError("ephemeral port was invalid")
        ready, reason = _SOURCE._reaper_boundary_status(signal_getter=signal_getter)
        if not ready:
            raise IdentityMismatchError(f"direct-child reaper boundary unavailable: {reason}")
        expected_command = _tool_launch_command(clone, config, port)
        process = _launch_tool_runtime(clone, config, port, password=password, popen_factory=popen_factory)
        root_pid = int(getattr(process, "pid"))
        incoming = _SOURCE.capture_process_identities(root_pid, command_runner=command_runner, session_id_getter=session_id_getter)
        root_identity = incoming.get(root_pid)
        _SOURCE._validate_source_process_identity(root_identity, expected_command, Path(source_paths["repo"]))
        root_pgid = root_identity.get("pgid") if isinstance(root_identity, Mapping) else None
        valid, reason = _SOURCE._validate_owned_process_group(root_pid, root_pgid if isinstance(root_pgid, int) else None, root_identity)
        if not valid:
            raise IdentityMismatchError("tool runtime process group was not identity-fenced")
        owned_pgid = root_pgid
        identity_conflicts.update(_SOURCE._merge_process_identities(owned, incoming))
        owned_pids.update(owned)
        if identity_conflicts:
            raise IdentityMismatchError("tool runtime process identity was reused")
        base_url = f"http://127.0.0.1:{port}"
        record["health_identity"] = _SOURCE._wait_for_health(http, base_url, timeout=config.startup_timeout_s, process=process, port=port, owned_pids=owned_pids, command_runner=command_runner)
        record["health_binding"] = {"loopback": base_url, "listener_verified": True, "launch_command_sha256": _sha256_json(expected_command), "tools_enabled": True}
        headers, auth = login_owner(http, base_url, password, config.request_timeout_s)
        record["auth"] = auth
        project = create_managed_project(
            http,
            base_url,
            headers,
            Path(clone.run_dir),
            timeout=config.request_timeout_s,
        )
        staged_fixture = stage_fixture_in_project(project, fixture)
        record["project"] = {
            "project_id": project["project_id"],
            "session_id_sha256": _sha256_bytes(f"project-{project['project_id']}".encode("utf-8")),
            "identity": project["identity"],
            "staged_fixture": staged_fixture,
        }
        load_response = _json_request(http, "POST", base_url, "/api/inference/load", headers=headers, payload=load_payload, timeout=config.request_timeout_s)
        if _SOURCE._operation_failed(load_response):
            raise HarnessError("local model load failed")
        loaded = True
        admitted_model = _SOURCE.select_model_request_name(load_response, model_spec["model_path"])
        public_model = _SOURCE.derive_public_model_identity(model_spec["model_path"], model_spec["config"])
        record["model"]["name"] = public_model
        record["load"] = {"status": load_response.get("status") if isinstance(load_response, Mapping) else None, "model_sha256": _sha256_bytes(str(admitted_model).encode()), "public_model_id": public_model}
        request_config = replace(config, model_name=public_model)
        objective = create_tool_objective(http, base_url, headers, request_config, fixture, project)
        request_payload = objective["request_payload"]
        record["objective"] = {"owner_subject": OWNER_USERNAME, "account_id": OWNER_ACCOUNT_ID, "run_id": objective["run_id"], "thread_id": objective["thread_id"], "user_message_id": objective["user_message_id"], "assistant_message_id": objective["assistant_message_id"], "request_hash": objective["request_hash"], "user_prompt_sha256": objective["user_prompt_sha256"], "user_message_content_sha256": objective["user_message_content_sha256"], "request_summary": summarize_request_payload(request_payload)}
        started = clock()
        sse = consume_tool_sse(http, base_url, headers, objective["run_id"], timeout=config.request_timeout_s, fixture=fixture, session_id=objective["session_id"], permission_mode=request_payload.get("permission_mode", ""), clock=clock)
        public_answer_text = _observed_answer_text(sse)
        answer_assessment = _assess_answer_text(public_answer_text, fixture)
        record["public_replay"] = {"event_count": sse["event_count"], "wire_bytes": sse["wire_bytes"], "event_cursor": sse["last_event_seq"], "proposal_count": len(sse.get("proposals") or ()), "private_trace_suppressed": True, "answer_sha256": answer_assessment["sha256"], "answer_length": answer_assessment["length"], "answer_exact": answer_assessment["exact"], "answer_semantic_match": answer_assessment["semantic_match"]}
        metric_record = collect_metrics(sse, elapsed_ms=max(0.0, (clock() - started) * 1000.0))
        record["metrics"] = metric_record["metrics"]
        record["metric_exposure"] = {"source": "public_sse", "usage": metric_record.get("usage"), "visible_output_chars": metric_record.get("visible_output_chars"), "visible_output_sha256": metric_record.get("visible_output_sha256")}
        record["assistant_persistence"] = persist_structured_answer(http, base_url, headers, objective=objective, sse=sse, answer={}, answer_text=public_answer_text, timeout=config.request_timeout_s)
        record["configuration"]["request_payload_sha256"] = _sha256_json(request_payload)
    except Exception as exc:
        fail(exc)
    finally:
        if loaded and not unloaded and base_url is not None and headers is not None:
            try:
                unload = _json_request(http, "POST", base_url, "/api/inference/unload", headers=headers, payload={"model_path": str(model_spec["model_path"]), "force_cancel_active": False}, timeout=config.request_timeout_s)
                if _SOURCE._operation_failed(unload):
                    raise HarnessError("local model unload failed")
                _SOURCE.validate_unload_response(unload, admitted_model_name=str(admitted_model), model_path=model_spec["model_path"])
                unloaded = True
                record["unload"] = {"status": unload.get("status") if isinstance(unload, Mapping) else None}
            except Exception as exc:
                record["unload"] = {"status": "error"}
                fail(exc)
        if base_url is not None and headers is not None and not shutdown_requested:
            try:
                shutdown = _json_request(http, "POST", base_url, "/api/shutdown", headers=headers, timeout=config.request_timeout_s)
                shutdown_requested = True
                if _SOURCE._operation_failed(shutdown):
                    raise HarnessError("tool runtime shutdown failed")
                shutdown_admitted = True
                record["shutdown"] = {"status": shutdown.get("status") if isinstance(shutdown, Mapping) else None, "ok": shutdown.get("ok") if isinstance(shutdown, Mapping) else None}
            except Exception as exc:
                shutdown_requested = True
                fail(exc)
        if process is not None and root_pid is not None:
            current = _SOURCE.capture_process_identities(root_pid, command_runner=command_runner, session_id_getter=session_id_getter)
            if current:
                identity_conflicts.update(_SOURCE._merge_process_identities(owned, current))
                owned_pids.update(owned)
            if port is not None:
                cleanup = _SOURCE.verify_cleanup(root_pid, owned_pids, port, command_runner=command_runner, owned_identities=owned, identity_conflicts=identity_conflicts, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter)
            if getattr(process, "poll", lambda: 0)() is None or cleanup.get("status") != "pass":
                termination = _SOURCE.terminate_owned_processes(process, owned_pids, owned_identities=owned, identity_conflicts=identity_conflicts, command_runner=command_runner, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter, group_signal_sender=group_signal_sender)
                cleanup = {**cleanup, "termination": termination}
                if port is not None:
                    cleanup = {**_SOURCE.verify_cleanup(root_pid, owned_pids, port, command_runner=command_runner, owned_identities=owned, identity_conflicts=identity_conflicts, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter), "termination": termination}
            process_reaped = getattr(process, "poll", lambda: None)() is not None
            if cleanup.get("status") != "pass" or not process_reaped:
                fail("cleanup_uncertain")
            elif shutdown_admitted and record.get("status") == "running" and objective is not None and clone is not None:
                try:
                    evidence = read_tool_durable_evidence(
                        clone.studio_home / "studio.db",
                        objective=objective,
                        fixture=fixture,
                        request_payload=objective["request_payload"],
                        provider_tier=config.provider_tier,
                        model_name=record["model"]["name"],
                        provider_identity=config.provider_identity,
                    )
                    record["durable_evidence"] = evidence
                    record["invocation_counts"] = evidence["trace"]["invocation_counts"]
                    record["metrics"] = _admit_durable_tool_metrics(record.get("metrics", unavailable_metrics()), evidence, expected_calls=len(fixture.expected_sequence))
                    task_success = bool(evidence["final_answer"].get("exact"))
                    record["correctness"] = {
                        "outcome": "pass" if task_success else "fail",
                        "reason": None if task_success else "exact_output_mismatch",
                        "semantic_match": bool(evidence["final_answer"].get("semantic_match")),
                        "evidence_refs": ["sqlite-durable-tool-evidence"],
                    }
                    required_metric_names = (
                        ("latency", "end_to_end_task_wall_ms"),
                        ("quality_and_work", "total_tokens"),
                        ("quality_and_work", "tool_calls"),
                        ("quality_and_work", "failed_tool_calls"),
                        ("quality_and_work", "prevented_tool_calls"),
                        ("quality_and_work", "redundant_tool_calls"),
                        ("latency", "tool_ms"),
                    )
                    metrics_admitted = all(
                        isinstance(record["metrics"].get(group, {}).get(name), Mapping)
                        and record["metrics"][group][name].get("value") is not None
                        and record["metrics"][group][name].get("provenance") not in {None, "unavailable"}
                        for group, name in required_metric_names
                    )
                    record["metric_admission"] = {"required": [f"{group}.{name}" for group, name in required_metric_names], "admitted": metrics_admitted, "h2_gate_requested": bool(h2_evidence_gate)}
                    record["mandatory_flags"].update({"backend_owned_receipt": True, "exact_objective_binding": True, "h2_repo_eligible": bool(h2_evidence_gate and metrics_admitted and task_success), "raw_evidence_retained": True})
                    record["negative_flags"].update({"model_only_success": False, "ambiguous_or_missing_read_receipt": False})
                    record["status"] = "complete"
                except Exception as exc:
                    fail(exc)
        record["cleanup"] = {"status": cleanup.get("status"), "provenance": cleanup.get("provenance"), "retained_on_uncertainty": cleanup.get("status") != "pass"}
        try:
            after_source = _SOURCE._source_identity(Path(source_paths["repo"]), command_runner=source_identity_runner)
            _SOURCE.verify_source_identity(before_source, after_source)
            record["source_identity"]["after"] = after_source
        except Exception as exc:
            record["source_identity"]["after_error"] = _failure_code(exc)
            fail(exc)
        after_fixture = fixture_tree_hash(fixture.root)
        record["fixture"]["after_sha256"] = after_fixture
        if after_fixture != fixture.fixture_sha256:
            record["negative_flags"]["write_or_mutation_seen"] = True
            fail("fixture_mutation_detected")
        staged_ok = True
        if project is not None:
            try:
                staged_fixture_after = verify_staged_fixture(project, fixture)
                record.setdefault("project", {})["staged_fixture_after"] = staged_fixture_after
            except Exception as exc:
                staged_ok = False
                record.setdefault("project", {})["staged_fixture_after_error"] = _failure_code(exc)
                record["negative_flags"]["write_or_mutation_seen"] = True
                fail("staged_fixture_mutation")
        removed = False
        if clone is not None:
            cleanup_proven = record.get("status") == "complete" and cleanup.get("status") == "pass" and after_fixture == fixture.fixture_sha256 and staged_ok
            try:
                if cleanup_proven and project is not None:
                    record.setdefault("project", {})["cleanup_permissions"] = prepare_staged_fixture_for_cleanup(project, fixture)
                removed = _SOURCE.remove_clone_if_proven(clone, cleanup_proven=cleanup_proven)
            except Exception as exc:
                record.setdefault("runtime", {})["clone_remove_error"] = type(exc).__name__
        record.setdefault("runtime", {})["clone_removed"] = removed
        if not removed and clone is not None:
            record["runtime"]["clone_retained"] = True
        if record.get("status") == "running":
            fail("lifecycle_incomplete")
        artifact = atomic_artifact(Path(source_paths["output_dir"]), record)
        record["artifact_path"] = str(artifact)
    return record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-home-template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "helix-v3" / "phase1" / "fixtures" / FIXTURE_ID)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--password-file", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--provider-tier", default=DEFAULT_PROVIDER_TIER)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ToolScenarioConfig:
    return ToolScenarioConfig(runtime_home_template=args.runtime_home_template, run_dir=args.run_dir, output_dir=args.output_dir, repo_root=args.repo_root, fixture_root=args.fixture_root, password_file=args.password_file, model_name=args.model_name, model_path=args.model_path, model_config_path=args.model_config, prompt=args.prompt, provider_tier=args.provider_tier, startup_timeout_s=args.startup_timeout, request_timeout_s=args.request_timeout, seed=args.seed, max_tokens=args.max_tokens, execute=bool(args.execute))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_benchmark(_config_from_args(_parse_args(argv)))
    except Exception as exc:
        result = {"schema_version": BENCHMARK_SCHEMA, "runner_version": SCRIPT_VERSION, "artifact_role": "baseline_control_only", "status": "error", "error": {"code": _failure_code(exc)}}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") in {"dry_run", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
