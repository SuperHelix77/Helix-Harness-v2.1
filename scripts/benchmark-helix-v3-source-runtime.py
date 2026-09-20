#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Secret-safe Phase 1 benchmark runner for the source Helix runtime.

The runner is intentionally opt-in.  Its default is a read-only contract
preflight; ``--execute`` is required before it clones or starts a server.  A
run uses an APFS copy-on-write clone of an explicitly supplied frozen runtime
home, starts the source backend from the clone's managed venv, and validates
the durable run and private optimization trace from a standalone read-only
SQLite connection after shutdown.  No backend database modules are imported
by this script.

The source benchmark is deliberately bounded to one no-tool local request.
Unavailable observations are represented as ``null`` with provenance rather
than fabricated zeros.  The durable trace is evidence, not a model-authored
claim, and is never sent to the frontend SSE consumer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import datetime as _datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


SCRIPT_VERSION = "helix-v3-source-runtime.v1"
BENCHMARK_SCHEMA = "helix.optimization-benchmark.v1"
TRACE_EVENT_SCHEMA = "helix.optimization-trace-event.v1"
TRACE_SCHEMA = "helix.optimization-trace.v1"
TRACE_PRIVATE_KEY = "_helix_optimization_trace_v1"
TRACE_EVENT_TYPE = "optimization.trace"
EXPECTED_BACKEND_CONTRACT = "helix.adaptive.backend.v1"
MAX_TRACE_EVENTS = 10_000
MAX_TRACE_WIRE_BYTES = 4 * 1024 * 1024
# Frontend replay is an untrusted transport boundary.  Keep each individual
# SSE line/block and the whole response bounded before retaining any decoded
# payload.  These limits are deliberately independent of the durable trace
# limits above: the public stream must never become an unbounded accumulator.
MAX_SSE_EVENTS = 10_000
MAX_SSE_WIRE_BYTES = 4 * 1024 * 1024
MAX_SSE_BLOCK_BYTES = 1 * 1024 * 1024
MAX_SSE_LINE_BYTES = 256 * 1024
MAX_SSE_JSON_DEPTH = 32
OWNER_USERNAME = "unsloth"
OWNER_ACCOUNT_ID = "owner"
DEFAULT_PROVIDER_TIER = "local"
DEFAULT_PROVIDER_IDENTITY = "local:mlx"
DEFAULT_MODEL_NAME = "local"
DEFAULT_PROMPT = "Reply with exactly HELIX_SOURCE_RUNTIME_OK and no other text."
EXPECTED_ANSWER = "HELIX_SOURCE_RUNTIME_OK"
DEFAULT_CONTEXT_LENGTH = 4096
COMMAND_TIMEOUT_S = 5.0
CLONE_COMMAND_TIMEOUT_S = 120.0
_PS_LSTART_RE = re.compile(
    r"^[A-Za-z]{3} [A-Za-z]{3} [0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}$"
)
_TIERS = ("tiny", "local", "senior", "frontier")
_SCOPES = ("foreground", "background")
_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed"})
_PRIVATE_KEYS = frozenset(
    {
        TRACE_PRIVATE_KEY,
        "optimization.trace",
        "optimization_trace",
        "access_token",
        "refresh_token",
        "authorization",
        "password",
        "secret",
        "token",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|authorization|access[_-]?token|refresh[_-]?token|"
    r"api[_-]?key|private[_-]?key|secret|credential|cookie)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}")


class HarnessError(RuntimeError):
    """Base class for fail-closed benchmark errors."""


class ContractError(HarnessError):
    """A supplied path, identity, protocol, or durable binding is invalid."""


class IdentityMismatchError(ContractError):
    """The source or loopback server identity changed or was not verified."""


class ProtocolMismatchError(ContractError):
    """A public SSE or private durable trace contract was violated."""


class AccountBindingError(ContractError):
    """A run, thread, message, or trace crossed the owner account boundary."""


class OutputPathError(ContractError):
    """An artifact path would escape the explicit output directory."""


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    payload: Any
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """All run inputs.  Paths are explicit so a benchmark cannot discover an app."""

    runtime_home_template: Path
    run_dir: Path
    output_dir: Path
    repo_root: Path
    password_file: Path | None = None
    model_name: str = DEFAULT_MODEL_NAME
    model_path: Path | None = None
    model_config_path: Path | None = None
    prompt: str = DEFAULT_PROMPT
    provider_tier: str = DEFAULT_PROVIDER_TIER
    provider_identity: str = DEFAULT_PROVIDER_IDENTITY
    startup_timeout_s: float = 60.0
    request_timeout_s: float = 300.0
    seed: int = 20260920
    max_tokens: int = 32
    execute: bool = False


@dataclass(frozen=True)
class CloneHandle:
    source: Path
    run_dir: Path
    studio_home: Path
    venv_python: Path
    source_manifest: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _sanitize(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Redact secret values and fields before anything is printed or persisted."""

    known = tuple(sorted({item for item in secrets if isinstance(item, str) and item}, key=len, reverse=True))

    def text(item: str) -> str:
        result = item
        for secret in known:
            result = result.replace(secret, "[REDACTED]")
        return _BEARER_RE.sub(r"\1[REDACTED]", result)

    if isinstance(value, str):
        return text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, nested in value.items():
            key = str(raw_key)
            result[key] = "[REDACTED]" if _SECRET_KEY_RE.search(key) else _sanitize(nested, known)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, known) for item in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return text(str(value))


def _require_abs_directory(path: Path, label: str, *, exists: bool = False) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    resolved = requested.resolve(strict=False)
    if exists and not resolved.is_dir():
        raise ContractError(f"{label} does not exist: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ContractError(f"{label} is not a directory: {resolved}")
    return resolved


def _require_regular_file(path: Path, label: str, *, exists: bool = True) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    resolved = requested.resolve(strict=False)
    if exists and (not resolved.is_file() or resolved.is_symlink()):
        raise ContractError(f"{label} is not a regular file: {resolved}")
    return resolved


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _venv_python(home: Path, *, require_exists: bool = True) -> Path:
    candidates = (
        home / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python",
        home / ".unsloth" / "studio" / "unsloth_studio" / "bin" / "python3",
        home / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe",
        home / ".unsloth" / "studio" / "venv" / "bin" / "python",
        home / ".unsloth" / "studio" / "venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
        # A managed venv commonly exposes Python as a symlink to its shared
        # interpreter.  It is safe to use; the clone owns HOME and its state.
        if candidate.is_symlink() and candidate.resolve(strict=False).is_file():
            return candidate
    if require_exists:
        raise ContractError("frozen runtime home has no managed venv Python")
    return candidates[0]


def _password_path(home: Path) -> Path:
    candidates = (
        home / ".unsloth" / "studio" / "auth" / ".bootstrap_password",
        home / ".unsloth" / "studio" / ".bootstrap_password",
    )
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return candidates[0]


def _validate_source_manifest_absent(repo_root: Path) -> Path:
    """A source run must not carry the installed overlay identity manifest."""

    manifest = repo_root / "studio" / "backend" / ".helix_backend_manifest.json"
    if manifest.exists() or manifest.is_symlink():
        raise IdentityMismatchError(
            "source backend unexpectedly contained a packaged overlay manifest"
        )
    return manifest


def read_password(path: Path) -> str:
    """Read the bootstrap password into memory; never return it in a record."""

    if path.is_symlink() or not path.is_file():
        raise ContractError("password file is not a regular file")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ContractError("password file could not be read") from exc
    if not value or any(char.isspace() for char in value):
        raise ContractError("password file is empty or contains whitespace")
    return value


def ephemeral_launch_password(current_password: str) -> str:
    """Generate a clone-local password distinct from the frozen bootstrap value."""

    for _ in range(4):
        candidate = "helix-v3-" + secrets.token_urlsafe(32)
        if candidate != current_password and not any(char.isspace() for char in candidate):
            return candidate
    raise ContractError("could not generate a distinct clone-local launch password")


def _file_manifest(root: Path) -> str:
    """Digest a runtime-home tree without following symlinks."""

    digest = hashlib.sha256()
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except OSError as exc:
        raise ContractError("could not inspect runtime-home source") from exc
    for entry in entries:
        relative = entry.relative_to(root).as_posix()
        try:
            stat = entry.lstat()
            if entry.is_symlink():
                target = os.readlink(entry)
                digest.update(f"L\0{relative}\0{target}\n".encode("utf-8", "surrogateescape"))
            elif entry.is_file():
                digest.update(f"F\0{relative}\0{stat.st_size}\0".encode())
                with entry.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                digest.update(b"\n")
            elif entry.is_dir():
                digest.update(f"D\0{relative}\n".encode())
        except OSError as exc:
            raise ContractError(f"could not inspect runtime-home entry {relative}") from exc
    return digest.hexdigest()


def validate_paths(config: RuntimeConfig, *, execute: bool | None = None) -> dict[str, Path | str | None]:
    source = _require_abs_directory(config.runtime_home_template, "runtime-home template", exists=True)
    run_dir = _require_abs_directory(config.run_dir, "run directory")
    output_dir = _require_abs_directory(config.output_dir, "output directory")
    repo = _require_abs_directory(config.repo_root, "repository root", exists=True)
    if source == run_dir or _within(run_dir, source):
        raise ContractError("run directory must not be the runtime-home source or a child of it")
    if _within(output_dir, source) or _within(output_dir, run_dir):
        raise ContractError("output directory must not be inside the source or run directory")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ContractError("run directory must not already contain files")
    backend_run = _require_regular_file(repo / "studio" / "backend" / "run.py", "source run.py")
    identity_script = _require_regular_file(repo / "scripts" / "helix-v3-source-identity.py", "source identity helper")
    source_manifest = _validate_source_manifest_absent(repo)
    template_python = _venv_python(source)
    password = config.password_file.resolve(strict=False) if config.password_file else _password_path(source)
    _require_regular_file(password, "password file")
    effective_execute = config.execute if execute is None else execute
    if effective_execute and (config.model_path is None or config.model_config_path is None):
        raise ContractError("execute requires both a local model path and model config")
    if config.model_path is not None:
        _require_abs_directory(config.model_path, "model path", exists=True)
    if config.model_config_path is not None:
        _require_regular_file(config.model_config_path, "model config")
    if effective_execute:
        # Validate the exact local model inputs before cloning/launching.  The
        # returned spec is intentionally discarded here; run_benchmark builds
        # the same spec again immediately before /inference/load and records
        # its authoritative load response.
        validate_offline_model_spec(Path(config.model_path), Path(config.model_config_path))
    if not config.model_name or not isinstance(config.model_name, str):
        raise ContractError("model name must be non-empty")
    if not config.prompt or not config.prompt.strip():
        raise ContractError("prompt must be non-empty")
    if config.provider_tier not in _TIERS:
        raise ContractError(f"provider tier must be one of {', '.join(_TIERS)}")
    if config.provider_tier != "local":
        raise ContractError("source runtime trace must use the local provider tier")
    if config.provider_identity != DEFAULT_PROVIDER_IDENTITY:
        raise ContractError("source runtime trace must use provider identity local:mlx")
    if config.startup_timeout_s <= 0 or config.request_timeout_s <= 0:
        raise ContractError("timeouts must be positive")
    if isinstance(config.max_tokens, bool) or config.max_tokens < 1:
        raise ContractError("max-tokens must be positive")
    return {
        "source": source,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "repo": repo,
        "backend_run": backend_run,
        "identity_script": identity_script,
        "source_manifest": source_manifest,
        "template_python": template_python,
        "password": password,
        "execute": config.execute if execute is None else execute,
    }


def build_environment(
    home: Path,
    repo_root: Path,
    *,
    provider_tier: str = DEFAULT_PROVIDER_TIER,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an allowlisted, offline child environment with clone-local state."""

    source = os.environ if base is None else base
    allow = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ", "USER", "LOGNAME")
    env = {key: str(source[key]) for key in allow if key in source and str(source[key])}
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    studio_home = home / ".unsloth" / "studio"
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "UNSLOTH_STUDIO_HOME": str(studio_home),
            "PYTHONPATH": str(repo_root / "studio" / "backend"),
            "HELIX_OPTIMIZATION_TRACE_V1": "1",
            "HELIX_OPTIMIZATION_TRACE_PROVIDER_TIER": provider_tier,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "UNSLOTH_OFFLINE_PROBE": "1",
            "UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_NO_INDEX": "1",
            "UV_NO_INDEX": "1",
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
        }
    )
    for key, relative in {
        "TMPDIR": "tmp",
        "XDG_CONFIG_HOME": ".config",
        "XDG_CACHE_HOME": ".cache",
        "XDG_DATA_HOME": ".local/share",
        "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub",
        "HF_DATASETS_CACHE": "hf/datasets",
        "HF_XET_CACHE": "hf/xet",
        "XET_CACHE": "hf/xet",
        "UV_CACHE_DIR": ".cache/uv",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
        "TRITON_CACHE_DIR": ".cache/triton",
        "MLX_METAL_CACHE_DIR": ".cache/mlx",
        "PIP_CACHE_DIR": ".cache/pip",
    }.items():
        path = home / relative
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def clone_runtime_home(
    template: Path,
    run_dir: Path,
    *,
    copy_runner: Callable[..., Any] | None = None,
    system: str | None = None,
) -> CloneHandle:
    """APFS clone *template* into the exact explicit *run_dir*."""

    if (system or platform.system()) != "Darwin":
        raise ContractError("source runtime execute requires Darwin APFS COW cloning")
    source = _require_abs_directory(template, "runtime-home template", exists=True)
    requested_destination = Path(run_dir).expanduser()
    if requested_destination.is_symlink():
        raise ContractError("run directory must not be a symlink")
    destination = _require_abs_directory(requested_destination, "run directory")
    if destination.exists():
        if any(destination.iterdir()):
            raise ContractError("run directory must not already contain files")
        try:
            destination.rmdir()
        except OSError as exc:
            raise ContractError("run directory could not be prepared") from exc
    if source == destination or _within(destination, source):
        raise ContractError("refusing to clone inside the source runtime home")
    try:
        source_dev = source.stat().st_dev
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.stat().st_dev != source_dev:
            raise ContractError("runtime-home source and run directory must share one APFS volume")
    except OSError as exc:
        raise ContractError("could not inspect APFS runtime-home volume") from exc
    before = _file_manifest(source)
    runner = subprocess.run if copy_runner is None else copy_runner
    try:
        completed = runner(
            ["/bin/cp", "-cR", str(source), str(destination)],
            capture_output=True,
            text=True,
            check=False,
            timeout=CLONE_COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError, TypeError) as exc:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        raise ContractError("APFS COW runtime-home clone failed") from exc
    if getattr(completed, "returncode", 0) != 0 or not destination.is_dir():
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        raise ContractError("APFS COW runtime-home clone failed")
    after = _file_manifest(source)
    if before != after:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        raise IdentityMismatchError("runtime-home source changed during clone")
    try:
        python_path = _venv_python(destination)
    except Exception:
        if destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    studio_home = destination / ".unsloth" / "studio"
    return CloneHandle(
        source=source,
        run_dir=destination,
        studio_home=studio_home,
        venv_python=python_path,
        source_manifest=before,
    )


def remove_clone_if_proven(clone: CloneHandle, *, cleanup_proven: bool) -> bool:
    """Remove only a clone whose process cleanup and identity checks passed."""

    if not cleanup_proven:
        return False
    if _file_manifest(clone.source) != clone.source_manifest:
        return False
    if clone.run_dir.is_symlink():
        return False
    shutil.rmtree(clone.run_dir)
    return not clone.run_dir.exists()


def allocate_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_health_identity(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise IdentityMismatchError("health response was not an object")
    nested = payload.get("helix_backend_identity")
    identity = nested if isinstance(nested, Mapping) else payload
    contract = identity.get("helix_backend_contract", identity.get("contract"))
    verified = identity.get("helix_backend_verified", identity.get("verified"))
    reason = identity.get("reason")
    tree = identity.get("helix_backend_tree_sha256", identity.get("tree_sha256"))
    # A source run intentionally has no packaged overlay manifest.  Health is
    # therefore a contract check only; accepting a verified packaged digest
    # would falsely bind this run to a different execution artifact.
    if (
        contract != EXPECTED_BACKEND_CONTRACT
        or verified is not False
        or reason not in (None, "runtime_manifest_missing")
        or tree is not None
    ):
        raise IdentityMismatchError("source loopback health identity was not the unverified runtime-manifest-missing mode")
    return {
        "contract": contract,
        "tree_sha256": None,
        "verified": False,
        "reason": "runtime_manifest_missing" if reason == "runtime_manifest_missing" else "not_exposed_source_manifest_absence_preflight",
    }


def _source_identity(
    repo_root: Path,
    *,
    command_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Use the repository's authoritative source identity helper."""

    command = [
        sys.executable,
        str(repo_root / "scripts" / "helix-v3-source-identity.py"),
        "--repo",
        str(repo_root),
    ]
    runner = subprocess.run if command_runner is None else command_runner
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError, TypeError) as exc:
        raise IdentityMismatchError("source identity helper could not run") from exc
    if getattr(completed, "returncode", 0) != 0:
        raise IdentityMismatchError("source identity helper failed")
    try:
        value = json.loads(getattr(completed, "stdout", ""))
    except (TypeError, ValueError) as exc:
        raise IdentityMismatchError("source identity helper returned invalid JSON") from exc
    if not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("tree_sha256", ""))):
        raise IdentityMismatchError("source identity did not contain a valid tree digest")
    return value


def verify_source_identity(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    fields = (
        "repo", "repo_root", "path", "git_head", "branch", "tree_sha256", "dirty",
        "status_count", "file_count", "total_bytes",
    )
    if any(before.get(field) != after.get(field) for field in fields):
        raise IdentityMismatchError("source identity drifted during the benchmark")


def runner_identity(repo_root: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "version": SCRIPT_VERSION,
        "path": str(script),
        "sha256": _sha256_bytes(script.read_bytes()),
        "repo_root": str(repo_root),
    }


def hardware_identity() -> dict[str, Any]:
    memory = None
    try:
        memory = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "system": platform.system() or None,
        "release": platform.release() or None,
        "machine": platform.machine() or None,
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
        "physical_memory_bytes": memory if memory and memory > 0 else None,
    }


def _load_model_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("model config must be a readable JSON object") from exc
    if not isinstance(payload, dict):
        raise ContractError("model config must be a JSON object")
    return payload


def validate_offline_model_spec(model_path: Path, config_path: Path) -> dict[str, Any]:
    """Validate the local MLX model and config used by the load request."""

    if not model_path.is_absolute() or not config_path.is_absolute():
        raise ContractError("offline model and config paths must be absolute")
    if "://" in str(model_path) or "://" in str(config_path):
        raise ContractError("network model/config references are not allowed")
    model = model_path.resolve(strict=False)
    config = config_path.resolve(strict=False)
    if not model.is_dir():
        raise ContractError("offline MLX model path must be an existing directory")
    if not config.is_file() or config.is_symlink():
        raise ContractError("offline MLX model config must be an existing regular file")
    model_config = _load_model_config(config)

    def reject_network_or_secret(value: Any) -> None:
        if isinstance(value, str) and "://" in value:
            raise ContractError("network model/config references are not allowed")
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in {"hf_token", "hub_token", "api_key", "access_token"}:
                    raise ContractError("credential-bearing model config is not allowed")
                reject_network_or_secret(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_network_or_secret(item)

    reject_network_or_secret(model_config)
    backend = str(model_config.get("backend", model_config.get("runtime", "mlx"))).lower()
    if backend not in {"mlx", "mlx-vlm", "mlx_vlm"}:
        raise ContractError("model config must select the MLX backend")
    return {"model_path": model, "config_path": config, "config": model_config}


def build_load_payload(model_spec: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    config = model_spec.get("config")
    model_path = model_spec.get("model_path")
    if not isinstance(config, Mapping) or not isinstance(model_path, Path):
        raise ContractError("model spec did not contain a local model config")
    load_config = config.get("load") if isinstance(config.get("load"), Mapping) else config
    try:
        max_seq_length = int(load_config.get("max_seq_length", DEFAULT_CONTEXT_LENGTH))
    except (TypeError, ValueError) as exc:
        raise ContractError("model config max_seq_length must be an integer") from exc
    if max_seq_length < 0:
        raise ContractError("model config max_seq_length must not be negative")
    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "force_reload": True,
        "load_request_id": f"helix-v3-{seed}",
        "max_seq_length": max_seq_length,
        "load_in_4bit": bool(load_config.get("load_in_4bit", False)),
        "speculative_type": str(load_config.get("speculative_type", "off")),
    }
    for key in (
        "mlx_kv_bits", "cache_type_kv", "spec_draft_n_max", "spec_draft_model_path",
        "n_parallel", "n_batch", "n_ubatch",
    ):
        if key in load_config and load_config[key] is not None:
            payload[key] = load_config[key]
    return payload


def _operation_failed(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return True
    if payload.get("ok") is False:
        return True
    status = payload.get("status")
    return isinstance(status, str) and status.lower() in {"error", "failed", "failure", "unavailable", "timeout"}


def _response_model_path(response: Mapping[str, Any]) -> str | None:
    """Return an optional response path, rejecting malformed path fields."""

    for key in ("model_path", "modelPath", "path"):
        if key not in response:
            continue
        value = response.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"model response {key} was not a non-empty path")
        return str(Path(value).expanduser().resolve(strict=False))
    return None


def _validate_local_mlx_response(response: Mapping[str, Any]) -> None:
    """Require the concrete local-MLX flags returned by ``LoadResponse``."""

    if response.get("is_mlx") is not True or response.get("is_local_model") is not True:
        raise ContractError("model load response did not prove a local MLX runtime")

    for key in ("backend", "runtime", "model_backend"):
        if key not in response:
            continue
        value = response.get(key)
        if not isinstance(value, str) or value.strip().lower() not in {"mlx", "mlx-vlm", "mlx_vlm"}:
            raise ContractError("model response did not identify the local MLX backend")


def _local_model_contract_names(
    model_path: Path,
    *,
    configured_model_name: str | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> set[str]:
    """Build the exact names admitted for one local model path/config."""

    names = {str(Path(model_path).resolve(strict=False))}
    if isinstance(configured_model_name, str) and configured_model_name.strip():
        names.add(configured_model_name.strip())
    if isinstance(model_config, Mapping):
        repository = _nested_model_value(model_config, {"repository", "repo", "model_repository"})
        if repository is not None:
            names.add(repository)
        for block_key in ("model_identity", "model"):
            block = model_config.get(block_key)
            if not isinstance(block, Mapping):
                continue
            for key in ("identifier", "model_id", "model_name", "name"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    names.add(value.strip())
        for key in ("identifier", "model_id", "model_name"):
            value = model_config.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


def _hf_cache_repo_id(path: str) -> str | None:
    """Recover ``org/model`` from one Hugging Face snapshot path."""

    parts = path.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part.startswith("models--") and parts[index + 1 : index + 2] == ["snapshots"]:
            return part[len("models--") :].replace("--", "/")
    return None


def public_model_id(identifier: str) -> str:
    """Mirror the backend's path-free model identity for request/trace binding."""

    raw = identifier.strip()
    if not raw:
        raise ContractError("local model identity was empty")
    repo_id = _hf_cache_repo_id(raw)
    if repo_id:
        return repo_id
    normalized = raw.replace("\\", "/").rstrip("/")
    looks_like_path = (
        normalized.lower().endswith(".gguf")
        or normalized.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~"))
        or (len(normalized) >= 2 and normalized[1] == ":")
        or normalized.count("/") >= 2
        or "\\" in raw
    )
    if not looks_like_path:
        return raw
    name = normalized.rsplit("/", 1)[-1]
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    if not name:
        raise ContractError("local model path did not yield a public identity")
    return name


def derive_public_model_identity(model_path: Path, model_config: Mapping[str, Any]) -> str:
    """Derive the backend-visible identity and bind it to configured provenance."""

    identity = public_model_id(str(Path(model_path).resolve(strict=False)))
    repository = _nested_model_value(
        model_config,
        {"repository", "repo", "model_repository"},
    )
    if repository is not None and repository != identity:
        raise IdentityMismatchError(
            "public model identity did not match the configured model repository"
        )
    return identity


def select_model_request_name(
    load_response: Any,
    model_path: Path,
    *,
    allowed_names: Iterable[str] = (),
) -> str:
    """Admit only a load identity bound to this exact local MLX model."""

    if not isinstance(load_response, Mapping) or not isinstance(load_response.get("model"), str) or not load_response["model"].strip():
        raise ContractError("model load response did not report an authoritative model identity")
    if load_response.get("status") != "loaded":
        raise ContractError("model load response did not report loaded status")
    _validate_local_mlx_response(load_response)
    expected_path = str(Path(model_path).resolve(strict=False))
    response_path = _response_model_path(load_response)
    if response_path is not None and response_path != expected_path:
        raise IdentityMismatchError("model load response path did not match the local MLX model")
    name = load_response["model"].strip()
    # The same frozen package/model pairing already reports the resolved local
    # snapshot path.  A generic alias (for example ``local``) is not an exact
    # model identity and must not be admitted merely because the caller supplied
    # it in configuration.
    if str(Path(name).expanduser().resolve(strict=False)) != expected_path:
        raise IdentityMismatchError("model load response identity did not match the local MLX model contract")
    return expected_path


def validate_unload_response(
    unload_response: Any,
    *,
    admitted_model_name: str,
    model_path: Path,
) -> str:
    """Require unload success and the exact model identity admitted at load."""

    if _operation_failed(unload_response) or not isinstance(unload_response, Mapping):
        raise HarnessError("local MLX model unload failed")
    model = unload_response.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ContractError("model unload response did not report its model identity")
    admitted = str(Path(model.strip()).expanduser().resolve(strict=False))
    expected_path = str(Path(model_path).resolve(strict=False))
    if admitted != admitted_model_name or admitted != expected_path:
        raise IdentityMismatchError("model unload response identity did not match the admitted model")
    response_path = _response_model_path(unload_response)
    if response_path is not None and response_path != expected_path:
        raise IdentityMismatchError("model unload response path did not match the admitted local MLX model")
    return admitted


def _nested_model_value(config: Mapping[str, Any], keys: set[str]) -> str | None:
    """Find a model identity field in nested model/model_identity blocks."""

    def visit(value: Any) -> str | None:
        if isinstance(value, Mapping):
            # Prefer the current object's identity keys before descending.  In
            # particular this handles {"model": {"model_identity": {...}}}
            # without losing the repository/revision pair.
            for key, item in value.items():
                if str(key).lower() in keys and isinstance(item, str) and item.strip():
                    return item.strip()
            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found
        return None

    return visit(config)


def model_identity(config: RuntimeConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": config.model_name,
        "provider": config.provider_identity,
        "backend": "mlx",
        "offline": True,
        "snapshot_path": None,
        "config_sha256": None,
        "repository": None,
        "revision": None,
    }
    if config.model_path is not None:
        result["snapshot_path"] = str(Path(config.model_path).resolve(strict=False))
    if config.model_config_path is not None:
        result["config_sha256"] = _sha256_bytes(Path(config.model_config_path).read_bytes())
        try:
            parsed = json.loads(Path(config.model_config_path).read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                result["repository"] = _nested_model_value(
                    parsed, {"repository", "repo", "model_repository"}
                )
                result["revision"] = _nested_model_value(
                    parsed, {"revision", "hf_revision", "hugging_face_revision", "commit_hash", "snapshot_revision"}
                )
        except (OSError, UnicodeError, ValueError):
            raise ContractError("model config could not be parsed")
    return result


def _unavailable(reason: str = "not_exposed") -> dict[str, Any]:
    return {"value": None, "provenance": "unavailable", "reason": reason}


_METRICS = {
    "quality_and_work": (
        "tool_calls", "deterministic_suboperations", "failed_tool_calls", "prevented_tool_calls",
        "redundant_tool_calls", "retrieval_calls", "redundant_retrieval_calls", "retrieval_recall",
        "irrelevant_hit_rate", "input_tokens", "output_tokens", "cached_tokens", "newly_evaluated_tokens",
        "prefill_tokens", "context_occupancy_high_water", "context_reconstructions", "context_compactions",
    ),
    "latency": (
        "end_to_end_task_wall_ms", "model_wall_ms", "ttft_first_protocol_event_ms",
        "ttft_first_user_visible_token_ms", "prefill_ms", "decode_ms", "decode_tokens_per_second",
        "tool_ms", "retrieval_ms", "context_reconstruction_ms", "controller_ms", "finalization_ms",
    ),
    "memory_and_runtime": (
        "peak_process_or_unified_memory_bytes", "host_available_memory_bytes", "memory_pressure_class",
        "swap_start_bytes", "swap_end_bytes", "swap_delta_bytes", "compression_start_bytes",
        "compression_end_bytes", "compression_delta_bytes", "resident_model_bytes", "kv_bytes",
        "speculative_companion_bytes", "small_model_bytes", "actual_peak_occupied_context_tokens",
    ),
    "recovery": (
        "crash_boundary", "restart_count", "recovery_attempts", "recovery_result", "ambiguous_outcome",
        "duplicate_side_effects", "approval_identity_preserved", "capability_identity_preserved",
        "call_identity_preserved", "checkpoint_identity_preserved", "payload_identity_preserved",
    ),
}


def unavailable_metrics() -> dict[str, dict[str, dict[str, Any]]]:
    return {group: {name: _unavailable() for name in names} for group, names in _METRICS.items()}


def _numeric_fields(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for key, raw in value.items():
        if key not in {"prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_n", "prompt_ms", "predicted_ms", "predicted_per_second"}:
            continue
        number = _safe_float(raw)
        if number is not None and number >= 0:
            result[str(key)] = int(number) if number.is_integer() else number
    return result


def _event_payload(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, str):
        return value if isinstance(value, Mapping) else None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _contains_private_trace(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _PRIVATE_KEYS or str(key).startswith("_helix_optimization_trace"):
                return True
            if _contains_private_trace(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_private_trace(item) for item in value)
    elif isinstance(value, str):
        return TRACE_EVENT_TYPE in value or TRACE_PRIVATE_KEY in value
    return False


class UrllibTransport:
    def __init__(self) -> None:
        # The harness itself must never honor a workstation proxy for its
        # loopback-only control plane.  The child receives NO_PROXY too, but
        # this opener makes the parent-side guarantee independent of ambient
        # HTTP(S)_PROXY variables.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request_json(self, method: str, url: str, *, headers: Mapping[str, str], payload: Any = None, timeout: float) -> HTTPResponse:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers = dict(headers)
        if body is not None:
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read()
                status = int(getattr(response, "status", response.getcode()))
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            raise HarnessError(f"loopback HTTP {method} returned {exc.code}") from None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise HarnessError(f"loopback HTTP {method} failed: {type(exc).__name__}") from None
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeError, ValueError) as exc:
            raise HarnessError(f"loopback HTTP {method} returned non-JSON data") from exc
        return HTTPResponse(status=status, payload=value, headers=response_headers)

    def open_stream(self, method: str, url: str, *, headers: Mapping[str, str], payload: Any, timeout: float) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json", "Accept": "text/event-stream"},
            method=method,
        )
        try:
            return self._opener.open(request, timeout=timeout)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise HarnessError(f"loopback stream failed: {type(exc).__name__}") from None


def _json_request(transport: Any, method: str, base_url: str, path: str, *, headers: Mapping[str, str], payload: Any = None, timeout: float) -> Any:
    response = transport.request_json(method, base_url.rstrip("/") + path, headers=headers, payload=payload, timeout=timeout)
    if isinstance(response, HTTPResponse):
        if response.status >= 400:
            raise HarnessError(f"loopback HTTP {method} returned HTTP {response.status}")
        return response.payload
    if isinstance(response, Mapping) and "status" in response and "payload" in response:
        if int(response["status"]) >= 400:
            raise HarnessError(f"loopback HTTP {method} returned HTTP {response['status']}")
        return response["payload"]
    return response


def _auth_headers(access_token: str) -> dict[str, str]:
    if not isinstance(access_token, str) or not access_token:
        raise AccountBindingError("login did not return an access token")
    return {"Authorization": f"Bearer {access_token}"}


def login_owner(transport: Any, base_url: str, password: str, timeout: float) -> tuple[dict[str, str], dict[str, Any]]:
    payload = _json_request(
        transport,
        "POST",
        base_url,
        "/api/auth/login",
        headers={"Accept": "application/json"},
        payload={"username": OWNER_USERNAME, "password": password},
        timeout=timeout,
    )
    if not isinstance(payload, Mapping) or payload.get("account_id") != OWNER_ACCOUNT_ID:
        raise AccountBindingError("login did not authenticate the installation owner")
    token = payload.get("access_token")
    return _auth_headers(token), {
        "username": OWNER_USERNAME,
        "account_id": OWNER_ACCOUNT_ID,
        "auth_mode": "password",
    }


def build_request_payload(config: RuntimeConfig, run_id: str, thread_id: str) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "messages": [{"role": "user", "content": config.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "tools": None,
        "tool_choice": "none",
        "enable_tools": False,
        "mcp_enabled": False,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "thread_id": thread_id,
        "cancel_id": run_id,
    }


def summarize_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded request summary without retaining prompt text."""

    messages = payload.get("messages")
    prompt_parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                prompt_parts.append(content)
    prompt = "".join(prompt_parts)
    summary: dict[str, Any] = {
        "request_sha256": _sha256_json(payload),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "prompt_length": len(prompt),
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")) if prompt else None,
    }
    for key in ("model", "stream", "max_tokens", "seed", "enable_tools", "mcp_enabled"):
        value = payload.get(key)
        if isinstance(value, (str, int, bool)):
            summary[key] = value
    return summary


def create_objective(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    config: RuntimeConfig,
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    stamp = int(clock() * 1000)
    suffix = f"{os.getpid()}_{stamp}"
    thread_id = f"helix-v3-source-thread-{suffix}"
    user_message_id = f"helix-v3-source-user-{suffix}"
    assistant_message_id = f"helix-v3-source-assistant-{suffix}"
    run_id = f"helix-v3-source-run-{suffix}"
    thread_payload = {
        "id": thread_id,
        "title": "Helix v3 source runtime benchmark",
        "modelType": "base",
        "modelId": config.model_name,
        "createdAt": stamp,
    }
    thread = _json_request(transport, "POST", base_url, "/api/chat/threads", headers=headers, payload=thread_payload, timeout=config.request_timeout_s)
    if not isinstance(thread, Mapping) or thread.get("id") != thread_id:
        raise AccountBindingError("server did not create the requested thread")
    message_payload = {
        "id": user_message_id,
        "threadId": thread_id,
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": config.prompt}],
        "createdAt": stamp,
    }
    message = _json_request(
        transport,
        "PUT",
        base_url,
        f"/api/chat/threads/{urllib.parse.quote(thread_id, safe='')}/messages/{urllib.parse.quote(user_message_id, safe='')}",
        headers=headers,
        payload=message_payload,
        timeout=config.request_timeout_s,
    )
    if not isinstance(message, Mapping) or message.get("id") != user_message_id or message.get("threadId") != thread_id or message.get("role") != "user":
        raise AccountBindingError("server did not persist the requested user message")
    request_payload = build_request_payload(config, run_id, thread_id)
    run_payload = {
        "runId": run_id,
        "threadId": thread_id,
        "userMessageId": user_message_id,
        "assistantMessageId": assistant_message_id,
        "requestPayload": request_payload,
    }
    run = _json_request(transport, "POST", base_url, "/api/inference/chat-runs", headers=headers, payload=run_payload, timeout=config.request_timeout_s)
    if not isinstance(run, Mapping):
        raise AccountBindingError("server did not return a durable run")
    for key, expected in (("id", run_id), ("threadId", thread_id), ("userMessageId", user_message_id), ("assistantMessageId", assistant_message_id)):
        if run.get(key) != expected:
            raise AccountBindingError(f"durable run {key} binding mismatch")
    return {
        "thread_id": thread_id,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "run_id": run_id,
        "request_payload": request_payload,
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


def _json_depth(value: Any, *, limit: int = MAX_SSE_JSON_DEPTH, _level: int = 0) -> int:
    """Return JSON nesting depth, rejecting pathological public frames."""

    if _level > limit:
        raise ProtocolMismatchError("frontend SSE JSON nesting exceeded the bound")
    if isinstance(value, Mapping):
        if not value:
            return 1
        depth = 1 + max(_json_depth(item, limit=limit, _level=_level + 1) for item in value.values())
    elif isinstance(value, (list, tuple)):
        if not value:
            return 1
        depth = 1 + max(_json_depth(item, limit=limit, _level=_level + 1) for item in value)
    else:
        return 0
    if depth > limit:
        raise ProtocolMismatchError("frontend SSE JSON nesting exceeded the bound")
    return depth


def consume_frontend_sse(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    run_id: str,
    *,
    timeout: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Consume only frontend-visible replay and reject any private trace frame."""

    started = clock()
    response = transport.open_stream(
        "POST",
        base_url.rstrip("/") + f"/api/inference/chat-runs/{urllib.parse.quote(run_id, safe='')}/events?after=0",
        headers={**headers, "Accept": "text/event-stream"},
        payload={},
        timeout=timeout,
    )
    events: list[dict[str, Any]] = []
    block: list[str] = []
    raw_bytes = 0
    block_bytes = 0
    last_cursor: int | None = None
    terminal_cursor: int | None = None

    def consume_block(lines: list[str]) -> None:
        nonlocal last_cursor, terminal_cursor
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
        encoded = "\n".join(data_lines)
        try:
            envelope = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ProtocolMismatchError("frontend SSE contained malformed JSON") from exc
        _json_depth(envelope)
        if _contains_private_trace(envelope) or event_type == TRACE_EVENT_TYPE:
            raise ProtocolMismatchError("private optimization trace leaked to frontend SSE")
        if event_type is None:
            raise ProtocolMismatchError("frontend SSE event type was missing")
        if event_id is None or not re.fullmatch(r"[0-9]+", event_id):
            raise ProtocolMismatchError("frontend SSE event cursor was missing or invalid")
        cursor = int(event_id)
        if not isinstance(envelope, Mapping):
            raise ProtocolMismatchError("frontend SSE envelope was not an object")
        expected_keys = {"seq", "type", "payload", "createdAt"}
        if event_type != "chunk":
            expected_keys.add("run")
        if set(envelope) != expected_keys:
            raise ProtocolMismatchError("frontend SSE envelope fields were not canonical")
        envelope_seq = envelope.get("seq")
        if isinstance(envelope_seq, bool) or not isinstance(envelope_seq, int) or envelope_seq < 0:
            raise ProtocolMismatchError("frontend SSE envelope sequence was invalid")
        if envelope_seq != cursor:
            raise ProtocolMismatchError("frontend SSE wire cursor did not match envelope seq")
        envelope_type = envelope.get("type")
        if not isinstance(envelope_type, str) or not envelope_type or envelope_type != event_type:
            raise ProtocolMismatchError("frontend SSE wire event did not match envelope type")
        created_at = envelope.get("createdAt")
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise ProtocolMismatchError("frontend SSE envelope createdAt was invalid")
        payload = envelope.get("payload")
        if not isinstance(payload, Mapping):
            raise ProtocolMismatchError("frontend SSE inner payload was not an object")
        run = envelope.get("run")
        if event_type != "chunk" and not isinstance(run, Mapping):
            raise ProtocolMismatchError("frontend SSE non-chunk envelope run was not an object")
        if last_cursor is not None and cursor <= last_cursor:
            raise ProtocolMismatchError("frontend SSE event cursor was not strictly monotonic")
        last_cursor = cursor
        if len(events) >= MAX_SSE_EVENTS:
            raise ProtocolMismatchError("frontend SSE exceeded the event-count bound")
        item = {
            "event": event_type,
            "id": event_id,
            "cursor": cursor,
            "payload": dict(payload),
        }
        if isinstance(run, Mapping):
            item["run"] = dict(run)
        events.append(item)

    try:
        for raw in _stream_lines(response):
            if isinstance(raw, bytes):
                wire_length = len(raw)
                text = raw.decode("utf-8", errors="strict")
            else:
                text = str(raw)
                wire_length = len(text.encode("utf-8"))
            if wire_length > MAX_SSE_LINE_BYTES:
                raise ProtocolMismatchError("frontend SSE line exceeded the bound")
            if raw_bytes + wire_length > MAX_SSE_WIRE_BYTES:
                raise ProtocolMismatchError("frontend SSE exceeded the wire-size bound")
            raw_bytes += wire_length
            text = text.rstrip("\r\n")
            if not text:
                consume_block(block)
                block = []
                block_bytes = 0
            else:
                block_bytes += wire_length
                if block_bytes > MAX_SSE_BLOCK_BYTES:
                    raise ProtocolMismatchError("frontend SSE block exceeded the bound")
                block.append(text)
        consume_block(block)
    except UnicodeError as exc:
        raise ProtocolMismatchError("frontend SSE was not valid UTF-8") from exc
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    terminal: Mapping[str, Any] | None = None
    for item in events:
        payload = item["payload"]
        candidate = item.get("run")
        if isinstance(candidate, Mapping) and candidate.get("id") == run_id:
            if candidate.get("status") in _TERMINAL_STATUSES:
                terminal = candidate
                terminal_cursor = int(item["cursor"])
        # Keep compatibility with older replay fixtures while requiring the
        # canonical outer envelope for every newly parsed event.
        nested_candidate = payload.get("run")
        if isinstance(nested_candidate, Mapping) and nested_candidate.get("id") == run_id:
            if nested_candidate.get("status") in _TERMINAL_STATUSES:
                terminal = nested_candidate
                terminal_cursor = int(item["cursor"])
        if payload.get("status") in _TERMINAL_STATUSES and payload.get("id") == run_id:
            terminal = payload
            terminal_cursor = int(item["cursor"])
    if terminal is None:
        raise ProtocolMismatchError("frontend SSE did not expose a terminal durable run")
    if terminal.get("status") != "completed":
        raise ProtocolMismatchError("durable source run did not complete")
    if terminal_cursor is None or last_cursor is None or terminal_cursor != last_cursor:
        raise ProtocolMismatchError("frontend SSE terminal cursor was not the last visible event")
    last_event_seq = terminal.get("lastEventSeq", terminal.get("last_event_seq"))
    if isinstance(last_event_seq, bool) or not isinstance(last_event_seq, int) or terminal_cursor != last_event_seq:
        raise ProtocolMismatchError("frontend SSE terminal cursor did not prove lastEventSeq")
    return {
        "status": "pass",
        "event_count": len(events),
        "events": events,
        "terminal_run": dict(terminal),
        "wire_bytes": raw_bytes,
        "elapsed_ms": max(0.0, (clock() - started) * 1000.0),
        "private_trace_suppressed": True,
        "last_event_seq": last_event_seq,
    }


def reconstruct_visible_answer(sse: Mapping[str, Any]) -> str:
    """Reconstruct the bounded public answer and require the exact control text."""

    events = sse.get("events") if isinstance(sse, Mapping) else None
    if not isinstance(events, list):
        raise ProtocolMismatchError("frontend SSE did not contain a bounded event list")
    pieces: list[str] = []
    encoded_bytes = 0
    for item in events:
        payload = item.get("payload") if isinstance(item, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        for choice in payload.get("choices") or ():
            if not isinstance(choice, Mapping):
                raise ProtocolMismatchError("frontend SSE choice was not an object")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ProtocolMismatchError("no-tool control emitted a tool proposal")
            content = delta.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                raise ProtocolMismatchError("frontend SSE content delta was not text")
            encoded_bytes += len(content.encode("utf-8"))
            if encoded_bytes > MAX_SSE_BLOCK_BYTES:
                raise ProtocolMismatchError("frontend visible answer exceeded the bound")
            pieces.append(content)
    answer = "".join(pieces)
    terminal = sse.get("terminal_run") if isinstance(sse, Mapping) else None
    if not isinstance(terminal, Mapping):
        raise ProtocolMismatchError("frontend SSE terminal run was missing")
    finish_reason = terminal.get("finishReason", terminal.get("finish_reason"))
    if finish_reason != "stop":
        raise ProtocolMismatchError("source control did not finish with stop")
    if answer != EXPECTED_ANSWER:
        raise ProtocolMismatchError("frontend SSE did not prove the exact expected answer")
    return answer


def persist_assistant_answer(
    transport: Any,
    base_url: str,
    headers: Mapping[str, str],
    *,
    objective: Mapping[str, str],
    sse: Mapping[str, Any],
    answer: str,
    timeout: float,
) -> dict[str, Any]:
    """Persist the verified public answer through the normal generation-message API."""

    if answer != EXPECTED_ANSWER:
        raise ProtocolMismatchError("refusing to persist an unverified assistant answer")
    thread_id = objective["thread_id"]
    user_message_id = objective["user_message_id"]
    assistant_message_id = objective["assistant_message_id"]
    run_id = objective["run_id"]
    quoted_thread = urllib.parse.quote(thread_id, safe="")
    quoted_message = urllib.parse.quote(assistant_message_id, safe="")
    path = f"/api/chat/threads/{quoted_thread}/messages/{quoted_message}"
    current = _json_request(
        transport,
        "GET",
        base_url,
        path,
        headers=headers,
        timeout=timeout,
    )
    if not isinstance(current, Mapping):
        raise AccountBindingError("assistant placeholder was not returned")
    if (
        current.get("id") != assistant_message_id
        or current.get("threadId") != thread_id
        or current.get("parentId") != user_message_id
        or current.get("role") != "assistant"
    ):
        raise AccountBindingError("assistant placeholder binding mismatch")
    if current.get("content") not in ([], None):
        raise ProtocolMismatchError("assistant placeholder already contained content")
    created_at = current.get("createdAt")
    metadata = current.get("metadata")
    last_event_seq = sse.get("last_event_seq")
    if isinstance(created_at, bool) or not isinstance(created_at, int):
        raise ProtocolMismatchError("assistant placeholder createdAt was invalid")
    if not isinstance(metadata, Mapping):
        raise AccountBindingError("assistant placeholder metadata was missing")
    if (
        metadata.get("generationRunId") != run_id
        or metadata.get("serverManaged") is not True
        or metadata.get("generationStatus") != "completed"
        or isinstance(last_event_seq, bool)
        or not isinstance(last_event_seq, int)
    ):
        raise AccountBindingError("assistant placeholder generation binding mismatch")
    next_metadata = dict(metadata)
    next_metadata.update(
        {
            "generationRunId": run_id,
            "generationSeq": last_event_seq,
            "generationStatus": "completed",
            "generationSettled": True,
            "serverManaged": True,
        }
    )
    payload = {
        "id": assistant_message_id,
        "threadId": thread_id,
        "parentId": user_message_id,
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
        "attachments": current.get("attachments"),
        "metadata": next_metadata,
        "createdAt": created_at,
    }
    saved = _json_request(
        transport,
        "PUT",
        base_url,
        path,
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    if not isinstance(saved, Mapping):
        raise AccountBindingError("assistant answer persistence returned no message")
    if (
        saved.get("id") != assistant_message_id
        or saved.get("threadId") != thread_id
        or saved.get("parentId") != user_message_id
        or saved.get("role") != "assistant"
        or saved.get("createdAt") != created_at
        or saved.get("metadata") != next_metadata
    ):
        raise AccountBindingError("persisted assistant message binding mismatch")
    saved_answer = _assistant_answer_from_parts(
        saved.get("content"),
        label="persisted assistant content",
    )
    if saved_answer != answer:
        raise ProtocolMismatchError("persisted assistant answer did not match public SSE")
    return {
        "status": "pass",
        "provenance": "authenticated_chat_message_put",
        "answer_length": len(answer),
        "answer_sha256": _sha256_bytes(answer.encode("utf-8")),
        "generation_seq": last_event_seq,
    }


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {scope: {tier: 0 for tier in _TIERS} for scope in _SCOPES}


def aggregate_trace_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive counts from strict event identities, never trust reported totals."""

    if len(events) > MAX_TRACE_EVENTS:
        raise ProtocolMismatchError("optimization trace exceeded the invocation limit")
    matrices = [_empty_matrix() for _ in range(4)]
    invocations, semantic, failed, wasted = matrices
    seen: set[str] = set()
    prior: set[str] = set()
    escalations = 0
    for event in events:
        if not isinstance(event, Mapping):
            raise ProtocolMismatchError("trace invocation event was not an object")
        required = {
            "invocation_id", "scope", "provider_tier", "model_identity", "provider_identity",
            "outcome", "semantic_contribution", "wasted", "wasted_reason", "retry_of", "escalation_from_tier",
        }
        if set(event) != required:
            raise ProtocolMismatchError("trace invocation event fields were not exact")
        invocation_id = event.get("invocation_id")
        scope = event.get("scope")
        tier = event.get("provider_tier")
        if not isinstance(invocation_id, str) or not invocation_id or invocation_id in seen:
            raise ProtocolMismatchError("trace invocation id was missing or duplicated")
        if (
            not isinstance(scope, str)
            or not isinstance(tier, str)
            or not isinstance(event.get("model_identity"), str)
            or not event.get("model_identity")
            or not isinstance(event.get("provider_identity"), str)
            or not event.get("provider_identity")
            or scope not in _SCOPES
            or tier not in _TIERS
        ):
            raise ProtocolMismatchError("trace invocation scope or tier was invalid")
        retry_of = event.get("retry_of")
        if retry_of is not None and (not isinstance(retry_of, str) or retry_of not in prior):
            raise ProtocolMismatchError("trace retry link did not identify an earlier event")
        outcome = event.get("outcome")
        contribution = event.get("semantic_contribution")
        wasted_flag = event.get("wasted")
        escalation = event.get("escalation_from_tier")
        if not isinstance(outcome, str) or outcome not in {"completed", "failed", "cancelled", "discarded"}:
            raise ProtocolMismatchError("trace invocation outcome was invalid")
        if not isinstance(contribution, str) or contribution not in {"none", "intent", "action_plan", "decision", "evidence_interpretation", "final_answer"}:
            raise ProtocolMismatchError("trace semantic contribution was invalid")
        if not isinstance(wasted_flag, bool):
            raise ProtocolMismatchError("trace wasted flag was invalid")
        if escalation is not None and not isinstance(escalation, str):
            raise ProtocolMismatchError("trace escalation tier was invalid")
        if contribution != "none" and (outcome != "completed" or wasted_flag):
            raise ProtocolMismatchError("trace semantic contribution was inadmissible")
        if wasted_flag and (
            not isinstance(event.get("wasted_reason"), str)
            or not event.get("wasted_reason").strip()
            or len(event.get("wasted_reason")) > 500
        ):
            raise ProtocolMismatchError("wasted trace invocation had no reason")
        if not wasted_flag and event.get("wasted_reason") is not None:
            raise ProtocolMismatchError("non-wasted trace invocation had a reason")
        if escalation is not None:
            if escalation not in _TIERS or _TIERS.index(escalation) >= _TIERS.index(tier):
                raise ProtocolMismatchError("trace escalation was not to a higher tier")
            escalations += 1
        invocations[scope][tier] += 1
        if contribution != "none":
            semantic[scope][tier] += 1
        if outcome == "failed":
            failed[scope][tier] += 1
        if wasted_flag:
            wasted[scope][tier] += 1
        seen.add(invocation_id)
        prior.add(invocation_id)
    expensive = sum(semantic[scope][tier] for scope in _SCOPES for tier in ("senior", "frontier"))
    return {
        "model_invocations": {"counts": invocations},
        "semantic_turns": {"counts": semantic},
        "failed_invocations": {"counts": failed},
        "wasted_invocations": {"counts": wasted},
        "escalations": escalations,
        "expensive_interventions": expensive,
    }


def _strict_binding(binding: Any, objective: Mapping[str, str], *, account_id: str) -> None:
    if not isinstance(binding, Mapping):
        raise AccountBindingError("trace binding was not an object")
    expected = {
        "run_id": objective["run_id"],
        "thread_id": objective["thread_id"],
        "user_message_id": objective["user_message_id"],
        "owner_subject": OWNER_USERNAME,
        "account_id": account_id,
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise AccountBindingError("trace account/run/thread/message binding mismatch")


def validate_trace_payload(
    payload: Mapping[str, Any],
    *,
    objective: Mapping[str, str],
    provider_tier: str,
    model_name: str,
    provider_identity: str,
    account_id: str = OWNER_ACCOUNT_ID,
) -> dict[str, Any]:
    expected_keys = {"schema_version", "status", "aggregation", "binding", "trace", "error"}
    if set(payload) != expected_keys or payload.get("schema_version") != TRACE_EVENT_SCHEMA:
        raise ProtocolMismatchError("optimization trace event envelope was invalid")
    if payload.get("status") != "available" or payload.get("error") is not None:
        raise ProtocolMismatchError("optimization trace was unavailable or inadmissible")
    aggregation = payload.get("aggregation")
    if not isinstance(aggregation, Mapping) or aggregation.get("admissible") is not True or aggregation.get("reason") is not None:
        raise ProtocolMismatchError("optimization trace aggregation was not admissible")
    _strict_binding(payload.get("binding"), objective, account_id=account_id)
    envelope = payload.get("trace")
    if not isinstance(envelope, Mapping):
        raise ProtocolMismatchError("optimization trace envelope was missing")
    if set(envelope) != {"schema_version", "status", "invocation_events", "invocation_counts", "all_handles_settled", "error"}:
        raise ProtocolMismatchError("optimization trace envelope fields were not exact")
    if envelope.get("schema_version") != TRACE_SCHEMA or envelope.get("status") != "available" or envelope.get("all_handles_settled") is not True or envelope.get("error") is not None:
        raise ProtocolMismatchError("optimization trace envelope was not available")
    raw_events = envelope.get("invocation_events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ProtocolMismatchError("optimization trace had no invocation events")
    if len(raw_events) > MAX_TRACE_EVENTS:
        raise ProtocolMismatchError("optimization trace exceeded the invocation limit")
    try:
        if len(_canonical_json(envelope).encode("utf-8")) > MAX_TRACE_WIRE_BYTES:
            raise ProtocolMismatchError("optimization trace exceeded the wire-size limit")
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatchError("optimization trace was not JSON-safe") from exc
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise ProtocolMismatchError("optimization trace event was malformed")
        if event.get("scope") != "foreground" or event.get("provider_tier") != provider_tier:
            raise ProtocolMismatchError("optimization trace scope or tier did not match the runner")
        if event.get("model_identity") != model_name or event.get("provider_identity") != provider_identity:
            raise ProtocolMismatchError("optimization trace model/provider identity mismatch")
    derived = aggregate_trace_events(raw_events)
    if envelope.get("invocation_counts") != derived:
        raise ProtocolMismatchError("optimization trace aggregation did not re-derive")
    return {"envelope": dict(envelope), "aggregation": derived}


def _sqlite_uri(path: Path) -> str:
    return "file:" + urllib.parse.quote(str(path.resolve(strict=True)), safe="/") + "?mode=ro"


def _load_json_text(value: Any, label: str) -> Any:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatchError(f"{label} was not valid JSON") from exc
    return parsed


def _assistant_answer_from_parts(content: Any, *, label: str) -> str:
    """Validate decoded assistant content parts and prove the exact task answer."""

    if not isinstance(content, list) or not content:
        raise ProtocolMismatchError(f"{label} was empty or not a list")
    pieces: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") not in {"text", "output_text"}:
            raise ProtocolMismatchError(f"{label} contained a non-text part")
        text = item.get("text")
        if not isinstance(text, str):
            raise ProtocolMismatchError(f"{label} text was not a string")
        pieces.append(text)
    answer = "".join(pieces)
    if answer != EXPECTED_ANSWER:
        raise ProtocolMismatchError(f"{label} did not prove the exact expected answer")
    return answer


def _reconstruct_assistant_answer(content_json: Any) -> str:
    """Decode SQLite JSON, then prove the exact durable assistant answer."""

    return _assistant_answer_from_parts(
        _load_json_text(content_json, "assistant content"),
        label="assistant content_json",
    )


def read_durable_evidence(
    studio_db_path: Path,
    *,
    objective: Mapping[str, str],
    model_name: str,
    provider_tier: str,
    provider_identity: str,
    account_id: str = OWNER_ACCOUNT_ID,
) -> dict[str, Any]:
    """Read only the exact run/threads/messages/trace rows in SQLite URI RO mode."""

    database = _require_regular_file(studio_db_path, "clone studio.db")
    try:
        connection = sqlite3.connect(_sqlite_uri(database), uri=True)
    except sqlite3.Error as exc:
        raise ProtocolMismatchError("clone studio.db could not be opened read-only") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        query_only = connection.execute("PRAGMA query_only").fetchone()[0]
        if int(query_only) != 1:
            raise ProtocolMismatchError("SQLite query_only was not enabled")
        run = connection.execute(
            """SELECT id, owner_subject, thread_id, user_message_id, assistant_message_id,
                      status, finish_reason, request_json, request_hash, last_event_seq,
                      finalization_status
                 FROM chat_generation_runs WHERE id=?""",
            (objective["run_id"],),
        ).fetchall()
        if len(run) != 1:
            raise AccountBindingError("exact durable run row was missing or duplicated")
        run_row = dict(run[0])
        expected_run = {
            "id": objective["run_id"],
            "owner_subject": OWNER_USERNAME,
            "thread_id": objective["thread_id"],
            "user_message_id": objective["user_message_id"],
            "assistant_message_id": objective["assistant_message_id"],
            "status": "completed",
        }
        if any(run_row.get(key) != value for key, value in expected_run.items()):
            raise AccountBindingError("durable run binding or status mismatch")
        if run_row.get("finish_reason") != "stop":
            raise ProtocolMismatchError("durable source run did not finish with stop")
        if run_row.get("finalization_status") != "none":
            raise ProtocolMismatchError("durable run finalization was not exactly none")
        request = _load_json_text(run_row.get("request_json"), "durable request")
        if not isinstance(request, Mapping) or request.get("thread_id") != objective["thread_id"] or request.get("generation_run_id") != objective["run_id"] or request.get("model") != model_name:
            raise AccountBindingError("durable request identity mismatch")
        thread_rows = connection.execute(
            "SELECT id FROM chat_threads WHERE id=?", (objective["thread_id"],)
        ).fetchall()
        if len(thread_rows) != 1:
            raise AccountBindingError("exact thread row was missing or duplicated")
        message_rows = connection.execute(
            """SELECT id, thread_id, parent_id, role, content_json, metadata_json
                 FROM chat_messages WHERE id IN (?, ?) ORDER BY id""",
            (objective["user_message_id"], objective["assistant_message_id"]),
        ).fetchall()
        by_id = {str(row["id"]): dict(row) for row in message_rows}
        if set(by_id) != {objective["user_message_id"], objective["assistant_message_id"]}:
            raise AccountBindingError("run messages were missing or crossed threads")
        user = by_id[objective["user_message_id"]]
        assistant = by_id[objective["assistant_message_id"]]
        if user["thread_id"] != objective["thread_id"] or user["role"] != "user":
            raise AccountBindingError("user message binding mismatch")
        if assistant["thread_id"] != objective["thread_id"] or assistant["role"] != "assistant" or assistant["parent_id"] != objective["user_message_id"]:
            raise AccountBindingError("assistant message binding mismatch")
        assistant_metadata = _load_json_text(assistant.get("metadata_json") or "{}", "assistant metadata")
        if not isinstance(assistant_metadata, Mapping) or assistant_metadata.get("generationRunId") != objective["run_id"]:
            raise AccountBindingError("assistant message was not bound to the durable run")
        assistant_answer = _reconstruct_assistant_answer(assistant.get("content_json"))
        task_success = {
            "status": "pass",
            "expected_answer_length": len(EXPECTED_ANSWER),
            "expected_answer_sha256": _sha256_bytes(EXPECTED_ANSWER.encode("utf-8")),
            "observed_answer_length": len(assistant_answer),
            "observed_answer_sha256": _sha256_bytes(assistant_answer.encode("utf-8")),
            "provenance": "client_observed_sse+authenticated_message_put+sqlite_ro",
            "objective_correctness": "unverified",
        }
        trace_rows = connection.execute(
            """SELECT seq, event_type, payload_json, created_at
                 FROM chat_generation_events WHERE run_id=? AND event_type=? ORDER BY seq""",
            (objective["run_id"], TRACE_EVENT_TYPE),
        ).fetchall()
        if len(trace_rows) != 1:
            raise ProtocolMismatchError("durable optimization trace was missing or duplicated")
        all_event_rows = connection.execute(
            """SELECT seq, event_type, payload_json, created_at
                 FROM chat_generation_events WHERE run_id=? ORDER BY seq""",
            (objective["run_id"],),
        ).fetchall()
        terminal_rows = [row for row in all_event_rows if row["event_type"] == "run.completed"]
        if len(terminal_rows) != 1:
            raise ProtocolMismatchError("durable completed terminal event was missing or duplicated")
        terminal_row = terminal_rows[0]
        trace_seq = trace_rows[0]["seq"]
        terminal_seq = terminal_row["seq"]
        last_event_seq = run_row.get("last_event_seq")
        if (
            isinstance(trace_seq, bool)
            or not isinstance(trace_seq, int)
            or isinstance(terminal_seq, bool)
            or not isinstance(terminal_seq, int)
            or isinstance(last_event_seq, bool)
            or not isinstance(last_event_seq, int)
            or terminal_seq != last_event_seq
            or trace_seq != terminal_seq - 1
        ):
            raise ProtocolMismatchError("optimization trace was not immediately before the terminal durable event")
        expected_metadata = {
            "generationRunId": objective["run_id"],
            "generationSeq": terminal_seq,
            "generationStatus": "completed",
            "generationSettled": True,
            "serverManaged": True,
        }
        if any(assistant_metadata.get(key) != value for key, value in expected_metadata.items()):
            raise AccountBindingError("assistant message settlement metadata was not exact")
        if assistant_metadata.get("incomplete") is not None:
            raise ProtocolMismatchError("completed assistant message was marked incomplete")
        optimization_rows = [row for row in all_event_rows if str(row["event_type"]).startswith("optimization.")]
        if not optimization_rows or optimization_rows[-1]["seq"] != trace_seq:
            raise ProtocolMismatchError("optimization trace was not the last optimization event")
        created_at = trace_rows[0]["created_at"]
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise ProtocolMismatchError("optimization trace timestamp was invalid")
        terminal_created_at = terminal_row["created_at"]
        if isinstance(terminal_created_at, bool) or not isinstance(terminal_created_at, int):
            raise ProtocolMismatchError("terminal run event timestamp was invalid")
        terminal_payload = _load_json_text(terminal_row["payload_json"], "terminal run event")
        if not isinstance(terminal_payload, Mapping) or terminal_payload.get("status") != "completed":
            raise ProtocolMismatchError("terminal run event did not prove completed status")
        trace_payload = _load_json_text(trace_rows[0]["payload_json"], "optimization trace row")
        if not isinstance(trace_payload, Mapping):
            raise ProtocolMismatchError("optimization trace row was not an object")
        validated = validate_trace_payload(
            trace_payload,
            objective=objective,
            provider_tier=provider_tier,
            model_name=model_name,
            provider_identity=provider_identity,
            account_id=account_id,
        )
        run_summary = {
            key: value for key, value in run_row.items() if key != "request_json"
        }
        run_summary["request_summary"] = summarize_request_payload(request)
        trace_envelope = validated["envelope"]
        trace_summary = {
            "status": trace_envelope.get("status"),
            "event_count": len(trace_envelope.get("invocation_events") or ()),
            "aggregation": validated["aggregation"],
            "invocation_counts": validated["aggregation"],
            "all_handles_settled": trace_envelope.get("all_handles_settled") is True,
        }
        return {
            "run": run_summary,
            "thread": {"id": objective["thread_id"]},
            "messages": {"user": {"id": user["id"], "thread_id": user["thread_id"], "role": user["role"]}, "assistant": {"id": assistant["id"], "thread_id": assistant["thread_id"], "role": assistant["role"], "content_length": len(assistant_answer), "content_sha256": _sha256_bytes(assistant_answer.encode("utf-8"))}},
            "task_success": task_success,
            "trace_availability": {"status": "available", "provenance": "sqlite:chat_generation_events.optimization.trace"},
            "trace_event": {"seq": trace_seq, "created_at": created_at, "event_type": TRACE_EVENT_TYPE, "payload_sha256": _sha256_json(trace_payload)},
            "terminal_event": {"seq": terminal_seq, "created_at": terminal_created_at, "event_type": "run.completed", "status": terminal_payload.get("status"), "finish_reason": terminal_payload.get("finishReason", terminal_payload.get("finish_reason")), "payload_sha256": _sha256_json(terminal_payload)},
            "trace": trace_summary,
            "sqlite": {"mode": "ro", "query_only": True},
        }
    finally:
        connection.close()


def _parse_sse_chunk_events(sse: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int | float]]:
    events = sse.get("events") if isinstance(sse, Mapping) else None
    if not isinstance(events, list):
        return {}, {}
    usage: dict[str, int | float] = {}
    timings: dict[str, int | float] = {}
    visible_chars = 0
    output_hash = hashlib.sha256()
    for item in events:
        payload = item.get("payload") if isinstance(item, Mapping) else None
        if not isinstance(payload, Mapping):
            continue
        usage.update(_numeric_fields(payload.get("usage")))
        timings.update(_numeric_fields(payload.get("timings")))
        for choice in payload.get("choices") or ():
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, Mapping) else None
            if isinstance(content, str) and content:
                visible_chars += len(content)
                output_hash.update(content.encode("utf-8"))
    return {"visible_output_chars": visible_chars, "visible_output_sha256": output_hash.hexdigest() if visible_chars else None}, {**usage, **{f"timing_{key}": value for key, value in timings.items()}}


def collect_metrics(sse: Mapping[str, Any], *, elapsed_ms: float) -> dict[str, Any]:
    metrics = unavailable_metrics()
    details, numeric = _parse_sse_chunk_events(sse)
    metrics["latency"]["end_to_end_task_wall_ms"] = {"value": max(0.0, elapsed_ms), "provenance": "measured", "exposure": ["durable-sse"]}
    if details["visible_output_chars"]:
        metrics["latency"]["ttft_first_user_visible_token_ms"] = _unavailable("frontend_replay_does_not_expose_first_token_timestamp")
    if "prompt_tokens" in numeric:
        metrics["quality_and_work"]["input_tokens"] = {"value": numeric["prompt_tokens"], "provenance": "measured", "exposure": ["sse:usage"]}
    if "completion_tokens" in numeric:
        metrics["quality_and_work"]["output_tokens"] = {"value": numeric["completion_tokens"], "provenance": "measured", "exposure": ["sse:usage"]}
    for source, target in (("timing_prompt_ms", "prefill_ms"), ("timing_predicted_ms", "decode_ms"), ("timing_predicted_per_second", "decode_tokens_per_second")):
        if source in numeric:
            metrics["latency"][target] = {"value": numeric[source], "provenance": "measured", "exposure": ["sse:timings"]}
    return {"metrics": metrics, **details, "usage": {key: value for key, value in numeric.items() if not key.startswith("timing_")} or None}


def summarize_frontend_sse(
    sse: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep only bounded SSE transport/content summaries for the artifact."""

    events = sse.get("events") if isinstance(sse, Mapping) else None
    if not isinstance(events, list):
        raise ProtocolMismatchError("frontend SSE did not contain a bounded event list")
    event_types: dict[str, int] = {}
    cursors: list[int] = []
    for item in events:
        if not isinstance(item, Mapping):
            raise ProtocolMismatchError("frontend SSE event summary encountered malformed event")
        event_type = item.get("event")
        cursor = item.get("cursor")
        if not isinstance(event_type, str) or not event_type:
            raise ProtocolMismatchError("frontend SSE event summary had no type")
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise ProtocolMismatchError("frontend SSE event summary had no cursor")
        event_types[event_type] = event_types.get(event_type, 0) + 1
        cursors.append(cursor)
    details, numeric = _parse_sse_chunk_events(sse)
    terminal = sse.get("terminal_run") if isinstance(sse, Mapping) else None
    if not isinstance(terminal, Mapping):
        raise ProtocolMismatchError("frontend SSE terminal run was missing")
    summary: dict[str, Any] = {
        "status": sse.get("status"),
        "event_count": len(events),
        "event_types": dict(sorted(event_types.items())),
        "first_event_seq": cursors[0] if cursors else None,
        "last_event_seq": sse.get("last_event_seq"),
        "wire_bytes": sse.get("wire_bytes"),
        "elapsed_ms": sse.get("elapsed_ms"),
        "private_trace_suppressed": sse.get("private_trace_suppressed") is True,
        "terminal": {
            "status": terminal.get("status"),
            "finish_reason": terminal.get("finishReason", terminal.get("finish_reason")),
            "last_event_seq": terminal.get("lastEventSeq", terminal.get("last_event_seq")),
        },
        "content": details,
        "usage": {key: value for key, value in numeric.items() if not key.startswith("timing_")} or None,
        "timings": {key.removeprefix("timing_"): value for key, value in numeric.items() if key.startswith("timing_")} or None,
    }
    if metrics is not None:
        summary["metrics_sha256"] = _sha256_json(metrics)
    return summary


def _ps_command(pid: int | None = None) -> list[str]:
    if pid is None:
        return ["/bin/ps", "-ww", "-axo", "pid=,ppid=,pgid=,rss=,lstart=,command="]
    return ["/bin/ps", "-ww", "-p", str(int(pid)), "-o", "pid=,ppid=,pgid=,rss=,lstart=,command="]


def _parse_ps_wire_snapshot(output: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(output, str):
        return result
    for line in output.splitlines():
        parts = line.strip().split(None, 9)
        if len(parts) < 10:
            continue
        try:
            pid, ppid, pgid, rss = (int(parts[index]) for index in range(4))
        except (IndexError, TypeError, ValueError):
            continue
        if pid <= 0 or ppid < 0 or pgid <= 0 or rss < 0:
            continue
        start_marker = " ".join(parts[4:9]).strip()
        command = parts[9].strip()
        if not start_marker or not _PS_LSTART_RE.fullmatch(start_marker) or not command:
            continue
        result[pid] = {
            "pid": pid, "ppid": ppid, "pgid": pgid, "sid": None,
            "session_id": None, "rss_kib": rss, "start_marker": start_marker,
            "command": command,
            "identity_token": {"start_marker": start_marker, "command": command, "pgid": pgid, "sid": None, "session_id": None},
        }
    return result


def _descendant_closure(snapshot: Mapping[int, Mapping[str, Any]], root_pid: int) -> set[int]:
    """Derive descendants from complete ps wire rows before getsid filtering."""

    root = int(root_pid)
    children: dict[int, set[int]] = {}
    for raw_pid, identity in snapshot.items():
        pid = int(raw_pid)
        parent = identity.get("ppid")
        if isinstance(parent, int) and parent >= 0:
            children.setdefault(parent, set()).add(pid)
    owned = {root}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in owned:
                owned.add(child)
                frontier.append(child)
    return owned


def _enrich_ps_sessions(
    snapshot: Mapping[int, Mapping[str, Any]],
    *,
    session_id_getter: Callable[[int], int],
    required_pids: Iterable[int] = (),
    required_pgid: int | None = None,
) -> dict[int, dict[str, Any]] | None:
    required = {int(pid) for pid in required_pids}
    enriched: dict[int, dict[str, Any]] = {}
    for raw_pid, raw_identity in snapshot.items():
        pid = int(raw_pid)
        try:
            sid = session_id_getter(pid)
        except (OSError, ProcessLookupError, TypeError, ValueError):
            if pid in required or (required_pgid is not None and raw_identity.get("pgid") == required_pgid):
                return None
            continue
        if isinstance(sid, bool) or not isinstance(sid, int) or sid <= 0:
            if pid in required or (required_pgid is not None and raw_identity.get("pgid") == required_pgid):
                return None
            continue
        identity = dict(raw_identity)
        identity["sid"] = sid
        identity["session_id"] = sid
        token = dict(identity.get("identity_token") or {})
        token["sid"] = sid
        token["session_id"] = sid
        identity["identity_token"] = token
        enriched[pid] = identity
    if required_pgid is not None and any(
        identity.get("pgid") == required_pgid and pid not in enriched
        for pid, identity in snapshot.items()
    ):
        return None
    return enriched


def _parse_ps_snapshot(output: str, *, session_id_getter: Callable[[int], int] | None = None) -> dict[int, dict[str, Any]]:
    getter = os.getsid if session_id_getter is None else session_id_getter
    try:
        enriched = _enrich_ps_sessions(_parse_ps_wire_snapshot(output), session_id_getter=getter)
    except (OSError, ProcessLookupError, TypeError, ValueError):
        return {}
    return enriched or {}


def _has_identity_token(identity: Mapping[str, Any] | None) -> bool:
    if not isinstance(identity, Mapping):
        return False
    token = identity.get("identity_token")
    return (
        isinstance(token, Mapping)
        and isinstance(token.get("start_marker"), str) and bool(token.get("start_marker"))
        and isinstance(token.get("command"), str) and bool(token.get("command"))
        and isinstance(token.get("pgid"), int) and token.get("pgid", 0) > 0
        and isinstance(token.get("sid"), int) and token.get("sid", 0) > 0
    )


def _reaper_boundary_status(*, signal_getter: Callable[[int], Any] | None = None) -> tuple[bool, str]:
    required = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if any(not hasattr(os, name) for name in required) or not callable(os.waitid) or any(
        isinstance(getattr(os, name), bool) or not isinstance(getattr(os, name), int)
        for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    ):
        return False, "waitid_unsupported"
    sigchld = getattr(signal, "SIGCHLD", None)
    if sigchld is None:
        return False, "sigchld_unsupported"
    getter = signal.getsignal if signal_getter is None else signal_getter
    try:
        disposition = getter(sigchld)
    except (OSError, TypeError, ValueError):
        return False, "sigchld_status_unavailable"
    return (True, "direct_child_reaper_boundary") if disposition == signal.SIG_DFL else (False, "sigchld_not_default")


def _probe_direct_child_lifetime(pid: int, *, waitid_fn: Callable[..., Any] | None = None, signal_getter: Callable[[int], Any] | None = None) -> dict[str, Any]:
    ready, reason = _reaper_boundary_status(signal_getter=signal_getter)
    if not ready:
        return {"status": "fail", "reason": reason, "state": "unavailable"}
    waiter = os.waitid if waitid_fn is None else waitid_fn
    try:
        result = waiter(os.P_PID, int(pid), os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return {"status": "fail", "reason": "waitability_lost", "state": "lost"}
    except OSError as exc:
        if exc.errno == errno.ECHILD:
            return {"status": "fail", "reason": "waitability_lost", "state": "lost"}
        return {"status": "fail", "reason": f"waitid_error:{type(exc).__name__}", "state": "unavailable"}
    except (TypeError, ValueError, AttributeError):
        return {"status": "fail", "reason": "waitid_unavailable", "state": "unavailable"}
    observed_pid = getattr(result, "si_pid", 0) if result is not None else 0
    if observed_pid in (None, 0):
        return {"status": "pass", "reason": "direct_child_alive", "state": "alive"}
    if observed_pid != int(pid):
        return {"status": "fail", "reason": "waitid_pid_mismatch", "state": "lost"}
    return {"status": "pass", "reason": "direct_child_exited_unreaped", "state": "exited_unreaped"}


def _root_defunct_transition_allowed(expected: Mapping[str, Any] | None, observed: Mapping[str, Any] | None, lifetime: Mapping[str, Any] | None) -> bool:
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping) or not isinstance(lifetime, Mapping) or lifetime.get("state") != "exited_unreaped":
        return False
    for field in ("pid", "ppid", "pgid", "sid", "start_marker"):
        if observed.get(field) != expected.get(field):
            return False
    expected_command, observed_command = expected.get("command"), observed.get("command")
    return isinstance(expected_command, str) and isinstance(observed_command, str) and observed_command in {"<defunct>", f"{expected_command} <defunct>"}


def _validate_owned_process_group(root_pid: int, owned_pgid: int | None, root_identity: Mapping[str, Any] | None) -> tuple[bool, str]:
    if isinstance(owned_pgid, bool) or not isinstance(owned_pgid, int) or owned_pgid <= 0:
        return False, "invalid_owned_pgid"
    if owned_pgid != int(root_pid):
        return False, "root_is_not_group_leader"
    if not _has_identity_token(root_identity):
        return False, "root_identity_missing"
    if root_identity.get("pgid") != owned_pgid or root_identity.get("sid") != int(root_pid):
        return False, "root_session_or_group_identity_mismatch"
    token = root_identity.get("identity_token")
    if not isinstance(token, Mapping) or token.get("pgid") != owned_pgid or token.get("sid") != int(root_pid):
        return False, "root_session_or_group_identity_mismatch"
    try:
        if os.getpgrp() == owned_pgid:
            return False, "owned_pgid_is_harness_group"
    except OSError:
        return False, "harness_group_unavailable"
    return True, "validated_root_owned_group"


def _run_ps_snapshot(
    *,
    command_runner: Callable[..., Any] | None = None,
    pid: int | None = None,
    session_id_getter: Callable[[int], int] | None = None,
    required_pids: Iterable[int] = (),
    required_pgid: int | None = None,
    protect_descendants_of: int | None = None,
) -> dict[int, dict[str, Any]] | None:
    runner = subprocess.run if command_runner is None else command_runner
    getsid = getattr(os, "getsid", None) if session_id_getter is None else session_id_getter
    if not callable(getsid):
        return None
    try:
        completed = runner(_ps_command(pid), capture_output=True, text=True, check=False, timeout=COMMAND_TIMEOUT_S)
        if getattr(completed, "returncode", 0) not in (0, None):
            return None
        output = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
    except (OSError, subprocess.SubprocessError, TypeError):
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if not isinstance(output, str) or (isinstance(stderr, str) and stderr.strip()):
        return None
    nonempty = [line for line in output.splitlines() if line.strip()]
    if not nonempty:
        return None
    # Validate every wire row before session enrichment.  Descendant and
    # same-PGID rows are protected ownership evidence; only unrelated rows may
    # disappear between ps and getsid without making the sample unsafe.
    base = _parse_ps_wire_snapshot(output)
    if len(base) != len(nonempty):
        return None
    protected_pids = {int(raw_pid) for raw_pid in required_pids}
    protected_pgid = required_pgid
    if protect_descendants_of is not None:
        protected_root = int(protect_descendants_of)
        protected_pids.update(_descendant_closure(base, protected_root))
        root_wire = base.get(protected_root)
        if protected_pgid is None and isinstance(root_wire, Mapping):
            candidate_pgid = root_wire.get("pgid")
            if isinstance(candidate_pgid, int) and candidate_pgid > 0:
                protected_pgid = candidate_pgid
    try:
        parsed = _enrich_ps_sessions(
            base,
            session_id_getter=getsid,
            required_pids=protected_pids,
            required_pgid=protected_pgid,
        )
    except (OSError, ProcessLookupError, TypeError, ValueError):
        return None
    if not parsed or any(not _has_identity_token(item) for item in parsed.values()):
        return None
    return parsed


def capture_process_identities(root_pid: int, *, command_runner: Callable[..., Any] | None = None, session_id_getter: Callable[[int], int] | None = None) -> dict[int, dict[str, Any]]:
    snapshot = _run_ps_snapshot(
        command_runner=command_runner,
        session_id_getter=session_id_getter,
        required_pids={int(root_pid)},
        protect_descendants_of=int(root_pid),
    )
    if snapshot is None or int(root_pid) not in snapshot:
        return {}
    children: dict[int, set[int]] = {}
    for pid, item in snapshot.items():
        children.setdefault(int(item["ppid"]), set()).add(pid)
    owned = {int(root_pid)}
    frontier = [int(root_pid)]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, ()):
            if child not in owned:
                owned.add(child)
                frontier.append(child)
    return {pid: snapshot[pid] for pid in sorted(owned) if pid in snapshot}


def _merge_process_identities(owned: dict[int, dict[str, Any]], incoming: Mapping[int, Mapping[str, Any]]) -> set[int]:
    incomplete = {int(pid) for pid, item in incoming.items() if not _has_identity_token(item)}
    if incomplete:
        return incomplete
    conflicts = {int(pid) for pid, item in incoming.items() if int(pid) in owned and owned[int(pid)].get("identity_token") != item.get("identity_token")}
    if conflicts:
        return conflicts
    for pid, item in incoming.items():
        if int(pid) not in owned:
            owned[int(pid)] = dict(item)
    return conflicts


def _revalidate_owned_process_group(root_pid: int, owned_pgid: int, root_identity: Mapping[str, Any], owned_identities: Mapping[int, Mapping[str, Any]], *, command_runner: Callable[..., Any] | None = None, session_id_getter: Callable[[int], int] | None = None, root_lifetime: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    current = _run_ps_snapshot(
        command_runner=command_runner,
        session_id_getter=session_id_getter,
        required_pids={int(root_pid), *map(int, owned_identities)},
        required_pgid=int(owned_pgid),
    )
    if current is None:
        return False, "group_snapshot_unavailable"
    members = {pid: item for pid, item in current.items() if item.get("pgid") == owned_pgid}
    mismatches: list[int] = []
    matched: list[int] = []
    expected_identities = dict(owned_identities)
    expected_identities.setdefault(int(root_pid), root_identity)
    for pid, expected in expected_identities.items():
        observed = current.get(int(pid))
        if observed is None:
            continue
        if observed.get("identity_token") != expected.get("identity_token"):
            if int(pid) == int(root_pid) and _root_defunct_transition_allowed(expected, observed, root_lifetime):
                matched.append(int(pid))
                continue
            mismatches.append(int(pid))
        elif observed.get("pgid") == owned_pgid:
            matched.append(int(pid))
    if mismatches:
        return False, "owned_group_identity_mismatch"
    if not members:
        return True, "owned_group_absent"
    if not matched:
        return False, "owned_group_identity_unproven"
    return True, "validated_owned_group_observed"


def _read_process_identity(pid: int, *, command_runner: Callable[..., Any] | None = None, session_id_getter: Callable[[int], int] | None = None) -> dict[str, Any] | None:
    snapshot = _run_ps_snapshot(
        command_runner=command_runner,
        pid=pid,
        session_id_getter=session_id_getter,
        required_pids={int(pid)},
    )
    return snapshot.get(int(pid)) if snapshot is not None else None


def _listener_pids(port: int, *, command_runner: Callable[..., Any] | None = None) -> set[int] | None:
    runner = subprocess.run if command_runner is None else command_runner
    try:
        completed = runner(["/usr/sbin/lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, check=False, timeout=COMMAND_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, TypeError):
        return None
    returncode = getattr(completed, "returncode", 0)
    stderr = getattr(completed, "stderr", "") or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if returncode not in (0, 1, None) or (isinstance(stderr, str) and stderr.strip()):
        return None
    result: set[int] = set()
    output = getattr(completed, "stdout", "") or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            result.add(int(line.strip()))
        except ValueError:
            return None
    return result


def _verify_cleanup_once(root_pid: int, owned_pids: Iterable[int], port: int, *, command_runner: Callable[..., Any] | None = None, owned_identities: Mapping[int, Mapping[str, Any]] | None = None, identity_conflicts: Iterable[int] = (), owned_pgid: int | None = None, session_id_getter: Callable[[int], int] | None = None, waitid_fn: Callable[..., Any] | None = None, signal_getter: Callable[[int], Any] | None = None) -> dict[str, Any]:
    owned = {int(pid) for pid in owned_pids}
    captured = {int(pid): dict(value) for pid, value in (owned_identities or {}).items() if int(pid) in owned}
    root_identity = captured.get(int(root_pid))
    if owned_pgid is None and isinstance(root_identity, Mapping):
        candidate = root_identity.get("pgid")
        if isinstance(candidate, int):
            owned_pgid = candidate
    group_valid, group_reason = _validate_owned_process_group(root_pid, owned_pgid, root_identity)
    missing = sorted(pid for pid in owned if not _has_identity_token(captured.get(pid)))
    current = _run_ps_snapshot(
        command_runner=command_runner,
        session_id_getter=session_id_getter,
        required_pids=owned,
        required_pgid=owned_pgid,
    )
    listeners = _listener_pids(port, command_runner=command_runner)
    if current is None:
        listener_status = "fail" if listeners or missing or not group_valid else "unavailable"
        return {"status": listener_status, "provenance": "ps+lsof_unavailable", "remaining_owned_pids": sorted(owned), "listener_pids": sorted(listeners) if listeners is not None else None, "unexpected_listener_pids": sorted(listeners) if listeners else [], "uncaptured_owned_pids": missing, "identity_tracking": "captured_tokens" if not missing else "incomplete_tokens", "owned_pgid": owned_pgid, "group_guard": group_reason, "remaining_owned_group_pids": None}
    remaining: list[int] = []
    mismatches = sorted({int(pid) for pid in identity_conflicts})
    for pid in sorted(owned):
        observed = current.get(pid)
        if observed is None:
            continue
        expected = captured.get(pid)
        if expected is None:
            remaining.append(pid)
        elif observed.get("identity_token") != expected.get("identity_token"):
            lifetime = _probe_direct_child_lifetime(root_pid, waitid_fn=waitid_fn, signal_getter=signal_getter) if pid == int(root_pid) else None
            if not _root_defunct_transition_allowed(expected, observed, lifetime):
                mismatches.append(pid)
            else:
                remaining.append(pid)
        else:
            remaining.append(pid)
    group_remaining = sorted(pid for pid, item in current.items() if item.get("pgid") == owned_pgid)
    group_complete = all(item.get("pgid") is not None for item in current.values())
    if listeners is None:
        return {"status": "fail" if missing or not group_valid else "unavailable", "provenance": "ps+lsof_unavailable", "remaining_owned_pids": remaining, "listener_pids": None, "identity_mismatches": sorted(set(mismatches)), "uncaptured_owned_pids": missing, "identity_tracking": "captured_tokens" if not missing else "incomplete_tokens", "owned_pgid": owned_pgid, "group_guard": group_reason, "remaining_owned_group_pids": group_remaining if group_complete else None}
    passed = group_valid and group_complete and not remaining and not group_remaining and not listeners and not mismatches and not missing
    return {"status": "pass" if passed else "fail", "provenance": "ps+lsof", "remaining_owned_pids": remaining, "remaining_owned_group_pids": group_remaining if group_complete else None, "listener_pids": sorted(listeners), "remaining_owned_listener_pids": sorted(listeners & owned), "unexpected_listener_pids": sorted(listeners - owned), "identity_mismatches": sorted(set(mismatches)), "uncaptured_owned_pids": missing, "identity_tracking": "captured_tokens" if not missing else "incomplete_tokens", "owned_pgid": owned_pgid, "group_guard": group_reason}


def verify_cleanup(root_pid: int, owned_pids: Iterable[int], port: int, *, command_runner: Callable[..., Any] | None = None, owned_identities: Mapping[int, Mapping[str, Any]] | None = None, identity_conflicts: Iterable[int] = (), owned_pgid: int | None = None, session_id_getter: Callable[[int], int] | None = None, waitid_fn: Callable[..., Any] | None = None, signal_getter: Callable[[int], Any] | None = None, clean_snapshots_required: int = 2, cleanup_poll_interval_s: float = 0.05, sleep_fn: Callable[[float], Any] = time.sleep) -> dict[str, Any]:
    if isinstance(clean_snapshots_required, bool) or not isinstance(clean_snapshots_required, int) or not 2 <= clean_snapshots_required <= 8:
        return {"status": "fail", "provenance": "cleanup_sampling", "reason": "invalid_clean_snapshot_requirement", "required_clean_snapshots": clean_snapshots_required}
    try:
        poll_interval = float(cleanup_poll_interval_s)
    except (TypeError, ValueError):
        poll_interval = -1.0
    if not math.isfinite(poll_interval) or not 0.0 <= poll_interval <= 1.0:
        return {"status": "fail", "provenance": "cleanup_sampling", "reason": "invalid_cleanup_poll_interval", "required_clean_snapshots": clean_snapshots_required}
    last: dict[str, Any] | None = None
    for index in range(clean_snapshots_required):
        observation = _verify_cleanup_once(root_pid, owned_pids, port, command_runner=command_runner, owned_identities=owned_identities, identity_conflicts=identity_conflicts, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter)
        observation["cleanup_sampling"] = "bounded_consecutive"
        observation["required_clean_snapshots"] = clean_snapshots_required
        if observation.get("status") != "pass":
            observation["consecutive_clean_snapshots"] = index
            return observation
        last = observation
        if index + 1 < clean_snapshots_required:
            try:
                sleep_fn(poll_interval)
            except Exception as exc:
                return {**observation, "status": "unavailable", "reason": f"cleanup_poll_error:{type(exc).__name__}", "consecutive_clean_snapshots": index + 1}
    assert last is not None
    last["consecutive_clean_snapshots"] = clean_snapshots_required
    return last


def terminate_owned_processes(process: Any, owned_pids: Iterable[int], *, owned_identities: Mapping[int, Mapping[str, Any]] | None = None, identity_conflicts: Iterable[int] = (), command_runner: Callable[..., Any] | None = None, signal_sender: Callable[[int, int], Any] | None = None, group_signal_sender: Callable[[int, int], Any] | None = None, owned_pgid: int | None = None, session_id_getter: Callable[[int], int] | None = None, waitid_fn: Callable[..., Any] | None = None, signal_getter: Callable[[int], Any] | None = None, wait_timeout_s: float = 10.0) -> dict[str, Any]:
    root_pid = int(getattr(process, "pid", -1))
    owned = {int(pid) for pid in owned_pids if int(pid) > 0}
    root_identity = (owned_identities or {}).get(root_pid)
    if owned_pgid is None and isinstance(root_identity, Mapping):
        candidate = root_identity.get("pgid")
        if isinstance(candidate, int):
            owned_pgid = candidate
    group_valid, group_guard = _validate_owned_process_group(root_pid, owned_pgid, root_identity)
    attempted: list[int] = []
    mismatches = set(int(pid) for pid in identity_conflicts)
    group_observation = None
    group_error = None
    root_lifetime: dict[str, Any] | None = None
    group_signal_attempted = False
    if group_valid and mismatches:
        group_observation = "identity_conflict"
        group_error = "identity_conflict"
    elif group_valid:
        reaper_ready, reaper_reason = _reaper_boundary_status(signal_getter=signal_getter)
        if not reaper_ready:
            group_observation, group_error = reaper_reason, reaper_reason
        else:
            root_lifetime = _probe_direct_child_lifetime(root_pid, waitid_fn=waitid_fn, signal_getter=signal_getter)
            if getattr(process, "returncode", None) is not None:
                group_observation, group_error = "root_lifetime_not_pinned", "root_lifetime_not_pinned"
            elif root_lifetime.get("status") != "pass":
                group_observation = str(root_lifetime.get("reason", "waitid_unavailable"))
                group_error = group_observation
            else:
                ready, group_observation = _revalidate_owned_process_group(root_pid, int(owned_pgid), root_identity, owned_identities or {}, command_runner=command_runner, session_id_getter=session_id_getter, root_lifetime=root_lifetime)
                if not ready and group_observation != "owned_group_absent":
                    group_error = group_observation
                elif group_observation != "owned_group_absent":
                    reaper_ready, reaper_reason = _reaper_boundary_status(signal_getter=signal_getter)
                    if not reaper_ready:
                        group_error = reaper_reason
                    else:
                        try:
                            (group_signal_sender or os.killpg)(int(owned_pgid), signal.SIGTERM)
                            group_signal_attempted = True
                        except ProcessLookupError:
                            group_signal_attempted = True
                        except OSError as exc:
                            group_error = type(exc).__name__
    # The group signal is always decided before Popen termination/reaping.  The
    # root is the only fallback owner; never signal descendants by PID.
    try:
        process.terminate()
        attempted.append(root_pid)
    except (OSError, subprocess.SubprocessError, TimeoutError):
        pass
    try:
        process.wait(timeout=wait_timeout_s)
    except (TimeoutError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=wait_timeout_s)
        except (OSError, subprocess.SubprocessError, TimeoutError):
            pass
    return {"status": "pass" if not mismatches and group_valid and group_error is None else "fail", "root_pid": root_pid, "attempted_pids": sorted(set(attempted)), "owned_pgid": owned_pgid, "group_guard": group_guard, "group_observation": group_observation, "group_signal_attempted": group_signal_attempted, "group_signal_error": group_error, "root_lifetime": root_lifetime, "identity_mismatches": sorted(mismatches), "identity_unavailable": []}


def _source_launch_command(clone: CloneHandle, config: RuntimeConfig, port: int) -> list[str]:
    return [
        str(clone.venv_python),
        str(Path(config.repo_root).resolve() / "studio" / "backend" / "run.py"),
        "--host", "127.0.0.1",
        "--port", str(int(port)),
        "--api-only", "--no-cloudflare", "--silent", "--disable-tools", "--password", "-",
    ]


def _validate_source_process_identity(
    identity: Mapping[str, Any] | None,
    expected_command: Sequence[str],
    repo_root: Path,
) -> None:
    if not _has_identity_token(identity):
        raise IdentityMismatchError("source process identity token was unavailable")
    command = identity.get("command") if isinstance(identity, Mapping) else None
    if not isinstance(command, str):
        raise IdentityMismatchError("source process command origin was unavailable")
    # ps presents one shell-escaped command string.  Requiring each exact
    # launch token binds the health response to this source run.py invocation,
    # including the interpreter and repository module origin.
    if any(str(token) not in command for token in expected_command):
        raise IdentityMismatchError("source process command/module origin did not match launch contract")
    if str(Path(expected_command[1]).resolve()) not in command or str(Path(repo_root).resolve()) not in command:
        raise IdentityMismatchError("source process was not launched from the expected repository module")


def _validate_owned_loopback_listener(
    port: int,
    owned_pids: Iterable[int],
    *,
    command_runner: Callable[..., Any] | None = None,
) -> set[int]:
    listeners = _listener_pids(port, command_runner=command_runner)
    if listeners is None or not listeners:
        raise HarnessError("owned loopback listener could not be observed")
    owned = {int(pid) for pid in owned_pids}
    unexpected = listeners - owned
    if unexpected:
        raise IdentityMismatchError(
            f"loopback health listener was not owned by the source process: {sorted(unexpected)}"
        )
    return listeners


def _launch_source(
    clone: CloneHandle,
    config: RuntimeConfig,
    port: int,
    *,
    password: str,
    popen_factory: Callable[..., Any] | None = None,
) -> Any:
    command = _source_launch_command(clone, config, port)
    factory = subprocess.Popen if popen_factory is None else popen_factory
    environment = build_environment(clone.run_dir, Path(config.repo_root).resolve(), provider_tier=config.provider_tier)
    process = factory(
        command,
        cwd=str(Path(config.repo_root).resolve()),
        env=environment,
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
        raise HarnessError("source process did not expose stdin for password handoff")
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
        raise HarnessError("source process password handoff failed") from exc
    return process


def _wait_for_health(
    transport: Any,
    base_url: str,
    *,
    timeout: float,
    process: Any | None = None,
    port: int | None = None,
    owned_pids: Iterable[int] = (),
    command_runner: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    deadline = clock() + timeout
    last_error: Exception | None = None
    while clock() < deadline:
        if process is not None and getattr(process, "poll", lambda: None)() is not None:
            raise HarnessError("source backend exited before health")
        try:
            payload = _json_request(transport, "GET", base_url, "/api/health", headers={}, timeout=min(5.0, timeout))
            identity = _validate_health_identity(payload)
            if port is not None:
                _validate_owned_loopback_listener(port, owned_pids, command_runner=command_runner)
            return identity
        except (HarnessError, IdentityMismatchError) as exc:
            if isinstance(exc, IdentityMismatchError):
                raise
            last_error = exc
            time.sleep(0.05)
    raise HarnessError("source backend did not become healthy before deadline") from last_error


def atomic_artifact(output_dir: Path, record: Mapping[str, Any], *, secrets: Iterable[str] = ()) -> Path:
    directory = _require_abs_directory(output_dir, "output directory")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = directory / f"source-runtime-{stamp}-{os.getpid()}.json"
    payload = json.dumps(_sanitize(record, secrets), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(directory))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _dry_run(config: RuntimeConfig, paths: Mapping[str, Any], source_identity: Mapping[str, Any]) -> dict[str, Any]:
    model = model_identity(config)
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "runner_version": SCRIPT_VERSION,
        "benchmark_id": "helix-v3-source-runtime",
        "scenario_id": "SOURCE-DURABLE-TRACE",
        "status": "dry_run",
        "would_launch": False,
        "offline": True,
        "source_identity": dict(source_identity),
        "runner_identity": runner_identity(Path(paths["repo"])),
        "hardware": hardware_identity(),
        "model": model,
        "runtime": {"runtime_home_template": str(paths["source"]), "run_dir": str(paths["run_dir"]), "venv_python": str(paths["template_python"]), "studio_home": str(paths["run_dir"] / ".unsloth" / "studio")},
        "configuration_sha256": _sha256_json({"model_name": config.model_name, "provider_tier": config.provider_tier, "provider_identity": config.provider_identity, "prompt": config.prompt, "seed": config.seed, "max_tokens": config.max_tokens}),
        "actions": ["APFS cp -cR clone", "source run.py launch with password on stdin", "owner login", "thread/message/durable run", "frontend SSE replay", "offline SQLite trace validation", "identity-fenced shutdown"],
    }


def run_benchmark(
    config: RuntimeConfig,
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
) -> dict[str, Any]:
    paths = validate_paths(config)
    before_source = _source_identity(Path(paths["repo"]), command_runner=source_identity_runner)
    if not config.execute:
        return _dry_run(config, paths, before_source)

    # The execute path has already required these inputs in validate_paths;
    # keep the explicit check here so a future caller cannot accidentally make
    # model loading optional by bypassing that helper.
    if config.model_path is None or config.model_config_path is None:
        raise ContractError("execute requires both a local model path and model config")
    model_spec = validate_offline_model_spec(config.model_path, config.model_config_path)
    load_payload = build_load_payload(model_spec, seed=config.seed)
    admitted_model_names = _local_model_contract_names(
        model_spec["model_path"],
        configured_model_name=config.model_name,
        model_config=model_spec["config"],
    )
    clone: CloneHandle | None = None
    process: Any | None = None
    password: str | None = None
    headers: dict[str, str] | None = None
    base_url: str | None = None
    port: int | None = None
    expected_command: list[str] | None = None
    owned: dict[int, dict[str, Any]] = {}
    owned_pids: set[int] = set()
    owned_pgid: int | None = None
    identity_conflicts: set[int] = set()
    root_pid: int | None = None
    cleanup: dict[str, Any] = {"status": "unavailable", "provenance": "not_started"}
    objective_record: dict[str, Any] | None = None
    admitted_model_path: str = str(model_spec["model_path"])
    public_model_name: str = derive_public_model_identity(
        model_spec["model_path"],
        model_spec["config"],
    )
    loaded = False
    unloaded = False
    shutdown_requested = False
    shutdown_admitted = False
    record: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA,
        "runner_version": SCRIPT_VERSION,
        "benchmark_id": "helix-v3-source-runtime",
        "scenario_id": "SOURCE-DURABLE-TRACE",
        "status": "running",
        "offline": True,
        "source_identity": {"before": before_source, "after": None},
        "runner_identity": runner_identity(Path(paths["repo"])),
        "hardware": hardware_identity(),
        "model": model_identity(config),
        "runtime": {"runtime_home_template": str(paths["source"]), "run_dir": str(paths["run_dir"]), "venv_python": None, "studio_home": None},
        "configuration": {
            "seed": config.seed,
            "max_tokens": config.max_tokens,
            "model_config_sha256": _sha256_bytes(config.model_config_path.read_bytes()),
            "load": load_payload,
            "benchmark_configuration_sha256": _sha256_json({"load": load_payload, "seed": config.seed, "max_tokens": config.max_tokens}),
        },
        "metrics": unavailable_metrics(),
    }
    safe_secrets: list[str] = []
    http = transport or UrllibTransport()

    def fail(exc: Exception | str) -> None:
        record["status"] = "failed"
        message = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
        if record.get("error"):
            record["error"] = f"{record['error']}; {message}"
        else:
            record["error"] = message

    try:
        clone = clone_runtime_home(Path(paths["source"]), Path(paths["run_dir"]), copy_runner=copy_runner)
        record["runtime"].update({"venv_python": str(clone.venv_python), "studio_home": str(clone.studio_home)})
        current_password = read_password(_password_path(clone.run_dir))
        safe_secrets.append(current_password)
        if config.password_file:
            password = read_password(Path(paths["password"]))
            if password == current_password:
                raise ContractError(
                    "explicit launch password must differ from the clone bootstrap password"
                )
        else:
            password = ephemeral_launch_password(current_password)
        safe_secrets.append(password)
        port = int((port_allocator or allocate_ephemeral_port)())
        if not 1 <= port <= 65535:
            raise ContractError("ephemeral port was invalid")
        reaper_ready, reaper_reason = _reaper_boundary_status(signal_getter=signal_getter)
        if not reaper_ready:
            raise IdentityMismatchError(f"direct-child reaper boundary unavailable: {reaper_reason}")
        expected_command = _source_launch_command(clone, config, port)
        process = _launch_source(clone, config, port, password=password, popen_factory=popen_factory)
        root_pid = int(getattr(process, "pid"))
        incoming = capture_process_identities(root_pid, command_runner=command_runner, session_id_getter=session_id_getter)
        root_identity = incoming.get(root_pid)
        _validate_source_process_identity(root_identity, expected_command, Path(paths["repo"]))
        root_pgid = root_identity.get("pgid") if isinstance(root_identity, Mapping) else None
        valid, reason = _validate_owned_process_group(root_pid, root_pgid if isinstance(root_pgid, int) else None, root_identity)
        if not valid:
            raise IdentityMismatchError(f"source process identity could not be fenced: {reason}")
        owned_pgid = root_pgid
        identity_conflicts.update(_merge_process_identities(owned, incoming))
        owned_pids.update(owned)
        if identity_conflicts:
            raise IdentityMismatchError("source process identity was reused during launch")
        base_url = f"http://127.0.0.1:{port}"
        record["health_identity"] = _wait_for_health(
            http,
            base_url,
            timeout=config.startup_timeout_s,
            process=process,
            port=port,
            owned_pids=owned_pids,
            command_runner=command_runner,
        )
        record["health_binding"] = {
            "loopback": base_url,
            "listener_pids": sorted(_validate_owned_loopback_listener(port, owned_pids, command_runner=command_runner)),
            "launch_command": list(expected_command),
            "source_run_module": str(Path(paths["repo"]) / "studio" / "backend" / "run.py"),
            "repo_root": str(Path(paths["repo"])),
            "listener_verified": True,
        }
        headers, auth_record = login_owner(http, base_url, password, config.request_timeout_s)
        record["auth"] = auth_record
        load_response = _json_request(
            http,
            "POST",
            base_url,
            "/api/inference/load",
            headers=headers,
            payload=load_payload,
            timeout=config.request_timeout_s,
        )
        if _operation_failed(load_response):
            raise HarnessError("local MLX model load failed")
        # A successful load may have installed weights even if its identity
        # payload is malformed; mark it loaded before parsing that identity so
        # the finally block still unloads before shutdown.
        loaded = True
        admitted_model_path = select_model_request_name(
            load_response,
            model_spec["model_path"],
            allowed_names=admitted_model_names,
        )
        public_model_name = derive_public_model_identity(
            model_spec["model_path"],
            model_spec["config"],
        )
        record["model"]["name"] = public_model_name
        record["model"]["public_model_id"] = public_model_name
        record["load"] = {
            "status": load_response.get("status") if isinstance(load_response, Mapping) else None,
            "model": admitted_model_path,
            "public_model_id": public_model_name,
            "backend": "mlx",
            "model_path": str(model_spec["model_path"]),
            "config_sha256": record["configuration"]["model_config_sha256"],
        }
        request_config = replace(config, model_name=public_model_name)
        objective = create_objective(http, base_url, headers, request_config)
        request_payload = objective.get("request_payload")
        if not isinstance(request_payload, Mapping):
            raise ProtocolMismatchError("objective request payload was not an object")
        objective_record = {
            "owner_subject": OWNER_USERNAME,
            "account_id": OWNER_ACCOUNT_ID,
            "run_id": objective["run_id"],
            "thread_id": objective["thread_id"],
            "user_message_id": objective["user_message_id"],
            "assistant_message_id": objective["assistant_message_id"],
            "request_summary": summarize_request_payload(request_payload),
        }
        record["objective"] = objective_record
        started = clock()
        sse = consume_frontend_sse(http, base_url, headers, objective["run_id"], timeout=config.request_timeout_s, clock=clock)
        visible_answer = reconstruct_visible_answer(sse)
        record["assistant_persistence"] = persist_assistant_answer(
            http,
            base_url,
            headers,
            objective=objective,
            sse=sse,
            answer=visible_answer,
            timeout=config.request_timeout_s,
        )
        metric_record = collect_metrics(sse, elapsed_ms=(clock() - started) * 1000.0)
        record["metrics"] = metric_record["metrics"]
        record["sse"] = summarize_frontend_sse(
            sse,
            metrics=metric_record["metrics"],
        )
        current = capture_process_identities(root_pid, command_runner=command_runner, session_id_getter=session_id_getter)
        identity_conflicts.update(_merge_process_identities(owned, current))
        owned_pids.update(owned)
        if identity_conflicts:
            raise IdentityMismatchError("source process identity changed during the run")
        if expected_command is not None:
            _validate_source_process_identity(owned.get(root_pid), expected_command, Path(paths["repo"]))
    except Exception as exc:
        fail(exc)
    finally:
        # Admission order is deliberate: unload first, then request shutdown;
        # durable SQLite is not opened until shutdown has been admitted and the
        # process has been proven reaped by the cleanup boundary below.
        if loaded and not unloaded and base_url is not None and headers is not None:
            try:
                unload = _json_request(
                    http,
                    "POST",
                    base_url,
                    "/api/inference/unload",
                    headers=headers,
                    payload={"model_path": str(model_spec["model_path"]), "force_cancel_active": False},
                    timeout=config.request_timeout_s,
                )
                unloaded_model_name = validate_unload_response(
                    unload,
                    admitted_model_name=admitted_model_path,
                    model_path=model_spec["model_path"],
                )
                record["unload"] = {
                    "status": unload.get("status") if isinstance(unload, Mapping) else None,
                    "model": unloaded_model_name,
                    "model_path": str(model_spec["model_path"]),
                }
                unloaded = True
            except Exception as exc:
                record["unload"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                fail(exc)
        if base_url is not None and headers is not None and not shutdown_requested:
            try:
                shutdown = _json_request(http, "POST", base_url, "/api/shutdown", headers=headers, timeout=config.request_timeout_s)
                shutdown_requested = True
                record["shutdown"] = {"status": shutdown.get("status") if isinstance(shutdown, Mapping) else None, "ok": shutdown.get("ok") if isinstance(shutdown, Mapping) else None}
                if _operation_failed(shutdown):
                    fail("source shutdown request failed")
                else:
                    shutdown_admitted = True
            except Exception as exc:
                shutdown_requested = True
                fail(exc)
        if process is not None and root_pid is not None:
            current = capture_process_identities(root_pid, command_runner=command_runner, session_id_getter=session_id_getter)
            if current:
                identity_conflicts.update(_merge_process_identities(owned, current))
                owned_pids.update(owned)
            if port is not None:
                cleanup = verify_cleanup(root_pid, owned_pids, port, command_runner=command_runner, owned_identities=owned, identity_conflicts=identity_conflicts, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter)
            if getattr(process, "poll", lambda: 0)() is None or cleanup.get("status") != "pass":
                termination = terminate_owned_processes(process, owned_pids, owned_identities=owned, identity_conflicts=identity_conflicts, command_runner=command_runner, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter, group_signal_sender=group_signal_sender)
                cleanup = {**cleanup, "termination": termination}
                if port is not None:
                    cleanup = {**verify_cleanup(root_pid, owned_pids, port, command_runner=command_runner, owned_identities=owned, identity_conflicts=identity_conflicts, owned_pgid=owned_pgid, session_id_getter=session_id_getter, waitid_fn=waitid_fn, signal_getter=signal_getter), "termination": termination}
            process_reaped = getattr(process, "poll", lambda: None)() is not None
            if cleanup.get("status") != "pass" or not process_reaped:
                fail("source process cleanup/reap was not proven")
            elif shutdown_admitted and record.get("status") == "running" and objective_record is not None and clone is not None:
                try:
                    evidence = read_durable_evidence(
                        clone.studio_home / "studio.db",
                        objective=objective_record,
                        model_name=public_model_name,
                        provider_tier=config.provider_tier,
                        provider_identity=config.provider_identity,
                    )
                    record["durable_evidence"] = evidence
                    record["trace_aggregation"] = evidence["trace"]["aggregation"]
                    record["task_success"] = evidence["task_success"]
                    record["trace_availability"] = evidence["trace_availability"]
                    record["status"] = "complete"
                except Exception as exc:
                    fail(exc)
        record["cleanup"] = cleanup
        try:
            after_source = _source_identity(Path(paths["repo"]), command_runner=source_identity_runner)
            verify_source_identity(before_source, after_source)
            record["source_identity"]["after"] = after_source
        except Exception as exc:
            record["source_identity"]["after_error"] = f"{type(exc).__name__}: {exc}"
            fail(exc)
        removed = False
        if clone is not None:
            try:
                removed = remove_clone_if_proven(clone, cleanup_proven=record.get("status") == "complete" and cleanup.get("status") == "pass")
            except Exception as exc:
                record["runtime"]["clone_remove_error"] = f"{type(exc).__name__}: {exc}"
        record["runtime"]["clone_removed"] = removed
        if record.get("status") == "running":
            fail("benchmark lifecycle did not reach a terminal state")
        # Output is explicit and never contains password/access-token values.
        record["artifact_path"] = str(atomic_artifact(Path(paths["output_dir"]), record, secrets=safe_secrets))
    return _sanitize(record, safe_secrets)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="clone and launch; default is dry-run")
    parser.add_argument("--runtime-home-template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="new explicit clone directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="explicit artifact directory")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--password-file", type=Path, default=None, help="read-only bootstrap password file; value is sent only to child stdin")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--provider-tier", default=DEFAULT_PROVIDER_TIER)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        runtime_home_template=args.runtime_home_template,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        password_file=args.password_file,
        model_name=args.model_name,
        model_path=args.model_path,
        model_config_path=args.model_config,
        prompt=args.prompt,
        provider_tier=args.provider_tier,
        startup_timeout_s=args.startup_timeout,
        request_timeout_s=args.request_timeout,
        seed=args.seed,
        max_tokens=args.max_tokens,
        execute=bool(args.execute),
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_benchmark(_config_from_args(_parse_args(argv)))
    except Exception as exc:
        result = {"schema_version": BENCHMARK_SCHEMA, "runner_version": SCRIPT_VERSION, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(_sanitize(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "dry_run" else 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
