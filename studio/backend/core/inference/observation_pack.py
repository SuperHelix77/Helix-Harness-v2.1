# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Disabled ObservationPack primitives.

This module deliberately stops before durable execution authority. Candidate
construction is an eligibility calculation, and publication is an immutable
blob write. Neither operation creates an account-database binding, marks a
binding available, or chooses the result of a durable tool finish. A later
backend adapter must bind the returned metadata in the same transaction as the
ordinary durable finish before handing an ``ObservationBinding`` to recall.
"""

from __future__ import annotations

import hashlib
import hmac
import errno
import os
import re
import stat
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Iterator, Mapping, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from auth.storage import (
    get_observation_master_key,
    get_observation_master_key_id,
    get_or_create_observation_master_key,
    observation_master_key_id_for,
)
from utils.account_context import is_owner_context
from utils.paths.storage_roots import RetiredAccountError, root_retirement_lock, workspace_root


OBSERVATION_PACK_ENABLED = False
OBSERVATION_FORMAT_VERSION = 1
CANONICALIZATION_VERSION = "helix.tool-text.v1"
PROJECTION_VERSION = "helix.observation-projection.v1"
OBSERVATION_HANDLE_VERSION = "v1"

MIN_OBSERVATION_BYTES = 10 * 1024
MAX_OBSERVATION_BYTES = 8 * 1024 * 1024
MAX_PROJECTION_CHARS = 4096
MAX_RECALL_BYTES = 16_384
MAX_RECALL_LINES = 400
MIN_ACTIVE_RESULT_BUDGET_BYTES = 256
_NONCE_BYTES = 12
_GCM_TAG_BYTES = 16
DEFAULT_QUOTA_BYTES = 64 * 1024 * 1024
_KEY_ID_RE = re.compile(r"^observation-key-v1:sha256:[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_BINDING_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_HANDLE_RE = re.compile(r"^obs:v1:([0-9a-f]{64}):sha256:([0-9a-f]{64})$")
_IDENTITY_RE = re.compile(r"^[^/\\\x00]+$")


class ObservationError(RuntimeError):
    """Base class for typed optional-packing failures."""


class ObservationPathError(ObservationError):
    """A blob path failed the regular-file/no-follow identity checks."""


class ObservationBlobConflict(ObservationError):
    """An immutable blob path already contains different authenticated bytes."""


@dataclass(frozen=True)
class BlobMetadata:
    binding_id: str
    blob_relpath: str
    key_id: str
    plaintext_length: int
    plaintext_digest: str
    ciphertext_length: int
    ciphertext_digest: str
    blob_dev: int
    blob_ino: int

    @property
    def relative_path(self) -> str:
        return self.blob_relpath

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ObservationCandidate:
    """Immutable staged or published candidate, never an authoritative result."""

    state: str
    eligible: bool
    account_id: str
    thread_id: str
    source_run_id: str
    execution_id: str
    tool_name: str
    tool_call_id: str
    approval_id: Optional[str]
    claim_token: Optional[str]
    worker_id: Optional[str]
    arguments_fingerprint: str
    binding_id: Optional[str]
    handle: Optional[str]
    projection: Optional[str]
    fallback_result: str
    is_error: bool
    classification: str
    reason: Optional[str]
    format_version: int
    canonicalization_version: str
    projection_version: str
    plaintext_length: Optional[int]
    plaintext_digest: Optional[str]
    ciphertext_length: Optional[int]
    key_id: Optional[str]
    blob_relpath: Optional[str]
    blob_metadata: Optional[BlobMetadata]
    quota_bytes: int
    created_at: str
    canonical_bytes: Optional[bytes] = None

    @property
    def available(self) -> bool:
        # A staged/published candidate is not available until the account DB
        # finish transaction creates and returns an available binding.
        return False

    @property
    def verified(self) -> bool:
        return False

    @property
    def selected_result(self) -> None:
        """No primitive in this module selects a durable finish result."""
        return None

    def to_dict(self) -> dict:
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "canonical_bytes"
        }
        if self.blob_metadata is not None:
            values["blob_metadata"] = self.blob_metadata.to_dict()
        return values

    def binding_metadata(self) -> Mapping[str, object]:
        """Return metadata for a later account-DB finish adapter.

        The returned mapping intentionally has no ``state=available`` and no
        selected result. The finish adapter must add those only while it
        commits the durable completion.
        """
        blob = self.blob_metadata
        values = {
            "binding_id": self.binding_id,
            "account_id": self.account_id,
            "source_run_id": self.source_run_id,
            "thread_id": self.thread_id,
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "approval_id": self.approval_id,
            "claim_token": self.claim_token,
            "worker_id": self.worker_id,
            "arguments_fingerprint": self.arguments_fingerprint,
            "handle": self.handle,
            "format_version": self.format_version,
            "canonicalization_version": self.canonicalization_version,
            "projection_version": self.projection_version,
            "plaintext_length": self.plaintext_length,
            "plaintext_digest": self.plaintext_digest,
            "ciphertext_length": blob.ciphertext_length if blob else self.ciphertext_length,
            "ciphertext_digest": blob.ciphertext_digest if blob else None,
            "key_id": blob.key_id if blob else self.key_id,
            "blob_relpath": blob.blob_relpath if blob else self.blob_relpath,
            "blob_dev": blob.blob_dev if blob else None,
            "blob_ino": blob.blob_ino if blob else None,
            "projection": self.projection,
            "fallback_result": self.fallback_result,
            "is_error": self.is_error,
            "classification": self.classification,
        }
        return MappingProxyType(values)


# Compatibility name for callers that only used the old preparation result.
ObservationPreparation = ObservationCandidate


@dataclass(frozen=True)
class ObservationBinding:
    """Caller-authorized available binding consumed by low-level recall.

    This is a value object, not an auth-db row and not an authorization check.
    The backend finish adapter owns constructing it after its durable commit.
    """

    binding_id: str
    account_id: str
    source_run_id: str
    thread_id: str
    execution_id: str
    tool_name: str
    tool_call_id: str
    approval_id: Optional[str]
    claim_token: Optional[str]
    worker_id: Optional[str]
    arguments_fingerprint: str
    handle: str
    state: str
    format_version: int
    canonicalization_version: str
    projection_version: str
    plaintext_length: int
    plaintext_digest: str
    ciphertext_length: int
    ciphertext_digest: str
    key_id: str
    blob_relpath: str
    blob_dev: int
    blob_ino: int
    projection: str
    fallback_result: str = ""
    is_error: bool = False
    classification: str = "ordinary"
    finished_event_seq: Optional[int] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    @property
    def available(self) -> bool:
        return self.state == "available"

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ObservationPage:
    status: str
    text: str
    handle: Optional[str]
    start_offset: int
    end_offset: int
    next_offset: Optional[int]
    byte_length: int
    byte_digest: Optional[str]
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def __getitem__(self, key: str):
        return getattr(self, key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_identity(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or _IDENTITY_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a non-empty single path component")
    return value


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_binding_id(value: str) -> str:
    if not isinstance(value, str) or _BINDING_ID_RE.fullmatch(value) is None:
        raise ValueError("binding_id must be a lowercase SHA-256 digest")
    return value


def _derive_key(master_key: bytes, domain: bytes, account_id: str) -> bytes:
    if not isinstance(master_key, bytes) or len(master_key) != 32:
        raise ValueError("Observation master key must be 32 bytes")
    _require_identity("account_id", account_id)
    return HKDF(
        algorithm = hashes.SHA256(),
        length = 32,
        salt = None,
        info = domain + b"\0" + account_id.encode("utf-8"),
    ).derive(master_key)


def derive_observation_encryption_key(
    account_id: str, *, master_key: Optional[bytes] = None
) -> bytes:
    return _derive_key(
        master_key or get_or_create_observation_master_key(),
        b"helix.observation.encryption.v1",
        account_id,
    )


def derive_observation_handle_key(
    account_id: str, *, master_key: Optional[bytes] = None
) -> bytes:
    return _derive_key(
        master_key or get_or_create_observation_master_key(),
        b"helix.observation.handle.v1",
        account_id,
    )


def observation_binding_id(
    account_id: str,
    source_run_id: str,
    execution_id: str,
    arguments_fingerprint: str,
) -> str:
    for name, value in (
        ("account_id", account_id),
        ("source_run_id", source_run_id),
        ("execution_id", execution_id),
        ("arguments_fingerprint", arguments_fingerprint),
    ):
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"{name} must be NUL-free text")
    payload = "\0".join(
        (
            "helix.observation.binding.v1",
            account_id,
            source_run_id,
            execution_id,
            arguments_fingerprint,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_observation_handle(
    account_id: str,
    thread_id: str,
    content_digest: str,
    *,
    master_key: Optional[bytes] = None,
) -> str:
    _require_identity("account_id", account_id)
    _require_identity("thread_id", thread_id)
    digest = _require_digest("content_digest", content_digest)
    key = derive_observation_handle_key(account_id, master_key = master_key)
    message = "\0".join(
        ("helix.observation.handle.v1", account_id, thread_id, OBSERVATION_HANDLE_VERSION, digest)
    ).encode("utf-8")
    token = hmac.new(key, message, hashlib.sha256).hexdigest()
    return f"obs:v1:{token}:sha256:{digest}"


def parse_observation_handle(handle: str) -> tuple[str, str]:
    if not isinstance(handle, str):
        raise ValueError("Observation handle must be text")
    match = _HANDLE_RE.fullmatch(handle)
    if match is None:
        raise ValueError("Invalid ObservationPack handle")
    return match.group(1), match.group(2)


def canonicalize_observation_text(value: str | bytes) -> tuple[str, bytes]:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors = "replace")
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("canonical observation must be text or bytes")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise ValueError("canonical observation contains an unpaired UTF-16 surrogate")
    return text, text.encode("utf-8")


def canonical_observation_digest(value: str | bytes) -> tuple[str, bytes]:
    _text, encoded = canonicalize_observation_text(value)
    return hashlib.sha256(encoded).hexdigest(), encoded


def _aad(account_id: str, key_id: str, digest: str, plaintext_length: int) -> bytes:
    _require_identity("account_id", account_id)
    _require_digest("plaintext_digest", digest)
    if not _KEY_ID_RE.fullmatch(key_id):
        raise ValueError("Invalid ObservationPack key id")
    if not isinstance(plaintext_length, int) or plaintext_length < 0:
        raise ValueError("Invalid plaintext length")
    fields = ("helix.observation.blob.v1", account_id, key_id, digest, str(plaintext_length))
    return b"".join(
        len(field.encode("utf-8")).to_bytes(4, "big") + field.encode("utf-8")
        for field in fields
    )


def _safe_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or "//" in value
    ):
        raise ObservationPathError("Observation blob identity is not a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ObservationPathError("Observation blob identity is not a safe relative path")
    return value


def observation_blob_root(account_id: str, *, storage_root: Optional[Path | str] = None) -> Path:
    """Return a private account workspace path, never a global account tree."""
    _require_identity("account_id", account_id)
    if storage_root is not None:
        base = Path(storage_root)
    else:
        # workspace_root is the account-retired/renamed root. Do not invent a
        # second studio_root()/accounts tree outside that lifecycle.
        base = workspace_root() / "observation-packs"
    return base / account_id


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _normalise_existing_system_prefix(root: Path) -> Path:
    """Resolve only known macOS compatibility links in the existing prefix.

    A user-controlled symlink anywhere else is rejected. The returned path is
    then traversed from ``/`` by descriptor-relative opens; it is never opened
    as one absolute path.
    """
    raw = Path(os.path.abspath(os.fspath(root)))
    current = Path(raw.anchor or os.sep)
    parts = raw.parts[1:] if raw.anchor else raw.parts
    for index, part in enumerate(parts):
        candidate = current / part
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            return current.joinpath(*parts[index:])
        if stat.S_ISLNK(info.st_mode):
            # macOS presents /tmp and /var as compatibility symlinks. They are
            # the only symlink components this storage primitive deliberately
            # resolves; account/private components remain no-follow.
            if candidate in {Path("/tmp"), Path("/var")}:
                current = Path(os.path.realpath(candidate))
                continue
            raise ObservationPathError("Observation directory has an unsafe symlink ancestor")
        if not stat.S_ISDIR(info.st_mode):
            raise ObservationPathError("Observation directory has a non-directory ancestor")
        current = candidate
    return current


def _check_publication_lifecycle() -> None:
    """Preserve account retirement fencing without being a path-security check."""
    if not is_owner_context():
        from core.training.account_jobs import account_is_retired

        if account_is_retired():
            raise RetiredAccountError("account has been deleted; refusing ObservationPack publication")


def _open_or_create_directory(root: Path, *, create: bool = True) -> int:
    """Traverse from a trusted anchor using retained no-follow dirfds."""
    target = _normalise_existing_system_prefix(root)
    if target.anchor != os.path.abspath(os.sep):
        raise ObservationPathError("Observation directory is not absolute")
    parts = target.parts[1:]
    fd = os.open(os.path.abspath(os.sep), _directory_flags())
    try:
        for part in parts:
            if part in {"", ".", ".."} or "/" in part or "\\" in part or "\x00" in part:
                raise ObservationPathError("Observation directory has an unsafe component")
            created_component = False
            try:
                next_fd = os.open(part, _directory_flags(), dir_fd = fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd = fd)
                created_component = True
                try:
                    next_fd = os.open(part, _directory_flags(), dir_fd = fd)
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise ObservationPathError("Observation directory changed to an unsafe path") from exc
                    raise
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ObservationPathError("Observation directory changed to an unsafe path") from exc
                raise
            if created_component:
                os.fchmod(next_fd, 0o700)
            os.close(fd)
            fd = next_fd
        if parts:
            final_info = os.fstat(fd)
            if not stat.S_ISDIR(final_info.st_mode) or final_info.st_mode & 0o077:
                raise ObservationPathError("Observation directory is not private")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _blob_directory(root: Path, *, create: bool = True) -> Iterator[tuple[int, int]]:
    lock = root_retirement_lock if create else nullcontext()
    with lock:
        if create:
            _check_publication_lifecycle()
        root_fd = _open_or_create_directory(root, create = create)
        blob_fd: Optional[int] = None
        try:
            blob_created = False
            if create:
                try:
                    os.mkdir("blobs", 0o700, dir_fd = root_fd)
                    blob_created = True
                except FileExistsError:
                    pass
            try:
                blob_fd = os.open("blobs", _directory_flags(), dir_fd = root_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ObservationPathError("Observation blob directory changed to an unsafe path") from exc
                raise
            blob_info = os.fstat(blob_fd)
            if not stat.S_ISDIR(blob_info.st_mode) or blob_info.st_mode & 0o077:
                raise ObservationPathError("Observation blob directory is not private")
            if blob_created:
                os.fchmod(blob_fd, 0o700)
            yield root_fd, blob_fd
        finally:
            if blob_fd is not None:
                os.close(blob_fd)
            os.close(root_fd)


def _fsync_directory(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        if getattr(exc, "errno", None) not in (
            getattr(os, "EINVAL", 22),
            getattr(os, "ENOTSUP", 95),
        ):
            raise


def _read_fd(fd: int, expected_length: Optional[int] = None) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_OBSERVATION_BYTES + _NONCE_BYTES + _GCM_TAG_BYTES:
            raise ObservationPathError("Observation blob exceeds bounded storage size")
        chunks.append(chunk)
    data = b"".join(chunks)
    if expected_length is not None and len(data) != expected_length:
        raise ObservationPathError("Observation blob length changed")
    return data


def _blob_metadata(
    *,
    binding_id: str,
    relpath: str,
    key_id: str,
    plaintext: bytes,
    blob: bytes,
    info: os.stat_result,
) -> BlobMetadata:
    return BlobMetadata(
        binding_id = binding_id,
        blob_relpath = relpath,
        key_id = key_id,
        plaintext_length = len(plaintext),
        plaintext_digest = hashlib.sha256(plaintext).hexdigest(),
        ciphertext_length = len(blob),
        ciphertext_digest = hashlib.sha256(blob).hexdigest(),
        blob_dev = int(info.st_dev),
        blob_ino = int(info.st_ino),
    )


def _existing_blob_matches(
    *,
    blob_fd: int,
    name: str,
    account_id: str,
    binding_id: str,
    plaintext: bytes,
    key_id: str,
    key: bytes,
    relpath: str,
) -> Optional[BlobMetadata]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd = blob_fd)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ObservationPathError("Observation blob is not a private regular file")
        blob = _read_fd(fd)
    finally:
        os.close(fd)
    if len(blob) < _NONCE_BYTES + _GCM_TAG_BYTES:
        raise ObservationPathError("Observation blob is truncated")
    digest = hashlib.sha256(plaintext).hexdigest()
    try:
        decoded = AESGCM(key).decrypt(
            blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], _aad(account_id, key_id, digest, len(plaintext))
        )
    except InvalidTag as exc:
        raise ObservationBlobConflict("Existing observation blob does not authenticate") from exc
    if decoded != plaintext:
        raise ObservationBlobConflict("Existing observation blob has different content")
    return _blob_metadata(
        binding_id = binding_id,
        relpath = relpath,
        key_id = key_id,
        plaintext = plaintext,
        blob = blob,
        info = info,
    )


def publish_observation_blob(
    account_id: str,
    binding_id: str,
    plaintext: str | bytes,
    *,
    key_id: Optional[str] = None,
    master_key: Optional[bytes] = None,
    storage_root: Optional[Path | str] = None,
) -> BlobMetadata:
    """Encrypt and immutably publish a blob; never creates a binding."""
    _require_identity("account_id", account_id)
    binding_id = _require_binding_id(binding_id)
    _text, encoded = canonicalize_observation_text(plaintext)
    if len(encoded) > MAX_OBSERVATION_BYTES:
        raise ObservationError("Observation exceeds bounded storage size")
    master = master_key or get_or_create_observation_master_key()
    actual_key_id = (
        get_observation_master_key_id()
        if master_key is None
        else observation_master_key_id_for(master)
    )
    if key_id is not None and key_id != actual_key_id:
        raise ObservationError("Observation key id does not fingerprint supplied key")
    key_id = actual_key_id
    digest = hashlib.sha256(encoded).hexdigest()
    key = derive_observation_encryption_key(account_id, master_key = master)
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, encoded, _aad(account_id, key_id, digest, len(encoded)))
    blob = nonce + ciphertext
    relpath = f"blobs/{binding_id}.blob"
    name = f"{binding_id}.blob"
    root = observation_blob_root(account_id, storage_root = storage_root)

    with _blob_directory(root) as (root_fd, blob_fd):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd: Optional[int] = None
        created = False
        info: Optional[os.stat_result] = None
        try:
            try:
                fd = os.open(name, flags, 0o600, dir_fd = blob_fd)
                created = True
            except FileExistsError:
                existing = _existing_blob_matches(
                    blob_fd = blob_fd,
                    name = name,
                    account_id = account_id,
                    binding_id = binding_id,
                    plaintext = encoded,
                    key_id = key_id,
                    key = key,
                    relpath = relpath,
                )
                if existing is None:
                    raise ObservationBlobConflict("Observation blob appeared during publication")
                return existing
            os.fchmod(fd, 0o600)
            view = memoryview(blob)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Observation blob write made no progress")
                view = view[written:]
            os.fsync(fd)
            info = os.fstat(fd)
        except BaseException:
            if fd is not None:
                os.close(fd)
                fd = None
            if created:
                try:
                    current = os.stat(name, dir_fd = blob_fd, follow_symlinks = False)
                    if info is None or (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                        os.unlink(name, dir_fd = blob_fd)
                except OSError:
                    pass
            raise
        finally:
            if fd is not None:
                os.close(fd)
        _fsync_directory(blob_fd)
        _fsync_directory(root_fd)
    if info is None:  # pragma: no cover
        raise ObservationError("Observation publication produced no metadata")
    return _blob_metadata(
        binding_id = binding_id,
        relpath = relpath,
        key_id = key_id,
        plaintext = encoded,
        blob = blob,
        info = info,
    )


def build_observation_projection(
    handle: str,
    canonical_text: str | bytes,
    *,
    content_digest: Optional[str] = None,
) -> str:
    text, encoded = canonicalize_observation_text(canonical_text)
    digest = hashlib.sha256(encoded).hexdigest()
    if content_digest is not None and content_digest != digest:
        raise ValueError("Projection digest does not match canonical content")
    prefix = (
        "[ObservationPack v1\n"
        f"handle={handle}\n"
        f"bytes={len(encoded)} sha256={digest}\n"
        "exact canonical text retained; excerpt is incomplete context]\n\n"
    )
    suffix = "\n\nCall read_observation with this handle and a byte offset to inspect more."
    budget = MAX_PROJECTION_CHARS - len(prefix) - len(suffix)
    if budget <= 0:
        return (prefix + suffix)[:MAX_PROJECTION_CHARS]
    if len(text) <= budget:
        excerpt = text
    else:
        marker = "\n…\n"
        side = max(0, (budget - len(marker)) // 2)
        excerpt = text[:side] + marker + text[-(budget - len(marker) - side):]
    return (prefix + excerpt + suffix)[:MAX_PROJECTION_CHARS]


def _unavailable(
    fallback_result: str,
    *,
    account_id: str,
    thread_id: str,
    source_run_id: str,
    execution_id: str,
    tool_name: str,
    tool_call_id: str,
    approval_id: Optional[str],
    claim_token: Optional[str],
    worker_id: Optional[str],
    arguments_fingerprint: str,
    binding_id: Optional[str],
    reason: str,
    is_error: bool,
    quota_bytes: int,
) -> ObservationCandidate:
    return ObservationCandidate(
        state = "ineligible",
        eligible = False,
        account_id = account_id,
        thread_id = thread_id,
        source_run_id = source_run_id,
        execution_id = execution_id,
        tool_name = tool_name,
        tool_call_id = tool_call_id,
        approval_id = approval_id,
        claim_token = claim_token,
        worker_id = worker_id,
        arguments_fingerprint = arguments_fingerprint,
        binding_id = binding_id,
        handle = None,
        projection = None,
        fallback_result = fallback_result,
        is_error = bool(is_error),
        classification = "ordinary",
        reason = reason,
        format_version = OBSERVATION_FORMAT_VERSION,
        canonicalization_version = CANONICALIZATION_VERSION,
        projection_version = PROJECTION_VERSION,
        plaintext_length = None,
        plaintext_digest = None,
        ciphertext_length = None,
        key_id = None,
        blob_relpath = None,
        blob_metadata = None,
        quota_bytes = quota_bytes,
        created_at = _now(),
    )


def build_observation_candidate(
    *,
    account_id: str,
    thread_id: str,
    source_run_id: str,
    execution_id: str,
    tool_name: str,
    arguments_fingerprint: str,
    canonical_text: str | bytes,
    fallback_result: str,
    observation_pack_version: Optional[int] = None,
    tool_call_id: str = "",
    approval_id: Optional[str] = None,
    claim_token: Optional[str] = None,
    worker_id: Optional[str] = None,
    complete: bool = True,
    process_exit_code: Optional[int] = 0,
    policy_eligible: bool = True,
    secret_bearing: bool = False,
    is_error: bool = False,
    quota_bytes: int = DEFAULT_QUOTA_BYTES,
    master_key: Optional[bytes] = None,
) -> ObservationCandidate:
    """Construct a candidate only; no blob or account-DB publication occurs."""
    if not isinstance(fallback_result, str):
        fallback_result = str(fallback_result)
    _require_identity("account_id", account_id)
    _require_identity("thread_id", thread_id)
    if not isinstance(quota_bytes, int) or isinstance(quota_bytes, bool) or quota_bytes < 0:
        raise ValueError("quota_bytes must be a non-negative integer")
    binding_id = observation_binding_id(account_id, source_run_id, execution_id, arguments_fingerprint)
    common = dict(
        fallback_result = fallback_result,
        account_id = account_id,
        thread_id = thread_id,
        source_run_id = source_run_id,
        execution_id = execution_id,
        tool_name = tool_name,
        tool_call_id = tool_call_id,
        approval_id = approval_id,
        claim_token = claim_token,
        worker_id = worker_id,
        arguments_fingerprint = arguments_fingerprint,
        binding_id = binding_id,
        is_error = is_error,
        quota_bytes = quota_bytes,
    )
    if observation_pack_version != OBSERVATION_FORMAT_VERSION:
        return _unavailable(reason = "not_admitted", **common)
    if not complete:
        return _unavailable(reason = "incomplete", **common)
    if process_exit_code != 0:
        return _unavailable(reason = "nonzero_exit", **common)
    if not policy_eligible or secret_bearing:
        return _unavailable(reason = "policy_excluded", **common)
    try:
        _text, encoded = canonicalize_observation_text(canonical_text)
    except (TypeError, ValueError):
        return _unavailable(reason = "invalid_utf8", **common)
    if not MIN_OBSERVATION_BYTES <= len(encoded) <= MAX_OBSERVATION_BYTES:
        return _unavailable(reason = "size_excluded", **common)
    digest = hashlib.sha256(encoded).hexdigest()
    master = master_key or get_or_create_observation_master_key()
    key_id = observation_master_key_id_for(master)
    handle = make_observation_handle(account_id, thread_id, digest, master_key = master)
    projection = build_observation_projection(handle, _text, content_digest = digest)
    return ObservationCandidate(
        state = "staged",
        eligible = True,
        handle = handle,
        projection = projection,
        classification = "ordinary",
        reason = None,
        format_version = OBSERVATION_FORMAT_VERSION,
        canonicalization_version = CANONICALIZATION_VERSION,
        projection_version = PROJECTION_VERSION,
        plaintext_length = len(encoded),
        plaintext_digest = digest,
        ciphertext_length = len(encoded) + _NONCE_BYTES + _GCM_TAG_BYTES,
        key_id = key_id,
        blob_relpath = f"blobs/{binding_id}.blob",
        blob_metadata = None,
        created_at = _now(),
        canonical_bytes = encoded,
        **common,
    )


def publish_observation_candidate(
    candidate: ObservationCandidate,
    *,
    storage_root: Optional[Path | str] = None,
    master_key: Optional[bytes] = None,
) -> ObservationCandidate:
    """Publish immutable evidence and return a new ``state=published`` value."""
    if not isinstance(candidate, ObservationCandidate) or not candidate.eligible:
        return candidate
    if candidate.state == "published" and candidate.blob_metadata is not None:
        return candidate
    if candidate.state != "staged" or candidate.binding_id is None or candidate.canonical_bytes is None:
        raise ObservationError("Only a staged eligible candidate can be published")
    blob = publish_observation_blob(
        candidate.account_id,
        candidate.binding_id,
        candidate.canonical_bytes,
        key_id = candidate.key_id,
        master_key = master_key,
        storage_root = storage_root,
    )
    return replace(
        candidate,
        state = "published",
        blob_metadata = blob,
        ciphertext_length = blob.ciphertext_length,
        blob_relpath = blob.blob_relpath,
    )


# Disabled-compatible entrypoint: it now constructs a staged value only.
prepare_observation_pack = build_observation_candidate


def _failure_page(status: str, handle: Optional[str], reason: str) -> ObservationPage:
    safe_reason = reason if reason in {"missing", "corrupt", "missing_key", "unavailable", "invalid_handle"} else "unavailable"
    message = f"[ObservationPack {safe_reason}; evidence is unavailable. Do not infer completeness or rerun the source tool.]"
    encoded = message.encode("utf-8")
    return ObservationPage(
        status = safe_reason,
        text = message,
        handle = handle,
        start_offset = 0,
        end_offset = 0,
        next_offset = None,
        byte_length = len(encoded),
        byte_digest = None,
        reason = safe_reason,
    )


def _binding_from_mapping(value: ObservationBinding | Mapping[str, object]) -> ObservationBinding:
    if isinstance(value, ObservationBinding):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("recall requires a caller-authorized ObservationBinding")
    fields = ObservationBinding.__dataclass_fields__
    kwargs = {}
    for name, field in fields.items():
        if name in value:
            kwargs[name] = value[name]
        elif field.default is not field.default_factory:
            kwargs[name] = field.default
        else:
            raise ValueError(f"available binding is missing {name}")
    return ObservationBinding(**kwargs)


def _authenticated_plaintext(
    binding: ObservationBinding,
    *,
    storage_root: Optional[Path | str],
    master_key: bytes,
) -> bytes:
    relpath = _safe_relative_path(binding.blob_relpath)
    parts = PurePosixPath(relpath).parts
    if len(parts) != 2 or parts[0] != "blobs":
        raise ObservationPathError("Observation blob path is outside private blob directory")
    name = parts[1]
    if not name.endswith(".blob") or not _BINDING_ID_RE.fullmatch(name.removesuffix(".blob")):
        raise ObservationPathError("Observation blob filename is invalid")
    root = observation_blob_root(binding.account_id, storage_root = storage_root)
    with _blob_directory(root, create = False) as (_root_fd, blob_fd):
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd = blob_fd,
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ObservationPathError("Observation blob is not a private regular file")
            if (int(info.st_dev), int(info.st_ino)) != (binding.blob_dev, binding.blob_ino):
                raise ObservationPathError("Observation blob identity changed")
            blob = _read_fd(fd, binding.ciphertext_length)
        finally:
            os.close(fd)
    if hashlib.sha256(blob).hexdigest() != binding.ciphertext_digest:
        raise ObservationPathError("Observation ciphertext digest mismatch")
    if len(blob) < _NONCE_BYTES + _GCM_TAG_BYTES:
        raise ObservationPathError("Observation ciphertext is truncated")
    try:
        plaintext = AESGCM(derive_observation_encryption_key(binding.account_id, master_key = master_key)).decrypt(
            blob[:_NONCE_BYTES],
            blob[_NONCE_BYTES:],
            _aad(binding.account_id, binding.key_id, binding.plaintext_digest, binding.plaintext_length),
        )
    except InvalidTag as exc:
        raise ObservationPathError("Observation authentication tag mismatch") from exc
    if len(plaintext) != binding.plaintext_length or hashlib.sha256(plaintext).hexdigest() != binding.plaintext_digest:
        raise ObservationPathError("Observation plaintext identity mismatch")
    plaintext.decode("utf-8", errors = "strict")
    return plaintext


def _bounded_page_int(value: object, ceiling: int) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return min(value, ceiling)


def recall_observation(
    binding: ObservationBinding | Mapping[str, object],
    *,
    offset: int = 0,
    max_bytes: int = MAX_RECALL_BYTES,
    max_lines: int = MAX_RECALL_LINES,
    active_result_budget_bytes: Optional[int] = None,
    storage_root: Optional[Path | str] = None,
    handle: Optional[str] = None,
) -> ObservationPage:
    """Page evidence from a caller-supplied already-authorized binding.

    There is deliberately no account DB lookup and no boolean-shaped
    ``run_authorized``/``current_run`` argument. Authorization belongs to the
    backend adapter that supplies this value object.
    """
    requested_handle = handle
    try:
        item = _binding_from_mapping(binding)
        _require_identity("account_id", item.account_id)
        _require_identity("thread_id", item.thread_id)
        _require_binding_id(item.binding_id)
        if item.state != "available":
            state = item.state if item.state in {"missing", "corrupt", "missing_key", "unavailable"} else "unavailable"
            return _failure_page(state, requested_handle, state)
        master = get_observation_master_key(refresh = True)
        if master is None:
            return _failure_page("missing_key", requested_handle, "missing_key")
        requested_handle = handle or make_observation_handle(
            item.account_id, item.thread_id, item.plaintext_digest, master_key = master
        )
        _token, digest = parse_observation_handle(requested_handle)
        if digest != item.plaintext_digest:
            return _failure_page("invalid_handle", requested_handle, "invalid_handle")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return _failure_page("invalid_handle", requested_handle, "invalid_handle")
        bounded_bytes = _bounded_page_int(max_bytes, MAX_RECALL_BYTES)
        bounded_lines = _bounded_page_int(max_lines, MAX_RECALL_LINES)
        if bounded_bytes is None or bounded_lines is None:
            return _failure_page("unavailable", requested_handle, "unavailable")
        if active_result_budget_bytes is not None:
            budget = _bounded_page_int(active_result_budget_bytes, MAX_RECALL_BYTES)
            if budget is None or budget < MIN_ACTIVE_RESULT_BUDGET_BYTES:
                return _failure_page("unavailable", requested_handle, "unavailable")
            bounded_bytes = min(bounded_bytes, budget - 64)
            if bounded_bytes <= 0:
                return _failure_page("unavailable", requested_handle, "unavailable")
        if item.key_id != observation_master_key_id_for(master):
            return _failure_page("missing_key", requested_handle, "missing_key")
        expected = make_observation_handle(
            item.account_id, item.thread_id, item.plaintext_digest, master_key = master
        )
        if not hmac.compare_digest(expected, requested_handle):
            return _failure_page("invalid_handle", requested_handle, "invalid_handle")
        if item.handle and not hmac.compare_digest(expected, item.handle):
            return _failure_page("invalid_handle", requested_handle, "invalid_handle")
        plaintext = _authenticated_plaintext(item, storage_root = storage_root, master_key = master)
        if offset > len(plaintext):
            return _failure_page("corrupt", requested_handle, "corrupt")
        try:
            plaintext[:offset].decode("utf-8", errors = "strict")
        except UnicodeDecodeError:
            return _failure_page("invalid_handle", requested_handle, "invalid_handle")
        end = min(len(plaintext), offset + bounded_bytes)
        while end > offset:
            try:
                text = plaintext[offset:end].decode("utf-8", errors = "strict")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            text = ""
        if text.count("\n") >= bounded_lines:
            cut = [index for index, char in enumerate(text) if char == "\n"][bounded_lines - 1] + 1
            text = text[:cut]
            end = offset + len(text.encode("utf-8"))
        page_bytes = text.encode("utf-8")
        if end < len(plaintext) and not page_bytes:
            return _failure_page("corrupt", requested_handle, "corrupt")
        return ObservationPage(
            status = "available",
            text = text,
            handle = requested_handle,
            start_offset = offset,
            end_offset = end,
            next_offset = end if end < len(plaintext) else None,
            byte_length = len(page_bytes),
            byte_digest = hashlib.sha256(page_bytes).hexdigest(),
        )
    except (TypeError, ValueError, KeyError):
        return _failure_page("invalid_handle", requested_handle, "invalid_handle")
    except FileNotFoundError:
        return _failure_page("missing", requested_handle, "missing")
    except (OSError, ObservationPathError, UnicodeDecodeError):
        return _failure_page("corrupt", requested_handle, "corrupt")


def read_observation(*args, **kwargs) -> ObservationPage:
    return recall_observation(*args, **kwargs)


def observation_pack_is_enabled() -> bool:
    return OBSERVATION_PACK_ENABLED


__all__ = [
    "OBSERVATION_PACK_ENABLED",
    "OBSERVATION_FORMAT_VERSION",
    "CANONICALIZATION_VERSION",
    "PROJECTION_VERSION",
    "MIN_OBSERVATION_BYTES",
    "MAX_OBSERVATION_BYTES",
    "MAX_PROJECTION_CHARS",
    "MAX_RECALL_BYTES",
    "MAX_RECALL_LINES",
    "MIN_ACTIVE_RESULT_BUDGET_BYTES",
    "ObservationError",
    "ObservationPathError",
    "ObservationBlobConflict",
    "BlobMetadata",
    "ObservationCandidate",
    "ObservationPreparation",
    "ObservationBinding",
    "ObservationPage",
    "derive_observation_encryption_key",
    "derive_observation_handle_key",
    "observation_binding_id",
    "make_observation_handle",
    "parse_observation_handle",
    "canonicalize_observation_text",
    "canonical_observation_digest",
    "observation_blob_root",
    "publish_observation_blob",
    "build_observation_projection",
    "build_observation_candidate",
    "publish_observation_candidate",
    "prepare_observation_pack",
    "recall_observation",
    "read_observation",
    "observation_pack_is_enabled",
]
