# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local Codex/Claude Code skill management.

Skill files are plain text and are never executed by this route. The manager only
reads the two user-owned skill roots and accepts GitHub HTTPS repositories (or an
explicit local directory) as installation sources.
"""

import re
import ctypes
import errno
import os
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from auth.authentication import get_current_subject
from core.inference.skills import (
    SkillError,
    SkillNotFoundError,
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_MD_BYTES,
    list_skills,
    _read_skill_document,
    set_skill_enabled,
)
from utils.account_context import is_owner_context
from utils.paths import workspace_root

router = APIRouter()

SkillTarget = Literal["codex", "claude", "both"]
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_GITHUB_SOURCE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$")
_MAX_LOCAL_PACKAGE_FILES = 2_048
_MAX_LOCAL_PACKAGE_DIRECTORIES = 2_048
_MAX_LOCAL_PACKAGE_DEPTH = 32
_MAX_LOCAL_PACKAGE_FILE_BYTES = MAX_SKILL_FILE_BYTES
_MAX_LOCAL_PACKAGE_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_LOCAL_INSTALL_SKILLS = 256
_MAX_LOCAL_INSTALL_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_LOCAL_DISCOVERY_ENTRIES = 1_000
_MAX_LOCAL_DISCOVERY_MARKER_BYTES = 4 * 1024 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_DIR_FD_READS = bool(_O_DIRECTORY and os.open in getattr(os, "supports_dir_fd", set()))
_DIR_FD_MUTATIONS = bool(
    _O_DIRECTORY
    and _O_NOFOLLOW
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.mkdir in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_dir_fd", set())
    and os.rename in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
    and os.rmdir in getattr(os, "supports_dir_fd", set())
)


@dataclass(frozen=True)
class _LocalSkill:
    path: Path
    identity: os.stat_result
    marker: bytes


@dataclass(frozen=True)
class _Package:
    marker: bytes
    files: tuple[tuple[tuple[str, ...], bytes, int], ...]


class CreateSkillRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 64)
    description: str = Field(min_length = 1, max_length = 500)
    instructions: str = Field(min_length = 1, max_length = 100_000)
    target: SkillTarget = "codex"


class InstallSkillRequest(BaseModel):
    source: str = Field(min_length = 1, max_length = 2_000)
    target: SkillTarget = "codex"


def _roots() -> dict[str, Path]:
    if not is_owner_context():
        # The local manager historically accepted two target labels, but a
        # managed account has one private Agent Skills root.  Keep accepting the
        # labels for API compatibility while making both resolve to this account.
        root = workspace_root() / "skills"
        return {"codex": root, "claude": root}
    home = Path.home()
    return {"codex": home / ".codex" / "skills", "claude": home / ".claude" / "skills"}


def _valid_name(name: str) -> str:
    normalized = name.strip().lower()
    if not _SKILL_NAME.fullmatch(normalized):
        raise HTTPException(
            status_code = 400,
            detail = "Skill names may use lowercase letters, numbers, dots, underscores, and hyphens.",
        )
    return normalized


def _unsafe_mutation() -> HTTPException:
    return HTTPException(status_code = 400, detail = "Skill filesystem operations are unavailable safely.")


def _require_secure_mutations() -> None:
    """Mutations fail closed unless every operation can stay descriptor-relative."""
    if not _DIR_FD_MUTATIONS:
        raise _unsafe_mutation()


def _is_link_status(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(reparse and attributes & reparse)


def _validate_absolute_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    """Reject links in every existing absolute ancestor without resolving them."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise OSError(errno.EINVAL, "path must be absolute")
    current = Path(candidate.anchor or os.sep)
    parts = candidate.parts[1:]
    for index, part in enumerate(parts):
        if part in {".", ".."}:
            raise OSError(errno.EINVAL, "unsafe path component")
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf:
                return
            raise
        if _is_link_status(status):
            raise OSError(errno.ELOOP, "linked ancestor")
        if index < len(parts) - 1 and not stat.S_ISDIR(status.st_mode):
            raise OSError(errno.ENOTDIR, "non-directory ancestor")


def _open_absolute_directory(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory by no-following every component."""
    candidate = Path(path)
    if not candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        raise OSError(errno.EINVAL, "unsafe directory path")
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        for part in candidate.parts[1:]:
            try:
                opened = os.open(part, flags, dir_fd = descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd = descriptor)
                opened = os.open(part, flags, dir_fd = descriptor)
            os.close(descriptor)
            descriptor = opened
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_relative_directory(parent: int, parts: tuple[str, ...], *, create: bool = False) -> int:
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    descriptor = os.dup(parent)
    try:
        for part in parts:
            if not part or part in {".", ".."} or "/" in part or "\\" in part:
                raise OSError(errno.EINVAL, "unsafe directory component")
            try:
                opened = os.open(part, flags, dir_fd = descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd = descriptor)
                opened = os.open(part, flags, dir_fd = descriptor)
            os.close(descriptor)
            descriptor = opened
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _target_root_parts(target_root: Path) -> tuple[Path, tuple[str, ...]]:
    base = Path(workspace_root() if not is_owner_context() else Path.home())
    base = Path(os.path.abspath(base))
    target = Path(os.path.abspath(target_root))
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise _unsafe_mutation() from exc
    parts = tuple(relative.parts)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise _unsafe_mutation()
    return base, parts


def _open_target_root(target_root: Path, *, create: bool) -> tuple[int, os.stat_result]:
    base, parts = _target_root_parts(target_root)
    if create:
        _require_secure_mutations()
    base_fd = _open_absolute_directory(base, create = create)
    try:
        root_fd = _open_relative_directory(base_fd, parts, create = create)
    finally:
        os.close(base_fd)
    return root_fd, os.fstat(root_fd)


def _read_fd_limited(parent: int, name: str, limit: int) -> bytes:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise OSError(errno.EINVAL, "unsafe file component")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | _O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd = parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        if before.st_nlink != 1:
            raise OSError(errno.ELOOP, "linked marker")
        if before.st_size > limit:
            raise OSError(errno.EFBIG, "file is too large")
        raw = b""
        while len(raw) <= limit:
            chunk = os.read(descriptor, min(64 * 1024, limit + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
        if (
            not os.path.samestat(before, after)
            or after.st_nlink != 1
            or len(raw) > limit
        ):
            raise OSError(errno.EAGAIN, "file changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _marker_from_directory(descriptor: int, *, strict: bool) -> bytes | None:
    try:
        return _read_fd_limited(descriptor, "SKILL.md", MAX_SKILL_MD_BYTES)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if strict:
            raise HTTPException(status_code = 400, detail = "Skill source contains an unsafe SKILL.md file.") from exc
        return None


def _safe_skill_entries_fallback(
    root: Path,
    *,
    target_root: bool,
    expected_identity: os.stat_result | None,
) -> list[_LocalSkill]:
    """Identity-revalidate read fallback for platforms without dir_fd support.

    Mutations never use this path.  It is intentionally conservative: any
    component replacement or link is treated as an unavailable/unsafe root.
    """
    raw_root = Path(root)
    if not raw_root.is_absolute() or any(part in {".", ".."} for part in raw_root.parts[1:]):
        if target_root:
            return []
        raise HTTPException(status_code = 400, detail = "Skill source is unsafe.")
    root = Path(os.path.abspath(raw_root))
    strict_markers = not target_root
    discovered_count = 0
    marker_bytes = 0
    visited_count = 0
    try:
        _validate_absolute_components(root, allow_missing_leaf = True)
        root_status = os.lstat(root)
        if _is_link_status(root_status) or not stat.S_ISDIR(root_status.st_mode):
            return []
        if expected_identity is not None and not os.path.samestat(root_status, expected_identity):
            raise OSError(errno.EAGAIN, "root changed")

        def marker(path: Path, expected_identity: os.stat_result) -> bytes | None:
            try:
                _validate_absolute_components(path)
            except OSError:
                if strict_markers:
                    raise
                return None
            if not (
                _O_DIRECTORY
                and _O_NOFOLLOW
                and os.open in getattr(os, "supports_dir_fd", set())
            ):
                if strict_markers:
                    raise OSError(errno.ENOTSUP, "safe fallback descriptors unavailable")
                return None
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                )
            except FileNotFoundError:
                return None
            except OSError:
                if strict_markers:
                    raise
                return None
            try:
                opened_status = os.fstat(descriptor)
                if (
                    _is_link_status(opened_status)
                    or not stat.S_ISDIR(opened_status.st_mode)
                    or not os.path.samestat(expected_identity, opened_status)
                ):
                    raise OSError(errno.EAGAIN, "skill directory changed before reading")
                try:
                    raw = _read_fd_limited(descriptor, "SKILL.md", MAX_SKILL_MD_BYTES)
                except FileNotFoundError:
                    return None
                after_directory = os.fstat(descriptor)
                if not os.path.samestat(opened_status, after_directory):
                    raise OSError(errno.EAGAIN, "skill directory changed while being read")
            except OSError:
                if strict_markers:
                    raise
                return None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            return raw

        candidates: list[_LocalSkill] = []

        def add_candidate(item: _LocalSkill) -> None:
            nonlocal discovered_count, marker_bytes
            discovered_count += 1
            marker_bytes += len(item.marker)
            if discovered_count > _MAX_LOCAL_DISCOVERY_ENTRIES:
                raise OSError(errno.E2BIG, "too many skill entries")
            if marker_bytes > _MAX_LOCAL_DISCOVERY_MARKER_BYTES:
                raise OSError(errno.EFBIG, "too much marker data")
            candidates.append(item)

        raw = marker(root, root_status)
        if raw is not None:
            add_candidate(_LocalSkill(root, root_status, raw))
        parents = [root]
        nested = root / "skills"
        try:
            nested_status = os.lstat(nested)
            if stat.S_ISDIR(nested_status.st_mode) and not _is_link_status(nested_status):
                parents.append(nested)
        except FileNotFoundError:
            pass
        for parent in parents:
            with os.scandir(parent) as children:
                for entry in children:
                    if entry.name.startswith("."):
                        continue
                    visited_count += 1
                    if visited_count > _MAX_LOCAL_DISCOVERY_ENTRIES:
                        raise OSError(errno.E2BIG, "too many entries")
                    child = parent / entry.name
                    try:
                        child_status = os.lstat(child)
                    except OSError:
                        continue
                    if _is_link_status(child_status) or not stat.S_ISDIR(child_status.st_mode):
                        continue
                    raw = marker(child, child_status)
                    if raw is not None:
                        add_candidate(_LocalSkill(child, child_status, raw))
        by_path: dict[Path, _LocalSkill] = {}
        for item in candidates:
            by_path.setdefault(item.path, item)
        return list(by_path.values())
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as exc:
        if target_root:
            return []
        raise HTTPException(status_code = 400, detail = "Skill source is unsafe.") from exc


def _safe_skill_entries(
    root: Path,
    *,
    target_root: bool = False,
    expected_identity: os.stat_result | None = None,
) -> list[_LocalSkill]:
    """Discover markers from descriptors retained for the whole read."""
    if not _DIR_FD_READS:
        return _safe_skill_entries_fallback(
            root, target_root = target_root, expected_identity = expected_identity
        )
    try:
        descriptor = _open_absolute_directory(Path(os.path.abspath(root)))
    except FileNotFoundError:
        return []
    except (OSError, HTTPException):
        if target_root:
            return []
        raise HTTPException(status_code = 400, detail = "Skill source is unsafe.")
    try:
        root_identity = os.fstat(descriptor)
        if expected_identity is not None and not os.path.samestat(root_identity, expected_identity):
            raise HTTPException(status_code = 400, detail = "Skill source changed while being read.")
        candidates: list[_LocalSkill] = []
        discovered_count = 0
        marker_bytes = 0

        def add_candidate(item: _LocalSkill) -> None:
            nonlocal discovered_count, marker_bytes
            discovered_count += 1
            marker_bytes += len(item.marker)
            if discovered_count > _MAX_LOCAL_DISCOVERY_ENTRIES:
                raise HTTPException(status_code = 400, detail = "Skill directory has too many entries.")
            if marker_bytes > _MAX_LOCAL_DISCOVERY_MARKER_BYTES:
                raise HTTPException(status_code = 400, detail = "Skill marker data is too large.")
            candidates.append(item)

        strict_markers = not target_root
        root_marker = _marker_from_directory(descriptor, strict = strict_markers)
        if root_marker is not None:
            add_candidate(_LocalSkill(Path(os.path.abspath(root)), root_identity, root_marker))

        visible_count = 0

        def scan_children(parent_fd: int, parent_path: Path) -> None:
            nonlocal visible_count
            with os.scandir(parent_fd) as children:
                for entry in children:
                    name = str(entry.name)
                    if name.startswith("."):
                        continue
                    visible_count += 1
                    if visible_count > _MAX_LOCAL_DISCOVERY_ENTRIES:
                        raise HTTPException(status_code = 400, detail = "Skill directory has too many entries.")
                    try:
                        child_fd = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = parent_fd)
                    except (FileNotFoundError, NotADirectoryError, PermissionError):
                        continue
                    except OSError as exc:
                        if exc.errno == errno.ELOOP:
                            continue
                        raise
                    try:
                        child_identity = os.fstat(child_fd)
                        marker = _marker_from_directory(child_fd, strict = strict_markers)
                        if marker is not None:
                            add_candidate(_LocalSkill(parent_path / name, child_identity, marker))
                    finally:
                        os.close(child_fd)

        scan_children(descriptor, Path(os.path.abspath(root)))
        try:
            nested_fd = os.open("skills", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = descriptor)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            nested_fd = None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                nested_fd = None
            else:
                raise
        if nested_fd is not None:
            try:
                scan_children(nested_fd, Path(os.path.abspath(root)) / "skills")
            finally:
                os.close(nested_fd)
        by_path: dict[Path, _LocalSkill] = {}
        for item in candidates:
            by_path.setdefault(item.path, item)
        return list(by_path.values())
    except HTTPException:
        raise
    except (OSError, UnicodeError) as exc:
        if target_root:
            return []
        raise HTTPException(status_code = 400, detail = "Skill source is unsafe.") from exc
    finally:
        os.close(descriptor)


def _description_from_marker(raw: bytes) -> str:
    try:
        text = raw[:4_000].decode("utf-8")
    except UnicodeError:
        return "Local skill"
    match = re.search(r"(?im)^description:\s*(.+?)\s*$", text)
    return (match.group(1).strip().strip('"\'') if match else "Local skill")[:500]


def _description(path: Path) -> str:
    try:
        entries = _safe_skill_entries(path.parent)
        entry = next(item for item in entries if item.path == path)
        return _description_from_marker(entry.marker)
    except (HTTPException, OSError, StopIteration):
        return "Local skill"


def _skill_marker(path: Path) -> Path:
    try:
        entry = next(item for item in _safe_skill_entries(path.parent) if item.path == path)
    except (HTTPException, OSError, StopIteration) as exc:
        raise HTTPException(status_code = 404, detail = "Local skill not found.") from exc
    if not entry.marker:
        raise HTTPException(status_code = 404, detail = "Local skill not found.")
    return path / "SKILL.md"


def _skill_dirs(root: Path, *, target_root: bool = False) -> list[Path]:
    return [item.path for item in _safe_skill_entries(root, target_root = target_root)]


def _list_skills() -> list[dict[str, object]]:
    roots = _roots()
    by_name: dict[str, dict[str, object]] = {}
    discovered: dict[Path, list[_LocalSkill]] = {}
    for ecosystem, root in roots.items():
        try:
            root_key = Path(os.path.abspath(root))
        except OSError:
            continue
        paths = discovered.setdefault(root_key, _safe_skill_entries(root, target_root = True))
        for entry in paths:
            path = entry.path
            name = path.name
            display_path = str(path)
            if not is_owner_context():
                try:
                    display_path = str(path.relative_to(workspace_root()))
                except ValueError:
                    display_path = f"skills/{name}"
            row = by_name.setdefault(
                name,
                {"name": name, "description": _description_from_marker(entry.marker), "path": display_path, "ecosystems": []},
            )
            ecosystems = row["ecosystems"]
            if ecosystem not in ecosystems:
                ecosystems.append(ecosystem)
    return sorted(by_name.values(), key = lambda row: str(row["name"]))


def _read_owner_local_skill(name: str) -> tuple[dict[str, str], str] | None:
    """Read a legacy owner-manager skill from its already pinned marker.

    The core Agent Skills resolver intentionally knows about ``~/.agents``;
    the local manager also owns ``~/.codex``.  Use the same bounded descriptor
    discovery as list/install so this compatibility path cannot widen that
    ownership or follow a swapped marker.
    """
    if not is_owner_context():
        return None
    seen: set[Path] = set()
    for root in _roots().values():
        key = Path(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        for entry in _safe_skill_entries(root, target_root = True):
            if entry.path.name != name:
                continue
            try:
                content = entry.marker.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code = 400, detail = "Could not read local skill.") from exc
            return {"description": _description_from_marker(entry.marker)}, content
    return None


def _targets(target: SkillTarget) -> list[tuple[str, Path]]:
    roots = _roots()
    selected: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for name in ("codex", "claude"):
        if target != "both" and target != name:
            continue
        root = roots[name]
        try:
            key = root.absolute()
        except OSError:
            key = root
        if key in seen:
            continue
        seen.add(key)
        selected.append((name, root))
    return selected


def _validate_target_root(target_root: Path) -> None:
    try:
        descriptor, _ = _open_target_root(target_root, create = False)
    except FileNotFoundError:
        return
    except (OSError, HTTPException) as error:
        raise HTTPException(status_code = 400, detail = "Skill destination is unsafe.") from error
    else:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short skill write")
        offset += written


def _open_or_create_child_directory(parent: int, name: str) -> int:
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd = parent)
    except FileNotFoundError:
        os.mkdir(name, 0o700, dir_fd = parent)
        return os.open(name, flags, dir_fd = parent)


def _write_package_at(stage: int, package: _Package) -> None:
    for parts, data, mode in package.files:
        if not parts or any(not component or component in {".", ".."} for component in parts):
            raise HTTPException(status_code = 400, detail = "Skill package contains an unsafe path.")
        parent = stage
        opened: list[int] = []
        try:
            for component in parts[:-1]:
                child = _open_or_create_child_directory(parent, component)
                opened.append(child)
                parent = child
            filename = parts[-1]
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | getattr(os, "O_BINARY", 0)
            descriptor = os.open(filename, flags, 0o600, dir_fd = parent)
            try:
                _write_all(descriptor, data)
                os.fchmod(descriptor, mode & 0o777)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)
    os.fsync(stage)


def _remove_tree_at(parent: int, name: str, expected: os.stat_result | None = None) -> bool:
    try:
        child = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = parent)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.ELOOP:
            return False
        if isinstance(exc, FileNotFoundError):
            return False
        # A non-directory file is safe to remove only when the caller explicitly
        # owns the exact identity.  Publications are always directories.
        return False
    try:
        if expected is not None and not os.path.samestat(os.fstat(child), expected):
            return False
        for item in os.listdir(child):
            item = str(item)
            try:
                nested = os.open(item, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = child)
            except (FileNotFoundError, NotADirectoryError):
                try:
                    os.unlink(item, dir_fd = child)
                except FileNotFoundError:
                    pass
                continue
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    return False
                raise
            try:
                if not _remove_tree_descriptor(nested):
                    return False
            finally:
                try:
                    os.close(nested)
                except OSError:
                    pass
            os.rmdir(item, dir_fd = child)
        os.rmdir(name, dir_fd = parent)
        return True
    finally:
        os.close(child)


def _remove_tree_descriptor(descriptor: int) -> bool:
    for item in os.listdir(descriptor):
        item = str(item)
        try:
            nested = os.open(item, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = descriptor)
        except (FileNotFoundError, NotADirectoryError):
            try:
                os.unlink(item, dir_fd = descriptor)
            except FileNotFoundError:
                pass
            continue
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                return False
            raise
        try:
            if not _remove_tree_descriptor(nested):
                return False
        finally:
            os.close(nested)
        os.rmdir(item, dir_fd = descriptor)
    return True


def _rename_noreplace_raw(parent: int, source: str, destination: str) -> None:
    """Perform the platform no-replace rename without path validation."""
    if not _DIR_FD_MUTATIONS:
        raise _unsafe_mutation()
    libc = ctypes.CDLL(None, use_errno = True)
    if sys_platform := getattr(os, "uname", None):
        platform_name = str(sys_platform().sysname).lower()
    else:
        platform_name = ""
    if platform_name == "linux":
        function = getattr(libc, "renameat2", None)
        flags = 1  # RENAME_NOREPLACE
    elif platform_name == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flags = 4  # RENAME_EXCL
    else:
        function = None
        flags = 0
    if function is None:
        raise _unsafe_mutation()
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    result = function(parent, source.encode("utf-8"), parent, destination.encode("utf-8"), flags)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _rename_noreplace(parent: int, source: str, destination: str) -> None:
    """Publish a private stage without replacing a concurrent destination."""
    # Keep the final basename check adjacent to the syscall.  This closes the
    # deterministic swap-at-rename case; the caller also verifies the exact
    # inode before and after this operation.
    descriptor = os.open(source, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = parent)
    try:
        _rename_noreplace_raw(parent, source, destination)
    finally:
        os.close(descriptor)


def _preserve_rejected_destination(parent: int, name: str, preferred: str) -> None:
    """Move a failed publication aside without unlinking its replacement."""
    candidates = [preferred]
    for _ in range(3):
        candidates.append(f".skill-rejected-{uuid.uuid4().hex}")
    for candidate in candidates:
        try:
            _rename_noreplace_raw(parent, name, candidate)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                continue
            return


def _verify_directory_name(parent: int, name: str, expected: os.stat_result) -> None:
    """Confirm that a basename still names the descriptor we staged.

    The descriptor keeps the written directory alive, but the subsequent
    rename API necessarily receives a basename.  Re-open that basename with
    no-follow semantics immediately before and after publication so a swapped
    stage can never be mistaken for the package we wrote.
    """
    descriptor = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = parent)
    try:
        observed = os.fstat(descriptor)
        if not os.path.samestat(observed, expected):
            raise OSError(errno.EAGAIN, "staged directory changed")
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _Publication:
    root: Path
    root_identity: os.stat_result
    name: str
    identity: os.stat_result


def _destination_available(target_root: Path, name: str) -> os.stat_result | None:
    try:
        root_fd, root_identity = _open_target_root(target_root, create = False)
    except FileNotFoundError:
        return None
    except (OSError, HTTPException) as exc:
        raise HTTPException(status_code = 400, detail = "Skill destination is unsafe.") from exc
    try:
        try:
            child = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = root_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise HTTPException(status_code = 409, detail = f"Skill destination is a symlink: {name}") from exc
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}") from exc
        else:
            os.close(child)
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")
    finally:
        os.close(root_fd)
    return root_identity


def _publish_package(
    package: _Package,
    target_root: Path,
    name: str,
    *,
    expected_root_identity: os.stat_result | None = None,
) -> _Publication:
    _require_secure_mutations()
    try:
        root_fd, root_identity = _open_target_root(target_root, create = True)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code = 400, detail = "Skill destination is unsafe.") from exc
    stage_name = f".skill-stage-{uuid.uuid4().hex}"
    stage_fd: int | None = None
    staged_identity: os.stat_result | None = None
    published = False
    try:
        if expected_root_identity is not None and not os.path.samestat(
            root_identity, expected_root_identity
        ):
            raise HTTPException(status_code = 400, detail = "Skill destination changed while being written.")
        try:
            child = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = root_fd)
        except FileNotFoundError:
            child = None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise HTTPException(status_code = 409, detail = f"Skill destination is a symlink: {name}") from exc
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}") from exc
        if child is not None:
            os.close(child)
            raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}")
        os.mkdir(stage_name, 0o700, dir_fd = root_fd)
        stage_fd = os.open(stage_name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd = root_fd)
        opened_identity = os.fstat(stage_fd)
        # A directory returned by mkdir must be empty when first pinned.  If a
        # basename was replaced before open, do not adopt or clean an existing
        # directory (which may contain another owner's files).
        if os.listdir(stage_fd):
            raise OSError(errno.EAGAIN, "staged directory was replaced before opening")
        staged_identity = opened_identity
        _write_package_at(stage_fd, package)
        _verify_directory_name(root_fd, stage_name, staged_identity)
        try:
            _rename_noreplace(root_fd, stage_name, name)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise HTTPException(status_code = 409, detail = f"Skill destination already exists: {name}") from exc
            raise HTTPException(status_code = 400, detail = "Skill destination publication failed.") from exc
        try:
            _verify_directory_name(root_fd, name, staged_identity)
        except OSError as exc:
            # Preserve a mismatched destination under a private basename rather
            # than unlinking an attacker-owned replacement.  If the stage was
            # swapped immediately before rename, this also leaves ``name``
            # unpublished in the final namespace.
            # The destination is ambiguous even for EACCES/EIO: never let the
            # identity-checked stage cleanup remove content we cannot inspect.
            staged_identity = None
            _preserve_rejected_destination(root_fd, name, stage_name)
            raise HTTPException(status_code = 400, detail = "Skill destination publication failed.") from exc
        published = True
        return _Publication(target_root, root_identity, name, staged_identity)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code = 400, detail = "Could not publish skill package safely.") from exc
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        if not published and staged_identity is not None:
            try:
                _remove_tree_at(root_fd, stage_name, staged_identity)
            except OSError:
                pass
        os.close(root_fd)


def _rollback_publications(publications: list[_Publication]) -> None:
    for publication in reversed(publications):
        try:
            root_fd = _open_absolute_directory(publication.root)
            try:
                if not os.path.samestat(os.fstat(root_fd), publication.root_identity):
                    continue
                _remove_tree_at(root_fd, publication.name, publication.identity)
            finally:
                os.close(root_fd)
        except OSError:
            continue


def _read_package(source: Path, expected_identity: os.stat_result, expected_marker: bytes) -> _Package:
    try:
        descriptor = _open_absolute_directory(source)
    except OSError as exc:
        raise HTTPException(status_code = 400, detail = "Skill source is unsafe.") from exc
    try:
        if not os.path.samestat(os.fstat(descriptor), expected_identity):
            raise HTTPException(status_code = 400, detail = "Skill source changed while being read.")
        files: list[tuple[tuple[str, ...], bytes, int]] = []
        total_bytes = 0
        directory_count = 0
        file_count = 0
        entry_count = 0

        def walk(parent: int, prefix: tuple[str, ...], depth: int) -> None:
            nonlocal total_bytes, directory_count, file_count, entry_count
            if depth > _MAX_LOCAL_PACKAGE_DEPTH:
                raise HTTPException(status_code = 400, detail = "Skill package is too deeply nested.")
            with os.scandir(parent) as children:
                for entry in children:
                    entry_count += 1
                    if entry_count > _MAX_LOCAL_PACKAGE_FILES + _MAX_LOCAL_PACKAGE_DIRECTORIES:
                        raise HTTPException(status_code = 400, detail = "Skill package has too many entries.")
                    name = str(entry.name)
                    if not name or name in {".", ".."} or "/" in name or "\\" in name:
                        raise HTTPException(status_code = 400, detail = "Skill package contains an unsafe path.")
                    try:
                        child = os.open(
                            name,
                            os.O_RDONLY | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                            dir_fd = parent,
                        )
                    except OSError as exc:
                        if exc.errno == errno.ELOOP:
                            raise HTTPException(status_code = 400, detail = "Skill package contains symlinks.") from exc
                        raise HTTPException(status_code = 400, detail = "Skill package could not be read safely.") from exc
                    try:
                        before = os.fstat(child)
                        path = prefix + (name,)
                        if stat.S_ISDIR(before.st_mode):
                            directory_count += 1
                            if directory_count > _MAX_LOCAL_PACKAGE_DIRECTORIES:
                                raise HTTPException(status_code = 400, detail = "Skill package has too many directories.")
                            walk(child, path, depth + 1)
                        elif stat.S_ISREG(before.st_mode):
                            if before.st_nlink != 1:
                                raise HTTPException(status_code = 400, detail = "Skill package contains hard-linked files.")
                            file_count += 1
                            if file_count > _MAX_LOCAL_PACKAGE_FILES:
                                raise HTTPException(status_code = 400, detail = "Skill package has too many files.")
                            if before.st_size > _MAX_LOCAL_PACKAGE_FILE_BYTES:
                                raise HTTPException(status_code = 400, detail = "Skill package file is too large.")
                            raw = b""
                            while len(raw) <= _MAX_LOCAL_PACKAGE_FILE_BYTES:
                                chunk = os.read(child, min(64 * 1024, _MAX_LOCAL_PACKAGE_FILE_BYTES + 1 - len(raw)))
                                if not chunk:
                                    break
                                raw += chunk
                                total_bytes += len(chunk)
                                if total_bytes > _MAX_LOCAL_PACKAGE_TOTAL_BYTES:
                                    raise HTTPException(status_code = 400, detail = "Skill package is too large.")
                            after = os.fstat(child)
                            if (
                                not os.path.samestat(before, after)
                                or after.st_nlink != 1
                                or len(raw) > _MAX_LOCAL_PACKAGE_FILE_BYTES
                            ):
                                raise HTTPException(status_code = 400, detail = "Skill package changed while being read.")
                            safe_mode = stat.S_IMODE(before.st_mode) & 0o777
                            files.append((path, raw, safe_mode))
                        else:
                            raise HTTPException(status_code = 400, detail = "Skill package contains a special file.")
                    finally:
                        os.close(child)

        walk(descriptor, (), 0)
    finally:
        os.close(descriptor)
    marker = next((data for parts, data, _ in files if parts == ("SKILL.md",)), None)
    if marker is None or marker != expected_marker:
        raise HTTPException(status_code = 400, detail = "Skill source changed while being read.")
    return _Package(marker, tuple(files))


def _copy_skill(source: Path, target_root: Path, name: str) -> _Publication:
    entries = _safe_skill_entries(source.parent)
    entry = next((item for item in entries if item.path == source), None)
    if entry is None:
        raise HTTPException(status_code = 400, detail = "Skill source does not contain a SKILL.md file.")
    package = _read_package(entry.path, entry.identity, entry.marker)
    return _publish_package(package, target_root, name)


def _source_directory(source: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    parsed = urlparse(source)
    if parsed.scheme or parsed.netloc:
        if not _GITHUB_SOURCE.fullmatch(source.strip()):
            raise HTTPException(status_code = 400, detail = "Only a direct GitHub HTTPS repository URL is allowed.")
        checkout = tempfile.TemporaryDirectory(prefix = "unsloth-skill-")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", source.strip(), checkout.name],
                check = True,
                capture_output = True,
                text = True,
                timeout = 180,
            )
        except FileNotFoundError as error:
            checkout.cleanup()
            raise HTTPException(status_code = 503, detail = "git is not installed on this Mac.") from error
        except subprocess.TimeoutExpired as error:
            checkout.cleanup()
            raise HTTPException(status_code = 504, detail = "The skill repository clone timed out.") from error
        except subprocess.CalledProcessError as error:
            checkout.cleanup()
            raise HTTPException(status_code = 400, detail = "GitHub clone failed.") from error
        # Git owns this private temporary checkout; canonicalize the platform
        # temp-root alias once so macOS ``/var`` -> ``/private/var`` does not
        # trip the no-follow walk used for the actual package read.
        return Path(checkout.name).resolve(strict = True), checkout
    raw = Path(source).expanduser()
    if not is_owner_context():
        # Managed accounts may import only from their own current workspace.  A
        # relative source is workspace-relative (never process-cwd-relative), and
        # lexical ``..`` is rejected even when it happens to resolve back inside.
        base = Path(os.path.abspath(workspace_root()))
        candidate = raw if raw.is_absolute() else base / raw
        candidate = Path(os.path.abspath(candidate))
        try:
            relative = candidate.relative_to(base)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = "Managed skill sources must stay inside the workspace.") from exc
        if not relative.parts or any(part in {".", ".."} for part in raw.parts):
            raise HTTPException(status_code = 400, detail = "Managed skill sources must stay inside the workspace.")
    else:
        candidate = raw if raw.is_absolute() else Path.cwd() / raw
        candidate = Path(os.path.abspath(candidate))
    try:
        descriptor = _open_absolute_directory(candidate)
    except (OSError, HTTPException) as exc:
        raise HTTPException(status_code = 400, detail = "Skill source must be a safe local directory.") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise HTTPException(status_code = 400, detail = "Skill source must be a safe local directory.")
    finally:
        os.close(descriptor)
    return candidate, None


def _install(source: str, target: SkillTarget) -> list[dict[str, object]]:
    root, checkout = _source_directory(source)
    try:
        try:
            root_descriptor = _open_absolute_directory(root)
            source_identity = os.fstat(root_descriptor)
            os.close(root_descriptor)
        except OSError as exc:
            raise HTTPException(status_code = 400, detail = "Skill source changed while being read.") from exc
        sources = _safe_skill_entries(root, expected_identity = source_identity)
        if not sources:
            raise HTTPException(status_code = 400, detail = "The source contains no SKILL.md skill package.")
        planned: list[tuple[_Package, Path, str, os.stat_result | None]] = []
        install_bytes = 0
        package_count = 0
        for skill in sources:
            package_count += 1
            if package_count > _MAX_LOCAL_INSTALL_SKILLS:
                raise HTTPException(status_code = 400, detail = "Skill source contains too many packages.")
            name = _valid_name(skill.path.name)
            package = _read_package(skill.path, skill.identity, skill.marker)
            install_bytes += sum(len(data) for _, data, _ in package.files)
            if install_bytes > _MAX_LOCAL_INSTALL_TOTAL_BYTES:
                raise HTTPException(status_code = 400, detail = "Skill source is too large.")
            for _, destination_root in _targets(target):
                root_identity = _destination_available(destination_root, name)
                planned.append((package, destination_root, name, root_identity))
        publications: list[_Publication] = []
        try:
            for package, destination_root, name, root_identity in planned:
                publications.append(
                    _publish_package(
                        package,
                        destination_root,
                        name,
                        expected_root_identity = root_identity,
                    )
                )
        except HTTPException:
            _rollback_publications(publications)
            raise
        return _list_skills()
    finally:
        if checkout is not None:
            checkout.cleanup()


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    description: str
    source: Literal["agents", "claude", "bundled"]
    enabled: bool
    valid: bool
    shadowed: bool
    shadowed_by: Optional[Literal["agents", "claude", "bundled"]] = None
    error: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[str] = None


class SkillEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: StrictBool


@router.get("", response_model = list[SkillRecord])
def get_skills(current_subject: str = Depends(get_current_subject)) -> list[dict[str, Any]]:
    """Preserve the upstream Agent Skills catalog contract."""
    try:
        records = list_skills()
    except SkillError as exc:
        raise HTTPException(status_code = 500, detail = "Could not read Agent Skills.") from exc
    try:
        from routes.inference import _invalidate_agent_skills_cache
        _invalidate_agent_skills_cache()
    except ImportError:
        pass
    return records


@router.get("/local")
def get_local_skills(current_subject: str = Depends(get_current_subject)):
    """Return the Codex/Claude local manager's simpler skill shape."""
    return {"skills": _list_skills()}


@router.put("/{name}/enabled", response_model = SkillRecord)
def update_skill_enabled(
    name: str,
    payload: SkillEnabledRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        updated = set_skill_enabled(name, payload.enabled)
        try:
            from routes.inference import _invalidate_agent_skills_cache
            _invalidate_agent_skills_cache()
        except ImportError:
            pass
        return updated
    except SkillNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


@router.get("/{name}")
def read_skill(name: str, current_subject: str = Depends(get_current_subject)):
    normalized = _valid_name(name)
    try:
        local = _read_owner_local_skill(normalized)
        if local is None:
            record, instructions = _read_skill_document(normalized, "SKILL.md")
        else:
            record, instructions = local
        if len(instructions) > 100_000:
            raise HTTPException(
                status_code=413,
                detail="Local skill instructions are too large to invoke.",
            )
    except HTTPException:
        raise
    except SkillNotFoundError as error:
        raise HTTPException(status_code=404, detail="Local skill not found.") from error
    except SkillError as error:
        raise HTTPException(status_code=400, detail="Could not read local skill.") from error
    except (OSError, UnicodeError) as error:
        raise HTTPException(status_code=400, detail="Could not read local skill.") from error

    return {
        "name": normalized,
        "description": str(record.get("description") or "Local skill")[:500],
        "instructions": instructions,
    }


@router.post("/create")
def create_skill(payload: CreateSkillRequest, current_subject: str = Depends(get_current_subject)):
    name = _valid_name(payload.name)
    description = payload.description.strip()
    instructions = payload.instructions.strip()
    if not description or not instructions:
        raise HTTPException(status_code = 400, detail = "Skill description and instructions are required.")
    body = f"---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n"
    targets = _targets(payload.target)
    checked_targets: list[tuple[str, Path, os.stat_result | None]] = []
    for ecosystem, root in targets:
        checked_targets.append((ecosystem, root, _destination_available(root, name)))
    encoded_body = body.encode("utf-8")
    package = _Package(encoded_body, ((("SKILL.md",), encoded_body, 0o600),))
    publications: list[_Publication] = []
    try:
        for _, root, root_identity in checked_targets:
            publications.append(
                _publish_package(
                    package,
                    root,
                    name,
                    expected_root_identity = root_identity,
                )
            )
    except HTTPException:
        _rollback_publications(publications)
        raise
    return {"skills": _list_skills()}


@router.post("/install")
def install_skill(payload: InstallSkillRequest, current_subject: str = Depends(get_current_subject)):
    return {"skills": _install(payload.source, payload.target)}
