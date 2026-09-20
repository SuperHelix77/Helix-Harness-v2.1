#!/usr/bin/env python3
"""Name a Helix v3 candidate source snapshot without changing the worktree.

The snapshot is deliberately based on Git's cached and non-ignored untracked
paths, rather than on a clean checkout or a temporary stash.  The digest is a
path/content manifest, so it remains meaningful for the intentionally dirty
candidate tree used by the v3 experiment ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Iterable


SNAPSHOT_SCHEMA = "path-nul-sha256-newline-v2-artifacts-excluded"
BASELINE_PATH = b"docs/helix-reliability-baseline-20260920.md"
ARTIFACTS_PREFIX = b"artifacts/helix-v3/"
EXCLUSIONS = [
    "docs/helix-reliability-baseline-20260920.md",
    "artifacts/helix-v3/**",
]
READ_CHUNK_BYTES = 1024 * 1024


class SourceIdentityError(RuntimeError):
    """Raised when the candidate source tree cannot be safely identified."""


class ExcludedSymlinkSource(SourceIdentityError):
    """A final symlink entry excluded by the regular-file snapshot schema."""


def _git(repo: Path, *args: str) -> bytes:
    """Run Git without invoking a shell and return its raw stdout bytes."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=os.fspath(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise SourceIdentityError(f"unable to execute git: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", "surrogateescape").strip()
        if detail:
            raise SourceIdentityError(f"git {' '.join(args)} failed: {detail}")
        raise SourceIdentityError(
            f"git {' '.join(args)} failed with status {completed.returncode}"
        )
    return completed.stdout


def _git_optional(repo: Path, *args: str) -> bytes | None:
    """Return Git stdout, treating an unavailable symbolic value as absent."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=os.fspath(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise SourceIdentityError(f"unable to execute git: {exc}") from exc
    if completed.returncode:
        return None
    return completed.stdout


def _real_repository(repo_argument: str | os.PathLike[str]) -> Path:
    """Resolve a repository argument to Git's real top-level directory."""

    requested = Path(repo_argument).expanduser()
    try:
        requested = requested.resolve(strict=True)
    except OSError as exc:
        raise SourceIdentityError(f"repository does not resolve: {repo_argument!r}") from exc
    if not requested.is_dir():
        raise SourceIdentityError(f"repository is not a directory: {requested}")

    top_level = _git(requested, "rev-parse", "--show-toplevel").rstrip(b"\r\n")
    if not top_level:
        raise SourceIdentityError("git returned an empty repository path")
    try:
        real = Path(os.fsdecode(top_level)).resolve(strict=True)
    except OSError as exc:
        raise SourceIdentityError("git repository path does not resolve") from exc
    if not real.is_dir():
        raise SourceIdentityError(f"git repository path is not a directory: {real}")
    return real


def _path_for_message(path_bytes: bytes) -> str:
    return os.fsdecode(path_bytes)


def _validate_relative_path(path_bytes: bytes) -> tuple[bytes, ...]:
    """Validate a Git path before using it as a relative filesystem path."""

    if not path_bytes:
        raise SourceIdentityError("git returned an empty path")
    if b"\x00" in path_bytes:
        raise SourceIdentityError("git returned a NUL-containing path")
    if path_bytes.startswith(b"/"):
        raise SourceIdentityError(
            f"unsafe absolute source path: {_path_for_message(path_bytes)!r}"
        )

    components = tuple(path_bytes.split(b"/"))
    if any(component in {b"", b".", b".."} for component in components):
        raise SourceIdentityError(
            f"unsafe source path: {_path_for_message(path_bytes)!r}"
        )
    return components


def _source_paths(repo: Path) -> list[bytes]:
    """Return cached and non-ignored untracked paths in bytewise order."""

    output = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = [path for path in output.split(b"\0") if path]
    paths.sort()

    seen: set[bytes] = set()
    for path_bytes in paths:
        _validate_relative_path(path_bytes)
        if path_bytes in seen:
            raise SourceIdentityError(
                f"git returned a duplicate source path: {_path_for_message(path_bytes)!r}"
            )
        seen.add(path_bytes)
    return paths


def _is_excluded(path_bytes: bytes) -> bool:
    return path_bytes == BASELINE_PATH or path_bytes.startswith(ARTIFACTS_PREFIX)


def _open_regular_nofollow(repo: Path, components: tuple[bytes, ...]) -> int:
    """Open a regular file while rejecting symlinked path components."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    directory_flags = os.O_RDONLY | cloexec | directory | nofollow
    file_flags = os.O_RDONLY | cloexec | nofollow

    try:
        current_fd = os.open(os.fsencode(repo), os.O_RDONLY | cloexec | directory)
    except OSError as exc:
        raise SourceIdentityError(f"cannot open repository root {repo}: {exc}") from exc

    try:
        for component in components[:-1]:
            try:
                before = os.lstat(component, dir_fd=current_fd)
            except OSError as exc:
                raise SourceIdentityError(
                    f"cannot inspect source path component {_path_for_message(component)!r}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise SourceIdentityError(
                    f"refusing symlinked source path component {_path_for_message(component)!r}"
                )
            if not stat.S_ISDIR(before.st_mode):
                raise SourceIdentityError(
                    f"source path component is not a directory: {_path_for_message(component)!r}"
                )
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise SourceIdentityError(
                    f"cannot open source path component {_path_for_message(component)!r}: {exc}"
                ) from exc
            try:
                after = os.fstat(next_fd)
            except OSError as exc:
                os.close(next_fd)
                raise SourceIdentityError(
                    f"cannot inspect opened source path component "
                    f"{_path_for_message(component)!r}: {exc}"
                ) from exc
            if (
                not stat.S_ISDIR(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            ):
                os.close(next_fd)
                raise SourceIdentityError(
                    f"source path component changed while opening: "
                    f"{_path_for_message(component)!r}"
                )
            os.close(current_fd)
            current_fd = next_fd

        final_component = components[-1]
        try:
            before = os.lstat(final_component, dir_fd=current_fd)
        except OSError as exc:
            raise SourceIdentityError(
                f"cannot inspect source file {_path_for_message(b'/'.join(components))!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(before.st_mode):
            raise ExcludedSymlinkSource(
                f"excluding symlink source file: {_path_for_message(b'/'.join(components))!r}"
            )
        if not stat.S_ISREG(before.st_mode):
            raise SourceIdentityError(
                f"source entry is not a regular file: "
                f"{_path_for_message(b'/'.join(components))!r}"
            )
        try:
            file_fd = os.open(final_component, file_flags, dir_fd=current_fd)
        except OSError as exc:
            raise SourceIdentityError(
                f"cannot open source file {_path_for_message(b'/'.join(components))!r}: {exc}"
            ) from exc
        try:
            after = os.fstat(file_fd)
        except OSError as exc:
            os.close(file_fd)
            raise SourceIdentityError(
                f"cannot inspect opened source file "
                f"{_path_for_message(b'/'.join(components))!r}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            os.close(file_fd)
            raise SourceIdentityError(
                f"source file changed while opening: "
                f"{_path_for_message(b'/'.join(components))!r}"
            )
        return file_fd
    finally:
        os.close(current_fd)


def _hash_source_file(repo: Path, path_bytes: bytes) -> tuple[str, int]:
    components = _validate_relative_path(path_bytes)
    file_fd = _open_regular_nofollow(repo, components)
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        try:
            initial_stat = os.fstat(file_fd)
        except OSError as exc:
            raise SourceIdentityError(
                f"cannot inspect source file {_path_for_message(path_bytes)!r}: {exc}"
            ) from exc
        while True:
            try:
                chunk = os.read(file_fd, READ_CHUNK_BYTES)
            except OSError as exc:
                raise SourceIdentityError(
                    f"cannot read source file {_path_for_message(path_bytes)!r}: {exc}"
                ) from exc
            if not chunk:
                break
            digest.update(chunk)
            total_bytes += len(chunk)
        try:
            final_stat = os.fstat(file_fd)
        except OSError as exc:
            raise SourceIdentityError(
                f"cannot inspect source file {_path_for_message(path_bytes)!r}: {exc}"
            ) from exc
        if not stat.S_ISREG(final_stat.st_mode):
            raise SourceIdentityError(
                f"source file became non-regular: {_path_for_message(path_bytes)!r}"
            )
        initial_identity = (
            initial_stat.st_dev,
            initial_stat.st_ino,
            initial_stat.st_size,
            initial_stat.st_mtime_ns,
            initial_stat.st_ctime_ns,
        )
        final_identity = (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        )
        if initial_identity != final_identity or total_bytes != final_stat.st_size:
            raise SourceIdentityError(
                f"source file changed while hashing: {_path_for_message(path_bytes)!r}"
            )
    finally:
        os.close(file_fd)
    return digest.hexdigest(), total_bytes


def _status_count(repo: Path, *, excluded_paths: frozenset[bytes] = frozenset()) -> int:
    """Count identity-relevant Git changes, excluding ledger output paths."""

    output = _git(repo, "status", "--porcelain=v2", "--untracked-files=all", "-z")
    records = output.split(b"\0")
    count = 0
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        record_type = record[:1]
        path: bytes | None = None
        if record_type == b"1":
            fields = record.split(b" ", 8)
            path = fields[8] if len(fields) == 9 else None
        elif record_type == b"u":
            fields = record.split(b" ", 10)
            path = fields[10] if len(fields) == 11 else None
        elif record_type == b"?":
            path = record[2:]
        elif record_type == b"2":
            fields = record.split(b" ", 9)
            path = fields[9] if len(fields) == 10 else None
            # Porcelain-v2 rename/copy records carry the old path as the next
            # NUL-delimited item.
            if index < len(records):
                index += 1
        elif record_type not in {b"#", b"!"}:
            raise SourceIdentityError(
                f"unrecognized git status record: {record[:40]!r}"
            )
        if record_type in {b"1", b"2", b"u", b"?"}:
            if path is None:
                raise SourceIdentityError(
                    f"malformed git status record: {record[:80]!r}"
                )
            _validate_relative_path(path)
            if not _is_excluded(path) and path not in excluded_paths:
                count += 1
    return count


def _git_text(repo: Path, *args: str) -> str | None:
    value = _git_optional(repo, *args)
    if value is None:
        return None
    return value.rstrip(b"\r\n").decode("utf-8", "surrogateescape") or None


def compute_identity(repo: str | os.PathLike[str]) -> dict[str, object]:
    """Compute the candidate source identity for ``repo``."""

    real_repo = _real_repository(repo)
    paths = _source_paths(real_repo)
    tree_digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    excluded_symlinks: list[bytes] = []

    for path_bytes in paths:
        if _is_excluded(path_bytes):
            continue
        try:
            file_sha256, byte_count = _hash_source_file(real_repo, path_bytes)
        except ExcludedSymlinkSource:
            # The frozen v1 schema includes regular files only. Keep the
            # exclusion explicit and never follow the link into another tree.
            excluded_symlinks.append(path_bytes)
            continue
        path_identity = os.fsdecode(path_bytes).encode("utf-8", "surrogateescape")
        tree_digest.update(path_identity)
        tree_digest.update(b"\0")
        tree_digest.update(file_sha256.encode("ascii"))
        tree_digest.update(b"\n")
        file_count += 1
        total_bytes += byte_count

    status_count = _status_count(
        real_repo, excluded_paths=frozenset(excluded_symlinks)
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "repo": str(real_repo),
        "git_head": _git_text(real_repo, "rev-parse", "--verify", "HEAD"),
        "branch": _git_text(real_repo, "symbolic-ref", "--quiet", "--short", "HEAD"),
        "dirty": status_count > 0,
        "status_count": status_count,
        "exclusions": list(EXCLUSIONS),
        "excluded_symlink_count": len(excluded_symlinks),
        "excluded_symlinks": [_path_for_message(path) for path in excluded_symlinks],
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
    }


def _output_path(repo: Path, output_argument: str) -> Path:
    candidate = Path(output_argument).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.absolute()
    artifact_root = (repo / "artifacts" / "helix-v3").resolve(strict=False)
    try:
        resolved_parent = candidate.parent.resolve(strict=False)
        resolved_parent.relative_to(artifact_root)
    except (OSError, ValueError) as exc:
        raise SourceIdentityError(
            "--output must name a file within artifacts/helix-v3/"
        ) from exc
    if candidate.name in {"", ".", ".."}:
        raise SourceIdentityError("--output must name a file")
    return candidate


def _write_atomic(path: Path, payload: bytes) -> None:
    """Atomically replace an output file beneath the already-validated root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=os.fspath(path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        except OSError as exc:
            raise SourceIdentityError(f"cannot atomically write {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    try:
        directory_fd = os.open(
            os.fsencode(path.parent),
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError:
        return
    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
    finally:
        os.close(directory_fd)


def _arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="repository path")
    parser.add_argument(
        "--output",
        help="optional atomic JSON output beneath artifacts/helix-v3/",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        identity = compute_identity(args.repo)
        rendered = json.dumps(
            identity,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        if args.output:
            output_path = _output_path(Path(identity["repo"]), args.output)
            _write_atomic(output_path, rendered)
    except SourceIdentityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
