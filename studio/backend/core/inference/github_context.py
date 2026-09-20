# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only GitHub context for local agent runs.

The desktop app cannot directly inherit Codex's connected GitHub MCP session. The
GitHub CLI, however, uses the user's macOS keychain-backed login. This adapter
keeps that boundary explicit: it only invokes ``gh api`` read endpoints, never
accepts a token from a model, and resolves repositories from local remotes or
the authenticated account when the user did not provide a URL.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote


class GitHubContextError(RuntimeError):
    """An actionable, credential-free GitHub context error."""


_REPOSITORY_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]{1,100})/(?P<name>[A-Za-z0-9_.-]{1,100})$"
)
_GITHUB_REMOTE_RE = re.compile(
    r"(?:github\.com[/:])(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".h",
    ".html",
    ".jinja",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_LEDGER_QUERY_RE = re.compile(
    r"ledger|future|v\s*1[._-]?1|mtp|dflash|dflare|qwen3\.8", re.IGNORECASE
)
_MAX_FILE_BYTES = 1_500_000


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GitHubContextError("GitHub context cancelled.")


def _run_gh(args: list[str], *, timeout: int, cancel_event: Any = None) -> str:
    _check_cancel(cancel_event)
    gh = shutil.which("gh")
    if not gh:
        raise GitHubContextError(
            "GitHub CLI is not installed. Install `gh` and run `gh auth login`; "
            "no repository URL is required after that."
        )
    try:
        result = subprocess.run(
            [gh, "api", *args],
            capture_output=True,
            text=True,
            timeout=max(2, min(int(timeout), 60)),
            check=False,
            env={**os.environ, "GH_PAGER": "cat"},
        )
    except subprocess.TimeoutExpired as exc:
        raise GitHubContextError("GitHub API request timed out; try again.") from exc
    except OSError as exc:
        raise GitHubContextError(f"Could not start GitHub CLI: {exc}") from exc
    _check_cancel(cancel_event)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "request failed").strip().splitlines()[0]
        raise GitHubContextError(f"GitHub API request failed: {detail[:300]}")
    return result.stdout


def _run_json(endpoint: str, *, timeout: int, cancel_event: Any = None) -> Any:
    raw = _run_gh([endpoint], timeout=timeout, cancel_event=cancel_event)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GitHubContextError("GitHub returned an unreadable response.") from exc


def _repository_from_remote(remote: str) -> str | None:
    match = _GITHUB_REMOTE_RE.search(remote.strip())
    if not match:
        return None
    return f"{match.group('owner')}/{match.group('name')}"


def _git_remote(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_branch(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except OSError:
        return None
    branch = result.stdout.strip()
    return branch if result.returncode == 0 and branch else None


def _candidate_paths() -> list[Path]:
    """Return a bounded set of likely local workspaces; never scan the home tree."""
    home = Path.home()
    roots = [
        Path.cwd(),
        home / "llmspeed-work",
        home / "Desktop",
        home / "Qwen",
        home / "projects",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.append(root)
            try:
                paths.extend(
                    child
                    for child in root.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                )
            except OSError:
                continue
    return list(dict.fromkeys(paths))


def _local_repositories() -> list[str]:
    repos: list[str] = []
    for path in _candidate_paths():
        repo = _repository_from_remote(_git_remote(path) or "")
        if repo and repo not in repos:
            repos.append(repo)
    return repos


def _local_branch(repository: str) -> str | None:
    """Prefer the checked-out branch when the app resolved a local repository."""
    for path in _candidate_paths():
        if _repository_from_remote(_git_remote(path) or "") == repository:
            branch = _git_branch(path)
            if branch:
                return branch
    return None


def _validate_repository(value: str) -> str:
    match = _REPOSITORY_RE.fullmatch(value.strip())
    if not match:
        raise GitHubContextError("repository must use the owner/name form.")
    return f"{match.group('owner')}/{match.group('name')}"


def _account_login(*, timeout: int, cancel_event: Any = None) -> str | None:
    try:
        data = _run_json("user", timeout=timeout, cancel_event=cancel_event)
    except GitHubContextError:
        return None
    login = data.get("login") if isinstance(data, dict) else None
    return login.strip() if isinstance(login, str) and login.strip() else None


def _score_repository(repo: str, query: str, login: str | None) -> int:
    lowered = repo.lower()
    words = re.findall(r"[a-z0-9]+", query.lower())
    score = 100 if login and lowered.startswith(login.lower() + "/") else 0
    score += sum(30 for word in words if word and word in lowered)
    for hint, points in {
        "q38": 60,
        "unsloth": 45,
        "ledger": 40,
        "mtp": 25,
        "dflash": 25,
        "v1.1": 35,
    }.items():
        if hint in lowered:
            score += points
    return score


def _resolve_repositories(
    repository: str | None,
    query: str,
    *,
    timeout: int,
    cancel_event: Any = None,
) -> list[str]:
    if repository and repository.strip():
        return [_validate_repository(repository)]

    login = _account_login(timeout=timeout, cancel_event=cancel_event)
    local = _local_repositories()
    if local:
        ordered = sorted(
            local,
            key=lambda item: _score_repository(item, query, login),
            reverse=True,
        )
        if login and any(item.lower().startswith(login.lower() + "/") for item in ordered):
            ordered = [
                item
                for item in ordered
                if item.lower().startswith(login.lower() + "/")
            ]
        return ordered[:3]

    data = _run_json(
        "user/repos?per_page=100&sort=updated",
        timeout=timeout,
        cancel_event=cancel_event,
    )
    if not isinstance(data, list):
        return []
    repos = [
        item.get("full_name")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("full_name"), str)
    ]
    return sorted(repos, key=lambda item: _score_repository(item, query, login), reverse=True)[:3]


def _default_branch(repository: str, *, timeout: int, cancel_event: Any = None) -> str:
    data = _run_json(f"repos/{repository}", timeout=timeout, cancel_event=cancel_event)
    branch = data.get("default_branch") if isinstance(data, dict) else None
    return branch if isinstance(branch, str) and branch else "main"


def _tree_paths(
    repository: str,
    branch: str,
    *,
    timeout: int,
    cancel_event: Any = None,
) -> list[str]:
    ref = quote(branch, safe="")
    ref_data = _run_json(
        f"repos/{repository}/git/ref/heads/{ref}",
        timeout=timeout,
        cancel_event=cancel_event,
    )
    sha = (
        ((ref_data or {}).get("object") or {}).get("sha")
        if isinstance(ref_data, dict)
        else None
    )
    if not isinstance(sha, str) or not sha:
        return []
    tree = _run_json(
        f"repos/{repository}/git/trees/{sha}?recursive=1",
        timeout=timeout,
        cancel_event=cancel_event,
    )
    entries = tree.get("tree", []) if isinstance(tree, dict) else []
    return [
        entry["path"]
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("type") == "blob"
        and isinstance(entry.get("path"), str)
    ]


def _read_file(
    repository: str,
    path: str,
    branch: str,
    *,
    timeout: int,
    cancel_event: Any = None,
) -> str | None:
    endpoint = (
        f"repos/{repository}/contents/{quote(path, safe='/')}"
        f"?ref={quote(branch, safe='')}"
    )
    data = _run_json(endpoint, timeout=timeout, cancel_event=cancel_event)
    if not isinstance(data, dict) or data.get("encoding") != "base64":
        return None
    encoded = data.get("content")
    if not isinstance(encoded, str):
        return None
    try:
        raw = base64.b64decode(encoded, validate=False)
    except (ValueError, binascii.Error):
        return None
    if len(raw) > _MAX_FILE_BYTES:
        raw = raw[:_MAX_FILE_BYTES]
    return raw.decode("utf-8", errors="replace")


def _rank_paths(paths: list[str], query: str) -> list[str]:
    query_words = re.findall(r"[a-z0-9]+", query.lower())

    def rank(path: str) -> tuple[int, int, str]:
        lowered = path.lower()
        score = 0
        if lowered in {"docs/experiment_ledger.md", "fieldlab/experiment_ledger.md"}:
            score += 1000
        if "ledger" in lowered:
            score += 300
        if "future" in lowered:
            score += 200
        if any(token in lowered for token in ("v1.1", "v1_1", "v1-1")):
            score += 180
        if "status" in lowered or "intake" in lowered:
            score += 60
        score += sum(80 for word in query_words if word and word in lowered)
        return (-score, len(path), path)

    return sorted(paths, key=rank)


def read_github_context(
    arguments: dict[str, Any] | None,
    *,
    timeout: int = 30,
    cancel_event: Any = None,
) -> str:
    """Resolve and read useful GitHub context without requiring a repository URL."""
    args = arguments if isinstance(arguments, dict) else {}
    query = str(args.get("query") or "").strip()
    path = str(args.get("path") or "").strip().lstrip("/")
    requested_ref = str(args.get("ref") or "").strip()
    repository = args.get("repository")
    repository = repository if isinstance(repository, str) else None
    try:
        max_chars = max(2_000, min(int(args.get("max_chars") or 24_000), 60_000))
    except (TypeError, ValueError):
        max_chars = 24_000

    repos = _resolve_repositories(
        repository,
        query,
        timeout=timeout,
        cancel_event=cancel_event,
    )
    if not repos:
        raise GitHubContextError(
            "No GitHub repository was found for the authenticated account or local remotes. "
            "Run `gh auth login`; a URL is optional."
        )

    sections: list[str] = ["GitHub context (read-only; resolved without a URL)"]
    for repo in repos:
        _check_cancel(cancel_event)
        branch = (
            requested_ref
            or _local_branch(repo)
            or _default_branch(repo, timeout=timeout, cancel_event=cancel_event)
        )
        selected = [path] if path else []
        if not selected:
            try:
                paths = [
                    item
                    for item in _tree_paths(
                        repo,
                        branch,
                        timeout=timeout,
                        cancel_event=cancel_event,
                    )
                    if Path(item).suffix.lower() in _TEXT_SUFFIXES
                ]
            except GitHubContextError:
                paths = []
            if _LEDGER_QUERY_RE.search(query):
                selected = _rank_paths(paths, query)[:4]
            else:
                selected = [
                    item
                    for item in _rank_paths(paths, query)
                    if Path(item).name.lower()
                    in {"readme.md", "experiment_ledger.md", "status.json"}
                ][:3]
        sections.append(
            f"\nRepository: {repo}\nBranch: {branch}\n"
            f"URL: https://github.com/{repo}"
        )
        added = 0
        for file_path in selected:
            if added >= max_chars:
                break
            content = _read_file(
                repo,
                file_path,
                branch,
                timeout=timeout,
                cancel_event=cancel_event,
            )
            if content is None:
                continue
            remaining = max_chars - added
            body = content[:remaining]
            if len(content) > remaining:
                body += "\n[…truncated by Unsloth GitHub context budget…]"
            sections.append(f"\n--- {repo}:{file_path} ---\n{body}")
            added += len(body)
        if not selected:
            sections.append(
                "\nNo matching text files were found; repository metadata was resolved successfully."
            )
    return "".join(sections)
