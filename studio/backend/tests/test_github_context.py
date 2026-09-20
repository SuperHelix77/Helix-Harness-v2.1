# SPDX-License-Identifier: AGPL-3.0-only

import base64

from core.inference import github_context


def test_repository_from_common_github_remotes():
    assert github_context._repository_from_remote("https://github.com/SuperHelix77/q38.git") == "SuperHelix77/q38"
    assert github_context._repository_from_remote("git@github.com:SuperHelix77/q38.git") == "SuperHelix77/q38"
    assert github_context._repository_from_remote("https://gitlab.com/example/repo.git") is None


def test_context_resolves_local_authenticated_repository_without_url(monkeypatch):
    monkeypatch.setattr(github_context, "_local_repositories", lambda: ["SuperHelix77/q38"])
    monkeypatch.setattr(github_context, "_local_branch", lambda _repository: None)

    def fake_json(endpoint, **_kwargs):
        if endpoint == "user":
            return {"login": "SuperHelix77"}
        if endpoint == "repos/SuperHelix77/q38":
            return {"default_branch": "main"}
        if endpoint == "repos/SuperHelix77/q38/git/ref/heads/main":
            return {"object": {"sha": "abc123"}}
        if endpoint == "repos/SuperHelix77/q38/git/trees/abc123?recursive=1":
            return {"tree": [{"path": "docs/EXPERIMENT_LEDGER.md", "type": "blob"}]}
        if endpoint == "repos/SuperHelix77/q38/contents/docs/EXPERIMENT_LEDGER.md?ref=main":
            return {
                "encoding": "base64",
                "content": base64.b64encode(b"Status: future ledger is experimental").decode(),
            }
        raise AssertionError(endpoint)

    monkeypatch.setattr(github_context, "_run_json", fake_json)
    result = github_context.read_github_context({"query": "future ledger v1.1"})
    assert "resolved without a URL" in result
    assert "SuperHelix77/q38" in result
    assert "future ledger is experimental" in result


def test_local_branch_is_preferred_for_automatically_resolved_repository(monkeypatch):
    monkeypatch.setattr(github_context, "_candidate_paths", lambda: ["/workspace"])
    monkeypatch.setattr(github_context, "_git_remote", lambda _path: "git@github.com:SuperHelix77/q38.git")
    monkeypatch.setattr(github_context, "_git_branch", lambda _path: "codex/q38-port-m3")
    assert github_context._local_branch("SuperHelix77/q38") == "codex/q38-port-m3"
