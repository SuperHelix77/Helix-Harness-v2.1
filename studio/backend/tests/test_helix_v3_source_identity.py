# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "helix-v3-source-identity.py"
BASELINE = "docs/helix-reliability-baseline-20260920.md"
ARTIFACT_ROOT = "artifacts/helix-v3"
SNAPSHOT_SCHEMA = "path-nul-sha256-newline-v2-artifacts-excluded"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("helix_v3_source_identity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HelixV3SourceIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self._temporary.name) / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        self._git("config", "user.email", "helix-tests@example.invalid")
        self._git("config", "user.name", "Helix tests")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _git(self, *args: str) -> bytes:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout

    def _write(self, relative: str, content: bytes) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _commit(self, message: str = "initial") -> None:
        self._git("add", "--all")
        self._git("commit", "-qm", message)

    def _identity(self, *extra: str, cwd: Path | None = None) -> tuple[subprocess.CompletedProcess[bytes], dict]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--repo", str(self.repo), *extra],
            cwd=cwd or self.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        payload = json.loads(completed.stdout) if completed.stdout else {}
        return completed, payload

    @staticmethod
    def _expected_tree(files: dict[bytes, bytes]) -> str:
        digest = hashlib.sha256()
        for relative, content in sorted(files.items()):
            digest.update(relative)
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def test_schema_path_sorting_and_repo_identity_are_deterministic(self) -> None:
        self._write("z.txt", b"z")
        self._write("a.txt", b"a")
        self._commit()

        completed, identity = self._identity(cwd=self.repo.parent)

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(identity["schema_version"], SNAPSHOT_SCHEMA)
        self.assertEqual(identity["repo"], str(self.repo.resolve()))
        self.assertEqual(identity["git_head"], self._git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(
            identity["branch"],
            self._git("symbolic-ref", "--quiet", "--short", "HEAD").decode().strip(),
        )
        self.assertEqual(identity["dirty"], False)
        self.assertEqual(identity["status_count"], 0)
        self.assertEqual(identity["file_count"], 2)
        self.assertEqual(identity["total_bytes"], 2)
        self.assertEqual(
            identity["tree_sha256"],
            self._expected_tree({b"a.txt": b"a", b"z.txt": b"z"}),
        )

    def test_content_change_changes_tree_hash(self) -> None:
        self._write("source.txt", b"before")
        self._commit()
        _, first = self._identity()

        self._write("source.txt", b"after")
        _, second = self._identity()

        self.assertNotEqual(first["tree_sha256"], second["tree_sha256"])
        self.assertEqual(second["file_count"], 1)
        self.assertEqual(second["total_bytes"], 5)
        self.assertTrue(second["dirty"])
        self.assertEqual(second["status_count"], 1)

    def test_untracked_files_are_included_and_ignored_files_are_not(self) -> None:
        self._write("tracked.txt", b"tracked")
        self._write(".gitignore", b"ignored.txt\n")
        self._commit()
        _, clean = self._identity()

        self._write("untracked.txt", b"untracked")
        self._write("ignored.txt", b"ignored")
        _, changed = self._identity()

        self.assertEqual(changed["file_count"], clean["file_count"] + 1)
        self.assertEqual(changed["total_bytes"], clean["total_bytes"] + len(b"untracked"))
        self.assertNotEqual(changed["tree_sha256"], clean["tree_sha256"])
        self.assertEqual(changed["status_count"], 1)

    def test_artifacts_and_output_are_excluded_from_identity(self) -> None:
        self._write("source.txt", b"source")
        self._commit()
        _, clean = self._identity()

        self._write(f"{ARTIFACT_ROOT}/raw.bin", b"first artifact")
        _, with_artifact = self._identity()
        self.assertEqual(with_artifact["tree_sha256"], clean["tree_sha256"])
        self.assertEqual(with_artifact["file_count"], clean["file_count"])

        output = self.repo / ARTIFACT_ROOT / "identity.json"
        completed, from_output = self._identity("--output", str(output))
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(from_output["tree_sha256"], clean["tree_sha256"])
        self.assertEqual(json.loads(output.read_text()), from_output)
        self.assertFalse(list(output.parent.glob(".identity.json.*.tmp")))

        _, after_output = self._identity()
        self.assertEqual(after_output, from_output)

    def test_baseline_document_is_excluded(self) -> None:
        self._write("source.txt", b"source")
        self._commit()
        _, first = self._identity()

        self._write(BASELINE, b"baseline content")
        _, second = self._identity()
        self._write(BASELINE, b"changed baseline content")
        _, third = self._identity()

        self.assertEqual(first["tree_sha256"], second["tree_sha256"])
        self.assertEqual(second["tree_sha256"], third["tree_sha256"])
        self.assertEqual(first["file_count"], second["file_count"])
        self.assertIn(BASELINE, second["exclusions"])
        self.assertIn(f"{ARTIFACT_ROOT}/**", second["exclusions"])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_source_entry_is_explicitly_excluded_without_following(self) -> None:
        self._write("real.txt", b"real")
        os.symlink("real.txt", self.repo / "link.txt")
        self._git("add", "real.txt", "link.txt")
        self._commit()

        completed, first = self._identity()
        self._write("real.txt", b"changed")
        _, second = self._identity()
        (self.repo / "link.txt").unlink()
        os.symlink("outside-and-missing.txt", self.repo / "link.txt")
        self._git("add", "link.txt")
        _, retargeted = self._identity()

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(first["excluded_symlink_count"], 1)
        self.assertEqual(first["excluded_symlinks"], ["link.txt"])
        self.assertEqual(first["file_count"], 1)
        self.assertNotEqual(first["tree_sha256"], second["tree_sha256"])
        self.assertEqual(second["tree_sha256"], retargeted["tree_sha256"])

    def test_output_must_be_inside_artifact_root_and_cwd_is_irrelevant(self) -> None:
        self._write("source.txt", b"source")
        self._commit()
        outside = Path(self._temporary.name) / "outside.json"

        completed, _ = self._identity("--output", str(outside), cwd=Path("/"))

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("artifacts/helix-v3", completed.stderr.decode())

    def test_file_change_during_hash_is_rejected(self) -> None:
        module = _load_script_module()
        self._write("source.txt", b"a" * (2 * 1024 * 1024))
        self._commit()
        source = self.repo / "source.txt"
        original_read = module.os.read
        calls = 0

        def changing_read(fd: int, size: int) -> bytes:
            nonlocal calls
            chunk = original_read(fd, size)
            calls += 1
            if calls == 1:
                source.write_bytes(b"b" * (2 * 1024 * 1024))
            return chunk

        with mock.patch.object(module.os, "read", side_effect=changing_read):
            with self.assertRaisesRegex(module.SourceIdentityError, "changed while hashing"):
                module._hash_source_file(self.repo, b"source.txt")


if __name__ == "__main__":
    unittest.main()
