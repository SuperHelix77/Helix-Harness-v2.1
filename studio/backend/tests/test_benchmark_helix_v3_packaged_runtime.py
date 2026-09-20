"""Focused, no-launch tests for the packaged Helix v3 benchmark harness."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import sys
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "benchmark-helix-v3-packaged-runtime.py"
)
_SPEC = importlib.util.spec_from_file_location("helix_v3_packaged_runtime", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HARNESS
_SPEC.loader.exec_module(_HARNESS)


def _fixture_session_id(pid):
    """Synthetic ps rows use PID==SID; production uses os.getsid(pid)."""

    return int(pid)


def _parse_fixture_ps(output, *, session_id_getter=_fixture_session_id):
    return _HARNESS._parse_ps_snapshot(output, session_id_getter=session_id_getter)


def test_identity_mismatch_fails_closed():
    with pytest.raises(_HARNESS.IdentityMismatchError):
        _HARNESS.verify_health_identity(
            {
                "helix_backend_contract": _HARNESS.EXPECTED_BACKEND_CONTRACT,
                "helix_backend_tree_sha256": "0" * 64,
                "helix_backend_verified": True,
            }
        )


def test_installed_desktop_app_path_is_rejected(tmp_path):
    installed = Path("/Applications/Helix Harness v2.app")
    with pytest.raises(_HARNESS.AppPathError, match="installed Desktop"):
        _HARNESS.validate_app_path(installed)

    temporary = tmp_path / "Helix Harness v2.app"
    assert _HARNESS.validate_app_path(temporary) == temporary.resolve()


def test_secret_redaction_applies_to_text_and_secret_fields():
    secret = "desktop-test-secret-123456"
    token = "eyJaccess-token-that-must-not-escape"
    value = {
        "authorization": f"Bearer {token}",
        "message": f"desktop secret={secret}",
        "nested": [{"access_token": token}],
    }
    sanitized = _HARNESS.sanitize_for_artifact(value, [secret, token])
    rendered = json.dumps(sanitized, sort_keys=True)
    assert secret not in rendered
    assert token not in rendered
    assert "[REDACTED]" in rendered


def test_cleanup_is_pid_scoped_and_does_not_target_unrelated_processes():
    ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 child\n"
        "900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n"
    )

    def runner(command, **_kwargs):
        if "-axo" in command:
            return SimpleNamespace(stdout=ps)
        assert "-p" in command
        pid = int(command[command.index("-p") + 1])
        row = next(line for line in ps.splitlines() if line.startswith(f"{pid} "))
        return SimpleNamespace(stdout=row + "\n")

    assert _HARNESS.process_tree_pids(
        100, command_runner=runner, session_id_getter=_fixture_session_id
    ) == {100, 101}
    identities = _HARNESS.capture_process_identities(
        100, command_runner=runner, session_id_getter=_fixture_session_id
    )

    class Process:
        pid = 100

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, **_kwargs):
            return 0

        def kill(self):
            self.terminated = True

    sent = []

    def send(pid, value):
        sent.append((pid, value))

    group_sent = []
    process = Process()
    _HARNESS.terminate_owned_processes(
        process,
        {100, 101},
        owned_identities=identities,
        command_runner=runner,
        signal_sender=send,
        group_signal_sender=lambda pgid, value: group_sent.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert sent == []
    assert group_sent == [(100, signal.SIGTERM)]
    assert process.terminated


def test_cleanup_detects_reparented_survivor_by_captured_identity():
    captured_ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 child --owned\n"
    )
    reparented_ps = "101 1 100 20 Sun Sep 20 10:00:01 2026 child --owned\n"

    def runner(command, **_kwargs):
        if "-axo" in command:
            return SimpleNamespace(stdout=reparented_ps)
        if "-p" in command:
            return SimpleNamespace(stdout=reparented_ps)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    identities = _parse_fixture_ps(captured_ps)
    result = _HARNESS.verify_cleanup(
        100,
        {100, 101},
        8888,
        command_runner=runner,
        owned_identities=identities,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "fail"
    assert result["remaining_owned_pids"] == [101]


def test_cleanup_detects_late_uncaptured_reparented_group_child():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
    )
    late_child = "103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker --owned\n"

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=late_child)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    result = _HARNESS.verify_cleanup(
        100,
        {100, 101},
        8888,
        command_runner=runner,
        owned_identities=captured,
        owned_pgid=100,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "fail"
    assert result["remaining_owned_pids"] == []
    assert result["remaining_owned_group_pids"] == [103]


def test_group_and_root_pid_reuse_fail_closed_without_group_signal():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
    )
    reused = (
        "100 1 101 10 Sun Sep 20 11:00:00 2026 unrelated-root --reused\n"
        "103 1 101 20 Sun Sep 20 11:00:01 2026 unrelated-child\n"
    )

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=reused)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    checked = _HARNESS.verify_cleanup(
        100,
        {100, 101},
        8888,
        command_runner=runner,
        owned_identities=captured,
        owned_pgid=100,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert checked["status"] == "fail"
    assert checked["identity_mismatches"] == [100]

    class Process:
        pid = 100

        def poll(self):
            return 0

    group_signals = []
    terminated = _HARNESS.terminate_owned_processes(
        Process(),
        {100, 101},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: group_signals.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert group_signals == []
    assert terminated["group_observation"] == "owned_group_identity_mismatch"
    assert terminated["status"] == "fail"


def test_group_revalidation_rejects_captured_child_reuse_before_group_signal():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
    )
    reused_child = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 1 100 20 Sun Sep 20 11:00:00 2026 unrelated --reused\n"
    )

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=reused_child)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        if "-p" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="101 1 100 20 Sun Sep 20 11:00:00 2026 unrelated --reused\n",
            )
        raise AssertionError(command)

    class Process:
        pid = 100

        def poll(self):
            return 0

    group_signals = []
    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100, 101},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: group_signals.append((pgid, value)),
        signal_sender=lambda _pid, _value: None,
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert group_signals == []
    assert result["group_observation"] == "owned_group_identity_mismatch"
    assert result["status"] == "fail"


def test_process_group_guard_refuses_invalid_or_harness_group(monkeypatch):
    root = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )[100]
    assert _HARNESS._validate_owned_process_group(100, 0, root) == (
        False,
        "invalid_owned_pgid",
    )
    monkeypatch.setattr(_HARNESS.os, "getpgrp", lambda: 100)
    assert _HARNESS._validate_owned_process_group(100, 100, root) == (
        False,
        "owned_pgid_is_harness_group",
    )

    class Process:
        pid = 100

        def poll(self):
            return 0

    signals = []
    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100},
        owned_identities={100: root},
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: signals.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert signals == []
    assert result["group_guard"] == "owned_pgid_is_harness_group"
    assert result["status"] == "fail"


def test_process_group_guard_requires_pid_pgid_sid_session_leader():
    root = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n",
        session_id_getter=lambda _pid: 99,
    )[100]
    assert _HARNESS._validate_owned_process_group(100, 100, root) == (
        False,
        "root_is_not_session_leader",
    )


def test_process_and_listener_sampling_uses_bounded_commands():
    timeouts = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
            )
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    assert _HARNESS._run_ps_snapshot(
        command_runner=runner, session_id_getter=_fixture_session_id
    )
    assert _HARNESS.listener_pids(8888, command_runner=runner) == set()
    assert timeouts == [_HARNESS.COMMAND_TIMEOUT_S, _HARNESS.COMMAND_TIMEOUT_S]


def test_host_ps_wire_is_supported_and_session_enriched_read_only():
    row = _HARNESS._run_ps_snapshot(pid=os.getpid())
    assert row is not None
    current_pid = os.getpid()
    identity = row[current_pid]
    assert "sid=" not in " ".join(_HARNESS._ps_command())
    assert identity["sid"] == os.getsid(current_pid)
    assert identity["pgid"] > 0
    assert identity["start_marker"]
    assert identity["command"]
    assert _HARNESS._has_identity_token(identity)


def test_ps_snapshot_drops_only_unrelated_getsid_race():
    ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n"
    )

    def vanished_unrelated(pid):
        if int(pid) == 900:
            raise ProcessLookupError(pid)
        return int(pid)

    snapshot = _HARNESS._run_ps_snapshot(
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=ps, stderr=""
        ),
        session_id_getter=vanished_unrelated,
        required_pids={100},
        required_pgid=100,
    )
    assert snapshot is not None
    assert set(snapshot) == {100}


def test_ps_snapshot_fails_when_owned_group_getsid_races():
    ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 child --owned\n"
    )

    def vanished_owned(pid):
        if int(pid) == 101:
            raise ProcessLookupError(pid)
        return int(pid)

    assert _HARNESS._run_ps_snapshot(
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=ps, stderr=""
        ),
        session_id_getter=vanished_owned,
        required_pids={100, 101},
        required_pgid=100,
    ) is None


def test_capture_protects_intermediate_descendant_before_getsid_filtering():
    ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
        "102 101 100 30 Sun Sep 20 10:00:02 2026 grandchild --owned\n"
        "900 1 900 40 Sun Sep 20 10:00:03 2026 unrelated\n"
    )

    def vanished_intermediate(pid):
        if int(pid) == 101:
            raise ProcessLookupError(pid)
        return int(pid)

    captured = _HARNESS.capture_process_identities(
        100,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=ps, stderr=""
        ),
        session_id_getter=vanished_intermediate,
    )
    assert captured == {}


def test_capture_protects_same_owned_pgid_even_after_reparenting():
    ps = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 1 100 20 Sun Sep 20 10:00:01 2026 reparented --owned\n"
        "900 1 900 40 Sun Sep 20 10:00:03 2026 unrelated\n"
    )

    def vanished_owned_group_member(pid):
        if int(pid) == 101:
            raise ProcessLookupError(pid)
        return int(pid)

    captured = _HARNESS.capture_process_identities(
        100,
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=ps, stderr=""
        ),
        session_id_getter=vanished_owned_group_member,
    )
    assert captured == {}


def test_short_numeric_and_legacy_no_session_rows_fail_closed():
    def short_row(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="100 1 100\n", stderr="")

    assert _HARNESS._run_ps_snapshot(
        command_runner=short_row,
        session_id_getter=_fixture_session_id,
    ) is None

    legacy_row = (
        "100 1 100 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )

    def legacy(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=legacy_row, stderr="")

    assert _HARNESS._run_ps_snapshot(
        command_runner=legacy,
        session_id_getter=_fixture_session_id,
    ) is None

    missing_session = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )

    assert _HARNESS._run_ps_snapshot(
        command_runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=missing_session, stderr=""
        ),
        session_id_getter=lambda _pid: None,
    ) is None


def test_reaper_boundary_rejects_non_default_sigchld_and_lost_waitability():
    assert _HARNESS._reaper_boundary_status(
        signal_getter=lambda _sig: _HARNESS.signal.SIG_IGN
    ) == (False, "sigchld_not_default")

    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )
    signals = []

    class Process:
        pid = 100

        def terminate(self):
            pass

        def wait(self, **_kwargs):
            return 0

    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100},
        owned_identities=captured,
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: signals.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: (_ for _ in ()).throw(ChildProcessError()),
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert signals == []
    assert result["group_observation"] == "waitability_lost"
    assert result["status"] == "fail"


def test_zombie_root_with_exact_identity_can_anchor_late_group_signal():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )
    current = (
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned <defunct>\n"
        "103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker\n"
    )
    signals = []
    waitid_calls = []

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=current, stderr="")
        raise AssertionError(command)

    class Process:
        pid = 100
        returncode = None

        def terminate(self):
            pass

        def wait(self, **_kwargs):
            return 0

    def waitid(idtype, pid, options):
        waitid_calls.append((idtype, pid, options))
        return SimpleNamespace(si_pid=pid, si_code=1)

    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        session_id_getter=_fixture_session_id,
        waitid_fn=waitid,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
        group_signal_sender=lambda pgid, value: signals.append((pgid, value)),
    )
    assert signals == [(100, signal.SIGTERM)]
    assert result["group_observation"] == "validated_owned_group_observed"
    assert result["root_lifetime"]["state"] == "exited_unreaped"
    assert waitid_calls == [
        (_HARNESS.os.P_PID, 100, _HARNESS.os.WEXITED | _HARNESS.os.WNOHANG | _HARNESS.os.WNOWAIT)
    ]


def test_group_signal_precedes_owner_aware_root_reap():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
    )
    events = []

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
                    "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
                ),
            )
        raise AssertionError(command)

    class Process:
        pid = 100

        def terminate(self):
            events.append("terminate")

        def wait(self, **_kwargs):
            events.append("wait")
            return 0

        def kill(self):
            events.append("kill")

    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100, 101},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        group_signal_sender=lambda _pgid, _value: events.append("group"),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert result["status"] == "pass"
    assert events.index("group") < events.index("terminate")


@pytest.mark.parametrize(
    ("current", "expected_signal"),
    [
        (
            "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
            "103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker\n",
            True,
        ),
        (
            "103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker\n",
            False,
        ),
    ],
)
def test_unknown_late_group_member_requires_unreaped_leader_anchor(current, expected_signal):
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )
    signals = []

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout=current)
        raise AssertionError(command)

    class Process:
        pid = 100

        def terminate(self):
            pass

        def wait(self, **_kwargs):
            return 0

    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: signals.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert bool(signals) is expected_signal
    if expected_signal:
        assert result["group_observation"] == "validated_owned_group_observed"
    else:
        assert result["group_observation"] == "owned_group_identity_unproven"
        assert result["status"] == "fail"


def test_group_signal_refuses_reaped_root_lifetime_anchor():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )
    signals = []

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker\n",
            )
        raise AssertionError(command)

    class Process:
        pid = 100
        returncode = 0

        def terminate(self):
            pass

        def wait(self, **_kwargs):
            return 0

    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100},
        owned_identities=captured,
        command_runner=runner,
        owned_pgid=100,
        group_signal_sender=lambda pgid, value: signals.append((pgid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: (_ for _ in ()).throw(ChildProcessError()),
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert signals == []
    assert result["group_observation"] == "root_lifetime_not_pinned"
    assert result["status"] == "fail"


def test_pid_reuse_fails_closed_before_signaling():
    captured_ps = "101 100 100 20 Sun Sep 20 10:00:01 2026 child --owned\n"
    reused_ps = "101 1 101 20 Sun Sep 20 11:00:01 2026 unrelated --reused\n"

    def runner(command, **_kwargs):
        assert "-p" in command
        return SimpleNamespace(stdout=reused_ps)

    class Process:
        pid = 100

        def poll(self):
            return 0

    sent = []
    result = _HARNESS.terminate_owned_processes(
        Process(),
        {100, 101},
        owned_identities=_parse_fixture_ps(captured_ps),
        command_runner=runner,
        signal_sender=lambda pid, value: sent.append((pid, value)),
        session_id_getter=_fixture_session_id,
        waitid_fn=lambda *_args: None,
        signal_getter=lambda _sig: _HARNESS.signal.SIG_DFL,
    )
    assert sent == []
    assert result["status"] == "fail"
    assert result["identity_mismatches"] == []


def test_pid_reuse_does_not_adopt_new_descendants():
    owned = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
        "101 100 100 20 Sun Sep 20 10:00:01 2026 worker --owned\n"
    )
    reused = _parse_fixture_ps(
        "100 1 101 10 Sun Sep 20 11:00:00 2026 unrelated-root\n"
        "999 100 101 20 Sun Sep 20 11:00:00 2026 unrelated-child\n"
    )
    conflicts = _HARNESS._merge_process_identities(owned, reused)
    assert conflicts == {100}
    assert 999 not in owned
    assert owned[100]["command"] == "app --owned"


def test_any_listener_on_dedicated_port_fails_cleanup():
    ps = "900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n"

    def runner(command, **_kwargs):
        if "-axo" in command:
            return SimpleNamespace(stdout=ps)
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(stdout="900\n")
        raise AssertionError(command)

    result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=runner,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "fail"
    assert result["unexpected_listener_pids"] == [900]


def test_lsof_no_match_is_a_clean_dedicated_port():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
            )
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=runner,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "pass"
    assert result["listener_pids"] == []
    assert result["consecutive_clean_snapshots"] == 2


def test_cleanup_never_passes_for_uncaptured_owned_pid():
    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
            )
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=runner,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "fail"
    assert result["uncaptured_owned_pids"] == [100]


def test_malformed_ps_or_lsof_output_never_admits_cleanup():
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )

    def malformed_ps(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=0, stdout="not a ps row\n")
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    ps_result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=malformed_ps,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert ps_result["status"] == "unavailable"
    assert ps_result["consecutive_clean_snapshots"] == 0

    def malformed_lsof(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
            )
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=0, stdout="not-a-pid\n")
        raise AssertionError(command)

    lsof_result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=malformed_lsof,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert lsof_result["status"] == "unavailable"
    assert lsof_result["consecutive_clean_snapshots"] == 0

    def diagnostic_lsof(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(
                returncode=0,
                stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
            )
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="", stderr="lsof diagnostic")
        raise AssertionError(command)

    diagnostic_result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=diagnostic_lsof,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert diagnostic_result["status"] == "unavailable"

    def empty_ps_error(command, **_kwargs):
        if command[0] == "/bin/ps":
            return SimpleNamespace(returncode=1, stdout="", stderr="ps diagnostic")
        if command[0] == "/usr/sbin/lsof":
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    empty_result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=empty_ps_error,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert empty_result["status"] == "unavailable"


@pytest.mark.parametrize("late_state", ["listener", "group"])
def test_cleanup_requires_consecutive_clean_snapshots_and_catches_late_state(late_state):
    captured = _parse_fixture_ps(
        "100 1 100 10 Sun Sep 20 10:00:00 2026 app --owned\n"
    )
    calls = {"ps": 0, "lsof": 0}

    def runner(command, **_kwargs):
        if command[0] == "/bin/ps":
            calls["ps"] += 1
            if calls["ps"] == 1 or late_state == "listener":
                return SimpleNamespace(
                    returncode=0,
                    stdout="900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="103 1 100 20 Sun Sep 20 10:00:02 2026 late-worker\n",
            )
        if command[0] == "/usr/sbin/lsof":
            calls["lsof"] += 1
            if late_state == "listener" and calls["lsof"] == 2:
                return SimpleNamespace(returncode=0, stdout="900\n")
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(command)

    result = _HARNESS.verify_cleanup(
        100,
        {100},
        8888,
        command_runner=runner,
        owned_identities=captured,
        session_id_getter=_fixture_session_id,
        sleep_fn=lambda _seconds: None,
    )
    assert result["status"] == "fail"
    assert result["consecutive_clean_snapshots"] == 1
    if late_state == "listener":
        assert result["unexpected_listener_pids"] == [900]
    else:
        assert result["remaining_owned_group_pids"] == [103]


def test_output_path_is_confined_to_versioned_artifact_directory(tmp_path):
    allowed = _HARNESS.confine_output_path(
        tmp_path, "artifacts/helix-v3/packaged-runtime.json"
    )
    assert allowed == (tmp_path / "artifacts/helix-v3/packaged-runtime.json").resolve()
    with pytest.raises(_HARNESS.OutputPathError):
        _HARNESS.confine_output_path(tmp_path, tmp_path / "outside.json")
    with pytest.raises(_HARNESS.OutputPathError):
        _HARNESS.confine_output_path(tmp_path, "artifacts/helix-v3/../escape.json")


def test_dry_run_never_launches_or_writes(monkeypatch, tmp_path):
    app_path = tmp_path / "frozen-control.app"
    app_path.mkdir()
    runtime_template = _make_runtime_template(tmp_path)
    launched = []

    def launch(*_args, **_kwargs):
        launched.append(True)
        raise AssertionError("dry-run launched the app")

    monkeypatch.setattr(_HARNESS, "_launch_app", launch)
    result = _HARNESS.run_benchmark(
        _HARNESS.HarnessConfig(
            app_path=app_path,
            runtime_home_template=runtime_template,
            execute=False,
        ),
        repo_root=tmp_path,
    )
    assert result["status"] == "dry_run"
    assert result["would_launch"] is False
    assert not launched
    assert not (tmp_path / "artifacts").exists()


def test_launch_starts_packaged_control_in_new_process_session(tmp_path):
    app = tmp_path / "Helix Harness v2.app"
    executable = app / "Contents" / "MacOS" / "unsloth-studio"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    captured = {}

    def popen_factory(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    _HARNESS._launch_app(app, home, popen_factory=popen_factory)
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["argv"] == [str(executable)]


def test_streaming_metrics_keep_usage_and_ttft_with_mocked_http():
    class Transport:
        def open_stream(self, *_args, **_kwargs):
            return io.BytesIO(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
                b'data: {"usage":{"prompt_tokens":3,"completion_tokens":1},'
                b'"timings":{"prompt_ms":2,"predicted_ms":4}}\n\n'
                b"data: [DONE]\n"
            )

    case = _HARNESS.PromptCase("prompt-00", "ok", 0, "AB", 0)
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {"Authorization": "Bearer in-memory"},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert result["usage"]["prompt_tokens"] == 3
    assert result["metrics"]["latency"]["ttft_first_user_visible_token_ms"]["value"] is not None
    assert result["metrics"]["quality_and_work"]["tool_calls"]["value"] == 0
    assert "top_k" not in result["request"]
    assert result["protocol"]["status"] == "pass"
    assert result["response_text_length"] == 2


def test_default_chat_payload_disables_thinking_and_reasoning():
    payload = _HARNESS.build_chat_payload("local-mlx", "prompt", seed=1, max_tokens=8)
    assert payload["enable_thinking"] is False
    assert payload["reasoning_effort"] == "none"


def test_default_exact_output_is_a_safe_task_success():
    captured = {}

    class Transport:
        def open_stream(self, *_args, **kwargs):
            captured["payload"] = kwargs["payload"]
            return io.BytesIO(
                b'data: {"choices":[{"delta":{"content":"HELIX_BENCHMARK_ALPHA"}}]}\n\n'
                b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n"
            )

    case = _HARNESS.PromptCase(
        "prompt-00", _HARNESS.DEFAULT_PROMPTS[0], 0, "AB", 0
    )
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["reasoning_effort"] == "none"
    assert result["correctness"]["outcome"] == "pass"
    assert result["correctness"]["reason"] == "exact_expected_output_match"
    assert result["response_text_length"] == len("HELIX_BENCHMARK_ALPHA")
    assert "HELIX_BENCHMARK_ALPHA" not in json.dumps(result, sort_keys=True)
    assert _HARNESS.task_success_gate([result])["status"] == "pass"


def test_terminal_length_with_no_visible_output_fails_task_success():
    class Transport:
        def open_stream(self, *_args, **_kwargs):
            return io.BytesIO(
                b'data: {"choices":[{"finish_reason":"length"}]}\n\n'
                b"data: [DONE]\n"
            )

    case = _HARNESS.PromptCase(
        "prompt-00", _HARNESS.DEFAULT_PROMPTS[0], 0, "AB", 0
    )
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert result["protocol"]["status"] == "pass"
    assert result["protocol"]["finish_reason"] == "length"
    assert result["correctness"] == {
        "outcome": "fail",
        "evidence": "safe_default_exact_output",
        "evidence_scope": "client_observed_output",
        "objective_correctness": "unverified",
        "protocol_status": "pass",
        "visible_output_length": 0,
        "visible_output_sha256": None,
        "expected_output_length": len("HELIX_BENCHMARK_ALPHA"),
        "expected_output_sha256": _HARNESS._sha256_bytes(
            b"HELIX_BENCHMARK_ALPHA"
        ),
        "exact_match": False,
        "reason": "length_without_visible_output",
    }
    assert _HARNESS.task_success_gate([result])["status"] == "fail"


def test_default_output_mismatch_fails_task_success():
    class Transport:
        def open_stream(self, *_args, **_kwargs):
            return io.BytesIO(
                b'data: {"choices":[{"delta":{"content":"WRONG"}}]}\n\n'
                b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n"
            )

    case = _HARNESS.PromptCase(
        "prompt-00", _HARNESS.DEFAULT_PROMPTS[0], 0, "AB", 0
    )
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert result["correctness"]["outcome"] == "fail"
    assert result["correctness"]["reason"] == "expected_output_mismatch"
    assert "WRONG" not in json.dumps(result, sort_keys=True)


def test_custom_prompt_remains_unverified_and_cannot_pass_gate():
    case = _HARNESS.PromptCase("custom", "Say anything", 0, "AB", 0)
    trial = {
        "protocol": {"status": "pass", "finish_reason": "stop"},
        "visible_output_chars": 3,
        "visible_output_sha256": "abc",
    }
    correctness = _HARNESS.evaluate_trial_correctness(case, trial)
    assert correctness["outcome"] == "unverified"
    assert _HARNESS.task_success_gate([{**trial, "correctness": correctness}])["status"] == "fail"


@pytest.mark.parametrize(
    ("wire", "reason", "saw_done"),
    [
        (b'data: {"error":{"message":"backend failed"}}\n', "error_frame", False),
        (
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
            b"data: [DONE]\n",
            "missing_terminal_finish",
            True,
        ),
        (
            b'data: {"choices":[{"finish_reason":"stop"}]}\n',
            "missing_done_marker",
            False,
        ),
    ],
)
def test_stream_protocol_failures_are_explicit_and_not_complete(wire, reason, saw_done):
    class Transport:
        def open_stream(self, *_args, **_kwargs):
            return io.BytesIO(wire)

    case = _HARNESS.PromptCase("prompt-00", "ok", 0, "AB", 0)
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert result["protocol"] == {
        "status": "fail",
        "reason": reason,
        "saw_done": saw_done,
    }


@pytest.mark.parametrize(
    ("wire", "reason"),
    [
        (
            b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n'
            b"data: [DONE]\n",
            "tools_disabled_finish_reason",
        ),
        (
            b'data: {not-json}\n'
            b'data: {"choices":[{"finish_reason":"stop"}]}\n'
            b"data: [DONE]\n",
            "malformed_data_frame",
        ),
        (
            b'data: {"type":"tool_start","name":"disabled"}\n'
            b'data: {"choices":[{"finish_reason":"stop"}]}\n'
            b"data: [DONE]\n",
            "tools_disabled_event",
        ),
    ],
)
def test_tools_disabled_and_malformed_frames_fail_even_with_terminal_marker(wire, reason):
    class Transport:
        def open_stream(self, *_args, **_kwargs):
            return io.BytesIO(wire)

    case = _HARNESS.PromptCase("prompt-00", "ok", 0, "AB", 0)
    result = _HARNESS.stream_prompt(
        Transport(),
        "http://127.0.0.1:8888",
        {},
        model_name="local-mlx",
        case=case,
        seed=1,
        max_tokens=8,
        timeout=1,
    )
    assert result["protocol"]["status"] == "fail"
    assert result["protocol"]["reason"] == reason


def _make_runtime_template(tmp_path: Path) -> Path:
    template = tmp_path / "runtime-template"
    shared_python = tmp_path / "shared-uv-python"
    shared_python.write_bytes(b"#!/usr/bin/env python3\n")
    venv = template / ".unsloth" / "studio" / "unsloth_studio"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /template\n", encoding="utf-8")
    (venv / "bin" / "unsloth").write_text(
        f"#!{template}/.unsloth/studio/unsloth_studio/bin/python\n", encoding="utf-8"
    )
    (venv / "bin" / "python").symlink_to(shared_python)
    auth = template / ".unsloth" / "studio" / "auth"
    auth.mkdir(parents=True)
    (auth / ".desktop_secret").write_text("desktop-clone-secret", encoding="utf-8")
    return template


def test_runtime_interpreter_identity_records_shared_symlink_and_rechecks(tmp_path, monkeypatch):
    template = _make_runtime_template(tmp_path)
    monkeypatch.setattr(_HARNESS.platform, "system", lambda: "Darwin")

    def copy_runner(command, **_kwargs):
        shutil.copytree(Path(command[2]), Path(command[3]), symlinks=True)
        return SimpleNamespace(returncode=0)

    def relocator(venv):
        for script in venv.joinpath("bin").iterdir():
            if not script.is_symlink():
                script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    isolated = _HARNESS._isolated_home(template, copy_runner=copy_runner, relocator=relocator)
    clone = isolated.__enter__()
    assert isolated.interpreter_identity["is_symlink"] is True
    assert isolated.interpreter_identity["is_interpreter_isolated"] is False
    assert isolated.interpreter_identity["target_sha256"]
    shared_target = Path(isolated.interpreter_identity["target_path"])
    shared_target.write_bytes(b"#!/usr/bin/env python3 changed\n")
    assert isolated.recheck_interpreter_identity() is False
    clone_root = clone.parent
    isolated.__exit__(None, None, None)
    assert clone_root.exists()
    shutil.rmtree(clone_root)


def test_dry_run_rejects_missing_app_or_runtime_template(tmp_path):
    runtime = _make_runtime_template(tmp_path)
    with pytest.raises(_HARNESS.HarnessError):
        _HARNESS.run_benchmark(
            _HARNESS.HarnessConfig(
                app_path=tmp_path / "missing.app",
                runtime_home_template=runtime,
                execute=False,
            ),
            repo_root=tmp_path,
        )
    app = tmp_path / "present.app"
    app.mkdir()
    with pytest.raises(_HARNESS.HarnessError):
        _HARNESS.run_benchmark(
            _HARNESS.HarnessConfig(
                app_path=app,
                runtime_home_template=tmp_path / "missing-runtime-home",
                execute=False,
            ),
            repo_root=tmp_path,
        )


def test_explicit_invalid_model_preflight_is_non_success_without_launch(tmp_path, monkeypatch):
    runtime = _make_runtime_template(tmp_path)
    app = tmp_path / "present.app"
    app.mkdir()
    monkeypatch.setattr(
        _HARNESS,
        "_launch_app",
        lambda *_args, **_kwargs: pytest.fail("preflight launched the app"),
    )
    result = _HARNESS.run_benchmark(
        _HARNESS.HarnessConfig(
            app_path=app,
            runtime_home_template=runtime,
            model_path=tmp_path / "missing-model",
            model_config_path=tmp_path / "missing-config.json",
            execute=False,
        ),
        repo_root=tmp_path,
    )
    assert result["status"] == "preflight_failed"
    assert result["model"]["status"] == "invalid"
    assert result["would_launch"] is False
    assert not (tmp_path / "artifacts").exists()


def test_preflight_records_nested_model_identity_revision_and_repository(tmp_path):
    runtime = _make_runtime_template(tmp_path)
    app = tmp_path / "present.app"
    app.mkdir()
    model = tmp_path / "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
    model.mkdir()
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "backend": "mlx",
                "model_identity": {
                    "hugging_face_revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
                    "repository": "mlx-community/Qwen3.8-27B-4bit",
                },
            }
        ),
        encoding="utf-8",
    )
    result = _HARNESS.run_benchmark(
        _HARNESS.HarnessConfig(
            app_path=app,
            runtime_home_template=runtime,
            model_path=model,
            model_config_path=config,
            execute=False,
        ),
        repo_root=tmp_path,
    )
    assert result["status"] == "dry_run"
    assert result["model"]["hf_revision"] == "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
    assert result["model"]["repository"] == "mlx-community/Qwen3.8-27B-4bit"
    assert result["model"]["name"] == "mlx-community/Qwen3.8-27B-4bit"


def test_runtime_home_uses_cow_clone_relocation_and_retains_template(tmp_path, monkeypatch):
    template = _make_runtime_template(tmp_path)
    commands = []
    relocated = []

    monkeypatch.setattr(_HARNESS.platform, "system", lambda: "Darwin")

    def copy_runner(command, **_kwargs):
        commands.append(command)
        shutil.copytree(Path(command[2]), Path(command[3]), symlinks=True)
        return SimpleNamespace(returncode=0)

    def relocator(venv):
        relocated.append(venv)
        for script in venv.joinpath("bin").iterdir():
            if script.is_symlink():
                continue
            script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    isolated = _HARNESS._isolated_home(template, copy_runner=copy_runner, relocator=relocator)
    with isolated as clone:
        clone_root = clone.parent
        assert clone != template
        assert template.exists()
        assert commands[0][:2] == ["/bin/cp", "-cR"]
        assert relocated == [clone / ".unsloth" / "studio" / "unsloth_studio"]
        assert (clone / ".unsloth" / "studio" / "auth" / ".desktop_secret").exists()
        isolated.mark_cleanup_proven()
    assert template.exists()
    assert not clone_root.exists()


def test_runtime_home_is_retained_when_cleanup_is_not_proven(tmp_path, monkeypatch):
    template = _make_runtime_template(tmp_path)
    monkeypatch.setattr(_HARNESS.platform, "system", lambda: "Darwin")

    def copy_runner(command, **_kwargs):
        shutil.copytree(Path(command[2]), Path(command[3]), symlinks=True)
        return SimpleNamespace(returncode=0)

    def relocator(venv):
        for script in venv.joinpath("bin").iterdir():
            if script.is_symlink():
                continue
            script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    isolated = _HARNESS._isolated_home(template, copy_runner=copy_runner, relocator=relocator)
    clone = isolated.__enter__()
    clone_root = clone.parent
    isolated.__exit__(None, None, None)
    assert clone_root.exists()
    assert template.exists()
    shutil.rmtree(clone_root)


def test_secret_is_read_in_memory_without_unlinking(tmp_path):
    path = tmp_path / ".desktop_secret"
    path.write_text("desktop-secret-that-stays", encoding="utf-8")
    assert _HARNESS.read_secret_in_memory(path) == "desktop-secret-that-stays"
    assert path.read_text(encoding="utf-8") == "desktop-secret-that-stays"


def test_child_environment_is_allowlisted_and_caches_are_pinned(tmp_path):
    home = tmp_path / "clone"
    external_cache = tmp_path / "hf-cache"
    external_cache.mkdir()
    external_cache.chmod(0o555)
    try:
        environment = _HARNESS.build_offline_environment(
            home,
            {
                "PATH": "/safe/bin",
                "SECRET_TOKEN": "must-not-cross",
                "PYTHONPATH": "/real/project",
                "HF_HUB_CACHE": "/real/cache",
            },
            hf_model_cache=external_cache,
        )
        assert environment["HOME"] == str(home)
        assert environment["HF_HUB_CACHE"] == str(external_cache.resolve())
        assert environment["HF_HOME"].startswith(str(home))
        assert environment["TMPDIR"].startswith(str(home))
        assert environment["UV_CACHE_DIR"].startswith(str(home))
        assert "SECRET_TOKEN" not in environment
        assert "PYTHONPATH" not in environment
        assert "STUDIO_HOME" not in environment
        assert "UNSLOTH_STUDIO_HOME" not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
    finally:
        external_cache.chmod(0o755)


def test_unobservable_tool_suboperations_are_not_measured_as_zero():
    result = _HARNESS._extract_response_metrics(
        [],
        started=0.0,
        finished=0.1,
        first_event_ms=None,
        first_visible_ms=None,
        exposure="test",
    )
    quality = result["metrics"]["quality_and_work"]
    for name in ("prevented_tool_calls", "redundant_tool_calls", "deterministic_suboperations"):
        assert quality[name]["value"] is None
        assert quality[name]["provenance"] == "unavailable"
    assert list(_HARNESS.build_chat_payload("m", "p", seed=1, max_tokens=2)).count("top_k") == 1


def test_model_request_fallback_is_exact_absolute_path(tmp_path):
    model = (tmp_path / "models" / "mlx-27b").resolve()
    assert _HARNESS.select_model_request_name({}, model) == str(model)
    assert _HARNESS.select_model_request_name({"model": "reported"}, model) == "reported"


def test_sampler_peak_has_lifecycle_provenance_and_no_end_swap_alias():
    sampler = _HARNESS.RuntimeSampler(123, interval_s=0.1)
    sampler._samples = [
        (
            1.0,
            {"total_rss_bytes": {"value": 10, "provenance": "measured"}},
            {
                "memory_pressure_class": {"value": "normal"},
                "swap_used_bytes": {"value": 2},
                "compression_bytes": {"value": 3},
            },
        ),
        (
            2.0,
            {"total_rss_bytes": {"value": 30, "provenance": "measured"}},
            {
                "memory_pressure_class": {"value": "warning"},
                "swap_used_bytes": {"value": 5},
                "compression_bytes": {"value": 8},
            },
        ),
    ]
    summary = sampler.summary()
    assert summary["peak_process_or_unified_memory_bytes"]["value"] == 30
    assert summary["peak_process_or_unified_memory_bytes"]["provenance"] == "background_sampler"
    assert summary["swap_delta_bytes"]["value"] == 3
    assert "end_swap" not in summary


def test_macos_free_percentage_does_not_become_normal_pressure(monkeypatch):
    def runner(command, **_kwargs):
        if command[0] == "/usr/bin/memory_pressure":
            return SimpleNamespace(stdout="System-wide memory free percentage: 12%\n")
        if command[0] == "/usr/sbin/sysctl":
            return SimpleNamespace(stdout="vm.swapusage: total = 1G  used = 128M  free = 896M\n")
        if command[0] == "/usr/bin/vm_stat":
            return SimpleNamespace(
                stdout="page size of 4096 bytes\nPages occupied by compressor: 4\n"
            )
        raise AssertionError(command)

    result = _HARNESS.sample_macos_memory(command_runner=runner, system="darwin")
    assert result["memory_pressure_class"]["value"] is None
    assert result["memory_pressure_class"]["provenance"] == "unavailable"
    assert result["swap_used_bytes"]["value"] == 128 * 1024 * 1024
    assert result["compression_bytes"]["value"] == 4 * 4096


def test_macos_explicit_pressure_class_is_measured():
    def runner(command, **_kwargs):
        if command[0] == "/usr/bin/memory_pressure":
            return SimpleNamespace(stdout="Memory pressure: warning\n")
        return SimpleNamespace(stdout="")

    result = _HARNESS.sample_macos_memory(command_runner=runner, system="darwin")
    assert result["memory_pressure_class"]["value"] == "warning"
    assert result["memory_pressure_class"]["provenance"] == "measured"


@pytest.mark.parametrize(
    ("clear_listener", "protocol_ok", "root_identity_available", "lifecycle_ok"),
    [
        (True, True, True, True),
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
    ],
)
def test_execute_refreshes_owned_pids_retries_cleanup_and_retains_failed_clone(
    tmp_path, monkeypatch, clear_listener, protocol_ok, root_identity_available, lifecycle_ok
):
    template = _make_runtime_template(tmp_path)
    app = tmp_path / "frozen-control.app"
    app.mkdir()
    model = tmp_path / "mlx-model"
    model.mkdir()
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "mlx",
                "model_identity": {
                    "hugging_face_revision": "abc123",
                    "repository": "mlx-community/Qwen3.8-27B-4bit",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "helix-v3" / "run.json"
    monkeypatch.setattr(_HARNESS.platform, "system", lambda: "Darwin")

    def copy_runner(command, **_kwargs):
        shutil.copytree(Path(command[2]), Path(command[3]), symlinks=True)
        return SimpleNamespace(returncode=0)

    def relocator(venv):
        for script in venv.joinpath("bin").iterdir():
            if script.is_symlink():
                continue
            script.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    captured_home = []

    class Process:
        pid = 100

        def __init__(self):
            self.stdout = io.StringIO("TAURI_PORT=8888\n")
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def wait(self, **_kwargs):
            self.alive = False
            return 0

        def terminate(self):
            self.alive = False

        def kill(self):
            self.alive = False

    process = Process()

    def launch(_app, home, **_kwargs):
        captured_home.append(home)
        return process

    monkeypatch.setattr(_HARNESS, "_launch_app", launch)

    class Sampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def summary(self):
            return {
                "sample_count": 3,
                "memory_kind": "process_tree_rss",
                "peak_process_or_unified_memory_bytes": _HARNESS._metric(
                    30, provenance="background_sampler"
                ),
            }

    ps_calls = 0
    lsof_calls = 0

    def command_runner(command, **_kwargs):
        nonlocal ps_calls, lsof_calls
        if command[0] == "/bin/ps":
            ps_calls += 1
            if not root_identity_available and ps_calls == 1:
                return SimpleNamespace(stdout="")
            output_text = (
                "100 1 100 10 Sun Sep 20 10:00:00 2026 app\n"
                "102 100 100 20 Sun Sep 20 10:00:01 2026 worker\n"
                if ps_calls <= 11
                else "900 1 900 30 Sun Sep 20 10:00:02 2026 unrelated\n"
            )
            return SimpleNamespace(stdout=output_text)
        if command[0] == "/usr/sbin/lsof":
            lsof_calls += 1
            return SimpleNamespace(stdout="102\n" if lsof_calls == 1 or not clear_listener else "")
        raise AssertionError(command)

    terminated = []

    def terminate(process_arg, pids, **_kwargs):
        terminated.append(set(pids))
        process_arg.alive = False
        return {"attempted_pids": sorted(pids), "status": "pass"}

    monkeypatch.setattr(_HARNESS, "terminate_owned_processes", terminate)
    requested_models = []

    def stream(_transport, _base_url, _headers, *, model_name, case, **_kwargs):
        requested_models.append(model_name)
        expected = _HARNESS.expected_output_for_prompt(case.prompt) or ""
        return {
            "metrics": _HARNESS.unavailable_metric_vector(),
            "visible_output_chars": len(expected) if protocol_ok else 0,
            "visible_output_sha256": (
                _HARNESS._sha256_bytes(expected.encode("utf-8")) if protocol_ok else None
            ),
            "expected_output_exact_match": protocol_ok,
            "protocol": {
                "status": "pass" if protocol_ok else "fail",
                "reason": "mocked_terminal" if protocol_ok else "missing_done_marker",
            },
        }

    monkeypatch.setattr(_HARNESS, "stream_prompt", stream)

    identity = {
        "helix_backend_contract": _HARNESS.EXPECTED_BACKEND_CONTRACT,
        "helix_backend_tree_sha256": _HARNESS.EXPECTED_BACKEND_TREE_SHA256,
        "helix_backend_verified": True,
    }

    class Transport:
        def request_json(self, method, url, **kwargs):
            path = url[url.index("/api") :]
            if path == "/api/health":
                return identity
            if path == "/api/auth/desktop-login":
                assert kwargs["payload"]["secret"] == "desktop-clone-secret"
                return {"access_token": "token-for-test"}
            if path == "/api/inference/load":
                return {"status": "loaded"}
            if path in {"/api/inference/unload", "/api/shutdown"}:
                return {"status": "ok" if lifecycle_ok else "error"}
            raise AssertionError((method, path))

    result = _HARNESS.run_benchmark(
        _HARNESS.HarnessConfig(
            app_path=app,
            runtime_home_template=template,
            model_path=model,
            model_config_path=config_path,
            output_path=output,
            execute=True,
        ),
        repo_root=tmp_path,
        transport=Transport(),
        command_runner=command_runner,
        session_id_getter=_fixture_session_id,
        copy_runner=copy_runner,
        relocator=relocator,
        popen_factory=None,
        sampler_factory=Sampler,
    )
    expected_complete = clear_listener and protocol_ok and root_identity_available and lifecycle_ok
    assert result["status"] == ("complete" if expected_complete else "failed")
    assert result["frozen_control_identity"]["source_snapshot_sha256"] == _HARNESS.FROZEN_SOURCE_SNAPSHOT_SHA256
    assert result["frozen_control_identity"]["baseline_report_sha256"] == _HARNESS.FROZEN_BASELINE_REPORT_SHA256
    assert result["model"]["absolute_snapshot_path"] == str(model.resolve())
    assert result["model"]["hf_revision"] == "abc123"
    assert result["model"]["repository"] == "mlx-community/Qwen3.8-27B-4bit"
    assert result["model"]["name"] == "mlx-community/Qwen3.8-27B-4bit"
    assert result["configuration"]["prompt_suite_sha256"]
    expected_task_success = protocol_ok and root_identity_available
    assert result["task_success"]["status"] == (
        "pass" if expected_task_success else "fail"
    )
    if root_identity_available:
        assert result["runtime_sampler"]["memory_kind"] == "process_tree_rss"
        assert result["protocol_gate"]["status"] == ("pass" if protocol_ok else "fail")
        assert result["lifecycle_gate"]["status"] == ("pass" if lifecycle_ok else "fail")
        assert requested_models == [str(model.resolve()), str(model.resolve())]
        assert terminated and {100, 102}.issubset(terminated[0])
        assert result["cleanup"]["after_termination"]["status"] == (
            "pass" if clear_listener else "fail"
        )
    else:
        assert result["identity_capture"]["status"] == "fail"
        assert result["identity_capture"]["root_captured"] is False
        assert result["protocol_gate"]["status"] == "fail"
        assert requested_models == []
        assert terminated and terminated[0] == {100}
        assert result["cleanup"]["status"] == "fail"
    rendered = json.dumps(result, sort_keys=True)
    assert "desktop-clone-secret" not in rendered
    assert "token-for-test" not in rendered
    assert output.exists()
    assert captured_home
    clone_root = captured_home[0].parent
    if expected_complete:
        assert not clone_root.exists()
    else:
        assert clone_root.exists()
        shutil.rmtree(clone_root)
