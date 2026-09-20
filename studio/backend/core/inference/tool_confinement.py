# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Confine managed-account tool procs: Landlock (Linux), sandbox-exec (macOS), else refuse."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import stat
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from core.inference.tool_capabilities import (
    ToolCapability,
    ToolCapabilityGrant,
    valid_local_execution_grant,
)
from utils.account_context import is_owner_context

_OVERRIDE_ENV = "UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS"

# Landlock syscall numbers are architecture independent.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_RULE_NET_PORT = 2
_PR_SET_NO_NEW_PRIVS = 38

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13  # ABI 2
_FS_TRUNCATE = 1 << 14  # ABI 3
_FS_IOCTL_DEV = 1 << 15  # ABI 5
_FS_RESOLVE_UNIX = 1 << 16  # ABI 9
_NET_BIND_TCP = 1 << 0  # ABI 4
_NET_CONNECT_TCP = 1 << 1  # ABI 4
_NET_BIND_UDP = 1 << 2  # ABI 10
_NET_CONNECT_SEND_UDP = 1 << 3  # ABI 10
_NET_BIND_MASK = _NET_BIND_TCP | _NET_BIND_UDP
_NET_ACCESS_MASK = _NET_BIND_MASK | _NET_CONNECT_TCP | _NET_CONNECT_SEND_UDP
_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0  # ABI 6
_SCOPE_SIGNAL = 1 << 1  # ABI 6: signals confined to the same domain
_FS_ABI1_MASK = (1 << 13) - 1

_SYSTEM_READ_ROOTS = (
    "/usr",
    "/lib",
    "/lib32",
    "/lib64",
    "/bin",
    "/sbin",
    "/etc",
    "/opt",
    "/run",
    "/snap",
    "/nix",
    "/var/lib",
    "/proc",
    "/sys",
)
# Pseudo-devices every tool needs, then the accelerator nodes. The rest of /dev stays out: other
# sessions' ptys, same-UID shm objects and, for a root-run Studio, every block and character device.
_DEVICE_NODES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty")
_ACCELERATOR_NODES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
    "/dev/nvidia-caps",
    "/dev/dri",
    "/dev/kfd",
)

# Readable by the service user, so DAC alone does not stop a tool. The rest of /run is readable.
_PRIVATE_RUNTIME_ROOTS = ("/run/user", "/run/secrets", "/run/credentials")
# Secrets a root-run Studio (the Docker image) could otherwise hand a managed tool; the rest of /etc
# (ld.so.cache, nsswitch, hosts, resolv.conf, ssl/certs) is what tool processes need.
_PRIVATE_SYSTEM_PATHS = (
    "/etc/shadow",
    "/etc/shadow-",
    "/etc/gshadow",
    "/etc/gshadow-",
    "/etc/security/opasswd",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssl/private",
    "/etc/pki/tls/private",
    "/etc/letsencrypt",
    "/etc/krb5.keytab",
    "/etc/ipsec.secrets",
    "/etc/docker/key.json",
    "/etc/ssh/ssh_host_*_key",
)


def _private_system_paths() -> list[str]:
    import glob

    out: list[str] = []
    for pattern in _PRIVATE_SYSTEM_PATHS:
        out.extend(sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern])
    return out


class ToolConfinementUnavailable(RuntimeError):
    """This host cannot confine a managed account's tool process."""


@dataclass(frozen = True)
class Confinement:
    """``preexec`` runs in the forked child (Linux); ``wrap`` rewrites argv (macOS)."""

    mechanism: str
    preexec: Optional[Callable[[], None]] = None
    wrapper: tuple[str, ...] = ()

    def wrap(self, argv: list[str]) -> list[str]:
        return [*self.wrapper, *argv] if self.wrapper else argv


def unconfined_tools_allowed() -> bool:
    return (os.environ.get(_OVERRIDE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def refusal_message() -> str:
    return (
        "Code execution is unavailable for this account: this host cannot confine tool "
        "processes to your workspace (Landlock ABI 10 on Linux, sandbox-exec "
        f"on macOS). The installation owner can set {_OVERRIDE_ENV}=1 to allow unconfined "
        "tool processes for managed accounts."
    )


def _existing(paths) -> list[str]:
    seen: list[str] = []
    for raw in paths:
        if not raw:
            continue
        try:
            path = os.path.realpath(raw)
        except (OSError, ValueError):
            continue
        if os.path.exists(path) and path not in seen:
            seen.append(path)
    return seen


def _interpreter_roots() -> list[str]:
    return _existing(
        (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            getattr(sys, "base_exec_prefix", ""),
            os.path.dirname(sys.executable),
            os.environ.get("VIRTUAL_ENV", ""),
        )
    )


def _ensure_dirs(paths) -> list[str]:
    """Landlock rules need existing paths; ``ensure_dir`` refuses a launch racing account delete."""
    from utils.paths.storage_roots import ensure_dir

    roots = []
    for root in paths:
        try:
            ensure_dir(Path(root))
        except OSError:
            continue
        roots.append(str(root))
    return _existing(roots)


def _readable_account_roots() -> list[str]:
    """Account workspace: readable, never writable, else a tool could rewrite its own grants."""
    from utils.paths.storage_roots import workspace_root
    return _ensure_dirs((workspace_root(),))


def _writable_roots() -> list[str]:
    from core.inference.tools import sandbox_root
    from utils.paths.storage_roots import project_workspaces_root, tmp_root
    return _ensure_dirs((sandbox_root(), tmp_root(), project_workspaces_root()))


def _hf_cache_roots() -> tuple[str, ...]:
    """Install-wide HF cache: an ancestor read grant would leak other accounts' repos and tokens."""
    try:
        from utils.hf_cache_settings import known_hf_cache_homes, known_hf_hub_caches
        return tuple(str(p) for p in (*known_hf_cache_homes(), *known_hf_hub_caches()))
    except Exception:
        return ()


def _protected_roots() -> list[str]:
    from core.inference.tools import shared_sandbox_root
    from utils.paths.storage_roots import (
        shared_project_workspaces_root,
        shared_tmp_root,
        studio_root,
    )
    return _with_shared_bases(
        _existing((studio_root(),)),
        (
            shared_sandbox_root(),
            shared_project_workspaces_root(),
            shared_tmp_root(),
            *_hf_cache_roots(),
        ),
    )


def _contains(ancestor: str, path: str) -> bool:
    return path == ancestor or path.startswith(ancestor.rstrip(os.sep) + os.sep)


def _with_shared_bases(roots: list[str], bases) -> list[str]:
    out = list(roots)
    for base in _existing(bases):
        if not any(_contains(root, base) for root in out):
            out.append(base)
    return out


def _grant_excluding(
    path: str, access: int, protected: list[str], rules: list[tuple[str, int]]
) -> None:
    """Landlock has no deny rule, so grant an ancestor child by child."""
    inside = [p for p in protected if _contains(path, p)]
    if not inside:
        # A plain-file rule may not carry directory rights.
        rules.append((path, access if os.path.isdir(path) else access & ~_FS_READ_DIR))
        return
    if any(p == path for p in inside):
        return
    try:
        children = sorted(os.listdir(path))
    except OSError:
        return
    for name in children:
        child = os.path.join(path, name)
        if not os.path.exists(child):
            # A dangling link cannot carry a rule.
            continue
        if os.path.islink(child):
            # A link opens as its target, so one near a protected root would grant that tree.
            target = os.path.realpath(child)
            if any(_contains(p, target) or _contains(target, p) for p in protected):
                continue
            # Store the resolved object rather than the link spelling: prepared rules
            # deliberately reject symlinks, including a path swapped after this scan.
            if os.path.exists(target):
                _grant_excluding(target, access, protected, rules)
            continue
        _grant_excluding(child, access, protected, rules)


class _RulesetAttr(ctypes.Structure):
    # Later ABIs accept this ABI 1 size, zeroing the rest.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _ScopedRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _NetPortAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


@dataclass(frozen = True)
class _PreparedLandlockRule:
    """A path rule pinned to the object observed in the parent before ``fork``."""

    path: str
    access: int
    st_dev: int
    st_ino: int
    st_mode: int


_libc = None
if sys.platform == "linux":
    try:
        import ctypes.util
        _name = ctypes.util.find_library("c")
        _libc = ctypes.CDLL(_name, use_errno = True) if _name else None
    except (OSError, AttributeError):
        _libc = None

_landlock_abi: Optional[int] = None


def landlock_abi() -> int:
    """Highest Landlock ABI of the running kernel, 0 when unavailable."""
    global _landlock_abi
    if _landlock_abi is not None:
        return _landlock_abi
    abi = 0
    if sys.platform == "linux" and _libc is not None:
        try:
            got = _libc.syscall(
                _SYS_LANDLOCK_CREATE_RULESET, None, 0, _LANDLOCK_CREATE_RULESET_VERSION
            )
            abi = int(got) if got > 0 else 0
        except (OSError, AttributeError, ValueError):
            abi = 0
    _landlock_abi = abi
    return abi


def _handled_mask(abi: int) -> int:
    mask = _FS_ABI1_MASK
    if abi >= 2:
        mask |= _FS_REFER
    if abi >= 3:
        mask |= _FS_TRUNCATE
    if abi >= 5:
        mask |= _FS_IOCTL_DEV
    if abi >= 9:
        mask |= _FS_RESOLVE_UNIX
    return mask


def _device_nodes() -> list[str]:
    import glob
    return _existing((*_DEVICE_NODES, *_ACCELERATOR_NODES, *sorted(glob.glob("/dev/nvidia[0-9]*"))))


def _landlock_rules(abi: int, sandbox_site_dir: str) -> list[tuple[str, int]]:
    handled = _handled_mask(abi)
    read = _FS_READ_FILE | _FS_READ_DIR | _FS_EXECUTE
    device = _FS_READ_FILE | _FS_WRITE_FILE | (_FS_IOCTL_DEV if abi >= 5 else 0)
    rules: list[tuple[str, int]] = []
    writable_roots = _writable_roots()
    protected = _protected_roots()
    # /dev is protected too, so a system-root link such as /run/shm cannot grant a device tree.
    system_protected = _with_shared_bases(
        protected, (*_PRIVATE_RUNTIME_ROOTS, "/dev", *_private_system_paths())
    )
    for path in _existing(_SYSTEM_READ_ROOTS):
        _grant_excluding(path, read, system_protected, rules)
    for path in _interpreter_roots():
        _grant_excluding(path, read, protected, rules)
    for path in _existing((sandbox_site_dir,)):
        _grant_excluding(path, read, protected, rules)
    for path in _readable_account_roots():
        rules.append((path, read))
    for path in _device_nodes():
        rules.append((path, device | (_FS_READ_DIR if os.path.isdir(path) else 0)))
    # No symlink creation: the server follows links, so a tool must not plant an escaping one.
    writable = handled & ~_FS_MAKE_SYM
    for path in writable_roots:
        rules.append((path, writable))
    return rules


def _capture_landlock_rules(rules: list[tuple[str, int]]) -> list[_PreparedLandlockRule]:
    """Capture rule-path identities before ``fork`` and refuse ambiguous paths."""
    prepared: list[_PreparedLandlockRule] = []
    for path, access in rules:
        try:
            observed = os.stat(path, follow_symlinks = False)
        except (OSError, ValueError) as exc:
            raise ToolConfinementUnavailable(
                f"Code execution refused: cannot identify Landlock rule path {path!r}."
            ) from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ToolConfinementUnavailable(
                f"Code execution refused: Landlock rule path is a symlink: {path!r}."
            )
        prepared.append(
            _PreparedLandlockRule(
                path = path,
                access = access,
                st_dev = observed.st_dev,
                st_ino = observed.st_ino,
                st_mode = observed.st_mode,
            )
        )
    return prepared


def _open_landlock_rule_fd(rule: _PreparedLandlockRule) -> int:
    """Open exactly the captured object, failing closed on replacement or a link."""
    flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(rule.path, flags)
    except (OSError, ValueError) as exc:
        raise ToolConfinementUnavailable(
            f"Code execution refused: Landlock rule path changed: {rule.path!r}."
        ) from exc
    try:
        current = os.fstat(parent_fd)
        identity = (current.st_dev, current.st_ino)
        expected = (rule.st_dev, rule.st_ino)
        if (
            identity != expected
            or stat.S_ISLNK(current.st_mode)
            or stat.S_IFMT(current.st_mode) != stat.S_IFMT(rule.st_mode)
        ):
            raise ToolConfinementUnavailable(
                f"Code execution refused: Landlock rule path was replaced: {rule.path!r}."
            )
        return parent_fd
    except Exception:
        os.close(parent_fd)
        raise


def _landlock_preexec(
    handled: int,
    rules: list[_PreparedLandlockRule],
    scoped: int = 0,
    handled_access_net: int = 0,
    network_rules: tuple[tuple[int, int], ...] = (),
) -> None:
    """Apply the already-prepared rules in the forked child without importing."""
    libc = _libc
    attr = (
        _ScopedRulesetAttr(handled, handled_access_net, scoped)
        if scoped or handled_access_net
        else _RulesetAttr(handled)
    )
    ruleset_fd = libc.syscall(
        _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for rule in rules:
            parent_fd = _open_landlock_rule_fd(rule)
            try:
                beneath = _PathBeneathAttr(rule.access & handled, parent_fd)
                rc = libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(beneath),
                    0,
                )
                if rc < 0:
                    raise OSError(
                        ctypes.get_errno(), f"landlock_add_rule failed for {rule.path}"
                    )
            finally:
                os.close(parent_fd)
        for access, port in network_rules:
            net_port = _NetPortAttr(access & handled_access_net, port)
            rc = libc.syscall(
                _SYS_LANDLOCK_ADD_RULE,
                ruleset_fd,
                _LANDLOCK_RULE_NET_PORT,
                ctypes.byref(net_port),
                0,
            )
            if rc < 0:
                raise OSError(
                    ctypes.get_errno(), f"landlock_add_rule failed for network port {port}"
                )
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if libc.syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


# ABI 10 adds UDP rights. ABI 6-9 can scope signals and TCP but cannot provide the
# broker's offline or outbound-only UDP boundary, so they are refused.
_MIN_LANDLOCK_ABI = 10


def _linux_confinement(
    sandbox_site_dir: str,
    *,
    allow_network: bool = False,
) -> Optional[Confinement]:
    abi = landlock_abi()
    if abi < _MIN_LANDLOCK_ABI:
        return None
    handled = _handled_mask(abi)
    rules = _capture_landlock_rules(_landlock_rules(abi, sandbox_site_dir))
    handled_access_net = _NET_BIND_MASK if allow_network else _NET_ACCESS_MASK
    # Handling BIND_UDP also covers implicit autobind before sendto(). Port 0 is
    # the kernel's documented wildcard for ephemeral autobind; nonzero binds stay denied.
    network_rules = ((_NET_BIND_UDP, 0),) if allow_network else ()
    return Confinement(
        mechanism = f"landlock-abi{abi}",
        preexec = partial(
            _landlock_preexec,
            handled,
            rules,
            _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL,
            handled_access_net = handled_access_net,
            network_rules = network_rules,
        ),
    )


def _sbpl(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _darwin_user_cache_dirs() -> tuple[str, ...]:
    try:
        cache = (os.confstr("CS_DARWIN_USER_CACHE_DIR") or "").rstrip(os.sep)
    except (AttributeError, ValueError, OSError):
        return ()
    if not cache:
        return ()
    bare = cache[len("/private") :] if cache.startswith("/private/") else cache
    return (bare, "/private" + bare)


def macos_profile(
    *,
    read_roots: list[str],
    hidden_roots: list[str],
    writable_roots: list[str],
    account_read_roots: list[str] = (),
    exact_hidden_roots: list[str] = (),
    allow_network: bool = False,
) -> str:
    """sandbox-exec profile; later rules win, so account roots outrank the install deny."""
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
        "(allow file-read-metadata)",
        # Pseudo-devices only; the rest of /dev (other sessions' ttys, disks) stays denied.
        '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random")'
        ' (literal "/dev/urandom") (literal "/dev/tty") (literal "/dev/dtracehelper"))',
        '(allow file-ioctl (literal "/dev/tty") (literal "/dev/dtracehelper"))',
        '(allow file-read* (subpath "/dev/fd"))',
        # No shared /private/tmp: the account tmp root is granted with the writable roots.
        '(allow file-read* (subpath "/private/var/db"))',
        # The per-user darwin tree holds every account tmp root: deny it, keep dyld's cache dir.
        '(deny file-read* file-write* (subpath "/private/var/folders") (subpath "/var/folders"))',
        *(f"(allow file-read* (subpath {_sbpl(path)}))" for path in _darwin_user_cache_dirs()),
    ]
    if allow_network:
        lines.insert(8, "(allow network-outbound)")
    for path in read_roots:
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
    for path in hidden_roots:
        lines.append(f"(deny file-read* file-write* (subpath {_sbpl(path)}))")
    for path in read_roots:
        if not any(_contains(root, path) and root != path for root in hidden_roots):
            continue
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
        for root in hidden_roots:
            if _contains(path, root) and root != path:
                lines.append(f"(deny file-read* file-write* (subpath {_sbpl(root)}))")
        # These roots may have been collapsed under a broader hidden ancestor. Keep
        # their exact denies after an interpreter subtree is allowed back.
        for root in exact_hidden_roots:
            if _contains(path, root):
                lines.append(f"(deny file-read* file-write* (subpath {_sbpl(root)}))")
    for path in account_read_roots:
        lines.append(f"(allow file-read* (subpath {_sbpl(path)}))")
    for path in writable_roots:
        lines.append(f"(allow file-read* file-write* (subpath {_sbpl(path)}))")
    # file-write* covers symlinks; deny last (later rules win), matching Landlock's held MAKE_SYM.
    lines.append("(deny file-write-create (vnode-type SYMLINK))")
    return "\n".join(lines) + "\n"


def _macos_confinement(
    sandbox_site_dir: str,
    *,
    allow_network: bool,
) -> Optional[Confinement]:
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return None
    from core.inference.tools import shared_sandbox_root
    from utils.paths.storage_roots import (
        shared_project_workspaces_root,
        shared_tmp_root,
        studio_root,
    )

    read_roots = _existing(
        (
            "/usr",
            "/bin",
            "/sbin",
            "/etc",
            "/private/etc",
            "/System",
            "/Library",
            "/opt",
            "/Applications",
            *_interpreter_roots(),
            sandbox_site_dir,
        )
    )
    writable_roots = _writable_roots()
    exact_hidden_roots = _existing(
        (
            shared_sandbox_root(),
            str(shared_project_workspaces_root()),
            str(shared_tmp_root()),
            *_hf_cache_roots(),
        )
    )
    hidden_roots = _with_shared_bases(
        _existing((str(studio_root()), str(shared_tmp_root()), os.path.expanduser("~"))),
        exact_hidden_roots,
    )
    profile = macos_profile(
        read_roots = read_roots,
        hidden_roots = hidden_roots,
        account_read_roots = _readable_account_roots(),
        writable_roots = writable_roots,
        exact_hidden_roots = exact_hidden_roots,
        allow_network = allow_network,
    )
    return Confinement(mechanism = "sandbox-exec", wrapper = (sandbox_exec, "-p", profile))


def _macos_owner_confinement(*, allow_network: bool) -> Optional[Confinement]:
    """Stable OS network capability boundary for the installation owner.

    Current macOS aborts ordinary Python/bash processes under the historical
    managed-account deny-default profile before user code starts. Owner mode was
    previously completely unconfined, so preserve ordinary filesystem/process
    behavior while removing ambient networking unless this call positively
    requests it.
    """
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return None
    profile = "(version 1)\n(allow default)\n(deny network*)\n"
    if allow_network:
        profile += "(allow network-outbound)\n"
    return Confinement(mechanism="sandbox-exec", wrapper=(sandbox_exec, "-p", profile))


def account_confinement(
    sandbox_site_dir: str,
    *,
    capability_grant: ToolCapabilityGrant,
    tool_name: str,
    payload: str,
) -> Optional[Confinement]:
    """Confinement for the next sandboxed tool child.

    macOS owner tools are confined too; explicit Bypass Permissions skips this
    function at the caller. Managed accounts continue to fail closed when the
    host has no supported confinement mechanism. Privileged options are derived
    only from a broker-issued grant bound to this exact account/tool/payload.
    """
    if not valid_local_execution_grant(
        capability_grant,
        tool_name=tool_name,
        payload=payload,
    ):
        raise ToolConfinementUnavailable(
            "Code execution refused: invalid or stale tool capability grant."
        )
    allow_network = capability_grant.allows(ToolCapability.NETWORK_EGRESS)
    owner = is_owner_context()
    confinement = None
    if sys.platform == "linux":
        confinement = _linux_confinement(sandbox_site_dir, allow_network = allow_network)
    elif sys.platform == "darwin":
        confinement = (
            _macos_owner_confinement(allow_network=allow_network)
            if owner
            else _macos_confinement(
                sandbox_site_dir,
                allow_network=allow_network,
            )
        )
    if confinement is not None:
        return confinement
    if unconfined_tools_allowed():
        return Confinement(mechanism = "unconfined-by-owner")
    raise ToolConfinementUnavailable(refusal_message())
