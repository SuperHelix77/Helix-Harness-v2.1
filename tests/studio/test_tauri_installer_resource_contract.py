# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Tauri repair helpers from mixing package versions, and keep each desktop bundle carrying
only the installer it can actually run."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TAURI = REPO / "studio/src-tauri"


def _resources(config_name: str) -> dict:
    config = json.loads((TAURI / config_name).read_text(encoding = "utf-8"))
    return config.get("bundle", {}).get("resources", {})


def _bundled_resources(platform: str) -> dict:
    # Tauri merges tauri.<platform>.conf.json over tauri.conf.json for the target being built.
    merged = dict(_resources("tauri.conf.json"))
    merged.update(_resources(f"tauri.{platform}.conf.json"))
    return merged


def test_tauri_never_overlays_install_python_stack() -> None:
    for platform in ("windows", "linux", "macos"):
        resources = _bundled_resources(platform)
        assert not any(
            "install_python_stack.py" in path for item in resources.items() for path in item
        ), platform

    installer = (REPO / "install.ps1").read_text(encoding = "utf-8")
    assert "Overlay Tauri-bundled studio fixes" not in installer
    assert (
        '"install_python_stack.py" = "Lib\\site-packages\\studio\\install_python_stack.py"'
        not in installer
    )


def test_each_bundle_ships_only_the_installer_it_runs() -> None:
    # resolve_install_script picks install.sh on unix and install.ps1 elsewhere, so the other was dead weight in every
    # bundle, and the largest script body a classifier walking the AppImage finds, which is where
    # Trojan:Script/Wacatac.B!ml landed.
    assert _bundled_resources("windows") == {"../../install.ps1": "install.ps1"}
    assert _bundled_resources("linux") == {"../../install.sh": "install.sh"}
    assert _bundled_resources("macos") == {
        "../../install.sh": "install.sh",
        "artifacts/helix-backend": "helix-backend",
        "../../scripts/apply_helix_backend_overlay.py": "apply_helix_backend_overlay.py",
    }



def test_macos_bundle_carries_the_exact_helix_backend_contract() -> None:
    resources = _bundled_resources("macos")
    assert resources.get("artifacts/helix-backend") == "helix-backend"
    tool_policy = (REPO / "studio/backend/state/tool_policy.py").read_text(encoding="utf-8")
    assert 'HELIX_HARNESS_BACKEND_CONTRACT = "helix.adaptive.backend.v1"' in tool_policy
    install_rs = (REPO / "studio/src-tauri/src/install.rs").read_text(encoding="utf-8")
    assert 'HELIX_HARNESS_BACKEND_OVERLAY' in install_rs
    assert resources.get("../../scripts/apply_helix_backend_overlay.py") == "apply_helix_backend_overlay.py"
    installer = (REPO / "install.sh").read_text(encoding="utf-8")
    assert '_apply_helix_backend_overlay' in installer
    applier = (REPO / "scripts/apply_helix_backend_overlay.py").read_text(encoding="utf-8")
    assert 'helix.adaptive.backend.v1' in applier
    assert 'mem0ai==2.0.20' in applier
    assert 'qdrant-client==1.19.1' in applier
    assert 'optional memory runtime unavailable' in applier
    assert '"-m", "pip", "install"' not in applier
    update_rs = (REPO / "studio/src-tauri/src/update.rs").read_text(encoding="utf-8")
    assert 'apply_bundled_helix_backend_overlay' in update_rs


def test_no_installer_resource_leaks_through_the_shared_config() -> None:
    # A resource in the shared config lands in every bundle, which is how the split regresses.
    assert _resources("tauri.conf.json") == {}


def test_windows_upgrade_removes_the_installer_it_no_longer_ships() -> None:
    # NSIS writes the current resource manifest and deletes nothing, and the uninstaller deletes only what is in that
    # manifest.
    # An in-place upgrade from a release that bundled both installers would therefore keep install.sh on a Windows
    # machine forever, and the non-recursive RMDir "$INSTDIR" would fail at uninstall.
    hooks = (REPO / "studio/src-tauri/windows/hooks.nsh").read_text(encoding = "utf-8")
    for macro in ("NSIS_HOOK_PREINSTALL", "NSIS_HOOK_PREUNINSTALL"):
        assert f"!macro {macro}" in hooks, f"hooks.nsh must define {macro}"
        body = hooks.split(f"!macro {macro}", 1)[1].split("!macroend", 1)[0]
        assert (
            'Delete "$INSTDIR\\install.sh"' in body
        ), f"{macro} must remove the install.sh a pre-split release left behind"
