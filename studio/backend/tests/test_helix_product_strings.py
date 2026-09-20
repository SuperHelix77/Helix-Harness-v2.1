# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_product_is_helix_harness_and_credits_unsloth():
    index = (ROOT / "studio/frontend/index.html").read_text(encoding="utf-8")
    en = (ROOT / "studio/frontend/src/i18n/locales/en.ts").read_text(encoding="utf-8")
    credits = (ROOT / "CREDITS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Helix Harness" in index
    assert 'product: "Helix Harness"' in en
    assert "Unsloth Studio" in credits
    assert "AGPL" in credits
    assert "chat-first" in readme.lower()
    assert "standalone Helix-branded Tauri desktop app" in readme


def test_train_and_media_tabs_are_filtered_from_sidebar():
    sidebar = (ROOT / "studio/frontend/src/components/app-sidebar.tsx").read_text(encoding="utf-8")
    assert 'item.id !== "train"' in sidebar
    assert 'item.id !== "images"' in sidebar
    assert 'item.id !== "video"' in sidebar


def test_v11_knobs_remain_in_speed_policy():
    policy = (ROOT / "studio/backend/core/inference/helix_speed_policy.py").read_text(encoding="utf-8")
    q38 = (ROOT / "studio/backend/core/inference/q38_v11_optimization.py").read_text(encoding="utf-8")
    assert "dflash" in policy
    assert "helix_gdn_host_path" in policy
    assert "65536" in q38 or "65_536" in q38
    assert "q4_0" in q38
    assert "kv-unified" in q38


def test_tauri_mac_app_is_named_helix_harness():
    conf = (ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    plist = (ROOT / "studio/src-tauri/Info.plist").read_text(encoding="utf-8")
    assert '"productName": "Helix Harness"' in conf
    assert '"identifier": "ai.helix.harness"' in conf
    assert '"title": "Helix Harness"' in conf
    assert "Helix Harness uses the microphone" in plist


def test_theme_engine_has_glass_and_smart_ink():
    css = (ROOT / "studio/frontend/src/index.css").read_text(encoding="utf-8")
    theme = (ROOT / "studio/frontend/src/features/settings/stores/theme-store.ts").read_text(
        encoding="utf-8"
    )
    ink = (ROOT / "studio/frontend/src/features/settings/stores/contrast-ink.ts").read_text(
        encoding="utf-8"
    )
    assert 'data-palette="glass"' in css
    assert "backdrop-filter" in css
    assert 'data-ink="light"' in css
    assert '"glass"' in theme
    assert "relativeLuminance" in ink
    welcome = (ROOT / "studio/frontend/src/components/assistant-ui/thread.tsx").read_text(
        encoding="utf-8"
    )
    assert "unsloth-welcome-sloth" not in welcome


def test_sidebar_uses_helix_branding_and_engine_is_chat_scoped():
    sidebar = (ROOT / "studio/frontend/src/components/app-sidebar.tsx").read_text(encoding="utf-8")
    assert "HELIX HARNESS" in sidebar
    assert "helix-mark.svg" not in sidebar
    assert "HELIX ENGINE" not in sidebar
    assert 'to: "/engine"' not in sidebar
    assert "circle-logo-small.png" not in sidebar
    engine = (ROOT / "studio/frontend/src/features/helix-engine/engine-page.tsx").read_text(
        encoding="utf-8"
    )
    assert "data-testid=\"helix-engine-compatibility-page\"" in engine
    assert "Workflow and Execution Graph controls in the chat header" in engine


def test_live_feed_gui_polls_execute_tool_session_and_hub_loads():
    panel = (ROOT / "studio/frontend/src/features/chat/components/chat-workspace-panel.tsx").read_text(
        encoding="utf-8"
    )
    hub_route = (ROOT / "studio/frontend/src/app/routes/hub.tsx").read_text(encoding="utf-8")
    assert "live-feed?session_id=default" not in panel
    assert "sandboxSessionIdFor" in panel
    assert "liveSessionId" in panel
    assert "@/features/hub/hub-page" in hub_route
