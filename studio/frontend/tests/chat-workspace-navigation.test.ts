import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const panel = readSrc("features/chat/components/chat-workspace-panel.tsx");
const chatPage = readSrc("features/chat/chat-page.tsx");

test("workspace has explicit back navigation and Escape closes the visible panel", () => {
  assert.match(panel, /Back to chat/);
  assert.match(panel, /aria-label="Back to chat"/);
  assert.match(panel, /event\.key !== "Escape"/);
  assert.match(panel, /event\.defaultPrevented/);
  assert.match(panel, /window\.addEventListener\("keydown", closeOnEscape\)/);
  assert.match(panel, /onClose\(\)/);
});

test("workspace tabs expose a complete keyboard-accessible tabs relationship", () => {
  assert.match(panel, /role="tablist"/);
  assert.match(panel, /aria-orientation="horizontal"/);
  assert.match(panel, /role="tab"/);
  assert.match(panel, /aria-controls=\{`workspace-panel-\$\{item\.id\}`\}/);
  assert.match(panel, /tabIndex=\{tab === item\.id \? 0 : -1\}/);
  assert.match(panel, /event\.key === "ArrowRight"/);
  assert.match(panel, /event\.key === "ArrowLeft"/);
  assert.match(panel, /event\.key === "Home"/);
  assert.match(panel, /event\.key === "End"/);
  assert.match(panel, /role="tabpanel"/);
  assert.match(panel, /aria-labelledby=\{`workspace-tab-\$\{tab\}`\}/);
});

test("workspace open state is shared with the toolbar and cleared by context takeover", () => {
  assert.match(chatPage, /const workspaceScope = JSON\.stringify\(\[/);
  assert.match(chatPage, /search\.compare \?\? null/);
  assert.match(chatPage, /search\.new \?\? null/);
  assert.match(chatPage, /search\.project \?\? null/);
  assert.match(chatPage, /search\.thread \?\? null/);
  assert.match(
    chatPage,
    /workspaceState\.scope === workspaceScope && workspaceState\.open/,
  );
  assert.match(chatPage, /workspaceOpen=\{workspaceOpen\}/);
  assert.match(chatPage, /onWorkspaceOpenChange=\{setWorkspaceOpen\}/);
  assert.match(chatPage, /aria-pressed=\{workspaceOpen\}/);
  assert.match(
    chatPage,
    /if \(workspaceOpen\) \{\s*setWorkspaceOpen\(false\);\s*return;\s*\}/,
  );
  assert.match(
    chatPage,
    /if \(showResearchPanel \|\| showArtifactPanel \|\| showPlanPanel\) \{[\s\S]*if \(workspaceOpen\) onWorkspaceOpenChange\(false\);/,
  );
  assert.doesNotMatch(chatPage, /openChatWorkspace|CHAT_WORKSPACE_OPEN_EVENT/);
});

test("persistent Chat mount closes manual rails across route and chat scope changes", () => {
  assert.match(
    chatPage,
    /if \(!active\) \{\s*setWorkspaceState\(\{ scope: workspaceScope, open: false \}\);\s*setHelixRailState\(\{ scope: workspaceScope, panel: null \}\);/,
  );
  assert.match(
    chatPage,
    /state\.scope === workspaceScope\s*\? state\s*: \{ scope: workspaceScope, open: false \}/,
  );
  assert.match(
    chatPage,
    /state\.scope === workspaceScope\s*\? state\s*: \{ scope: workspaceScope, panel: null \}/,
  );
  assert.match(chatPage, /\}, \[active, workspaceScope\]\);/);
});

test("persistent Chat mount clears route-owned overlays instead of resurfacing them", () => {
  assert.match(chatPage, /const transientUiScopeRef = useRef\(workspaceScope\)/);
  assert.match(
    chatPage,
    /const scopeChanged = transientUiScopeRef\.current !== workspaceScope;[\s\S]*if \(active && !scopeChanged\) return;/,
  );
  for (const close of [
    "setModelSelectorOpen(false)",
    "setModelSelectorLocked(false)",
    "setProjectPickerOpen(false)",
    "setSettingsOpen(false)",
    "closeResearchPanel()",
    "closeArtifactSurface()",
    'useChatRuntimeStore.getState().setChatMode("normal")',
  ]) {
    assert.ok(chatPage.includes(close), `${close} no longer resets with Chat scope`);
  }
});
