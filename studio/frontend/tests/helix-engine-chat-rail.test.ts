// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const chatPage = readSrc("features/chat/chat-page.tsx");
const panels = readSrc("features/helix-engine/engine-panels.tsx");
const appSidebar = readSrc("components/app-sidebar.tsx");

test("chat header exposes mutually exclusive Workflow and Execution Graph toggles", () => {
  assert.match(chatPage, /type HelixRailPanel = "workflow" \| "execution" \| null/);
  assert.match(chatPage, /helixRailState\.scope === workspaceScope/);
  assert.match(chatPage, /aria-pressed=\{helixRailPanel === "workflow"\}/);
  assert.match(chatPage, /aria-pressed=\{helixRailPanel === "execution"\}/);
  assert.match(chatPage, /Open Helix Workflow/);
  assert.match(chatPage, /Open Execution Graph/);
  assert.match(chatPage, /setWorkspaceOpen\(false\);\s*setHelixRailPanel\("workflow"\)/);
  assert.match(chatPage, /setWorkspaceOpen\(false\);\s*setHelixRailPanel\("execution"\)/);
  assert.match(chatPage, /setHelixRailPanel\(null\);\s*setWorkspaceOpen\(true\)/);
});

test("Helix panels share the existing right context rail and yield to higher-priority surfaces", () => {
  assert.match(chatPage, /showHelixWorkflowPanel/);
  assert.match(chatPage, /showHelixExecutionPanel/);
  assert.match(chatPage, /showContextPanel =[\s\S]*showHelixWorkflowPanel[\s\S]*showHelixExecutionPanel/);
  assert.match(chatPage, /collapsible=\{[\s\S]*showHelixWorkflowPanel[\s\S]*showHelixExecutionPanel/);
  assert.match(chatPage, /<HelixWorkflowPanel[\s\S]*onClose=\{\(\) => onHelixRailPanelChange\(null\)\}/);
  assert.match(chatPage, /<HelixExecutionGraphPanel[\s\S]*onClose=\{\(\) => onHelixRailPanelChange\(null\)\}/);
  assert.match(
    chatPage,
    /if \(showResearchPanel \|\| showArtifactPanel \|\| showPlanPanel\) \{[\s\S]*if \(helixRailPanel\) onHelixRailPanelChange\(null\);/,
  );
});

test("Helix right-rail panels have explicit close and Escape semantics", () => {
  assert.match(panels, /aria-label="Back to chat"/);
  assert.match(panels, /event\.key !== "Escape"/);
  assert.match(panels, /event\.defaultPrevented/);
  assert.match(panels, /window\.addEventListener\("keydown", closeOnEscape\)/);
  assert.match(panels, /data-testid=\{`helix-\$\{mode\}-panel`\}/);
});

test("the left sidebar keeps Helix Harness primary while exposing Engine as a secondary control plane", () => {
  assert.match(appSidebar, />\s*HELIX HARNESS\s*</);
  assert.match(appSidebar, /navigate\(\{ to: "\/engine" \}\)/);
  assert.match(appSidebar, /isActive=\{pathname === "\/engine"\}/);
  assert.match(appSidebar, />\s*HELIX ENGINE\s*</);
  assert.match(appSidebar, />\s*in-app control plane\s*</);
});
