// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const panels = readSrc("features/helix-engine/engine-panels.tsx");
const page = readSrc("features/helix-engine/engine-page.tsx");

test("Helix right-rail panels preserve execution graph and workflow controls", () => {
  assert.match(panels, /export function HelixExecutionGraphPanel/);
  assert.match(panels, /export function HelixWorkflowPanel/);
  assert.match(panels, /data-testid="helix-execution-graph"/);
  assert.match(panels, /Execution graph/);
  assert.match(panels, /Observed fact/);
  assert.match(panels, /Model \/ self-audit/);
  assert.match(panels, /Policy decision/);
  assert.match(panels, /Missing evidence/);
  assert.match(panels, /data-testid="helix-execution-details"/);
  assert.match(panels, /data-testid="helix-provenance-gaps"/);
  assert.match(panels, /ordering is left unknown/);
  assert.match(panels, /data-testid="helix-primary-execution-timeline"/);
  assert.match(panels, /Solid links follow the shared backend sequence exactly/);
  assert.match(panels, /Same-turn correlated receipts/);
  assert.match(panels, />\s*Analyze\s*</);
  assert.match(panels, />\s*Monitor\s*</);
  assert.match(panels, /Live computer \/ browse feed/);
  assert.match(panels, /Trajectory/);
});

test("execution graph preserves partial and empty states", () => {
  assert.match(panels, /No receipts in this lane\./);
  assert.match(panels, /No execution receipts are available for this session yet/);
  assert.match(panels, /already loaded data stays\s+visible/);
});

test("legacy engine route is compatibility-only and points back to Chat", () => {
  assert.match(page, /Helix Engine moved into Chat/);
  assert.match(page, /Workflow and Execution Graph controls in the chat header/);
  assert.match(page, /navigate\(\{ to: "\/chat" \}\)/);
  assert.doesNotMatch(page, /helix-execution-graph/);
});
