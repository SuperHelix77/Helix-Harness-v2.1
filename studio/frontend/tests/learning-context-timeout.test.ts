// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("getLearningContext aborts after 4s and still uses jsonOrThrow", () => {
  const src = readSrc("features/chat/api/learning-api.ts");
  const fnStart = src.indexOf("export async function getLearningContext");
  assert.ok(fnStart >= 0);
  const fn = src.slice(fnStart, src.indexOf("export async function createLearningProposal"));
  assert.match(fn, /new AbortController/);
  assert.match(fn, /setTimeout\(\(\) => \{\s*controller\.abort\(\);/);
  assert.match(fn, /4_000/);
  assert.match(fn, /jsonOrThrow/);
  assert.match(fn, /signal: controller\.signal/);
});

test("resolveLearningContext degrades learning failures to empty instruction", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(src, /getLearningContext\(\)\.catch\(/);
  assert.match(src, /enabled: false,/);
  assert.match(src, /instruction: "",/);
});

test("send path traces the required stages", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  for (const event of [
    "send.received",
    "memory.start",
    "memory.end",
    "memory.error",
    "memory.timeout",
    "prompt.start",
    "prompt.end",
    "run.created",
    "backend.request",
    "backend.first_token",
    "backend.complete",
    "run.finalized",
    "ui.terminal",
  ]) {
    assert.match(src, new RegExp(`"${event}"`));
  }
});

test("send path sanitizes dangling tool calls before the completions request", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(src, /sanitizeDanglingToolCalls\(outboundRaw\)/);
  assert.match(src, /deleteOrphanAssistantMessage/);
});

test("background self-audit disables checkpoint memory-tool recovery", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  const start = src.indexOf('"X-Helix-Background-Audit": "1"');
  assert.ok(start >= 0);
  const end = src.indexOf("const payload = response.ok", start);
  assert.ok(end > start);
  const auditRequest = src.slice(start, end);
  assert.match(auditRequest, /enable_tools: false/);
  assert.match(auditRequest, /enable_thinking: false/);
  assert.match(auditRequest, /context_policy: "rolling"/);
  assert.doesNotMatch(auditRequest, /thread_id:/);
});
