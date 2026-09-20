// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DANGLING_TOOL_STUB,
  sanitizeDanglingToolCalls,
} from "../src/features/chat/utils/outbound-tool-sanitize.ts";

test("keeps a tool call that has a matching tool result", () => {
  const { messages, unresolvedToolCalls } = sanitizeDanglingToolCalls([
    {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "call_1", type: "function", function: { name: "python", arguments: "{}" } }],
    },
    { role: "tool", tool_call_id: "call_1", content: "ok" },
    { role: "user", content: "continue" },
  ]);
  assert.equal(unresolvedToolCalls, 0);
  assert.equal(messages.length, 3);
  assert.equal((messages[0]?.tool_calls ?? []).length, 1);
});

test("drops unmatched tool-call parts and stubs an empty assistant turn", () => {
  const { messages, unresolvedToolCalls } = sanitizeDanglingToolCalls([
    {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "dangling", type: "function", function: { name: "python", arguments: "{}" } }],
    },
    { role: "user", content: "continue" },
  ]);
  assert.equal(unresolvedToolCalls, 1);
  assert.equal(messages[0]?.role, "assistant");
  assert.equal(messages[0]?.content, DANGLING_TOOL_STUB);
  assert.equal(messages[0]?.tool_calls, undefined);
  assert.equal(messages[1]?.content, "continue");
});

test("keeps assistant text when only some tool calls are unmatched", () => {
  const { messages, unresolvedToolCalls } = sanitizeDanglingToolCalls([
    {
      role: "assistant",
      content: "working",
      tool_calls: [
        { id: "keep", type: "function", function: { name: "python", arguments: "{}" } },
        { id: "drop", type: "function", function: { name: "python", arguments: "{}" } },
      ],
    },
    { role: "tool", tool_call_id: "keep", content: "1" },
  ]);
  assert.equal(unresolvedToolCalls, 1);
  assert.deepEqual(
    (messages[0]?.tool_calls ?? []).map((call) => call.id),
    ["keep"],
  );
  assert.equal(messages[0]?.content, "working");
  assert.equal(messages.length, 2);
});

test("drops orphan role=tool rows whose call was removed", () => {
  const { messages } = sanitizeDanglingToolCalls([
    {
      role: "assistant",
      content: "hi",
      tool_calls: [{ id: "gone", type: "function", function: { name: "python", arguments: "{}" } }],
    },
    { role: "tool", tool_call_id: "unrelated", content: "stale" },
  ]);
  assert.equal(messages.length, 1);
  assert.equal(messages[0]?.content, "hi");
});
