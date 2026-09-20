// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

installLocalStorageFake().store.set(
  "unsloth_chat_settings_imported_to_studio_db",
  "true",
);
register("./store-settings-resolver.mjs", import.meta.url);

const { useChatRuntimeStore: store } = await import(
  "../src/features/chat/stores/chat-runtime-store.ts"
);

const exactA = {
  promptTokens: 90,
  completionTokens: 10,
  totalTokens: 100,
  cachedTokens: 20,
};
const liveA = {
  promptTokens: 105,
  completionTokens: 7,
  totalTokens: 112,
  cachedTokens: 0,
};

function reset(): void {
  store.setState({
    activeThreadId: "A",
    contextUsage: exactA,
    contextUsageByThreadId: { A: exactA },
    liveContextUsageByThreadId: {},
    runningByThreadId: {},
    localRunByThreadId: {},
    runOwnerByThreadId: {},
  });
}

test("live usage is transient, per-thread, and cannot repaint a background thread", () => {
  reset();
  store.getState().setThreadLiveContextUsage("A", liveA);
  assert.deepEqual(store.getState().contextUsage, liveA);
  assert.deepEqual(store.getState().contextUsageByThreadId.A, exactA);
  assert.deepEqual(store.getState().liveContextUsageByThreadId.A, liveA);

  const liveB = { ...liveA, totalTokens: 999 };
  store.getState().setThreadLiveContextUsage("B", liveB);
  assert.deepEqual(store.getState().contextUsage, liveA);
  assert.deepEqual(store.getState().liveContextUsageByThreadId.B, liveB);
  store.getState().setActiveThreadId("B");
  assert.deepEqual(store.getState().contextUsage, liveB);
  store.getState().setActiveThreadId("A");
  assert.deepEqual(store.getState().contextUsage, liveA);

  const exactB = { ...exactA, totalTokens: 500 };
  store.getState().setThreadContextUsage("B", exactB);
  assert.equal(store.getState().liveContextUsageByThreadId.B, undefined);
  assert.deepEqual(store.getState().contextUsage, liveA);

  store.getState().setActiveThreadId("B");
  assert.deepEqual(store.getState().contextUsage, exactB);
});

test("terminal exact usage immediately replaces the visible live estimate", () => {
  reset();
  store.getState().setThreadLiveContextUsage("A", liveA);
  const terminal = {
    promptTokens: 106,
    completionTokens: 8,
    totalTokens: 114,
    cachedTokens: 42,
  };
  store.getState().setThreadContextUsage("A", terminal);
  assert.equal(store.getState().liveContextUsageByThreadId.A, undefined);
  assert.deepEqual(store.getState().contextUsageByThreadId.A, terminal);
  assert.deepEqual(store.getState().contextUsage, terminal);
});

test("run lifecycle clears a stale live estimate and restores the exact thread value", () => {
  reset();
  const owner = () => undefined;
  store.getState().setThreadLiveContextUsage("A", liveA);
  store.getState().setThreadRunning("A", true, { owner });
  assert.equal(store.getState().liveContextUsageByThreadId.A, undefined);
  assert.deepEqual(store.getState().contextUsage, exactA);

  store.getState().setThreadLiveContextUsage("A", liveA);
  store.getState().setThreadRunning("A", false, { owner });
  assert.equal(store.getState().liveContextUsageByThreadId.A, undefined);
  assert.deepEqual(store.getState().contextUsage, exactA);
});

test("first-turn adoption moves the transient usage with the run key", () => {
  reset();
  const owner = () => undefined;
  const firstTurn = { ...liveA, totalTokens: 12 };
  store.setState({
    activeThreadId: null,
    contextUsage: null,
    contextUsageByThreadId: {},
    liveContextUsageByThreadId: {},
    runningByThreadId: {},
    localRunByThreadId: {},
    runOwnerByThreadId: {},
  });
  store.getState().setThreadRunning("__default", true, { owner });
  store.getState().setThreadLiveContextUsage("__default", firstTurn);
  assert.deepEqual(store.getState().contextUsage, firstTurn);

  store.getState().adoptDefaultThreadRun("real-thread");
  assert.equal(store.getState().liveContextUsageByThreadId.__default, undefined);
  assert.deepEqual(
    store.getState().liveContextUsageByThreadId["real-thread"],
    firstTurn,
  );
  store.getState().setActiveThreadId("real-thread");
  assert.deepEqual(store.getState().contextUsage, firstTurn);
});

test("an unresolved background run can advance its own estimate without repainting a new chat", () => {
  reset();
  store.setState({
    activeThreadId: null,
    contextUsage: null,
    contextUsageByThreadId: {},
    liveContextUsageByThreadId: {},
  });
  store
    .getState()
    .setThreadLiveContextUsage("__default", liveA, { visible: false });
  assert.deepEqual(store.getState().liveContextUsageByThreadId.__default, liveA);
  assert.equal(store.getState().contextUsage, null);
});

test("the bar identifies stream-time usage as an estimate", () => {
  const state = deriveContextUsageBar({
    used: 120,
    total: 1000,
    promptTokens: 100,
    completionTokens: 20,
    estimated: true,
  });
  assert.ok(state);
  assert.equal(state.estimated, true);
  assert.match(state.label, /^Estimated Context usage:/);
});

test("live usage is written only after the existing coalesced stream gate opens", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const gate = adapter.indexOf("!canPublish(streamedChars)");
  const liveWrite = adapter.indexOf(".setThreadLiveContextUsage(", gate);
  const streamedYield = adapter.indexOf("yield {", liveWrite);
  assert.ok(gate >= 0, "stream publish gate not found");
  assert.ok(liveWrite > gate, "live usage must share the gated publish cadence");
  assert.ok(streamedYield > liveWrite, "meter update should precede the matching UI yield");
  assert.match(
    adapter,
    /Math\.max\([\s\S]*estimatePromptTokens\(requestPayload\.messages\)[\s\S]*exactUsageBeforeRun\?\.totalTokens/,
  );
  assert.match(
    adapter,
    /liveUsageKey === "__default"[\s\S]*activeThreadEpochAtRunStart[\s\S]*visible: liveUsageIsVisible/,
  );
});

test("terminal exact usage and the tooltip replace/identify transient estimates", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const storeSource = readSrc("features/chat/stores/chat-runtime-store.ts");
  const bar = readSrc("features/chat/components/context-usage-bar.tsx");

  assert.match(adapter, /setThreadContextUsage\(usageThreadKey, usage\)/);
  assert.match(
    storeSource,
    /setThreadContextUsage:[\s\S]*delete live\[threadId\]/,
  );
  assert.match(bar, /liveContextUsageByThreadId\[key\] != null/);
  assert.match(bar, /Live estimate/);
});
