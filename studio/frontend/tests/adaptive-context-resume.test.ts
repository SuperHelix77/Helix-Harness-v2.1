// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AdaptiveCheckpointControl } from "../src/features/chat/types/api.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  ADAPTIVE_AUTO_CONTINUE_LIMIT,
  AUTO_CONTINUE_LIMIT,
  autoContinueCount,
  nextAdaptiveResumeRound,
  readAdaptiveCheckpointMetadata,
  readContinuationRequest,
  recordAutoContinue,
  resetAutoContinue,
  shouldAdaptiveAutoContinue,
  shouldAutoContinue,
} = await import("../src/features/chat/utils/continuation.ts");

const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);

const CONTROL: AdaptiveCheckpointControl = {
  reason: "context_ratio",
  ratio: 0.8,
  context_length: 10_000,
  trigger_tokens: 8_000,
  prompt_tokens: 6_000,
  completion_tokens: 2_000,
  occupancy_tokens: 8_000,
  segment_max_tokens: 2_000,
};

function metadata(
  status: "pending" | "completed" | "failed",
  resumeRound = 0,
) {
  return {
    custom: {
      adaptiveCheckpoint: {
        ...CONTROL,
        resumeRound,
        checkpointStatus: status,
        ...(status === "failed" ? { checkpointError: "optional learning failed" } : {}),
      },
    },
  };
}

test("durable generation replay preserves the Studio adaptive checkpoint control frame", () => {
  const normalized = normalizeChatGenerationChunkPayload({
    type: "adaptive_checkpoint",
    ...CONTROL,
  });
  assert.deepEqual(normalized, { _adaptiveCheckpoint: CONTROL });
});

test("adaptive metadata is backend-authoritative and carries the durable resume round", () => {
  const checkpoint = readAdaptiveCheckpointMetadata(metadata("completed", 4));
  assert.ok(checkpoint);
  assert.equal(checkpoint.reason, "context_ratio");
  assert.equal(checkpoint.resumeRound, 4);
  assert.equal(nextAdaptiveResumeRound(checkpoint), 5);

  assert.equal(
    readAdaptiveCheckpointMetadata({
      custom: {
        adaptiveCheckpoint: { resumeRound: 4, checkpointStatus: "completed" },
      },
    }),
    null,
  );
});

test("adaptive resumes wait while checkpoint work is pending and fail open after optional learning fails", () => {
  const pending = readAdaptiveCheckpointMetadata(metadata("pending"));
  const failed = readAdaptiveCheckpointMetadata(metadata("failed"));
  assert.ok(pending);
  assert.ok(failed);
  assert.equal(shouldAdaptiveAutoContinue(pending, "length", "parent"), false);
  assert.equal(shouldAdaptiveAutoContinue(failed, "length", "parent"), true);
  assert.equal(shouldAdaptiveAutoContinue(failed, "cancelled", "parent"), false);
  assert.equal(shouldAdaptiveAutoContinue(failed, "interrupted", "parent"), false);
  assert.equal(shouldAdaptiveAutoContinue(failed, "context_window", "parent"), false);
});

test("explicit Stop at a recovered checkpoint never auto-resumes even if length later wins", () => {
  const cancelledCheckpoint = readAdaptiveCheckpointMetadata({
    custom: {
      adaptiveCheckpoint: {
        ...CONTROL,
        resumeRound: 2,
        checkpointStatus: "failed",
        checkpointError: "checkpoint_cancelled",
      },
    },
  });
  assert.ok(cancelledCheckpoint);
  assert.equal(
    shouldAdaptiveAutoContinue(cancelledCheckpoint, "length", "parent"),
    false,
  );
  const ordinaryFailure = readAdaptiveCheckpointMetadata(metadata("failed", 2));
  assert.ok(ordinaryFailure);
  assert.equal(
    shouldAdaptiveAutoContinue(ordinaryFailure, "length", "parent"),
    true,
  );
});

test("adaptive resume budget is separate from ordinary Max Tokens budget", () => {
  resetAutoContinue();
  assert.equal(AUTO_CONTINUE_LIMIT, 3);
  assert.ok(ADAPTIVE_AUTO_CONTINUE_LIMIT > AUTO_CONTINUE_LIMIT);

  for (let round = 0; round < AUTO_CONTINUE_LIMIT; round += 1) {
    assert.equal(shouldAutoContinue("length", "parent"), true);
    recordAutoContinue("parent");
  }
  assert.equal(autoContinueCount("parent"), AUTO_CONTINUE_LIMIT);
  assert.equal(shouldAutoContinue("length", "parent"), false);

  const checkpoint = readAdaptiveCheckpointMetadata(
    metadata("completed", AUTO_CONTINUE_LIMIT),
  );
  assert.ok(checkpoint);
  assert.equal(
    shouldAdaptiveAutoContinue(checkpoint, "length", "parent"),
    true,
    "ordinary three-round budget must not cap an adaptive checkpoint",
  );

  const exhausted = readAdaptiveCheckpointMetadata(
    metadata("completed", ADAPTIVE_AUTO_CONTINUE_LIMIT),
  );
  assert.ok(exhausted);
  assert.equal(shouldAdaptiveAutoContinue(exhausted, "length", "parent"), false);
});

test("adaptive sibling request reuses the exact continuation envelope", () => {
  const request = readContinuationRequest({
    custom: {
      unslothContinuation: {
        partial: "partial answer",
        thoughtSignature: "sig",
        adaptiveResumeRound: 7,
      },
    },
  });
  assert.deepEqual(request, {
    partial: "partial answer",
    thoughtSignature: "sig",
    adaptiveResumeRound: 7,
  });
});

test("legacy SSE parser recognizes adaptive control without turning it into a terminal finish", () => {
  const api = readSrc("features/chat/api/chat-api.ts");
  const adaptiveAt = api.indexOf('parsed.type === "adaptive_checkpoint"');
  const yieldAt = api.indexOf("_adaptiveCheckpoint: checkpoint", adaptiveAt);
  const terminalAt = api.indexOf("if (finishReason)", adaptiveAt);
  assert.ok(adaptiveAt >= 0);
  assert.ok(yieldAt > adaptiveAt);
  assert.ok(terminalAt > yieldAt);
  const branch = api.slice(adaptiveAt, terminalAt);
  assert.doesNotMatch(branch, /sawTerminalSignal\s*=\s*true/);
});

test("adapter awaits Helix checkpoint before the public length terminal can complete", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const adaptiveAt = adapter.indexOf("if (chunk._adaptiveCheckpoint)");
  const awaitAt = adapter.indexOf("await runAdaptiveCheckpoint", adaptiveAt);
  const finishAt = adapter.indexOf(
    'chunk.choices?.[0]?.finish_reason === "length"',
    adaptiveAt,
  );
  assert.ok(adaptiveAt >= 0);
  assert.ok(awaitAt > adaptiveAt);
  assert.ok(finishAt > awaitAt);
  assert.match(
    adapter.slice(adaptiveAt, finishAt),
    /sessionId: sandboxSessionId[\s\S]*threadId: resolvedThreadId[\s\S]*turnId: cancelId[\s\S]*prompt: generationUserText[\s\S]*partialResult: visiblePartial[\s\S]*modelId: params\.checkpoint[\s\S]*effectiveModelId[\s\S]*telemetry:/,
  );
});

test("adaptive checkpoint learning has a bounded fail-open safety deadline", () => {
  const api = readSrc("features/chat/api/learning-api.ts");
  const start = api.indexOf("export async function runAdaptiveCheckpoint");
  const end = api.indexOf("export async function createLearningProposal", start);
  const body = api.slice(start, end);
  assert.match(body, /setTimeout\(\(\) => \{[\s\S]*timedOut = true;[\s\S]*controller\.abort\(\);[\s\S]*12_000/);
  assert.match(body, /if \(timedOut && !input\.signal\?\.aborted\)/);
  assert.match(body, /Adaptive checkpoint exceeded its 12s safety deadline/);
});

test("preflight memory and skill receipts survive into checkpoint and final Helix telemetry", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /type ChatPreflightReceipt = \{[\s\S]*memory:[\s\S]*skills:/);
  assert.match(adapter, /receipt\.memory\.hits = mem\.results\.slice\(0, 3\)/);
  assert.match(adapter, /receipt\.skills\.relevant = relevantSkills\.map/);
  assert.match(adapter, /const helixPreflightReceipt = createChatPreflightReceipt\(\)/);
  const checkpointAt = adapter.indexOf("await runAdaptiveCheckpoint");
  assert.ok(checkpointAt >= 0);
  assert.match(adapter.slice(checkpointAt, checkpointAt + 2_500), /preflight: helixPreflightReceipt/);
  const reflectAt = adapter.indexOf("scheduleSelfReflect(", checkpointAt);
  assert.ok(reflectAt > checkpointAt);
  assert.match(adapter.slice(reflectAt, reflectAt + 3_500), /preflight: helixPreflightReceipt/);
});

test("multimodal user text still drives Mem0 and skill preflight", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const preflightAt = adapter.indexOf("const helixPreflightReceipt = createChatPreflightReceipt()");
  const promptAt = adapter.indexOf("const combinedSystemPrompt = await resolveChatInstructions", preflightAt);
  assert.ok(preflightAt >= 0);
  assert.ok(promptAt > preflightAt);
  const block = adapter.slice(promptAt, promptAt + 1_000);
  assert.match(block, /lastUserText\(outboundMessages\)/);
  assert.doesNotMatch(
    block,
    /typeof latestOutboundUser\?\.content === "string"/,
    "array-backed multimodal user turns must not collapse to an empty preflight query",
  );
});

test("adaptive auto-resume keeps the existing lease/tool/cancellation continuation gates", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /resumable\s*=\s*[\s\S]*continuable[\s\S]*modeAllowsContinuation/,
  );
  assert.match(thread, /claimAutoContinue\(messageId, runThreadId/);
  assert.match(thread, /holdAutoContinueRun\(messageId, runThreadId\)/);
  assert.match(
    thread,
    /adaptiveCheckpoint[\s\S]*shouldAdaptiveAutoContinueMessage\([\s\S]*reason[\s\S]*parentId/,
  );
  assert.match(
    thread,
    /if \(!adaptiveCheckpoint\) \{\s*recordAutoContinue\(parentId\);/,
  );
  assert.match(
    thread,
    /adaptiveResumeRound:\s*nextAdaptiveResumeRound\(adaptiveCheckpoint\)/,
  );
});

test("partial adaptive segments stay out of final post-task learning until the logical answer completes", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const postTaskAt = adapter.indexOf("// Post-task collection");
  const gateAt = adapter.indexOf("finalIncompleteReason === null", postTaskAt);
  const trainingAt = adapter.indexOf("const selfTrainingWrite = recordSelfTrainingExample", gateAt);
  const reflectAt = adapter.indexOf("scheduleSelfReflect(", trainingAt);
  assert.ok(postTaskAt >= 0);
  assert.ok(gateAt > postTaskAt);
  assert.ok(trainingAt > gateAt);
  assert.ok(reflectAt > trainingAt);
});
