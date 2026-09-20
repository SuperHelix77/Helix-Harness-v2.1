// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Minimal Send-path tracing for the conversation-stuck P0. */

export type SendTraceFields = Record<string, unknown>;

export function sendTrace(event: string, fields: SendTraceFields = {}): void {
  console.info(
    JSON.stringify({
      ts: Date.now(),
      event,
      ...fields,
    }),
  );
}

export class SendTraceSession {
  readonly startedAt = Date.now();
  threadId: string | undefined;
  runId: string | null | undefined;
  promptTokens: number | undefined;
  unresolvedToolCalls = 0;

  emit(event: string, extra: SendTraceFields = {}): void {
    sendTrace(event, {
      threadId: this.threadId,
      runId: this.runId,
      promptTokens: this.promptTokens,
      unresolvedToolCalls: this.unresolvedToolCalls,
      elapsedMs: Date.now() - this.startedAt,
      ...extra,
    });
  }
}

export function estimatePromptTokens(messages: unknown): number {
  try {
    return Math.ceil(JSON.stringify(messages).length / 4);
  } catch {
    return 0;
  }
}
