// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Replay hygiene: drop assistant tool_calls that have no matching role=tool result
 *  in the outbound history. Does not truncate context. */

export const DANGLING_TOOL_STUB = "Response interrupted";

export type SanitizableToolCall = {
  id?: unknown;
  [key: string]: unknown;
};

export type SanitizableMessage = {
  role?: unknown;
  content?: unknown;
  tool_calls?: SanitizableToolCall[] | null;
  tool_call_id?: unknown;
  reasoning_content?: unknown;
  [key: string]: unknown;
};

function toolCallId(call: SanitizableToolCall): string | null {
  return typeof call.id === "string" && call.id.length > 0 ? call.id : null;
}

function hasReplayText(content: unknown): boolean {
  if (typeof content === "string") return content.trim().length > 0;
  return Boolean(content);
}

export function sanitizeDanglingToolCalls<T extends SanitizableMessage>(
  messages: T[],
): { messages: T[]; unresolvedToolCalls: number } {
  const resultIds = new Set<string>();
  for (const message of messages) {
    if (message.role !== "tool") continue;
    if (typeof message.tool_call_id === "string" && message.tool_call_id) {
      resultIds.add(message.tool_call_id);
    }
  }

  let unresolvedToolCalls = 0;
  const rewritten: T[] = [];
  const keptCallIds = new Set<string>();

  for (const message of messages) {
    if (message.role !== "assistant" || !Array.isArray(message.tool_calls)) {
      rewritten.push(message);
      continue;
    }

    const original = message.tool_calls;
    const kept = original.filter((call) => {
      const id = toolCallId(call);
      return id !== null && resultIds.has(id);
    });
    unresolvedToolCalls += original.length - kept.length;

    if (kept.length === original.length) {
      for (const call of kept) {
        const id = toolCallId(call);
        if (id) keptCallIds.add(id);
      }
      rewritten.push(message);
      continue;
    }

    const next = { ...message } as T;
    if (kept.length > 0) {
      next.tool_calls = kept;
      for (const call of kept) {
        const id = toolCallId(call);
        if (id) keptCallIds.add(id);
      }
      rewritten.push(next);
      continue;
    }

    delete next.tool_calls;
    const hasText = hasReplayText(next.content);
    const hasReasoning =
      typeof next.reasoning_content === "string" &&
      next.reasoning_content.trim().length > 0;
    if (!hasText && !hasReasoning) {
      next.content = DANGLING_TOOL_STUB;
    } else if (next.content === null && !hasText) {
      next.content = hasReasoning ? "" : DANGLING_TOOL_STUB;
    }
    rewritten.push(next);
  }

  const messagesOut = rewritten.filter((message) => {
    if (message.role !== "tool") return true;
    return (
      typeof message.tool_call_id === "string" &&
      keptCallIds.has(message.tool_call_id)
    );
  });

  return { messages: messagesOut, unresolvedToolCalls };
}
