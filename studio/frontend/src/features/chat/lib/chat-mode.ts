// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ChatMode = "normal" | "plan" | "goal";

export const DEFAULT_CHAT_MODE: ChatMode = "normal";

export const CHAT_MODE_OPTIONS: readonly {
  value: ChatMode;
  label: string;
  description: string;
}[] = [
  {
    value: "normal",
    label: "Normal",
    description: "Answer normally, with the tools enabled in this chat",
  },
  {
    value: "plan",
    label: "Plan",
    description: "Analyze and propose a plan without taking actions",
  },
  {
    value: "goal",
    label: "Goal",
    description: "Work toward the objective and verify completion",
  },
] as const;

/** System guidance for the selected interaction mode. Tool gating is enforced separately in the
 * adapter so a prompt cannot accidentally re-enable actions in Plan mode. */
export function chatModeInstruction(mode: ChatMode): string | null {
  if (mode === "plan") {
    return [
      "Plan mode is active.",
      "Do not call tools, edit files, run commands, or perform external side effects.",
      "Analyze the request and return a concrete, ordered implementation plan with acceptance criteria, risks, dependencies, and an exact verification step for each milestone.",
      "Wait for the user's approval before carrying out the plan.",
    ].join(" ");
  }
  if (mode === "goal") {
    return [
      "Goal mode is active.",
      "Treat the user's latest request as the objective, keep progress focused on it, and use available tools when useful.",
      "Use a planner-to-executor-to-skeptical-verifier loop: choose the smallest safe next action, execute one milestone, run its narrowest relevant check, and repair or narrow the claim when the check fails.",
      "Preserve existing functionality, keep evidence references current, and do not claim completion until the result is checked.",
    ].join(" ");
  }
  return null;
}
