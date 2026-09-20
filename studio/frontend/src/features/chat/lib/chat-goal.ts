// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Durable state for the Codex-like Plan -> Goal workflow. Keep this object small: it is stored
 * inside a thread's settings snapshot and is carried through exports and older-server writes. */
export type ChatGoalStatus = "draft" | "active" | "paused" | "completed";
export type ChatGoalMilestoneStatus =
  | "pending"
  | "active"
  | "completed"
  | "blocked";

export interface ChatGoalMilestone {
  id: string;
  title: string;
  status: ChatGoalMilestoneStatus;
}

export interface ChatGoalState {
  objective: string;
  constraints: string[];
  verification: string[];
  milestones: ChatGoalMilestone[];
  evidenceRefs: string[];
  blockers: string[];
  status: ChatGoalStatus;
  createdAt: number;
  updatedAt: number;
  /** Wall-clock start for the visible elapsed timer. Omitted while paused. */
  startedAt?: number;
  /** Accumulated active time, preserved across pause/resume and reload. */
  elapsedMs?: number;
  plan?: string;
  claimCeiling?: string;
  nextAction?: string;
  completedAt?: number;
}

export const CHAT_GOAL_OBJECTIVE_MAX_CHARS = 4_096;
export const CHAT_GOAL_PLAN_MAX_CHARS = 32_768;
export const CHAT_GOAL_ITEM_MAX_CHARS = 1_000;
export const CHAT_GOAL_ITEMS_MAX = 16;
export const CHAT_GOAL_MILESTONE_ID_MAX_CHARS = 64;
export const CHAT_GOAL_MILESTONE_TITLE_MAX_CHARS = 500;
export const CHAT_GOAL_MILESTONES_MAX = 32;

const GOAL_STATUSES: readonly ChatGoalStatus[] = [
  "draft",
  "active",
  "paused",
  "completed",
];
const GOAL_MILESTONE_STATUSES: readonly ChatGoalMilestoneStatus[] = [
  "pending",
  "active",
  "completed",
  "blocked",
];

function boundedText(value: unknown, max: number): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const text = value.trim();
  return text ? text.slice(0, max) : undefined;
}

function boundedItems(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => boundedText(item, CHAT_GOAL_ITEM_MAX_CHARS))
    .filter((item): item is string => item !== undefined)
    .slice(0, CHAT_GOAL_ITEMS_MAX);
}

function boundedMilestones(value: unknown): ChatGoalMilestone[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return undefined;
      }
      const raw = item as Record<string, unknown>;
      const id = boundedText(raw.id, CHAT_GOAL_MILESTONE_ID_MAX_CHARS);
      const title = boundedText(raw.title, CHAT_GOAL_MILESTONE_TITLE_MAX_CHARS);
      const status = GOAL_MILESTONE_STATUSES.includes(
        raw.status as ChatGoalMilestoneStatus,
      )
        ? (raw.status as ChatGoalMilestoneStatus)
        : undefined;
      return id && title && status ? { id, title, status } : undefined;
    })
    .filter((item): item is ChatGoalMilestone => item !== undefined)
    .slice(0, CHAT_GOAL_MILESTONES_MAX);
}

function finiteTimestamp(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.round(value)
    : fallback;
}

function finiteDuration(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.round(value)
    : Math.max(0, Math.round(fallback));
}

/** Normalize data read from a durable row. Unknown or malformed goal data is ignored so an old
 * build can still open the conversation and a partial write cannot poison the chat UI. */
export function sanitizeChatGoal(value: unknown): ChatGoalState | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const raw = value as Record<string, unknown>;
  const objective = boundedText(raw.objective, CHAT_GOAL_OBJECTIVE_MAX_CHARS);
  const status = GOAL_STATUSES.includes(raw.status as ChatGoalStatus)
    ? (raw.status as ChatGoalStatus)
    : undefined;
  if (!(objective && status)) {
    return undefined;
  }
  const now = Date.now();
  const createdAt = finiteTimestamp(raw.createdAt, now);
  const updatedAt = Math.max(createdAt, finiteTimestamp(raw.updatedAt, now));
  const completedAt =
    status === "completed"
      ? finiteTimestamp(raw.completedAt, updatedAt)
      : undefined;
  const elapsedMs = finiteDuration(
    raw.elapsedMs,
    status === "active"
      ? 0
      : (completedAt ?? updatedAt) - createdAt,
  );
  const startedAt =
    status === "active"
      ? finiteTimestamp(raw.startedAt, createdAt)
      : undefined;
  const plan = boundedText(raw.plan, CHAT_GOAL_PLAN_MAX_CHARS);
  const claimCeiling = boundedText(raw.claimCeiling, CHAT_GOAL_ITEM_MAX_CHARS);
  const nextAction = boundedText(raw.nextAction, CHAT_GOAL_ITEM_MAX_CHARS);
  return {
    objective,
    constraints: boundedItems(raw.constraints),
    verification: boundedItems(raw.verification),
    milestones: boundedMilestones(raw.milestones),
    evidenceRefs: boundedItems(raw.evidenceRefs),
    blockers: boundedItems(raw.blockers),
    status,
    createdAt,
    updatedAt,
    elapsedMs,
    ...(startedAt ? { startedAt } : {}),
    ...(plan ? { plan } : {}),
    ...(claimCeiling ? { claimCeiling } : {}),
    ...(nextAction ? { nextAction } : {}),
    ...(completedAt ? { completedAt } : {}),
  };
}

export function createChatGoal(
  objective: string,
  options: {
    status?: ChatGoalStatus;
    plan?: string;
    constraints?: string[];
    verification?: string[];
    milestones?: ChatGoalMilestone[];
    evidenceRefs?: string[];
    blockers?: string[];
    claimCeiling?: string;
    nextAction?: string;
  } = {},
): ChatGoalState {
  const now = Date.now();
  const goal = sanitizeChatGoal({
    objective,
    constraints: options.constraints ?? ["Preserve existing functionality."],
    verification: options.verification ?? [
      "Run the relevant tests or checks.",
      "Confirm the requested behavior works end to end.",
    ],
    milestones: options.milestones ?? [],
    evidenceRefs: options.evidenceRefs ?? [],
    blockers: options.blockers ?? [],
    claimCeiling: options.claimCeiling,
    nextAction: options.nextAction,
    status: options.status ?? "active",
    plan: options.plan,
    createdAt: now,
    updatedAt: now,
    startedAt: options.status === "draft" ? undefined : now,
    elapsedMs: 0,
  });
  if (!goal) {
    throw new Error("Could not create a valid goal.");
  }
  return goal;
}

export function goalInstruction(goal: ChatGoalState): string {
  const lines = [
    "A durable Goal is active for this conversation.",
    `Objective: ${goal.objective}`,
    goal.constraints.length > 0
      ? `Constraints: ${goal.constraints.join("; ")}`
      : null,
    goal.verification.length > 0
      ? `Verification required: ${goal.verification.join("; ")}`
      : null,
    goal.milestones.length > 0
      ? `Milestones: ${goal.milestones.map((milestone) => `${milestone.id} [${milestone.status}] ${milestone.title}`).join("; ")}`
      : null,
    goal.blockers.length > 0 ? `Blockers: ${goal.blockers.join("; ")}` : null,
    goal.evidenceRefs.length > 0
      ? `Evidence references to inspect: ${goal.evidenceRefs.join("; ")}`
      : null,
    goal.claimCeiling ? `Claim ceiling: ${goal.claimCeiling}` : null,
    goal.nextAction ? `Next action: ${goal.nextAction}` : null,
    goal.plan ? `Approved plan context: ${goal.plan}` : null,
    "Continue making measurable progress toward this objective. Use a planner-to-executor-to-skeptical-verifier loop: choose the smallest safe next action, execute one milestone, run its narrowest relevant check, and repair or narrow the claim when the check fails. Preserve existing functionality, keep evidence references current, and report what was verified. Do not claim completion until the verification requirements have actually been checked.",
  ];
  return lines.filter((line): line is string => line !== null).join("\n");
}

export function goalObjectiveFromText(text: string): string {
  const trimmed = text.trim();
  return (trimmed || "Complete the requested task.").slice(
    0,
    CHAT_GOAL_OBJECTIVE_MAX_CHARS,
  );
}
