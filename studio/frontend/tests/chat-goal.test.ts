// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createChatGoal,
  goalInstruction,
  sanitizeChatGoal,
} from "../src/features/chat/lib/chat-goal.ts";

test("goal state has bounded durable fields and defaults", () => {
  const goal = createChatGoal("  Ship the v1.1 changes  ");
  assert.equal(goal.objective, "Ship the v1.1 changes");
  assert.equal(goal.status, "active");
  assert.deepEqual(goal.constraints, ["Preserve existing functionality."]);
  assert.equal(goal.verification.length, 2);
  assert.deepEqual(goal.milestones, []);
  assert.deepEqual(goal.evidenceRefs, []);
  assert.deepEqual(goal.blockers, []);
});

test("malformed goal state is ignored and valid plan state is preserved", () => {
  assert.equal(sanitizeChatGoal({ status: "active" }), undefined);
  const goal = sanitizeChatGoal({
    objective: "Build it",
    constraints: ["Keep compatibility", "x".repeat(2000)],
    verification: ["Run tests"],
    milestones: [
      { id: "M1", title: "Implement", status: "active" },
      { id: "M2", title: "Ignore malformed", status: "unknown" },
    ],
    evidenceRefs: ["commit:abc", "x".repeat(2000)],
    blockers: ["Awaiting benchmark"],
    claimCeiling: "Measured baseline only",
    nextAction: "Run the matched experiment",
    status: "draft",
    createdAt: 10,
    updatedAt: 11,
    plan: "Inspect, implement, verify.",
  });
  assert.deepEqual(goal?.constraints, ["Keep compatibility", "x".repeat(1000)]);
  assert.deepEqual(goal?.milestones, [
    { id: "M1", title: "Implement", status: "active" },
  ]);
  assert.deepEqual(goal?.evidenceRefs, ["commit:abc", "x".repeat(1000)]);
  assert.deepEqual(goal?.blockers, ["Awaiting benchmark"]);
  if (!goal) {
    throw new Error("expected a valid goal");
  }
  assert.match(goalInstruction(goal), /Objective: Build it/);
  assert.match(goalInstruction(goal), /Approved plan context/);
  assert.match(goalInstruction(goal), /Milestones: M1 \[active\] Implement/);
  assert.match(goalInstruction(goal), /Claim ceiling/);
});

test("completed goals carry a completion timestamp but active goals do not", () => {
  const active = sanitizeChatGoal({
    objective: "A",
    constraints: [],
    verification: [],
    milestones: [],
    evidenceRefs: [],
    blockers: [],
    status: "active",
    createdAt: 10,
    updatedAt: 11,
    completedAt: 12,
  });
  const completed = sanitizeChatGoal({
    ...active,
    status: "completed",
    completedAt: 12,
  });
  assert.equal(active?.completedAt, undefined);
  assert.equal(completed?.completedAt, 12);
});
