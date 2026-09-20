// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type ChatGoalState,
  createChatGoal,
  sanitizeChatGoal,
} from "../lib/chat-goal";
import {
  getStoredChatThread,
  updateStoredChatThread,
} from "../utils/chat-history-storage";

type ChatGoalStore = {
  goalsByThreadId: Record<string, ChatGoalState | null | undefined>;
  loadingByThreadId: Record<string, boolean | undefined>;
  savingByThreadId: Record<string, boolean | undefined>;
  load: (threadId: string) => Promise<ChatGoalState | null>;
  save: (threadId: string, goal: ChatGoalState | null) => Promise<void>;
  start: (threadId: string, objective: string) => Promise<ChatGoalState>;
  update: (
    threadId: string,
    update: (goal: ChatGoalState) => ChatGoalState,
  ) => Promise<void>;
  clear: (threadId: string) => Promise<void>;
};

const loadVersions = new Map<string, number>();
const saveVersions = new Map<string, number>();

function bump(versions: Map<string, number>, threadId: string): number {
  const next = (versions.get(threadId) ?? 0) + 1;
  versions.set(threadId, next);
  return next;
}

export const useChatGoalStore = create<ChatGoalStore>((set, get) => ({
  goalsByThreadId: {},
  loadingByThreadId: {},
  savingByThreadId: {},
  load: async (threadId) => {
    const version = bump(loadVersions, threadId);
    set((state) => ({
      loadingByThreadId: { ...state.loadingByThreadId, [threadId]: true },
    }));
    try {
      const thread = await getStoredChatThread(threadId);
      const goal = sanitizeChatGoal(thread?.settings?.goal);
      if (loadVersions.get(threadId) === version) {
        set((state) => ({
          goalsByThreadId: {
            ...state.goalsByThreadId,
            [threadId]: goal ?? null,
          },
          loadingByThreadId: { ...state.loadingByThreadId, [threadId]: false },
        }));
      }
      return goal ?? null;
    } catch (error) {
      if (loadVersions.get(threadId) === version) {
        set((state) => ({
          loadingByThreadId: { ...state.loadingByThreadId, [threadId]: false },
        }));
      }
      throw error;
    }
  },
  save: async (threadId, goal) => {
    const version = bump(saveVersions, threadId);
    const previous = get().goalsByThreadId[threadId] ?? null;
    const next = goal ? (sanitizeChatGoal(goal) ?? null) : null;
    set((state) => ({
      goalsByThreadId: { ...state.goalsByThreadId, [threadId]: next },
      savingByThreadId: { ...state.savingByThreadId, [threadId]: true },
    }));
    try {
      await updateStoredChatThread(threadId, {
        settingsPatch: { goal: next },
      });
      if (saveVersions.get(threadId) === version) {
        set((state) => ({
          savingByThreadId: { ...state.savingByThreadId, [threadId]: false },
        }));
      }
    } catch (error) {
      if (saveVersions.get(threadId) === version) {
        set((state) => ({
          goalsByThreadId: { ...state.goalsByThreadId, [threadId]: previous },
          savingByThreadId: { ...state.savingByThreadId, [threadId]: false },
        }));
      }
      throw error;
    }
  },
  start: async (threadId, objective) => {
    const existing = get().goalsByThreadId[threadId];
    const goal =
      existing && existing.status === "draft"
        ? {
            ...existing,
            status: "active" as const,
            updatedAt: Date.now(),
            startedAt: existing.startedAt ?? Date.now(),
            completedAt: undefined,
          }
        : createChatGoal(objective);
    await get().save(threadId, goal);
    return goal;
  },
  update: async (threadId, update) => {
    const existing = get().goalsByThreadId[threadId];
    if (!existing) {
      throw new Error("This chat has no active goal.");
    }
    await get().save(threadId, update(existing));
  },
  clear: async (threadId) => {
    await get().save(threadId, null);
  },
}));
