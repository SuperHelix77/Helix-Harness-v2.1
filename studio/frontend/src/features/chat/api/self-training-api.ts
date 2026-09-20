// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type SelfTrainingEvaluation = {
  candidateAdapterPath: string;
  baseScore: number;
  candidateScore: number;
  baseTokPerSec: number;
  candidateTokPerSec: number;
  rubric: string;
  adapterExists: boolean;
  intelligenceImproved: boolean;
  speedPreserved: boolean;
  speedMeasured?: boolean;
  promoted: boolean;
  createdAt: number;
};

export type SelfTrainingState = {
  enabled: boolean;
  autoTrain: boolean;
  minExamples: number;
  maxSeqLength: number;
  baseModelId: string | null;
  baseServingModelId?: string | null;
  baseSnapshotPath: string | null;
  baseServingSnapshotPath?: string | null;
  baseContextLength: number | null;
  activeAdapterPath: string | null;
  activeServingModelId?: string | null;
  lastCandidateAdapterPath?: string | null;
  candidateQualifiedOnly?: boolean;
  datasetPath: string;
  exampleCount: number;
  eligibleExampleCount?: number;
  status: string;
  lastJobId: string | null;
  lastEvaluation: SelfTrainingEvaluation | null;
  lastRecommendation?: {
    action: "skill" | "qlora" | "runtime-fix" | "none";
    reason: string;
    createdAt: number;
    advisoryOnly?: boolean;
  } | null;
  lastError: string | null;
  acceptanceCriteria: string[];
};

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export async function getSelfTrainingState(): Promise<SelfTrainingState> {
  return jsonOrThrow<SelfTrainingState>(await authFetch("/api/self-training"));
}

export async function setSelfTrainingConfig(input: {
  enabled?: boolean;
  autoTrain?: boolean;
  minExamples?: number;
  maxSeqLength?: number;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function setSelfTrainingRecommendation(input: {
  action: "skill" | "qlora" | "runtime-fix" | "none";
  reason?: string;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/recommendation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function setSelfTrainingBaseline(input: {
  modelId: string;
  snapshotPath?: string;
  contextLength?: number;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/baseline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function startSelfTraining(): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/start", { method: "POST" }),
  );
}

export async function recordSelfTrainingExample(input: {
  modelId: string;
  prompt: string;
  completion: string;
  sourceThreadId?: string;
  /** Stable logical-turn identity. The backend binds the first committed payload to this key and
   * returns the same durable receipt on lost-response/reload replay. */
  idempotencyKey?: string;
  score?: number;
  critique?: string;
}): Promise<
  SelfTrainingState & {
    recorded: boolean;
    reason?: string;
    idempotent?: boolean;
    exampleId?: string;
    idempotencyReceipt?: Record<string, unknown>;
  }
> {
  return jsonOrThrow(
    await authFetch("/api/self-training/examples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function evaluateSelfTraining(input: {
  candidateAdapterPath: string;
  baseScore: number;
  candidateScore: number;
  baseTokPerSec: number;
  candidateTokPerSec: number;
  rubric?: string;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function benchmarkSelfTraining(input: {
  candidateAdapterPath: string;
  adapterName?: string;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function hotSwapSelfTrainingAdapter(input: {
  adapterPath: string;
  adapterName?: string;
}): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/hot-swap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function revertSelfTrainingAdapter(): Promise<SelfTrainingState> {
  return jsonOrThrow(
    await authFetch("/api/self-training/revert", { method: "POST" }),
  );
}
