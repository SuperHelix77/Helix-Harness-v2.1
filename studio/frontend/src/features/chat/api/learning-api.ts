// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LearningKind = "memory" | "user" | "skill";
export type LearningTarget = "codex" | "claude" | "both";
export type LearningRecommendationAction = "skill" | "qlora" | "runtime-fix" | "none";

export type LearningEntry = {
  title: string;
  content: string;
  createdAt?: number;
};

export type LearningProposal = LearningEntry & {
  id: string;
  kind: LearningKind;
  reason?: string;
  name?: string | null;
  target?: LearningTarget;
  sourceThreadId?: string | null;
  createdAt: number;
  recommendationAction?: LearningRecommendationAction | null;
  recommendationReason?: string;
};

export type LearningState = {
  enabled: boolean;
  mem0Enabled?: boolean;
  onTheFlySkills?: boolean;
  decisionMode?: "ask" | "autonomous";
  allowSkillCreation?: boolean;
  allowQloraTraining?: boolean;
  allowRuntimeFix?: boolean;
  memory: LearningEntry[];
  user: LearningEntry[];
  pending: LearningProposal[];
  limits: { memoryChars: number; userChars: number };
  context: string;
  lastRecommendation?: {
    action: LearningRecommendationAction;
    reason: string;
    createdAt: number;
  } | null;
};

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export type HelixFinalIngestRequest = {
  idempotency_key?: string;
  session_id: string;
  thread_id?: string;
  turn_id?: string;
  prompt: string;
  final_result: string;
  critic?: Record<string, unknown>;
  self_audit?: Record<string, unknown>;
  claims?: Array<Record<string, unknown>>;
  telemetry?: Record<string, unknown>;
  acceptance_criteria?: string[];
  model_id: string;
  effective_model_id?: string;
  adapter_state?: boolean;
};

export async function ingestHelixTurn(
  input: HelixFinalIngestRequest,
): Promise<Record<string, unknown>> {
  return jsonOrThrow(
    await authFetch("/api/helix-engine/ingest-turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function getLearningState(): Promise<LearningState> {
  return jsonOrThrow<LearningState>(await authFetch("/api/learning"));
}
export async function getLearningContext(): Promise<{
  enabled: boolean;
  instruction: string;
}> {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, 4_000);
  try {
    return jsonOrThrow(
      await authFetch("/api/learning/context", { signal: controller.signal }),
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function runAdaptiveCheckpoint(input: {
  checkpointId?: string;
  checkpointEventSeq?: number;
  runId?: string;
  resumeRound?: number;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  prompt: string;
  partialResult: string;
  modelId: string;
  effectiveModelId?: string;
  adapterState?: boolean;
  telemetry?: Record<string, unknown>;
  signal?: AbortSignal;
}): Promise<{
  available: boolean;
  checkpoint_id?: string;
  checkpoint_event_seq?: number | null;
  run_id?: string | null;
  resume_round?: number;
  training_deferred: boolean;
  qlora_candidate?: boolean;
  qlora_eligible?: boolean;
  actions?: string[];
}> {
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort(input.signal?.reason);
  if (input.signal?.aborted) {
    forwardAbort();
  } else {
    input.signal?.addEventListener("abort", forwardAbort, { once: true });
  }
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 12_000);
  try {
    return await jsonOrThrow(
      await authFetch("/api/helix-engine/adaptive-checkpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          checkpoint_id: input.checkpointId,
          checkpoint_event_seq: input.checkpointEventSeq,
          run_id: input.runId,
          resume_round: input.resumeRound,
          session_id: input.sessionId ?? "",
          thread_id: input.threadId,
          turn_id: input.turnId,
          prompt: input.prompt,
          partial_result: input.partialResult,
          model_id: input.modelId,
          effective_model_id: input.effectiveModelId ?? "",
          adapter_state: input.adapterState,
          telemetry: input.telemetry ?? {},
        }),
      }),
    );
  } catch (error) {
    // A checkpoint is optional learning at a safe segment boundary. If its own
    // backend work wedges, surface a normal Error so chat-adapter records the
    // failed receipt and resumes fail-open. A real user Stop keeps its AbortError.
    if (timedOut && !input.signal?.aborted) {
      throw new Error("Adaptive checkpoint exceeded its 12s safety deadline.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
    input.signal?.removeEventListener("abort", forwardAbort);
  }
}

export async function createLearningProposal(input: {
  kind: LearningKind;
  title: string;
  content: string;
  reason?: string;
  name?: string;
  target?: LearningTarget;
  sourceThreadId?: string;
  recommendationAction?: LearningRecommendationAction;
  recommendationReason?: string;
}): Promise<LearningState> {
  return jsonOrThrow(
    await authFetch("/api/learning/proposals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function approveLearningProposal(id: string): Promise<LearningState> {
  return jsonOrThrow(
    await authFetch(`/api/learning/proposals/${encodeURIComponent(id)}/approve`, {
      method: "POST",
    }),
  );
}

export async function rejectLearningProposal(id: string): Promise<LearningState> {
  return jsonOrThrow(
    await authFetch(`/api/learning/proposals/${encodeURIComponent(id)}/reject`, {
      method: "POST",
    }),
  );
}

export async function setLearningEnabled(enabled: boolean): Promise<LearningState> {
  return setLearningConfig({ enabled });
}

export async function setLearningConfig(input: {
  enabled?: boolean;
  mem0Enabled?: boolean;
  onTheFlySkills?: boolean;
  decisionMode?: "ask" | "autonomous";
  allowSkillCreation?: boolean;
  allowQloraTraining?: boolean;
  allowRuntimeFix?: boolean;
}): Promise<LearningState> {
  return jsonOrThrow(
    await authFetch("/api/learning/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}
