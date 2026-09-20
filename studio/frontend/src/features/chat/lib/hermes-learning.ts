// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LearningKind,
  LearningTarget,
  LearningRecommendationAction,
} from "../api/learning-api";

export const HERMES_LEARNING_REVIEW_PREFIX = "[UNSLOTH_HERMES_LEARNING_REVIEW]";
export const HELIX_SELF_CRITIC_PREFIX = "[HELIX_SELF_CRITIC]";
export const HELIX_SELF_AUDIT_PREFIX = "[HELIX_SELF_AUDIT]";
export const HERMES_LEARNING_TAG = "unsloth-learning";
export const HELIX_SELF_CRITIC_TAG = "helix-self-critic";
export const HELIX_SELF_AUDIT_TAG = "helix-self-audit";
export const ON_THE_FLY_SKILL_TAG = "unsloth-skill-draft";
export const HERMES_LEARNING_CHANGED_EVENT = "unsloth-hermes-learning-changed";
export const HERMES_LEARNING_OPEN_EVENT = "unsloth-open-hermes-learning";
export const SELF_QLORA_OPEN_EVENT = "unsloth-open-self-qlora";

export type HelixSelfCritic = {
  finished: "full" | "partial" | "none";
  rightTools: boolean;
  rightSkills: boolean;
  tooManyTools: boolean;
  toolCount?: number;
  usedSkills?: string[];
  notes?: string;
  recommendation: LearningRecommendationAction;
  skillTitle?: string;
  skillContent?: string;
  reason?: string;
};

export type HelixEvidenceClaim = {
  claim_id?: string;
  claim: string;
  supporting_evidence?: string[];
  evidence_refs?: string[];
  contradicting_evidence?: string[];
  missing_evidence?: string[];
  confidence?: number;
};

export type HelixSelfAudit = {
  objective: string;
  achieved: boolean | null;
  contributing_actions: string[];
  unnecessary_actions: string[];
  failures: string[];
  retries: string[];
  rediscovered_information: string[];
  excess_retrieval: string[];
  avoidable_cache_disruption: string[];
  tool_selection_correct: boolean | null;
  expensive_resource_misuse: string[];
  overclaimed_claims: string[];
  stopped_too_early: boolean;
  continued_too_long: boolean;
  better_trajectory: string[];
  reusable_lessons: string[];
  likely_behavioral_pattern: boolean;
  recommendation: "IGNORE" | "RUNTIME_POLICY" | "MEMORY" | "SKILL" | "QLORA_CANDIDATE" | "CAPABILITY_GAP";
  recommendation_reason: string;
  self_assessment_confidence: number;
  claims: HelixEvidenceClaim[];
};

export function openHermesLearningManager(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(HERMES_LEARNING_OPEN_EVENT));
  }
}

export function openSelfQloraManager(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(SELF_QLORA_OPEN_EVENT));
  }
}

export type ParsedLearningProposal = {
  kind: LearningKind;
  title: string;
  content: string;
  reason?: string;
  name?: string;
  target?: LearningTarget;
  recommendationAction?: LearningRecommendationAction;
  recommendationReason?: string;
};

export function parseOnTheFlySkillDraft(text: string): ParsedLearningProposal | null {
  const match = text.match(
    new RegExp(`<${ON_THE_FLY_SKILL_TAG}>\\s*([\\s\\S]*?)\\s*</${ON_THE_FLY_SKILL_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  try {
    const value = JSON.parse(match[1]) as Record<string, unknown>;
    const name = typeof value.name === "string" ? value.name.trim() : "";
    const title = typeof value.title === "string" ? value.title.trim() : "";
    const content = typeof value.content === "string" ? value.content.trim() : "";
    if (!name || !title || !content || !/^[a-z0-9][a-z0-9._-]{0,63}$/i.test(name)) return null;
    return {
      kind: "skill",
      name: name.toLowerCase(),
      title: title.slice(0, 240),
      content: content.slice(0, 100_000),
      target: "codex",
      recommendationAction: "skill",
      ...(typeof value.reason === "string" ? { reason: value.reason.slice(0, 2_000) } : {}),
      recommendationReason: "A repeatable procedure was identified; stage it as a skill before considering QLoRA.",
    };
  } catch {
    return null;
  }
}

export function stripOnTheFlySkillDraft(text: string): string {
  return text
    .replace(
      new RegExp(`<${ON_THE_FLY_SKILL_TAG}>\\s*[\\s\\S]*?\\s*</${ON_THE_FLY_SKILL_TAG}>`, "gi"),
      "",
    )
    .trim();
}

/** The model proposes; the app stages. It cannot silently edit a memory file or install code. */
export function hermesLearningReviewPrompt(focus: string): string {
  return [
    HERMES_LEARNING_REVIEW_PREFIX,
    "Review the current conversation as a skeptical Hermes-style learning reviewer.",
    "Find at most one durable, reusable lesson or procedure that would improve future work.",
    "Before choosing, ask yourself in this order: (1) Do I already have a skill I should use? (2) Do we need a new skill I can reuse? (3) Do I fix the runtime? (4) Do I need to QLoRA-train myself on this task? QLoRA is expensive; prefer a skill or memory when the gap is procedural or preference-based. Only recommend QLoRA for a repeated, measurable behavior gap that a skill cannot fix. This is a recommendation, never a score or approval.",
    "Prefer a verified engineering fact, user preference, or repeatable plain-text procedure; do not save secrets, transient details, or unsupported guesses.",
    "Return the normal concise explanation first, then exactly one tag with a JSON object, or kind=none if there is no durable lesson:",
    `<${HERMES_LEARNING_TAG}>{\"kind\":\"memory|user|skill|none\",\"title\":\"short title\",\"content\":\"compact lesson or SKILL.md instructions\",\"reason\":\"why it is reusable\",\"name\":\"skill-name when kind is skill\",\"target\":\"codex\"}</${HERMES_LEARNING_TAG}>`,
    "You may additionally include recommendationAction=skill, qlora, runtime-fix, or none and recommendationReason in that JSON. This never authorizes training or promotion.",
    "Do not claim that anything was saved. The app will place a valid proposal in a review inbox for approval.",
    focus.trim() ? `Review focus: ${focus.trim()}` : "Review focus: the current task and its verified outcome.",
  ].join("\n");
}

export function isHermesLearningReviewRequest(text: string): boolean {
  return text.includes(HERMES_LEARNING_REVIEW_PREFIX);
}

export function helixSelfAuditPrompt(
  focus: string,
  observableArtifacts: Record<string, unknown>,
): string {
  const rawTools = Array.isArray(observableArtifacts.tool_steps)
    ? observableArtifacts.tool_steps.slice(-16)
    : [];
  const boundedTools = rawTools.map((value) => {
    if (!value || typeof value !== "object") return value;
    const raw = value as Record<string, unknown>;
    return {
      index: raw.index,
      evidence_ids: Array.isArray(raw.evidence_ids)
        ? raw.evidence_ids.filter((item): item is string => typeof item === "string").slice(0, 4)
        : [],
      name: typeof raw.name === "string" ? raw.name.slice(0, 160) : raw.name,
      arguments: typeof raw.arguments === "string" ? raw.arguments.slice(0, 600) : raw.arguments,
      result: typeof raw.result === "string" ? raw.result.slice(0, 1_200) : raw.result,
      useful_hint: raw.useful_hint,
      error: typeof raw.error === "string" ? raw.error.slice(0, 500) : raw.error,
      retry: raw.retry,
    };
  });
  const objective = String(observableArtifacts.objective ?? "").slice(0, 4_000);
  const finalResult = String(observableArtifacts.final_result ?? "").slice(0, 4_000);
  const telemetry = boundAuditJsonValue(observableArtifacts.telemetry ?? {}, 0);
  const presentedContext = String(observableArtifacts.presented_context ?? "").slice(0, 6_000);
  const acceptanceCriteria = boundAuditJsonValue(observableArtifacts.acceptance_criteria ?? [], 0);
  const cacheIntegrity = boundAuditJsonValue(observableArtifacts.cache_integrity ?? {}, 0);
  const evidence = boundAuditJsonValue(observableArtifacts.evidence ?? [], 0);
  const controlEvents = boundAuditJsonValue(observableArtifacts.control_events ?? [], 0);
  const objectiveOutcomeEvidence = boundAuditJsonValue(
    observableArtifacts.objective_outcome_evidence ?? [],
    0,
  );
  const edits = boundAuditJsonValue(observableArtifacts.edits ?? [], 0);
  const tests = boundAuditJsonValue(observableArtifacts.tests ?? [], 0);
  const benchmarks = boundAuditJsonValue(observableArtifacts.benchmarks ?? [], 0);
  const preAuditDecision = boundAuditJsonValue(observableArtifacts.pre_audit_decision ?? {}, 0);
  const rawTrajectory =
    observableArtifacts.trajectory && typeof observableArtifacts.trajectory === "object"
      ? (observableArtifacts.trajectory as Record<string, unknown>)
      : {};
  const trajectory = boundAuditJsonValue(
    Object.fromEntries(
      Object.entries(rawTrajectory).filter(
        ([key]) => key !== "tool_steps" && key !== "reasoning",
      ),
    ),
    0,
  );
  let artifactPayload: Record<string, unknown> = {
    schema_version: observableArtifacts.schema_version ?? "helix.audit-input.v1",
    objective,
    presented_context: presentedContext,
    final_result: finalResult,
    acceptance_criteria: acceptanceCriteria,
    telemetry,
    cache_integrity: cacheIntegrity,
    evidence,
    control_events: controlEvents,
    objective_outcome_evidence: objectiveOutcomeEvidence,
    edits,
    tests,
    benchmarks,
    trajectory,
    pre_audit_decision: preAuditDecision,
    tool_steps: boundedTools,
  };
  let artifacts = JSON.stringify(artifactPayload);
  if (artifacts.length > 28_000) {
    artifactPayload = {
      objective: objective.slice(0, 3_000),
      presented_context: presentedContext.slice(0, 2_000),
      final_result: finalResult.slice(0, 3_000),
      acceptance_criteria: acceptanceCriteria,
      telemetry: summarizeAuditTelemetry(observableArtifacts.telemetry),
      cache_integrity: boundAuditJsonValue(observableArtifacts.cache_integrity ?? {}, 1),
      control_events: Array.isArray(observableArtifacts.control_events)
        ? boundAuditJsonValue(observableArtifacts.control_events.slice(-16), 1)
        : controlEvents,
      evidence: Array.isArray(observableArtifacts.evidence)
        ? boundAuditJsonValue(observableArtifacts.evidence.slice(0, 12), 1)
        : evidence,
      tests: Array.isArray(observableArtifacts.tests)
        ? boundAuditJsonValue(observableArtifacts.tests.slice(-8), 1)
        : tests,
      benchmarks: Array.isArray(observableArtifacts.benchmarks)
        ? boundAuditJsonValue(observableArtifacts.benchmarks.slice(-8), 1)
        : benchmarks,
      pre_audit_decision: preAuditDecision,
      tool_steps: boundedTools.slice(-8),
      helix_artifact_compaction: "structured",
    };
    artifacts = JSON.stringify(artifactPayload);
  }
  if (artifacts.length > 28_000) {
    artifactPayload = {
      objective: objective.slice(0, 2_000),
      final_result: finalResult.slice(0, 2_000),
      telemetry: { helix_artifact_compaction: "telemetry_omitted_for_size" },
      cache_integrity: boundAuditJsonValue(observableArtifacts.cache_integrity ?? {}, 2),
      control_events: Array.isArray(observableArtifacts.control_events)
        ? boundAuditJsonValue(observableArtifacts.control_events.slice(-8), 2)
        : [],
      evidence: Array.isArray(observableArtifacts.evidence)
        ? boundAuditJsonValue(observableArtifacts.evidence.slice(0, 6), 2)
        : [],
      tests: Array.isArray(observableArtifacts.tests)
        ? boundAuditJsonValue(observableArtifacts.tests.slice(-4), 2)
        : [],
      benchmarks: Array.isArray(observableArtifacts.benchmarks)
        ? boundAuditJsonValue(observableArtifacts.benchmarks.slice(-4), 2)
        : [],
      tool_steps: boundedTools.slice(-4).map((value) => {
        if (!value || typeof value !== "object") return value;
        const raw = value as Record<string, unknown>;
        return {
          ...raw,
          arguments: typeof raw.arguments === "string" ? raw.arguments.slice(0, 300) : raw.arguments,
          result: typeof raw.result === "string" ? raw.result.slice(0, 600) : raw.result,
        };
      }),
      helix_artifact_compaction: "minimal",
    };
    artifacts = JSON.stringify(artifactPayload);
  }
  return [
    HELIX_SELF_AUDIT_PREFIX,
    "Audit the task you just performed using ONLY the observable artifacts below. Do not reveal or reconstruct hidden chain-of-thought.",
    "Judge objective completion, useful and unnecessary actions, failures/retries, repeated rediscovery/retrieval, cache/context waste, tool choice, resource escalation, unsupported claims, stopping behavior, a shorter equivalent trajectory, reusable lessons, and whether the issue is likely a repeated behavioral pattern.",
    "Your self-report is advisory. Hermes will compare it with tests, tool outputs, telemetry and other objective evidence. You cannot authorize training.",
    "For claims, supporting_evidence is descriptive only. Put evidence IDs from the observable artifact list into evidence_refs; only backend-resolved evidence IDs can count as proof.",
    "Recommendation must be exactly one of IGNORE, RUNTIME_POLICY, MEMORY, SKILL, QLORA_CANDIDATE, CAPABILITY_GAP. QLORA_CANDIDATE is appropriate only for a repeated behavioral tendency, never a one-off error.",
    "Return strict valid JSON. Every top-level array other than claims must contain strings only and at most 4 items; claims may contain at most 8 objects, and each claim evidence array must contain strings only. Keep each string concise. Close every array/object and the final audit tag. Do not place recommendation fields inside reusable_lessons or any other array.",
    `Return exactly one tag: <${HELIX_SELF_AUDIT_TAG}>{"objective":"","achieved":true,"contributing_actions":[],"unnecessary_actions":[],"failures":[],"retries":[],"rediscovered_information":[],"excess_retrieval":[],"avoidable_cache_disruption":[],"tool_selection_correct":true,"expensive_resource_misuse":[],"overclaimed_claims":[],"stopped_too_early":false,"continued_too_long":false,"better_trajectory":[],"reusable_lessons":[],"likely_behavioral_pattern":false,"recommendation":"IGNORE","recommendation_reason":"","self_assessment_confidence":0.5,"claims":[{"claim_id":"","claim":"","supporting_evidence":[],"evidence_refs":[],"contradicting_evidence":[],"missing_evidence":[],"confidence":0.5}]}</${HELIX_SELF_AUDIT_TAG}>`,
    focus.trim() ? `Task summary: ${focus.trim()}` : "Task summary: completed task.",
    `Observable artifacts JSON: ${artifacts}`,
  ].join("\n");
}

function boundAuditJsonValue(value: unknown, depth: number): unknown {
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (typeof value === "string") return value.slice(0, 800);
  if (depth >= 4) return "[depth-limited]";
  if (Array.isArray(value)) {
    const bounded = value.slice(0, 32).map((item) => boundAuditJsonValue(item, depth + 1));
    if (value.length > bounded.length) bounded.push(`[${value.length - bounded.length} more items]`);
    return bounded;
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    const bounded: Record<string, unknown> = {};
    for (const [key, item] of entries.slice(0, 64)) {
      bounded[key.slice(0, 120)] = boundAuditJsonValue(item, depth + 1);
    }
    if (entries.length > 64) bounded.helix_omitted_keys = entries.length - 64;
    return bounded;
  }
  return String(value ?? "").slice(0, 800);
}

function summarizeAuditTelemetry(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { helix_artifact_compaction: "telemetry_unavailable" };
  }
  const raw = value as Record<string, unknown>;
  const keys = [
    "prompt_tokens", "promptTokens", "cached_tokens", "cachedTokens", "stable_prefix_tokens",
    "newly_evaluated_tokens", "prefill_ms", "decode_ms", "ttft_ms", "context_compaction",
    "context_compactions", "prompt_reconstruction", "prompt_reconstructions", "system_prompt_change",
    "system_prompt_changes", "tool_schema_change", "tool_schema_changes", "context_reorder",
    "context_reorders", "repeated_context_insertions", "speculative_requested", "speculative_engaged",
    "accepted_drafts", "acceptedDrafts", "rejected_drafts", "rejectedDrafts", "runtime_config",
    "runtimeConfig", "latency_ms", "completion_tokens", "completionTokens", "trajectory_id",
    "objective_verified",
  ];
  const summary: Record<string, unknown> = { helix_artifact_compaction: "telemetry_summary" };
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(raw, key)) {
      summary[key] = boundAuditJsonValue(raw[key], 0);
    }
  }
  const omitted = Object.keys(raw).filter((key) => !keys.includes(key)).length;
  if (omitted) summary.helix_omitted_keys = omitted;
  return summary;
}

export function isHelixSelfAuditRequest(text: string): boolean {
  return text.includes(HELIX_SELF_AUDIT_PREFIX);
}

function auditStrings(value: unknown, limit = 4): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string").slice(0, limit)
    : [];
}

const HELIX_EVIDENCE_REF_RE = /^(?:trajectory:objective_verified|benchmark:speed|speculation:accepted|tool:\d+:(?:error|result|verification))$/;

export function parseHelixSelfAudit(text: string): HelixSelfAudit | null {
  const match = text.match(
    new RegExp(`<${HELIX_SELF_AUDIT_TAG}>\\s*([\\s\\S]*?)\\s*</${HELIX_SELF_AUDIT_TAG}>`, "i"),
  );
  // Some otherwise well-formed local-model audits have reproducibly emitted the
  // opening tag plus one complete JSON object but omitted only the closing tag.
  // Accept that narrow shape only when the opening tag begins the response and
  // *all* remaining non-whitespace text is the JSON payload. JSON.parse below
  // therefore still rejects partial JSON and any trailing prose.
  const unterminated = match?.[1]
    ? null
    : text.match(new RegExp(`^\\s*<${HELIX_SELF_AUDIT_TAG}>\\s*([\\s\\S]*?)\\s*$`, "i"));
  const payloadText = match?.[1] ?? unterminated?.[1];
  if (!payloadText) return null;
  let value: unknown;
  try { value = JSON.parse(payloadText); } catch { return null; }
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const allowed = new Set(["IGNORE", "RUNTIME_POLICY", "MEMORY", "SKILL", "QLORA_CANDIDATE", "CAPABILITY_GAP"]);
  const recommendation = String(raw.recommendation ?? "IGNORE").toUpperCase();
  if (!allowed.has(recommendation)) return null;
  const claims = Array.isArray(raw.claims)
    ? raw.claims
        .filter((item): item is Record<string, unknown> => Boolean(
          item && typeof item === "object" && typeof (item as Record<string, unknown>).claim === "string",
        ))
        .slice(0, 8)
        .map((claim) => ({
          claim_id: typeof claim.claim_id === "string" ? claim.claim_id.slice(0, 200) : undefined,
          claim: String(claim.claim).slice(0, 2_000),
          supporting_evidence: auditStrings(claim.supporting_evidence, 4),
          evidence_refs: auditStrings(claim.evidence_refs, 4)
            .map((item) => item.slice(0, 200))
            .filter((item) => HELIX_EVIDENCE_REF_RE.test(item)),
          contradicting_evidence: auditStrings(claim.contradicting_evidence, 4),
          missing_evidence: auditStrings(claim.missing_evidence, 4),
          confidence: typeof claim.confidence === "number"
            ? Math.max(0, Math.min(1, claim.confidence))
            : 0.5,
        }))
    : [];
  const confidenceRaw = typeof raw.self_assessment_confidence === "number" ? raw.self_assessment_confidence : 0.5;
  return {
    objective: typeof raw.objective === "string" ? raw.objective.slice(0, 4_000) : "",
    achieved: typeof raw.achieved === "boolean" ? raw.achieved : null,
    contributing_actions: auditStrings(raw.contributing_actions),
    unnecessary_actions: auditStrings(raw.unnecessary_actions),
    failures: auditStrings(raw.failures),
    retries: auditStrings(raw.retries),
    rediscovered_information: auditStrings(raw.rediscovered_information),
    excess_retrieval: auditStrings(raw.excess_retrieval),
    avoidable_cache_disruption: auditStrings(raw.avoidable_cache_disruption),
    tool_selection_correct: typeof raw.tool_selection_correct === "boolean" ? raw.tool_selection_correct : null,
    expensive_resource_misuse: auditStrings(raw.expensive_resource_misuse),
    overclaimed_claims: auditStrings(raw.overclaimed_claims),
    stopped_too_early: raw.stopped_too_early === true,
    continued_too_long: raw.continued_too_long === true,
    better_trajectory: auditStrings(raw.better_trajectory),
    reusable_lessons: auditStrings(raw.reusable_lessons),
    likely_behavioral_pattern: raw.likely_behavioral_pattern === true,
    recommendation: recommendation as HelixSelfAudit["recommendation"],
    recommendation_reason: typeof raw.recommendation_reason === "string" ? raw.recommendation_reason.slice(0, 2_000) : "",
    self_assessment_confidence: Math.max(0, Math.min(1, confidenceRaw)),
    claims,
  };
}

export function auditToLegacyCritic(audit: HelixSelfAudit): HelixSelfCritic {
  const rec: LearningRecommendationAction = "none";
  return {
    finished: audit.achieved === true ? "full" : audit.achieved === false ? "none" : "partial",
    rightTools: audit.tool_selection_correct !== false,
    rightSkills: true,
    tooManyTools: audit.unnecessary_actions.length > 0 || audit.excess_retrieval.length > 0,
    recommendation: rec,
    notes: audit.recommendation_reason,
    reason: audit.recommendation_reason,
    skillContent: audit.reusable_lessons.join("\n").slice(0, 8_000),
  };
}

export function helixSelfCriticPrompt(focus: string): string {
  return [
    HELIX_SELF_CRITIC_PREFIX,
    "Ask yourself about the turn you just completed. Answer only with the tag below.",
    "1. Did I finish the user's work? full, partial, or none.",
    "2. Did I use the right tools and skills? Prefer an existing skill over inventing a workflow.",
    "3. Did I use too many tools (rereads, duplicate searches, extra calls)?",
    "Then recommend skill, qlora, runtime-fix, or none. QLoRA is never authorized by this answer; Helix Engine decides.",
    `Return exactly: <${HELIX_SELF_CRITIC_TAG}>{"finished":"full|partial|none","rightTools":true,"rightSkills":true,"tooManyTools":false,"toolCount":0,"usedSkills":[],"notes":"short","recommendation":"skill|qlora|runtime-fix|none","skillTitle":"","skillContent":"","reason":""}</${HELIX_SELF_CRITIC_TAG}>`,
    focus.trim() ? `Turn: ${focus.trim()}` : "Turn: the current completed task.",
  ].join("\n");
}

export function isHelixSelfCriticRequest(text: string): boolean {
  return text.includes(HELIX_SELF_CRITIC_PREFIX);
}

function asBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (lowered === "true" || lowered === "yes") return true;
    if (lowered === "false" || lowered === "no") return false;
  }
  return fallback;
}

export function parseHelixSelfCritic(text: string): HelixSelfCritic | null {
  const match = text.match(
    new RegExp(`<${HELIX_SELF_CRITIC_TAG}>\\s*([\\s\\S]*?)\\s*</${HELIX_SELF_CRITIC_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  let value: unknown;
  try {
    value = JSON.parse(match[1]);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const finishedRaw = String(raw.finished ?? "partial").toLowerCase();
  const finished =
    finishedRaw === "full" || finishedRaw === "partial" || finishedRaw === "none"
      ? finishedRaw
      : "partial";
  const rec = raw.recommendation;
  const recommendation: LearningRecommendationAction =
    rec === "skill" || rec === "qlora" || rec === "runtime-fix" || rec === "none"
      ? rec
      : "none";
  const used = Array.isArray(raw.usedSkills)
    ? raw.usedSkills.filter((item): item is string => typeof item === "string").slice(0, 16)
    : [];
  const toolCount =
    typeof raw.toolCount === "number" && Number.isFinite(raw.toolCount)
      ? Math.max(0, Math.round(raw.toolCount))
      : undefined;
  return {
    finished,
    rightTools: asBool(raw.rightTools, true),
    rightSkills: asBool(raw.rightSkills, true),
    tooManyTools: asBool(raw.tooManyTools, false),
    ...(toolCount !== undefined ? { toolCount } : {}),
    ...(used.length ? { usedSkills: used } : {}),
    ...(typeof raw.notes === "string" ? { notes: raw.notes.slice(0, 2_000) } : {}),
    recommendation,
    ...(typeof raw.skillTitle === "string" ? { skillTitle: raw.skillTitle.slice(0, 240) } : {}),
    ...(typeof raw.skillContent === "string"
      ? { skillContent: raw.skillContent.slice(0, 8_000) }
      : {}),
    ...(typeof raw.reason === "string" ? { reason: raw.reason.slice(0, 2_000) } : {}),
  };
}

function isKind(value: unknown): value is LearningKind {
  return value === "memory" || value === "user" || value === "skill";
}

function isTarget(value: unknown): value is LearningTarget {
  return value === "codex" || value === "claude" || value === "both";
}

export function parseHermesLearningProposal(text: string): ParsedLearningProposal | null {
  const match = text.match(
    new RegExp(`<${HERMES_LEARNING_TAG}>\\s*([\\s\\S]*?)\\s*</${HERMES_LEARNING_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  let value: unknown;
  try {
    value = JSON.parse(match[1]);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const candidate = value as Record<string, unknown>;
  if (!isKind(candidate.kind)) return null;
  const title = typeof candidate.title === "string" ? candidate.title.trim() : "";
  const content = typeof candidate.content === "string" ? candidate.content.trim() : "";
  if (!title || !content) return null;
  return {
    kind: candidate.kind,
    title: title.slice(0, 240),
    content: content.slice(0, 100_000),
    ...(typeof candidate.reason === "string" ? { reason: candidate.reason.slice(0, 2_000) } : {}),
    ...(typeof candidate.name === "string" ? { name: candidate.name.slice(0, 64) } : {}),
    ...(isTarget(candidate.target) ? { target: candidate.target } : {}),
    ...(candidate.recommendationAction === "skill" ||
    candidate.recommendationAction === "qlora" ||
    candidate.recommendationAction === "runtime-fix" ||
    candidate.recommendationAction === "none"
      ? { recommendationAction: candidate.recommendationAction }
      : {}),
    ...(typeof candidate.recommendationReason === "string"
      ? { recommendationReason: candidate.recommendationReason.slice(0, 2_000) }
      : {}),
  };
}

export function stripHermesLearningProposal(text: string): string {
  return text
    .replace(
      new RegExp(`<${HERMES_LEARNING_TAG}>\\s*[\\s\\S]*?\\s*</${HERMES_LEARNING_TAG}>`, "gi"),
      "",
    )
    .trim();
}
