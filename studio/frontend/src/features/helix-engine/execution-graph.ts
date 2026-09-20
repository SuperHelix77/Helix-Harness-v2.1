// SPDX-License-Identifier: AGPL-3.0-only

import type {
  FeedEvent,
  HelixProvenance,
  SessionStep,
  ToolControlEvent,
} from "./engine-data";

export type ExecutionProvenance = "observed" | "model-audit" | "policy" | "missing";

export type ExecutionCategory =
  | "memory"
  | "skill"
  | "tool"
  | "browse"
  | "policy"
  | "evidence"
  | "checkpoint"
  | "runtime"
  | "adaptation"
  | "qlora"
  | "learning";

export type ExecutionNode = {
  id: string;
  category: ExecutionCategory;
  provenance: ExecutionProvenance;
  title: string;
  summary: string;
  source: string;
  sourceIndex: number | null;
  sequence: number | null;
  createdAtMs: number | null;
  status: "ok" | "error" | "advisory" | "unknown" | "missing";
  reason: string;
  evidence: string[];
  receipts: Array<{ label: string; value: string }>;
  missingEvidence: string[];
};

export type ExecutionEdge = {
  id: string;
  from: string;
  to: string;
  relation: "observed-sequence";
};

export type ExecutionGraph = {
  nodes: ExecutionNode[];
  edges: ExecutionEdge[];
  orderingMode: "exact" | "source-local";
  coverageGaps: string[];
};

export type HelixAnalyzeResult = {
  gates?: string[];
  credit?: number[];
  compressed?: { tool_calls?: number; final_decision?: string };
  route?: string;
  decision?: string;
  success_authorizes_weight_update?: boolean;
  corrections?: Array<{ bad_action?: string; correct_action?: string; why?: string }>;
  adaptive_cycle?: unknown;
  actions?: string[];
  training_target_receipt?: unknown;
};

type ExecutionGraphInput = {
  events: FeedEvent[];
  steps: SessionStep[];
  controlEvents: ToolControlEvent[];
  analysis?: HelixAnalyzeResult | null;
  provenance?: HelixProvenance | null;
  provenanceError?: string | null;
};

const MEMORY_TOOLS = new Set(["search_memory", "search_conversation"]);
const SKILL_READ_TOOLS = new Set(["read_skill"]);
const SKILL_CREATE_TOOLS = new Set(["create_skill", "learn_skill"]);
const CHECKPOINT_BOUNDARY_SOURCE = "provenance:adaptive-checkpoint-boundary";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => text(item)).filter(Boolean)
    : [];
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function jsonRecord(value: string): Record<string, unknown> | null {
  try {
    return asRecord(JSON.parse(value));
  } catch {
    return null;
  }
}

function bounded(value: unknown, limit = 900): string {
  if (value == null) return "";
  const raw = typeof value === "string" ? value : JSON.stringify(value);
  return raw.length > limit ? `${raw.slice(0, limit)}…` : raw;
}

function addSequentialEdges(
  edges: ExecutionEdge[],
  ids: string[],
  prefix: string,
): void {
  for (let index = 1; index < ids.length; index += 1) {
    edges.push({
      id: `${prefix}:${index - 1}->${index}`,
      from: ids[index - 1]!,
      to: ids[index]!,
      relation: "observed-sequence",
    });
  }
}

function stepNode(
  step: SessionStep,
  index: number,
  verification: Record<string, unknown> | null = null,
): ExecutionNode {
  const name = step.name || "tool";
  const error = text(step.error);
  const evidenceIds = strings(step.evidence_ids);
  const verificationStatus = text(verification?.status);
  let category: ExecutionCategory = "tool";
  let title = `Tool · ${name}`;
  if (MEMORY_TOOLS.has(name)) {
    category = "memory";
    title = `Memory retrieval · ${name}`;
  } else if (SKILL_READ_TOOLS.has(name)) {
    category = "skill";
    title = "Existing skill selected";
  } else if (SKILL_CREATE_TOOLS.has(name)) {
    category = "skill";
    title = name === "learn_skill" ? "Temporary skill created" : "Skill created";
  }
  const result = bounded(step.result, 1_200);
  const receipts = [
    { label: "Tool", value: name },
    { label: "Arguments", value: bounded(step.arguments, 1_200) || "None recorded" },
    { label: "Result", value: result || "No result text recorded" },
  ];
  if (typeof step.retry === "number") {
    receipts.push({ label: "Retry index", value: String(step.retry) });
  }
  if (evidenceIds.length) {
    receipts.push({ label: "Evidence receipts", value: evidenceIds.join(", ") });
  }
  if (verification) {
    receipts.push({ label: "Verification", value: bounded(verification, 1_500) });
  }
  return {
    id: `step:${step.index ?? index}`,
    category,
    provenance: "observed",
    title,
    summary: error || result || "Captured execution with no result text.",
    source: "trajectory",
    sourceIndex: step.index ?? index,
    sequence: finiteNumber(step.sequence),
    createdAtMs: finiteNumber(step.created_at_ms),
    status: error || verificationStatus === "failed" ? "error" : "ok",
    reason: [
      step.useful_hint
        ? `Capture classified this step as ${step.useful_hint}.`
        : "Observed tool execution; no usefulness classification was recorded.",
      verification
        ? `Backend verification ${verificationStatus || "status unknown"}${text(verification.detail) ? `: ${text(verification.detail)}` : "."}`
        : "",
    ].filter(Boolean).join(" "),
    evidence: [
      ...evidenceIds,
      text(verification?.claim),
      text(verification?.subject),
    ].filter(Boolean),
    receipts,
    missingEvidence:
      evidenceIds.some((id) => id.endsWith(":verification")) && !verification
        ? ["A typed verification evidence ID exists, but its detailed receipt is unavailable."]
        : evidenceIds.length
          ? []
          : ["No typed backend evidence receipt was attached to this step."],
  };
}

function controlNode(event: ToolControlEvent, index: number): ExecutionNode {
  const action = text(event.action) || "policy decision";
  const toolName = text(event.tool_name) || "tool";
  const reason = text(event.reason);
  const equivalent = text(event.equivalent_to);
  const progress = asRecord(event.progress);
  return {
    id: `control:${index}`,
    category: "policy",
    provenance: "policy",
    title: `Tool policy · ${action}`,
    summary: reason || `Runtime policy intercepted ${toolName}.`,
    source: text(event.provenance) || "runtime_tool_loop",
    sourceIndex: index,
    sequence: finiteNumber(event.sequence),
    createdAtMs: finiteNumber(event.created_at_ms),
    status: "advisory",
    reason: reason || "The session API did not include a human-readable policy reason.",
    evidence: equivalent ? [`equivalent_to:${equivalent}`] : [],
    receipts: [
      { label: "Tool", value: toolName },
      { label: "Arguments", value: bounded(event.arguments, 1_000) || "None recorded" },
      { label: "Equivalent to", value: equivalent || "Not recorded" },
      { label: "Failed attempts", value: String(event.failed_attempts ?? 0) },
      { label: "Progress", value: progress ? bounded(progress, 1_000) : "Not recorded" },
    ],
    missingEvidence: reason ? [] : ["Policy reason was not included in this receipt."],
  };
}

function feedNode(event: FeedEvent, index: number): ExecutionNode {
  const kind = text(event.kind) || "computer";
  const action = text(event.action) || "event";
  const detail = text(event.title) || text(event.snippet) || text(event.url);
  return {
    id: `feed:${index}`,
    category: "browse",
    provenance: "observed",
    title: `${kind} · ${action}`,
    summary: detail || "Live-feed event with no descriptive payload.",
    source: `live-feed${event.engine ? `:${event.engine}` : ""}`,
    sourceIndex: index,
    sequence: finiteNumber(event.sequence),
    createdAtMs: finiteNumber(event.created_at_ms),
    status: event.ok === false ? "error" : "ok",
    reason: "Observed by the Helix live computer/browse feed.",
    evidence: [text(event.url), text(event.title)].filter(Boolean),
    receipts: [
      { label: "Turn", value: text(event.turn_id) || "Not recorded" },
      { label: "URL", value: text(event.url) || "Not recorded" },
      { label: "Title", value: text(event.title) || "Not recorded" },
      { label: "Snippet", value: bounded(event.snippet, 1_000) || "Not recorded" },
    ],
    missingEvidence: [],
  };
}

function evidenceNodes(
  adaptive: Record<string, unknown>,
  options: { idPrefix?: string; source?: string; createdAtMs?: number | null } = {},
): ExecutionNode[] {
  const claims = Array.isArray(adaptive.evidence) ? adaptive.evidence : [];
  return claims.flatMap((value, index) => {
    const claim = asRecord(value);
    if (!claim) return [];
    const supporting = strings(claim.supporting_evidence);
    const refs = strings(claim.evidence_refs);
    const missing = strings(claim.missing_evidence);
    const contradicting = strings(claim.contradicting_evidence);
    const status = text(claim.status) || "UNVERIFIED";
    return [{
      id: `${options.idPrefix ?? "analysis"}:evidence:${index}`,
      category: "evidence" as const,
      provenance: "policy" as const,
      title: `Evidence · ${text(claim.claim_id) || `claim ${index + 1}`}`,
      summary: text(claim.claim) || "Evidence claim",
      source: options.source ?? "manual-analysis:evidence-gate",
      sourceIndex: index,
      sequence: null,
      createdAtMs: options.createdAtMs ?? null,
      status: contradicting.length ? "error" as const : supporting.length ? "ok" as const : "unknown" as const,
      reason: `Helix evidence gate classified this claim as ${status}.`,
      evidence: [...refs, ...supporting, ...contradicting],
      receipts: [
        { label: "Status", value: status },
        { label: "Evidence refs", value: refs.join(", ") || "None" },
        { label: "Supporting", value: supporting.join(" · ") || "None" },
        { label: "Contradicting", value: contradicting.join(" · ") || "None" },
      ],
      missingEvidence: missing,
    }];
  });
}

function analysisNodes(analysis: HelixAnalyzeResult): ExecutionNode[] {
  const nodes: ExecutionNode[] = [];
  nodes.push({
    id: "analysis:route",
    category: "learning",
    provenance: "policy",
    title: "Manual Helix analysis",
    summary: `Route ${analysis.route || "unknown"} · decision ${analysis.decision || "unknown"}`,
    source: "manual /api/helix-engine/analyze",
    sourceIndex: null,
    sequence: null,
    createdAtMs: null,
    status: "advisory",
    reason: "This is the explicit Analyze control result, not the final post-turn ingest receipt.",
    evidence: [],
    receipts: [
      { label: "Route", value: analysis.route || "Not reported" },
      { label: "Decision", value: analysis.decision || "Not reported" },
      { label: "Gates", value: (analysis.gates ?? []).join(" → ") || "Not reported" },
      { label: "Kept step indexes", value: (analysis.credit ?? []).join(", ") || "None" },
      {
        label: "Success authorizes weights",
        value: String(analysis.success_authorizes_weight_update ?? false),
      },
    ],
    missingEvidence: ["Final turn-ingest action receipt is not exposed by the current session API."],
  });

  const adaptive = asRecord(analysis.adaptive_cycle);
  if (!adaptive) return nodes;

  const audit = asRecord(adaptive.self_audit);
  if (audit) {
    const auditSource = text(audit.source) || "model";
    const modelAuthored = auditSource !== "deterministic_fallback";
    nodes.push({
      id: "analysis:self-audit",
      category: "adaptation",
      provenance: modelAuthored ? "model-audit" : "policy",
      title: modelAuthored ? "Model / self-audit" : "Deterministic audit fallback",
      summary: text(audit.recommendation_reason) || text(audit.objective) || "Self-audit report",
      source: auditSource,
      sourceIndex: null,
      sequence: null,
      createdAtMs: null,
      status: "advisory",
      reason: modelAuthored
        ? "Self-assessment is model-authored analysis, not objective execution evidence."
        : "Backend fallback analysis was derived deterministically from observable execution receipts; it is policy analysis, not model self-assessment or objective proof.",
      evidence: [
        ...strings(audit.contributing_actions),
        ...strings(audit.failures),
        ...strings(audit.reusable_lessons),
      ],
      receipts: [
        { label: "Achieved", value: String(audit.achieved ?? "unknown") },
        { label: "Recommendation", value: text(audit.recommendation) || "Not reported" },
        { label: "Confidence", value: String(audit.self_assessment_confidence ?? "unknown") },
        { label: "Model", value: text(audit.model_id) || "Not reported" },
      ],
      missingEvidence: [],
    });
  }

  nodes.push(...evidenceNodes(adaptive));

  const adaptation = asRecord(adaptive.adaptation);
  if (adaptation) {
    const action = text(adaptation.action) || "IGNORE";
    nodes.push({
      id: "analysis:adaptation",
      category: action === "QLORA_CANDIDATE" ? "qlora" : "adaptation",
      provenance: "policy",
      title: `Hermes / Helix adaptation · ${action}`,
      summary: text(adaptation.reason) || "No adaptation reason reported.",
      source: "closed-loop adjudication",
      sourceIndex: null,
      sequence: null,
      createdAtMs: null,
      status: "advisory",
      reason: text(adaptation.reason) || "The adaptation receipt omitted a reason.",
      evidence: strings(adaptation.evidence_ids),
      receipts: [
        { label: "Action", value: action },
        { label: "QLoRA eligible", value: String(adaptation.qlora_eligible ?? false) },
        { label: "Recurrence", value: String(adaptation.recurrence_count ?? "unknown") },
        {
          label: "Source trajectories",
          value: strings(adaptation.source_trajectory_ids).join(", ") || "None reported",
        },
        {
          label: "Rejected claims",
          value: strings(adaptation.rejected_claim_ids).join(", ") || "None",
        },
      ],
      missingEvidence: [],
    });
  }

  const shadows = asRecord(adaptive.shadow_decisions);
  if (shadows) {
    for (const [kind, value] of Object.entries(shadows).sort(([a], [b]) => a.localeCompare(b))) {
      const decision = asRecord(value);
      if (!decision) continue;
      nodes.push({
        id: `analysis:shadow:${kind}`,
        category: kind === "QLORA_CANDIDATE" ? "qlora" : "policy",
        provenance: "policy",
        title: `Shadow decision · ${kind}`,
        summary: `${text(decision.choice) || "SKIP"} · p=${String(decision.probability ?? "unknown")}`,
        source: text(decision.policy_version) || "shadow policy",
        sourceIndex: null,
        sequence: null,
        createdAtMs: null,
        status: "advisory",
        reason: "Advisory policy output; it does not itself authorize execution or training.",
        evidence: [],
        receipts: [
          { label: "Choice", value: text(decision.choice) || "Not reported" },
          { label: "Probability", value: String(decision.probability ?? "unknown") },
          { label: "Confidence", value: String(decision.confidence ?? "unknown") },
          { label: "Features", value: bounded(decision.evidence_features, 1_400) || "None" },
        ],
        missingEvidence: [],
      });
    }
  }

  for (const action of analysis.actions ?? []) {
    nodes.push({
      id: `analysis:action:${action}`,
      category: action.includes("qlora") ? "qlora" : "learning",
      provenance: "policy",
      title: `Learning action · ${action}`,
      summary: "Action receipt returned by Helix.",
      source: "analysis action receipt",
      sourceIndex: null,
      sequence: null,
      createdAtMs: null,
      status: "advisory",
      reason: "Returned action receipt; authority depends on the endpoint that produced it.",
      evidence: [],
      receipts: [{ label: "Action", value: action }],
      missingEvidence: [],
    });
  }
  return nodes;
}

function checkpointBoundaryNode(
  checkpoint: Record<string, unknown>,
  index: number,
  boundary: "start" | "resume",
): ExecutionNode | null {
  const sequence = finiteNumber(checkpoint[`${boundary}_sequence`]);
  const createdAtMs = finiteNumber(checkpoint[`${boundary}_created_at_ms`]);
  if (sequence == null && createdAtMs == null) return null;
  const trajectory = text(checkpoint.trajectory_id);
  const missingEvidence = [
    ...(sequence == null ? [`${boundary}_sequence was not persisted for this checkpoint boundary.`] : []),
    ...(createdAtMs == null
      ? [`${boundary}_created_at_ms was not persisted for this checkpoint boundary.`]
      : []),
  ];
  return {
    id: `provenance:checkpoint:${index}:${boundary}`,
    category: "checkpoint",
    provenance: "observed",
    title: `Adaptive checkpoint · ${boundary}`,
    summary:
      boundary === "start"
        ? "Foreground execution entered the adaptive checkpoint boundary."
        : "Foreground execution resumed after the adaptive checkpoint boundary.",
    source: CHECKPOINT_BOUNDARY_SOURCE,
    sourceIndex: index,
    sequence,
    createdAtMs,
    status: missingEvidence.length ? "unknown" : "ok",
    reason:
      "The backend persisted this boundary from the shared Helix event clock for the correlated adaptive checkpoint.",
    evidence: trajectory ? [`trajectory:${trajectory}`] : [],
    receipts: [
      { label: "Boundary", value: boundary },
      { label: "Trajectory", value: trajectory || "Not reported" },
      { label: "Sequence", value: sequence == null ? "Not reported" : String(sequence) },
      {
        label: "Created at ms",
        value: createdAtMs == null ? "Not reported" : String(createdAtMs),
      },
    ],
    missingEvidence,
  };
}

function checkpointNodes(provenance: HelixProvenance): ExecutionNode[] {
  return (provenance.adaptive_checkpoints ?? []).flatMap((checkpoint, index) => {
    const mechanisms = asRecord(checkpoint.mechanisms);
    const mem0 = asRecord(mechanisms?.mem0);
    const retrieval = asRecord(mem0?.retrieval);
    const update = asRecord(mem0?.update);
    const helixHermes = asRecord(mechanisms?.helix_hermes);
    const skill = asRecord(mechanisms?.temporary_skill);
    const qlora = asRecord(mechanisms?.qlora);
    const actions = strings(checkpoint.actions);
    const missing: string[] = [];
    if (!retrieval) missing.push("Memory retrieval receipt was not attached to this checkpoint.");
    if (!helixHermes) missing.push("Helix/Hermes checkpoint analysis receipt was not attached.");
    if (!skill) missing.push("Temporary-skill disposition receipt was not attached.");
    if (!qlora) missing.push("QLoRA candidacy receipt was not attached.");
    const checkpointNode = {
      id: `provenance:checkpoint:${index}`,
      category: "checkpoint",
      provenance: "policy",
      title: "Adaptive checkpoint receipt",
      summary:
        text(helixHermes?.adaptation_action) || actions.join(", ") || "Checkpoint completed",
      source: "provenance:adaptive-checkpoint",
      sourceIndex: index,
      sequence: finiteNumber(checkpoint.audit_sequence) ?? finiteNumber(checkpoint.sequence),
      createdAtMs:
        finiteNumber(checkpoint.audit_created_at_ms) ?? finiteNumber(checkpoint.created_at_ms),
      status: helixHermes?.analysis_performed === false ? "error" : "advisory",
      reason:
        text(retrieval?.reason) ||
        "Backend checkpoint receipt records memory, Helix/Hermes, skill, and QLoRA mechanisms.",
      evidence: [
        ...strings(asRecord(checkpoint.adaptation)?.evidence_ids),
        ...strings(asRecord(qlora?.shadow_decision)?.evidence_ids),
      ],
      receipts: [
        { label: "Trajectory", value: text(checkpoint.trajectory_id) || "Not reported" },
        {
          label: "Start boundary",
          value:
            finiteNumber(checkpoint.start_sequence) == null &&
            finiteNumber(checkpoint.start_created_at_ms) == null
              ? "Not reported"
              : `sequence ${finiteNumber(checkpoint.start_sequence) ?? "?"} · ${finiteNumber(checkpoint.start_created_at_ms) ?? "timestamp unavailable"}`,
        },
        {
          label: "Audit boundary",
          value:
            finiteNumber(checkpoint.audit_sequence) == null &&
            finiteNumber(checkpoint.audit_created_at_ms) == null
              ? "Not reported"
              : `sequence ${finiteNumber(checkpoint.audit_sequence) ?? "?"} · ${finiteNumber(checkpoint.audit_created_at_ms) ?? "timestamp unavailable"}`,
        },
        {
          label: "Resume boundary",
          value:
            finiteNumber(checkpoint.resume_sequence) == null &&
            finiteNumber(checkpoint.resume_created_at_ms) == null
              ? "Not reported"
              : `sequence ${finiteNumber(checkpoint.resume_sequence) ?? "?"} · ${finiteNumber(checkpoint.resume_created_at_ms) ?? "timestamp unavailable"}`,
        },
        { label: "Memory retrieval", value: bounded(retrieval, 1_300) || "Not reported" },
        { label: "Memory update", value: bounded(update, 1_300) || "Not reported" },
        { label: "Helix / Hermes", value: bounded(helixHermes, 1_300) || "Not reported" },
        { label: "Temporary skill", value: bounded(skill, 1_300) || "Not reported" },
        { label: "QLoRA", value: bounded(qlora, 1_300) || "Not reported" },
        { label: "Actions", value: actions.join(", ") || "None" },
        { label: "Training deferred", value: String(checkpoint.training_deferred ?? true) },
      ],
      missingEvidence: missing,
    } satisfies ExecutionNode;
    return [
      checkpointBoundaryNode(checkpoint, index, "start"),
      checkpointNode,
      checkpointBoundaryNode(checkpoint, index, "resume"),
    ].filter((node): node is ExecutionNode => Boolean(node));
  });
}

function preflightNodes(provenance: HelixProvenance): ExecutionNode[] {
  const telemetry = asRecord(provenance.trajectory?.telemetry);
  const preflight = asRecord(telemetry?.preflight);
  if (!preflight) return [];
  const nodes: ExecutionNode[] = [];
  const memory = asRecord(preflight.memory) ?? asRecord(preflight.mem0);
  if (memory) {
    const hits = Array.isArray(memory.hits)
      ? memory.hits.map(asRecord).filter((hit): hit is Record<string, unknown> => Boolean(hit))
      : [];
    nodes.push({
      id: "provenance:preflight:memory",
      category: "memory",
      provenance: "observed",
      title: "Memory preflight retrieval",
      summary:
        hits.length > 0
          ? `${hits.length} memory hit${hits.length === 1 ? "" : "s"} retrieved before foreground execution.`
          : text(memory.reason) || "Memory preflight completed with no recorded hits.",
      source: text(memory.provenance) || "provenance:trajectory.telemetry.preflight.mem0",
      sourceIndex: null,
      sequence: finiteNumber(memory.sequence),
      createdAtMs: finiteNumber(memory.created_at_ms),
      status: memory.available === false ? "unknown" : "ok",
      reason: text(memory.reason) || "Preflight retrieval receipt captured before foreground execution.",
      evidence: hits.flatMap((hit) => [text(hit.id), text(hit.title)].filter(Boolean)),
      receipts: [
        { label: "Available", value: String(memory.available ?? "unknown") },
        { label: "Reason", value: text(memory.reason) || "Not reported" },
        {
          label: "Hits",
          value:
            hits
              .map((hit) => {
                const label = text(hit.title) || text(hit.id) || "memory";
                const score = finiteNumber(hit.score);
                return `${label}${score == null ? "" : ` (${score})`}${text(hit.kind) ? ` · ${text(hit.kind)}` : ""}`;
              })
              .join("\n") || "None",
        },
        {
          label: "Previews",
          value: hits.map((hit) => text(hit.excerpt) || text(hit.preview)).filter(Boolean).join("\n\n") || "None",
        },
        { label: "Attempted", value: String(memory.attempted ?? "unknown") },
        { label: "Hit count", value: String(memory.hitCount ?? hits.length) },
      ],
      missingEvidence:
        memory.available === false
          ? ["Memory preflight was unavailable, so hit evidence could not be collected."]
          : [],
    });
  }
  const skills = asRecord(preflight.skills);
  if (skills) {
    const relevantRecords = Array.isArray(skills.relevant)
      ? skills.relevant.map(asRecord).filter((item): item is Record<string, unknown> => Boolean(item))
      : [];
    const matchedSkills = relevantRecords.length
      ? relevantRecords.map((item) => text(item.name)).filter(Boolean)
      : strings(skills.matchedSkills);
    const enabledSkills = strings(skills.enabledSkills);
    nodes.push({
      id: "provenance:preflight:skills",
      category: "skill",
      provenance: "policy",
      title: "Skill preflight match",
      summary:
        matchedSkills.length > 0
          ? `Matched ${matchedSkills.join(", ")}.`
          : skills.noExistingMatch === true
            ? skills.learnSkillInstructionOffered === true
              ? "No existing skill matched; learn-skill instructions were offered."
              : "No existing skill matched."
            : "Skill preflight completed without a recorded match.",
      source: text(skills.provenance) || "provenance:trajectory.telemetry.preflight.skills",
      sourceIndex: null,
      sequence: finiteNumber(skills.sequence),
      createdAtMs: finiteNumber(skills.created_at_ms),
      status: "advisory",
      reason: "Preflight match narrows candidate skills; actual use requires an observed read_skill/tool receipt.",
      evidence: matchedSkills,
      receipts: [
        { label: "Attempted", value: String(skills.attempted ?? "unknown") },
        { label: "Enabled count", value: String(skills.enabledSkillCount ?? skills.enabledCount ?? enabledSkills.length) },
        { label: "Enabled skills", value: enabledSkills.join(", ") || "None recorded" },
        { label: "Matched skills", value: matchedSkills.join(", ") || "None" },
        { label: "No existing match", value: String(skills.noExistingMatch ?? (matchedSkills.length === 0)) },
        {
          label: "Learn-skill instruction offered",
          value: String(skills.learnSkillInstructionOffered ?? "unknown"),
        },
        { label: "Error", value: text(skills.error) || "None" },
      ],
      missingEvidence: text(skills.error)
        ? [`Skill preflight error: ${text(skills.error)}`]
        : [],
    });
  }
  return nodes;
}

function finalSelfAuditNode(provenance: HelixProvenance): ExecutionNode[] {
  const audit = provenance.self_audit;
  if (!audit) return [];
  const auditSource = text(audit.source) || "model";
  const modelAuthored = auditSource !== "deterministic_fallback";
  return [{
    id: "provenance:self-audit",
    category: "adaptation",
    provenance: modelAuthored ? "model-audit" : "policy",
    title: modelAuthored ? "Final self-audit" : "Final deterministic audit fallback",
    summary: text(audit.recommendation_reason) || text(audit.objective) || "Completed-turn self-audit",
    source: auditSource,
    sourceIndex: null,
    sequence: finiteNumber(audit.sequence),
    createdAtMs: finiteNumber(audit.created_at_ms),
    status: "advisory",
    reason: modelAuthored
      ? "Persisted self-audit is model/self-assessment, not objective proof."
      : "Persisted fallback audit is deterministic backend analysis over observable receipts, not model self-assessment or objective proof.",
    evidence: [
      ...strings(audit.contributing_actions),
      ...strings(audit.failures),
      ...strings(audit.reusable_lessons),
    ],
    receipts: [
      { label: "Achieved", value: String(audit.achieved ?? "unknown") },
      { label: "Recommendation", value: text(audit.recommendation) || "Not reported" },
      { label: "Reason", value: text(audit.recommendation_reason) || "Not reported" },
      { label: "Confidence", value: String(audit.self_assessment_confidence ?? "unknown") },
      { label: "Model", value: text(audit.model_id) || "Not reported" },
    ],
    missingEvidence: [],
  }];
}

function finalEvidenceNodes(provenance: HelixProvenance): ExecutionNode[] {
  const evidence = provenance.evidence;
  if (!evidence) return [];
  return evidenceNodes(
    { evidence: Array.isArray(evidence.claims) ? evidence.claims : [] },
    {
      idPrefix: "provenance",
      source: "provenance:evidence",
      createdAtMs: finiteNumber(evidence.created_at_ms),
    },
  );
}

function finalAdaptationNode(provenance: HelixProvenance): ExecutionNode[] {
  const adaptation = provenance.adaptation;
  if (!adaptation) return [];
  const action = text(adaptation.action) || "IGNORE";
  return [{
    id: "provenance:adaptation",
    category: action === "QLORA_CANDIDATE" ? "qlora" : "adaptation",
    provenance: "policy",
    title: `Final Hermes / Helix adaptation · ${action}`,
    summary: text(adaptation.reason) || "No adaptation rationale was persisted.",
    source: "provenance:adaptation",
    sourceIndex: null,
    sequence: finiteNumber(adaptation.sequence),
    createdAtMs: finiteNumber(adaptation.created_at_ms),
    status: "advisory",
    reason: text(adaptation.reason) || "Persisted adaptation receipt omitted a reason.",
    evidence: strings(adaptation.evidence_ids),
    receipts: [
      { label: "Action", value: action },
      { label: "QLoRA eligible", value: String(adaptation.qlora_eligible ?? false) },
      { label: "Recurrence", value: String(adaptation.recurrence_count ?? "unknown") },
      { label: "Pattern", value: text(adaptation.pattern_fingerprint) || "Not reported" },
      {
        label: "Source trajectories",
        value: strings(adaptation.source_trajectory_ids).join(", ") || "None reported",
      },
      {
        label: "Rejected claims",
        value: strings(adaptation.rejected_claim_ids).join(", ") || "None",
      },
    ],
    missingEvidence: [],
  }];
}

function decisionNodes(provenance: HelixProvenance): ExecutionNode[] {
  const runtimeKinds = new Set([
    "ESCALATE_MODEL",
    "USE_LOCAL_MODEL",
    "USE_ASTRA",
    "USE_SPECULATIVE_DECODING",
  ]);
  return (provenance.decisions ?? []).map((decision, index) => {
    const kind = text(decision.decision) || "UNKNOWN";
    const category: ExecutionCategory = kind === "QLORA_CANDIDATE"
      ? "qlora"
      : runtimeKinds.has(kind)
        ? "runtime"
        : kind === "EVIDENCE_SUFFICIENT"
          ? "evidence"
          : "policy";
    return {
      id: `provenance:decision:${text(decision.decision_id) || index}`,
      category,
      provenance: "policy",
      title: `Policy decision · ${kind}`,
      summary: `${text(decision.choice) || "SKIP"} · p=${String(decision.probability ?? "unknown")}`,
      source: text(decision.policy_version) || "provenance:decision",
      sourceIndex: index,
      sequence: finiteNumber(decision.sequence),
      createdAtMs: finiteNumber(decision.created_at_ms),
      status: "advisory",
      reason: "Persisted advisory policy decision and its backend-observed feature vector.",
      evidence: [],
      receipts: [
        { label: "Choice", value: text(decision.choice) || "Not reported" },
        { label: "Probability", value: String(decision.probability ?? "unknown") },
        { label: "Confidence", value: String(decision.confidence ?? "unknown") },
        { label: "Features", value: bounded(decision.evidence_features, 1_500) || "None" },
        { label: "Eventual outcome", value: String(decision.eventual_outcome ?? "not labelled") },
        {
          label: "Retrospective usefulness",
          value: String(decision.retrospective_usefulness ?? "not labelled"),
        },
      ],
      missingEvidence: [],
    } satisfies ExecutionNode;
  });
}

function qloraReceiptNodes(provenance: HelixProvenance): ExecutionNode[] {
  const nodes: ExecutionNode[] = [];
  if (provenance.training_target_receipt) {
    const receipt = provenance.training_target_receipt;
    nodes.push({
      id: "provenance:training-target",
      category: "evidence",
      provenance: "observed",
      title: "Verified training-target receipt",
      summary: `${text(receipt.source) || "backend verification"} · ${text(receipt.trajectory_id) || "trajectory"}`,
      source: text(receipt.provenance) || "backend_verified_target_ingress",
      sourceIndex: null,
      sequence: finiteNumber(receipt.sequence),
      createdAtMs: finiteNumber(receipt.created_at_ms),
      status: "ok",
      reason: "Backend-owned target-verification receipt; this is evidence, not a model assertion.",
      evidence: strings(receipt.evidence_refs),
      receipts: [
        { label: "Source", value: text(receipt.source) || "Not reported" },
        { label: "Source ref", value: text(receipt.source_ref) || "Not reported" },
        { label: "Thread", value: text(receipt.source_thread_id) || "Not reported" },
        { label: "Evidence refs", value: strings(receipt.evidence_refs).join(", ") || "None" },
      ],
      missingEvidence: [],
    });
  }
  if (provenance.qlora_admission) {
    const receipt = provenance.qlora_admission;
    nodes.push({
      id: "provenance:qlora-admission",
      category: "qlora",
      provenance: "policy",
      title: "QLoRA admission receipt",
      summary: `Recurrence ${String(receipt.recurrence_count ?? "unknown")} · training target ${receipt.training_target_verified === true ? "verified" : "unverified"}`,
      source: "provenance:qlora-admission",
      sourceIndex: null,
      sequence: finiteNumber(receipt.sequence),
      createdAtMs: finiteNumber(receipt.created_at_ms),
      status: "advisory",
      reason: "Persisted admission gate; it proves candidacy admission, not that training started.",
      evidence: strings(receipt.evidence_ids),
      receipts: [
        { label: "Model", value: text(receipt.model_id) || "Not reported" },
        { label: "Pattern", value: text(receipt.pattern_fingerprint) || "Not reported" },
        { label: "Evidence digest", value: text(receipt.evidence_sha256) || "Not reported" },
        { label: "Training target verified", value: String(receipt.training_target_verified ?? false) },
      ],
      missingEvidence: [],
    });
  }
  return nodes;
}

function structuredSkillRetentionReceipts(provenance: HelixProvenance) {
  return (provenance.skill_retention ?? []).filter(
    (receipt) => text(receipt.skill_name) && text(receipt.disposition),
  );
}

function skillRetentionNodes(provenance: HelixProvenance): ExecutionNode[] {
  const createdAtMs = finiteNumber(provenance.turn_receipt?.created_at_ms);
  return structuredSkillRetentionReceipts(provenance).map((receipt, index) => {
    const skillName = text(receipt.skill_name);
    const disposition = text(receipt.disposition);
    const reason = text(receipt.reason);
    const evidence = asRecord(receipt.evidence);
    const evidenceItems = evidence
      ? Object.entries(evidence).map(([key, value]) => `${key}=${bounded(value, 240)}`)
      : [];
    return {
      id: `provenance:skill-retention:${index}`,
      category: "skill",
      provenance: "policy",
      title: `Temporary skill outcome · ${disposition}`,
      summary: skillName,
      source: "provenance:skill-retention",
      sourceIndex: index,
      sequence: null,
      createdAtMs,
      status:
        disposition === "promoted" || disposition === "discarded"
          ? "ok"
          : disposition === "unknown"
            ? "unknown"
            : "advisory",
      reason: reason || "The structured skill-retention receipt omitted its rationale.",
      evidence: evidenceItems,
      receipts: [
        { label: "Skill", value: skillName },
        { label: "Disposition", value: disposition },
        { label: "Reason", value: reason || "Not reported" },
        { label: "Retention predicates", value: bounded(evidence, 1_500) || "Not reported" },
      ],
      missingEvidence: reason
        ? []
        : ["Skill-retention rationale was not included in the structured backend receipt."],
    } satisfies ExecutionNode;
  });
}

function qloraOutcomeNode(provenance: HelixProvenance): ExecutionNode[] {
  const receipt = provenance.qlora_outcome;
  if (!receipt) return [];
  const outcome = text(receipt.outcome);
  const reason = text(receipt.reason);
  if (!outcome && !reason) return [];
  return [{
    id: "provenance:qlora-outcome",
    category: "qlora",
    provenance: "policy",
    title: `QLoRA outcome · ${outcome || "unreported"}`,
    summary: reason || "Final QLoRA outcome receipt omitted its reason.",
    source: "provenance:qlora-outcome",
    sourceIndex: null,
    sequence: null,
    createdAtMs: finiteNumber(provenance.turn_receipt?.created_at_ms),
    status: outcome === "queued" ? "ok" : outcome ? "advisory" : "unknown",
    reason: reason || "The structured QLoRA outcome receipt omitted its reason.",
    evidence: [],
    receipts: [
      { label: "Outcome", value: outcome || "Not reported" },
      { label: "Reason", value: reason || "Not reported" },
    ],
    missingEvidence: [
      ...(!outcome ? ["QLoRA outcome was not included in the structured backend receipt."] : []),
      ...(!reason ? ["QLoRA outcome reason was not included in the structured backend receipt."] : []),
    ],
  }];
}

function finalLearningNode(provenance: HelixProvenance): ExecutionNode[] {
  const receipt = provenance.turn_receipt;
  const actions = provenance.final_actions ?? strings(receipt?.actions);
  const hasStructuredSkillRetention = structuredSkillRetentionReceipts(provenance).length > 0;
  const hasStructuredQloraOutcome = Boolean(
    text(provenance.qlora_outcome?.outcome) || text(provenance.qlora_outcome?.reason),
  );
  if (!receipt && !actions.length) return [];
  const nodes: ExecutionNode[] = [{
    id: "provenance:final-learning",
    category: "learning",
    provenance: "policy",
    title: "Final learning decision",
    summary: actions.join(", ") || "Completed turn with no learning action",
    source: "provenance:turn-receipt",
    sourceIndex: null,
    sequence: finiteNumber(receipt?.sequence),
    createdAtMs: finiteNumber(receipt?.created_at_ms),
    status: "advisory",
    reason: "Final post-turn receipt records the actions that actually survived ingestion and gating.",
    evidence: [],
    receipts: [
      { label: "Turn", value: text(receipt?.turn_id) || "Not reported" },
      { label: "Trajectory", value: text(receipt?.trajectory_id) || "Not reported" },
      { label: "Model", value: text(receipt?.model) || "Not reported" },
      { label: "Base model", value: text(receipt?.base_model_id) || "Not reported" },
      { label: "Final actions", value: actions.join(", ") || "None" },
    ],
    missingEvidence: [],
  }];
  for (const [index, action] of actions.entries()) {
    if (!hasStructuredSkillRetention && /^(?:promote|discard|retain)_temp_skill/.test(action)) {
      const [verb, name = "unknown"] = action.split(":", 2);
      nodes.push({
        id: `provenance:skill-retention:${index}`,
        category: "skill",
        provenance: "policy",
        title: `Temporary skill outcome · ${verb.replaceAll("_", " ")}`,
        summary: name,
        source: "provenance:turn-receipt",
        sourceIndex: index,
        sequence: null,
        createdAtMs: finiteNumber(receipt?.created_at_ms),
        status: verb.startsWith("retain_") ? "unknown" : "ok",
        reason: "The final turn receipt proves the retention outcome. Its predicate-level rationale was not persisted.",
        evidence: [action],
        receipts: [{ label: "Final action", value: action }],
        missingEvidence: [
          "Skill-retention rationale (for example created/read/clean-finish predicates) is not attached to the final action receipt.",
        ],
      });
    }
    if (
      !hasStructuredQloraOutcome &&
      (action === "stage_qlora_candidate" || action === "queue_qlora_training")
    ) {
      nodes.push({
        id: `provenance:qlora-outcome:${index}`,
        category: "qlora",
        provenance: "policy",
        title:
          action === "queue_qlora_training"
            ? "QLoRA outcome · queued"
            : "QLoRA outcome · candidate staged",
        summary: action,
        source: "provenance:turn-receipt",
        sourceIndex: index,
        sequence: null,
        createdAtMs: finiteNumber(receipt?.created_at_ms),
        status: "advisory",
        reason:
          action === "queue_qlora_training"
            ? "Final turn receipt confirms the training queue accepted the eligible candidate."
            : "Final turn receipt confirms the candidate survived final ingest and was staged for queue admission.",
        evidence: [action],
        receipts: [{ label: "Final action", value: action }],
        missingEvidence:
          action === "stage_qlora_candidate"
            ? ["If the candidate was not queued, the queue denial/defer reason is not persisted in this receipt."]
            : [],
      });
    }
  }
  return nodes;
}

function provenanceNodes(provenance: HelixProvenance | null | undefined): ExecutionNode[] {
  if (!provenance?.available) return [];
  return [
    ...preflightNodes(provenance),
    ...checkpointNodes(provenance),
    ...finalEvidenceNodes(provenance),
    ...finalSelfAuditNode(provenance),
    ...finalAdaptationNode(provenance),
    ...decisionNodes(provenance),
    ...qloraReceiptNodes(provenance),
    ...skillRetentionNodes(provenance),
    ...qloraOutcomeNode(provenance),
    ...finalLearningNode(provenance),
  ];
}

export function normalizeExecutionGraph(input: ExecutionGraphInput): ExecutionGraph {
  const nodes: ExecutionNode[] = [];
  const edges: ExecutionEdge[] = [];

  const controlNodes = input.controlEvents.map(controlNode);
  const provenanceToolSteps = Array.isArray(input.provenance?.trajectory?.tool_steps)
    ? input.provenance.trajectory.tool_steps
        .map(asRecord)
        .filter((step): step is Record<string, unknown> => Boolean(step))
    : [];
  const stepNodes = input.steps.map((step, index) => {
    const matchingReceipt = provenanceToolSteps.find((candidate) => {
      const candidateSequence = finiteNumber(candidate.sequence);
      if (step.sequence != null && candidateSequence != null) {
        return candidateSequence === step.sequence;
      }
      return finiteNumber(candidate.index) === (step.index ?? index);
    });
    return stepNode(step, index, asRecord(matchingReceipt?.verification));
  });
  const feedNodes = input.events.map(feedNode);
  const persistedNodes = provenanceNodes(input.provenance);
  const sequencedCheckpointEvents = persistedNodes.filter(
    (node) =>
      node.sequence != null &&
      (node.source === CHECKPOINT_BOUNDARY_SOURCE || node.source === "provenance:adaptive-checkpoint"),
  );
  const correlatedProvenanceNodes = persistedNodes.filter(
    (node) =>
      !(
        node.sequence != null &&
        (node.source === CHECKPOINT_BOUNDARY_SOURCE || node.source === "provenance:adaptive-checkpoint")
      ),
  );
  const observedNodes = [
    ...controlNodes,
    ...stepNodes,
    ...feedNodes,
    ...sequencedCheckpointEvents,
  ];
  const orderingMode: ExecutionGraph["orderingMode"] =
    observedNodes.length > 0 && observedNodes.every((node) => node.sequence != null)
      ? "exact"
      : "source-local";

  if (orderingMode === "exact") {
    observedNodes.sort(
      (left, right) =>
        (left.sequence ?? 0) - (right.sequence ?? 0) ||
        (left.createdAtMs ?? 0) - (right.createdAtMs ?? 0) ||
        left.id.localeCompare(right.id),
    );
    nodes.push(...observedNodes);
    addSequentialEdges(
      edges,
      observedNodes.map((node) => node.id),
      "exact-order",
    );
  } else {
    nodes.push(...controlNodes, ...stepNodes, ...feedNodes, ...sequencedCheckpointEvents);
    addSequentialEdges(edges, controlNodes.map((node) => node.id), "control-order");
    addSequentialEdges(edges, stepNodes.map((node) => node.id), "step-order");
    addSequentialEdges(edges, feedNodes.map((node) => node.id), "feed-order");
    addSequentialEdges(
      edges,
      sequencedCheckpointEvents.map((node) => node.id),
      "checkpoint-boundary-order",
    );
  }

  if (input.analysis) {
    nodes.push(...analysisNodes(input.analysis));
  }
  nodes.push(...correlatedProvenanceNodes);

  const hasMemoryTool = input.steps.some((step) => MEMORY_TOOLS.has(step.name));
  const hasSkillTool = input.steps.some((step) =>
    SKILL_READ_TOOLS.has(step.name) || SKILL_CREATE_TOOLS.has(step.name),
  );
  const createdTemporarySkill = input.steps.some((step) => {
    if (step.name === "learn_skill") return true;
    if (step.name !== "create_skill") return false;
    return jsonRecord(step.arguments)?.temporary === true;
  });
  const telemetry = asRecord(input.provenance?.trajectory?.telemetry);
  const preflight = asRecord(telemetry?.preflight);
  const memoryPreflight = asRecord(preflight?.memory) ?? asRecord(preflight?.mem0);
  const skillPreflight = asRecord(preflight?.skills);
  const relevantSkillRecords = Array.isArray(skillPreflight?.relevant)
    ? skillPreflight.relevant
        .map(asRecord)
        .filter((item): item is Record<string, unknown> => Boolean(item))
    : [];
  const relevantSkillMatches = Math.max(
    relevantSkillRecords.filter((item) => Boolean(text(item.name))).length,
    strings(skillPreflight?.matchedSkills).length,
  );
  const hasMemoryReceipt = hasMemoryTool || Boolean(memoryPreflight);
  const hasSkillPreflight = Boolean(skillPreflight);
  const finalActions = input.provenance?.final_actions ?? strings(input.provenance?.turn_receipt?.actions);
  const checkpointReceipts = input.provenance?.adaptive_checkpoints ?? [];
  const hasCheckpoint = checkpointReceipts.length > 0;
  const hasCompleteCheckpointBoundaries =
    hasCheckpoint &&
    checkpointReceipts.every(
      (checkpoint) =>
        finiteNumber(checkpoint.start_sequence) != null &&
        finiteNumber(checkpoint.start_created_at_ms) != null &&
        finiteNumber(checkpoint.audit_sequence) != null &&
        finiteNumber(checkpoint.audit_created_at_ms) != null &&
        finiteNumber(checkpoint.resume_sequence) != null &&
        finiteNumber(checkpoint.resume_created_at_ms) != null,
    );
  const hasSkillRetention =
    structuredSkillRetentionReceipts(input.provenance ?? {}).length > 0 ||
    finalActions.some((action) => /(?:promote|discard|retain)_temp_skill/.test(action));
  const runtimeDecisionKinds = new Set([
    "ESCALATE_MODEL",
    "USE_LOCAL_MODEL",
    "USE_ASTRA",
    "USE_SPECULATIVE_DECODING",
  ]);
  const hasRuntimeDecision = (input.provenance?.decisions ?? []).some((decision) =>
    runtimeDecisionKinds.has(text(decision.decision)),
  );
  const hasFinalLearning = Boolean(
    input.provenance?.turn_receipt || input.provenance?.final_actions?.length,
  );
  const finalAdaptationAction = text(input.provenance?.adaptation?.action);
  const finalQloraCandidate =
    finalAdaptationAction === "QLORA_CANDIDATE" ||
    finalActions.includes("stage_qlora_candidate") ||
    Boolean(input.provenance?.qlora_admission);
  const qloraQueued = finalActions.includes("queue_qlora_training");
  const hasQloraOutcomeReason = Boolean(
    text(input.provenance?.qlora_outcome?.outcome) && text(input.provenance?.qlora_outcome?.reason),
  );
  const coverageGaps = [
    ...(!hasMemoryReceipt
      ? ["Preflight Mem0 retrieval/hit details are not exposed as a turn-correlated receipt."]
      : []),
    ...(!hasSkillPreflight && !hasSkillTool
      ? ["Skill preflight match/selection receipt is not exposed for this turn."]
      : []),
    ...(hasSkillPreflight && relevantSkillMatches > 0 && !hasSkillTool
      ? ["Skill preflight matched candidates, but no read_skill/learn_skill execution receipt proves actual use."]
      : []),
    ...(createdTemporarySkill && !hasSkillRetention
      ? ["Temporary-skill retention/promotion/discard outcome is not present in the final turn receipt."]
      : []),
    ...(!hasCheckpoint
      ? ["Adaptive checkpoint receipt is not available for this turn."]
      : !hasCompleteCheckpointBoundaries
        ? ["Adaptive checkpoint start/resume boundary metadata is incomplete; a start, audit, or resume sequence/timestamp is missing."]
        : []),
    ...(orderingMode !== "exact"
      ? ["Cross-stream timestamps/sequence IDs are incomplete, so tool, policy, and browse lanes cannot be interleaved without guessing."]
      : []),
    ...(input.events.length > 0 && input.events.some((event) => !text(event.turn_id))
      ? ["Live-feed observations are thread-scoped but lack turn_id, so membership in the latest logical turn cannot be proven."]
      : []),
    ...(!hasRuntimeDecision
      ? ["Model escalation/runtime-selection decision receipts are not present for this turn."]
      : []),
    ...(finalQloraCandidate && !qloraQueued && !hasQloraOutcomeReason
      ? ["QLoRA candidate was retained/staged, but the final queue/start decision reason is not recorded."]
      : []),
    ...(!hasFinalLearning
      ? ["Final post-turn learning/ingest decision receipt is not available for this turn."]
      : []),
    ...(input.provenanceError
      ? [`Provenance endpoint unavailable: ${input.provenanceError}`]
      : []),
    ...(input.provenance?.response_truncated ? ["Provenance response was truncated by the backend size bound."] : []),
    ...((input.provenance?.truncated_sources ?? []).map(
      (source) => `Backend provenance source omitted by the response size bound: ${source}.`,
    )),
    ...((input.provenance?.missing_sources ?? [])
      .filter(
        (source) =>
          source !== "mem0_preflight_retrieval_details" || !memoryPreflight,
      )
      .map((source) => `Backend provenance source missing: ${source}.`)),
  ];

  return {
    nodes,
    edges,
    orderingMode,
    coverageGaps: [...new Set(coverageGaps)],
  };
}
