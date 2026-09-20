// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import {
  auditToLegacyCritic,
  helixSelfAuditPrompt,
  helixSelfCriticPrompt,
  hermesLearningReviewPrompt,
  isHelixSelfAuditRequest,
  isHelixSelfCriticRequest,
  isHermesLearningReviewRequest,
  parseHelixSelfAudit,
  parseHelixSelfCritic,
  parseOnTheFlySkillDraft,
  parseHermesLearningProposal,
  stripOnTheFlySkillDraft,
  stripHermesLearningProposal,
} from "../src/features/chat/lib/hermes-learning.ts";

function auditArtifacts(prompt: string): Record<string, unknown> {
  const marker = "Observable artifacts JSON: ";
  const offset = prompt.lastIndexOf(marker);
  assert.notEqual(offset, -1);
  return JSON.parse(prompt.slice(offset + marker.length)) as Record<string, unknown>;
}

test("Hermes review prompt is detectable and uses the staged proposal protocol", () => {
  const prompt = hermesLearningReviewPrompt("the loader fix");
  assert.equal(isHermesLearningReviewRequest(prompt), true);
  assert.match(prompt, /unsloth-learning/);
  assert.match(prompt, /Do not claim that anything was saved/);
});

test("learning marker is parsed and removed from the visible answer", () => {
  const answer = `Verified result.\n<unsloth-learning>{"kind":"memory","title":"Keep checks","content":"Run the focused test after changes.","reason":"Reusable verification habit."}</unsloth-learning>`;
  assert.deepEqual(parseHermesLearningProposal(answer), {
    kind: "memory",
    title: "Keep checks",
    content: "Run the focused test after changes.",
    reason: "Reusable verification habit.",
  });
  assert.equal(stripHermesLearningProposal(answer), "Verified result.");
});

test("on-the-fly skill drafts are parsed, bounded, and hidden from the answer", () => {
  const answer = `Completed the task.\n<unsloth-skill-draft>{"name":"json-checks","title":"Verify JSON contracts","content":"Compare parsed values against the contract.","reason":"The same procedure recurred."}</unsloth-skill-draft>`;
  assert.deepEqual(parseOnTheFlySkillDraft(answer), {
    kind: "skill",
    name: "json-checks",
    title: "Verify JSON contracts",
    content: "Compare parsed values against the contract.",
    target: "codex",
    recommendationAction: "skill",
    reason: "The same procedure recurred.",
    recommendationReason:
      "A repeatable procedure was identified; stage it as a skill before considering QLoRA.",
  });
  assert.equal(stripOnTheFlySkillDraft(answer), "Completed the task.");
});

test("self-critic prompt is per-turn and parsed into Helix ingest fields", () => {
  const prompt = helixSelfCriticPrompt("User: ship the check\nAssistant: still missing a test");
  assert.equal(isHelixSelfCriticRequest(prompt), true);
  assert.match(prompt, /Did I finish the user's work/);
  assert.match(prompt, /too many tools/);
  const critic = parseHelixSelfCritic(
    '<helix-self-critic>{"finished":"partial","rightTools":false,"rightSkills":false,"tooManyTools":true,"toolCount":9,"usedSkills":[],"recommendation":"skill","skillTitle":"json-checks","skillContent":"Read the skill first.","reason":"Existing skill covers this."}</helix-self-critic>',
  );
  assert.equal(critic?.finished, "partial");
  assert.equal(critic?.tooManyTools, true);
  assert.equal(critic?.recommendation, "skill");
  assert.equal(critic?.skillTitle, "json-checks");
});

test("learning recommendations accept runtime repair without turning it into a score", () => {
  const proposal = parseHermesLearningProposal(
    '<unsloth-learning>{"kind":"memory","title":"Runtime repair","content":"Revert a failing adapter before retraining.","recommendationAction":"runtime-fix","recommendationReason":"The active candidate failed."}</unsloth-learning>',
  );
  assert.equal(proposal?.recommendationAction, "runtime-fix");
  assert.equal(proposal?.recommendationReason, "The active candidate failed.");
});

test("self-audit artifacts remain valid JSON under oversized telemetry", () => {
  const telemetry: Record<string, unknown> = Object.fromEntries(
    Array.from({ length: 120 }, (_, index) => [`noise_${index}`, "x".repeat(2_000)]),
  );
  telemetry.prompt_tokens = 1_234;
  telemetry.cached_tokens = 800;
  telemetry.accepted_drafts = 7;
  const prompt = helixSelfAuditPrompt("oversized telemetry", {
    objective: "inspect the run",
    final_result: "done",
    telemetry,
    tool_steps: [],
  });
  assert.equal(isHelixSelfAuditRequest(prompt), true);
  const artifacts = auditArtifacts(prompt);
  assert.equal((artifacts.telemetry as Record<string, unknown>).prompt_tokens, 1_234);
  assert.equal((artifacts.telemetry as Record<string, unknown>).cached_tokens, 800);
  assert.match(String((artifacts.telemetry as Record<string, unknown>).helix_artifact_compaction), /summary|structured/);
});

test("self-audit tool slicing preserves backend evidence indexes", () => {
  const toolSteps = Array.from({ length: 20 }, (_, index) => ({
    index,
    evidence_ids: [`tool:${index}:result`],
    name: "read_file",
    arguments: `file-${index}`,
    result: `result-${index}`,
  }));
  const artifacts = auditArtifacts(helixSelfAuditPrompt("index preservation", {
    objective: "inspect files",
    final_result: "done",
    telemetry: {},
    tool_steps: toolSteps,
  }));
  const tools = artifacts.tool_steps as Array<Record<string, unknown>>;
  assert.equal(tools.length, 16);
  assert.equal(tools[0]?.index, 4);
  assert.deepEqual(tools[0]?.evidence_ids, ["tool:4:result"]);
  assert.equal(tools.at(-1)?.index, 19);
  assert.deepEqual(tools.at(-1)?.evidence_ids, ["tool:19:result"]);
});

test("self-audit prompt preserves backend-resolved evidence/cache/test artifacts without hidden reasoning", () => {
  const prompt = helixSelfAuditPrompt("rich backend bundle", {
    schema_version: "helix.audit-input.v1",
    objective: "verify the implementation",
    presented_context: "public context only",
    final_result: "implemented",
    acceptance_criteria: ["tests pass"],
    telemetry: { prompt_tokens: 100, cached_tokens: 40 },
    cache_integrity: { cache_reuse_ratio: 0.4, unavailable_fields: ["kv_cache_resets"] },
    evidence: [{ claim_id: "task-outcome", status: "UNVERIFIED", missing_evidence: ["test"] }],
    control_events: [
      {
        schema_version: "helix.tool-control.v1",
        action: "equivalent_duplicate",
        tool_name: "search_conversation",
        equivalent_to: "search_memory",
      },
    ],
    objective_outcome_evidence: [{ claim_id: "task-outcome", status: "UNVERIFIED" }],
    edits: [{ name: "edit_file", result: "changed" }],
    tests: [{ name: "pytest", verification: { kind: "test", status: "passed" } }],
    benchmarks: [{ name: "bench", verification: { kind: "benchmark", status: "passed" } }],
    trajectory: {
      schema_version: "helix.trajectory.v1",
      trajectory_id: "t1",
      reasoning: "private reasoning must never be forwarded",
      tool_steps: [{ result: "duplicate that should be stripped from trajectory envelope" }],
    },
    pre_audit_decision: { decision: "DEEP_SELF_AUDIT", choice: "TAKE" },
    tool_steps: [],
  });
  const artifacts = auditArtifacts(prompt);
  assert.deepEqual(artifacts.acceptance_criteria, ["tests pass"]);
  assert.equal((artifacts.cache_integrity as Record<string, unknown>).cache_reuse_ratio, 0.4);
  assert.equal((artifacts.evidence as Array<Record<string, unknown>>)[0]?.status, "UNVERIFIED");
  assert.equal(
    (artifacts.control_events as Array<Record<string, unknown>>)[0]?.action,
    "equivalent_duplicate",
  );
  assert.equal((artifacts.tests as Array<Record<string, unknown>>)[0]?.name, "pytest");
  assert.equal((artifacts.benchmarks as Array<Record<string, unknown>>)[0]?.name, "bench");
  const trajectory = artifacts.trajectory as Record<string, unknown>;
  assert.equal(trajectory.tool_steps, undefined);
  assert.equal(trajectory.reasoning, undefined);
  assert.match(prompt, /Every top-level array other than claims must contain strings only and at most 4 items/);
  assert.match(prompt, /Do not place recommendation fields inside reusable_lessons/);
});

test("self-audit parser keeps only evidence-reference grammar the backend can resolve", () => {
  const audit = parseHelixSelfAudit(
    '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"IGNORE","claims":[{"claim":"checked","evidence_refs":["tool:19:verification","benchmark:speed","tool:abc:result","tool:0:../../","made-up"]}]}</helix-self-audit>',
  );
  assert.deepEqual(audit?.claims[0]?.evidence_refs, ["tool:19:verification", "benchmark:speed"]);
});

test("self-audit parser bounds model-authored arrays and claim count", () => {
  const claims = Array.from({ length: 12 }, (_, index) => ({
    claim: `claim-${index}`,
    supporting_evidence: Array.from({ length: 8 }, (__, item) => `support-${item}`),
    missing_evidence: Array.from({ length: 8 }, (__, item) => `missing-${item}`),
  }));
  const audit = parseHelixSelfAudit(
    `<helix-self-audit>${JSON.stringify({
      objective: "x",
      achieved: true,
      recommendation: "IGNORE",
      reusable_lessons: Array.from({ length: 8 }, (_, index) => `lesson-${index}`),
      claims,
    })}</helix-self-audit>`,
  );
  assert.equal(audit?.reusable_lessons.length, 4);
  assert.equal(audit?.claims.length, 8);
  assert.equal(audit?.claims[0]?.supporting_evidence?.length, 4);
  assert.equal(audit?.claims[0]?.missing_evidence?.length, 4);
});

test("self-audit parser accepts only a complete JSON payload when the closing tag is omitted", () => {
  const audit = parseHelixSelfAudit(
    '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"IGNORE"}',
  );
  assert.ok(audit);
  assert.equal(audit.objective, "x");
  assert.equal(
    parseHelixSelfAudit(
      '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"IGNORE"} trailing prose',
    ),
    null,
  );
  assert.equal(
    parseHelixSelfAudit('<helix-self-audit>{"objective":"x","achieved":true'),
    null,
  );
});

test("self-audit QLoRA recommendation cannot re-enter the legacy critic path", () => {
  const audit = parseHelixSelfAudit(
    '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"QLORA_CANDIDATE","recommendation_reason":"model request"}</helix-self-audit>',
  );
  assert.ok(audit);
  assert.equal(auditToLegacyCritic(audit).recommendation, "none");
});
