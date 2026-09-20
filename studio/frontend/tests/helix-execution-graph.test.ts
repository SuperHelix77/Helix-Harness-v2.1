// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";

import { normalizeExecutionGraph } from "../src/features/helix-engine/execution-graph.ts";

test("execution graph preserves source-local order without inventing cross-stream causality", () => {
  const graph = normalizeExecutionGraph({
    controlEvents: [
      {
        action: "equivalent_duplicate",
        tool_name: "search_memory",
        reason: "same query already resolved",
        equivalent_to: "tool:0",
        provenance: "runtime_tool_loop",
      },
      {
        action: "repeated_failure",
        tool_name: "terminal",
        reason: "retry budget exhausted",
        provenance: "runtime_tool_loop",
      },
    ],
    steps: [
      {
        index: 0,
        evidence_ids: ["tool:0:result", "tool:0:verification"],
        name: "search_memory",
        arguments: '{"query":"prior decision"}',
        result: "1 memory hit",
        useful_hint: "evidence",
        retry: 0,
      },
      {
        index: 1,
        evidence_ids: ["tool:1:result"],
        name: "read_skill",
        arguments: '{"name":"repo-audit"}',
        result: "skill instructions",
        useful_hint: "useful",
        retry: 0,
      },
      {
        index: 2,
        evidence_ids: ["tool:2:error"],
        name: "terminal",
        arguments: "npm test",
        result: "Error: exit code 1",
        useful_hint: "wrong",
        error: "Error: exit code 1",
        retry: 0,
      },
    ],
    events: [{ kind: "browse", action: "navigate", url: "https://example.test", ok: true }],
  });

  assert.equal(graph.nodes.find((node) => node.id === "step:0")?.category, "memory");
  assert.equal(graph.nodes.find((node) => node.id === "step:1")?.category, "skill");
  assert.equal(graph.nodes.find((node) => node.id === "step:2")?.status, "error");
  assert.equal(graph.nodes.find((node) => node.id === "control:0")?.provenance, "policy");
  assert.equal(graph.nodes.find((node) => node.id === "feed:0")?.provenance, "observed");

  assert.deepEqual(
    graph.edges.map((edge) => [edge.from, edge.to, edge.relation]),
    [
      ["control:0", "control:1", "observed-sequence"],
      ["step:0", "step:1", "observed-sequence"],
      ["step:1", "step:2", "observed-sequence"],
    ],
  );
  assert.ok(
    graph.coverageGaps.some((gap) => gap.includes("Cross-stream timestamps/sequence IDs")),
  );
});

test("manual analysis distinguishes self-audit, evidence gates, Hermes adaptation, and QLoRA policy", () => {
  const graph = normalizeExecutionGraph({
    events: [],
    steps: [],
    controlEvents: [],
    analysis: {
      route: "hermes",
      decision: "promote_hermes",
      gates: ["observe", "critic", "route"],
      credit: [0],
      success_authorizes_weight_update: false,
      adaptive_cycle: {
        self_audit: {
          source: "model",
          objective: "finish task",
          achieved: true,
          recommendation: "QLORA_CANDIDATE",
          recommendation_reason: "repeated behavior gap",
          self_assessment_confidence: 0.7,
          model_id: "local-model",
        },
        evidence: [
          {
            claim_id: "task-outcome",
            claim: "task completed",
            status: "UNVERIFIED",
            supporting_evidence: [],
            evidence_refs: [],
            contradicting_evidence: [],
            missing_evidence: ["objective outcome verification"],
          },
        ],
        adaptation: {
          action: "QLORA_CANDIDATE",
          reason: "recurring evidence-backed pattern",
          qlora_eligible: false,
          recurrence_count: 3,
          evidence_ids: ["task-outcome"],
          source_trajectory_ids: ["t1", "t2", "t3"],
        },
        shadow_decisions: {
          QLORA_CANDIDATE: {
            choice: "TAKE",
            probability: 0.78,
            confidence: 0.56,
            policy_version: "helix-shadow-v1",
            evidence_features: { recurrence_count: 3 },
          },
        },
      },
    },
  });

  assert.equal(graph.nodes.find((node) => node.id === "analysis:self-audit")?.provenance, "model-audit");
  assert.equal(graph.nodes.find((node) => node.id === "analysis:evidence:0")?.provenance, "policy");
  assert.deepEqual(
    graph.nodes.find((node) => node.id === "analysis:evidence:0")?.missingEvidence,
    ["objective outcome verification"],
  );
  assert.equal(graph.nodes.find((node) => node.id === "analysis:adaptation")?.category, "qlora");
  assert.equal(
    graph.nodes.find((node) => node.id === "analysis:shadow:QLORA_CANDIDATE")?.status,
    "advisory",
  );
  assert.ok(graph.coverageGaps.some((gap) => gap.includes("Final post-turn learning/ingest")));
  assert.ok(graph.coverageGaps.some((gap) => gap.includes("Adaptive checkpoint receipt")));
  assert.ok(
    !graph.coverageGaps.some((gap) => gap.includes("QLoRA candidate was retained/staged")),
    "manual Analyze candidacy is advisory and must not masquerade as a final queue obligation",
  );
});

test("deterministic fallback audit is not mislabeled as model self-assessment", () => {
  const graph = normalizeExecutionGraph({
    steps: [],
    controlEvents: [],
    events: [],
    analysis: {
      adaptive_cycle: {
        self_audit: {
          source: "deterministic_fallback",
          recommendation: "IGNORE",
          recommendation_reason: "derived from observable trajectory",
        },
      },
    },
    provenance: {
      available: true,
      self_audit: {
        source: "deterministic_fallback",
        recommendation: "IGNORE",
        recommendation_reason: "derived from observable trajectory",
      },
      missing_sources: [],
    },
  });

  const manual = graph.nodes.find((node) => node.id === "analysis:self-audit");
  const persisted = graph.nodes.find((node) => node.id === "provenance:self-audit");
  assert.equal(manual?.provenance, "policy");
  assert.equal(manual?.source, "deterministic_fallback");
  assert.equal(persisted?.provenance, "policy");
  assert.equal(persisted?.source, "deterministic_fallback");
  assert.ok(persisted?.reason.includes("not model self-assessment"));
});

test("skill and memory tool receipts suppress only the gaps they actually satisfy", () => {
  const graph = normalizeExecutionGraph({
    events: [],
    controlEvents: [],
    steps: [
      {
        name: "search_memory",
        arguments: "{}",
        result: "hit",
        useful_hint: "useful",
      },
      {
        name: "learn_skill",
        arguments: "{}",
        result: "created temporary skill",
        useful_hint: "useful",
      },
    ],
  });

  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Preflight Mem0 retrieval")));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Existing/temporary skill selection")));
  assert.ok(graph.coverageGaps.some((gap) => gap.startsWith("Temporary-skill retention")));
});

test("shared backend sequence interleaves observed tool, policy, and browse receipts exactly", () => {
  const graph = normalizeExecutionGraph({
    steps: [
      {
        sequence: 2,
        created_at_ms: 200,
        name: "search_memory",
        arguments: "{}",
        result: "hit",
        useful_hint: "useful",
      },
    ],
    controlEvents: [
      {
        sequence: 1,
        created_at_ms: 100,
        action: "duplicate",
        tool_name: "search_memory",
        reason: "deduplicated",
      },
    ],
    events: [
      {
        sequence: 3,
        created_at_ms: 300,
        kind: "browse",
        action: "navigate",
        url: "https://example.test",
      },
    ],
  });

  assert.equal(graph.orderingMode, "exact");
  assert.deepEqual(graph.nodes.slice(0, 3).map((node) => node.id), [
    "control:0",
    "step:0",
    "feed:0",
  ]);
  assert.deepEqual(graph.edges.map((edge) => [edge.from, edge.to]), [
    ["control:0", "step:0"],
    ["step:0", "feed:0"],
  ]);
  assert.ok(!graph.coverageGaps.some((gap) => gap.includes("Cross-stream timestamps/sequence IDs")));
});

test("turn provenance renders preflight, checkpoint, final adaptation, and final learning receipts", () => {
  const graph = normalizeExecutionGraph({
    steps: [],
    controlEvents: [],
    events: [],
    provenance: {
      available: true,
      trajectory: {
        telemetry: {
          preflight: {
            mem0: {
              attempted: true,
              available: true,
              reason: "preflight lookup",
              provenance: "frontend_preflight_mem0",
              hitCount: 1,
              hits: [
                { id: "mem-1", title: "Prior choice", score: 0.91, kind: "mem0", threadId: "t1", preview: "Use path A" },
              ],
            },
            skills: {
              attempted: true,
              enabledCount: 4,
              enabledSkills: ["repo-audit", "skill-creator"],
              matchedSkills: ["repo-audit"],
              noExistingMatch: false,
              learnSkillInstructionOffered: false,
              error: "",
              provenance: "frontend_skill_preflight",
            },
          },
        },
      },
      adaptive_checkpoints: [
        {
          trajectory_id: "turn-1",
          created_at_ms: 1_000,
          actions: ["checkpoint_mem0_update"],
          training_deferred: true,
          mechanisms: {
            mem0: {
              retrieval: { status: "preflight_owned", reason: "preflight owns retrieval" },
              update: { status: "updated" },
            },
            helix_hermes: { analysis_performed: true, adaptation_action: "MEMORY" },
            temporary_skill: { disposition: "KEEP" },
            qlora: { candidate: false, eligible: false },
          },
        },
      ],
      self_audit: { source: "model", recommendation: "MEMORY", recommendation_reason: "retain finding" },
      evidence: {
        claims: [
          {
            claim_id: "outcome",
            claim: "task done",
            status: "SUPPORTED",
            supporting_evidence: ["tool:0:verification"],
            missing_evidence: [],
          },
        ],
      },
      adaptation: { action: "MEMORY", reason: "one-off durable fact", evidence_ids: ["outcome"] },
      decisions: [],
      turn_receipt: {
        turn_id: "turn-1",
        trajectory_id: "turn-1",
        model: "local-model",
        actions: ["promote_temp_skill:repo-audit", "queue_qlora_training"],
        created_at_ms: 2_000,
      },
      final_actions: ["promote_temp_skill:repo-audit", "queue_qlora_training"],
      missing_sources: ["mem0_preflight_retrieval_details"],
    },
  });

  assert.equal(graph.nodes.find((node) => node.id === "provenance:preflight:memory")?.provenance, "observed");
  assert.equal(graph.nodes.find((node) => node.id === "provenance:preflight:skills")?.provenance, "policy");
  assert.ok(graph.nodes.some((node) => node.id === "provenance:checkpoint:0"));
  assert.ok(graph.nodes.some((node) => node.id === "provenance:adaptation"));
  assert.ok(graph.nodes.some((node) => node.id === "provenance:final-learning"));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Preflight Mem0 retrieval")));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Temporary-skill retention")));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("QLoRA queue/start")));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Final post-turn learning")));
  assert.ok(graph.coverageGaps.some((gap) => gap.includes("Adaptive checkpoint start/resume")));
});

test("coverage recognizes the real chat preflight memory and relevant-skill receipt shape", () => {
  const graph = normalizeExecutionGraph({
    steps: [],
    controlEvents: [],
    events: [],
    provenance: {
      available: true,
      trajectory: {
        telemetry: {
          preflight: {
            memory: {
              attempted: true,
              available: true,
              hitCount: 1,
              hits: [{ id: "mem-1", excerpt: "prior result" }],
              provenance: "mem0_rest_search",
            },
            skills: {
              attempted: true,
              enabledSkillCount: 2,
              relevant: [{ name: "repo-audit", description: "Audit repository state" }],
              catalogEmpty: false,
              learnSkillInstructionOffered: true,
              provenance: "skills_catalog_preflight",
            },
          },
        },
      },
      missing_sources: ["mem0_preflight_retrieval_details"],
    },
  });

  assert.ok(graph.nodes.some((node) => node.id === "provenance:preflight:memory"));
  assert.ok(graph.nodes.some((node) => node.id === "provenance:preflight:skills"));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Preflight Mem0 retrieval")));
  assert.ok(
    graph.coverageGaps.some((gap) =>
      gap.includes("Skill preflight matched candidates, but no read_skill/learn_skill execution receipt proves actual use"),
    ),
  );
  assert.ok(
    !graph.coverageGaps.some((gap) => gap === "Backend provenance source missing: mem0_preflight_retrieval_details."),
  );
});

test("typed backend verification is attached to its observed tool receipt", () => {
  const graph = normalizeExecutionGraph({
    controlEvents: [],
    events: [],
    steps: [
      {
        index: 0,
        sequence: 9,
        created_at_ms: 900,
        evidence_ids: ["tool:0:result", "tool:0:verification"],
        name: "terminal",
        arguments: "npm test",
        result: "all tests passed",
        useful_hint: "evidence",
      },
    ],
    provenance: {
      available: true,
      trajectory: {
        tool_steps: [
          {
            index: 0,
            sequence: 9,
            verification: {
              kind: "test",
              status: "passed",
              provenance: "backend_tool_capture",
              detail: "70 tests passed",
              claim: "focused regression suite passes",
              subject: "frontend",
            },
          },
        ],
      },
      missing_sources: [],
    },
  });

  const tool = graph.nodes.find((node) => node.id === "step:0");
  assert.equal(tool?.status, "ok");
  assert.match(tool?.reason ?? "", /Backend verification passed: 70 tests passed/);
  assert.ok(tool?.evidence.includes("focused regression suite passes"));
  assert.ok(tool?.evidence.includes("frontend"));
  assert.match(
    tool?.receipts.find((receipt) => receipt.label === "Verification")?.value ?? "",
    /backend_tool_capture/,
  );
});

test("a final QLoRA candidate without a queue receipt exposes the missing queue reason", () => {
  const graph = normalizeExecutionGraph({
    events: [],
    steps: [],
    controlEvents: [],
    provenance: {
      available: true,
      adaptation: {
        action: "QLORA_CANDIDATE",
        reason: "recurring verified correction",
        qlora_eligible: true,
      },
      qlora_admission: {
        training_target_verified: true,
      },
      turn_receipt: {
        turn_id: "turn-q",
        actions: ["stage_qlora_candidate"],
      },
      final_actions: ["stage_qlora_candidate"],
      missing_sources: [],
    },
  });

  assert.ok(
    graph.coverageGaps.some((gap) =>
      gap.includes("QLoRA candidate was retained/staged, but the final queue/start decision reason is not recorded"),
    ),
  );
});

test("live-feed turn_id is rendered and removes only the turn-membership gap it proves", () => {
  const withTurn = normalizeExecutionGraph({
    steps: [],
    controlEvents: [],
    events: [
      {
        sequence: 1,
        created_at_ms: 100,
        turn_id: "turn-live",
        kind: "browse",
        action: "navigate",
        url: "https://example.test",
      },
    ],
  });

  const feed = withTurn.nodes.find((node) => node.id === "feed:0");
  assert.equal(feed?.receipts.find((receipt) => receipt.label === "Turn")?.value, "turn-live");
  assert.ok(
    !withTurn.coverageGaps.some((gap) => gap.startsWith("Live-feed observations are thread-scoped")),
  );

  const withoutTurn = normalizeExecutionGraph({
    steps: [],
    controlEvents: [],
    events: [{ sequence: 1, created_at_ms: 100, kind: "browse", action: "navigate" }],
  });
  assert.ok(
    withoutTurn.coverageGaps.some((gap) => gap.startsWith("Live-feed observations are thread-scoped")),
  );
});

test("adaptive checkpoint boundary metadata becomes source-backed nodes in shared sequence order", () => {
  const graph = normalizeExecutionGraph({
    controlEvents: [],
    events: [
      {
        sequence: 1,
        created_at_ms: 100,
        turn_id: "turn-checkpoint",
        kind: "computer",
        action: "observe",
      },
    ],
    steps: [
      {
        sequence: 5,
        created_at_ms: 500,
        name: "terminal",
        arguments: "status",
        result: "done",
        useful_hint: "evidence",
      },
    ],
    provenance: {
      available: true,
      adaptive_checkpoints: [
        {
          trajectory_id: "turn-checkpoint",
          start_sequence: 2,
          start_created_at_ms: 200,
          audit_sequence: 3,
          audit_created_at_ms: 300,
          resume_sequence: 4,
          resume_created_at_ms: 400,
          actions: ["checkpoint_mem0_update"],
        },
      ],
      missing_sources: [],
    },
  });

  assert.equal(graph.orderingMode, "exact");
  assert.deepEqual(graph.nodes.slice(0, 5).map((node) => node.id), [
    "feed:0",
    "provenance:checkpoint:0:start",
    "provenance:checkpoint:0",
    "provenance:checkpoint:0:resume",
    "step:0",
  ]);
  assert.deepEqual(graph.edges.map((edge) => [edge.from, edge.to]), [
    ["feed:0", "provenance:checkpoint:0:start"],
    ["provenance:checkpoint:0:start", "provenance:checkpoint:0"],
    ["provenance:checkpoint:0", "provenance:checkpoint:0:resume"],
    ["provenance:checkpoint:0:resume", "step:0"],
  ]);
  const start = graph.nodes.find((node) => node.id === "provenance:checkpoint:0:start");
  assert.equal(start?.sequence, 2);
  assert.equal(start?.createdAtMs, 200);
  assert.equal(start?.source, "provenance:adaptive-checkpoint-boundary");
  const audit = graph.nodes.find((node) => node.id === "provenance:checkpoint:0");
  assert.equal(audit?.sequence, 3);
  assert.equal(audit?.createdAtMs, 300);
  assert.equal(audit?.provenance, "policy");
  assert.equal(
    start?.receipts.find((receipt) => receipt.label === "Trajectory")?.value,
    "turn-checkpoint",
  );
  assert.ok(!graph.coverageGaps.some((gap) => gap.includes("Adaptive checkpoint start/resume")));
});

test("structured skill retention and QLoRA outcome render their backend reasons without inference", () => {
  const graph = normalizeExecutionGraph({
    events: [],
    controlEvents: [],
    steps: [
      {
        name: "learn_skill",
        arguments: '{"name":"draft-skill"}',
        result: "temporary skill created",
        useful_hint: "useful",
      },
    ],
    provenance: {
      available: true,
      turn_receipt: {
        turn_id: "turn-learning",
        created_at_ms: 2_000,
        actions: ["discard_temp_skill:draft-skill", "stage_qlora_candidate"],
      },
      final_actions: ["discard_temp_skill:draft-skill", "stage_qlora_candidate"],
      adaptation: {
        action: "QLORA_CANDIDATE",
        reason: "recurring verified correction",
      },
      skill_retention: [
        {
          skill_name: "draft-skill",
          disposition: "discarded",
          reason: "skill_not_read_successfully_after_creation",
          evidence: {
            created_in_turn: true,
            read_successfully: false,
            task_finished_full: true,
          },
        },
      ],
      qlora_outcome: {
        outcome: "denied",
        reason: "verified_training_target_required",
      },
      missing_sources: [],
    },
  });

  const retention = graph.nodes.find((node) => node.id === "provenance:skill-retention:0");
  assert.equal(retention?.source, "provenance:skill-retention");
  assert.equal(retention?.reason, "skill_not_read_successfully_after_creation");
  assert.ok(retention?.evidence.includes("read_successfully=false"));

  const qlora = graph.nodes.find((node) => node.id === "provenance:qlora-outcome");
  assert.equal(qlora?.summary, "verified_training_target_required");
  assert.equal(qlora?.receipts.find((receipt) => receipt.label === "Outcome")?.value, "denied");
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("Temporary-skill retention")));
  assert.ok(!graph.coverageGaps.some((gap) => gap.startsWith("QLoRA candidate was retained/staged")));

  const partialOutcome = normalizeExecutionGraph({
    events: [],
    controlEvents: [],
    steps: [],
    provenance: {
      available: true,
      adaptation: { action: "QLORA_CANDIDATE" },
      turn_receipt: { turn_id: "turn-partial", actions: ["stage_qlora_candidate"] },
      final_actions: ["stage_qlora_candidate"],
      qlora_outcome: { outcome: "denied" },
      missing_sources: [],
    },
  });
  assert.ok(
    partialOutcome.coverageGaps.some((gap) =>
      gap.includes("QLoRA candidate was retained/staged, but the final queue/start decision reason is not recorded"),
    ),
  );
});

test("response-size truncation names each backend source omitted from the graph", () => {
  const graph = normalizeExecutionGraph({
    events: [],
    controlEvents: [],
    steps: [],
    provenance: {
      available: true,
      response_truncated: true,
      truncated_sources: ["trajectory", "evidence"],
      missing_sources: ["response_size_limit"],
    },
  });

  assert.ok(graph.coverageGaps.includes("Provenance response was truncated by the backend size bound."));
  assert.ok(
    graph.coverageGaps.includes(
      "Backend provenance source omitted by the response size bound: trajectory.",
    ),
  );
  assert.ok(
    graph.coverageGaps.includes(
      "Backend provenance source omitted by the response size bound: evidence.",
    ),
  );
});
