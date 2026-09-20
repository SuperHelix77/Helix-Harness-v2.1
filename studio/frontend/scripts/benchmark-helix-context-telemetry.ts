// SPDX-License-Identifier: AGPL-3.0-only
// Reproducible matched microbenchmark for the synchronous chat hot-path collector.

import {
  observeHelixContextTelemetry,
  resetHelixContextTelemetryForTests,
} from "../src/features/chat/lib/helix-context-telemetry.ts";

const WARMUP = 2_000;
const ITERATIONS = 20_000;

const input = {
  scope: "benchmark-thread::qwen",
  systemPrompt: [
    "<skill_preflight>read-file: inspect files before edits</skill_preflight>",
    "<project_instructions>Preserve tests.</project_instructions>",
    "<mem0_hits>- The target uses the adaptive cycle.</mem0_hits>",
  ].join("\n"),
  userSystemPrompt: "Be precise.",
  toolCatalogSignature: JSON.stringify({
    localCode: ["python", "terminal"],
    hostedCode: [],
    webSearch: false,
    webFetch: false,
    codeExecution: true,
    imageGeneration: false,
  }),
  messages: Array.from({ length: 24 }, (_, index) => ({
    role: index % 3 === 2 ? "tool" : index % 2 === 0 ? "user" : "assistant",
    content: `bounded-message-${index}-${"x".repeat(240)}`,
  })),
  promptTokens: 8_192,
  cachedTokens: 6_144,
};

function baselineMatchedPreparation(): void {
  // The disabled path still performs the same caller-side property reads and
  // arithmetic that exist regardless of Helix collection.
  const prompt = Math.max(0, Math.trunc(input.promptTokens));
  const cached = Math.max(0, Math.min(prompt, Math.trunc(input.cachedTokens)));
  void (prompt - cached);
  void input.scope;
  void input.systemPrompt;
  void input.toolCatalogSignature;
  void input.messages;
}

function percentile(sorted: number[], fraction: number): number {
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1));
  return sorted[index] ?? 0;
}

function runSample(fn: () => void): { median_us: number; p95_us: number; mean_us: number } {
  const values: number[] = [];
  for (let i = 0; i < WARMUP; i += 1) fn();
  for (let i = 0; i < ITERATIONS; i += 1) {
    const started = process.hrtime.bigint();
    fn();
    values.push(Number(process.hrtime.bigint() - started) / 1_000);
  }
  values.sort((a, b) => a - b);
  return {
    median_us: percentile(values, 0.5),
    p95_us: percentile(values, 0.95),
    mean_us: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

resetHelixContextTelemetryForTests();
const baseline = runSample(baselineMatchedPreparation);
resetHelixContextTelemetryForTests();
const instrumented = runSample(() => {
  observeHelixContextTelemetry(input);
});

const result = {
  schema_version: "helix.telemetry-overhead-benchmark.v1",
  iterations: ITERATIONS,
  warmup_iterations: WARMUP,
  workload: {
    messages: input.messages.length,
    prompt_tokens: input.promptTokens,
    cached_tokens: input.cachedTokens,
    system_prompt_chars: input.systemPrompt.length,
  },
  baseline,
  instrumented,
  overhead: {
    median_us: instrumented.median_us - baseline.median_us,
    p95_us: instrumented.p95_us - baseline.p95_us,
    mean_us: instrumented.mean_us - baseline.mean_us,
    median_percent:
      baseline.median_us > 0
        ? ((instrumented.median_us - baseline.median_us) / baseline.median_us) * 100
        : null,
  },
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
