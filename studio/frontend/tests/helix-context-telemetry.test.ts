// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import {
  observeHelixContextTelemetry,
  resetHelixContextTelemetryForTests,
} from "../src/features/chat/lib/helix-context-telemetry.ts";

test("context telemetry derives stable-prefix/new-prefill counts and fingerprints context", () => {
  resetHelixContextTelemetryForTests();
  const telemetry = observeHelixContextTelemetry({
    scope: "thread-1::model",
    systemPrompt: "<skill_preflight>same skill catalog</skill_preflight>",
    toolCatalogSignature: "terminal,read_file",
    messages: [
      { role: "user", content: "inspect" },
      { role: "tool", content: "same output" },
      { role: "tool", content: "same output" },
    ],
    promptTokens: 1_000,
    cachedTokens: 700,
  });
  assert.equal(telemetry.stable_prefix_tokens, 700);
  assert.equal(telemetry.newly_evaluated_tokens, 300);
  assert.equal(telemetry.repeated_context_insertions, 1);
  const insertions = telemetry.context_insertions as Array<Record<string, unknown>>;
  assert.ok(insertions.some((item) => item.kind === "skill"));
  assert.ok(insertions.some((item) => item.kind === "tool_output" && item.repeated === true));
  assert.equal((telemetry.telemetry_provenance as Record<string, unknown>).kv_cache_resets, "unavailable");
});

test("context telemetry detects system/tool-catalog changes and only calls reorder for same members", () => {
  resetHelixContextTelemetryForTests();
  const base = {
    scope: "thread-2::model",
    promptTokens: 100,
    cachedTokens: 10,
  };
  observeHelixContextTelemetry({
    ...base,
    systemPrompt: "alpha",
    toolCatalogSignature: "read",
    messages: [{ role: "user", content: "a" }, { role: "assistant", content: "b" }],
  });
  const changed = observeHelixContextTelemetry({
    ...base,
    systemPrompt: "beta",
    toolCatalogSignature: "read,terminal",
    messages: [{ role: "assistant", content: "b" }, { role: "user", content: "a" }],
  });
  assert.equal(changed.system_prompt_changes, 1);
  assert.equal(changed.tool_schema_changes, 1);
  assert.equal(changed.context_reorders, 1);
});
