// biome-ignore lint/correctness/noNodejsModules: this file runs in Node's test runner
import assert from "node:assert/strict";
// biome-ignore lint/correctness/noNodejsModules: this file runs in Node's test runner
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveLoadMaxSeqLength } = await import(
  "../src/features/chat/presets/preset-policy.ts"
);
const {
  isQwen38ModelIdentifier,
  preferQwen38MacNativeContext,
  Q38_V11_PREFERRED_CONTEXT_LENGTH,
} = await import("../src/lib/qwen38-v11.ts");

const base = {
  modelId: "unsloth/Qwen3.8-27B-GGUF",
  ggufVariant: "UD-Q4_K_XL",
  isGguf: true,
  customContextLength: null,
  loadedContextLength: 8192,
  currentCheckpoint: "unsloth/Qwen3.8-27B-GGUF",
  activeGgufVariant: "UD-Q4_K_XL",
  pinnedMaxSeqLength: null,
  defaultMaxSeqLength: 4096,
  presetSource: "custom" as const,
};

test("Qwen3.8 Mac Auto reload returns to backend sizing", () => {
  assert.equal(
    preferQwen38MacNativeContext({
      modelId: base.modelId,
      isGguf: true,
      deviceType: "mac",
    }),
    true,
  );
  assert.equal(
    resolveLoadMaxSeqLength({
      ...base,
      preferNativeContext: true,
    }),
    0,
    "a fitted 8K result must not become the next request",
  );
  assert.equal(
    resolveLoadMaxSeqLength(base),
    8192,
    "the legacy replay is retained for other models/platforms",
  );
});

test("an explicit Qwen3.8 context still wins over the native preference", () => {
  assert.equal(
    resolveLoadMaxSeqLength({
      ...base,
      customContextLength: 32768,
      preferNativeContext: true,
    }),
    32768,
  );
});

test("Qwen3.8 context profile is scoped to GGUF Macs", () => {
  assert.equal(isQwen38ModelIdentifier("Qwen3.8-27B-GGUF"), true);
  assert.equal(isQwen38ModelIdentifier("Qwen3.6-35B-GGUF"), false);
  assert.equal(
    preferQwen38MacNativeContext({
      modelId: base.modelId,
      isGguf: true,
      deviceType: "linux",
    }),
    false,
  );
  assert.equal(
    preferQwen38MacNativeContext({
      modelId: base.modelId,
      isGguf: false,
      deviceType: "mac",
    }),
    false,
  );
  assert.equal(Q38_V11_PREFERRED_CONTEXT_LENGTH, 65536);
});
