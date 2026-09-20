// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { parseUnslothDeepLink } from "../src/features/deep-links/parse-deep-link.ts";

test("the registered Helix scheme reaches the existing Hugging Face intent parser", () => {
  assert.deepEqual(
    parseUnslothDeepLink(
      "helixharness://open_from_hf?model=Qwen/Qwen3-8B&file=Qwen3-8B-Q4_K_M.gguf",
    ),
    {
      model: "Qwen/Qwen3-8B",
      file: "Qwen3-8B-Q4_K_M.gguf",
    },
  );
});

test("legacy Unsloth deep links remain parseable during the app transition", () => {
  assert.deepEqual(
    parseUnslothDeepLink("unsloth://open_from_hf?model=Qwen/Qwen3-8B"),
    { model: "Qwen/Qwen3-8B" },
  );
});

test("unregistered or malformed schemes are rejected", () => {
  assert.equal(
    parseUnslothDeepLink("https://open_from_hf?model=Qwen/Qwen3-8B"),
    null,
  );
  assert.equal(
    parseUnslothDeepLink("helixharness://open_from_hf/extra?model=Qwen/Qwen3-8B"),
    null,
  );
});
