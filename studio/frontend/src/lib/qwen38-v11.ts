// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The context target used by the Qwen3.8 V1.1 profile on Mac. */
export const Q38_V11_PREFERRED_CONTEXT_LENGTH = 65_536;

function compactModelIdentifier(modelId: string | null | undefined): string {
  return String(modelId ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

/** Keep the client-side identity test aligned with the backend's scoped Qwen3.8 policy. */
export function isQwen38ModelIdentifier(
  modelId: string | null | undefined,
): boolean {
  return compactModelIdentifier(modelId).includes("qwen38");
}

/** Whether an unpinned Mac GGUF reload should return to native backend sizing. */
export function preferQwen38MacNativeContext(options: {
  modelId: string | null | undefined;
  isGguf: boolean | null | undefined;
  deviceType: string | null | undefined;
}): boolean {
  return (
    options.deviceType === "mac" &&
    options.isGguf === true &&
    isQwen38ModelIdentifier(options.modelId)
  );
}
