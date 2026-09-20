// SPDX-License-Identifier: AGPL-3.0-only
// Cheap, bounded context/cache observations for the Helix adaptive cycle.

type MessageLike = { role?: unknown; content?: unknown };

export type HelixContextObservationInput = {
  scope: string;
  systemPrompt: string;
  userSystemPrompt?: string;
  toolCatalogSignature: string;
  messages: MessageLike[];
  promptTokens: number;
  cachedTokens: number;
};

type PreviousContext = {
  systemHash: string;
  userSystemHash: string;
  toolCatalogHash: string;
  messageHashes: string[];
};

const previousByScope = new Map<string, PreviousContext>();
const MAX_SCOPES = 256;

function hashText(value: string): string {
  // FNV-1a is sufficient here: this is a change detector, not a security digest.
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function boundedContent(value: unknown): string {
  if (typeof value === "string") return value.slice(0, 16_000);
  if (value === null || value === undefined) return "";
  try {
    return JSON.stringify(value).slice(0, 16_000);
  } catch {
    return String(value).slice(0, 16_000);
  }
}

function messageFingerprint(message: MessageLike): string {
  return hashText(`${String(message.role ?? "")}\u0000${boundedContent(message.content)}`);
}

function markerInsertion(systemPrompt: string, marker: string, kind: string) {
  const start = `<${marker}>`;
  const end = `</${marker}>`;
  const at = systemPrompt.indexOf(start);
  if (at < 0) return null;
  const endAt = systemPrompt.indexOf(end, at + start.length);
  const content = systemPrompt.slice(
    at,
    endAt >= 0 ? Math.min(systemPrompt.length, endAt + end.length) : Math.min(systemPrompt.length, at + 8_000),
  );
  return {
    kind,
    fingerprint: hashText(content),
    repeated: false,
    required: true,
    estimated_tokens: Math.max(1, Math.ceil(content.length / 4)),
    provenance: "frontend_context_constructor",
  };
}

export function observeHelixContextTelemetry(
  input: HelixContextObservationInput,
): Record<string, unknown> {
  const scope = input.scope.trim().slice(0, 500) || "default";
  const systemHash = hashText(input.systemPrompt.slice(0, 64_000));
  const userSystemHash = hashText(String(input.userSystemPrompt ?? "").slice(0, 32_000));
  const toolCatalogHash = hashText(input.toolCatalogSignature.slice(0, 16_000));
  const boundedMessages = input.messages.slice(-64);
  const messageHashes = boundedMessages.map(messageFingerprint);
  const prior = previousByScope.get(scope);

  const contextInsertions: Array<Record<string, unknown>> = [];
  for (const marker of [
    markerInsertion(input.systemPrompt, "skill_preflight", "skill"),
    markerInsertion(input.systemPrompt, "project_instructions", "project_instructions"),
    markerInsertion(input.systemPrompt, "mem0_hits", "memory"),
  ]) {
    if (marker) contextInsertions.push(marker);
  }

  const seenToolOutputs = new Set<string>();
  for (const message of boundedMessages) {
    if (message.role !== "tool") continue;
    const text = boundedContent(message.content);
    if (!text) continue;
    const fingerprint = hashText(text);
    const repeated = seenToolOutputs.has(fingerprint);
    seenToolOutputs.add(fingerprint);
    contextInsertions.push({
      kind: "tool_output",
      fingerprint,
      repeated,
      required: false,
      estimated_tokens: Math.max(1, Math.ceil(text.length / 4)),
      provenance: "frontend_outbound_tool_history",
    });
  }

  let contextReorders = 0;
  if (prior && prior.messageHashes.length === messageHashes.length) {
    const sameMembers = [...prior.messageHashes].sort().join("|") === [...messageHashes].sort().join("|");
    if (sameMembers && prior.messageHashes.join("|") !== messageHashes.join("|")) contextReorders = 1;
  }

  const systemPromptChanges = prior && prior.systemHash !== systemHash ? 1 : 0;
  const userChangedSystemPrompt = Boolean(prior && prior.userSystemHash !== userSystemHash);
  const toolSchemaChanges = prior && prior.toolCatalogHash !== toolCatalogHash ? 1 : 0;
  const repeatedContextInsertions = contextInsertions.filter((item) => item.repeated === true).length;

  previousByScope.delete(scope);
  previousByScope.set(scope, { systemHash, userSystemHash, toolCatalogHash, messageHashes });
  while (previousByScope.size > MAX_SCOPES) {
    const oldest = previousByScope.keys().next().value as string | undefined;
    if (!oldest) break;
    previousByScope.delete(oldest);
  }

  const promptTokens = Math.max(0, Math.trunc(input.promptTokens || 0));
  const cachedTokens = Math.max(0, Math.min(promptTokens || Number.MAX_SAFE_INTEGER, Math.trunc(input.cachedTokens || 0)));
  return {
    stable_prefix_tokens: cachedTokens,
    newly_evaluated_tokens: promptTokens ? Math.max(0, promptTokens - cachedTokens) : 0,
    system_prompt_changes: systemPromptChanges,
    user_changed_system_prompt: userChangedSystemPrompt,
    tool_schema_changes: toolSchemaChanges,
    tool_schema_change_required: Boolean(toolSchemaChanges),
    // A fingerprint delta tells us the schema changed, but not which actor caused
    // the feature/catalog transition. Preserve that uncertainty instead of
    // attributing it to the user merely because a UI flag changed.
    tool_schema_change_cause: "UNKNOWN",
    context_reorders: contextReorders,
    prompt_reconstructions: 0,
    context_insertions: contextInsertions,
    repeated_context_insertions: repeatedContextInsertions,
    context_fingerprint: hashText(messageHashes.join("|")),
    system_prompt_fingerprint: systemHash,
    tool_schema_fingerprint: toolCatalogHash,
    telemetry_provenance: {
      stable_prefix_tokens: "server_prompt_cache_cached_tokens",
      newly_evaluated_tokens: "derived_server_prompt_minus_cached",
      system_prompt_changes: prior ? "frontend_system_prompt_fingerprint_delta" : "first_observation",
      tool_schema_changes: prior ? "frontend_tool_catalog_fingerprint_delta" : "first_observation",
      context_reorders: prior ? "frontend_bounded_message_fingerprint_order" : "first_observation",
      prompt_reconstructions: "frontend_observed_no_reconstruction_event",
      context_insertions: "frontend_context_constructor_and_outbound_tool_history",
      repeated_context_insertions: "frontend_duplicate_tool_output_fingerprint",
      kv_cache_resets: "unavailable",
    },
  };
}

export function resetHelixContextTelemetryForTests(): void {
  previousByScope.clear();
}
