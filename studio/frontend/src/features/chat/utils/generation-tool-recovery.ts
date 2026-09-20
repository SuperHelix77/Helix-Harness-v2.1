// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SANDBOX_FILE_TOOLS,
  extractCreatedFiles,
} from "@/components/assistant-ui/sandbox-files";
import type { ChatGenerationPendingApproval } from "../api/chat-generation-api";
import {
  SEARCH_IMAGE_TOOL,
  extractSearchImages,
  searchResultText,
} from "../search-images/search-images";
import {
  mergedToolCallArgumentsText,
  toolCallArgumentsText,
} from "../tool-call-arguments";
import type { CarriedPart } from "./chat-generation-recovery";
import {
  newDeepResearchHandoff,
  readDeepResearchToolEvent,
} from "./deep-research-handoff";
import {
  documentCitationToSource,
  parseSourcesFromResult,
} from "./document-citation-source";
import { mergeGoogleNativeParts } from "./google-native-parts";

export type RecoveredToolConfirmation = {
  toolCallId: string;
  approvalId: string;
  sessionId: string;
};

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function recoveredToolResult(
  event: Record<string, unknown>,
  toolName: unknown,
  sessionId: string,
): unknown {
  if (toolName === "image_generation" && typeof event.image_b64 === "string") {
    return {
      image_b64: event.image_b64,
      image_mime: event.image_mime ?? "image/png",
      size: event.size,
      quality: event.quality,
      background: event.background,
      prompt: event.prompt,
    };
  }
  if (typeof event.result !== "string") {
    return event.result ?? "";
  }
  const sandbox =
    typeof toolName === "string" && SANDBOX_FILE_TOOLS.has(toolName);
  const { text, files } = sandbox
    ? extractCreatedFiles(event.result)
    : { text: event.result, files: [] };
  const mcpMarker = "\n__MCP_IMAGES__:";
  const mcpAt = text.lastIndexOf(mcpMarker);
  if (mcpAt !== -1) {
    try {
      const images: unknown = JSON.parse(text.slice(mcpAt + mcpMarker.length));
      if (
        Array.isArray(images) &&
        images.length > 0 &&
        images.every(
          (image) =>
            typeof record(image)?.data === "string" &&
            typeof record(image)?.mimeType === "string",
        )
      ) {
        return { text: text.slice(0, mcpAt), images };
      }
    } catch {
      // Keep malformed envelopes as text.
    }
  }
  const imageMarker = "\n__IMAGES__:";
  const imageAt = text.lastIndexOf(imageMarker);
  if (imageAt !== -1) {
    try {
      const images: unknown = JSON.parse(
        text.slice(imageAt + imageMarker.length),
      );
      if (
        Array.isArray(images) &&
        images.every((image) => typeof image === "string")
      ) {
        return { text: text.slice(0, imageAt), images, sessionId, files };
      }
    } catch {
      // Keep malformed envelopes as text.
    }
  }
  if (sandbox) {
    return { text, images: [], sessionId, files };
  }
  if (toolName === SEARCH_IMAGE_TOOL) {
    const search = extractSearchImages(text);
    if (search.images.length > 0) {
      return { text: search.text, webImages: search.images };
    }
  }
  return text;
}

export function createGenerationToolRecovery(
  carried: CarriedPart[],
  runId: string,
  snapshotSeq = 0,
) {
  const pending = new Map<string, CarriedPart>();
  const endedApprovalIds = new Set<string>();
  const endedConfirmationCardIds: string[] = [];
  const researchHandoff = newDeepResearchHandoff();
  /** The sources a finished search card yields, parsed once per card rather than per publish.
   *  Every rebuild used to re-run the parse over every finished search result in the turn,
   *  which measured as the bulk of a recovery's per-publish cost on a long tool-using reply.
   *  `apply` replaces a card object wholesale rather than editing one, so an entry that is
   *  still the same object still holds the same result; the result is compared as well, so
   *  an in-place edit somewhere else could not make this serve a stale list either. */
  const parsedSources = new WeakMap<
    object,
    { result: unknown; sources: ReturnType<typeof parseSourcesFromResult> }
  >();
  const searchCard = (part: unknown) => {
    const card = record(part);
    return card?.type === "tool-call" &&
      card.result !== undefined &&
      (card.toolName === "web_search" || card.toolName === "web_fetch")
      ? card
      : undefined;
  };
  const cardSources = (card: Record<string, unknown>) => {
    const hit = parsedSources.get(card);
    if (hit && hit.result === card.result) return hit.sources;
    const sources = parseSourcesFromResult(searchResultText(card.result));
    parsedSources.set(card, { result: card.result, sources });
    return sources;
  };
  // A source a previous recovery appended is carried at the offset it was appended AT, so text
  // replayed after it lands behind it and cuts the reply in two, breaking any markdown that
  // spans the cut. `withSources` rebuilds these from the card, so drop them and let every
  // rebuild re-append them, which is also where the live adapter puts them. A citation source
  // is anchored where it arrived and has no card to rebuild it, so it stays.
  const rebuildableSourceIds = new Set(
    carried.flatMap(({ part }) => {
      const card = searchCard(part);
      return card ? cardSources(card).map((source) => source.id) : [];
    }),
  );
  for (let i = carried.length - 1; i >= 0; i--) {
    const part = record(carried[i].part);
    if (
      part?.type === "source" &&
      typeof part.id === "string" &&
      rebuildableSourceIds.has(part.id)
    ) {
      carried.splice(i, 1);
    }
  }
  const sourceIds = new Set(
    carried.flatMap(({ part }) => {
      const source = record(part);
      return source?.type === "source" && typeof source.id === "string"
        ? [source.id]
        : [];
    }),
  );
  const savedPending = carried.filter((entry) => {
    const part = record(entry.part);
    return part?.type === "tool-call" && part.result === undefined;
  });
  // Id-less cards share the empty id: one slot each, else all but the last stay running forever.
  let savedIdless = 0;
  for (const entry of savedPending) {
    const id = record(entry.part)?.backendToolCallId;
    if (typeof id === "string") {
      pending.set(id || `#idless:saved:${savedIdless++}`, entry);
    }
  }
  const replayFrom = savedPending.some(
    (entry) => typeof record(entry.part)?.backendToolCallId !== "string",
  )
    ? 0
    : snapshotSeq;
  const legacyPending = savedPending.filter(
    (entry) => typeof record(entry.part)?.backendToolCallId !== "string",
  );
  const claimLegacy = (entry: CarriedPart | undefined) => {
    const at = entry ? legacyPending.indexOf(entry) : -1;
    if (at !== -1) legacyPending.splice(at, 1);
    return entry;
  };
  // Seeded from saves: a reload between two completions of one card leaves it in no other lookup.
  const completed = new Map<string, CarriedPart>();
  /** Most recent finished card a provider gave no id, for a repeated id-less ending. */
  let lastIdless: CarriedPart | undefined;
  for (const entry of carried) {
    const part = record(entry.part);
    const id = part?.backendToolCallId;
    if (part?.type !== "tool-call" || part.result === undefined) continue;
    if (typeof part.toolApprovalId === "string" && part.toolApprovalId) {
      endedApprovalIds.add(part.toolApprovalId);
    }
    if (typeof id === "string" && id) completed.set(id, entry);
    else if (id === "") lastIdless = entry;
  }
  /** A card the previous frontend saved carries its backend id inside toolCallId and nowhere
   *  else, so a later completion has to recognise it the way the pending lookup already does. */
  const findCompletedLegacy = (backendId: string) => {
    if (!backendId) return undefined;
    for (let i = carried.length - 1; i >= 0; i--) {
      const part = record(carried[i].part);
      const id = part?.toolCallId;
      if (
        part?.type !== "tool-call" ||
        part.result === undefined ||
        part.backendToolCallId !== undefined ||
        typeof id !== "string"
      ) {
        continue;
      }
      if (id === backendId || id.startsWith(`${backendId}:`)) return carried[i];
    }
    return undefined;
  };
  const findSavedEntry = (backendId: string, approvalId: unknown) => {
    const matches = savedPending.filter((entry) => {
      const part = record(entry.part);
      const id = part?.toolCallId;
      if (!part || typeof id !== "string" || part.result !== undefined)
        return false;
      if (typeof approvalId === "string" && approvalId) {
        return (
          part.toolApprovalId === approvalId ||
          id === approvalId ||
          id.endsWith(`:${approvalId}`)
        );
      }
      return (
        Boolean(backendId) &&
        (id === backendId || id.startsWith(`${backendId}:`))
      );
    });
    return matches.length === 1 ? matches[0] : undefined;
  };
  const findApprovalEntry = (approval: ChatGenerationPendingApproval) => {
    const backendId = approval.cardCallId || approval.toolCallId;
    const matches = carried.filter((entry) => {
      const part = record(entry.part);
      if (part?.type !== "tool-call" || part.result !== undefined) return false;
      const id = typeof part.toolCallId === "string" ? part.toolCallId : "";
      return (
        part.toolApprovalId === approval.approvalId ||
        (Boolean(backendId) && id === backendId) ||
        (Boolean(backendId) && part.backendToolCallId === backendId) ||
        id === approval.approvalId ||
        id.endsWith(`:${approval.approvalId}`)
      );
    });
    return matches.length === 1 ? matches[0] : undefined;
  };
  /** Reconcile the cards against the run row's authoritative pending set. This may rebuild a
   *  missing display card, but deliberately has no event-cursor input or output: a snapshot
   *  watermark is not proof that any content event was applied. */
  const syncPendingApprovals = (
    approvals: readonly ChatGenerationPendingApproval[] | undefined,
    at: number,
  ):
    | { changed: boolean; confirmations: RecoveredToolConfirmation[] }
    | undefined => {
    if (approvals === undefined) return undefined;
    let changed = false;
    const confirmations: RecoveredToolConfirmation[] = [];
    for (const approval of approvals) {
      if (
        approval.status !== "pending" ||
        approval.runId !== runId ||
        !approval.approvalId ||
        endedApprovalIds.has(approval.approvalId)
      ) {
        continue;
      }
      let entry = findApprovalEntry(approval);
      if (!entry) {
        entry = {
          at,
          part: {
            type: "tool-call",
            toolCallId: `${approval.sessionId || "_default"}:${runId}:${approval.approvalId}`,
          },
        };
        carried.push(entry);
        changed = true;
      }
      const part = record(entry.part) ?? {};
      const backendId = approval.cardCallId || approval.toolCallId;
      const args = approval.arguments ?? record(part.args) ?? {};
      const argsText =
        typeof approval.argumentsText === "string"
          ? toolCallArgumentsText(approval.argumentsText, args)
          : approval.arguments !== undefined
            ? toolCallArgumentsText(undefined, args)
            : typeof part.argsText === "string"
              ? part.argsText
              : toolCallArgumentsText(undefined, args);
      const next = {
        ...part,
        type: "tool-call",
        toolCallId:
          typeof part.toolCallId === "string" && part.toolCallId
            ? part.toolCallId
            : `${approval.sessionId || "_default"}:${runId}:${approval.approvalId}`,
        backendToolCallId: backendId,
        toolApprovalId: approval.approvalId,
        toolName: approval.toolName,
        args,
        argsText,
      };
      if (
        part.backendToolCallId !== next.backendToolCallId ||
        part.toolApprovalId !== next.toolApprovalId ||
        part.toolName !== next.toolName ||
        part.argsText !== next.argsText ||
        JSON.stringify(part.args ?? {}) !== JSON.stringify(next.args)
      ) {
        entry.part = next;
        changed = true;
      }
      if (backendId) pending.set(backendId, entry);
      confirmations.push({
        toolCallId: String(next.toolCallId),
        approvalId: approval.approvalId,
        sessionId: approval.sessionId,
      });
    }
    return { changed, confirmations };
  };
  let appliedSeq = 0;
  const apply = (
    payload: unknown,
    at: number,
    seq: number,
    sessionId = "_default",
  ) => {
    const chunk = record(payload);
    const event = record(chunk?._toolEvent) ?? chunk;
    if (
      event?.type !== "tool_start" &&
      event?.type !== "tool_end" &&
      event?.type !== "document_citations"
    ) {
      return;
    }
    if (seq <= appliedSeq) return;
    appliedSeq = seq;
    const backendId =
      typeof event.tool_call_id === "string" ? event.tool_call_id : "";
    if (event.tool_name === "deep_research") {
      if (event.type === "tool_start")
        researchHandoff.hiddenCallIds.delete(backendId);
      if (readDeepResearchToolEvent(researchHandoff, event)) return;
    }
    // Older approval cards need their original start event to recover the backend id.
    if (seq <= snapshotSeq) {
      const entry =
        event.type === "tool_start"
          ? claimLegacy(
              findSavedEntry(backendId, event.approval_id) ??
                (backendId ? undefined : legacyPending[0]),
            )
          : undefined;
      if (entry) {
        entry.part = {
          ...record(entry.part),
          backendToolCallId: backendId,
          generationToolCallId: `${runId}:${seq}`,
        };
        pending.set(backendId || `#idless:legacy:${seq}`, entry);
      }
      return;
    }
    if (event.type === "document_citations") {
      if (Array.isArray(event.citations)) {
        event.citations.forEach((value, index) => {
          const citation = record(value);
          const part = citation
            ? documentCitationToSource(citation, index)
            : null;
          if (!part || sourceIds.has(part.id)) return;
          carried.push({ at, part });
          sourceIds.add(part.id);
        });
      }
      return;
    }
    const toolName = typeof event.tool_name === "string" ? event.tool_name : "";
    let entry =
      (backendId ? pending.get(backendId) : undefined) ??
      findSavedEntry(backendId, event.approval_id);
    // Id-less end closes the most recent start, as the adapter does; else two calls recover as one.
    if (
      event.type === "tool_end" &&
      !(entry || backendId) &&
      pending.size > 0
    ) {
      for (const active of pending.values()) entry = active;
    }
    // OpenAI Responses ends a web search twice (placeholder, then citations); a start clears this.
    if (!entry && event.type === "tool_end") {
      entry = backendId
        ? (completed.get(backendId) ?? findCompletedLegacy(backendId))
        : lastIdless;
    }
    // Gemini can emit a second completion carrying a generated image.
    if (
      !entry &&
      event.type === "tool_end" &&
      record(record(event.google)?.native_part)
    ) {
      for (let i = carried.length - 1; i >= 0; i--) {
        const candidate = record(carried[i].part);
        if (
          candidate?.type !== "tool-call" ||
          !record(record(record(candidate.args)?.google)?.native_part)
        )
          continue;
        const id = candidate.toolCallId;
        if (
          backendId &&
          (candidate.backendToolCallId === backendId ||
            (candidate.backendToolCallId === undefined &&
              typeof id === "string" &&
              (id === backendId || id.startsWith(`${backendId}:`))))
        ) {
          entry = carried[i];
          break;
        }
      }
    }
    if (event.type === "tool_start") {
      if (!toolName) {
        return;
      }
      if (!entry) {
        entry = {
          at,
          part: {
            type: "tool-call",
            toolCallId: `${backendId || "tool"}:${runId}:${seq}`,
          },
        };
        carried.push(entry);
      }
      const args = record(event.arguments) ?? {};
      entry.part = {
        ...record(entry.part),
        backendToolCallId: backendId,
        generationToolCallId: `${runId}:${seq}`,
        ...(typeof event.approval_id === "string" && event.approval_id
          ? { toolApprovalId: event.approval_id }
          : {}),
        toolName,
        args,
        argsText: toolCallArgumentsText(event.arguments_text, args),
        ...(record(event.provenance) ? { provenance: event.provenance } : {}),
      };
      pending.set(backendId || `#idless:${runId}:${seq}`, entry);
      if (backendId) completed.delete(backendId);
      else lastIdless = undefined;
      return;
    }
    if (!entry) {
      return;
    }
    const part = record(entry.part);
    if (!part) {
      return;
    }
    if (typeof part.toolApprovalId === "string" && part.toolApprovalId) {
      endedApprovalIds.add(part.toolApprovalId);
    }
    if (typeof part.toolCallId === "string" && part.toolCallId) {
      endedConfirmationCardIds.push(part.toolCallId);
    }
    const nextArgs = record(event.arguments);
    const args = mergeGoogleNativeParts(
      { ...record(part.args), ...nextArgs },
      event.google,
    );
    entry.part = {
      ...part,
      args,
      argsText: mergedToolCallArgumentsText(
        part.argsText,
        args,
        Object.keys(nextArgs ?? {}),
      ),
      result: recoveredToolResult(event, part.toolName, sessionId),
      ...(record(event.provenance)
        ? {
            provenance: {
              ...record(part.provenance),
              ...record(event.provenance),
            },
          }
        : {}),
    };
    for (const [id, active] of pending) {
      if (active === entry) {
        pending.delete(id);
      }
    }
    if (backendId) completed.set(backendId, entry);
    else lastIdless = entry;
  };
  // Recovery never reaches the live path's end-of-stream source yield, so rebuild those entries.
  // Per occurrence, not per url, because the live path flat-maps the cards: two rounds finding
  // the same page carry their own title and snippet, and the Sources panel lists both.
  const withSources = <TPart>(parts: TPart[]): TPart[] => {
    const out: TPart[] = [...parts];
    for (const { part } of carried) {
      const card = searchCard(part);
      if (!card) continue;
      for (const source of cardSources(card)) {
        // Copied rather than handed out from the cache, `metadata` included: before the
        // cache each rebuild yielded its own objects all the way down, and a caller that
        // edits a part it was given must not reach back into what the next rebuild yields.
        out.push({
          ...source,
          ...(source.metadata ? { metadata: { ...source.metadata } } : {}),
        } as TPart);
      }
    }
    return out;
  };
  const takeEndedConfirmationCardIds = (): string[] =>
    endedConfirmationCardIds.splice(0, endedConfirmationCardIds.length);
  return {
    replayFrom,
    apply,
    syncPendingApprovals,
    takeEndedConfirmationCardIds,
    withSources,
  };
}
