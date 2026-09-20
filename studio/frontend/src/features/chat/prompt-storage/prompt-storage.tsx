// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Lightweight entry point for Saved Prompts and chat export.
 *
 * `prompt-storage-dialog.tsx` intentionally owns the heavy implementation: dialog
 * UI, markdown preview, RAG/project-source helpers, conversation serializers and
 * fine-tune export. Importing it from the Chat shell pulled all of that into first
 * paint even when Saved Prompts was never opened. Keep this file dependency-light
 * and load the implementation only when an action actually needs it.
 */

import { lazy, Suspense, useEffect, useState, type ReactElement } from "react";

type Implementation = typeof import("./prompt-storage-dialog");

let implementationPromise: Promise<Implementation> | null = null;

function implementation(): Promise<Implementation> {
  implementationPromise ??= import("./prompt-storage-dialog");
  return implementationPromise;
}

const LazyPromptStorageDialog = lazy(async () => {
  const module = await implementation();
  return { default: module.PromptStorageDialog };
});

export type PromptStorageDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUse: (text: string) => void;
  onRunList?: (items: string[]) => void;
};

/** Mount lazily on first open, then keep the dialog mounted so unsaved drafts
 * retain the same lifetime they had before code splitting. */
export function PromptStorageDialog(
  props: PromptStorageDialogProps,
): ReactElement | null {
  const [wasOpened, setWasOpened] = useState(props.open);
  useEffect(() => {
    if (props.open) setWasOpened(true);
  }, [props.open]);
  if (!wasOpened && !props.open) return null;
  return (
    <Suspense fallback={null}>
      <LazyPromptStorageDialog {...props} />
    </Suspense>
  );
}

export type ConvExportFormat =
  | "jsonl-raw"
  | "jsonl-messages"
  | "csv"
  | "sharegpt";

const EXPORT_FORMAT_LABELS: Record<ConvExportFormat, string> = {
  "jsonl-raw": "Training JSONL",
  "jsonl-messages": "Message JSONL",
  csv: "CSV",
  sharegpt: "ShareGPT JSONL",
};

export const EXPORT_FORMATS_LIST = (
  Object.keys(EXPORT_FORMAT_LABELS) as ConvExportFormat[]
).map((fmt) => ({ fmt, label: EXPORT_FORMAT_LABELS[fmt] }));

// Message JSONL preserves branch siblings and is intentionally per-chat rather
// than a merged multi-conversation body; mirror the implementation's predicate
// without importing it.
export const COMBINED_EXPORT_FORMATS_LIST = EXPORT_FORMATS_LIST.filter(
  ({ fmt }) => fmt !== "jsonl-messages",
);

export type FineTuneFormat = "openai" | "sharegpt" | "alpaca";

export type FineTuneExportResult = {
  lines: string[];
  conversations: number;
  skipped: number;
};

export async function exportConversationShareGPT(threadId: string): Promise<void> {
  return (await implementation()).exportConversationShareGPT(threadId);
}

export async function exportConversationRawJsonl(threadId: string): Promise<void> {
  return (await implementation()).exportConversationRawJsonl(threadId);
}

export async function exportConversationMessagesJsonl(threadId: string): Promise<void> {
  return (await implementation()).exportConversationMessagesJsonl(threadId);
}

export async function exportConversationCsv(threadId: string): Promise<void> {
  return (await implementation()).exportConversationCsv(threadId);
}

export async function exportConversationMarkdown(threadId: string): Promise<void> {
  return (await implementation()).exportConversationMarkdown(threadId);
}

export async function saveChatItemAsProjectSource(
  item: { id: string; title: string; type: string },
  projectId: string,
): Promise<void> {
  return (await implementation()).saveChatItemAsProjectSource(item, projectId);
}

export async function buildChatItemMarkdown(item: {
  id: string;
  title: string;
  type: string;
}): Promise<string> {
  return (await implementation()).buildChatItemMarkdown(item);
}

export async function exportBulkConversationsMerged(
  threadIds: string[],
  format: ConvExportFormat,
  basename: string,
): Promise<void> {
  return (await implementation()).exportBulkConversationsMerged(
    threadIds,
    format,
    basename,
  );
}

export async function exportBulkConversationsSeparate(
  threadIds: string[],
  format: ConvExportFormat,
  basename: string,
): Promise<void> {
  return (await implementation()).exportBulkConversationsSeparate(
    threadIds,
    format,
    basename,
  );
}

export async function bulkExportConversationsByScope(
  scope: "recents" | "all",
  format: ConvExportFormat,
  merged: boolean,
): Promise<void> {
  return (await implementation()).bulkExportConversationsByScope(
    scope,
    format,
    merged,
  );
}

export async function exportProjectConversations(
  threadIds: string[],
  format: ConvExportFormat,
  projectName: string,
): Promise<void> {
  return (await implementation()).exportProjectConversations(
    threadIds,
    format,
    projectName,
  );
}

export async function buildFineTuneJsonl(
  format: FineTuneFormat = "openai",
): Promise<FineTuneExportResult> {
  return (await implementation()).buildFineTuneJsonl(format);
}

export async function exportFineTuneJsonl(
  format: FineTuneFormat = "openai",
): Promise<number> {
  return (await implementation()).exportFineTuneJsonl(format);
}
