// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CheckCircle2, ClipboardList, X } from "lucide-react";
import { useEffect, type ReactElement } from "react";

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import { useChatGoalStore } from "../stores/chat-goal-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

export function ChatPlanPanel({
  threadId,
  onClose,
}: {
  threadId?: string | null;
  onClose: () => void;
}): ReactElement {
  const goal = useChatGoalStore((state) =>
    threadId ? state.goalsByThreadId[threadId] : undefined,
  );

  useEffect(() => {
    if (!threadId) return;
    const load = () => {
      void useChatGoalStore.getState().load(threadId).catch(() => undefined);
    };
    load();
    const onHistoryUpdated = () => load();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    return () => window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
  }, [threadId]);

  const usePlanAsGoal = () => {
    if (!threadId || !goal?.plan) {
      toast.info("Finish a plan first", {
        description: "Ask the assistant to produce an ordered plan in the chat.",
      });
      return;
    }
    void useChatGoalStore
      .getState()
      .start(threadId, goal.objective)
      .then(() => {
        useChatRuntimeStore.getState().setChatMode("goal");
        toast.success("Goal started from plan");
      })
      .catch((error) => {
        toast.error("Could not start goal", {
          description: error instanceof Error ? error.message : undefined,
        });
      });
  };

  return (
    <aside className="flex h-full min-h-0 flex-col border-l border-border/70 bg-background/80">
      <div className="flex items-start gap-3 border-b border-border/70 px-4 py-4">
        <ClipboardList className="mt-0.5 size-5 shrink-0 text-primary" />
        <div className="min-w-0 flex-1">
          <h2 className="font-medium">Working plan</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Plan mode keeps the assistant in analysis-only mode until you choose to act.
          </p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close plan panel" title="Close plan panel">
          <X className="size-4" />
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {goal?.objective ? (
          <div className="mb-4 rounded-xl border border-primary/20 bg-primary/[0.045] p-3">
            <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-primary">Objective</div>
            <p className="mt-1 text-sm text-foreground">{goal.objective}</p>
          </div>
        ) : null}
        {goal?.plan ? (
          <div className="whitespace-pre-wrap text-sm leading-6 text-foreground">{goal.plan}</div>
        ) : (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>Your generated plan will appear here after the assistant finishes its plan-only response.</p>
            <ol className="list-inside list-decimal space-y-2">
              <li>Inspect the relevant context.</li>
              <li>Propose ordered, reversible steps.</li>
              <li>Call out risks and verification checks.</li>
            </ol>
          </div>
        )}
      </div>
      <div className="border-t border-border/70 p-3">
        <Button className="w-full" disabled={!goal?.plan} onClick={usePlanAsGoal}>
          <CheckCircle2 className="size-4" />
          Use plan as Goal
        </Button>
      </div>
    </aside>
  );
}
