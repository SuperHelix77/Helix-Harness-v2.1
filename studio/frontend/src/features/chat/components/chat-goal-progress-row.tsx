// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CheckCircle2, Pause, Pencil, Play, Target, X } from "lucide-react";
import { type ReactElement, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import {
  CHAT_GOAL_ITEMS_MAX,
  CHAT_GOAL_ITEM_MAX_CHARS,
  CHAT_GOAL_MILESTONES_MAX,
  CHAT_GOAL_MILESTONE_ID_MAX_CHARS,
  CHAT_GOAL_MILESTONE_TITLE_MAX_CHARS,
} from "../lib/chat-goal";
import type {
  ChatGoalMilestone,
  ChatGoalMilestoneStatus,
  ChatGoalState,
} from "../lib/chat-goal";
import { useChatGoalStore } from "../stores/chat-goal-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

function linesToItems(value: string): string[] {
  return value
    .split("\n")
    .map((line) => line.trim().slice(0, CHAT_GOAL_ITEM_MAX_CHARS))
    .filter(Boolean)
    .slice(0, CHAT_GOAL_ITEMS_MAX);
}

function itemsToLines(items: string[]): string {
  return items.join("\n");
}

function milestonesToLines(milestones: ChatGoalMilestone[]): string {
  return milestones
    .map(
      (milestone) =>
        `${milestone.id} | ${milestone.title} | ${milestone.status}`,
    )
    .join("\n");
}

function elapsedAt(goal: ChatGoalState, now: number): number {
  const accumulated = goal.elapsedMs ?? 0;
  return Math.max(
    0,
    accumulated +
      (goal.status === "active" && goal.startedAt
        ? Math.max(0, now - goal.startedAt)
        : 0),
  );
}

function formatElapsed(milliseconds: number): string {
  const totalSeconds = Math.floor(milliseconds / 1_000);
  const hours = Math.floor(totalSeconds / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function linesToMilestones(value: string): ChatGoalMilestone[] {
  const statuses: readonly ChatGoalMilestoneStatus[] = [
    "pending",
    "active",
    "completed",
    "blocked",
  ];
  return value
    .split("\n")
    .map((line) => {
      const [rawId, rawTitle, rawStatus] = line.split("|");
      const id = rawId?.trim().slice(0, CHAT_GOAL_MILESTONE_ID_MAX_CHARS);
      const title = rawTitle
        ?.trim()
        .slice(0, CHAT_GOAL_MILESTONE_TITLE_MAX_CHARS);
      const status = rawStatus?.trim() as ChatGoalMilestoneStatus;
      return id && title
        ? {
            id,
            title,
            status: statuses.includes(status) ? status : "pending",
          }
        : undefined;
    })
    .filter(
      (milestone): milestone is ChatGoalMilestone => milestone !== undefined,
    )
    .slice(0, CHAT_GOAL_MILESTONES_MAX);
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: the row intentionally combines the finite goal lifecycle actions with its compact editor.
export function ChatGoalProgressRow({
  threadId,
}: {
  threadId?: string | null;
}): ReactElement | null {
  const goal = useChatGoalStore((state) =>
    threadId ? state.goalsByThreadId[threadId] : undefined,
  );
  const loading = useChatGoalStore((state) =>
    threadId ? state.loadingByThreadId[threadId] === true : false,
  );
  const [editing, setEditing] = useState(false);
  const [objective, setObjective] = useState("");
  const [constraints, setConstraints] = useState("");
  const [verification, setVerification] = useState("");
  const [milestones, setMilestones] = useState("");
  const [evidenceRefs, setEvidenceRefs] = useState("");
  const [blockers, setBlockers] = useState("");
  const [claimCeiling, setClaimCeiling] = useState("");
  const [nextAction, setNextAction] = useState("");
  const [clock, setClock] = useState(() => Date.now());

  useEffect(() => {
    if (!threadId) {
      return;
    }
    const load = () => {
      useChatGoalStore
        .getState()
        .load(threadId)
        .catch(() => undefined);
    };
    load();
    const onHistoryUpdated = (event: Event) => {
      const detail = (event as CustomEvent<{ thread?: { id?: string } }>)
        .detail;
      if (!detail?.thread?.id || detail.thread.id === threadId) {
        load();
      }
    };
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    return () =>
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
  }, [threadId]);

  useEffect(() => {
    if (!goal || goal.status !== "active") return;
    const timer = window.setInterval(() => setClock(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [goal]);

  if (!threadId || loading || !goal) {
    return null;
  }

  const run = async (
    action: () => Promise<void>,
    success?: string,
  ): Promise<boolean> => {
    try {
      await action();
      if (success) {
        toast.success(success);
      }
      return true;
    } catch (error) {
      toast.error("Could not update goal", {
        description: error instanceof Error ? error.message : undefined,
      });
      return false;
    }
  };

  const openEditor = () => {
    setObjective(goal.objective);
    setConstraints(itemsToLines(goal.constraints));
    setVerification(itemsToLines(goal.verification));
    setMilestones(milestonesToLines(goal.milestones));
    setEvidenceRefs(itemsToLines(goal.evidenceRefs));
    setBlockers(itemsToLines(goal.blockers));
    setClaimCeiling(goal.claimCeiling ?? "");
    setNextAction(goal.nextAction ?? "");
    setEditing(true);
  };

  const saveEditor = () => {
    const nextObjective = objective.trim();
    if (!nextObjective) {
      toast.error("A goal needs an objective.");
      return;
    }
    run(
      () =>
        useChatGoalStore.getState().update(threadId, (current) => ({
          ...current,
          objective: nextObjective,
          constraints: linesToItems(constraints),
          verification: linesToItems(verification),
          milestones: linesToMilestones(milestones),
          evidenceRefs: linesToItems(evidenceRefs),
          blockers: linesToItems(blockers),
          claimCeiling: claimCeiling.trim() || undefined,
          nextAction: nextAction.trim() || undefined,
          updatedAt: Date.now(),
        })),
      "Goal updated",
    ).then((saved) => {
      if (saved) {
        setEditing(false);
      }
    });
  };

  const setStatus = (status: ChatGoalState["status"]) => {
    run(
      () => {
        const now = Date.now();
        return useChatGoalStore.getState().update(threadId, (current) => ({
          ...current,
          status,
          updatedAt: now,
          elapsedMs: elapsedAt(current, now),
          ...(status === "active"
            ? { startedAt: now, completedAt: undefined }
            : {
                startedAt: undefined,
                ...(status === "completed" ? { completedAt: now } : { completedAt: undefined }),
              }),
        }));
      },
      status === "completed"
        ? "Goal marked complete"
        : status === "active"
          ? "Goal resumed"
          : "Goal paused",
    );
  };

  return (
    <>
      <section
        aria-label="Goal progress"
        className="mx-auto mb-2 flex w-[calc(100%-2.5rem)] max-w-[44rem] shrink-0 flex-col gap-2 rounded-2xl border border-primary/20 bg-primary/[0.045] px-3.5 py-2.5 text-sm"
      >
        <div className="flex items-start gap-2">
          <Target
            className="mt-0.5 size-4 shrink-0 text-primary"
            aria-hidden="true"
          />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-foreground">Goal</span>
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] capitalize text-primary">
                {goal.status}
              </span>
              <span className="text-[11px] text-muted-foreground" data-goal-elapsed>
                Elapsed {formatElapsed(elapsedAt(goal, clock))}
              </span>
            </div>
            <p className="mt-0.5 line-clamp-2 text-muted-foreground">
              {goal.objective}
            </p>
            {goal.milestones.length > 0 ? (
              <ul className="mt-1 space-y-0.5 text-xs text-muted-foreground">
                {goal.milestones.map((milestone) => (
                  <li key={milestone.id}>
                    <span className="font-medium text-foreground/80">
                      {milestone.id}
                    </span>{" "}
                    <span className="capitalize">({milestone.status})</span>{" "}
                    {milestone.title}
                  </li>
                ))}
              </ul>
            ) : null}
            {goal.blockers.length > 0 ? (
              <p className="mt-1 text-xs text-destructive">
                Blocked: {goal.blockers.join("; ")}
              </p>
            ) : null}
            {goal.evidenceRefs.length > 0 || goal.nextAction ? (
              <p className="mt-1 text-xs text-muted-foreground">
                {goal.evidenceRefs.length > 0
                  ? `${goal.evidenceRefs.length} evidence reference${goal.evidenceRefs.length === 1 ? "" : "s"}`
                  : null}
                {goal.evidenceRefs.length > 0 && goal.nextAction ? " · " : null}
                {goal.nextAction ? `Next: ${goal.nextAction}` : null}
              </p>
            ) : null}
            {goal.claimCeiling ? (
              <p className="mt-1 text-xs text-muted-foreground">
                Claim ceiling: {goal.claimCeiling}
              </p>
            ) : null}
            {goal.evidenceRefs.length > 0 ? (
              <details className="mt-1 text-xs text-muted-foreground">
                <summary className="cursor-pointer select-none text-primary/80">
                  View evidence references
                </summary>
                <ul className="mt-1 list-inside list-disc space-y-0.5">
                  {goal.evidenceRefs.map((reference) => (
                    <li key={reference}>{reference}</li>
                  ))}
                </ul>
              </details>
            ) : null}
            {goal.plan ? (
              <details className="mt-1 text-xs text-muted-foreground">
                <summary className="cursor-pointer select-none text-primary/80">
                  {goal.status === "draft"
                    ? "View proposed plan"
                    : "View approved plan"}
                </summary>
                <p className="mt-1 whitespace-pre-wrap">{goal.plan}</p>
              </details>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-1">
            {goal.status === "draft" ? (
              <Button
                size="sm"
                onClick={() => {
                  run(async () => {
                    await useChatGoalStore
                      .getState()
                      .start(threadId, goal.objective);
                    useChatRuntimeStore.getState().setChatMode("goal");
                  }, "Goal started");
                }}
              >
                <Play className="size-3.5" />
                Start
              </Button>
            ) : goal.status === "active" ? (
              <Button
                variant="ghost"
                size="icon-sm"
                title="Pause goal"
                aria-label="Pause goal"
                onClick={() => setStatus("paused")}
              >
                <Pause className="size-4" />
              </Button>
            ) : goal.status === "paused" ? (
              <Button
                variant="ghost"
                size="icon-sm"
                title="Resume goal"
                aria-label="Resume goal"
                onClick={() => setStatus("active")}
              >
                <Play className="size-4" />
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="icon-sm"
                title="Restart goal"
                aria-label="Restart goal"
                onClick={() => setStatus("active")}
              >
                <Play className="size-4" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon-sm"
              title="Edit goal"
              aria-label="Edit goal"
              onClick={openEditor}
            >
              <Pencil className="size-4" />
            </Button>
            {goal.status === "active" || goal.status === "paused" ? (
              <Button
                variant="ghost"
                size="icon-sm"
                title="Mark goal complete"
                aria-label="Mark goal complete"
                onClick={() => setStatus("completed")}
              >
                <CheckCircle2 className="size-4" />
              </Button>
            ) : null}
            <Button
              variant="ghost"
              size="icon-sm"
              title="Clear goal"
              aria-label="Clear goal"
              onClick={() => {
                run(
                  () => useChatGoalStore.getState().clear(threadId),
                  "Goal cleared",
                );
              }}
            >
              <X className="size-4" />
            </Button>
          </div>
        </div>
      </section>

      <Dialog open={editing} onOpenChange={setEditing}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Edit goal</DialogTitle>
            <DialogDescription>
              Keep the objective precise. Put one constraint or verification
              requirement per line.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-objective">Objective</label>
              <Textarea
                id="chat-goal-objective"
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-constraints">Constraints</label>
              <Textarea
                id="chat-goal-constraints"
                value={constraints}
                onChange={(event) => setConstraints(event.target.value)}
                placeholder="Preserve existing functionality"
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-verification">Verification</label>
              <Textarea
                id="chat-goal-verification"
                value={verification}
                onChange={(event) => setVerification(event.target.value)}
                placeholder="Run the relevant tests"
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-milestones">Milestones</label>
              <Textarea
                id="chat-goal-milestones"
                value={milestones}
                onChange={(event) => setMilestones(event.target.value)}
                placeholder="M1 | Implement the change | active"
              />
              <span className="text-xs font-normal text-muted-foreground">
                One per line: id | title | pending, active, completed, or
                blocked
              </span>
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-evidence">Evidence references</label>
              <Textarea
                id="chat-goal-evidence"
                value={evidenceRefs}
                onChange={(event) => setEvidenceRefs(event.target.value)}
                placeholder="test output, commit, benchmark, or source URL"
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-blockers">Blockers</label>
              <Textarea
                id="chat-goal-blockers"
                value={blockers}
                onChange={(event) => setBlockers(event.target.value)}
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-claim-ceiling">Claim ceiling</label>
              <Textarea
                id="chat-goal-claim-ceiling"
                value={claimCeiling}
                onChange={(event) => setClaimCeiling(event.target.value)}
                placeholder="What this goal may claim before further evidence"
              />
            </div>
            <div className="grid gap-1.5 text-sm font-medium">
              <label htmlFor="chat-goal-next-action">Next action</label>
              <Textarea
                id="chat-goal-next-action"
                value={nextAction}
                onChange={(event) => setNextAction(event.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(false)}>
              Cancel
            </Button>
            <Button onClick={saveEditor}>Save goal</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
