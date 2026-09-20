// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState, type ReactElement } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import {
  approveLearningProposal,
  createLearningProposal,
  getLearningState,
  rejectLearningProposal,
  setLearningConfig,
  setLearningEnabled,
  type LearningKind,
  type LearningState,
  type LearningTarget,
} from "../api/learning-api";
import {
  HERMES_LEARNING_CHANGED_EVENT,
  HERMES_LEARNING_OPEN_EVENT,
} from "../lib/hermes-learning";

const EMPTY_STATE: LearningState = {
  enabled: true,
  mem0Enabled: true,
  onTheFlySkills: true,
  memory: [],
  user: [],
  pending: [],
  limits: { memoryChars: 2_200, userChars: 1_375 },
  context: "",
};

export function ChatLearningDialog(): ReactElement {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<LearningState>(EMPTY_STATE);
  const [kind, setKind] = useState<LearningKind>("memory");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [reason, setReason] = useState("");
  const [name, setName] = useState("");
  const [target, setTarget] = useState<LearningTarget>("codex");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    void getLearningState()
      .then(setState)
      .catch((error) =>
        toast.error("Could not load Hermes learning", {
          description: error instanceof Error ? error.message : undefined,
        }),
      );
  }, []);

  useEffect(() => {
    const openManager = () => {
      setOpen(true);
      refresh();
    };
    window.addEventListener(HERMES_LEARNING_OPEN_EVENT, openManager);
    window.addEventListener(HERMES_LEARNING_CHANGED_EVENT, refresh);
    return () => {
      window.removeEventListener(HERMES_LEARNING_OPEN_EVENT, openManager);
      window.removeEventListener(HERMES_LEARNING_CHANGED_EVENT, refresh);
    };
  }, [refresh]);

  const apply = async (action: () => Promise<LearningState>, success: string) => {
    setBusy(true);
    try {
      setState(await action());
      window.dispatchEvent(new Event(HERMES_LEARNING_CHANGED_EVENT));
      toast.success(success);
    } catch (error) {
      toast.error("Hermes learning update failed", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  };

  const stage = () => {
    void apply(
      () =>
        createLearningProposal({
          kind,
          title: title.trim(),
          content: content.trim(),
          reason: reason.trim(),
          ...(kind === "skill" && name.trim() ? { name: name.trim() } : {}),
          target,
        }),
      "Learning proposal staged for review",
    ).then(() => {
      setTitle("");
      setContent("");
      setReason("");
      setName("");
    });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="corner-squircle dialog-soft-surface max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Hermes self-improvement</DialogTitle>
          <DialogDescription>
            The loop is experience → Mem0 recurrence → Hermes skill → deterministic proof of model deficiency → QLoRA. Nothing becomes active until its policy allows it; executable code and permissions are never self-edited.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-5">
          <section className="flex items-center justify-between gap-3 rounded-xl border border-border/70 bg-muted/20 p-3 text-sm">
            <span>
              <span className="block font-medium">Use approved learning in future prompts</span>
              <span className="text-xs text-muted-foreground">Memory budget: {state.limits.memoryChars} chars · user preferences: {state.limits.userChars} chars</span>
            </span>
            <div className="flex gap-2">
              <Button
                variant={state.enabled ? "default" : "outline"}
                disabled={busy}
                onClick={() => void apply(() => setLearningEnabled(!state.enabled), state.enabled ? "Hermes learning paused" : "Hermes learning enabled")}
              >
                {state.enabled ? "Enabled" : "Paused"}
              </Button>
              <Button
                variant={state.onTheFlySkills ? "default" : "outline"}
                disabled={busy || !state.enabled}
                onClick={() => void apply(() => setLearningConfig({ onTheFlySkills: !state.onTheFlySkills }), state.onTheFlySkills ? "On-the-fly skills paused" : "On-the-fly skills enabled")}
              >
                {state.onTheFlySkills ? "Draft skills" : "No skill drafts"}
              </Button>
              <Button
                variant={state.mem0Enabled !== false ? "default" : "outline"}
                disabled={busy || !state.enabled}
                onClick={() => void apply(() => setLearningConfig({ mem0Enabled: state.mem0Enabled === false }), state.mem0Enabled === false ? "Mem0 experience memory enabled" : "Mem0 experience memory paused")}
              >
                {state.mem0Enabled !== false ? "Mem0 on" : "Mem0 off"}
              </Button>
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Pending review</h3>
            <div className="max-h-56 overflow-y-auto rounded-xl border border-border/70 bg-muted/20 p-2 text-xs">
              {state.pending.length ? state.pending.map((proposal) => (
                <div key={proposal.id} className="border-b border-border/50 py-2 last:border-0">
                  <div className="flex items-start justify-between gap-2">
                    <span className="min-w-0"><span className="font-medium">{proposal.title}</span> · {proposal.kind}</span>
                    <span className="flex shrink-0 gap-1">
                      <Button size="sm" disabled={busy} onClick={() => void apply(() => approveLearningProposal(proposal.id), "Learning approved")}>Approve</Button>
                      <Button size="sm" variant="ghost" disabled={busy} onClick={() => void apply(() => rejectLearningProposal(proposal.id), "Learning rejected")}>Reject</Button>
                    </span>
                  </div>
                  <p className="mt-1 whitespace-pre-wrap text-muted-foreground">{proposal.content}</p>
                  {proposal.reason ? <p className="mt-1 text-muted-foreground">Why: {proposal.reason}</p> : null}
                </div>
              )) : <span className="text-muted-foreground">No pending proposals. Use /learn after a verified task to ask the agent for one.</span>}
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Stage a proposal manually</h3>
            <div className="flex gap-2">
              <select value={kind} onChange={(event) => setKind(event.target.value as LearningKind)} className="h-9 rounded-md border border-input bg-background px-2 text-sm">
                <option value="memory">Verified lesson</option>
                <option value="user">User preference</option>
                <option value="skill">Reusable skill</option>
              </select>
              {kind === "skill" ? <Input value={name} onChange={(event) => setName(event.target.value)} placeholder="skill-name" /> : null}
              {kind === "skill" ? (
                <select value={target} onChange={(event) => setTarget(event.target.value as LearningTarget)} className="h-9 rounded-md border border-input bg-background px-2 text-sm">
                  <option value="codex">Codex</option>
                  <option value="claude">Claude Code</option>
                  <option value="both">Codex + Claude</option>
                </select>
              ) : null}
            </div>
            <Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Short title" />
            <Textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder={kind === "skill" ? "Plain-text SKILL.md instructions" : "Compact, durable fact or preference"} className="min-h-24" />
            <Input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why this should survive future sessions (optional)" />
            <div className="flex justify-end">
              <Button disabled={busy || !state.enabled || !title.trim() || !content.trim() || (kind === "skill" && !name.trim())} onClick={stage}>Stage for approval</Button>
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="text-sm font-medium">Active learning</h3>
            <div className="rounded-xl border border-border/70 bg-muted/20 p-3 text-xs">
              {state.user.length || state.memory.length ? (
                <>{state.user.map((entry) => <p key={`user-${entry.title}`}><span className="font-medium">Preference · {entry.title}:</span> {entry.content}</p>)}{state.memory.map((entry) => <p key={`memory-${entry.title}`}><span className="font-medium">Lesson · {entry.title}:</span> {entry.content}</p>)}</>
              ) : <span className="text-muted-foreground">No approved learning yet.</span>}
            </div>
          </section>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
