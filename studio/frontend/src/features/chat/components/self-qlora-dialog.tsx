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
  getLearningState,
  setLearningConfig,
  type LearningState,
} from "../api/learning-api";
import {
  benchmarkSelfTraining,
  evaluateSelfTraining,
  getSelfTrainingState,
  hotSwapSelfTrainingAdapter,
  revertSelfTrainingAdapter,
  setSelfTrainingBaseline,
  setSelfTrainingConfig,
  startSelfTraining,
  type SelfTrainingState,
} from "../api/self-training-api";
import {
  HERMES_LEARNING_CHANGED_EVENT,
  SELF_QLORA_OPEN_EVENT,
} from "../lib/hermes-learning";

const EMPTY_STATE: SelfTrainingState = {
  enabled: true,
  autoTrain: false,
  minExamples: 8,
  maxSeqLength: 8_192,
  baseModelId: null,
  baseSnapshotPath: null,
  baseContextLength: null,
  activeAdapterPath: null,
  datasetPath: "",
  exampleCount: 0,
  status: "idle",
  lastJobId: null,
  lastEvaluation: null,
  lastError: null,
  acceptanceCriteria: [],
};

const EMPTY_LEARNING_STATE: LearningState = {
  enabled: true,
  onTheFlySkills: true,
  decisionMode: "ask",
  allowSkillCreation: false,
  allowQloraTraining: false,
  allowRuntimeFix: false,
  memory: [],
  user: [],
  pending: [],
  limits: { memoryChars: 2_200, userChars: 1_375 },
  context: "",
};

function numberValue(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function SelfQloraDialog(): ReactElement {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<SelfTrainingState>(EMPTY_STATE);
  const [learningState, setLearningState] = useState<LearningState>(EMPTY_LEARNING_STATE);
  const [modelId, setModelId] = useState("");
  const [snapshotPath, setSnapshotPath] = useState("");
  const [contextLength, setContextLength] = useState("65536");
  const [candidatePath, setCandidatePath] = useState("");
  const [baseScore, setBaseScore] = useState("0");
  const [candidateScore, setCandidateScore] = useState("0");
  const [baseTokPerSec, setBaseTokPerSec] = useState("0");
  const [candidateTokPerSec, setCandidateTokPerSec] = useState("0");
  const [rubric, setRubric] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => {
    void Promise.all([getSelfTrainingState(), getLearningState()])
      .then(([next, nextLearning]) => {
        setState(next);
        setLearningState(nextLearning);
        if (!modelId && next.baseModelId) setModelId(next.baseModelId);
        if (!candidatePath && next.lastCandidateAdapterPath) {
          setCandidatePath(next.lastCandidateAdapterPath);
        }
      })
      .catch((error) =>
        toast.error("Could not load self-QLoRA", {
          description: error instanceof Error ? error.message : undefined,
        }),
      );
  }, [candidatePath, modelId]);

  useEffect(() => {
    const openManager = () => {
      setOpen(true);
      refresh();
    };
    window.addEventListener(SELF_QLORA_OPEN_EVENT, openManager);
    window.addEventListener(HERMES_LEARNING_CHANGED_EVENT, refresh);
    return () => {
      window.removeEventListener(SELF_QLORA_OPEN_EVENT, openManager);
      window.removeEventListener(HERMES_LEARNING_CHANGED_EVENT, refresh);
    };
  }, [refresh]);

  const apply = async (
    action: () => Promise<SelfTrainingState>,
    success: string,
  ) => {
    setBusy(true);
    try {
      setState(await action());
      window.dispatchEvent(new Event(HERMES_LEARNING_CHANGED_EVENT));
      toast.success(success);
    } catch (error) {
      toast.error("Self-QLoRA update failed", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  };

  const saveBaseline = () => {
    const trimmed = modelId.trim();
    if (!trimmed) return;
    void apply(
      () =>
        setSelfTrainingBaseline({
          modelId: trimmed,
          ...(snapshotPath.trim() ? { snapshotPath: snapshotPath.trim() } : {}),
          contextLength: numberValue(contextLength, 65_536),
        }),
      "Original model baseline saved",
    );
  };

  const updateConfig = (input: Parameters<typeof setSelfTrainingConfig>[0]) => {
    void apply(() => setSelfTrainingConfig(input), "Self-QLoRA settings saved");
  };

  const updateDecision = (input: Parameters<typeof setLearningConfig>[0]) => {
    setBusy(true);
    void setLearningConfig(input)
      .then((next) => {
        setLearningState(next);
        window.dispatchEvent(new Event(HERMES_LEARNING_CHANGED_EVENT));
      })
      .catch((error) => toast.error("Decision policy update failed", { description: error instanceof Error ? error.message : undefined }))
      .finally(() => setBusy(false));
  };

  const evaluate = () => {
    if (!candidatePath.trim()) return;
    void apply(
      () =>
        evaluateSelfTraining({
          candidateAdapterPath: candidatePath.trim(),
          baseScore: numberValue(baseScore, 0),
          candidateScore: numberValue(candidateScore, 0),
          baseTokPerSec: numberValue(baseTokPerSec, 0),
          candidateTokPerSec: numberValue(candidateTokPerSec, 0),
          rubric: rubric.trim(),
        }),
      "Candidate evaluated",
    );
  };

  const benchmark = () => {
    if (!candidatePath.trim()) return;
    void apply(
      () => benchmarkSelfTraining({ candidateAdapterPath: candidatePath.trim() }),
      "Independent holdout benchmark completed",
    );
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="corner-squircle dialog-soft-surface max-h-[88vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Self-QLoRA</DialogTitle>
          <DialogDescription>
            Completed local tasks become bounded ChatML examples. The original
            model stays the baseline; adapters are promoted only after measured
            intelligence improves and throughput does not regress.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-5 text-sm">
          <section className="grid gap-3 rounded-xl border border-border/70 bg-muted/20 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="font-medium">Learning permissions</p>
                <p className="text-xs text-muted-foreground">
                  Collection is local and bounded. Automatic training is a separate permission.
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={state.enabled ? "default" : "outline"}
                  disabled={busy}
                  onClick={() => updateConfig({ enabled: !state.enabled })}
                >
                  {state.enabled ? "Collecting" : "Paused"}
                </Button>
                <Button
                  size="sm"
                  variant={state.autoTrain ? "default" : "outline"}
                  disabled={busy || !state.enabled}
                  onClick={() => updateConfig({ autoTrain: !state.autoTrain })}
                >
                  {state.autoTrain ? "Auto-train on" : "Auto-train off"}
                </Button>
              </div>
            </div>
            <div className="grid gap-2 border-t border-border/60 pt-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <p className="font-medium">Agent decision policy</p>
                  <p className="text-xs text-muted-foreground">Ask me stages a proposal. Autonomous may choose only enabled actions.</p>
                </div>
                <select
                  value={learningState.decisionMode ?? "ask"}
                  onChange={(event) => updateDecision({ decisionMode: event.target.value as "ask" | "autonomous" })}
                  disabled={busy || !learningState.enabled}
                  className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                  aria-label="Agent decision policy"
                >
                  <option value="ask">Ask me</option>
                  <option value="autonomous">Autonomous</option>
                </select>
              </div>
              <div className="flex flex-wrap gap-2 text-xs">
                {([
                  ["allowSkillCreation", "Skill creation"],
                  ["allowQloraTraining", "QLoRA training"],
                  ["allowRuntimeFix", "Runtime fix"],
                ] as const).map(([key, label]) => (
                  <Button
                    key={key}
                    size="sm"
                    variant={learningState[key] ? "default" : "outline"}
                    disabled={busy || !learningState.enabled}
                    onClick={() => updateDecision({ [key]: !learningState[key] })}
                  >
                    {learningState[key] ? "✓ " : ""}{label}
                  </Button>
                ))}
                <span className="inline-flex items-center rounded-md border border-border/60 px-2 py-1.5 text-muted-foreground">Nothing · always allowed</span>
              </div>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <Input
                value={String(state.minExamples)}
                onChange={(event) =>
                  setState((current) => ({
                    ...current,
                    minExamples: numberValue(event.target.value, current.minExamples),
                  }))
                }
                onBlur={() => updateConfig({ minExamples: state.minExamples })}
                type="number"
                min={4}
                max={64}
                aria-label="Minimum examples"
                placeholder="Minimum examples"
              />
              <Input
                value={String(state.maxSeqLength)}
                onChange={(event) =>
                  setState((current) => ({
                    ...current,
                    maxSeqLength: numberValue(event.target.value, current.maxSeqLength),
                  }))
                }
                onBlur={() => updateConfig({ maxSeqLength: state.maxSeqLength })}
                type="number"
                min={512}
                max={65536}
                aria-label="Training sequence length"
                placeholder="Training sequence length"
              />
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="font-medium">Immutable original baseline</h3>
            <p className="text-xs text-muted-foreground">
              The trainable source and the serving checkpoint are tracked separately. A GGUF serving model is never passed to the PEFT/MLX trainer.
            </p>
            {state.baseServingModelId && state.baseServingModelId !== state.baseModelId ? (
              <p className="text-xs text-muted-foreground">
                Serving: <span className="font-mono">{state.baseServingModelId}</span> · trainable source: <span className="font-mono">{state.baseModelId}</span>
              </p>
            ) : null}
            <Input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="Model ID (for example Qwen3.8-27B)" />
            <Input value={snapshotPath} onChange={(event) => setSnapshotPath(event.target.value)} placeholder="Optional local base snapshot path" />
            <div className="flex gap-2">
              <Input value={contextLength} onChange={(event) => setContextLength(event.target.value)} type="number" placeholder="Base context length" />
              <Button disabled={busy || !modelId.trim()} onClick={saveBaseline}>Save baseline</Button>
            </div>
          </section>

          <section className="grid gap-2 rounded-xl border border-border/70 bg-muted/20 p-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-medium">Task dataset</h3>
              <span className="text-xs text-muted-foreground">
                {state.exampleCount} / 256 examples · {state.eligibleExampleCount ?? 0} verified · {state.status}
              </span>
            </div>
            <p className="break-all text-xs text-muted-foreground">{state.datasetPath || "Dataset path will appear after the backend is ready."}</p>
            {state.lastRecommendation ? (
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-2 text-xs">
                <p className="font-medium">Model recommendation: {state.lastRecommendation.action}</p>
                <p className="text-muted-foreground">{state.lastRecommendation.reason || "No reason recorded."} · advisory only; the holdout bench decides promotion.</p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">After a task, use /learn to decide whether a reusable skill is enough. QLoRA is reserved for repeated, measurable gaps.</p>
            )}
            {state.lastCandidateAdapterPath ? (
              <p className="break-all text-xs text-muted-foreground">Latest candidate: {state.lastCandidateAdapterPath}</p>
            ) : null}
            {state.activeServingModelId ? (
              <p className="text-xs text-muted-foreground">Active adapted serving model: {state.activeServingModelId}</p>
            ) : null}
            {state.lastError ? <p className="text-xs text-destructive">{state.lastError}</p> : null}
            <div className="flex justify-end">
              <Button disabled={busy || !state.enabled} onClick={() => void apply(startSelfTraining, "Self-QLoRA queued")}>Train now</Button>
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="font-medium">Acceptance gate</h3>
            <p className="text-xs text-muted-foreground">The independent holdout bench is the trusted path. Both conditions are required: same speed passes; a speed regression never passes. Manual scores are for human review only.</p>
            <div className="grid gap-2 sm:grid-cols-2">
              <Input value={baseScore} onChange={(event) => setBaseScore(event.target.value)} type="number" min={0} max={1} step={0.01} placeholder="Base intelligence score" />
              <Input value={candidateScore} onChange={(event) => setCandidateScore(event.target.value)} type="number" min={0} max={1} step={0.01} placeholder="Candidate intelligence score" />
              <Input value={baseTokPerSec} onChange={(event) => setBaseTokPerSec(event.target.value)} type="number" min={0} step={0.1} placeholder="Base tok/s" />
              <Input value={candidateTokPerSec} onChange={(event) => setCandidateTokPerSec(event.target.value)} type="number" min={0} step={0.1} placeholder="Candidate tok/s" />
            </div>
            <Input value={candidatePath} onChange={(event) => setCandidatePath(event.target.value)} placeholder="Candidate adapter directory" />
            <Textarea value={rubric} onChange={(event) => setRubric(event.target.value)} placeholder="Evaluation rubric or notes (optional)" className="min-h-16" />
            <div className="flex flex-wrap justify-end gap-2">
              <Button variant="secondary" disabled={busy || !candidatePath.trim() || !state.baseModelId} onClick={benchmark}>Run independent holdout bench</Button>
              <Button disabled={busy || !candidatePath.trim()} onClick={evaluate}>Record human evaluation</Button>
            </div>
            {state.lastEvaluation ? (
              <div className="rounded-lg border border-border/70 bg-muted/20 p-3 text-xs">
                <p className="font-medium">{state.lastEvaluation.promoted ? "Promoted" : "Rejected"}</p>
                <p>Intelligence: {state.lastEvaluation.intelligenceImproved ? "pass" : "fail"} · Throughput: {state.lastEvaluation.speedPreserved ? "pass" : "fail"} · Adapter: {state.lastEvaluation.adapterExists ? "found" : "missing"}</p>
              </div>
            ) : null}
          </section>

          <section className="flex flex-wrap justify-end gap-2 border-t border-border/60 pt-3">
            <Button
              variant="outline"
              disabled={busy || !state.activeAdapterPath}
              onClick={() => void apply(() => hotSwapSelfTrainingAdapter({ adapterPath: state.activeAdapterPath ?? "" }), "Adapter hotswapped")}
            >
              Hotswap active adapter
            </Button>
            <Button
              variant="outline"
              disabled={busy || !state.baseModelId}
              onClick={() => void apply(revertSelfTrainingAdapter, "Reverted to original base")}
            >
              Revert to original
            </Button>
          </section>
          <ul className="grid gap-1 text-xs text-muted-foreground">
            {state.acceptanceCriteria.map((criterion) => <li key={criterion}>• {criterion}</li>)}
          </ul>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
