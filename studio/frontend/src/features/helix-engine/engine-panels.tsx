// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft } from "lucide-react";

import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { getStoredChatThreadReadResult, useChatRuntimeStore } from "@/features/chat";

import {
  engineProjectIdForThread,
  engineSessionIdFor,
  fetchHelixJson,
  helixErrorMessage,
  loadHelixSnapshot,
  type FeedEvent,
  type HelixProvenance,
  type SessionStep,
  type ToolControlEvent,
} from "./engine-data";
import {
  type ExecutionNode,
  type HelixAnalyzeResult,
  normalizeExecutionGraph,
} from "./execution-graph";

type SessionBinding = {
  threadId: string;
  sessionId: string;
  scope: string;
  warning: string | null;
};

type PollState = {
  key: string;
  events: FeedEvent[];
  steps: SessionStep[];
  controlEvents: ToolControlEvent[];
  provenance: HelixProvenance | null;
  provenanceError: string | null;
  status: "connected" | "degraded";
  error: string | null;
  lastUpdatedAt: number | null;
};

type AnalysisState = {
  key: string;
  value: HelixAnalyzeResult;
};

const EMPTY_FEED_EVENTS: FeedEvent[] = [];
const EMPTY_SESSION_STEPS: SessionStep[] = [];
const EMPTY_CONTROL_EVENTS: ToolControlEvent[] = [];

function provenanceLabel(node: ExecutionNode): string {
  if (node.provenance === "observed") return "Observed fact";
  if (node.provenance === "model-audit") return "Model / self-audit";
  if (node.provenance === "policy") return "Policy decision";
  return "Missing evidence";
}

function authorityLabel(node: ExecutionNode): string {
  if (node.id.startsWith("analysis:")) return "Manual Analyze · advisory";
  if (node.provenance === "observed") return "Backend-observed receipt";
  if (node.provenance === "model-audit") return "Model-authored self-assessment";
  if (node.provenance === "policy") return "Backend policy/adjudication receipt";
  return "No authoritative receipt";
}

function ExecutionLane({
  title,
  nodes,
  selectedId,
  onSelect,
}: {
  title: string;
  nodes: ExecutionNode[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="rounded-lg border border-border/60 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </h3>
        <span className="text-[11px] text-muted-foreground">{nodes.length}</span>
      </div>
      {nodes.length ? (
        <ol className="space-y-2">
          {nodes.map((node, index) => (
            <li key={node.id} className="relative pl-4">
              {index > 0 ? (
                <span
                  aria-hidden
                  className="absolute -top-2 bottom-[calc(50%+0.35rem)] left-[3px] w-px bg-border"
                />
              ) : null}
              {index < nodes.length - 1 ? (
                <span
                  aria-hidden
                  className="absolute top-[calc(50%+0.35rem)] -bottom-2 left-[3px] w-px bg-border"
                />
              ) : null}
              <span
                aria-hidden
                className="absolute left-0 top-1/2 size-[7px] -translate-y-1/2 rounded-full border border-foreground/50 bg-background"
              />
              <button
                type="button"
                onClick={() => onSelect(node.id)}
                aria-pressed={selectedId === node.id}
                className="w-full rounded-md border border-border/60 px-2.5 py-2 text-left transition-colors hover:bg-muted/40 aria-pressed:border-foreground/35 aria-pressed:bg-muted/50"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs font-medium">{node.title}</span>
                  <span className="shrink-0 text-[10px] uppercase tracking-wide text-muted-foreground">
                    {provenanceLabel(node)}
                  </span>
                </div>
                <p className="mt-1 line-clamp-2 text-[11px] text-muted-foreground">
                  {node.summary}
                </p>
                {node.sourceIndex != null ? (
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    Source order {node.sourceIndex + 1}
                  </p>
                ) : null}
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-xs text-muted-foreground">No receipts in this lane.</p>
      )}
    </div>
  );
}

function PrimaryExecutionTimeline({
  observed,
  correlated,
  selectedId,
  onSelect,
}: {
  observed: ExecutionNode[];
  correlated: ExecutionNode[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="rounded-lg border border-border/60 p-3" data-testid="helix-primary-execution-timeline">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Backend execution order
          </h3>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            Solid links follow the shared backend sequence exactly.
          </p>
        </div>
        <span className="text-[11px] text-muted-foreground">{observed.length}</span>
      </div>
      <ol className="space-y-2">
        {observed.map((node, index) => (
          <li key={node.id} className="relative pl-5">
            {index > 0 ? (
              <span aria-hidden className="absolute -top-2 bottom-1/2 left-[5px] w-px bg-foreground/35" />
            ) : null}
            {index < observed.length - 1 ? (
              <span aria-hidden className="absolute top-1/2 -bottom-2 left-[5px] w-px bg-foreground/35" />
            ) : null}
            <span
              aria-hidden
              className="absolute left-[1px] top-1/2 size-[9px] -translate-y-1/2 rounded-full border border-foreground/60 bg-background"
            />
            <button
              type="button"
              onClick={() => onSelect(node.id)}
              aria-pressed={selectedId === node.id}
              className="w-full rounded-md border border-border/60 px-2.5 py-2 text-left hover:bg-muted/40 aria-pressed:border-foreground/35 aria-pressed:bg-muted/50"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-medium">{node.title}</span>
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                  seq {node.sequence ?? "?"} · {provenanceLabel(node)}
                </span>
              </div>
              <p className="mt-1 line-clamp-2 text-[11px] text-muted-foreground">{node.summary}</p>
            </button>
          </li>
        ))}
      </ol>

      {correlated.length ? (
        <div className="mt-4 border-t border-dashed border-border pt-3">
          <div className="mb-2 rounded-md border border-dashed border-border px-2.5 py-2 text-[11px] text-muted-foreground">
            Same-turn correlated receipts. A dashed boundary means the backend proved turn
            correlation, but did not provide a shared causal sequence for these receipts.
          </div>
          <div className="space-y-2 border-l border-dashed border-border pl-3">
            {correlated.map((node) => (
              <button
                key={node.id}
                type="button"
                onClick={() => onSelect(node.id)}
                aria-pressed={selectedId === node.id}
                className="block w-full rounded-md border border-border/60 px-2.5 py-2 text-left hover:bg-muted/40 aria-pressed:border-foreground/35 aria-pressed:bg-muted/50"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-medium">{node.title}</span>
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                    {provenanceLabel(node)}
                  </span>
                </div>
                <p className="mt-1 line-clamp-2 text-[11px] text-muted-foreground">{node.summary}</p>
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ExecutionNodeDetails({ node }: { node: ExecutionNode | null }) {
  if (!node) {
    return (
      <div className="rounded-lg border border-dashed border-border p-4 text-xs text-muted-foreground">
        Select a receipt to inspect why Helix recorded or decided it.
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-border/70 p-4" data-testid="helix-execution-details">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">{node.title}</h3>
        <span className="rounded border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
          {provenanceLabel(node)}
        </span>
        <span className="rounded border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
          {node.category}
        </span>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">{node.summary}</p>
      {node.sequence != null || node.createdAtMs != null ? (
        <p className="mt-1 text-[11px] text-muted-foreground">
          {node.sequence != null ? `Sequence ${node.sequence}` : "Sequence unavailable"}
          {node.createdAtMs != null
            ? ` · ${new Date(node.createdAtMs).toLocaleTimeString()}`
            : ""}
        </p>
      ) : null}
      <dl className="mt-3 space-y-2 text-xs">
        <div>
          <dt className="font-medium">Why</dt>
          <dd className="mt-0.5 whitespace-pre-wrap text-muted-foreground">{node.reason}</dd>
        </div>
        <div>
          <dt className="font-medium">Source</dt>
          <dd className="mt-0.5 break-all text-muted-foreground">{node.source}</dd>
        </div>
        <div>
          <dt className="font-medium">Authority</dt>
          <dd className="mt-0.5 text-muted-foreground">{authorityLabel(node)}</dd>
        </div>
        <div>
          <dt className="font-medium">Status</dt>
          <dd className="mt-0.5 text-muted-foreground">{node.status}</dd>
        </div>
        {node.receipts.map((receipt) => (
          <div key={`${receipt.label}:${receipt.value.slice(0, 80)}`}>
            <dt className="font-medium">{receipt.label}</dt>
            <dd className="mt-0.5 whitespace-pre-wrap break-words text-muted-foreground">
              {receipt.value}
            </dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 border-t border-border/60 pt-3">
        <p className="text-xs font-medium">Evidence</p>
        {node.evidence.length ? (
          <ul className="mt-1 space-y-1 text-xs text-muted-foreground">
            {node.evidence.map((item) => (
              <li key={item} className="break-words">{item}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-xs text-muted-foreground">No evidence items were attached.</p>
        )}
      </div>
      {node.missingEvidence.length ? (
        <div className="mt-3 border-t border-border/60 pt-3">
          <p className="text-xs font-medium">Missing evidence</p>
          <ul className="mt-1 space-y-1 text-xs text-muted-foreground">
            {node.missingEvidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

type HelixPanelMode = "workflow" | "execution";

type HelixEnginePanelProps = {
  threadId: string | null;
  onClose: () => void;
};

function HelixEnginePanel({
  mode,
  threadId,
  onClose,
}: HelixEnginePanelProps & { mode: HelixPanelMode }) {
  const activeProjectId = useChatRuntimeStore((state) => state.activeProjectId);
  const fallbackSessionId = engineSessionIdFor(threadId, activeProjectId);
  const [sessionBinding, setSessionBinding] = useState<SessionBinding | null>(null);
  const resolvedBinding =
    threadId && sessionBinding?.threadId === threadId ? sessionBinding : null;
  const sessionId = resolvedBinding?.sessionId ?? fallbackSessionId;
  const sessionScope = !threadId
    ? "No active chat selected"
    : resolvedBinding?.scope ?? "Resolving saved chat scope…";
  const sessionWarning = resolvedBinding?.warning ?? null;
  const pollKey = JSON.stringify([threadId ?? "", sessionId]);
  const [pollState, setPollState] = useState<PollState | null>(null);
  const currentPoll = pollState?.key === pollKey ? pollState : null;
  const events = currentPoll?.events ?? EMPTY_FEED_EVENTS;
  const steps = currentPoll?.steps ?? EMPTY_SESSION_STEPS;
  const controlEvents = currentPoll?.controlEvents ?? EMPTY_CONTROL_EVENTS;
  const provenance = currentPoll?.provenance ?? null;
  const provenanceError = currentPoll?.provenanceError ?? null;
  const pollStatus = currentPoll?.status ?? "connecting";
  const pollError = currentPoll?.error ?? null;
  const lastUpdatedAt = currentPoll?.lastUpdatedAt ?? null;
  const [turns, setTurns] = useState("");
  const [turnSignal, setTurnSignal] = useState<{
    unnecessary_turns?: number;
    feed_self_improvement?: boolean;
  } | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState | null>(null);
  const analysis = analysisState?.key === pollKey ? analysisState.value : null;
  const executionGraph = useMemo(
    () =>
      normalizeExecutionGraph({
        events,
        steps,
        controlEvents,
        analysis,
        provenance,
        provenanceError,
      }),
    [analysis, controlEvents, events, provenance, provenanceError, steps],
  );
  const observedExecutionNodes = executionGraph.nodes.filter(
    (node) =>
      node.source === "trajectory" ||
      node.source.includes("runtime_tool_loop") ||
      node.source.startsWith("live-feed") ||
      ((node.source === "provenance:adaptive-checkpoint-boundary" ||
        node.source === "provenance:adaptive-checkpoint") &&
        node.sequence != null),
  );
  const correlatedExecutionNodes = executionGraph.nodes.filter(
    (node) =>
      node.id.startsWith("provenance:") &&
      !(
        (node.source === "provenance:adaptive-checkpoint-boundary" ||
          node.source === "provenance:adaptive-checkpoint") &&
        node.sequence != null
      ),
  );
  const manualAnalysisNodes = executionGraph.nodes.filter((node) =>
    node.id.startsWith("analysis:"),
  );
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);
  const selectedExecution =
    executionGraph.nodes.find((node) => node.id === selectedExecutionId) ??
    executionGraph.nodes[0] ??
    null;
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  useEffect(() => {
    if (!threadId) return;
    let live = true;
    const controller = new AbortController();
    void getStoredChatThreadReadResult(threadId, {
      bounded: true,
      timeoutMs: 2_500,
      signal: controller.signal,
    })
      .then(({ thread }) => {
        if (!live) return;
        const projectId = engineProjectIdForThread(thread, activeProjectId);
        setSessionBinding({
          threadId,
          sessionId: engineSessionIdFor(threadId, projectId),
          scope: projectId
            ? `Project workspace · thread ${threadId}`
            : `Thread workspace · ${threadId}`,
          warning: null,
        });
      })
      .catch((cause: unknown) => {
        if (!live || controller.signal.aborted) return;
        setSessionBinding({
          threadId,
          sessionId: fallbackSessionId,
          scope: activeProjectId
            ? `Project workspace · thread ${threadId}`
            : `Thread workspace · ${threadId}`,
          warning: `Saved chat scope could not be verified (${helixErrorMessage(cause)}). Using the current app scope.`,
        });
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [threadId, activeProjectId, fallbackSessionId]);

  useEffect(() => {
    let live = true;
    let timer: number | null = null;
    const load = async () => {
      const snapshot = await loadHelixSnapshot(authFetch, sessionId, threadId);
      if (!live) return;
      setPollState((previous) => {
        const prior = previous?.key === pollKey ? previous : null;
        return {
          key: pollKey,
          events: snapshot.events ?? prior?.events ?? [],
          steps: snapshot.steps ?? prior?.steps ?? [],
          controlEvents: snapshot.controlEvents ?? prior?.controlEvents ?? [],
          provenance: snapshot.provenance ?? prior?.provenance ?? null,
          provenanceError: snapshot.provenanceError ?? null,
          status: snapshot.errors.length ? "degraded" : "connected",
          error: snapshot.errors.length ? snapshot.errors.join(" · ") : null,
          lastUpdatedAt:
            snapshot.successfulRequests > 0 ? Date.now() : (prior?.lastUpdatedAt ?? null),
        };
      });
      if (live) timer = window.setTimeout(() => void load(), 1_000);
    };
    void load();
    return () => {
      live = false;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [pollKey, sessionId, threadId]);

  async function analyzeSession() {
    setError(null);
    const payload = {
      prompt_state: `session ${sessionId}`,
      steps: steps.map((step) => ({
        name: step.name,
        arguments: step.arguments,
        result: step.result,
        useful_hint: step.useful_hint,
      })),
      // Manual inspection is not objective verification. The backend also
      // enforces this distinction even if an old client sends verified=true.
      verified: false,
      final_result: steps.at(-1)?.result ?? "",
    };
    try {
      const body = await fetchHelixJson<HelixAnalyzeResult>(
        authFetch,
        "/api/helix-engine/analyze",
        "Analyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      setAnalysisState({ key: pollKey, value: body });
    } catch (cause) {
      setError(helixErrorMessage(cause));
    }
  }

  async function checkTurns() {
    setError(null);
    try {
      const body = await fetchHelixJson<{
        unnecessary_turns?: number;
        feed_self_improvement?: boolean;
      }>(authFetch, "/api/helix-engine/semantic-turns", "Semantic-turn check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turns: turns
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean),
        }),
      });
      setTurnSignal(body);
    } catch (cause) {
      setError(helixErrorMessage(cause));
    }
  }

  return (
    <section
      className="flex h-full min-h-0 flex-col border-l border-border/70 bg-background"
      data-testid={`helix-${mode}-panel`}
      aria-label={mode === "execution" ? "Execution Graph" : "Helix Workflow"}
    >
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-border/70 px-3 py-2">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold">
            {mode === "execution" ? "Execution Graph" : "Helix Workflow"}
          </h2>
          <p className="truncate text-xs text-muted-foreground">Session {sessionId}</p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          aria-label="Back to chat"
          className="shrink-0 gap-1.5 px-2 text-xs"
        >
          <ArrowLeft className="size-4" />
          Back to chat
        </Button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3">
      <section className="rounded-xl border border-border p-4" data-testid="helix-engine-status">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Engine status</h2>
          <span className="text-xs text-muted-foreground" aria-live="polite">
            {pollStatus === "connecting"
              ? "Connecting"
              : pollStatus === "degraded"
                ? "Retrying"
                : "Connected"}
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{sessionScope}</p>
        {!threadId ? (
          <p className="mt-2 text-sm">
            Helix Engine is ready. Select a chat and complete a turn; this page will follow that
            conversation automatically.
          </p>
        ) : pollStatus === "connected" &&
          events.length === 0 &&
          steps.length === 0 &&
          controlEvents.length === 0 ? (
          <p className="mt-2 text-sm">
            Connected and idle. Tool activity and the latest completed trajectory will appear here
            as the selected chat runs.
          </p>
        ) : (
          <p className="mt-2 text-sm">
            Tracking {events.length} live event{events.length === 1 ? "" : "s"} and {steps.length}{" "}
            trajectory step{steps.length === 1 ? "" : "s"}
            {controlEvents.length
              ? ` plus ${controlEvents.length} runtime policy receipt${controlEvents.length === 1 ? "" : "s"}.`
              : "."}
          </p>
        )}
        {lastUpdatedAt ? (
          <p className="mt-1 text-xs text-muted-foreground">
            Last refresh {new Date(lastUpdatedAt).toLocaleTimeString()}
          </p>
        ) : null}
        {sessionWarning ? (
          <p className="mt-2 text-xs text-muted-foreground" role="status">
            {sessionWarning}
          </p>
        ) : null}
        {pollError ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            Telemetry refresh issue: {pollError}. Retrying automatically; already loaded data stays
            visible.
          </p>
        ) : null}
      </section>

      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : null}

      {mode === "execution" ? (
      <section
        className="rounded-xl border border-border p-4"
        data-testid="helix-execution-graph"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">Execution graph</h2>
            <p className="mt-1 max-w-2xl text-xs text-muted-foreground">
              Deterministic receipt timeline for “why did Helix do this?”. When every observed
              receipt carries the backend sequence, tool, policy, and browse lanes are interleaved
              exactly. Otherwise edges remain inside each ordered source stream and cross-stream
              ordering is left unknown.
            </p>
          </div>
          <div className="text-right text-[11px] text-muted-foreground">
            <p>{executionGraph.nodes.length} receipts</p>
            <p>{executionGraph.edges.length} observed-order edges</p>
          </div>
        </div>

        {executionGraph.nodes.length ? (
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.55fr)_minmax(18rem,1fr)]">
            <div className="grid gap-3 sm:grid-cols-2">
              {executionGraph.orderingMode === "exact" ? (
                <div className="sm:col-span-2">
                  <PrimaryExecutionTimeline
                    observed={observedExecutionNodes}
                    correlated={correlatedExecutionNodes}
                    selectedId={selectedExecution?.id ?? null}
                    onSelect={setSelectedExecutionId}
                  />
                  {manualAnalysisNodes.length ? (
                    <div className="mt-3">
                      <ExecutionLane
                        title="Manual Analyze · advisory"
                        nodes={manualAnalysisNodes}
                        selectedId={selectedExecution?.id ?? null}
                        onSelect={setSelectedExecutionId}
                      />
                    </div>
                  ) : null}
                </div>
              ) : (
                <>
                  <ExecutionLane
                    title="Runtime policy"
                    nodes={executionGraph.nodes.filter((node) => node.source.includes("runtime_tool_loop"))}
                    selectedId={selectedExecution?.id ?? null}
                    onSelect={setSelectedExecutionId}
                  />
                  <ExecutionLane
                    title="Tool trajectory"
                    nodes={executionGraph.nodes.filter((node) => node.source === "trajectory")}
                    selectedId={selectedExecution?.id ?? null}
                    onSelect={setSelectedExecutionId}
                  />
                  <ExecutionLane
                    title="Browse / computer"
                    nodes={executionGraph.nodes.filter((node) => node.source.startsWith("live-feed"))}
                    selectedId={selectedExecution?.id ?? null}
                    onSelect={setSelectedExecutionId}
                  />
                  <ExecutionLane
                    title="Persisted Helix learning"
                    nodes={correlatedExecutionNodes}
                    selectedId={selectedExecution?.id ?? null}
                    onSelect={setSelectedExecutionId}
                  />
                  {manualAnalysisNodes.length ? (
                    <ExecutionLane
                      title="Manual Analyze · advisory"
                      nodes={manualAnalysisNodes}
                      selectedId={selectedExecution?.id ?? null}
                      onSelect={setSelectedExecutionId}
                    />
                  ) : null}
                </>
              )}
            </div>
            <ExecutionNodeDetails node={selectedExecution} />
          </div>
        ) : (
          <div className="mt-4 rounded-lg border border-dashed border-border p-4 text-xs text-muted-foreground">
            No execution receipts are available for this session yet. Complete a tool-using turn,
            browse action, or run Analyze to populate the graph.
          </div>
        )}

        <div className="mt-4 border-t border-border/60 pt-3" data-testid="helix-provenance-gaps">
          <h3 className="text-xs font-semibold">Provenance gaps</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">
            These are missing backend receipts, not inferred failures.
          </p>
          <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
            {executionGraph.coverageGaps.map((gap) => (
              <li key={gap}>• {gap}</li>
            ))}
          </ul>
        </div>
      </section>
      ) : null}

      {mode === "workflow" ? (
      <>
      <section className="rounded-xl border border-border p-4">
        <h2 className="text-sm font-semibold">Live computer / browse feed</h2>
        {events.length ? (
          events.slice(-12).map((event, index) => (
            <div key={`${event.action}-${index}`} className="mt-2 rounded-lg border border-border/70 p-2 text-sm">
              <p className="font-medium">
                {event.kind || "computer"} · {event.action}
              </p>
              {event.title ? <p className="text-xs">{event.title}</p> : null}
              {event.url ? <p className="break-all text-xs text-muted-foreground">{event.url}</p> : null}
            </div>
          ))
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">No browse or computer-use events yet for this session.</p>
        )}
      </section>

      <section className="rounded-xl border border-border p-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Trajectory</h2>
          <button
            type="button"
            className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void analyzeSession()}
            disabled={steps.length === 0}
            title={steps.length ? "Analyze the captured trajectory" : "No captured tool steps yet"}
          >
            Analyze
          </button>
        </div>
        {steps.length ? (
          <ol className="mt-2 space-y-1 text-xs">
            {steps.map((step, index) => (
              <li key={`${step.name}-${index}`} className="rounded-md border border-border/60 p-2">
                <span className="font-medium">{step.name}</span>
                <span className="text-muted-foreground"> · {step.useful_hint || "useful"}</span>
                <p className="mt-1 truncate text-muted-foreground">{step.arguments}</p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">Tool steps appear here as the model works.</p>
        )}
        {analysis ? (
          <div className="mt-3 space-y-1 text-xs">
            <p>Route: {analysis.route} · Decision: {analysis.decision}</p>
            <p>Success authorizes weights: {String(analysis.success_authorizes_weight_update)}</p>
            <p>Kept steps: {(analysis.credit ?? []).join(", ") || "none"}</p>
            <p>Gates: {(analysis.gates ?? []).join(" → ")}</p>
          </div>
        ) : null}
      </section>

      <section className="rounded-xl border border-border p-4">
        <h2 className="text-sm font-semibold">Unnecessary semantic turns</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Paste ChatGPT/Codex-style filler turns, one per line. Helix Engine feeds this to self-improvement.
        </p>
        <textarea
          className="mt-2 min-h-24 w-full rounded-md border border-border bg-background p-2 text-sm"
          value={turns}
          onChange={(event) => setTurns(event.target.value)}
          placeholder={"ok\nsure\nlet me think"}
        />
        <button
          type="button"
          className="mt-2 rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground"
          onClick={() => void checkTurns()}
        >
          Monitor
        </button>
        {turnSignal ? (
          <p className="mt-2 text-xs">
            Unnecessary: {turnSignal.unnecessary_turns} · Feed self-improvement:{" "}
            {String(turnSignal.feed_self_improvement)}
          </p>
        ) : null}
      </section>
      </>
      ) : null}
      </div>
    </section>
  );
}

export function HelixWorkflowPanel(props: HelixEnginePanelProps) {
  return <HelixEnginePanel {...props} mode="workflow" />;
}

export function HelixExecutionGraphPanel(props: HelixEnginePanelProps) {
  return <HelixEnginePanel {...props} mode="execution" />;
}
