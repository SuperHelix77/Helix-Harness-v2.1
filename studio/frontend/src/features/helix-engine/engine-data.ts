// SPDX-License-Identifier: AGPL-3.0-only
import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";

export type FeedEvent = {
  sequence?: number;
  created_at_ms?: number;
  turn_id?: string;
  action?: string;
  url?: string;
  kind?: string;
  title?: string;
  snippet?: string;
  ok?: boolean;
  engine?: string;
};

export type SessionStep = {
  sequence?: number;
  created_at_ms?: number;
  index?: number;
  evidence_ids?: string[];
  name: string;
  arguments: string;
  result: string;
  useful_hint: string;
  error?: string | null;
  retry?: number;
};

export type ToolControlEvent = {
  sequence?: number;
  created_at_ms?: number;
  schema_version?: string;
  action?: string;
  tool_name?: string;
  arguments?: string;
  reason?: string;
  equivalent_to?: string;
  failed_attempts?: number;
  progress?: Record<string, unknown>;
  provenance?: string;
};

export type AdaptiveCheckpointReceipt = Record<string, unknown> & {
  start_sequence?: number;
  start_created_at_ms?: number;
  audit_sequence?: number;
  audit_created_at_ms?: number;
  resume_sequence?: number;
  resume_created_at_ms?: number;
};

export type SkillRetentionReceipt = {
  skill_name?: string;
  disposition?: string;
  reason?: string;
  evidence?: Record<string, unknown> | null;
};

export type QloraOutcomeReceipt = {
  outcome?: string;
  reason?: string;
};

export type HelixProvenance = {
  schema_version?: string;
  available?: boolean;
  response_truncated?: boolean;
  selection?: {
    source?: string;
    session_id?: string | null;
    thread_id?: string | null;
    turn_id?: string | null;
    trajectory_id?: string | null;
    scope_ambiguous?: boolean;
  };
  turn_receipt?: Record<string, unknown> | null;
  adaptive_checkpoints?: AdaptiveCheckpointReceipt[];
  mechanisms?: Record<string, unknown> | null;
  trajectory?: Record<string, unknown> | null;
  evidence?: Record<string, unknown> | null;
  self_audit?: Record<string, unknown> | null;
  adaptation?: Record<string, unknown> | null;
  quality?: Record<string, unknown> | null;
  decisions?: Array<Record<string, unknown>>;
  decision_outcomes?: Array<Record<string, unknown>>;
  counterfactual?: Record<string, unknown> | null;
  qlora_admission?: Record<string, unknown> | null;
  training_target_receipt?: Record<string, unknown> | null;
  skill_retention?: SkillRetentionReceipt[];
  qlora_outcome?: QloraOutcomeReceipt | null;
  final_actions?: string[];
  missing_sources?: string[];
  truncated_sources?: string[];
};

type JsonFetcher = (input: string, init?: RequestInit) => Promise<Response>;

export function engineProjectIdForThread(
  storedThread: { projectId?: string | null } | undefined,
  activeProjectId: string | null,
): string | null {
  return storedThread ? (storedThread.projectId ?? null) : activeProjectId;
}

export function engineSessionIdFor(
  threadId: string | null | undefined,
  projectId: string | null | undefined,
): string {
  return sandboxSessionIdFor(threadId ?? undefined, projectId) ?? threadId ?? "default";
}

export function helixErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message.trim();
  if (typeof error === "string" && error.trim()) return error.trim();
  return "Unknown Helix Engine error";
}

function responseDetail(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const detail = (body as { detail?: unknown }).detail;
  return typeof detail === "string" ? detail.trim() : "";
}

export async function fetchHelixJson<T>(
  fetcher: JsonFetcher,
  url: string,
  label: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetcher(url, init);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      `${label}: ${response.ok ? "server returned an unreadable response" : `HTTP ${response.status}`}`,
    );
  }
  if (!response.ok) {
    throw new Error(`${label}: ${responseDetail(body) || `HTTP ${response.status}`}`);
  }
  return body as T;
}

export type HelixSnapshot = {
  events?: FeedEvent[];
  steps?: SessionStep[];
  controlEvents?: ToolControlEvent[];
  turnId?: string | null;
  provenance?: HelixProvenance;
  provenanceError?: string | null;
  errors: string[];
  successfulRequests: number;
};

export async function loadHelixSnapshot(
  fetcher: JsonFetcher,
  sessionId: string,
  threadId?: string | null,
): Promise<HelixSnapshot> {
  const traceQuery = threadId
    ? `?thread_id=${encodeURIComponent(threadId)}`
    : "";
  const feedParams = new URLSearchParams({ session_id: sessionId });
  if (threadId) feedParams.set("thread_id", threadId);
  const requests = await Promise.allSettled([
    fetchHelixJson<{ events?: FeedEvent[] }>(
      fetcher,
      `/api/helix-engine/live-feed?${feedParams.toString()}`,
      "Live feed",
    ),
    fetchHelixJson<{
      turn_id?: string | null;
      steps?: SessionStep[];
      control_events?: ToolControlEvent[];
      provenance?: HelixProvenance;
    }>(
      fetcher,
      `/api/helix-engine/session/${encodeURIComponent(sessionId)}${traceQuery}`,
      "Trajectory",
    ),
  ]);
  const snapshot: HelixSnapshot = { errors: [], successfulRequests: 0 };
  const [feed, trace] = requests;
  if (feed.status === "fulfilled") {
    snapshot.events = feed.value.events ?? [];
    snapshot.successfulRequests += 1;
  } else {
    snapshot.errors.push(helixErrorMessage(feed.reason));
  }
  if (trace.status === "fulfilled") {
    snapshot.steps = trace.value.steps ?? [];
    snapshot.controlEvents = trace.value.control_events ?? [];
    snapshot.turnId = trace.value.turn_id ?? null;
    const traceSequences = [
      ...snapshot.steps.map((step) => step.sequence),
      ...snapshot.controlEvents.map((event) => event.sequence),
    ].filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    const latestTaggedFeed = (snapshot.events ?? []).reduce<FeedEvent | null>((latest, event) => {
      if (!event.turn_id || typeof event.sequence !== "number" || !Number.isFinite(event.sequence)) {
        return latest;
      }
      if (!latest || (latest.sequence ?? -1) < event.sequence) return event;
      return latest;
    }, null);
    const traceMaxSequence = traceSequences.length ? Math.max(...traceSequences) : null;
    const feedProvesNewerTurn = Boolean(
      latestTaggedFeed?.turn_id &&
      latestTaggedFeed.turn_id !== snapshot.turnId &&
      (snapshot.turnId == null ||
        (traceMaxSequence != null && (latestTaggedFeed.sequence ?? -1) > traceMaxSequence)),
    );
    if (feedProvesNewerTurn && latestTaggedFeed?.turn_id) {
      // Computer/browse observation can precede the enclosing tool result. The
      // shared backend sequence clock proves that this tagged event belongs to a
      // newer live turn, so do not combine it with the stale archived trace.
      snapshot.turnId = latestTaggedFeed.turn_id;
      snapshot.steps = [];
      snapshot.controlEvents = [];
    }
    if (snapshot.turnId && snapshot.events) {
      // The live feed is thread-scoped and intentionally retains recent history.
      // Exclude events that are positively correlated to another logical turn;
      // keep legacy untagged events so the graph can expose their membership gap
      // instead of silently inventing or discarding provenance.
      snapshot.events = snapshot.events.filter(
        (event) => !event.turn_id || event.turn_id === snapshot.turnId,
      );
    }
    snapshot.provenance = trace.value.provenance;
    snapshot.successfulRequests += 1;
  } else {
    snapshot.errors.push(helixErrorMessage(trace.reason));
  }

  if (trace.status === "fulfilled") {
    const params = new URLSearchParams({ session_id: sessionId });
    if (threadId) params.set("thread_id", threadId);
    if (snapshot.turnId) params.set("turn_id", snapshot.turnId);
    try {
      snapshot.provenance = await fetchHelixJson<HelixProvenance>(
        fetcher,
        `/api/helix-engine/provenance?${params.toString()}`,
        "Provenance",
      );
      snapshot.provenanceError = null;
      snapshot.successfulRequests += 1;
    } catch (cause) {
      // Provenance is an additive debug surface. Older/partial backends may not
      // expose it; keep live feed + trajectory usable and make the gap explicit.
      snapshot.provenanceError = helixErrorMessage(cause);
    }
  }
  return snapshot;
}
