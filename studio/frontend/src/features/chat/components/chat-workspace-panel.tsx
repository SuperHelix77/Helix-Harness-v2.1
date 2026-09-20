// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowLeft } from "lucide-react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";

import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { cn } from "@/lib/utils";
import { type MemoryGraphSnapshot, fetchMemoryGraph } from "../api/memory-api";
import { getSelfTrainingState } from "../api/self-training-api";
import { useChatArtifactsStore } from "../artifacts/store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useResearchRunStore } from "../stores/research-run-store";
import { terminalResearchStatuses } from "../stores/research-run-store";
import type { MessageRecord } from "../types";
import { listStoredChatMessages } from "../utils/chat-history-storage";

type WorkspaceTab =
	| "outputs"
	| "background"
	| "sources"
	| "subagents"
	| "memory"
	| "live";

function messageText(message: MessageRecord): string {
	if (!Array.isArray(message.content)) return "";
	return message.content
		.map((part) => {
			if (!part || typeof part !== "object") return "";
			const value = part as { text?: unknown; content?: unknown };
			return typeof value.text === "string"
				? value.text
				: typeof value.content === "string"
					? value.content
					: "";
		})
		.join("\n");
}

type ToolEvent = { id: string; name: string; status: string };

function toolEvents(message: MessageRecord): ToolEvent[] {
	if (!Array.isArray(message.content)) return [];
	const events: ToolEvent[] = [];
	for (const [index, part] of message.content.entries()) {
		if (!part || typeof part !== "object") continue;
		const typed = part as {
			type?: unknown;
			name?: unknown;
			toolName?: unknown;
			status?: unknown;
		};
		const type = typed.type;
		if (
			type !== "tool-call" &&
			type !== "tool-call-start" &&
			type !== "tool-result"
		)
			continue;
		const name =
			typeof typed.name === "string"
				? typed.name
				: typeof typed.toolName === "string"
					? typed.toolName
					: "tool";
		events.push({
			id: `${message.id}-${index}`,
			name,
			status:
				type === "tool-result" ? "done" : String(typed.status || "running"),
		});
	}
	return events;
}

export function ChatWorkspacePanel({
	threadId,
	onClose,
}: {
	threadId: string | null;
	onClose: () => void;
}): ReactElement {
	const [tab, setTab] = useState<WorkspaceTab>("outputs");
	const [messages, setMessages] = useState<MessageRecord[]>([]);
	const [graph, setGraph] = useState<MemoryGraphSnapshot>({
		nodes: [],
		edges: [],
		available: false,
	});
	const [trainingStatus, setTrainingStatus] = useState<string>("idle");
	const [liveEvents, setLiveEvents] = useState<
		Array<{
			action?: string;
			url?: string;
			kind?: string;
			title?: string;
			snippet?: string;
		}>
	>([]);
	const tabRefs = useRef<
		Partial<Record<WorkspaceTab, HTMLButtonElement | null>>
	>({});
	const activeProjectId = useChatRuntimeStore((state) => state.activeProjectId);
	const liveSessionId =
		sandboxSessionIdFor(threadId ?? undefined, activeProjectId) ??
		threadId ??
		"default";
	const artifactsById = useChatArtifactsStore((state) => state.artifactsById);
	const artifacts = useMemo(
		() =>
			Object.values(artifactsById).filter(
				(artifact) =>
					!threadId || !artifact.threadId || artifact.threadId === threadId,
			),
		[artifactsById, threadId],
	);
	const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
	const sessions = useResearchRunStore((state) => state.sessions);
	const threadResearch = useMemo(
		() =>
			Object.values(sessions).filter(
				(session) => session.run.threadId === threadId,
			),
		[sessions, threadId],
	);

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
		let live = true;
		const load = () => {
			if (!threadId) {
				setMessages([]);
				return;
			}
			void listStoredChatMessages(threadId)
				.then((next) => {
					if (live) {
						setMessages((prev) => (prev === next ? prev : next));
					}
				})
				.catch(() => {
					if (live) setMessages([]);
				});
		};
		load();
		const timer = window.setInterval(load, 2_000);
		return () => {
			live = false;
			window.clearInterval(timer);
		};
	}, [threadId]);

	useEffect(() => {
		let live = true;
		const load = () => {
			void fetchMemoryGraph()
				.then((next) => {
					if (live) setGraph(next);
				})
				.catch(() => undefined);
			void authFetch(
				`/api/helix-engine/live-feed?session_id=${encodeURIComponent(liveSessionId)}`,
			)
				.then((response) => response.json())
				.then(
					(body: {
						events?: Array<{
							action?: string;
							url?: string;
							kind?: string;
							title?: string;
							snippet?: string;
						}>;
					}) => {
						if (live) setLiveEvents(body.events ?? []);
					},
				)
				.catch(() => undefined);
			void getSelfTrainingState()
				.then((state) => {
					if (live) setTrainingStatus(String(state.status || "idle"));
				})
				.catch(() => undefined);
		};
		load();
		const timer = window.setInterval(load, 1_000);
		return () => {
			live = false;
			window.clearInterval(timer);
		};
	}, [threadId, tab, liveSessionId]);

	const sources = useMemo(() => {
		const values = new Set<string>();
		for (const message of messages) {
			for (const url of messageText(message).match(
				/https?:\/\/[^\s)\]}>,]+/g,
			) ?? []) {
				values.add(url);
			}
		}
		for (const session of threadResearch) {
			for (const source of session.run.sources ?? []) {
				if (source.url) values.add(source.url);
			}
		}
		return [...values];
	}, [messages, threadResearch]);

	const tools = messages.flatMap(toolEvents);
	const activeResearch = threadResearch.filter(
		(session) => !terminalResearchStatuses.has(session.run.status),
	);
	const memoryNodes = graph.nodes.filter(
		(node) => !threadId || !node.thread_id || node.thread_id === threadId,
	);
	const tabs: Array<{ id: WorkspaceTab; label: string; count: number }> = [
		{ id: "outputs", label: "Outputs", count: artifacts.length },
		{
			id: "background",
			label: "Background",
			count: activeResearch.length + (trainingStatus !== "idle" ? 1 : 0),
		},
		{ id: "sources", label: "Sources", count: sources.length },
		{ id: "subagents", label: "Subagents", count: tools.length },
		{ id: "memory", label: "Memory", count: memoryNodes.length },
		{ id: "live", label: "Live feed", count: liveEvents.length },
	];

	return (
		<section
			className="helix-right-rail flex h-full min-h-0 flex-col border-l border-border/70"
			aria-label="Chat workspace"
		>
			<header className="flex shrink-0 items-center justify-between border-b border-border/70 px-3 py-2">
				<div>
					<h2 className="text-sm font-semibold">Workspace</h2>
					<p className="text-xs text-muted-foreground">
						Outputs, memory graph, and agent activity
					</p>
				</div>
				<Button
					variant="ghost"
					size="sm"
					onClick={onClose}
					aria-label="Back to chat"
					className="gap-1.5 px-2 text-xs"
				>
					<ArrowLeft className="size-4" />
					Back to chat
				</Button>
			</header>
			<nav
				className="flex shrink-0 gap-1 overflow-x-auto border-b border-border/70 px-2 py-1"
				aria-label="Workspace tabs"
				role="tablist"
				aria-orientation="horizontal"
			>
				{tabs.map((item) => (
					<button
						key={item.id}
						ref={(node) => {
							tabRefs.current[item.id] = node;
						}}
						type="button"
						onClick={() => setTab(item.id)}
						onKeyDown={(event) => {
							const currentIndex = tabs.findIndex(
								(candidate) => candidate.id === item.id,
							);
							let nextIndex: number | null = null;
							if (event.key === "ArrowRight")
								nextIndex = (currentIndex + 1) % tabs.length;
							if (event.key === "ArrowLeft")
								nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
							if (event.key === "Home") nextIndex = 0;
							if (event.key === "End") nextIndex = tabs.length - 1;
							if (nextIndex === null) return;
							event.preventDefault();
							const nextTab = tabs[nextIndex];
							setTab(nextTab.id);
							tabRefs.current[nextTab.id]?.focus();
						}}
						className={cn(
							"rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
							tab === item.id && "bg-muted font-medium text-foreground",
						)}
						id={`workspace-tab-${item.id}`}
						aria-controls={`workspace-panel-${item.id}`}
						aria-selected={tab === item.id}
						tabIndex={tab === item.id ? 0 : -1}
						role="tab"
					>
						{item.label}
						{item.count ? ` · ${item.count}` : ""}
					</button>
				))}
			</nav>
			<div
				className="min-h-0 flex-1 overflow-y-auto p-3 text-sm"
				id={`workspace-panel-${tab}`}
				role="tabpanel"
				aria-labelledby={`workspace-tab-${tab}`}
			>
				{tab === "outputs" ? (
					artifacts.length ? (
						artifacts.map((artifact) => (
							<button
								key={artifact.id}
								type="button"
								className="mb-2 block w-full rounded-lg border border-border/70 p-3 text-left hover:bg-muted/40"
								onClick={() =>
									openArtifact(artifact, { surface: "panel", view: "preview" })
								}
							>
								<span className="block font-medium">{artifact.title}</span>
								<span className="text-xs text-muted-foreground">
									{artifact.source === "tool" ? "Tool output" : "Code output"}
								</span>
							</button>
						))
					) : (
						<p className="text-xs text-muted-foreground">
							Outputs appear here when the agent creates an artifact.
						</p>
					)
				) : null}
				{tab === "background" ? (
					<>
						<div className="mb-2 rounded-lg border border-border/70 p-3">
							<p className="font-medium">Self-QLoRA</p>
							<p className="text-xs text-muted-foreground">{trainingStatus}</p>
						</div>
						{activeResearch.length ? (
							activeResearch.map((session) => (
								<div
									key={session.run.id}
									className="mb-2 rounded-lg border border-border/70 p-3"
								>
									<p className="font-medium">Research activity</p>
									<p className="text-xs text-muted-foreground">
										{session.run.status} · {session.activities.length} events
									</p>
								</div>
							))
						) : (
							<p className="text-xs text-muted-foreground">
								No active background research. Training and memory handoff
								status show above.
							</p>
						)}
					</>
				) : null}
				{tab === "sources" ? (
					sources.length ? (
						sources.map((source) => (
							<a
								key={source}
								href={source}
								target="_blank"
								rel="noreferrer"
								className="mb-2 block break-all rounded-lg border border-border/70 p-2 text-xs text-primary hover:underline"
							>
								{source}
							</a>
						))
					) : (
						<p className="text-xs text-muted-foreground">
							Sources cited by this task will appear here.
						</p>
					)
				) : null}
				{tab === "subagents" ? (
					tools.length ? (
						tools.map((event) => (
							<div
								key={event.id}
								className="mb-2 rounded-lg border border-border/70 p-3"
							>
								<p className="font-medium">{event.name}</p>
								<p className="text-xs text-muted-foreground">{event.status}</p>
							</div>
						))
					) : (
						<div className="rounded-lg border border-border/70 p-3">
							<p className="font-medium">Local agent activity</p>
							<p className="mt-1 text-xs text-muted-foreground">
								Tool calls from this task appear here as they run.
							</p>
						</div>
					)
				) : null}
				{tab === "memory" ? (
					memoryNodes.length ? (
						<>
							{memoryNodes.map((node) => (
								<div
									key={node.id || node.title}
									className="mb-2 rounded-lg border border-border/70 p-3"
								>
									<p className="font-medium">{node.title || "Memory"}</p>
									<p className="mt-1 text-xs text-muted-foreground">
										{(node.text || "").slice(0, 240)}
									</p>
									{node.entities?.length ? (
										<p className="mt-1 text-xs text-muted-foreground">
											{node.entities.join(" · ")}
										</p>
									) : null}
								</div>
							))}
							{graph.edges.length ? (
								<p className="mt-2 text-xs text-muted-foreground">
									{graph.edges
										.slice(0, 12)
										.map(
											(edge) =>
												`${edge.from} —${edge.relation || "related"}→ ${edge.to}`,
										)
										.join(" · ")}
								</p>
							) : null}
						</>
					) : (
						<p className="text-xs text-muted-foreground">
							Permanent memories appear here after a context handoff or
							self-reflect. The model recalls them with search_memory.
						</p>
					)
				) : null}
				{tab === "live" ? (
					<div data-testid="helix-live-feed">
						{liveEvents.length ? (
							liveEvents.map((event, index) => (
								<div
									key={`${event.action}-${index}`}
									className="mb-2 rounded-lg border border-border/70 p-3"
								>
									<p className="font-medium">
										{event.kind || "computer"} · {event.action}
									</p>
									{event.title ? (
										<p className="text-xs">{event.title}</p>
									) : null}
									{event.url ? (
										<p className="text-xs text-muted-foreground break-all">
											{event.url}
										</p>
									) : null}
									{event.snippet ? (
										<p className="mt-1 text-xs text-muted-foreground">
											{event.snippet.slice(0, 240)}
										</p>
									) : null}
								</div>
							))
						) : (
							<p className="text-xs text-muted-foreground">
								Live feed of browse and computer-use actions appears here as the
								model works.
							</p>
						)}
					</div>
				) : null}
			</div>
		</section>
	);
}
