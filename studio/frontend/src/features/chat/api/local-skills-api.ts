// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LocalSkill = {
  name: string;
  description: string;
  path: string;
  ecosystems: Array<"codex" | "claude">;
};

export type LocalSkillDocument = {
  name: string;
  description: string;
  instructions: string;
};

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export async function listLocalSkills(): Promise<LocalSkill[]> {
  const response = await authFetch("/api/skills/local");
  return (await jsonOrThrow<{ skills: LocalSkill[] }>(response)).skills;
}

export async function readLocalSkill(name: string): Promise<LocalSkillDocument> {
  const response = await authFetch("/api/skills/" + encodeURIComponent(name));
  return await jsonOrThrow<LocalSkillDocument>(response);
}

export async function createLocalSkill(input: {
  name: string;
  description: string;
  instructions: string;
  target: "codex" | "claude" | "both";
}): Promise<LocalSkill[]> {
  const response = await authFetch("/api/skills/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return (await jsonOrThrow<{ skills: LocalSkill[] }>(response)).skills;
}

export async function installLocalSkill(input: {
  source: string;
  target: "codex" | "claude" | "both";
}): Promise<LocalSkill[]> {
  const response = await authFetch("/api/skills/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return (await jsonOrThrow<{ skills: LocalSkill[] }>(response)).skills;
}
