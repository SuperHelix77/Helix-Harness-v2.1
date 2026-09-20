// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatMode } from "./chat-mode";
import { hermesLearningReviewPrompt } from "./hermes-learning.ts";

export const LOCAL_SKILLS = [
  {
    command: "/github",
    label: "GitHub context",
    description: "Read your linked repositories and ledgers without a URL",
    mode: null,
  },
  {
    command: "/plan",
    label: "Plan mode",
    description: "Analyze and propose changes before acting",
    mode: "plan" as ChatMode,
  },
  {
    command: "/goal",
    label: "Goal mode",
    description: "Work toward an objective and verify the result",
    mode: "goal" as ChatMode,
  },
  {
    command: "/speed",
    label: "Speed diagnostics",
    description: "Report actual tok/s and speculative-decoding state",
    mode: null,
  },
  {
    command: "/learn",
    label: "Hermes learning review",
    description: "Propose one durable lesson for approval without self-editing code",
    mode: null,
  },
  {
    command: "/reflect",
    label: "Self-reflect",
    description: "Ask: new skill, runtime fix, or QLoRA? Store durable facts in Mem0",
    mode: null,
  },
] as const;

export type InstalledLocalSkill = {
  name: string;
  description: string;
  path: string;
  ecosystems: readonly ("codex" | "claude")[];
};

export type LocalSkillMenuItem = {
  command: string;
  label: string;
  description: string;
  mode?: ChatMode | null;
};

export type LocalSlashCommand = {
  command: string;
  args: string;
  message: string;
  mode?: ChatMode;
  handled: boolean;
  help?: boolean;
  skillName?: string;
};

/** Return the command fragment while the entire draft is still a local-command prefix.
 * This deliberately accepts both `/` and `/ ` so the picker can open before a command is chosen. */
export function localSlashQuery(input: string): string | null {
  const match = input.match(/^\s*\/([a-z0-9._-]*)\s*$/i);
  return match ? match[1].toLowerCase() : null;
}

export function localSkillsForQuery(
  query: string,
  installedSkills: readonly InstalledLocalSkill[] = [],
): LocalSkillMenuItem[] {
  const dynamicSkills = installedSkills
    .filter((skill) => !LOCAL_SKILLS.some((builtin) => builtin.command.slice(1) === skill.name))
    .map((skill) => ({
      command: "/" + skill.name,
      label: skill.name,
      description: skill.description || "Installed SKILL.md",
    }));
  return [...LOCAL_SKILLS, ...dynamicSkills].filter(
    (skill) =>
      !query ||
      skill.command.slice(1).startsWith(query) ||
      skill.label.toLowerCase().includes(query),
  );
}

export const LOCAL_SKILLS_OPEN_EVENT = "unsloth-open-local-skills";
export const LOCAL_SKILLS_CHANGED_EVENT = "unsloth-local-skills-changed";

/** Open the manager from a slash-menu item without moving focus to a second composer. */
export function openLocalSkillsManager(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(LOCAL_SKILLS_OPEN_EVENT));
  }
}

export function localSkillPrompt(
  name: string,
  instructions: string,
  request: string,
): string {
  // Keep an accidentally enormous skill from consuming the whole model window. The
  // manager stores the complete file; invocation only needs the instruction body.
  const bounded = instructions.slice(0, 32_000);
  const truncation =
    instructions.length > bounded.length
      ? "\n[The skill file was truncated to keep room for the user request.]"
      : "";
  return [
    "Apply the installed local skill /" + name + " to this request.",
    "",
    "Skill instructions:",
    bounded + truncation,
    "",
    "User request:",
    request || "Apply /" + name + " to the current task.",
  ].join("\n");
}

const HELP_TEXT =
  "Local commands: /github (read linked GitHub), /plan, /goal, /speed, /learn, /reflect, and /skills. " +
  "You can also choose them from the Skills button.";

/** Parse only commands owned by Helix Harness; ordinary slash-prefixed prose is preserved. */
export function parseLocalSlashCommand(
  input: string,
  installedSkills: readonly InstalledLocalSkill[] = [],
): LocalSlashCommand | null {
  const match = input.trim().match(/^\/([a-z][a-z0-9._-]*)(?:\s+([\s\S]*))?$/i);
  if (!match) return null;
  const name = match[1].toLowerCase();
  const args = match[2]?.trim() ?? "";
  if (name === "skills" || name === "help") {
    return { command: "/skills", args, message: "", handled: true, help: true };
  }
  if (name === "plan" || name === "goal") {
    return {
      command: "/" + name,
      args,
      message: args,
      mode: name,
      handled: args.length === 0,
    };
  }
  if (name === "github" || name === "git") {
    return {
      command: "/github",
      args,
      message:
        "Use github_context to inspect my authenticated GitHub repositories and relevant ledger files. " +
        "Do not ask me for a repository URL. " +
        (args || "Find the relevant v1.1, future-ledger, MTP, DFlash, and DFlare evidence."),
      handled: false,
    };
  }
  if (name === "speed" || name === "perf" || name === "performance") {
    return {
      command: "/speed",
      args,
      message:
        "Inspect the current local inference runtime and report measured tok/s, context length, " +
        "requested speculative mode, engaged speculative mode, and accepted draft tokens. " +
        "Do not claim a multiplier unless the runtime metrics support it. " +
        (args ? "Additional focus: " + args : ""),
      handled: false,
    };
  }
  if (name === "learn" || name === "hermes" || name === "reflect") {
    return {
      command: name === "reflect" ? "/reflect" : "/learn",
      args,
      message: hermesLearningReviewPrompt(args),
      handled: false,
    };
  }
  if (installedSkills.some((skill) => skill.name === name)) {
    return {
      command: "/" + name,
      args,
      message: args,
      handled: false,
      skillName: name,
    };
  }
  return null;
}

export function localSkillsHelp(): string {
  return HELP_TEXT;
}
