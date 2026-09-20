// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ClipboardList, Gauge, GitBranch, Sparkles, Target } from "lucide-react";
import type { ReactElement } from "react";

import {
  type InstalledLocalSkill,
  localSkillsForQuery,
  openLocalSkillsManager,
} from "./lib/local-skills";

const SKILL_ICONS = {
  "/github": GitBranch,
  "/plan": ClipboardList,
  "/goal": Target,
  "/speed": Gauge,
} as const;

export function LocalSkillsCommandMenu({
  query,
  activeIndex,
  onSelect,
  onManage,
  installedSkills,
}: {
  query: string;
  activeIndex: number;
  onSelect: (command: string) => void;
  onManage?: () => void;
  installedSkills?: readonly InstalledLocalSkill[];
}): ReactElement | null {
  const skills = localSkillsForQuery(query, installedSkills);
  if (skills.length === 0 && query.length > 0) return null;
  const selected = Math.min(Math.max(activeIndex, 0), Math.max(skills.length - 1, 0));

  return (
    <div
      className="absolute bottom-full left-2 z-50 mb-2 w-[min(22rem,calc(100%-1rem))] overflow-hidden rounded-2xl border border-border/80 bg-popover p-1.5 text-popover-foreground shadow-xl"
      role="listbox"
      aria-label="Skills and slash commands"
      data-local-skills-menu="true"
    >
      <div className="px-2.5 py-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
        Skills · type to filter
      </div>
      {skills.map((skill, index) => {
        const Icon =
          SKILL_ICONS[skill.command as keyof typeof SKILL_ICONS] ?? Sparkles;
        return (
          <button
            key={skill.command}
            type="button"
            role="option"
            aria-selected={index === selected}
            className="flex w-full items-start gap-2 rounded-xl px-2.5 py-2 text-left text-sm outline-none transition-colors hover:bg-accent aria-selected:bg-accent"
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => onSelect(skill.command)}
          >
            <Icon className="mt-0.5 size-4 shrink-0 text-primary" strokeWidth={1.8} />
            <span className="min-w-0">
              <span className="block font-medium">{skill.command} · {skill.label}</span>
              <span className="block text-xs text-muted-foreground">{skill.description}</span>
            </span>
          </button>
        );
      })}
      {query.length === 0 ? (
        <button
          type="button"
          role="option"
          aria-selected={false}
          className="mt-1 flex w-full items-start gap-2 rounded-xl border-t border-border/60 px-2.5 py-2 text-left text-sm outline-none hover:bg-accent"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            onManage?.();
            openLocalSkillsManager();
          }}
        >
          <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" strokeWidth={1.8} />
          <span>
            <span className="block font-medium">/skills · Manage skills</span>
            <span className="block text-xs text-muted-foreground">Create or install Codex/Claude Code SKILL.md files</span>
          </span>
        </button>
      ) : null}
    </div>
  );
}
