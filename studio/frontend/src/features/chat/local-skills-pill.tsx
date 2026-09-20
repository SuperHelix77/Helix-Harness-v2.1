// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Brain, Check, ChevronDown, ClipboardList, Gauge, GitBranch, Sparkles, Target } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

import { LocalSkillsManagerDialog } from "./components/local-skills-manager-dialog";
import { LOCAL_SKILLS, openLocalSkillsManager } from "./lib/local-skills";
import { openHermesLearningManager } from "./lib/hermes-learning";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

const SKILL_ICONS = {
  "/github": GitBranch,
  "/plan": ClipboardList,
  "/goal": Target,
  "/speed": Gauge,
  "/learn": Sparkles,
  "/reflect": Brain,
} as const;

export function LocalSkillsPill({
  side = "top",
  onSelectCommand,
}: {
  side?: "top" | "bottom";
  onSelectCommand: (command: string) => void;
}) {
  const chatMode = useChatRuntimeStore((state) => state.chatMode);

  return (
    <>
      <LocalSkillsManagerDialog />
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className="composer-pill-btn"
            data-pill-label="Skills"
            aria-label="Local skills and slash commands"
          >
            <Sparkles className="size-[15px]" strokeWidth={1.8} />
            <span>Skills</span>
            <ChevronDown className="composer-pill-caret size-[15px]" strokeWidth={1.5} />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align="start"
          sideOffset={0}
          className="unsloth-plus-menu w-[300px]"
        >
          <DropdownMenuLabel>Local skills</DropdownMenuLabel>
          {LOCAL_SKILLS.map((skill) => {
            const Icon = SKILL_ICONS[skill.command];
            const active = skill.mode === chatMode;
            return (
              <DropdownMenuItem
                key={skill.command}
                onSelect={() => onSelectCommand(skill.command)}
                className={cn("items-start gap-2", active && "text-primary font-medium")}
              >
                <Icon className="mt-0.5 size-4 shrink-0" strokeWidth={1.8} />
                <span className="min-w-0">
                  <span className="block">{skill.label}</span>
                  <span className="text-muted-foreground block text-xs font-normal">
                    {skill.description}
                  </span>
                </span>
                {active ? <Check className="ml-auto mt-0.5 size-4 shrink-0" /> : null}
              </DropdownMenuItem>
            );
          })}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => openHermesLearningManager()}>
            <Sparkles className="size-4 shrink-0" strokeWidth={1.8} />
            <span>
              <span className="block">Hermes self-improvement</span>
              <span className="text-muted-foreground block text-xs font-normal">
                Review, approve, and reuse durable lessons
              </span>
            </span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openLocalSkillsManager()}>
            <Sparkles className="size-4 shrink-0" strokeWidth={1.8} />
            <span>
              <span className="block">Manage skills</span>
              <span className="text-muted-foreground block text-xs font-normal">
                Install or create Codex/Claude Code SKILL.md files
              </span>
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}
