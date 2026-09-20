// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Check, ChevronDown, ClipboardList, MessageCircle, Target } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

import { CHAT_MODE_OPTIONS, type ChatMode } from "./lib/chat-mode";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

const MODE_ICONS = {
  normal: MessageCircle,
  plan: ClipboardList,
  goal: Target,
} as const;

export function ChatModePill({ side = "bottom" }: { side?: "top" | "bottom" }) {
  const chatMode = useChatRuntimeStore((state) => state.chatMode);
  const setChatMode = useChatRuntimeStore((state) => state.setChatMode);
  const selected =
    CHAT_MODE_OPTIONS.find((option) => option.value === chatMode) ??
    CHAT_MODE_OPTIONS[0];
  const Icon = MODE_ICONS[selected.value];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          className={cn("composer-pill-btn", chatMode !== "normal" && "text-primary")}
          data-pill-label="Mode"
          data-active={chatMode !== "normal" ? "true" : "false"}
          aria-label={`Chat mode: ${selected.label}`}
        >
          <Icon className="size-[15px]" strokeWidth={1.8} />
          <span>{selected.label}</span>
          <ChevronDown className="composer-pill-caret size-[15px]" strokeWidth={1.5} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align="start"
        sideOffset={0}
        className="unsloth-plus-menu w-[250px]"
      >
        <DropdownMenuLabel>Chat mode</DropdownMenuLabel>
        {CHAT_MODE_OPTIONS.map((option) => {
          const OptionIcon = MODE_ICONS[option.value];
          return (
            <DropdownMenuItem
              key={option.value}
              onSelect={() => setChatMode(option.value as ChatMode)}
              className={cn(
                "items-start gap-2",
                option.value === chatMode && "text-primary font-medium",
              )}
            >
              <OptionIcon className="mt-0.5 size-4 shrink-0" strokeWidth={1.8} />
              <span className="min-w-0">
                <span className="block">{option.label}</span>
                <span className="text-muted-foreground block text-xs font-normal">
                  {option.description}
                </span>
              </span>
              {option.value === chatMode ? (
                <Check className="ml-auto mt-0.5 size-4 shrink-0" strokeWidth={2} />
              ) : null}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
