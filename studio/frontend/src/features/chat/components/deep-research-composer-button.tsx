// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Telescope02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { XIcon } from "lucide-react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/** Lightweight composer pill. The website-policy dialog lives in a separate lazy
 * module so its forms/dialog primitives do not sit on Chat's first-paint path. */
export function DeepResearchComposerButton({
  onConfigure,
}: {
  onConfigure: () => void;
}) {
  const enabled = useChatRuntimeStore((state) => state.deepResearchEnabled);
  const setEnabled = useChatRuntimeStore(
    (state) => state.setDeepResearchEnabled,
  );

  if (!enabled) return null;

  return (
    <button
      type="button"
      onClick={onConfigure}
      className="composer-pill-btn"
      data-pill-label="Deep research"
      data-active="true"
      aria-label="Configure Deep Research website access"
      title="Configure website access"
    >
      <span
        role="button"
        aria-label="Disable deep research"
        tabIndex={-1}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => {
          event.stopPropagation();
          setEnabled(false);
        }}
        className="composer-pill-glyph cursor-pointer"
      >
        <HugeiconsIcon icon={Telescope02Icon} className="size-[15px]" />
        <XIcon className="composer-pill-x" />
      </span>
      <span>Deep research</span>
      <HugeiconsIcon
        icon={ChevronDownStandardIcon}
        strokeWidth={1.5}
        className="composer-pill-caret size-[15px] text-primary/70"
      />
    </button>
  );
}
