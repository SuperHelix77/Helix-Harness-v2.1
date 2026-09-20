// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, useEffect, useState } from "react";

const LazyChatSkillsDialog = lazy(() =>
  import("./chat-skills-dialog").then((module) => ({
    default: module.ChatSkillsDialog,
  })),
);

/** Load the skills-management UI only after first use, then keep it mounted so
 * it preserves the same close/reopen state lifetime as the original dialog. */
export function ChatSkillsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [mounted, setMounted] = useState(open);
  useEffect(() => {
    if (open) setMounted(true);
  }, [open]);
  if (!mounted && !open) return null;
  return (
    <Suspense fallback={null}>
      <LazyChatSkillsDialog open={open} onOpenChange={onOpenChange} />
    </Suspense>
  );
}

