// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useNavigate } from "@tanstack/react-router";

/** Compatibility surface for old /engine links. Operational Helix UI now lives in Chat. */
export function HelixEnginePage() {
  const navigate = useNavigate();
  return (
    <div
      className="mx-auto flex h-full max-w-xl flex-col items-start justify-center gap-3 p-6"
      data-testid="helix-engine-compatibility-page"
    >
      <h1 className="font-heading text-xl font-semibold">Helix Engine moved into Chat</h1>
      <p className="text-sm text-muted-foreground">
        Use the Workflow and Execution Graph controls in the chat header. They open beside the
        active conversation so telemetry, provenance, and analysis stay scoped to the task you are
        debugging.
      </p>
      <Button onClick={() => navigate({ to: "/chat" })}>Open Chat</Button>
    </div>
  );
}
