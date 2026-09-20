// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FloatingMonitor } from "@/components/floating-monitor";
import { lazy, Suspense } from "react";
import { useSettingsDialogStore } from "./stores/settings-dialog-store";

const SettingsDialog = lazy(() =>
  import("./settings-dialog").then((module) => ({
    default: module.SettingsDialog,
  })),
);

export function SettingsDialogMount({ active }: { active: boolean }) {
  const open = useSettingsDialogStore((state) => state.open);
  if (!active) return null;
  return (
    <>
      {open ? (
        <Suspense fallback={null}>
          <SettingsDialog />
        </Suspense>
      ) : null}
      <FloatingMonitor />
    </>
  );
}
