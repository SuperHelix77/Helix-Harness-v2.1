// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SETTINGS_SEARCH_KEYWORDS,
  createSettingsSearchIndex,
} from "../src/features/settings/settings-search.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const UPDATE_ENTRY = "settings.about.updates";
const SETTINGS_DIALOG = readSrc("features/settings/settings-dialog.tsx");

test("settings search remains available on narrow screens", () => {
  assert.match(
    SETTINGS_DIALOG,
    /className="relative mx-1 mt-3 mb-2 shrink-0"/,
  );
  assert.match(SETTINGS_DIALOG, /max-sm:max-h-\[40dvh\]/);
  assert.match(SETTINGS_DIALOG, /results !== null && "hidden"/);
  assert.doesNotMatch(
    SETTINGS_DIALOG,
    /relative mx-1 mt-3 mb-2 shrink-0 max-sm:hidden/,
  );
});

test("desktop update searches route to General", () => {
  const index = createSettingsSearchIndex({ desktop: true, closeToTray: true });

  assert.ok(index.general.includes(UPDATE_ENTRY));
  assert.ok(!index.about.includes(UPDATE_ENTRY));
});

test("browser update searches keep routing to About", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(!index.general.includes(UPDATE_ENTRY));
  assert.ok(index.about.includes(UPDATE_ENTRY));
});

// The words a user types for this feature are not substrings of any of its
// labels, so without keywords the rows it is named after were unfindable.
test("model memory rows are reachable by the terms the feature is about", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });
  const rows = [
    "settings.resources.modelMemory.title",
    "settings.resources.modelMemory.keepResident",
    "settings.resources.modelMemory.noRamReserve",
  ] as const;

  for (const row of rows) {
    assert.ok(index.resources.includes(row), `${row} is indexed under Resources`);
    assert.equal(
      SETTINGS_SEARCH_KEYWORDS[row],
      "settings.resources.modelMemory.modelMemoryKeywords",
      `${row} has synonyms`,
    );
  }

  for (const term of ["mlock", "vram", "ulimit", "memlock", "pin"]) {
    assert.ok(
      en.settings.resources.modelMemory.modelMemoryKeywords.includes(term),
      `search matches "${term}"`,
    );
  }
});

const DESKTOP_STARTUP_ENTRIES = [
  "settings.general.startup.sectionTitle",
  "settings.general.startup.launchAtLogin",
] as const;
const CLOSE_TO_TRAY_ENTRY = "settings.general.startup.closeToTray";
const CURRENT_DATE_ENTRY = "settings.chat.currentDate.label";
const MAIN_TRANSPARENCY_ENTRY = "settings.appearance.palette.mainTransparency";

test("the current date prompt setting is searchable under Chat", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(index.chat.includes(CURRENT_DATE_ENTRY));
});

test("settings search follows the reduced Settings surface", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(index.appearance.includes(MAIN_TRANSPARENCY_ENTRY));
  assert.ok(!index.general.includes("settings.general.helperLlm.sectionTitle"));
  assert.ok(!index.general.includes("settings.general.helperLlm.preloadOnStartup"));
  assert.ok(!index.about.includes("settings.about.hardware"));

  for (const hiddenAdvancedAppearanceEntry of [
    "settings.appearance.custom.pointerCursors.label",
    "settings.appearance.custom.uiFontSize.label",
    "settings.appearance.custom.codeFontSize.label",
    "settings.appearance.custom.fontSmoothing.label",
  ] as const) {
    assert.ok(!index.appearance.includes(hiddenAdvancedAppearanceEntry));
  }

  assert.ok(index.resources.includes("settings.resources.modelMemory.title"));
  assert.ok(index.debugging.includes("settings.debugging.logSection"));
});

test("desktop startup entries are absent from browser search", () => {
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  for (const entry of DESKTOP_STARTUP_ENTRIES) {
    assert.ok(desktop.general.includes(entry));
    assert.ok(!browser.general.includes(entry));
  }
});

test("the repair row is searchable on the desktop, where it exists", () => {
  // The capability message for a host whose PyTorch cannot use its GPUs sends the user to
  // "Repair installation in Settings", so searching Settings for it has to find it. Only
  // on the desktop: DesktopRepairControl renders nothing in a browser, and an index entry
  // there would scroll to a row that is not on the page.
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(desktop.general.includes("settings.general.repairInstall.label"));
  assert.ok(!browser.general.includes("settings.general.repairInstall.label"));
});

test("close to tray is searchable only on supported desktops", () => {
  const supported = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const mac = createSettingsSearchIndex({ desktop: true, closeToTray: false });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(supported.general.includes(CLOSE_TO_TRAY_ENTRY));
  assert.ok(!mac.general.includes(CLOSE_TO_TRAY_ENTRY));
  assert.ok(!browser.general.includes(CLOSE_TO_TRAY_ENTRY));
});
