import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const general = readSrc("features/settings/tabs/general-tab.tsx");
const appearance = readSrc("features/settings/tabs/appearance-tab.tsx");
const about = readSrc("features/settings/tabs/about-tab.tsx");
const resources = readSrc("features/settings/tabs/resources-tab.tsx");

test("General omits the helper-model pre-cache implementation detail", () => {
  assert.doesNotMatch(general, /helper-precache/);
  assert.doesNotMatch(general, /settings\.general\.helperLlm/);
  assert.doesNotMatch(general, /helperPrecache/);
});

test("About no longer duplicates hardware already covered by Resources", () => {
  assert.doesNotMatch(about, /settings\.about\.hardware/);
  assert.doesNotMatch(about, /acceleratorRuntimes/);
  assert.match(about, /StudioVersionSection/);
  assert.match(resources, /settings\.resources\.gpu\.title/);
  assert.match(resources, /settings\.resources\.environment\.title/);
});

test("low-level appearance knobs are preserved behind one Advanced disclosure", () => {
  assert.match(appearance, /<AdvancedDisclosure/);
  assert.match(appearance, /open=\{advancedAppearanceOpen\}/);
  assert.match(appearance, /<PointerCursorsSwitch \/>/);
  assert.match(appearance, /<UiFontSizeRow \/>/);
  assert.match(appearance, /<CodeFontSizeRow \/>/);
  assert.match(appearance, /<FontSmoothingSwitch \/>/);
  assert.match(appearance, /<ReduceMotionSegmented \/>/);
  assert.match(appearance, /<Switch checked=\{pinned\} onCheckedChange=\{setPinned\} \/>/);
});
