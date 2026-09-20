// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function readStoredNumber(key, fallback, min, max) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
  } catch {
    return fallback;
  }
}

try {
  let theme = "system";
  let palette = null;
  try {
    theme = localStorage.getItem("theme") || "system";
    palette = localStorage.getItem("helix-palette") || "glass";
  } catch {}
  const dark =
    theme === "dark" ||
    (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", !dark);
  root.style.colorScheme = dark ? "dark" : "light";

  const preferences = [
    ["glass-opacity", "glass-alpha", 36, 8, 88, "%", false],
    ["glass-blur", "glass-blur", 28, 0, 48, "px", false],
    ["main-background-transparency", "main-background-alpha", 34, 0, 100, "%", true],
    ["sidebar-transparency", "sidebar-material-alpha", 46, 0, 88, "%", true],
    ["chat-surface-transparency", "chat-material-alpha", 72, 0, 92, "%", true],
    ["composer-transparency", "composer-material-alpha", 18, 0, 72, "%", true],
    ["right-rail-transparency", "right-rail-material-alpha", 38, 0, 88, "%", true],
  ];
  for (const [key, property, fallback, min, max, unit, invert] of preferences) {
    const value = readStoredNumber(`helix-${key}`, fallback, min, max);
    root.style.setProperty(`--${property}`, `${invert ? 100 - value : value}${unit}`);
  }

  if (["classic", "minimal", "glass"].includes(palette)) {
    root.setAttribute("data-palette", palette);
  }
  if (palette === "glass") root.setAttribute("data-glass", "");
  root.setAttribute("data-ink", dark ? "light" : "dark");
} catch {}
