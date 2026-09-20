// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSyncExternalStore } from "react";

import { syncInkContrast } from "./contrast-ink";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";
export type Palette = "standard" | "classic" | "minimal" | "glass";

const STORAGE_KEY = "theme";
const PALETTE_STORAGE_KEY = "helix-palette";
const GLASS_OPACITY_KEY = "helix-glass-opacity";
const GLASS_BLUR_KEY = "helix-glass-blur";
const MAIN_BACKGROUND_TRANSPARENCY_KEY = "helix-main-background-transparency";
const SIDEBAR_TRANSPARENCY_KEY = "helix-sidebar-transparency";
const CHAT_SURFACE_TRANSPARENCY_KEY = "helix-chat-surface-transparency";
const COMPOSER_TRANSPARENCY_KEY = "helix-composer-transparency";
const RIGHT_RAIL_TRANSPARENCY_KEY = "helix-right-rail-transparency";

export const PALETTES: readonly Palette[] = [
	"standard",
	"classic",
	"minimal",
	"glass",
];

export const GLASS_OPACITY_RANGE = { min: 8, max: 88, default: 36 } as const;
export const GLASS_BLUR_RANGE = { min: 0, max: 48, default: 28 } as const;
export const MAIN_BACKGROUND_TRANSPARENCY_RANGE = {
	min: 0,
	max: 100,
	default: 34,
} as const;
export const SIDEBAR_TRANSPARENCY_RANGE = {
	min: 0,
	max: 88,
	default: 46,
} as const;
export const CHAT_SURFACE_TRANSPARENCY_RANGE = {
	min: 0,
	max: 92,
	default: 72,
} as const;
export const COMPOSER_TRANSPARENCY_RANGE = {
	min: 0,
	max: 72,
	default: 18,
} as const;
export const RIGHT_RAIL_TRANSPARENCY_RANGE = {
	min: 0,
	max: 88,
	default: 38,
} as const;

export function isPalette(value: unknown): value is Palette {
	return (
		value === "standard" ||
		value === "classic" ||
		value === "minimal" ||
		value === "glass"
	);
}

// Persist a re-derived literal from a fixed allow-list rather than the argument,
// so a value arriving via the authenticated personalization sync is not tracked
// as sensitive data flowing into storage (these are plain UI preferences).
const STORED_THEME: Record<Theme, Theme> = {
	light: "light",
	dark: "dark",
	system: "system",
};
const STORED_PALETTE: Record<Palette, Palette> = {
	standard: "standard",
	classic: "classic",
	minimal: "minimal",
	glass: "glass",
};

function readStoredTheme(): Theme {
	if (typeof window === "undefined") return "system";
	let stored: string | null = null;
	try {
		stored = window.localStorage.getItem(STORAGE_KEY);
	} catch {
		return "system";
	}
	if (stored === "light" || stored === "dark" || stored === "system")
		return stored;
	return "system";
}

function readStoredPalette(): Palette {
	if (typeof window === "undefined") return "glass";
	let stored: string | null = null;
	try {
		stored = window.localStorage.getItem(PALETTE_STORAGE_KEY);
	} catch {
		return "glass";
	}
	return isPalette(stored) ? stored : "glass";
}

function readStoredNumber(
	key: string,
	fallback: number,
	min: number,
	max: number,
): number {
	if (typeof window === "undefined") return fallback;
	try {
		const raw = window.localStorage.getItem(key);
		const value = raw == null ? fallback : Number(raw);
		if (!Number.isFinite(value)) return fallback;
		return Math.min(max, Math.max(min, Math.round(value)));
	} catch {
		return fallback;
	}
}

// In-memory source of truth so a selected value survives even when localStorage is blocked (private
// browsing). Without it the snapshots would re-read empty storage and revert React state to the
// default while the DOM already changed.
let currentTheme: Theme = readStoredTheme();
let currentPalette: Palette = readStoredPalette();
let currentGlassOpacity = readStoredNumber(
	GLASS_OPACITY_KEY,
	GLASS_OPACITY_RANGE.default,
	GLASS_OPACITY_RANGE.min,
	GLASS_OPACITY_RANGE.max,
);
let currentGlassBlur = readStoredNumber(
	GLASS_BLUR_KEY,
	GLASS_BLUR_RANGE.default,
	GLASS_BLUR_RANGE.min,
	GLASS_BLUR_RANGE.max,
);

let currentMainBackgroundTransparency = readStoredNumber(
	MAIN_BACKGROUND_TRANSPARENCY_KEY,
	MAIN_BACKGROUND_TRANSPARENCY_RANGE.default,
	MAIN_BACKGROUND_TRANSPARENCY_RANGE.min,
	MAIN_BACKGROUND_TRANSPARENCY_RANGE.max,
);
let currentSidebarTransparency = readStoredNumber(
	SIDEBAR_TRANSPARENCY_KEY,
	SIDEBAR_TRANSPARENCY_RANGE.default,
	SIDEBAR_TRANSPARENCY_RANGE.min,
	SIDEBAR_TRANSPARENCY_RANGE.max,
);
let currentChatSurfaceTransparency = readStoredNumber(
	CHAT_SURFACE_TRANSPARENCY_KEY,
	CHAT_SURFACE_TRANSPARENCY_RANGE.default,
	CHAT_SURFACE_TRANSPARENCY_RANGE.min,
	CHAT_SURFACE_TRANSPARENCY_RANGE.max,
);
let currentComposerTransparency = readStoredNumber(
	COMPOSER_TRANSPARENCY_KEY,
	COMPOSER_TRANSPARENCY_RANGE.default,
	COMPOSER_TRANSPARENCY_RANGE.min,
	COMPOSER_TRANSPARENCY_RANGE.max,
);
let currentRightRailTransparency = readStoredNumber(
	RIGHT_RAIL_TRANSPARENCY_KEY,
	RIGHT_RAIL_TRANSPARENCY_RANGE.default,
	RIGHT_RAIL_TRANSPARENCY_RANGE.min,
	RIGHT_RAIL_TRANSPARENCY_RANGE.max,
);

function systemPrefersDark(): boolean {
	if (typeof window === "undefined") return false;
	return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolveTheme(theme: Theme): ResolvedTheme {
	if (theme === "system") return systemPrefersDark() ? "dark" : "light";
	return theme;
}

function applyToDocument(resolved: ResolvedTheme) {
	if (typeof document === "undefined") return;
	const el = document.documentElement;
	el.classList.toggle("dark", resolved === "dark");
	el.classList.toggle("light", resolved === "light");
	// Native controls (scrollbars, spinners, pickers) follow the app mode.
	el.style.colorScheme = resolved;
}

function applyPaletteToDocument(palette: Palette) {
	if (typeof document === "undefined") return;
	const el = document.documentElement;
	// Standard is the base :root/.dark palette; no attribute keeps the DOM
	// (and CSS selectors) simple for the default look.
	if (palette === "standard") {
		el.removeAttribute("data-palette");
	} else {
		el.setAttribute("data-palette", palette);
	}
	el.toggleAttribute("data-glass", palette === "glass");
	applyGlassToDocument();
	applyMainBackgroundToDocument();
	queueMicrotask(() => syncInkContrast());
}

function applyGlassToDocument() {
	if (typeof document === "undefined") return;
	const el = document.documentElement;
	el.style.setProperty("--glass-alpha", `${currentGlassOpacity}%`);
	el.style.setProperty("--glass-blur", `${currentGlassBlur}px`);
	el.style.setProperty(
		"--sidebar-material-alpha",
		`${100 - currentSidebarTransparency}%`,
	);
	el.style.setProperty(
		"--chat-material-alpha",
		`${100 - currentChatSurfaceTransparency}%`,
	);
	el.style.setProperty(
		"--composer-material-alpha",
		`${100 - currentComposerTransparency}%`,
	);
	el.style.setProperty(
		"--right-rail-material-alpha",
		`${100 - currentRightRailTransparency}%`,
	);
}

function applyMainBackgroundToDocument() {
	if (typeof document === "undefined") return;
	document.documentElement.style.setProperty(
		"--main-background-alpha",
		`${100 - currentMainBackgroundTransparency}%`,
	);
}

const listeners = new Set<() => void>();
function subscribe(cb: () => void) {
	listeners.add(cb);
	if (typeof window === "undefined") {
		return () => listeners.delete(cb);
	}
	const mq = window.matchMedia("(prefers-color-scheme: dark)");
	// OS scheme flip: only the resolved value changes; keep the in-memory choice
	// (re-reading storage here would clobber it when storage is blocked).
	const onSchemeChange = () => {
		applyToDocument(resolveTheme(currentTheme));
		cb();
	};
	// Another tab wrote storage (only fires when storage is available): adopt it.
	const onStorage = (e: StorageEvent) => {
		if (
			e.key === STORAGE_KEY ||
			e.key === PALETTE_STORAGE_KEY ||
			e.key === GLASS_OPACITY_KEY ||
			e.key === GLASS_BLUR_KEY ||
			e.key === MAIN_BACKGROUND_TRANSPARENCY_KEY ||
			e.key === SIDEBAR_TRANSPARENCY_KEY ||
			e.key === CHAT_SURFACE_TRANSPARENCY_KEY ||
			e.key === COMPOSER_TRANSPARENCY_KEY ||
			e.key === RIGHT_RAIL_TRANSPARENCY_KEY ||
			e.key === null
		) {
			currentTheme = readStoredTheme();
			currentPalette = readStoredPalette();
			currentGlassOpacity = readStoredNumber(
				GLASS_OPACITY_KEY,
				GLASS_OPACITY_RANGE.default,
				GLASS_OPACITY_RANGE.min,
				GLASS_OPACITY_RANGE.max,
			);
			currentGlassBlur = readStoredNumber(
				GLASS_BLUR_KEY,
				GLASS_BLUR_RANGE.default,
				GLASS_BLUR_RANGE.min,
				GLASS_BLUR_RANGE.max,
			);
			currentMainBackgroundTransparency = readStoredNumber(
				MAIN_BACKGROUND_TRANSPARENCY_KEY,
				MAIN_BACKGROUND_TRANSPARENCY_RANGE.default,
				MAIN_BACKGROUND_TRANSPARENCY_RANGE.min,
				MAIN_BACKGROUND_TRANSPARENCY_RANGE.max,
			);
			currentSidebarTransparency = readStoredNumber(
				SIDEBAR_TRANSPARENCY_KEY,
				SIDEBAR_TRANSPARENCY_RANGE.default,
				SIDEBAR_TRANSPARENCY_RANGE.min,
				SIDEBAR_TRANSPARENCY_RANGE.max,
			);
			currentChatSurfaceTransparency = readStoredNumber(
				CHAT_SURFACE_TRANSPARENCY_KEY,
				CHAT_SURFACE_TRANSPARENCY_RANGE.default,
				CHAT_SURFACE_TRANSPARENCY_RANGE.min,
				CHAT_SURFACE_TRANSPARENCY_RANGE.max,
			);
			currentComposerTransparency = readStoredNumber(
				COMPOSER_TRANSPARENCY_KEY,
				COMPOSER_TRANSPARENCY_RANGE.default,
				COMPOSER_TRANSPARENCY_RANGE.min,
				COMPOSER_TRANSPARENCY_RANGE.max,
			);
			currentRightRailTransparency = readStoredNumber(
				RIGHT_RAIL_TRANSPARENCY_KEY,
				RIGHT_RAIL_TRANSPARENCY_RANGE.default,
				RIGHT_RAIL_TRANSPARENCY_RANGE.min,
				RIGHT_RAIL_TRANSPARENCY_RANGE.max,
			);
			applyToDocument(resolveTheme(currentTheme));
			applyPaletteToDocument(currentPalette);
			cb();
		}
	};
	// Apply on mount so this store is the single source of truth for the DOM
	// class after the index.html bootstrap script painted the first frame.
	applyToDocument(resolveTheme(currentTheme));
	applyPaletteToDocument(currentPalette);
	mq.addEventListener("change", onSchemeChange);
	window.addEventListener("storage", onStorage);
	return () => {
		listeners.delete(cb);
		mq.removeEventListener("change", onSchemeChange);
		window.removeEventListener("storage", onStorage);
	};
}

function getSnapshot(): Theme {
	return currentTheme;
}

function getServerSnapshot(): Theme {
	return "system";
}

// Snapshot the RESOLVED mode too: under "system" the theme string never
// changes when the OS scheme flips, so consumers keyed on `resolved`
// (customization applier, mode-scoped settings) would not re-render.
function getResolvedSnapshot(): ResolvedTheme {
	return resolveTheme(currentTheme);
}

function getResolvedServerSnapshot(): ResolvedTheme {
	return "light";
}

/**
 * Single source of truth for setting the theme. All writers (Settings dialog
 * control, sidebar dropdown toggler) route through this so the DOM class,
 * localStorage, and React subscribers stay in sync.
 */
export function setTheme(next: Theme): void {
	if (typeof window === "undefined") return;
	currentTheme = next;
	// Persist "system" explicitly so a reload keeps following the OS.
	try {
		window.localStorage.setItem(STORAGE_KEY, STORED_THEME[next]);
	} catch {
		// ignore storage failures
	}
	applyToDocument(resolveTheme(next));
	listeners.forEach((cb) => cb());
}

export function useTheme(): {
	theme: Theme;
	resolved: ResolvedTheme;
	setTheme: (next: Theme) => void;
} {
	const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
	const resolved = useSyncExternalStore(
		subscribe,
		getResolvedSnapshot,
		getResolvedServerSnapshot,
	);
	return { theme, resolved, setTheme };
}

function getPaletteSnapshot(): Palette {
	return currentPalette;
}

function getPaletteServerSnapshot(): Palette {
	return "glass";
}

/**
 * Single source of truth for setting the color palette; mirrors setTheme so
 * the data-palette attribute, localStorage, and React subscribers stay in
 * sync.
 */
export function setPalette(next: Palette): void {
	if (typeof window === "undefined") return;
	currentPalette = next;
	try {
		window.localStorage.setItem(PALETTE_STORAGE_KEY, STORED_PALETTE[next]);
	} catch {
		// ignore storage failures
	}
	applyPaletteToDocument(next);
	listeners.forEach((cb) => cb());
}

export function usePalette(): {
	palette: Palette;
	setPalette: (next: Palette) => void;
} {
	const palette = useSyncExternalStore(
		subscribe,
		getPaletteSnapshot,
		getPaletteServerSnapshot,
	);
	return { palette, setPalette };
}

function persistNumber(key: string, value: number) {
	try {
		window.localStorage.setItem(key, String(value));
	} catch {
		// ignore storage failures
	}
}

export function setGlassOpacity(next: number): void {
	if (typeof window === "undefined") return;
	currentGlassOpacity = Math.min(
		GLASS_OPACITY_RANGE.max,
		Math.max(GLASS_OPACITY_RANGE.min, Math.round(next)),
	);
	persistNumber(GLASS_OPACITY_KEY, currentGlassOpacity);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function setGlassBlur(next: number): void {
	if (typeof window === "undefined") return;
	currentGlassBlur = Math.min(
		GLASS_BLUR_RANGE.max,
		Math.max(GLASS_BLUR_RANGE.min, Math.round(next)),
	);
	persistNumber(GLASS_BLUR_KEY, currentGlassBlur);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function setMainBackgroundTransparency(next: number): void {
	if (typeof window === "undefined") return;
	currentMainBackgroundTransparency = Math.min(
		MAIN_BACKGROUND_TRANSPARENCY_RANGE.max,
		Math.max(MAIN_BACKGROUND_TRANSPARENCY_RANGE.min, Math.round(next)),
	);
	persistNumber(
		MAIN_BACKGROUND_TRANSPARENCY_KEY,
		currentMainBackgroundTransparency,
	);
	applyMainBackgroundToDocument();
	listeners.forEach((cb) => cb());
}

function clampTransparency(
	next: number,
	range: { min: number; max: number },
): number {
	return Math.min(range.max, Math.max(range.min, Math.round(next)));
}

export function setSidebarTransparency(next: number): void {
	if (typeof window === "undefined") return;
	currentSidebarTransparency = clampTransparency(
		next,
		SIDEBAR_TRANSPARENCY_RANGE,
	);
	persistNumber(SIDEBAR_TRANSPARENCY_KEY, currentSidebarTransparency);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function setChatSurfaceTransparency(next: number): void {
	if (typeof window === "undefined") return;
	currentChatSurfaceTransparency = clampTransparency(
		next,
		CHAT_SURFACE_TRANSPARENCY_RANGE,
	);
	persistNumber(CHAT_SURFACE_TRANSPARENCY_KEY, currentChatSurfaceTransparency);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function setComposerTransparency(next: number): void {
	if (typeof window === "undefined") return;
	currentComposerTransparency = clampTransparency(
		next,
		COMPOSER_TRANSPARENCY_RANGE,
	);
	persistNumber(COMPOSER_TRANSPARENCY_KEY, currentComposerTransparency);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function setRightRailTransparency(next: number): void {
	if (typeof window === "undefined") return;
	currentRightRailTransparency = clampTransparency(
		next,
		RIGHT_RAIL_TRANSPARENCY_RANGE,
	);
	persistNumber(RIGHT_RAIL_TRANSPARENCY_KEY, currentRightRailTransparency);
	applyGlassToDocument();
	listeners.forEach((cb) => cb());
}

export function useGlass(): {
	opacity: number;
	blur: number;
	mainBackgroundTransparency: number;
	sidebarTransparency: number;
	chatSurfaceTransparency: number;
	composerTransparency: number;
	rightRailTransparency: number;
	setOpacity: (next: number) => void;
	setBlur: (next: number) => void;
	setMainBackgroundTransparency: (next: number) => void;
	setSidebarTransparency: (next: number) => void;
	setChatSurfaceTransparency: (next: number) => void;
	setComposerTransparency: (next: number) => void;
	setRightRailTransparency: (next: number) => void;
} {
	const opacity = useSyncExternalStore(
		subscribe,
		() => currentGlassOpacity,
		() => GLASS_OPACITY_RANGE.default,
	);
	const blur = useSyncExternalStore(
		subscribe,
		() => currentGlassBlur,
		() => GLASS_BLUR_RANGE.default,
	);
	const mainBackgroundTransparency = useSyncExternalStore(
		subscribe,
		() => currentMainBackgroundTransparency,
		() => MAIN_BACKGROUND_TRANSPARENCY_RANGE.default,
	);
	const sidebarTransparency = useSyncExternalStore(
		subscribe,
		() => currentSidebarTransparency,
		() => SIDEBAR_TRANSPARENCY_RANGE.default,
	);
	const chatSurfaceTransparency = useSyncExternalStore(
		subscribe,
		() => currentChatSurfaceTransparency,
		() => CHAT_SURFACE_TRANSPARENCY_RANGE.default,
	);
	const composerTransparency = useSyncExternalStore(
		subscribe,
		() => currentComposerTransparency,
		() => COMPOSER_TRANSPARENCY_RANGE.default,
	);
	const rightRailTransparency = useSyncExternalStore(
		subscribe,
		() => currentRightRailTransparency,
		() => RIGHT_RAIL_TRANSPARENCY_RANGE.default,
	);
	return {
		opacity,
		blur,
		mainBackgroundTransparency,
		sidebarTransparency,
		chatSurfaceTransparency,
		composerTransparency,
		rightRailTransparency,
		setOpacity: setGlassOpacity,
		setBlur: setGlassBlur,
		setMainBackgroundTransparency,
		setSidebarTransparency,
		setChatSurfaceTransparency,
		setComposerTransparency,
		setRightRailTransparency,
	};
}
