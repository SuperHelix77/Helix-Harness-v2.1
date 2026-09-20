import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { readSrc } from "./helpers/kit.ts";

const themeStore = readSrc("features/settings/stores/theme-store.ts");
const controls = readSrc(
	"features/settings/components/appearance-custom-controls.tsx",
);
const css = readSrc("index.css");
const sidebar = readSrc("components/ui/sidebar.tsx");
const dashboard = readSrc("components/layout/dashboard-layout.tsx");
const titlebar = readSrc("components/tauri/window-titlebar.tsx");
const provider = readSrc("app/provider.tsx");
const chatPage = readSrc("features/chat/chat-page.tsx");
const hubPage = readSrc("features/hub/hub-page.tsx");
const studioPage = readSrc("features/studio/studio-page.tsx");
const recipeStudioPage = readSrc(
	"features/recipe-studio/recipe-studio-page.tsx",
);

test("main background transparency is persisted and synchronized separately from glass surfaces", () => {
	assert.match(themeStore, /helix-main-background-transparency/);
	assert.match(themeStore, /helix-sidebar-transparency/);
	assert.match(themeStore, /helix-chat-surface-transparency/);
	assert.match(themeStore, /helix-composer-transparency/);
	assert.match(themeStore, /helix-right-rail-transparency/);
	assert.match(themeStore, /--main-background-alpha/);
	assert.match(
		themeStore,
		/e\.key === GLASS_OPACITY_KEY[\s\S]*e\.key === GLASS_BLUR_KEY[\s\S]*e\.key === MAIN_BACKGROUND_TRANSPARENCY_KEY/,
	);
	assert.match(controls, /glass\.mainBackgroundTransparency/);
	assert.match(controls, /glass\.setMainBackgroundTransparency/);
	assert.match(controls, /MAIN_BACKGROUND_TRANSPARENCY_RANGE/);
	assert.match(
		controls,
		/t\("settings\.appearance\.palette\.advancedMaterials"\)/,
	);
	assert.match(controls, /glass\.setSidebarTransparency/);
	assert.match(controls, /glass\.setChatSurfaceTransparency/);
	assert.match(controls, /glass\.setComposerTransparency/);
	assert.match(controls, /glass\.setRightRailTransparency/);
	assert.doesNotMatch(controls, /if \(palette !== "glass"\) return null/);
	assert.match(controls, /palette === "glass"/);
	assert.match(provider, /glass\.mainBackgroundTransparency/);
});

test("main canvas is painted once while nested app surfaces remain transparent", () => {
	assert.match(
		css,
		/--main-background-surface:\s*color-mix\(\s*in srgb,\s*var\(--background\)\s*var\(--main-background-alpha\),\s*transparent\s*\)/,
	);
	assert.match(
		css,
		/\.app-main-background\s*\{\s*background-color:\s*transparent;\s*\}/,
	);
	assert.doesNotMatch(css, /\.app-main-background\s*\{[^}]*\bopacity\s*:/s);
	assert.doesNotMatch(
		css,
		/html\[data-palette="glass"\][^{]*\.app-main-background[^{]*\{[^}]*backdrop-filter/s,
	);
	assert.match(
		css,
		/html\.tauri:not\(\[data-palette="glass"\]\) body\s*\{\s*background-color:\s*var\(--main-background-surface\);\s*\}/,
	);
	assert.match(
		css,
		/body\s*\{[\s\S]*?background-color:\s*var\(--background\);/,
	);
	assert.match(css, /\.app-main-chrome-background\s*\{/);
	assert.match(chatPage, /app-main-chrome-background/);
	for (const source of [
		sidebar,
		dashboard,
		titlebar,
		provider,
		chatPage,
		hubPage,
		studioPage,
		recipeStudioPage,
	]) {
		assert.match(source, /app-main-background/);
	}
	assert.match(
		css,
		/html\[data-palette="glass"\] body::before\s*\{[^}]*opacity:\s*var\(--main-background-alpha\)/s,
	);
	assert.match(
		css,
		/--sidebar:\s*rgb\(246 248 249 \/ var\(--sidebar-material-alpha\)\)/,
	);
	assert.match(css, /--popover:\s*rgb\(250 251 251 \/ 0\.9\)/);
	assert.match(css, /--composer-material-alpha/);
	assert.match(css, /--right-rail-material-alpha/);
	assert.match(css, /@media \(prefers-reduced-transparency: reduce\)/);
});

test("theme bootstrap applies persisted glass and main alpha before React", () => {
	const source = readFileSync(
		new URL("../public/theme-boot.js", import.meta.url),
		"utf8",
	);
	const stored = new Map<string, string>([
		["theme", "light"],
		["helix-palette", "glass"],
		["helix-glass-opacity", "999"],
		["helix-glass-blur", "-9"],
		["helix-main-background-transparency", "37.6"],
		["helix-sidebar-transparency", "999"],
		["helix-chat-surface-transparency", "51"],
		["helix-composer-transparency", "7"],
		["helix-right-rail-transparency", "-3"],
	]);
	const vars = new Map<string, string>();
	const attrs = new Map<string, string>();
	const classes = new Map<string, boolean>();
	const style = {
		colorScheme: "",
		setProperty(name: string, value: string) {
			vars.set(name, value);
		},
	};
	const root = {
		style,
		classList: {
			toggle(name: string, value: boolean) {
				classes.set(name, value);
			},
		},
		setAttribute(name: string, value: string) {
			attrs.set(name, value);
		},
	};

	vm.runInNewContext(source, {
		document: { documentElement: root },
		localStorage: { getItem: (key: string) => stored.get(key) ?? null },
		matchMedia: () => ({ matches: false }),
	});

	assert.equal(vars.get("--glass-alpha"), "88%");
	assert.equal(vars.get("--glass-blur"), "0px");
	assert.equal(vars.get("--main-background-alpha"), "62%");
	assert.equal(vars.get("--sidebar-material-alpha"), "12%");
	assert.equal(vars.get("--chat-material-alpha"), "49%");
	assert.equal(vars.get("--composer-material-alpha"), "93%");
	assert.equal(vars.get("--right-rail-material-alpha"), "100%");
	assert.equal(style.colorScheme, "light");
	assert.equal(classes.get("light"), true);
	assert.equal(attrs.get("data-palette"), "glass");
});
