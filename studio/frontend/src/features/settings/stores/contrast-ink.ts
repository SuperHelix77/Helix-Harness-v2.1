// SPDX-License-Identifier: AGPL-3.0-only
/** Flip ink (text/logos) to white on dark surfaces, dark on light surfaces. */

const LIGHT_INK = "#f7f5ff";
const DARK_INK = "#16141c";
const LUMINANCE_CUTOFF = 0.42;

function srgbToLinear(channel: number): number {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function relativeLuminance(r: number, g: number, b: number): number {
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
}

export function parseCssRgb(value: string): [number, number, number] | null {
  const hex = value.trim();
  const short = /^#([0-9a-f]{3})$/i.exec(hex);
  if (short) {
    const n = short[1];
    return [
      Number.parseInt(n[0] + n[0], 16),
      Number.parseInt(n[1] + n[1], 16),
      Number.parseInt(n[2] + n[2], 16),
    ];
  }
  const long = /^#([0-9a-f]{6})$/i.exec(hex);
  if (long) {
    const n = long[1];
    return [
      Number.parseInt(n.slice(0, 2), 16),
      Number.parseInt(n.slice(2, 4), 16),
      Number.parseInt(n.slice(4, 6), 16),
    ];
  }
  const rgb = /^rgba?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*[,\s]\s*([\d.]+)/i.exec(value);
  if (!rgb) return null;
  return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
}

function resolveBackgroundRgb(): [number, number, number] | null {
  if (typeof document === "undefined") return null;
  const probe = document.createElement("div");
  probe.style.cssText =
    "position:absolute;left:-9999px;top:0;width:8px;height:8px;pointer-events:none;";
  probe.style.backgroundColor = getComputedStyle(document.documentElement)
    .getPropertyValue("--background")
    .trim();
  document.documentElement.appendChild(probe);
  const painted = getComputedStyle(probe).backgroundColor;
  probe.remove();
  return parseCssRgb(painted);
}

export function syncInkContrast(customForeground?: string | null): boolean {
  if (typeof document === "undefined") return false;
  const rgb = resolveBackgroundRgb();
  const darkSurface = rgb ? relativeLuminance(...rgb) < LUMINANCE_CUTOFF : false;
  const el = document.documentElement;
  el.setAttribute("data-ink", darkSurface ? "light" : "dark");
  el.style.setProperty("--smart-foreground", darkSurface ? LIGHT_INK : DARK_INK);
  if (!customForeground) {
    el.style.setProperty("--foreground", darkSurface ? LIGHT_INK : DARK_INK);
    el.style.setProperty("--card-foreground", darkSurface ? LIGHT_INK : DARK_INK);
    el.style.setProperty("--popover-foreground", darkSurface ? LIGHT_INK : DARK_INK);
  }
  return darkSurface;
}
