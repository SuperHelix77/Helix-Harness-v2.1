// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig, minify } from "vite";

const CLASSIC_BOOT_SCRIPTS = [
  "crypto-boot.js",
  "theme-boot.js",
  "reload-snapshot.js",
] as const;

/**
 * Public scripts bypass Vite's module transform but block the first render.
 * Keep their reviewed sources readable and compact only the emitted copies.
 */
function minifyClassicBootScripts(): Plugin {
  let outputDirectory = "";
  return {
    name: "minify-classic-boot-scripts",
    apply: "build",
    enforce: "post",
    configResolved(config) {
      outputDirectory = path.resolve(config.root, config.build.outDir);
    },
    async closeBundle() {
      const license = "/* SPDX-License-Identifier: AGPL-3.0-only */\n";
      await Promise.all(
        CLASSIC_BOOT_SCRIPTS.map(async (filename) => {
          const emittedPath = path.join(outputDirectory, filename);
          const source = await readFile(emittedPath, "utf8");
          const result = await minify(filename, source, {
            compress: true,
            mangle: true,
            codegen: { removeWhitespace: true },
          });
          if (result.errors.length > 0) {
            throw new Error(
              `Could not minify ${filename}: ${result.errors.map((error) => error.message).join("; ")}`,
            );
          }
          await writeFile(emittedPath, `${license}${result.code}\n`, "utf8");
        }),
      );
    },
  };
}

function smokeModuleDelay(): Plugin {
  const match = process.env.SMOKE_MODULE_DELAY_MATCH;
  const delayMs = Number(process.env.SMOKE_MODULE_DELAY_MS ?? "0");
  return {
    name: "smoke-module-delay",
    configureServer(server) {
      if (!match || !Number.isFinite(delayMs) || delayMs <= 0) return;
      server.middlewares.use((request, _response, next) => {
        if (!request.url?.includes(match)) {
          next();
          return;
        }
        setTimeout(next, delayMs);
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    smokeModuleDelay(),
    minifyClassicBootScripts(),
  ],
  // Keep an unrelated PostCSS config in an ancestor directory from leaking
  // into Unsloth installs. Tailwind is provided by its dedicated Vite plugin.
  css: {
    postcss: {
      plugins: [],
    },
  },
  optimizeDeps: {
    include: ["@dagrejs/dagre", "@dagrejs/graphlib"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/seed/inspect": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/seed/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/validate": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/tools": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@dagrejs/dagre": path.resolve(
        __dirname,
        "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
      ),
    },
  },
  build: {
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
  },
});
