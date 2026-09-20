// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Session = {
  AUTH_TOKEN_KEY: string;
  AUTH_REFRESH_TOKEN_KEY: string;
  getAuthToken(): string | null;
  hasRefreshToken(): boolean;
  storeDesktopAccessToken(access: string, managed?: boolean): void;
  storeAuthTokens(access: string, refresh: string): void;
  clearAuthTokens(): void;
};

test("Tauri bearer and refresh credentials never persist in localStorage", (t) => {
  const originalWindow = globalThis.window;
  const originalLocalStorage = globalThis.localStorage;
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
    key: () => null,
    get length() {
      return values.size;
    },
  } as Storage;
  Object.assign(globalThis, {
    localStorage: storage,
    window: {
      localStorage: storage,
      addEventListener: () => {},
      dispatchEvent: () => true,
    },
  });
  t.after(() => {
    Object.assign(globalThis, {
      window: originalWindow,
      localStorage: originalLocalStorage,
    });
  });

  const session = loadWithStubs<Session>(
    new URL("../src/features/auth/session.ts", import.meta.url),
    {
      "@/lib/api-base": { isTauri: true },
      "@/lib/account-transition": { installAccountTransitionListener: () => {} },
      "./login-client": { getLoginMode: () => "single" },
      "./session-events": {
        AUTH_SESSION_CLEARED_EVENT: "auth-cleared",
        AUTH_SESSION_STORED_EVENT: "auth-stored",
      },
    },
  );

  values.set(session.AUTH_TOKEN_KEY, "legacy-access");
  values.set(session.AUTH_REFRESH_TOKEN_KEY, "legacy-refresh");
  session.storeDesktopAccessToken("memory-access", true);
  assert.equal(session.getAuthToken(), "memory-access");
  assert.equal(session.hasRefreshToken(), true);
  assert.equal(values.has(session.AUTH_TOKEN_KEY), false);
  assert.equal(values.has(session.AUTH_REFRESH_TOKEN_KEY), false);

  // Even the browser-oriented compatibility helper must not write credentials
  // when the module is running inside Tauri.
  session.storeAuthTokens("second-access", "must-not-persist");
  assert.equal(session.getAuthToken(), "second-access");
  assert.equal(values.has(session.AUTH_TOKEN_KEY), false);
  assert.equal(values.has(session.AUTH_REFRESH_TOKEN_KEY), false);
});
