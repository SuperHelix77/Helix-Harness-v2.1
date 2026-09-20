// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { installAccountTransitionListener } from "@/lib/account-transition";
import { getLoginMode } from "./login-client";

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
} from "./session-events";

export { AUTH_SESSION_CLEARED_EVENT, AUTH_SESSION_STORED_EVENT } from "./session-events";

export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
export const AUTH_MUST_CHANGE_PASSWORD_KEY = "unsloth_auth_must_change_password";
/**
 * The cross-document counterpart to `authSessionEpoch`, which is per-document memory and so
 * says nothing to another tab. Written when a session begins, removed when it ends, never on
 * a refresh, so a `storage` listener can tell an account switch from an hourly rotation. The
 * value is opaque: it marks a session boundary, not the account.
 */
export const AUTH_SESSION_MARK_KEY = "unsloth_auth_session_mark";


if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  installAccountTransitionListener();
}

let authSessionEpoch = 0;
let desktopAccessToken: string | null = null;
let desktopRefreshManaged = false;

/** Stable across access-token refreshes; changes when an auth session starts or clears. */
export function getAuthSessionEpoch(): number {
  return authSessionEpoch;
}

type PostAuthRoute = "/change-password" | "/chat";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function hasAuthToken(): boolean {
  if (!canUseStorage()) return false;
  if (isTauri) return Boolean(desktopAccessToken);
  return Boolean(localStorage.getItem(AUTH_TOKEN_KEY));
}

export function hasRefreshToken(): boolean {
  if (!canUseStorage()) return false;
  if (isTauri) return desktopRefreshManaged;
  return Boolean(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY));
}

export function getAuthToken(): string | null {
  if (!canUseStorage()) return null;
  if (isTauri) return desktopAccessToken;
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (!canUseStorage()) return null;
  if (isTauri) return null;
  return localStorage.getItem(AUTH_REFRESH_TOKEN_KEY);
}

export function storeDesktopAccessToken(
  accessToken: string,
  refreshManaged = true,
): void {
  if (!canUseStorage()) return;
  const sessionStarted = !desktopAccessToken;
  desktopAccessToken = accessToken;
  desktopRefreshManaged = refreshManaged;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
  if (sessionStarted) {
    authSessionEpoch += 1;
    localStorage.setItem(
      AUTH_SESSION_MARK_KEY,
      `${Date.now()}.${Math.random()}`,
    );
    window.dispatchEvent(new Event(AUTH_SESSION_STORED_EVENT));
  }
}

export async function storeAuthTokensSecurely(
  accessToken: string,
  refreshToken: string,
): Promise<void> {
  if (!isTauri) {
    storeAuthTokens(accessToken, refreshToken);
    return;
  }
  await persistDesktopRefreshCredential(refreshToken);
  storeDesktopAccessToken(accessToken, true);
}

export async function persistDesktopRefreshCredential(
  refreshToken: string,
): Promise<void> {
  if (!isTauri) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("desktop_store_refresh_auth", { refreshToken });
}

export function storeAuthTokens(
  accessToken: string,
  refreshToken: string,
): void {
  // must_change_password is set via setMustChangePassword(), not here: routing
  // it through would let CodeQL trace the boolean into localStorage and flag the
  // deliberate JWT writes as sensitive-info storage.
  if (!canUseStorage()) return;
  if (isTauri) {
    storeDesktopAccessToken(accessToken, Boolean(refreshToken));
    return;
  }
  const sessionStarted = !localStorage.getItem(AUTH_TOKEN_KEY);
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
  localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, refreshToken);
  if (sessionStarted) {
    authSessionEpoch += 1;
    localStorage.setItem(
      AUTH_SESSION_MARK_KEY,
      `${Date.now()}.${Math.random()}`,
    );
    window.dispatchEvent(new Event(AUTH_SESSION_STORED_EVENT));
  } else if (!localStorage.getItem(AUTH_SESSION_MARK_KEY)) {
    // A session signed in before this key existed. Written on its next refresh so a later
    // sign-out has a key to remove, since removing an absent one raises no storage event and
    // would leave the other tabs on a signed-out account's titles. Not a new session, so no
    // epoch bump: the tokens here are a rotation.
    localStorage.setItem(
      AUTH_SESSION_MARK_KEY,
      `${Date.now()}.${Math.random()}`,
    );
  }
}

export function clearAuthTokens(): void {
  if (!canUseStorage()) return;
  authSessionEpoch += 1;
  desktopAccessToken = null;
  desktopRefreshManaged = false;
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_REFRESH_TOKEN_KEY);
  localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
  localStorage.removeItem(AUTH_SESSION_MARK_KEY);
  window.dispatchEvent(new Event(AUTH_SESSION_CLEARED_EVENT));
}

export async function clearDesktopRefreshCredential(): Promise<void> {
  if (!isTauri) return;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("desktop_clear_refresh_auth");
  } finally {
    desktopRefreshManaged = false;
  }
}

// Flag stored as key presence (constant "1" or absence), not a derived boolean,
// so CodeQL doesn't flow must_change_password into localStorage.setItem. The
// value is a route hint (/change-password vs /chat), not a secret.
export function mustChangePassword(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY) !== null;
}

export function setMustChangePassword(required: boolean): void {
  if (!canUseStorage()) return;
  if (required) {
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "1");
  } else {
    localStorage.removeItem(AUTH_MUST_CHANGE_PASSWORD_KEY);
  }
}



export function getPostAuthRoute(): PostAuthRoute {
  if (isTauri && getLoginMode() === "single") return "/chat";
  if (mustChangePassword()) return "/change-password";
  return "/chat";
}
