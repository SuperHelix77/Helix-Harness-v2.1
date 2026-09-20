// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import {
  LOCAL_SKILLS,
  localSkillsForQuery,
  localSkillPrompt,
  localSkillsHelp,
  localSlashQuery,
  parseLocalSlashCommand,
} from "../src/features/chat/lib/local-skills.ts";

test("local slash commands select modes and strip the command from the prompt", () => {
  assert.deepEqual(parseLocalSlashCommand("/plan inspect the loader"), {
    command: "/plan",
    args: "inspect the loader",
    message: "inspect the loader",
    mode: "plan",
    handled: false,
  });
  assert.equal(parseLocalSlashCommand("/goal ship the fix")?.mode, "goal");
  assert.match(parseLocalSlashCommand("/github future ledger")?.message ?? "", /github_context/);
});

test("help is handled locally and unknown slash prose is preserved", () => {
  assert.equal(parseLocalSlashCommand("/skills")?.help, true);
  assert.match(localSkillsHelp(), /\/github/);
  assert.equal(parseLocalSlashCommand("/not-a-command"), null);
});

test("the visible menu and parser share the same local commands", () => {
  assert.deepEqual(
    LOCAL_SKILLS.map((skill) => skill.command),
    ["/github", "/plan", "/goal", "/speed", "/learn", "/reflect"],
  );
});

test("the picker opens for slash and slash-space and filters by command", () => {
  assert.equal(localSlashQuery("/"), "");
  assert.equal(localSlashQuery("/ "), "");
  assert.equal(localSlashQuery("  /go"), "go");
  assert.equal(localSlashQuery("/github inspect ledger"), null);
  assert.deepEqual(
    localSkillsForQuery("go").map((skill) => skill.command),
    ["/goal"],
  );
});

test("installed skills appear in the picker and parse into bounded skill prompts", () => {
  const installed = [{
    name: "review-code",
    description: "Review code carefully",
    path: "/tmp/review-code",
    ecosystems: ["codex"] as const,
  }];
  assert.deepEqual(
    localSkillsForQuery("review", installed).map((skill) => skill.command),
    ["/learn", "/review-code"],
  );
  const parsed = parseLocalSlashCommand("/review-code check this", installed);
  assert.equal(parsed?.skillName, "review-code");
  assert.equal(
    localSkillPrompt("review-code", "Be skeptical.", parsed?.args ?? ""),
    "Apply the installed local skill /review-code to this request.\n\nSkill instructions:\nBe skeptical.\n\nUser request:\ncheck this",
  );
});
