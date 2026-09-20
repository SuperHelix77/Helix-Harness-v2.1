// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  engineProjectIdForThread,
  engineSessionIdFor,
  fetchHelixJson,
  loadHelixSnapshot,
} = await import("../src/features/helix-engine/engine-data.ts");

test("engine session scope matches chat sandbox identity", () => {
  assert.equal(engineSessionIdFor("thread-1", null), "thread-1");
  assert.equal(engineSessionIdFor("thread-1", "project-1"), "project-project-1");
  assert.equal(engineSessionIdFor(null, null), "default");

  assert.equal(
    engineProjectIdForThread({ projectId: "saved-project" }, "stale-project"),
    "saved-project",
  );
  assert.equal(
    engineProjectIdForThread({ projectId: null }, "stale-project"),
    null,
    "a persisted non-project thread must not inherit a stale active project",
  );
  assert.equal(engineProjectIdForThread(undefined, "fresh-project"), "fresh-project");
});

test("Helix JSON errors keep the backend detail", async () => {
  await assert.rejects(
    fetchHelixJson(
      async () =>
        new Response(JSON.stringify({ detail: "engine unavailable" }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      "/api/helix-engine/example",
      "Trajectory",
    ),
    /Trajectory: engine unavailable/,
  );
});

test("snapshot polling preserves a successful half and reports the failed half", async () => {
  const urls: string[] = [];
  const snapshot = await loadHelixSnapshot(async (url) => {
    urls.push(url);
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(
        JSON.stringify({ events: [{ kind: "computer", action: "click" }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(JSON.stringify({ detail: "trace temporarily unavailable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }, "project-p 1", "thread/a");

  assert.deepEqual(snapshot.events, [{ kind: "computer", action: "click" }]);
  assert.equal(snapshot.steps, undefined);
  assert.equal(snapshot.successfulRequests, 1);
  assert.deepEqual(snapshot.errors, ["Trajectory: trace temporarily unavailable"]);
  assert.deepEqual(urls, [
    "/api/helix-engine/live-feed?session_id=project-p+1&thread_id=thread%2Fa",
    "/api/helix-engine/session/project-p%201?thread_id=thread%2Fa",
  ]);
});

test("snapshot polling keeps tool-control provenance from the session trace", async () => {
  const snapshot = await loadHelixSnapshot(async (url) => {
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(JSON.stringify({ events: [{ kind: "computer", action: "click", turn_id: "turn-1" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.startsWith("/api/helix-engine/provenance")) {
      return new Response(
        JSON.stringify({
          schema_version: "helix.provenance.v1",
          available: true,
          selection: { turn_id: "turn-1", trajectory_id: "turn-1" },
          adaptive_checkpoints: [
            {
              trajectory_id: "turn-1",
              start_sequence: 10,
              start_created_at_ms: 1_000,
              resume_sequence: 11,
              resume_created_at_ms: 1_100,
            },
          ],
          skill_retention: [
            {
              skill_name: "repo-audit",
              disposition: "promoted",
              reason: "created and read successfully",
              evidence: { read_successfully: true },
            },
          ],
          qlora_outcome: { outcome: "deferred", reason: "training worker busy" },
          final_actions: ["promote_temp_skill:repo-audit"],
          missing_sources: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(
      JSON.stringify({
        turn_id: "turn-1",
        steps: [{ index: 0, name: "search_memory", arguments: "{}", result: "hit", useful_hint: "useful" }],
        control_events: [
          {
            action: "equivalent_duplicate",
            tool_name: "search_memory",
            reason: "same retrieval already completed",
            provenance: "runtime_tool_loop",
          },
        ],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }, "session-1", "thread-1");

  assert.equal(snapshot.events?.[0]?.turn_id, "turn-1");
  assert.equal(snapshot.steps?.[0]?.name, "search_memory");
  assert.equal(snapshot.controlEvents?.[0]?.action, "equivalent_duplicate");
  assert.equal(snapshot.controlEvents?.[0]?.provenance, "runtime_tool_loop");
  assert.equal(snapshot.provenance?.selection?.turn_id, "turn-1");
  assert.equal(snapshot.provenance?.adaptive_checkpoints?.[0]?.start_sequence, 10);
  assert.equal(snapshot.provenance?.adaptive_checkpoints?.[0]?.resume_created_at_ms, 1_100);
  assert.equal(snapshot.provenance?.skill_retention?.[0]?.reason, "created and read successfully");
  assert.equal(snapshot.provenance?.qlora_outcome?.outcome, "deferred");
  assert.deepEqual(snapshot.provenance?.final_actions, ["promote_temp_skill:repo-audit"]);
  assert.equal(snapshot.successfulRequests, 3);
  assert.deepEqual(snapshot.errors, []);
});

test("snapshot excludes feed events explicitly correlated to another turn", async () => {
  const snapshot = await loadHelixSnapshot(async (url) => {
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(
        JSON.stringify({
          events: [
            { kind: "computer", action: "click", turn_id: "turn-old" },
            { kind: "computer", action: "legacy-untagged" },
            { kind: "browse", action: "navigate", turn_id: "turn-current" },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    if (url.startsWith("/api/helix-engine/provenance")) {
      return new Response(JSON.stringify({ available: false, missing_sources: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(
      JSON.stringify({ turn_id: "turn-current", steps: [], control_events: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }, "session-1", "thread-1");

  assert.deepEqual(snapshot.events?.map((event) => event.action), [
    "legacy-untagged",
    "navigate",
  ]);
  assert.equal(snapshot.turnId, "turn-current");
});

test("newer tagged feed sequence pivots snapshot away from a stale archived turn", async () => {
  const urls: string[] = [];
  const snapshot = await loadHelixSnapshot(async (url) => {
    urls.push(url);
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(
        JSON.stringify({
          events: [
            { kind: "browse", action: "old", turn_id: "turn-old", sequence: 9 },
            { kind: "browse", action: "current", turn_id: "turn-live", sequence: 11 },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    if (url.startsWith("/api/helix-engine/provenance")) {
      return new Response(
        JSON.stringify({
          available: true,
          selection: { source: "adaptive_checkpoint", turn_id: "turn-live" },
          missing_sources: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(
      JSON.stringify({
        turn_id: "turn-old",
        steps: [
          {
            name: "terminal",
            arguments: "old",
            result: "old",
            useful_hint: "useful",
            sequence: 10,
          },
        ],
        control_events: [],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }, "session-1", "thread-1");

  assert.equal(snapshot.turnId, "turn-live");
  assert.deepEqual(snapshot.steps, []);
  assert.deepEqual(snapshot.controlEvents, []);
  assert.deepEqual(snapshot.events?.map((event) => event.action), ["current"]);
  assert.ok(urls.some((url) => url.includes("turn_id=turn-live")));
  assert.equal(snapshot.provenance?.selection?.turn_id, "turn-live");
});

test("missing provenance is additive and does not discard live/session telemetry", async () => {
  const snapshot = await loadHelixSnapshot(async (url) => {
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(JSON.stringify({ events: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (url.startsWith("/api/helix-engine/session/")) {
      return new Response(JSON.stringify({ turn_id: "turn-old", steps: [], control_events: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ detail: "not supported" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }, "session-1", "thread-1");

  assert.equal(snapshot.successfulRequests, 2);
  assert.deepEqual(snapshot.errors, []);
  assert.match(snapshot.provenanceError ?? "", /Provenance: not supported/);
});
