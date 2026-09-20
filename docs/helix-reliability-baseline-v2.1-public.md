# Helix Harness v2.1 — Public Reliability Baseline

## Control identity

The hostile-audit control was frozen on 2026-09-20 from the authoritative shared source tree.

- Base Git commit: `06a151d0e944e648cbd7f64044b7f01b08f7ed93`
- Base tag: `v2.0.0-rc.0.1`
- Dirty-tree snapshot schema: `path-nul-sha256-newline-v1`
- Frozen snapshot file count: 5,580
- Frozen snapshot byte count: 114,135,323
- Frozen snapshot SHA-256: `dbc6170d22db03c99580a0f5ef824c5bedcf3d9c390639005baf022ab0e1c309`

This digest, rather than the base commit alone, names the validated control because the reliability work intentionally existed in a shared dirty tree.

## What the frozen control established

- Durable generation and tool-call state across approval, claim, start, output, finish, reload, and replay.
- Replay-only committed receipts and fail-closed handling of unknown mutating outcomes.
- Worker ownership, lease fencing, account scoping, exact payload binding, cancellation, deadlines, and bounded output.
- A backend-owned durable Turn Finalizer with idempotent memory, audit, Hermes, skill, and learning boundaries.
- Exact backend-authoritative 80% checkpoint semantics plus hard-overflow protection.
- Runtime governance that distinguishes configured context from observed context exposure.
- Foreground-inference priority and a verified-target gate for post-answer QLoRA admission.
- Evidence/provenance classes that do not elevate model claims into verified facts.
- Temporary-skill retention only after post-task evidence.
- Packaged backend-overlay identity and clean desktop/backend/process teardown.

## Executed control validation

| Surface | Result |
| --- | ---: |
| Durable tool/recovery matrix | 851 passed |
| Account, memory, finalizer, learning | 1,108 passed, 1 skipped |
| Capability and confinement | 625 passed, 33 platform-specific skips |
| Runtime, context, checkpoint, governor | 1,051 passed, 1 skipped |
| Final skills/account matrix | 182 passed, 1 filesystem-specific skip |
| Packaging contracts | 60 passed |
| Overlay and identity contracts | 7 passed |
| Frontend suite | 7,588 passed |
| Frontend strict typecheck | passed |
| Frontend production build | passed |
| Bundle budget | passed |
| Rust locked tests | 480 passed |
| Rust `cargo check` | passed |
| `git diff --check` | passed |

The repository contains more backend test modules than the hostile priority matrices above. The control therefore records the executed slices exactly rather than representing them as one exhaustive monolithic run.

## Failure attribution discipline

The freeze distinguished product defects from environmental contamination. Examples include macOS skips for Linux-only confinement primitives, an isolated packaged-smoke warning caused by a source-tree working directory, and inaccurate test fixtures around memory-estimation and lazy PyTorch registration. Production behavior was not weakened to satisfy those contaminated cases.

## v2.1 release delta

v2.1 preserves the frozen invariants and adds:

- further terminal-replay and sibling-continuation recovery coverage;
- neutral macOS native-material styling;
- independently persisted transparency for the main background, sidebar, chat surface, composer, and contextual right rail;
- opaque/balanced/airy presets and reduced-transparency fallback;
- a separate resizable application identity: `ai.helix.harness.v21`.

The v2.1 candidate must pass its own focused backend, frontend, Rust, packaging, identity, launch/shutdown, and privacy gates before publication. This document does not convert the ad-hoc macOS build into a notarized Apple release.

## Preserved invariants

- Exact 80% adaptive checkpoint boundary and hard-overflow safety.
- No duplicate side effects after durable replay.
- Unknown mutating outcomes fail closed.
- Approval and execution identity bind account, run, tool, payload, claim, and checkpoint.
- Account isolation and typed capability boundaries remain authoritative.
- Stop/cancel and bounded tool-output semantics remain intact.
- Foreground inference preempts background audit/training.
- QLoRA remains post-answer and requires a verified target.
- Mem0 identity remains account-stable.
- Model self-audit remains non-objective enrichment.
- Execution Graph never fabricates causality.
- A 27B model is not admitted above the empirically safe context class merely because its model card advertises more.
