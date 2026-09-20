# Credits and upstream attribution

Helix Harness is built on **Unsloth Studio**, developed by the Unsloth AI Inc. team.

Upstream Unsloth provides the core foundation this fork depends on, including substantial portions of the training stack, model loading, quantized inference integrations, Apple-Silicon/MLX support, GGUF/llama.cpp integration, desktop/runtime infrastructure, and compatibility work across model families.

- Upstream project: https://github.com/unslothai/unsloth
- License: AGPL-3.0-only; see `LICENSE` and `studio/LICENSE.AGPL-3.0`.
- Upstream lineage is recorded in the public release documentation; machine-local baseline checkouts are not distributed.
- Unsloth's original copyright notices and SPDX/license headers are retained where applicable.

## Helix Harness additions

Helix Harness adds and maintains the Helix-specific product/control layer around that foundation:

- Helix-branded desktop UX and packaging.
- Local agentic chat/tool orchestration and reliability guards.
- Mem0/Qdrant-backed persistent memory integration.
- Helix Engine trajectory, cache/context, retry, prevented-action, and quality telemetry.
- Hermes post-task self-audit/learning control.
- Jev-inspired typed advisory decision control.
- Governed QLoRA/live-tuning/hot-swap workflows that require corroborated and independently verified targets.
- Foreground-loop semantic deduplication, bounded failure recovery, verification obligations, and control-event provenance.
- Additional regression coverage and fixes discovered while integrating these paths, including Qwen3.8 MLX runtime-routing correctness.

Helix Harness does not imply sponsorship, endorsement, or release ownership by Unsloth. Unsloth remains explicitly credited for the upstream groundwork on which this fork is built.
