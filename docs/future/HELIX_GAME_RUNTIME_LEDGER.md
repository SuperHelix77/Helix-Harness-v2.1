# Future Ledger — Helix Game Runtime

**Status:** HOLD / research parked  
**Last updated:** 2026-09-22

This ledger preserves the current graphics/runtime research direction without making it part of the active Helix Harness roadmap.

The immediate priority is to improve the efficiency of the current agent/orchestration workflow ("Steve") and continue the active Helix work. Nothing in this document should be treated as a committed release feature or active implementation task.

## Parked direction

Long-term concept: a high-performance Windows-games-on-Apple-Silicon runtime built around a measurable, modular stack:

```text
x86/x64 Windows game
        ↓
native ARM64 Wine / CrossOver
        ↓
FEX for x86/x64 CPU translation
        ↓
D3D11 → DXMT
D3D12 → DXMT12 / experimental DXMT D3D12 path
        ↓
Metal-native rendering layer
        ↓
Helix reconstruction / presentation / texture systems
        ↓
Metal 4
        ↓
Apple Silicon
```

The research target is not "port one vendor feature at any cost." The target is to reduce translation, rendering, reconstruction, frame-generation, and memory overhead while preserving correctness and image quality.

## 1. D3D12 → Metal

Investigate the existing experimental D3D12 work in DXMT rather than assuming a clean-sheet implementation is required.

Primary hypothesis:

> D3D12 should be translated as a dependency/semantic graph, not mechanically as one D3D12 synchronization operation per Metal synchronization operation.

Potential architecture:

- correctness-first SAFE translation mode;
- optimized GRAPH mode;
- explicit resource/subresource state tracking;
- barrier coalescing where semantics permit;
- queue/fence dependency modeling;
- Metal-native batching and synchronization;
- unified-memory-aware resource implementation;
- differential validation between SAFE and GRAPH.

No performance claim is accepted without output/correctness parity and attributed timing data.

## 2. Temporal super-resolution

FSR 4.x is a reference point, not a requirement to reuse or redistribute AMD's proprietary neural implementation.

Future goal:

- Metal-native temporal upscaler;
- compatible interception/front-end for game-provided DLSS/FSR/XeSS-style inputs where feasible;
- prioritize temporal stability, especially in Performance and Ultra Performance modes;
- explicitly target shimmer, ghosting, thin-geometry instability, foliage breakup, particle loss, disocclusion artifacts, specular instability, and UI contamination;
- train/distill smaller models when an open/trainable teacher and data pipeline are available.

The desired metric is a quality/latency Pareto improvement, not merely a smaller model file.

## 3. Adaptive frame generation up to 6×

Future target: variable-ratio frame generation aimed at the display refresh rate, with **6× as a ceiling rather than a permanent mode**.

Preferred architecture:

```text
shared temporal encoder(frame0, frame1, motion/depth/metadata)
                    ↓
          shared latent representation
                    ↓
          time-conditioned decoder(t)
                    ↓
      t = arbitrary positions between frames
```

Expensive shared features should be computed once per pair of real frames where possible.

The controller should choose 1×–6× using:

- real render rate;
- target display refresh;
- frame-generation latency;
- reconstruction confidence;
- disocclusion/motion complexity;
- frame pacing;
- hysteresis;
- latency constraints.

Generated/displayed FPS must never be confused with real simulation or input FPS.

## 4. Texture memory / neural texture compression

Goal: reduce physical unified-memory pressure rather than only reducing files on disk.

Staged direction:

1. better residency/eviction policy;
2. sparse/virtual textures;
3. demand-decoded BC/ASTC tile cache;
4. neural texture or neural residual representation where compute economics justify it.

Conceptually:

```text
logical D3D texture
        ↓
Metal sparse/virtual resource
        ↓
resident tile cache
        ↓
compressed / neural backing representation
```

Compare decode-on-load, decode-on-demand-tile, and decode-on-sample. Neural decoding is useful only if the memory/bandwidth savings outweigh its compute and latency cost.

## 5. CrossOver / FEX integration

The eventual runtime should keep CPU translation and graphics translation separately measurable.

Candidate comparison:

- Intel Wine + Rosetta + current D3D12 path;
- ARM64 Wine/CrossOver + FEX + DXMT/DXMT12.

Do not assume FEX is automatically faster. Measure:

- CPU translation cost;
- Wine/API overhead;
- graphics translation;
- shader compilation;
- synchronization stalls;
- GPU execution;
- unified-memory high-water mark.

Initial game targets should be offline/single-player. Anti-cheat bypass is out of scope.

## Evidence discipline

Every subsystem must be independently switchable and benchmarkable.

Future measurements should separate at least:

- real FPS;
- displayed/generated FPS;
- 1% low;
- frame-time variance;
- CPU translation time;
- D3D→Metal translation time;
- command encoding;
- shader compilation;
- render GPU time;
- SR GPU time;
- FG GPU time;
- texture-management GPU time;
- unified-memory high-water mark;
- texture logical vs resident bytes;
- sparse-cache hit rate;
- PSNR / SSIM / LPIPS where useful;
- temporal consistency;
- disocclusion/artifact-suite results.

Negative results stay negative. "Looks faster" is not evidence.

## Resume conditions

Do not reactivate this work until:

1. the current Helix/Steve orchestration path has been made materially more usage-efficient;
2. active Helix v2.1 product work is stable enough that graphics research will not compete with release-critical work;
3. a bounded first macOS experiment is selected.

Preferred first experiment when resumed:

> Build and inspect the existing DXMT experimental D3D12 path on Apple Silicon, establish the first working D3D12 sample, measure its current synchronization behavior, and implement only a correctness-first resource/barrier baseline before any SR, frame generation, texture compression, or FEX expansion.

## Non-claims

This ledger does **not** claim:

- FSR 4.x has been ported to Metal;
- a custom SR model currently beats FSR/MetalFX/DLSS;
- 6× frame generation is currently usable;
- neural texture compression currently reduces memory in CrossOver;
- DXMT D3D12 currently outperforms D3DMetal;
- FEX currently outperforms Rosetta on this workload.

Those are future hypotheses to test.

---

**Current decision:** park this research, preserve the findings, improve Steve/orchestration efficiency first, then return with a bounded experiment.
