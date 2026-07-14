# Fidelity milestone roadmap

This roadmap defines what the current fidelity milestone accepts as complete
and, equally importantly, what the repository does **not** claim. It tracks the
independent ST-FLIP implementation described by Braun et al.
([DOI 10.1145/3811289](https://doi.org/10.1145/3811289)); it is not a claim of
source-code identity with the authors' production solver.

## Acceptance status

| Area | Acceptance criterion | Status |
|---|---|---|
| Scene units | One documented conversion boundary: geometry/velocity/gravity remain Blender-unit RNA values; SI density is multiplied by `scale_length³`, SI kinematic viscosity divided by `scale_length²`, and surface tension unchanged | Complete; policy and scale are recorded in bake metadata |
| Inflows | Occupancy is checked before every active global simulation step; cells below half target PPC receive a full PPC packet; a future schedule start splits the step at its exact start time | Complete and regression-tested, including equivalent global-step partitions; this is not RK-substep refill or a prescribed rate |
| Pressure projection | A non-finite terminal residual or finite residual above tolerance cannot be silently accepted | Complete; an explicit pressure-solve error aborts the uncommitted bad step, with no automatic fallback |
| Paper-facing preset | One curated settings-only action resets the experiment profile to Custom and selects listed temporal controls plus Appendix-B Paper MCF without deleting cache data or changing unlisted controls | Complete: **Final / Paper Fidelity** |
| Large-world Paper MCF | Reconstruct in the Domain-local frame, conditionally preserve synchronized local positions when world float32 is inadequate, place the Blender object by translation, and reject a lossy world-only legacy cache | Complete; near-origin legacy fallback remains supported |
| Real Blender/OpenVDB integration | Install the built archive into an isolated Blender 4.2 profile; run a tiny CPU step, two-iteration Paper field, Geometry Nodes, and a non-empty bundled-OpenVDB isosurface | Complete in pull-request CI with a pinned/checksummed release archive from an NLUUG mirror; not a full bake or throughput test |
| CUDA integration | A missing GPU must never count as coverage; a real core CUDA step must pass CPU-parity tolerances | Smoke tool and manual self-hosted NVIDIA workflow complete. Coverage exists only when that workflow passes; it does not exercise Blender/Paper and is not a required public PR job |
| Kleefsman scene | Published geometry and H2/H4 x-locations are executable; gauge estimator and unpublished sampling choices are explicit; experimental input is attributable and hashed | Complete as a geometry/gauge harness. No experimental trace is bundled, so experimental validation remains external-data-dependent |
| Paper-constrained glug scene | Published ST-FLIP dimension ratios are executable; every unpublished wall/layout choice is recorded | Complete as a compact two-phase regression, not an exact production/Blender scene, golden-baseline comparison, or PF-FLIP comparison |
| Tiled storage Phase 1 | Deterministic core/halo layout, dense coarse lookup table, neighbours, dense-boundary pack/unpack, packed halo exchange, and callable sparsity telemetry have round-trip/halo/boundary tests | Complete as a standalone NumPy representation; no solver/backend/cache path uses it and no memory/time gain follows |
| Fully tiled solver | P2G/G2P, pressure/multigrid, material stencils, cache schema, and CUDA kernels execute over packed tiles with dense-path parity | Not implemented; see [tiled sparse grid](tiled-sparse-grid.md) Phases 2–5 |
| Comparison solvers | Standard-FLIP/GFM, implicit-density-projection, and PF-FLIP branches reproduce the paper's comparison curves | Not implemented and not claimed |
| Production scale | Multi-billion-particle scenes are demonstrated on suitable distributed or multi-GPU hardware | Not implemented and not claimed |

## Preset provenance

**Final / Paper Fidelity** is a curated mixed preset that resets **Experiment
Profile** to Custom. It combines published method-facing values (`γ=1`,
`ηφ=0.5`, FLIP fraction `0.98`, local CFL 1, PPE tolerance `1e-4`, and the
Appendix-B `0.5Δx` reconstruction with 30 MCF iterations) with explicit add-on
workflow choices: target CFL 8, 8 particles/cell, seed 0, and zero OpenVDB mesh
adaptivity. Those workflow choices are useful reproducible defaults inside the
paper's explored regime; they are not presented as a unique configuration
mandated by the paper. It does not set geometry, resolution, FPS, backend,
pressure-solver choice/maximum iterations, densities, two-phase controls,
reconstruction memory limits, or other unlisted settings.

The action preserves existing cache files. Cache fingerprints remain the
authority: rebake the simulation when solver inputs differ, or rebuild only the
derived Paper surface when its output settings differ.

## Evidence tiers

1. Unit tests close deterministic numerical and boundary contracts.
2. Pull-request CI validates a narrow installed-extension path in a Blender 4.2
   release archive from an NLUUG mirror and requires OpenVDB polygonization.
3. The manual self-hosted workflow is the only CUDA smoke and requires a real
   device; it tests the core, not Blender or Paper surfacing.
4. `tools/run_paper_validation.py` produces strict JSON artifacts for the
   reproducible Kleefsman and paper-constrained glug scenes. Its defaults are
   compact smoke settings; evidence-scale resolution/duration are user choices.
5. Kleefsman becomes an experimental comparison only when an attributable
   external CSV and citation are supplied. See
   [validation/README.md](../../validation/README.md).

No lower evidence tier is promoted into a stronger claim: a unit test is not a
Blender integration test, a smoke is not a throughput benchmark, the
paper-constrained glug is not PF-FLIP parity, and Phase-1 tiled storage is not a
tiled solver.

## Next research gates

The remaining work is gated by evidence rather than release language:

- integrate automatic tile telemetry into representative bakes, then proceed
  to a tiled Poisson operator only where tile footprint materially beats the
  dense active boxes;
- require dense/tiled operator and full-step parity before selecting a tiled
  path in production;
- add comparison-solver branches only with independently verified definitions
  and reference configurations;
- publish Kleefsman error claims only with a redistributable or user-supplied,
  attributable numeric dataset;
- make production-scale claims only from archived hardware, memory, particle,
  and timing artifacts at that scale.
