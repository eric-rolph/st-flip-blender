# Performance and scaling

## What costs time

The solver works on a 3-D grid, so cost scales steeply with **Resolution**:
doubling resolution multiplies cell count (and roughly the work) by ~8. The two
big consumers per step are usually:

- **Advection + transfer** — proportional to particle count.
- **Pressure projection** — an iterative solve whose cost grows with grid size
  and with how hard the system is to solve.

Everything below is about making those cheaper without lowering resolution.

## GPU acceleration

Install GPU support from **ST-FLIP → Solver**. When CUDA compute is unavailable,
press **Install GPU Support (CUDA)**. GPU usually helps most at production
resolutions where there is enough work to saturate the device.

A fixed seed gives CPU and CUDA the same initial particle random state. Evolved
results should be numerically close, not bitwise-identical across backends,
hardware, drivers or numerical-library versions.

GPU memory is the ceiling on resolution/particle count — a single consumer GPU
cannot hold a billion-particle production scene, so scale resolution to fit VRAM
and use previews for look-dev.

## The multigrid pressure solver

Set **Solver → Pressure Solver = Multigrid-PCG** for large grids. Jacobi-PCG's
iteration count grows with resolution; the geometric multigrid V-cycle keeps it
nearly flat (e.g. on a constant-coefficient 64³ Poisson, ~160 Jacobi iterations
vs. ~8 with multigrid). On grids too small to coarsen it falls back to Jacobi
automatically, so it is safe to leave enabled. It changes only convergence
speed while solving the same discretized PPE to tolerance. Reduction order and
iteration paths can still produce small roundoff-level differences.

This holds for **two-phase** bakes too, even though the variable-density
pressure system is severely ill-conditioned at the water/air density ratio
(~800:1). Measured on a real two-phase step, Jacobi-PCG needs ~150 iterations
per solve while multigrid needs ~20, and the multigrid count stays
grid-independent up to 800:1 and 10000:1 ratios — so multigrid is the
recommended solver precisely when two-phase meets production resolution.

## Automatic pressure crops

Both pressure solvers use one or more tight active boxes when the projected
reduction is worthwhile; otherwise they solve the full grid.

Regions separated by complete empty lattice planes can use independent boxes
even when the **Sparse Grid** toggle is off.

This helps localized pours with drains or cut-cell obstacles. It is still dense
inside each box, and a thin full-span sheet may see little benefit.

## The sparse grid

Enable **Sparse Grid** when the fluid occupies a small fraction of a large
Domain (a splash in a big room, a stream across a wide floor). The solver crops
each step to the active region plus a halo, cutting both time and memory. It
disengages when it cannot safely help (outflows, cut-cell obstacles, or
Two-Phase filling the domain).

This is a dense active-window optimization, not fully tiled sparse storage.
The repository now ships Phase-1 tiled-storage primitives—deterministic
core/halo layouts, a dense coarse lookup table, dense-field pack/unpack,
one-cell packed halo exchange, neighbour slots, and callable bbox/tile
telemetry—but they are
solver-independent. They do not yet reduce a bake's memory or compute cost.
See the [tiled-grid design](design/tiled-sparse-grid.md) for the parity-gated
integration phases.

## Resolution strategy

1. **Block out motion at low resolution.** Get timing, camera, and gross
   behaviour right where each bake is seconds, not minutes.
2. **Raise resolution for the final bake**, turning on Multigrid + GPU (+ Sparse
   where applicable).
3. **Bake once, render many times.** The cache is the expensive artifact; keep
   it and iterate on shading/lighting freely.

## Headless / scripted baking

Bakes can run without the UI for overnight or farm jobs — bake from a Python
script via Blender's background mode. See the top-level
[README](../README.md#headless--scripted-baking) for the exact invocation. This
is also how you drive a bake on a remote machine.

## What CI performance coverage does—and does not—prove

Pull-request CI installs the extension into a pinned/checksummed Blender 4.2
release archive from an NLUUG mirror, runs a tiny real CPU step, performs a
two-iteration Paper reconstruction, builds Geometry Nodes, and requires
Blender's bundled OpenVDB binding to emit a non-empty mesh. This validates a
narrow installed-extension integration path, not a complete bake or production
throughput.

CUDA validation is a separate manual workflow for a labelled self-hosted
NVIDIA runner. It passes `--require-gpu` and compares a tiny core CUDA step with
CPU within declared tolerances; a CPU-only machine cannot be reported as a CUDA
pass. It does not run Blender or Paper surfacing, and it is not a required
public pull-request job. Neither smoke establishes billion-particle scale or
PF-FLIP equivalence.

## Cloud / remote baking and rendering

You do not have to tie up your workstation for a long bake or render:

1. **Provision a GPU instance** (any cloud GPU VM, or a render-farm service that
   accepts a `.blend` + add-on).
2. **Bake headless** on that machine, writing the cache to its disk.
3. **Render the frames** there (or pull the cache back and render locally).

Because the cache is plain files, you can bake remotely and render locally, or
vice-versa. Cost and wall-clock depend entirely on the instance and the scene;
budget by baking a short frame range first to measure per-frame time, then
extrapolate. Primary-solver checkpoints are uncompressed and can be large at
high particle counts. Provision disk accordingly, or keep only compressed
playback frames if you do not need Resume.

## Quick checklist for a slow bake

- [ ] Resolution higher than it needs to be for this shot?
- [ ] GPU installed and detected?
- [ ] Multigrid pressure solver on for a large grid?
- [ ] Sparse grid on for a localized flow (and not blocked by an outflow/solid)?
- [ ] CFL as high as accuracy allows (8+)?
- [ ] Whitewater/Two-Phase on only when you actually need them?
