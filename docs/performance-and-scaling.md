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

Install GPU support from the **ST-FLIP → GPU** panel (NVIDIA + CuPy). The same
solver runs on the GPU array backend, typically a large speed-up at production
resolutions where there is enough work to saturate the device. Results are
reproducible across CPU and GPU for the same seed and inputs.

GPU memory is the ceiling on resolution/particle count — a single consumer GPU
cannot hold a billion-particle production scene, so scale resolution to fit VRAM
and use previews for look-dev.

## The multigrid pressure solver

Set **Solver → Pressure Solver = Multigrid-PCG** for large grids. Jacobi-PCG's
iteration count grows with resolution; the geometric multigrid V-cycle keeps it
nearly flat (e.g. on a constant-coefficient 64³ Poisson, ~160 Jacobi iterations
vs. ~8 with multigrid). On grids too small to coarsen it falls back to Jacobi
automatically, so it is safe to leave enabled. It changes only convergence
speed, never the solution.

This holds for **two-phase** bakes too, even though the variable-density
pressure system is severely ill-conditioned at the water/air density ratio
(~800:1). Measured on a real two-phase step, Jacobi-PCG needs ~150 iterations
per solve while multigrid needs ~20, and the multigrid count stays
grid-independent up to 800:1 and 10000:1 ratios — so multigrid is the
recommended solver precisely when two-phase meets production resolution.

## The sparse grid

Enable **Sparse Grid** when the fluid occupies a small fraction of a large
Domain (a splash in a big room, a stream across a wide floor). The solver crops
each step to the active region plus a halo, cutting both time and memory. It
disengages when it cannot safely help (outflows, cut-cell obstacles, or
Two-Phase filling the domain).

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

## Cloud / remote baking and rendering

You do not have to tie up your workstation for a long bake or render:

1. **Provision a GPU instance** (any cloud GPU VM, or a render-farm service that
   accepts a `.blend` + add-on).
2. **Bake headless** on that machine, writing the cache to its disk.
3. **Render the frames** there (or pull the cache back and render locally).

Because the cache is plain files, you can bake remotely and render locally, or
vice-versa. Cost and wall-clock depend entirely on the instance and the scene;
budget by baking a short frame range first to measure per-frame time, then
extrapolate. Exact solver checkpoints are uncompressed and can be large at high
particle counts — provision disk accordingly, or keep only the compressed
playback frames if you do not need to resume.

## Quick checklist for a slow bake

- [ ] Resolution higher than it needs to be for this shot?
- [ ] GPU installed and detected?
- [ ] Multigrid pressure solver on for a large grid?
- [ ] Sparse grid on for a localized flow (and not blocked by an outflow/solid)?
- [ ] CFL as high as accuracy allows (8+)?
- [ ] Whitewater/Two-Phase on only when you actually need them?
