# ST-FLIP Fluid for Blender

Large-time-step FLIP liquid simulation as an easy-to-use Blender addon, with
optional NVIDIA CUDA GPU acceleration.

This is an independent implementation of **ST-FLIP**:

> Bernhard Braun, Rene Winchenbach, Jan Bender, Nils Thuerey.
> *Spatiotemporal FLIP for Fast Free-Surface and Two-Phase Simulation With
> Very Large Time Steps.* ACM Transactions on Graphics 45(4), Article 76,
> SIGGRAPH 2026. <https://doi.org/10.1145/3811289>

## Why ST-FLIP?

Classic FLIP solvers need small time steps (CFL 1–2, i.e. fluid moves at most
1–2 grid cells per step) or the free surface dissolves into temporal-aliasing
ripples. ST-FLIP treats particles as **Monte Carlo samples in 4D space-time**:
each particle carries a small random time offset, particle-to-grid transfers
use a separable spatial × temporal kernel, and the accumulated kernel weights
double as a phase field that replaces per-step surface reconstruction. The
result: time steps **up to an order of magnitude larger** (CFL 8–15+) with
coherent, detailed flow.

Measured on a 64³ dam break (344k particles, 8 frames at 24 fps) with
Blender 5.1's Python on Windows 11, Ryzen 7 9800X3D, and RTX 5090. Times
include solver stepping and render-particle re-synchronization:

| Configuration                  | s / frame | vs. baseline |
|--------------------------------|-----------|--------------|
| Instantaneous P2G, CFL 1, CPU  | 5.43      | 1×           |
| ST-FLIP, CFL 8, CPU            | 4.57      | 1.2×         |
| ST-FLIP, CFL 8, RTX 5090       | 0.53      | **10.2×**    |
| ST-FLIP, CFL 15, RTX 5090      | 0.52      | 10.5×        |

The timing baseline disables temporal weighting and jitter but retains this
add-on's phase-field projection. It is an ablation control, not the paper's
standard-FLIP/GFM reference solver.

## Install

1. Download a release ZIP, or build a clean extension archive from a clone:

   ```bash
   python tools/build_extension.py
   ```

   The archive is written to `dist/st_flip-<version>.zip`. Do not install a
   GitHub source ZIP directly: its extra top-level folder and development files
   are not a valid Blender extension package layout.
2. Blender → *Edit → Preferences → Get Extensions → Install from Disk…* and
   pick the ZIP.
3. Enable **ST-FLIP Fluid**. Requires Blender **4.2+** (tested on 5.1).

### GPU acceleration (optional, recommended)

*ST-FLIP panel → Solver → Install GPU Support (CUDA)* installs a pinned CuPy
runtime into an isolated Blender user-module directory, then runs allocation,
kernel, reduction, scatter, and synchronization checks before enabling it.
The preferred CUDA 13 bundle is roughly a 1.1 GB download / 1.5 GB installed;
it does not require a separately installed CUDA toolkit. A CUDA 12 wheel is
tried only if the compute preflight fails. Needs a current NVIDIA driver.
Blackwell GPUs (RTX 50xx) are supported via PTX JIT.

AMD GPU acceleration is not integrated. CuPy's ROCm support is experimental
on Linux and recent CuPy versions do not publish official ROCm wheels; the
add-on therefore uses its CPU backend unless the user supplies a compatible
custom build. A vendor-neutral wgpu backend is on the roadmap.

## Use

1. *3D Viewport → Sidebar (N) → ST-FLIP → Quick Dam-Break Setup*, **or**:
   - Create a box, set it as the **Domain** (defines the grid).
   - Select any closed mesh, set its role to **Liquid**, **Inflow**
     (with a velocity), or **Obstacle**.
2. Pick **Resolution** (cells along the longest domain axis) and
   **Target CFL** (8 is a good start; raise for speed, lower for accuracy).
3. Press **Bake Simulation**. Frames are cached to disk (`Cache Directory`)
   and playback is driven by a frame-change handler.

The bake produces two objects:

- **STFLIP Particles** — a point cloud with a `velocity` point attribute
  (usable for motion blur or Geometry Nodes).
- **STFLIP Liquid Surface** — a Geometry Nodes
  points → volume → mesh surface ready for materials and rendering.
  Tune *Radius* / *Voxel Size* on its modifier.

Gravity comes from the scene's gravity settings; frame rate from the render
settings.

### Solver settings

| Setting | Paper ref | Meaning |
|---|---|---|
| Target CFL | Algorithm 1, §4 | Global step target from 0.5–30; paper examples use values through 30 |
| Particles / Cell | §4.5 | Initial samples per occupied cell, 1–64; paper sweeps 1–16 against a 50-particle reference |
| Spatiotemporal Sampling | §3 | Disable temporal weighting and jitter for an instantaneous-P2G ablation; this is not a full standard-FLIP/GFM baseline |
| Jitter Strength (γ) | Eq. 10 | Base temporal jitter amplitude; 1 permits full-slab jitter before adaptive attenuation |
| Adaptive Attenuation | §3.10 | Less jitter noise on calm surfaces |
| Interface Steepness (η) | Eq. 13 | Lower = steeper/stronger leveling; higher = finer detail/noise |
| FLIP Fraction | §4 | FLIP/PIC blend (default 0.98) |
| Random Seed | §3.10 | Reproducible particle placement and temporal jitter; seed 0 is an add-on default, not a published paper value |
| Initial / Inflow Velocity | §4.8 | Per-liquid uniform or solid-body rotational initial velocity; inflow meshes remain uniform |

### Solid-body initial velocity

For each **Liquid** object, choose **Solid Body Rotation** to initialize the
actual jittered particles with
`u(x) = v_linear + omega × (x - center)`. The center and axis are entered in
Blender world coordinates. The axis is normalized, the signed angular speed
is in radians per scene second, and positive speed follows the right-hand
rule. The linear velocity is superposed on the rotation. Field sampling uses
a deterministic host-float32 path, so a fixed seed produces the same initial
particle positions and velocities on CPU and CUDA.

With the paper's vertical axis mapped through Blender's world origin, its
whirlpool initialization is represented by center `(0, 0, 0)`, axis `+Z`,
zero linear velocity, and angular speed `0.1 rad/s`. This reproduces the
published initial velocity field, not the complete experiment: the
cylindrical bottom outflow and an exact scene preset are still unavailable.

### Experiment profiles and diagnostics

The **Experiment Diagnostics** panel provides parameter-only, paper-inspired
profiles. They apply reproducible solver settings to the current Blender
scene; they do **not** recreate the paper's geometry, reference solvers, or
surface reconstruction:

- Laminar dam break (§4.1): target CFL 1, 3, 5, 10, or 20.
- Standard dam break (§4.3): target CFL 1, 2, 4, 8, or 16, plus an explicitly
  labeled instantaneous-P2G ablation that is not standard FLIP/GFM.
- Enstrophy (§4.4): ST-FLIP analogs for `(CFL, FLIP)` pairs `(1, .99)`,
  `(5, .99)`, `(10, .99)`, `(1, .95)`, and `(1, .90)`. The paper's CFL 1
  comparison curves use the unavailable standard-FLIP/GFM solver.
- Particle count (§4.5): the ST-FLIP/CFL 10 branch at 1, 2, 4, 8, or 16
  particles/cell plus the 50-PPC reference. The paper's standard-FLIP/CFL 1
  branch is unavailable.

Enable **Record Frame Metrics** before baking to append strict schema-v1
records to `stflip_metrics.jsonl` in the cache. The export button produces an
atomic, self-contained CSV or JSON file. Evolved output frames record
solver-only wall time, time-step and observed particle-CFL summaries, PCG
iteration/residual summaries, particle count, speed, center of mass, momentum,
and equal-particle-mass kinetic-energy and volume estimates. **Compute
Enstrophy** additionally records
`0.5 * integral(|curl(u)|^2) dV` and a `phi >= 0.5` phase-threshold volume
estimate from the MAC grid.

Distances and derived quantities remain in raw Blender/solver units; the
scene unit system and scale are saved alongside them. Particle-volume and
phase-threshold volume are explicitly estimates, not the paper's unspecified
volume estimator or normalization. The observed CFL fields use this solver's
maximum particle speed and are therefore not labeled as paper-equivalent grid
CFL. Enstrophy adds an O(grid) diagnostic and synchronization, but it is timed
outside the reported solver-only wall time.

### Paper coverage

Version 0.5 implements the paper's **single-phase, free-surface ST-FLIP
core**; it is not a full reproduction of every solver variant and production
example in the paper.

| Paper capability | Status in this add-on |
|---|---|
| One-sided temporal kernel, slab P2G, residual jitter, phase field, variable-coefficient projection | Implemented |
| Large target CFL and instantaneous-P2G temporal ablation | Implemented; target CFL 0.5–30 |
| Reproducible reruns over paper-inspired CFL, particle-count, and FLIP/PIC parameter matrices | ST-FLIP-side profiles and auditable frame diagnostics implemented; no true FLIP/GFM branches, automated batch runner, or exact scene presets |
| Single-phase liquid scenes with static mesh obstacles and inflows | Implemented |
| Fractional solid face apertures in Eq. 14–17 | Implemented for stationary mesh obstacles using an add-on-specific node-SDF reconstruction; moving/deforming solids remain unsupported |
| Paper render reconstruction (`0.5Δx` spheres, 2× grid, MCF) | Partial; radius/voxel defaults match the first two values, but Geometry Nodes replaces MCF |
| Two-phase liquid/gas coupling | Not implemented |
| APIC and implicit-density-projection comparison solvers | Not implemented |
| Surface-tension examples | Not implemented |
| Sparse/adaptive grids and billion-particle production scale | Not implemented |
| Appendix B mean-curvature-flow output reconstruction | Not implemented; Geometry Nodes volume meshing is an approximation |
| Animated/deforming obstacle boundary conditions | Not implemented |

Experiment-level coverage is narrower than method-level coverage:

| Paper experiment | Reproduction level |
|---|---|
| Laminar/standard dam breaks over large CFL values | Partial: parameter profiles, qualitative setup, and raw diagnostics, without exact geometry, GFM/APIC baselines, or the paper's unspecified normalization |
| Static-obstacle wake and thin-obstacle inflow jet | Partial: stationary obstacles use fractional face apertures and inflows are supported, but the paper's exact geometry and jet parameters are unpublished |
| Particle-count and FLIP-blend studies | ST-FLIP-side parameter profiles, deterministic seed, and enstrophy diagnostic are available; true FLIP/GFM branches, SDF RMSE, and batch-sweep tooling are not |
| Kleefsman obstacle validation | Missing water-height gauges and experimental-data comparison |
| MCF reconstruction study | Missing MCF and normal-RMSE evaluation |
| Whirlpool, rotational fields, and outflow scenes | Partial: the published solid-body rotational initialization is controllable; the cylindrical outflow and exact scene preset are missing |
| Two-phase glugging/discharge and production-scale scenes | Missing two-phase solver and sparse production grid |

The Blender UI exposes these inputs for the implemented single-phase solver:
resolution, target CFL, particles/cell, `γ`, adaptive attenuation, `η`,
FLIP/PIC blend, seed, backend, gravity, frame rate/range, uniform inflow
velocity, and per-liquid uniform or solid-body initial velocity (linear
velocity, world center/axis, and signed angular speed), plus surface display
radius/voxel size. Every
bake records these settings in its cache metadata. The solver's Python
`Params` API additionally exposes liquid density, pressure tolerance/iteration
limit, local advection CFL, the under-sampling threshold, and the relative
density floor. Paper-stated defaults include `γ = 1`, `ηφ = 0.5`,
`αFLIP = 0.98`, local CFL `= 1`, and PPE tolerance `= 1e-4`; the add-on also
uses `ρ = 1000` and a 400-iteration PCG limit. The paper's `kψ = 30` belongs
to the unimplemented MCF reconstruction. Controls for gas properties, surface
tension, APIC, sparse-grid adaptivity, and MCF reconstruction do not exist
because the corresponding algorithms are not implemented.

## What's implemented

- MAC-grid FLIP/PIC with separable poly6 spatial kernel (Eq. 18) and
  one-sided temporal kernel (Eq. 19)
- 4D→3D slab-integrated P2G with weight-accumulator phase field (Eq. 8–13)
- Variational variable-coefficient pressure projection, matrix-free
  Jacobi-PCG (Eq. 14–17), including node-SDF-derived fractional solid face
  apertures, `div(alpha * u)`, and aperture-weighted PPE coefficients
- Temporal jitter with residual carryover and the Appendix A boundedness
  guarantee (tested), adaptive γ attenuation
- Globally adaptive time stepping with even frame subdivision, sub-stepped
  RK3 advection at local CFL ≤ 1
- Particle re-synchronisation (un-jittering) at output frames
- Stationary solid obstacles via cell/node-sampled voxelised SDF, fractional
  face apertures, and particle push-back; inflow emitters
- Uniform and right-handed solid-body initial velocity fields, sampled at the
  actual jittered particle positions with deterministic CPU/CUDA setup
- NumPy CPU + CuPy CUDA backends sharing one code path
- Paper-inspired parameter profiles plus strict JSONL frame diagnostics and
  atomic CSV/JSON export; optional discrete MAC-grid enstrophy

Not (yet) implemented from the paper: two-phase air–liquid coupling, surface
tension (CSF), APIC transfers, sparse/adaptive grids, mean-curvature-flow
surface smoothing (Appendix B — the Geometry Nodes volume meshing stands in),
and the paper's standard-FLIP/GFM and implicit-density-projection baselines.
Outflow boundaries, including the whirlpool experiment's cylindrical bottom
pipe, are also not yet implemented.

Known limitations: scene voxelization (mesh → grid masks/SDF) is a pure
Python BVH loop and gets slow above resolution ~128 — vectorizing it is on
the roadmap. Temporal jitter randoms are drawn on the host for cross-backend
determinism, costing a small per-step transfer on GPU.

### Headless / scripted baking

`bpy.ops.stflip.bake()` (EXEC context) runs the bake synchronously, so it
works from scripts and `blender -b`:

```bash
blender -b scene.blend --python-expr "import bpy; bpy.ops.stflip.bake()"
```

For a diagnostic bake, set `scene.stflip.collect_metrics = True` (and
optionally `collect_enstrophy = True`) before invoking the synchronous bake.
The canonical JSONL remains usable after an interrupted bake; malformed or
partial records are ignored and exports include only frames with valid cache
files.

## Development

```bash
# CPU tests
uv run --no-project --with pytest --with numpy pytest tests -v
# GPU parity test (NVIDIA GPU required)
uv run --no-project --with pytest --with numpy --with "cupy-cuda13x[ctk]==14.1.1" pytest tests -m gpu -v
```

The solver (`stflip/`) is bpy-free and usable standalone:

```python
from stflip import Params, STFLIPSolver
import numpy as np

p = Params(resolution=(64, 64, 64), dx=1/64, cfl_target=10.0)
s = STFLIPSolver(p, "auto")          # "cpu" | "cuda" | "auto"
mask = np.zeros((64, 64, 64), bool)
mask[:20, :, :32] = True
s.add_liquid_mask(mask)
for frame in range(48):
    s.step_frame()
    pos, vel = s.get_render_particles()
```

## License

MIT. The ST-FLIP paper is CC-BY 4.0; this repository is an independent
implementation of the published method.
