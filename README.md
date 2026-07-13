# ST-FLIP Fluid for Blender

Large-time-step FLIP liquid simulation as an easy-to-use Blender addon, with
optional NVIDIA CUDA GPU acceleration and a paper-defined Appendix-B render
surface path.

This is an independent implementation of **ST-FLIP**:

> Bernhard Braun, Rene Winchenbach, Jan Bender, Nils Thuerey.
> *Spatiotemporal FLIP for Fast Free-Surface and Two-Phase Simulation With
> Very Large Time Steps.* ACM Transactions on Graphics 45(4), Article 76,
> SIGGRAPH 2026. <https://doi.org/10.1145/3811289>

[![ST Flip Blender](https://img.youtube.com/vi/VwIIyES4_Gg/maxresdefault.jpg)](https://youtu.be/VwIIyES4_Gg)

**New here?** The [User Guide](docs/README.md) has step-by-step tutorials:
[getting started](docs/getting-started.md) ·
[object roles](docs/object-roles.md) ·
[recipes](docs/recipes.md) ·
[settings guide](docs/settings-guide.md) ·
[performance](docs/performance-and-scaling.md) ·
[rendering & export](docs/rendering-and-export.md) ·
[troubleshooting](docs/troubleshooting.md).

## Why ST-FLIP?

Hybrid FLIP solvers can remain numerically stable when particles cross several
grid cells in one step, but stability is not fidelity. The paper shows that
instantaneous particle-to-grid deposition leaves gaps in space-time coverage;
pressure projection amplifies the resulting temporal aliasing into unphysical
surface waves, often becoming visually objectionable above roughly CFL 2–3.

ST-FLIP treats particles as **Monte Carlo samples in 4D space-time**. Each
particle carries a bounded temporal residual, deposition uses a separable
spatial × temporal kernel, and the accumulated weights become a space-time
phase field for variable-coefficient pressure projection. This is more than
naively randomizing advection times: the tracked residual and temporal kernel
keep particle state and deposited mass/momentum consistent. Locally sub-stepped
RK3 advection still enforces collision-safe travel; the larger global step
reduces expensive P2G/projection updates rather than eliminating all substeps.

The paper reports benchmark-dependent **2×–8×** wall-clock improvements at
high effective resolutions and target CFL values through 30. Its often-quoted
`≈ O(n⁴)` observation is a resolution-scaling argument for 3D CFL-conforming
simulations—`n³` cells plus about `n` times as many steps—not a universal
complexity guarantee. The authors' sparse/adaptive CPU implementation and
multi-billion-particle scenes also do not directly predict this add-on's
dense-grid CPU or CUDA performance.

The algorithm is lightweight, not free: the paper adds one particle-time
attribute and changes advection/P2G/interface handling, while avoiding extra
grids, linear solves, or particle passes in its host architecture. One reported
comparison measured 3.7% additional peak memory. This repository is a
standalone solver, not a binary drop-in patch for Mantaflow or FLIP Fluids.

### Reproducible core validation

`tools/run_validation.py` runs a deterministic four-case matrix on one fixed
backend: ST-FLIP and instantaneous-P2G at matched target CFL 1 and 16. It
requires identical initial particle hashes, records actual (not merely target)
CFL, residual bounds, trajectory/occupancy/energy differences, output hashes,
and timing in a separate section. The instantaneous branch is an ablation of
this solver; it is not the paper's standard-FLIP/GFM reference implementation.

```bash
python tools/run_validation.py
python tools/run_validation.py --scale evidence --seeds 0 1 2 --require-validation-ready
```

The quick scale is a 2-PPC CI/developer stress smoke and may correctly report
that it did not reach high observed CFL or the expected coherence trend. The
evidence scale uses the paper/add-on default of 8 PPC. Its internal regression
gate requires lower mean error in the bounded Eq. 13 pressure/interface phase
field, higher mean threshold-interface IoU, and strict-majority agreement for
both across three seeds. Raw Eq. 8 deposited-mass RMSE remains prominent as a
non-gating Monte Carlo-variance diagnostic. CUDA timing must be run separately
at the same ST-FLIP configuration; changing the method, target CFL, and backend
in one speedup ratio would conflate three effects.

This matrix validates the implemented temporal mechanism and an internal
phase-field coherence surrogate, not an exact paper reproduction: it uses a
dense unit-cube dam break and an internal diffuse-phase instantaneous ablation
instead of standard FLIP/GFM or APIC. It does not replace the paper's
predeclared `T=7` SDF/mass-slice studies or MCF-smoothed render-surface normal
metric. The add-on now implements that surface operator, but the validator does
not reproduce the paper's geometry or reference data. Output hashes are audit
fingerprints for the artifact's declared
Python/NumPy/backend environment, not bitwise-determinism claims across
hardware or numerical-library versions.

The checked [three-seed CPU evidence artifact](https://github.com/eric-rolph/st-flip-blender/blob/main/validation/stflip_matched_cpu.json)
was regenerated from its embedded source hashes. Every run reached observed
particle CFL above 10.6. Mean Eq. 13 phase RMSE was `0.07880` for ST-FLIP
versus `0.08574` for the instantaneous ablation (ratio `0.9191`, ST better in
3/3 seeds); mean threshold-interface IoU was `0.92829` versus `0.92175` (ST
better in 3/3). Raw deposited-mass RMSE was `0.23379` versus `0.23034` (ratio
`1.0150`, ST better in only 1/3), so the artifact reports that diagnostic as
inconclusive rather than disguising Monte Carlo noise as a successful paper
reproduction.

## Install

1. Download `st_flip-<version>.zip` from the
   [latest release](https://github.com/eric-rolph/st-flip-blender/releases/latest),
   or build the same deterministic extension archive from a clone:

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

1. Save the `.blend` file, or choose an absolute **Cache Directory**.
   Blend-file-relative caches (`//...`) are refused in an unsaved file so a
   later Save As cannot silently move the cache path.
2. *3D Viewport → Sidebar (N) → ST-FLIP* and choose **Dam Break**,
   **Whirlpool Preview**, or the approximate **High-CFL Jet Preview**;
   open the **Presets** sub-panel for one-click feature demos
   (**Viscous Pour**, **Stormy Pool**, **Two-Phase Glug**, **Fountain**),
   each of which builds a ready-to-bake scene and configures the relevant
   solver features; **or** build a scene by hand:
   - Create a box, set it as the **Domain** (defines the grid).
   - Select any closed mesh and set its role to **Liquid**, **Inflow**
     (with a velocity), **Outflow**, or **Obstacle**.
3. Pick **Resolution** (cells along the longest domain axis) and
   **Target CFL** (8 is a good start; higher values may reduce global solves
   when the frame interval does not cap the step, while lower values generally
   trade more work for accuracy).
4. Press **Bake Simulation**. Frames are cached to disk (`Cache Directory`)
   and playback is driven by a frame-change handler. To continue a cancelled,
   failed, or completed long bake, extend the scene's End frame and press
   **Resume**.

Every committed frame has both a compressed playback frame and an atomic raw
solver checkpoint. Resume re-voxelizes the scene and refuses to continue if
trajectory-defining inputs, geometry, outlet modes, or the compute backend no
longer match the checkpoint. Existing output and metric history are preserved.
Raw checkpoints trade disk space for exact restart state, so plan cache storage
accordingly for high particle counts and long frame ranges.

Save the `.blend` before using the default relative cache. An unsaved file has
no stable base directory, so Bake/Resume asks you to save first; an explicitly
absolute Cache Directory can be used without saving the scene.

The default cache root is namespaced by a persistent scene ID in saved
`.blend` files, preventing one scene from overwriting or freeing another
scene's bake. Explicit custom cache paths remain exact and carry ownership
metadata; foreign caches are readable only by their owning scene.

The bake produces two objects:

- **STFLIP Particles** — a point cloud with a `velocity` point attribute
  (usable for motion blur or Geometry Nodes).
- **STFLIP Liquid Surface** — choose **Fast Preview** for deterministic
  Geometry Nodes points → volume → mesh output, or **Paper MCF** for the
  Appendix-B `0.5Δx` sphere union, 2× scalar grid, fixed feature mask, and
  level-set mean-curvature flow before OpenVDB polygonization. Paper MCF uses
  `kψ = 30` by default; the dense scalar stage follows the selected NumPy/CuPy
  backend while OpenVDB meshing is CPU-only.

Paper surfaces are derived cache entries, separate from restart checkpoints.
When selected before a simulation bake, each committed output frame receives a
matching surface mesh. **Rebuild Paper Surface Cache** regenerates every
committed particle frame after changing MCF/adaptivity settings, is cancellable
between frames, and activates the new configuration only after the full range
finishes. A conservative preflight budgets the configured field cap, dense
temporaries, particle copies, OpenVDB work, and any live solver. If the combined
CUDA allocation is unsafe but host RAM is sufficient, the simulation remains
on CUDA while the entire derived surface configuration is pinned to CPU—frames
are never mixed across reconstruction backends. The voxel setting is a hard
field-size cap, not by itself a memory guarantee. **Fast Preview** remains the
recommended interactive/scrubbing mode.
Each derived mesh is bound to the exact source-position hash, reconstruction
configuration hash, frame number, and its own mesh hash.

Surface reconstruction is not part of the authoritative solver transaction. A
runtime OpenVDB, allocation, or derived-cache failure is recorded as a failed
Paper cache while particle frames and exact resume checkpoints continue to
commit. Resume follows the current Surface Method; it never revives an old
Paper configuration after the user switches to Fast Preview or disables
surfacing. A missing, failed, changed, or memory-unsafe Paper configuration is
left inactive and can be regenerated over all committed frames with **Rebuild
Paper Surface Cache**.

Gravity comes from the scene's gravity settings; frame rate from the render
settings.

One-click setups replace generated setup objects, clear this scene's owned
bake, and may change scene-global units, gravity, FPS, and frame range. When a
bake exists Blender asks for confirmation because the disk-cache deletion
cannot be restored by Undo. Use a new scene or save first if those settings
matter. Editing the Jet Preview's
domain, resolution, FPS, source speed, or plate dimensions breaks its authored
one-cell / 16-cells-per-frame ratios; current values and an intact/modified flag
are recorded in the next bake's metadata.

Outflows have two explicit modes. **Volume Sink** removes particles inside a
mesh and is useful anywhere in the domain. **Pressure Outlet** opens only the
covered exterior domain faces at atmospheric pressure and removes particles
after they cross; its mesh must intersect a domain boundary. The latter is
the closer analog for the paper's bottom drain. Use **Whirlpool** beside the
quick setup button for a clearly labeled, low-resolution preview constrained
by the paper's published dimensions and `0.1 rad/s` initial rotation.

### Workflow fit

| Workflow | Practical support | Boundary |
|---|---|---|
| Cropped environment puddles and standing water | GPU/CPU bakes, static cut-cell obstacles, resumable caches, high global CFL, and Blender surfacing | Uniform dense grid and Python scene voxelization make full landscapes impractical; build a shot-sized interaction domain |
| Rain, leaks, and jets | Static inflow volumes support uniform or solid-body velocity plus an optional inclusive evolved-output frame range; the High-CFL Jet Preview demonstrates a high-speed static collision | Refill is occupancy based, not a physical flow-rate or pressure/head boundary; no droplet distribution, viscosity, wetting, surface tension, or air coupling |
| Vortex and sci-fi prop look development | Solid-body rotation can initialize liquid or drive a rotating inflow; the whirlpool setup reproduces the published rotation field and dimensions at preview scale | No torque model or rotating propeller boundary; this is a visual-effects flow field, not engineering validation |
| Moving tire or mechanism interaction | A stationary proxy can be a fractional-aperture obstacle | Animated/deforming obstacles and moving-wall velocity are not implemented; a moving tire splash is therefore not a supported claim |
| AI/procedural detail enhancement | A portable playback handoff exports committed world-space particle positions, velocities, settings, units, timing, hashes, and optional metrics | No stable particle IDs, foam/spray labels, training pairs, inference model, or generated microdetail; temporal AI consistency remains a downstream responsibility |

This distinction is intentional. Re-voxelizing a moving tire without imposing
its boundary velocity in collision response and pressure projection would be
nonphysical, so the add-on does not present static-obstacle support as moving
solid support.

### Solver settings

| Setting | Paper ref | Meaning |
|---|---|---|
| Target CFL | Algorithm 1, §4 | Global step target from 0.5–30; paper examples use values through 30 |
| Particles / Cell | §4.5 | Initial samples per occupied cell, 1–64; paper sweeps 1–16 against a 50-particle reference |
| Spatiotemporal Sampling | §3 | Disable temporal weighting and jitter for an instantaneous-P2G ablation; this is not a full standard-FLIP/GFM baseline |
| Jitter Strength (γ) | Eq. 10 | Base temporal jitter amplitude; 1 permits full-slab jitter before adaptive attenuation |
| Adaptive Attenuation | §3.10 | Less jitter noise on calm surfaces |
| Interface Steepness (η) | Eq. 13 | Lower = steeper/stronger leveling; higher = finer detail/noise |
| Transfer | §3.9 | Velocity transfer: FLIP (detail, noisier), APIC (low-dissipation, smooth), PIC (very smooth) |
| FLIP Fraction | §4 | FLIP/PIC blend (default 0.98); FLIP transfer only |
| Two-Phase (Gas) + Gas Density | §3.1, 3.6 | Couple a light gas so air drives splashes and rising bubbles (glugging) |
| Surface Tension (σ) | §3.9 | CSF surface tension; small-scale, needs high resolution |
| Sparse Grid | — | Crop the solver to the active fluid region each step |
| Whitewater | §4.9 | Foam/spray/bubble secondaries; with Two-Phase, spray and bubbles ride the simulated air field |
| Random Seed | §3.10 | Reproducible particle placement and temporal jitter; seed 0 is an add-on default, not a published paper value |
| Initial / Inflow Velocity | §4.8 | Liquid and inflow sources can use uniform or solid-body rotational fields; inflows can be limited to an inclusive scene-frame range |
| Outflow Mode | §4.8 | Interior particle-removal volume or exterior half-cell `p=0` pressure outlet |
| Advanced Solver | §3.3, §3.7 | Liquid density, local advection CFL, PCG tolerance/limit, and relative density floor |
| Pressure Solver | §3.6–3.7 | PPE preconditioner: Jacobi-PCG (default) or a geometric multigrid V-cycle whose iteration count is nearly resolution-independent; multigrid falls back to Jacobi on grids too small to coarsen |
| Materials & Look | — | One-click fluid materials (water, clear, honey, juice, milk, lava) plus a Studio Look setup (EEVEE Next raytracing + sky world + sun) so a bake renders with real refraction immediately |
| Paper MCF Surface | Appendix B | Fixed radius/voxel `0.5Δx`, Gaussian `σ=2Δx`, feature mask `θ=2, ζ=5`, and `0.5` isovalue; `kψ` defaults to 30 |

### Source velocity fields

For each **Liquid** or **Inflow** object, choose **Solid Body Rotation** to
sample the actual jittered particles with
`u(x) = v_linear + omega × (x - center)`. The center and axis are entered in
Blender world coordinates. The axis is normalized, the signed angular speed
is in radians per scene second, and positive speed follows the right-hand
rule. The linear velocity is superposed on the rotation. Field sampling uses
a deterministic host-float32 path, so a fixed seed produces the same initial
particle positions and velocities on CPU and CUDA.

An inflow may also be restricted to inclusive evolved **Start Frame / End
Frame** outputs. The cache's first frame is a pre-step snapshot; subsequent
frames receive emission during their preceding simulation interval. Scheduling
is checkpoint-safe because it is derived from the restored solver clock; it
does not turn the occupancy refill source into a prescribed volume-flow
boundary.

With the paper's vertical axis mapped through Blender's world origin, its
whirlpool initialization is represented by center `(0, 0, 0)`, axis `+Z`,
zero linear velocity, and angular speed `0.1 rad/s`. This reproduces the
published initial velocity field. Version 0.7 also provides an approximate
whirlpool preview with the published `200 x 200 x 80 m` domain proportions
and an authored `20 m` diameter by `10 m` bottom outlet mesh. The solver uses
the mesh's circular boundary footprint; its displayed `10 m` length is
reference geometry, not a simulated conduit. Fill height, resolution,
duration, and unpublished production details are explicit preview choices,
so this is not presented as an exact reproduction.

### Experiment profiles and diagnostics

The **Experiment Diagnostics** panel provides parameter-only, paper-inspired
profiles. They apply reproducible solver settings to the current Blender
scene; they do **not** recreate the paper's geometry, reference solvers, or
surface-error evaluation datasets:

- Laminar dam break (§4.1): target CFL 1, 3, 5, 10, or 20.
- Standard dam break (§4.3): target CFL 1, 2, 4, 8, or 16, plus explicitly
  labeled instantaneous-P2G ablations at CFL 1 and 16 that are not standard
  FLIP/GFM.
- Enstrophy (§4.4): ST-FLIP analogs for `(CFL, FLIP)` pairs `(1, .99)`,
  `(5, .99)`, `(10, .99)`, `(1, .95)`, and `(1, .90)`. The paper's CFL 1
  comparison curves use the unavailable standard-FLIP/GFM solver.
- Particle count (§4.5): the ST-FLIP/CFL 10 branch at 1, 2, 4, 8, or 16
  particles/cell plus the 50-PPC reference. The paper's standard-FLIP/CFL 1
  branch is unavailable.

Enable **Record Frame Metrics** before baking to append strict schema-v2
records to `stflip_metrics.jsonl` in the cache. The export button produces an
atomic, self-contained CSV or JSON file. Evolved output frames record
solver-only wall time, particle-free idle time, time-step and observed
particle-CFL summaries, PCG
iteration/residual summaries, particle and outflow-removal counts, speed,
center of mass, momentum,
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

The open **Downstream Export** panel exposes **Export Playback Handoff**, which
writes an atomic ZIP for downstream procedural or AI-assisted enhancement. It
contains only committed, validated playback NPZ
frames (`positions`, `velocities`), a strict versioned manifest, per-frame
SHA-256 hashes and counts, Blender world-space/unit/timing declarations,
sanitized solver settings, and validated metrics when available. Raw resume
checkpoints are deliberately excluded. The manifest explicitly declares that
stable particle IDs, foam/spray labels, inferred microdetail, and an AI model
are absent, so consumers cannot mistake a transport package for a turnkey
detail generator.

### Paper coverage

Version 0.9 extends the free-surface ST-FLIP core with **two-phase gas
coupling, APIC/PIC transfers, CSF surface tension, animated moving-wall
boundaries, and a sparse active-block production grid** (see the table below);
it is still not a full reproduction of every production example in the paper
(billion-particle scale, exact scene geometry, GFM baselines).

| Paper capability | Status in this add-on |
|---|---|
| One-sided temporal kernel, slab P2G, residual jitter, phase field, variable-coefficient projection | Implemented |
| Large target CFL and instantaneous-P2G temporal ablation | Implemented; target CFL 0.5–30 |
| Reproducible reruns over paper-inspired CFL, particle-count, and FLIP/PIC parameter matrices | ST-FLIP-side profiles, auditable frame diagnostics, and a matched four-case batch validator are implemented; no true FLIP/GFM branches or exact paper scenes |
| Single-phase liquid scenes with static mesh obstacles, inflows, and outflows | Implemented; inflows support uniform/rotational fields and active frame ranges; outflows support volume sinks and exterior atmospheric-pressure faces |
| Fractional solid face apertures in Eq. 14–17 | Implemented; a moving-wall solid-velocity flux term `div((1-alpha) u_solid)` couples animated obstacles into the same projection |
| Paper render reconstruction (`0.5Δx` spheres, 2× grid, feature mask, MCF, `0.5` iso) | Implemented as an opt-in, versioned derived mesh cache; the paper does not prescribe the subvoxel rasterizer, so this implementation records its linear one-voxel coverage ramp explicitly |
| Two-phase liquid/gas coupling | Implemented (v0.9): liquid volume-fraction phase field `m_l/(m_l+m_g)`, variable-density projection over both phases, gas seeding (`fill_gas`/`add_gas_mask`) |
| APIC / PIC transfers | Implemented (v0.9): per-particle affine matrix with the same temporal weighting, MAC-grid `B·D⁻¹` reconstruction; implicit-density-projection comparison solver still not implemented |
| Surface tension (CSF) | Implemented (v0.9): curvature from a cubic-B-spline-smoothed phase field, `σ·κ·∇φ` face acceleration |
| Sparse/adaptive grids | Implemented (v0.9) as a block-aligned active-window crop (bitwise-identical to dense); billion-particle production scale still out of reach on a single workstation |
| Appendix B mean-curvature-flow output reconstruction | Implemented with default `kψ=30`; OpenVDB extracts the render mesh and Fast Preview remains a separate approximation |
| Animated/deforming obstacle boundary conditions | Implemented (v0.9) for rigid moving walls: per-object "Animated (Moving Wall)" toggle re-voxelizes each output frame with a differenced rigid velocity (`set_solid_sdf(..., solid_vel=...)`); walls push and separate from fluid without tunneling. Deforming solids remain unsupported |

Experiment-level coverage is narrower than method-level coverage:

| Paper experiment | Reproduction level |
|---|---|
| Laminar/standard dam breaks over large CFL values | Partial: parameter profiles, qualitative setup, and raw diagnostics, without exact geometry, GFM/APIC baselines, or the paper's unspecified normalization |
| Static-obstacle wake and thin-obstacle inflow jet | Partial: an explicitly approximate High-CFL Jet Preview exercises a scheduled high-speed inflow, thin stationary cut-cell obstacle, and pressure outlet; the paper's exact geometry and jet parameters are unpublished |
| Particle-count and FLIP-blend studies | ST-FLIP-side parameter profiles, deterministic seed, and enstrophy diagnostic are available; true FLIP/GFM branches, SDF RMSE, and batch-sweep tooling are not |
| Kleefsman obstacle validation | Missing water-height gauges and experimental-data comparison |
| MCF reconstruction study | MCF is implemented; the paper's reference surfaces and normal-RMSE evaluation dataset are unavailable |
| Whirlpool, rotational fields, and outflow scenes | Partial: published rotation and pipe/domain dimensions are available in an approximate preview with a pressure outlet; exact fill height, timing, production scale, and rendering remain unpublished/unreproduced |
| Two-phase glugging/discharge and production-scale scenes | Missing two-phase solver and sparse production grid |

The Blender UI exposes these inputs for the implemented single-phase solver:
resolution, target CFL, particles/cell, `γ`, adaptive attenuation, `η`,
FLIP/PIC blend, seed, backend, gravity, frame rate/range, per-source uniform
or solid-body velocity (linear velocity, world center/axis, and signed angular
speed), optional inflow active frames, both outflow modes, density,
local CFL, PCG controls, and the relative density floor. Surface controls
include deterministic Geometry Nodes preview settings plus a paper-MCF mode
with iterations, OpenVDB adaptivity, a dense-grid voxel guard, cancellable
full-cache rebuild, and live cached-frame refresh. Blender Laplacian smoothing
remains preview-only and is deliberately not labeled as paper MCF. Every
bake records these settings in its cache metadata. The solver's Python
`Params` API additionally exposes liquid density, pressure tolerance/iteration
limit, local advection CFL, the under-sampling threshold, and the relative
density floor. Paper-stated defaults include `γ = 1`, `ηφ = 0.5`,
`αFLIP = 0.98`, local CFL `= 1`, and PPE tolerance `= 1e-4`; the add-on also
uses `ρ = 1000` and a 400-iteration PCG limit. Paper MCF defaults to `kψ=30`.
The Solver panel additionally exposes the velocity transfer scheme
(FLIP/APIC/PIC), two-phase gas coupling with gas density and gas particles/cell,
the surface-tension coefficient, and the sparse-grid toggle; inflows gain an
"Emit Gas" option in two-phase mode, and obstacles an "Animated (Moving Wall)"
toggle that re-voxelizes them per output frame with a differenced rigid
velocity (also scriptable via `set_solid_sdf(..., solid_vel=...)`).

## What's implemented

- MAC-grid FLIP/PIC with separable poly6 spatial kernel (Eq. 18) and
  one-sided temporal kernel (Eq. 19)
- 4D→3D slab-integrated P2G with weight-accumulator phase field (Eq. 8–13)
- Variational variable-coefficient pressure projection, matrix-free
  Jacobi-PCG (Eq. 14–17), including node-SDF-derived fractional solid face
  apertures, `div(alpha * u)`, and aperture-weighted PPE coefficients
- Temporal jitter with residual carryover and the Appendix A boundedness
  guarantee (tested), adaptive γ attenuation
- Globally adaptive time stepping with even frame subdivision and strict
  grid-advector-bounded RK3 substeps (local CFL defaults to 1)
- Particle re-synchronisation (un-jittering) at output frames
- Appendix-B render reconstruction with a twice-resolution cropped density,
  fixed Gaussian/self-quotient feature mask, 30-iteration default level-set
  mean-curvature flow, strict derived mesh caches, and OpenVDB `0.5` isosurface
- Stationary solid obstacles via cell/node-sampled voxelised SDF, fractional
  face apertures, and particle push-back; optionally scheduled inflow emitters
- Uniform and right-handed solid-body liquid/inflow velocity fields, sampled
  at actual jittered particle positions with deterministic CPU/CUDA setup
- Interior volume sinks and exterior half-cell atmospheric-pressure outlets,
  including synchronized particle-state removal and auditable metrics
- Scene-owned caches with missing/foreign-cache reconciliation, explicit bake
  lifecycle/progress/cancellation, stale-output clearing, and strict resumable
  per-frame solver checkpoints
- **Two-phase gas coupling** (v0.9, §3.1/3.6–3.7): liquid volume-fraction phase
  field, variable-density projection over both phases, gas seeding — glugging,
  rising bubbles, air-driven spray
- **APIC and PIC transfers** (v0.9, §3.9): per-particle affine matrix with the
  same temporal weighting; batched analytic 3×3 inverse (no `cupy.linalg`)
- **Viscosity** (v0.13): implicit (unconditionally stable) diffusion solve for
  oil/honey/lava — thickness that survives ST-FLIP's large time steps
- **Surface tension** (v0.9, §3.9): CSF from a B-spline-smoothed phase field
- **Animated moving-wall obstacles** (v0.9): per-cell solid velocity enters the
  projection as a `(1-alpha) u_solid` flux; near-wall particles shed only the
  penetrating normal velocity, so walls push yet fluid still separates
- **Whitewater** (v0.10): foam/spray/bubble secondary particles emitted from
  energetic interface regions; with Two-Phase on, spray and bubbles are driven
  by the simulated air velocity field -- the paper's stated purpose for
  two-phase simulation (§4.9). Streams to a `STFLIP Whitewater` point cloud
  with `velocity`/`ww_kind`/`ww_life` attributes
- **Vectorized voxelization** (v0.10): masks via NumPy ray-parity and obstacle
  SDFs exact in a narrow surface band (256³ masks in seconds instead of
  minutes); the BVH loop remains as an automatic fallback
- **Sparse production grid** (v0.9): every step crops to a block-aligned active
  window (fluid + extrapolation band), bitwise-identical to the dense solve;
  disengages when outflows or cut-cell node-SDF solids are present
- **Shading attributes** on the particle point cloud: `velocity`, `speed`,
  `age` (seconds since seeding), and `source` (per-source id) for age-fade,
  speed-driven effects, and per-source colouring in shaders/Geometry Nodes
- **Alembic / USD export** of the baked liquid surface (and optionally the
  particle/whitewater clouds) as an animated cache for render farms and
  other DCCs, from the ST-FLIP > Downstream Export panel or
  `bpy.ops.stflip.export_cache(filepath=...)` headless
- **Motion blur setup**: one click enables render motion blur and samples
  per-particle velocity onto the topology-changing surface (Geometry Nodes)
  so the surface and point clouds motion-blur; surface deformation blur
  needs Cycles
- **Force fields / guides** (v0.16): a `Force Field` object role adds
  directional (wind), vortex, or divergence-free curl-noise turbulence body
  forces for art-directable flow (object +Z = axis, origin = centre)
- **Particle sheeting / anti-clumping** (v0.17): a position-only nudge that
  spreads genuinely over-dense clumps (density-gated so it never inflates
  the free surface) to keep thin splashes and sheets intact; adds no energy
- NumPy CPU + CuPy CUDA backends sharing one code path
- Paper-inspired parameter profiles plus strict JSONL frame diagnostics and
  atomic CSV/JSON export; optional discrete MAC-grid enstrophy
- Matched four-case ST/instantaneous CFL 1/16 validation with observed-CFL,
  quality, state-hash, residual, and separately scoped timing evidence
- Atomic playback-only downstream handoff with a strict manifest and explicit
  absence declarations for IDs, foam/spray labels, inferred detail, and AI

Not (yet) implemented from the paper: the standard-FLIP/GFM and
implicit-density-projection comparison baselines, and billion-particle
production-scale scenes.

Known limitations: two-phase and the sparse grid do not combine usefully (gas
fills the domain, so the active window is the whole grid), and the sparse
window also disengages when outflows or cut-cell node-SDF obstacles are present.
The disk resume checkpoint (schema v2) persists the gas tag, APIC affine matrix,
and shading attributes, so a resumed two-phase/APIC bake continues without a
discontinuity; older v1 checkpoints still load and fall back to single-phase
liquid with a fresh affine field. Obstacles marked
"Animated (Moving Wall)" are re-voxelized every output frame with a differenced
rigid velocity (slower; resume re-samples their motion from the current frame);
source/outlet and static-obstacle geometry is still voxelized once at the
first frame. Scene voxelization (mesh → grid masks/SDF) is vectorized in NumPy:
ray-parity inside tests and the banded signed-distance field both walk a
Morton (Z-order) curve so their per-chunk triangle bounding-box prefilters prune
to nearby triangles (a ray or band cell only ever meets a handful), with the
Blender BVH loop kept only as an automatic fallback; the residual per-frame cost
for animated obstacles is the band point-to-triangle distance. Temporal jitter
randoms are drawn on the host for cross-backend
determinism, costing a small per-step transfer on GPU. Exact resume checkpoints
are intentionally uncompressed and can be substantially larger than playback
frames. Paper surfacing uses a dense cropped grid with `O(kψ V)` work and can
also be memory-heavy; it is output-only, guarded by a field-size cap plus a
conservative preflight, and does not change the solver trajectory.

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

`bpy.ops.stflip.resume_bake()` is also synchronous in an EXEC context. It
requires an owned v0.8 checkpoint, unchanged simulation inputs, and a scene End
frame later than the latest committed frame.

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
