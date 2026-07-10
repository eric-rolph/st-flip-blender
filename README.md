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

Measured on a 64³ dam break (344k particles, 8 frames @ 24 fps):

| Configuration                  | s / frame | vs. baseline |
|--------------------------------|-----------|--------------|
| Plain FLIP, CFL 1, CPU         | 19.1      | 1×           |
| ST-FLIP, CFL 8, CPU            | 11.9      | 1.6×         |
| ST-FLIP, CFL 8, RTX 5090       | 0.67      | **28.6×**    |
| ST-FLIP, CFL 15, RTX 5090      | 0.54      | 35×          |

## Install

1. Download this repository as a ZIP (or `git clone` and zip the folder).
2. Blender → *Edit → Preferences → Get Extensions → Install from Disk…*
   (or *Add-ons → Install…* on 4.1) and pick the ZIP.
3. Enable **ST-FLIP Fluid**. Requires Blender **4.2+** (tested on 5.1).

### GPU acceleration (optional, recommended)

*ST-FLIP panel → Solver → Install GPU Support (CUDA)* pip-installs CuPy into
Blender's user modules (~100 MB download). Needs an NVIDIA GPU with a
CUDA 12/13 driver. Blackwell GPUs (RTX 50xx) are supported via PTX JIT.

AMD GPUs: CuPy's ROCm builds work on Linux (`pip install cupy-rocm-5-0` into
Blender's Python, untested); on Windows the CPU backend is used. A
vendor-neutral wgpu backend is on the roadmap.

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
| Spatiotemporal Sampling | §3 | Disable to compare with standard FLIP |
| Jitter Strength (γ) | Eq. 10 | Temporal jitter amplitude, 1 = full slab |
| Adaptive Attenuation | §3.10 | Less jitter noise on calm surfaces |
| Interface Steepness (η) | Eq. 13 | Lower = smoother phase interface |
| FLIP Fraction | §4 | FLIP/PIC blend (default 0.98) |

## What's implemented

- MAC-grid FLIP/PIC with separable poly6 spatial kernel (Eq. 18) and
  one-sided temporal kernel (Eq. 19)
- 4D→3D slab-integrated P2G with weight-accumulator phase field (Eq. 8–13)
- Variational variable-coefficient pressure projection, matrix-free
  Jacobi-PCG (Eq. 14–16)
- Temporal jitter with residual carryover and the Appendix A boundedness
  guarantee (tested), adaptive γ attenuation
- Globally adaptive time stepping with even frame subdivision, sub-stepped
  RK3 advection at local CFL ≤ 1
- Particle re-synchronisation (un-jittering) at output frames
- Solid obstacles via voxelised SDF with push-back; inflow emitters
- NumPy CPU + CuPy CUDA backends sharing one code path

Not (yet) implemented from the paper: two-phase air–liquid coupling, surface
tension (CSF), APIC transfers, sparse/adaptive grids, mean-curvature-flow
surface smoothing (Appendix B — the Geometry Nodes volume meshing stands in).

## Development

```bash
# CPU tests
uv run --no-project --with pytest --with numpy pytest tests -v
# GPU parity test (NVIDIA GPU required)
uv run --no-project --with pytest --with numpy --with cupy-cuda13x pytest tests -m gpu -v
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
