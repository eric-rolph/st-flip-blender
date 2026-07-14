# ST-FLIP for Blender — User Guide

Task-oriented guides for the ST-FLIP fluid add-on. The top-level
[README](../README.md) is the reference (paper coverage, install matrix, the
full settings table); the pages here walk you through *doing* things.

## Start here

1. **[Getting started](getting-started.md)** — install, enable the add-on,
   optionally install GPU support, and bake your first splash in five minutes.
2. **[Object roles](object-roles.md)** — how the Domain, Liquid, Inflow,
   Outflow, Obstacle, and Force roles fit together to define a scene.
3. **[Recipes](recipes.md)** — the one-click Presets explained, plus
   step-by-step builds for pours, fountains, air entrainment, viscous fluids,
   and whirlpools.

## Go deeper

4. **[Settings guide](settings-guide.md)** — what to actually turn when you
   want more detail, a thicker fluid, foam, air, or a faster bake.
5. **[Performance and scaling](performance-and-scaling.md)** — resolution vs.
   cost, GPU acceleration, the multigrid pressure solver, the sparse grid,
   headless baking, and cloud-rendering options.
6. **[Rendering and export](rendering-and-export.md)** — surface
   reconstruction modes, shading the water, motion blur, and exporting an
   Alembic/USD cache to another DCC.
7. **[Troubleshooting](troubleshooting.md)** — the failures you are most likely
   to hit, and how to fix each one.

## Design notes

- **[Tiled sparse grid](design/tiled-sparse-grid.md)** — the architecture for
  true tiled sparsity, plus the shipped active-box and axis-separable-region
  pressure crops that are its safe first increments.

## One-paragraph mental model

ST-FLIP is a FLIP fluid solver built to take **large time steps** (high CFL).
You mark a box as the **Domain**, assign scene objects their **roles**, pick a
**Resolution** and **Target CFL**, and **Bake**. Each baked frame writes both a
compressed playback frame and a primary-solver checkpoint to the **Cache
Directory**. Resume requires the same add-on version and matching simulation
inputs; whitewater state is not checkpointed.

Playback is driven by a frame-change handler. Rendering uses either a fast
Geometry-Nodes preview or the paper's mean-curvature-flow reconstruction.
