# Rendering and export

A bake produces **particles**. To render water you turn those particles into a
**surface**, then shade it. This page covers both, plus motion blur and
exporting to other applications.

## Surface reconstruction methods

Set **Surface Method** (in the surface panel):

- **Fast Preview** — a deterministic Geometry-Nodes points-to-volume surface.
  Interactive and cheap; regenerates as you scrub. Use it for look-dev, layout,
  and checking motion. It is *not* the paper's reconstruction.
- **Paper MCF** — the paper's Appendix-B pipeline: a dense union-of-spheres
  reconstruction, mean-curvature-flow smoothing, then CPU OpenVDB meshing. This
  is the high-quality, feature-preserving surface for final frames. It is
  heavier (dense grid + OpenVDB), so it is output-only and guarded by size
  caps — reserve it for the resolutions and frames you will actually render.

Tuning knobs:

- **Particle Radius** — the surfacing sphere radius in cell widths. Larger =
  smoother, more "filled"; smaller = more detail and more gaps.
- **Surface Voxel** — surfacing voxel size in cell widths. Smaller = finer mesh
  and more cost.
- **Geometric Smoothing** — an optional Laplacian-smooth modifier pass on top.

The Paper-MCF surface is cached per frame. Use **Rebuild Paper Surface Cache**
to (re)generate it for the whole range, and **Refresh Surface** to update the
current frame after changing settings.

## Shading the water

Assign a glass/water material to the generated surface object as you would any
mesh. Because the surface is regenerated per frame, put the material on the
surface object the add-on manages so it persists across frames.

For a believable look: a transmissive/refractive shader with a slight tint and
appropriate IOR, over a lit environment. Raytraced reflections/refractions
(EEVEE Next raytracing, or Cycles where available) give the convincing result;
the exact engine depends on your Blender build.

## Motion blur

Fluid moves fast, so motion blur matters. Use **Set Up Motion Blur** to
configure the scene and the surface object for velocity-based blur, so fast
splashes streak correctly instead of strobing.

## Exporting to another application

Use **Export Cache (Alembic/USD)** to write an animated mesh cache for Houdini,
Maya, Unreal, etc.:

- **Alembic (`.abc`)** — a widely supported animated-mesh cache.
- **USD (`.usdc`)** — Universal Scene Description cache.

Export bakes the reconstructed surface across the frame range into a single
animated cache file, decoupling downstream rendering from Blender and from the
solver cache.

## A clean final-frame workflow

1. Bake the simulation (see [performance-and-scaling](performance-and-scaling.md)).
2. Switch **Surface Method** to **Paper MCF** and **Rebuild Paper Surface
   Cache** for the render range.
3. Shade the surface; set up lighting and, if needed, **Motion Blur**.
4. Render — or **Export** an Alembic/USD cache and render elsewhere.

Keep the solver cache: you can re-surface and re-render endlessly without
re-baking.
