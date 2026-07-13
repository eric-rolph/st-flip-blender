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

## Shading the water — the fast path

Open **ST-FLIP → Materials & Look**:

1. Pick a **Fluid Material** — Water, Clear, Honey, Juice, Milk, or Lava — and
   press **Apply Fluid Material**. The add-on builds a physically-tuned shader
   and assigns it to the managed surface object (it persists across the
   per-frame surface rebuild).
2. Press **Setup Studio Look**. This configures EEVEE Next raytracing, a sky
   world, and a sun so the fluid renders with real refraction immediately —
   no hand-built node graph or lighting needed. It is safe to run more than
   once, and there is an option to drop in a neutral ground plane.

The refractive materials (Water, Clear, Honey, Juice) need scene raytracing,
which **Setup Studio Look** turns on; the panel reminds you when the selected
material is refractive. Milk uses subsurface scattering; Lava is emissive and
glows on its own. Presets pick a matching material automatically (for example
Viscous Pour selects Honey).

## Shading the water — by hand

If you would rather build your own shader, assign a glass/water material to the
managed surface object (so it survives the per-frame rebuild): a
transmissive/refractive Principled BSDF with a light base-colour tint,
`IOR ≈ 1.33`, low roughness, and `use_raytrace_refraction` enabled, plus
`scene.eevee.use_raytracing = True`. EEVEE Next raytraced refraction traces
screen buffers, so tint via the Base Color rather than a Volume Absorption node
(which is unreliable through refraction on the thin per-frame surface).

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
3. In **Materials & Look**, pick a Fluid Material, **Apply** it, and press
   **Setup Studio Look**. Add **Motion Blur** if the fluid moves fast.
4. Render — or **Export** an Alembic/USD cache and render elsewhere.

Keep the solver cache: you can re-surface and re-render endlessly without
re-baking.
