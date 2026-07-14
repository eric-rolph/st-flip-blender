# Getting started

## 1. Install the add-on

ST-FLIP is a Blender **extension** for Blender 4.2+. This guide follows the
current v0.23.1 source; the latest public package is still v0.11.0 and lacks
features added later.

Build current source or install the older release's `.zip` via *Edit →
Preferences → Get Extensions → Install from Disk*. See the top-level
[README](../README.md#install) for the exact choices.

Finish resumable bakes before updating. Resume requires the same add-on version
that created the checkpoint.

Once enabled you get a **ST-FLIP** tab in the 3D Viewport sidebar
(press <kbd>N</kbd> to open the sidebar).

## 2. (Optional) install GPU acceleration

The solver runs on CPU (NumPy) out of the box. For an NVIDIA GPU, open
**ST-FLIP → Solver** and, when CUDA compute is unavailable, press **Install GPU
Support (CUDA)**. The panel reports the detected device after setup.

GPU support is optional. The same features run on CPU, usually more slowly at
production resolutions.

See [performance-and-scaling](performance-and-scaling.md) for what the GPU
actually buys you.

## 3. Bake your first splash (the fast path)

The quickest possible result:

1. In the **ST-FLIP** panel, click **Quick Dam-Break Setup**, or open the
   **Presets** sub-panel and pick **Stormy Pool**. Either builds a complete,
   ready-to-bake scene: a Domain box and the fluid objects.
2. Save the `.blend` file. The default cache path is blend-relative, so an
   unsaved file is refused rather than silently writing to a temp location.
   (Alternatively set an absolute **Cache Directory** and skip saving.)
3. Press **Bake Simulation**. A progress bar tracks the bake; frames stream to
   disk as they finish.
4. Scrub the timeline. The fluid plays back from the cache.

That is the whole loop: **setup → save → bake → scrub**.

## 4. Build a scene by hand

When you want your own geometry instead of a preset:

1. Add a cube, scale it to enclose your scene, and with it selected set
   **Active Object → Role = Domain** (or use the Domain picker in the main
   panel). The Domain defines the simulation grid.
2. For a voxelized role, add a closed mesh inside or intersecting the Domain as
   the role requires, then choose:

   - **Liquid** — starts full of fluid (a body of water).
   - **Inflow** — emits fluid over time (a faucet), optionally with a velocity
     and a limited frame range.
   - **Obstacle** — a solid the fluid flows around.
   - **Outflow** — a volume drain, or a pressure outlet intersecting a Domain
     boundary.
3. For a **Force Field**, add a mesh or Empty; it is not voxelized. Directional
   uses +Z, vortex uses origin/+Z, and turbulence is domain-wide.
4. Set **Resolution** (cells along the longest Domain axis) and **Target CFL**
   (start at 8). Higher resolution = more detail and more cost.
5. **Bake Simulation**.

The full role system is covered in [object-roles](object-roles.md).

## 5. Resume, cancel, and free

- **Bake** runs from the scene Start frame. To extend a finished or cancelled
  bake, raise Scene End and press **Resume**. It restores the last committed
  primary-solver checkpoint.
- **Cancel** stops a running bake; the frames already written stay valid and
  are resumable.
- **Free** (trash icon) clears the cache for a fresh start.

Resume requires a scene-owned schema-v2 checkpoint from the same add-on
version, the same backend, a matching simulation fingerprint, and Scene End
later than the last committed frame.

Output-only surface and material controls may change; changed Paper MCF
settings require a surface-cache rebuild. Whitewater restarts after Resume,
and animated obstacle motion is re-sampled.

## Next steps

- Learn the roles: [object-roles](object-roles.md)
- Get a specific look: [recipes](recipes.md)
- Tune detail/behaviour: [settings-guide](settings-guide.md)
