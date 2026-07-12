# Getting started

## 1. Install the add-on

ST-FLIP is a Blender **extension** (Blender 4.2+). Install the packaged
`.zip` via *Edit → Preferences → Get Extensions → Install from Disk*, or build
it from the repository (see the top-level [README](../README.md#install)).

Once enabled you get a **ST-FLIP** tab in the 3D Viewport sidebar
(press <kbd>N</kbd> to open the sidebar).

## 2. (Optional) install GPU acceleration

The solver runs on CPU (NumPy) out of the box. For a large speed-up on an
NVIDIA GPU, open the **ST-FLIP → GPU** panel and press **Install GPU Support**.
This provisions a CuPy runtime the add-on can import; the panel then reports the
detected device. GPU is optional — everything works on CPU, just slower.

See [performance-and-scaling](performance-and-scaling.md) for what the GPU
actually buys you.

## 3. Bake your first splash (the fast path)

The quickest possible result:

1. In the **ST-FLIP** panel, click **Quick Setup** (dam-break), or open the
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
2. Add another closed mesh inside the Domain and set its **Role**:
   - **Liquid** — starts full of fluid (a body of water).
   - **Inflow** — emits fluid over time (a faucet), optionally with a velocity
     and a limited frame range.
   - **Obstacle** — a solid the fluid flows around.
   - **Outflow** — a drain that removes fluid.
   - **Force Field** — an art-directed wind/vortex/turbulence body force.
3. Set **Resolution** (cells along the longest Domain axis) and **Target CFL**
   (start at 8). Higher resolution = more detail and more cost.
4. **Bake Simulation**.

The full role system is covered in [object-roles](object-roles.md).

## 5. Resume, cancel, and free

- **Bake** runs from the scene Start frame. To extend a finished or cancelled
  bake, raise the scene End frame and press **Resume** — the solver restarts
  from the exact checkpoint of the last committed frame.
- **Cancel** stops a running bake; the frames already written stay valid and
  are resumable.
- **Free** (trash icon) clears the cache for a fresh start.

Resume refuses to continue if trajectory-defining inputs changed (geometry,
resolution, outlet modes, the compute backend, …) so you can never silently
splice two different simulations together.

## Next steps

- Learn the roles: [object-roles](object-roles.md)
- Get a specific look: [recipes](recipes.md)
- Tune detail/behaviour: [settings-guide](settings-guide.md)
