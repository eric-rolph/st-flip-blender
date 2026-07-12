# Troubleshooting

## Bake / setup

**"Save the file first" / bake refuses to start.**
The default cache path is blend-relative, which needs a saved `.blend`. Either
save the file, or set an absolute **Cache Directory** and bake without saving.

**Resume refuses to continue ("inputs changed").**
Resume compares the current scene against the checkpoint's fingerprint and
refuses if a trajectory-defining input changed — Domain size, Resolution, scene
FPS, outlet modes, geometry, or the compute backend (CPU↔GPU). To extend a
bake, change *only* the scene End frame, then Resume. To change anything else,
**Free** the cache and bake fresh.

**A source covers no cells / nothing emits.**
The object must be a **closed mesh** that actually overlaps Domain cells at the
current Resolution. Very thin or tiny objects can fall between cells — thicken
them or raise Resolution. Check the object's Role is set.

**Fluid immediately drains or never fills.**
Check your **Outflow**: a Volume Outflow deletes everything inside it, and a
Pressure Outflow must intersect a Domain boundary to act as an opening. A
continuous **Inflow** with no Outflow will simply fill the Domain.

## Look / behaviour

**Water looks like blobs, not a surface.**
That is the raw particle preview. Assign a **Surface Method** and shade the
generated surface — see [rendering-and-export](rendering-and-export.md). For
final quality use **Paper MCF**.

**Splashes tear into chunky blobs.**
Raise **Resolution**, and add some **Sheeting** (`0.3–0.6`) to hold thin sheets
together. **FLIP** transfer preserves splash energy better than PIC.

**Motion is too noisy / jittery.**
Lower the **FLIP Fraction** (more PIC smoothing) or switch **Transfer** to
**APIC**. **Adaptive Attenuation** also calms flat surfaces.

**Motion is too smooth / mushy.**
The opposite: use **FLIP** with a high FLIP Fraction (0.95–0.98), and make sure
Resolution is high enough to carry the detail.

**Surface tension / beading does nothing.**
It is a small-scale effect; it only shows at high Resolution. On a coarse grid
you will not see droplets bead.

**Two-Phase with Sparse Grid seems to ignore Sparse.**
Correct — gas fills the whole domain, so there is no localized active region to
crop to, and Sparse disengages. They are not meant to combine.

## Performance

**Bake is very slow.**
Work through the checklist in
[performance-and-scaling](performance-and-scaling.md): lower Resolution for
look-dev, install the **GPU**, enable **Multigrid-PCG** on large grids, enable
**Sparse Grid** for localized flows, and raise **CFL**.

**GPU not detected.**
Reinstall from the **ST-FLIP → GPU** panel. The runtime needs a compatible
NVIDIA GPU and driver; if detection still fails, the solver keeps running on
CPU. (On some setups the CuPy runtime path must be short — the installer places
it accordingly.)

**Cache is enormous.**
Exact solver checkpoints are uncompressed so a bake can resume exactly; they
dominate cache size at high particle counts. If you do not need to resume, you
can rely on the compressed playback frames. Plan disk for long, high-resolution
bakes.

## Still stuck?

Check the top-level [README](../README.md) for paper coverage and known
limitations, and the [settings guide](settings-guide.md) for what each control
actually does.
