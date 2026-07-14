# Troubleshooting

## Bake / setup

**"Save the file first" / bake refuses to start.**
The default cache path is blend-relative, which needs a saved `.blend`. Either
save the file, or set an absolute **Cache Directory** and bake without saving.

**Resume refuses to continue ("inputs changed").**
Resume requires a scene-owned schema-v2 checkpoint from the same add-on version,
the same backend, a matching simulation fingerprint, and Scene End later than
the last committed frame.

Among simulation inputs, change only Scene End. Output-only surface, material
and display controls may change; changed Paper MCF settings require a surface-
cache rebuild. Otherwise **Free** the cache and bake fresh.

Animated obstacle motion is re-sampled after Resume and may differ slightly.
Whitewater state is not checkpointed, so whitewater restarts even though the
primary fluid resumes from its committed state.

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
Open **ST-FLIP → Solver** and press **Install GPU Support (CUDA)** when CUDA is
unavailable. A compatible NVIDIA GPU and driver are required; otherwise the
solver keeps running on CPU.

On some setups the CuPy runtime path must be short. The installer places it in
a shallow per-user directory automatically.

**Cache is enormous.**
Primary-solver checkpoints are uncompressed and dominate cache size at high
particle counts. If you do not need Resume, compressed playback frames are
enough for playback and re-surfacing. Plan disk for long, high-resolution bakes.

## Still stuck?

Check the top-level [README](../README.md) for paper coverage and known
limitations, and the [settings guide](settings-guide.md) for what each control
actually does.
