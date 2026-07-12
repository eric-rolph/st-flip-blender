# Recipes

Concrete scenes. The four **Presets** (in the *ST-FLIP → Presets* sub-panel)
build these for you in one click; each section below also shows how to build the
same thing by hand so you can adapt it.

Every preset builds a fresh Domain and objects, sets a sensible frame range,
and configures exactly the solver features it needs — switching presets resets
feature settings so a viscous pour never leaves a later fountain sticky.

---

## Viscous Pour (honey, lava, thick paint)

**Preset:** *Viscous Pour*. A downward **Inflow** of thick fluid piling onto a
floor and folding over itself (coiling).

Build it by hand:

1. Domain box; an **Inflow** near the top with a downward velocity.
2. **Solver → Transfer = APIC** (low dissipation keeps the coil crisp).
3. **Viscosity** ≈ `0.05–0.1` for honey; higher for lava/tar. Viscosity is
   solved implicitly, so it stays stable even at high CFL.

Coiling and folding need resolution — expect to raise **Resolution** for a
clean rope of fluid.

---

## Stormy Pool (choppy water, wind-driven surface)

**Preset:** *Stormy Pool*. A **Liquid** pool agitated by a **Turbulence**
force, throwing spray.

Build it by hand:

1. Domain box; a **Liquid** slab filling the lower third.
2. Add an Empty or mesh as a **Force Field**, **Force Type = Turbulence**,
   raise **Strength** (~8) and set a **Scale** for the eddy size.
3. Enable **Whitewater** for spray/foam on the crests.
4. A little **Sheeting** (~0.5) keeps thin splash sheets from tearing into
   blobs.

---

## Two-Phase Glug (bottle emptying, bubbling)

**Preset:** *Two-Phase Glug*. Air is simulated as a second phase, so it can
push back up through liquid as bubbles — the "glug-glug" of a draining bottle.

Build it by hand:

1. Domain box; a **Liquid** volume with a downward **Inflow** or a narrow neck.
2. Enable **Two-Phase (Gas)** in the Solver panel. This fills the empty domain
   with gas particles.
3. Optionally mark an **Inflow** as **Emit as gas** to inject air.

Two-Phase does **not** combine with the Sparse grid (gas fills the whole
domain, so there is no localized active region to crop to).

---

## Fountain (jet + drain)

**Preset:** *Fountain*. An upward **Inflow** jet with a **Volume Outflow** at
the base so the domain reaches a steady state instead of overflowing.

Build it by hand:

1. Domain box; an **Inflow** at the bottom with a strong upward velocity.
2. A **Volume Outflow** around the base to recycle the water that falls back.
3. Enable **Whitewater** for the spray plume.

The drain is what keeps a continuous source from simply filling the box — any
long-running inflow scene needs an outflow.

---

## Whirlpool / draining vortex

Use the built-in **Whirlpool Preview** button (an *approximate* preview of the
paper's published whirlpool dimensions with a pressure outlet). It sets domain,
rotation, and a `p = 0` drain for you.

To approximate one by hand: a **Liquid** pool given a **solid-body rotation**
initial velocity about the vertical axis, plus a **Pressure Outflow** at the
centre of the floor. Because a true production whirlpool is huge, treat this as
a look-dev preview rather than a reproduction — see
[performance-and-scaling](performance-and-scaling.md).

---

## Where to go next

- Not enough detail, or the wrong *feel*? → [settings-guide](settings-guide.md)
- Bake too slow? → [performance-and-scaling](performance-and-scaling.md)
- Water looks like blobs, not a surface? → [rendering-and-export](rendering-and-export.md)
