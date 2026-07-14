# Settings guide

This is the *practical* companion to the full settings table in the top-level
[README](../README.md#solver-settings). It answers "which knob do I turn?"

## The two you always set

- **Resolution** — cells along the Domain's longest axis. This is your single
  biggest quality/cost lever. Double it for finer detail and ~8× the work (grid
  is 3-D). Start low while you block out motion, raise it for the final bake.
- **Target CFL** — how far the fluid may move per step. ST-FLIP is built for
  **large** steps; 8 is a good default, and the paper uses values up to 30.
  Higher CFL = fewer, bigger steps (faster) but coarser time accuracy; lower
  CFL trades speed for accuracy.

## Detail vs. smoothness — Transfer

- **FLIP** — most energetic and detailed, but noisier. Good for violent
  splashes. Pair with **FLIP Fraction** (default 0.98) to dial in a little PIC
  smoothing.
- **APIC** — low-dissipation *and* smooth/stable. The best default for most
  swirly, coherent motion (vortices, coils). Used by the Viscous Pour preset.
- **PIC** — very smooth and dissipative; rarely what you want except for
  deliberately calm fluid.

## Making a thicker fluid — Viscosity

Raise **Viscosity** for oil, honey, lava, paint. It is solved implicitly
(backward-Euler), so it stays stable even at high CFL and high viscosity. `0`
is inviscid water. Try `0.05–0.1` for honey-like coiling.

## Foam and spray — Whitewater

Enable **Whitewater** to emit foam/spray/bubble secondary particles from
energetic interface regions, and scale emission with **Whitewater Rate**. With
**Two-Phase** on, spray and bubbles are driven by the actual simulated air
field instead of a heuristic.

## Simulating air — Two-Phase (Gas)

Enable **Two-Phase (Gas)** so air is a real second phase that can drive
splashes and rise as bubbles (glugging). Set **Gas Density** (air ≈ 1.2) and
**Gas Particles / Cell**. Note: Two-Phase fills the whole domain with gas, so
it does **not** combine with the Sparse grid.

## Keeping thin sheets alive — Sheeting

**Sheeting** is anti-clumping: it spreads over-dense particle clusters so thin
splash sheets and crowns do not tear into blobs. It is position-only (adds no
energy) and density-gated, so it will not inflate the free surface. Try
`0.3–0.6` for splashy scenes.

## Surface tension

**Surface Tension (σ)** adds a CSF surface-tension force for beading and thin
filaments. It is a *small-scale* effect and needs high resolution to show —
don't expect droplets to bead on a coarse grid.

## Pressure solver (speed at high resolution)

**Pressure Solver** chooses the preconditioner for the pressure projection:

- **Jacobi-PCG** (default) — fine at low/medium resolution.
- **Multigrid-PCG** — a geometric multigrid V-cycle whose iteration count is
  nearly independent of resolution. At production resolutions this is a large
  speed-up; on small grids it transparently falls back to Jacobi, so it is safe
  to leave on. Turn it on when your grid is large (roughly ≥ 64³) and the
  pressure solve dominates the step time.

Both choices solve the same discretized PPE to the configured tolerance.
Different reduction order and iteration paths can produce small roundoff-level
differences, which may eventually separate chaotic trajectories.

Both use one or more tight active boxes when the projected reduction is
worthwhile; otherwise they solve the full grid. Empty lattice planes may split
independent boxes. This is separate from the **Sparse Grid** toggle.

## Faster localized flows — Sparse Grid

**Sparse Grid** crops the solver each step to the active fluid region — a big
speed/memory win when the fluid occupies a small part of a large domain (a
splash in a big room). It disengages automatically when it cannot help (outflows
or cut-cell solids present, or Two-Phase filling the domain).

This remains dense storage inside the active window, not a tiled sparse grid.

## Reproducibility — Random Seed

**Random Seed** fixes particle placement, temporal jitter and force randomness.
With the same backend and software environment, it supports comparable reruns.
CPU and CUDA evolved results are close, not guaranteed bitwise-identical.

## A sensible starting point

| Goal | Transfer | CFL | Notable toggles |
|---|---|---|---|
| General water | FLIP | 8 | — |
| Swirls / vortices | APIC | 8 | — |
| Honey / lava | APIC | 6–8 | Viscosity 0.05–0.1 |
| Big splash | FLIP | 8 | Whitewater, Sheeting 0.5 |
| Bubbling / glug | FLIP | 6 | Two-Phase |
| Large final bake | (as above) | 8 | Multigrid, GPU |
