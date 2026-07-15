# TIME-M2/M3 two-time-level reconstruction: KEEP-EXPERIMENTAL

Status: **implemented behind `Params.temporal_levels` (default 1,
bit-identical); scene-level A/B run; decision KEEP-EXPERIMENTAL
(Params-only, no promotion, no addon UI).** This resolves the K = 2
linear case of paper Outlook O2 for this repo; K > 2 remains a declared
non-goal. The paper's own preliminary negative (thin-4D-grid
linear-in-time reconstruction: "no significant improvement of visual
simulation quality", p.6) is hereby REPLICATED with quantified numbers.

## What shipped (TIME-M2)

`temporal_levels = 2` retains first temporal moments during P2G (three
extra scatters per velocity axis per tap, the same weights as the
momentum normalization including two-phase mass weights) and
reconstructs face velocities linearly in tau at the slab end:
`q(1/2) = qbar + b (1/2 - tbar)`, `b = cov / (var + lam / 12)`, with
`temporal_fit_reg = 0.1` from the TIME-M1 GO. `lam -> inf` degrades
EXACTLY to today's one-sided mean (tested); phase and validity keep
their zeroth-moment recipes, so the pressure system is identical by
construction (tested: byte-equal phi/validity grids and equal PCG
iteration counts). Linear-in-tau signals are recovered exactly through
the real P2G at reg = 0 (tested); checkpoint/resume is bit-identical
with the flag on (tested; no new checkpoint keys).

## The A/B numbers (TIME-M3, validation/temporal_levels_ab.json, 20^3)

| gate | target | measured |
| --- | --- | --- |
| calm-pool noise reduction (8 seeds) | >= 20 percent | **-0.06 percent (exact null)** |
| dam-break KE within seed spread (3 seeds) | <= 3x spread | **38x spread (11.4 percent systematic)** |
| smooth-swirl no compounding (200 frames) | late-KE ratio <= 1.10 | **2.09** |
| wall-clock overhead (multigrid) | <= 45 percent | **~0 percent** |

## Reading the numbers honestly

1. **The noise claim is dead at this scale.** The fit shifts the
   effective evaluation time (removes the one-sided kernel's ~0.27 dt
   phase lag); it does not touch per-frame Monte Carlo noise -- the same
   structural fact the SAMP-M5 and CALM-M4 studies hit from other
   directions.
2. **The dynamics shift is systematic, not noise.** 11 percent KE
   deviation at 38x seed spread on the dam means level 2 simulates a
   measurably different flow with no demonstrated quality gain to
   justify it.
3. **The swirl result is the reviewers' predicted slope feedback, with
   an unexpected sign.** Level 2 retains 2.1x the late-time kinetic
   energy of level 1 (13.3 vs 7.4 percent of initial) -- LESS numerical
   dissipation, the direction users usually want. But this retention is
   an estimator side effect (the slope term recycling through the 0.98
   FLIP blend), not a derived conservation property like
   advection-reflection's, and it arrives coupled to the uncontrolled
   dam shift. It is a lead, not a feature.
4. Scatter cost is invisible at accessible scale (per-frame overheads
   dominate -- the recurring finding).

## Decision and revival conditions

KEEP-EXPERIMENTAL: the flag stays for research (it is clean, gated, and
structurally safe), no promotion. The prescribed follow-up (a) has now
RUN at production scale (128^3, 5.2M particles, CUDA;
validation/production_tank_128.json): level-2 at CFL 16 retains
floor-relative L_z 0.729 -- essentially plain CFL-8 grade (0.734) at
plain CFL-16 substep counts, versus plain CFL-16's 0.602.  The
energy-retention lead is therefore REAL and replicates, but reflection
dominates it outright (1.078 at the same CFL for ~9 percent less wall
time), so the decision stands: keep experimental, prefer
`Params.reflection`.  The combination cell (reflection +
temporal_levels, same tank and scale) has now RUN and the mechanisms
INTERFERE: floor-relative L_z 0.935 -- worse than reflection alone
(1.078) -- at 487 substeps versus reflection's 285 (the fit's slope
extrapolation amplifies velocity extremes, shrinking dt) and ~10
percent more wall time.  The plausible mechanism: the reflected mirror
u_hat = 2 u1 - u* DOUBLES whatever estimator noise the fit adds to the
transport field.  Do not combine; the question is closed.  (b) a
production-scale visual A/B is still required before any
temporal-levels quality claim, per the paper's own negative --
reflection's own visual A/B exists (Videos/stflip_reflection_ab.mp4).
