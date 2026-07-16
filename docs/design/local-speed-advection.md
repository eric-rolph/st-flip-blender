# Local speed bound for sub-stepped advection (PERF-M1)

Status: **design pre-registered; gates below were fixed before any study
ran.** Follow-up to the structural finding in
[advection-reflection.md](advection-reflection.md): after the band crop
and active-subset optimizations, the remaining linear-in-CFL cost term
is that every particle's sub-step COUNT derives from the domain-wide
maximum speed, so the calm bulk of a production scene pays the splash
region's count (with early exit, but the count itself is inflated).

## Problem

`_advect` computes `nsub_i = ceil(vmax_global * |dt_act_i| / (dx *
cfl_local))` from `_grid_velocity_bound` -- a strict global bound on the
trilinearly sampled speed anywhere in the domain. At CFL 16 a particle
sitting in a still pool takes ~16-32 RK3 sub-steps whose displacements
are nearly zero. The bound is what guarantees the documented local-CFL
displacement contract (<= cfl_local * dx per sub-step), which
`_segments_hit_mask`'s fixed traversal count and the half-band
extrapolation rely on -- so any replacement must preserve it STRICTLY,
not statistically.

## Scheme: adaptive per-sub-step h from a dilated cell bound

Behind `Params.advection_bound = "global" | "local"` (default
`"global"`; the default path is byte-for-byte the existing code).

Once per `_advect` call, build a cell-centred bound grid `b`:

1. Per component, per cell, the max |face value| of the two faces
   bounding that cell along the component's own axis.
2. Dilate EACH component field by `R = ceil(cfl_local) + 1` cells
   (Chebyshev ball, separable per-axis iterated 3-point max,
   edge-clamped -- NOT wrapping).
3. Combine: `b = sqrt(su^2 + sv^2 + sw^2)` of the dilated components.

The order matters for the proof: trilinear sampling is a convex
combination per component, so each sampled component at q is bounded by
that component's maximum over the cells its stencil touches, and the
sampled SPEED is bounded by the Euclidean norm of the three
per-component neighborhood maxima -- which is exactly what
dilate-then-combine computes.  The tighter combine-then-dilate variant
(max of per-cell norms) held in adversarial optimization (empirical
supremum ratio exactly 1.0, apparently via MAC face-sharing between
neighboring cells), but no written proof covers it, so the code
computes the provable quantity.

Then the sub-step loop becomes, per active particle:

    b_p  = b[cell(pos_p)]                     (nearest-cell gather)
    hmag = min(remaining_p, cfl_local * dx / max(b_p, tiny))
    he   = sign(dt_act_p) * hmag
    RK3 as before; remaining_p -= hmag

## Why the contract holds (the load-bearing lemma)

Claim: `b[cell(p)]` bounds |sampled velocity| at every point q the RK3
sub-step started at p can evaluate.

- The sub-step's displacement is `<= hmag * (max sampled speed along the
  stages)`. Stage 2 evaluates at `p + 0.5 he k1`, stage 3 at
  `p + 0.75 he k2`; if every stage speed is `<= b_p` then every stage
  point lies within `0.75 * cfl_local * dx` of p and the final point
  within `cfl_local * dx` -- i.e. inside the Chebyshev ball of
  `ceil(cfl_local)` cells around `cell(p)`.
- A sample at any point q touches faces belonging to cells within 1 cell
  of `cell(q)` (the `floor(pos/dx - offset)` staggering spill), hence
  within `ceil(cfl_local) + 1 = R` cells of `cell(p)` -- exactly the
  dilation radius. So each stage speed is `<= b_p`, closing the
  induction (stage 1 samples at p itself, distance 0, grounding it).
- Therefore displacement `<= hmag * b_p <= cfl_local * dx`: the same
  strict per-sub-step bound the global scheme provides. Backward
  advection (negative `dt_act`, the un-jitter path) only flips the sign
  of `he`; all magnitudes are unchanged.

Termination: `b <= vmax_global` everywhere, so
`hmag >= min(remaining, cfl_local dx / vmax_global)` and the iteration
count is bounded by the global scheme's `nsub`. The remaining-time
ledger is kept in FLOAT64: adversarial review demonstrated that a
float32 ledger's rounding drift grows as Theta(N^2 2^-24) and overruns
any O(1) headroom from roughly 1.2e4 sub-steps (reachable only with
extreme `cfl_target / cfl_local` ratios, but silently truncating
`dt_act` when hit). In float64 the same drift needs N^2 > 2^53. When
`remaining <= cap/b` the min takes `remaining` and the subtraction
yields exactly 0.0, so there is no infinite tail; the loop keeps the
global `max_n` (+2 headroom) as its bound and breaks when no particle
remains active. The float32 cast of `he` for the RK3 kernels can
exceed the cap by at most 0.5 ulp -- the same ulp-level slack the
global scheme's `h = dt_act / nsub` always had, absorbed by
`_segments_hit_mask`'s +3 traversal margin.

Both `_advect` paths (fast and `track_outflows`) use the same `he`
formula, so trajectories are identical whether or not outflow tracking
is engaged, as today. The outflow path keeps its full-array structure
(it never had the fast path's active-subset gathers), so local mode
buys trajectory consistency there, not speed; its early-out host sync
is amortized to every 8th iteration. The collision push-out and domain
clamp keep their existing full-array-every-iteration semantics, and a
call in which no sub-step runs (all `dt_act` exactly zero) still
applies the one relaxation+clamp pass the global branch would, never
returning the caller's array aliased.

## What changes and what does not

- Flag off: bit-identical (the branch wraps only new code; verified with
  the eight-config worktree harness before merge).
- Flag on: trajectories CHANGE -- calm-region particles take fewer,
  larger-in-time sub-steps. Spatial error per sub-step is still bounded
  by the same one-cell displacement; temporal integration error grows
  where h grows, i.e. exactly where the field is slow and smooth. This
  is arguably closer to the paper's own "locally sub-stepped advection
  at CFL_local = 1" than the global-bound conservative reading.
- Cost per sub-step adds one integer-indexed gather (vs 72 taps of RK3
  gathers) plus one elementwise min; the bound grid costs a handful of
  elementwise kernels once per `_advect` call.

## Pre-registered gates (fixed before any measurement)

- **G-H (hydrostatic)**: still pool with `advection_bound="local"`
  stays calm (existing still-pool tolerance); a uniform-translation
  field advects to the global mode's endpoint within float32
  accumulation tolerance (both bounds are equal there, but the
  sub-step partitions differ: equal splits vs cap-sized steps + tail).
- **G-L (lemma property test)**: random spiky fields (including a
  single-cell velocity spike), random query points q within
  `cfl_local * dx` of random p: `|sample(q)| <= b[cell(p)] * (1+1e-6)`.
- **G-A (accuracy, ship gate)**: 32^3 rotating tank, CFL {8, 16}, plain
  and reflection: local mode's floor-relative L_z retention `>=`
  global mode's `- 0.05`; KE retention `>=` global's `- 0.10`.
- **G-P (performance, ship gate)**: 128^3 production tank, CFL 16,
  plain, CUDA: local per-substep wall time `<= 0.75x` global.
  **Kill** if `> 0.9x` (not worth a non-bit-identical mode).
- **G-S (suite)**: full test suite + ruff; checkpoint resume roundtrip
  with the flag on reproduces the continuous run.

Ship = all gates pass -> flag stays default-off for one release
(product conservatism, same as `reflection`), recommended alongside
high Target CFL. Kill = G-P kill bar or a G-A miss; the branch then
lands docs-only with the measured tables.

## Adversarial review (pre-study)

A four-lens refutation review (contract math, float-point/termination,
integration, CUDA perf; 12 agents, findings adversarially verified) ran
BEFORE the gate studies. Four findings survived and were fixed before
any number below was measured: (1) the lemma/code mismatch resolved by
dilating per component before combining (above); (2) the float64
remaining-time ledger (above); (3) the all-zero-dt call now applies the
single relaxation+clamp pass the global branch would and never returns
the caller's array aliased; (4) the outflow path's early-out sync is
amortized to every 8th iteration and documented as consistency-only.
Two additional attacks came back clean: the full-array push-out cost
claim was refuted, and the fast path's sync/subset pattern matches the
already-profiled global subset path.

## Verdict: SHIP (measured 2026-07-15/16, RTX 5090 CUDA)

All pre-registered gates pass; `advection_bound="local"` stays
default-off for one release (product conservatism, same as
`reflection`), recommended alongside high Target CFL.

G-H, G-L, G-S: hydrostatic, lemma property test, uniform-translation,
zero-dt, outflow, resume tests all pass (tests/test_advection_bound.py);
full suite 812; flag-off verified bit-identical across the eight-config
worktree matrix.

G-A at 32^3 (48-frame rotating tank, floor L_z 0.4002,
validation/local_bound_32.json), floor-relative L_z deltas local vs
global: plain@8 +0.005, reflect@8 -0.001, plain@16 +0.001,
reflect@16 -0.017 -- all within the -0.05 gate; KE within +/-0.005
absolute everywhere.

G-P at 128^3 (5.2M particles, CFL 16, 48 frames,
validation/local_bound_128.json):

| case | bound | substeps | wall | s/substep | ratio | L_z ret. |
| --- | --- | --- | --- | --- | --- | --- |
| plain | global | 187 | 1530 s | 8.18 | 1.00 | 0.3551 |
| plain | local | 177 | 974 s | 5.50 | **0.67** | 0.3475 |
| reflect | global | 282 | 1651 s | 5.86 | 1.00 | 0.6316 |
| reflect | local | 285 | 1220 s | 4.28 | **0.73** | 0.6321 |

Both ratios clear the <= 0.75 ship bar (kill bar was > 0.9). Headline:
reflection + local at CFL 16 costs 1220 s where the plain CFL-1 floor
cost 2602 s -- less than half the wall time while retaining MORE
angular momentum than the floor (0.632 vs 0.589 absolute, 1.07
floor-relative). The remaining advection cost is now genuinely
per-particle; the old global-count linear term is gone.
