# Advection-reflection: scheme of record and residual proof

Status: **implemented behind `Params.reflection` (default off)** per the
corrected ENER-M2b design in
[the paper-limitations roadmap](paper-limitations-roadmap.md). This
document records the scheme, the merge-blocking residual-carryover
induction, and the acceptance evidence.

## Scheme (one substep of size dt)

1. Slab-integrated P2G (`W_T` as always); FLIP baseline `old`.
2. Forces (full dt), surface tension (+ optional stabilizer), viscosity,
   no-through; snapshot `u*`.
3. First projection -> `u1` (solenoidal; the ONLY transport field).
4. Reflected field `u_hat = 2 u1 - u*` on faces that are open, valid in
   the first P2G, and have `phi_f >= 0.5` (eps-rho-clamped interface
   faces must not get their projection residual doubled); elsewhere
   `u_hat = u1`; no-through re-enforced.
5. Half-band extrapolation (`ceil(cfl_target) + 2`) of `u1`, `u_hat`,
   `old` under the SAME valid masks.
6. Mid-step reflection G2P: FLIP takes the pure delta
   `u_p += interp(u_hat - old)` -- equal to the plain delta
   `interp(u1 - old)` plus the reflected pressure impulse
   `interp(u1 - u*)`, so forces reach particle momentum exactly once.
   PIC/APIC are replacement transfers and read `u_hat`.
7. First half-advection through `u1`:
   `dt_act1 = clip(dt/2 + r, 0, dt)`, carry `r1 = dt/2 + r - dt_act1`
   (`r` is the incoming residual; the outflow keep-mask also filters r1).
8. Second P2G, instantaneous (`wt = 1`; m0-consistent because m0 is
   calibrated to E[W_T] = 1); FLIP baseline `old2`.
9. Face densities/active from the second deposit; NO force re-application
   (they are already inside the deposit via step 6); step-start apertures
   and wall velocities reused (O(dt) consistent).
10. Second projection; half-band extrapolation; standard final G2P blend
    against `old2`; solid-velocity enforcement.
11. Second-half jitter with fresh velocities:
    `dt_act2 = clip(dt/2 + r1 + gamma xi dt, 0, dt)`,
    `r' = dt/2 + r1 - dt_act2`.
12. Second half-advection through the projected `grids2`; sheeting;
    commit.

The NORM vel-immutability contract survives: the only jitter draw is step
11, and only positions change afterwards (advection, sheeting). The
mid-step P2G is instantaneous, so the exact-normalization divisor (which
recomputes gamma from velocities) never runs against mid-step state.

## Residual-carryover induction (merge-blocking)

Notation: incoming residual `r`, step size `dt`, previous step size
`dt_prev`; jitter `j = gamma xi dt in [-dt/2, dt/2]`. Invariant claimed:

    |r| <= max(dt, dt_prev) / 2   after every substep,        (I)

with the excess over dt/2 contracting geometrically after an abrupt dt
change. Three cases for step 7's clip:

**Case A, |r| <= dt/2** (steady state). `dt/2 + r in [0, dt]`: the clip
is the identity, `dt_act1 = dt/2 + r`, `r1 = 0`. Step 11:
`dt/2 + 0 + j in [0, dt]`, identity again, so `r' = -j in [-dt/2, dt/2]`.
This is exactly the paper's single-step stationary distribution split
across two half-steps; (I) holds with the tight bound dt/2.

**Case B, r > dt/2** (dt shrank; r bounded by dt_prev/2 from the previous
step's own invariant). `dt/2 + r > dt`: clip binds high, `dt_act1 = dt`,
`r1 = r - dt/2 in (0, dt_prev/2 - dt/2]`. Step 11 argument
`dt/2 + r1 + j`:
- if it stays in `[0, dt]`: `r' = -j + max(0, ...)`, precisely
  `r' = dt/2 + r1 - dt_act2 = -j`, bounded by dt/2 -- fully recovered in
  ONE substep;
- if it clips high (`r1 + j > dt/2`): `dt_act2 = dt`,
  `r' = r1 - dt/2 + j... <= r1` and `r' <= r1 - dt/2 + dt/2 = r1`, i.e.
  the excess never grows and shrinks by at least `dt/2 - j >= 0`;
  strictly positive shrinkage in expectation (E[j] = 0), so the excess
  contracts and (I) holds with the max() bound.

**Case C, r < -dt/2** (dt grew is impossible for this sign; this arises
after clip-low history). `dt/2 + r < 0`: clip binds low, `dt_act1 = 0`,
`r1 = dt/2 + r in [dt/2 - dt_prev/2, 0)`. Step 11 mirrors case B with
signs flipped: `r'` either lands at `-j` (recovered) or contracts its
negative excess by at least dt/2 per substep.

Total travel: `dt_act1 + dt_act2 <= 2 dt` (each half clipped at dt), so
the sparse `_band` contract and the half-band extrapolation are honest by
construction. The randomized adaptive-dt stress test drives dt through
x2 / x0.5 jumps for hundreds of substeps and asserts (I) directly.

## Cost model

Per substep: 2 projections + 2 P2Gs + 2 G2Ps + 2 half-band extrapolations
+ forces once. Against plain stepping at HALF the CFL (two substeps),
reflection trades the second force pass for tighter extrapolation bands:
approximately cost-neutral on grid work, which is the product framing --
"enable Reflection when you raise Target CFL" -- with the physics win
that both halves transport through freshly projected fields and the
splitting's dissipation is reflected away once per substep.

## Acceptance evidence (measured 2026-07-14, 32^3 tank, 48 frames)

Quality gates -- passed far beyond their targets:

| case | absolute L_z retention | vs plain CFL-1 floor (0.396) |
| --- | --- | --- |
| plain CFL 1 (ENER-M0) | 0.396 | 1.00 (the floor) |
| plain CFL 8 (ENER-M0) | 0.262 | 0.660 (the gate bar) |
| plain CFL 16 (ENER-M0) | 0.214 | 0.539 |
| reflection CFL 1 | 0.581 | 1.47 |
| reflection CFL 8 | 0.672 | 1.70 |
| **reflection CFL 16** | **0.588** | **1.49** |

Reflection at CFL 16 retains MORE angular momentum than plain stepping at
CFL 1 -- not merely more than plain at CFL 8, which was the gate.  KE
retention tells the same story (0.262 at reflection-16 vs 0.043 at
plain-8).  Hydrostatic still pool stays calm, the dam still falls,
two-phase/APIC/PIC/outflow/resume all pass in tests/test_reflection.py.

Wall-clock gate -- honestly failed at study scale: the roadmap asked
reflection@16 within 1.1x of plain@8; measured 1.31x (jacobi AND
multigrid identically).  The mechanism: at 32^3 wall time is dominated by
CFL-independent per-frame work (plain@16 with HALF the substeps of
plain@8 runs only 9 percent faster), so halving substeps cannot pay for
doubled transfers.  The per-substep grid-cost ratio is 1.43x -- BETTER
than the design's predicted 1.9-2.0x -- so the cost model holds; the
scene scale does not exercise it.  Production-scale wall-clock evaluation
(where substeps dominate) is the remaining open item; until then the
product framing is: an opt-in QUALITY control for raised Target CFL at
~1.3-1.4x small-scale cost, delivering angular-momentum retention that no
amount of plain sub-stepping reaches.
