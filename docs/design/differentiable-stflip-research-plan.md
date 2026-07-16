# Differentiable ST-FLIP: phased research plan (R0-R4)

Status: **plan only -- no phase is funded until its predecessor's gate
passes.** Motivated by N. Thuerey's differentiable-PDE-solver program
(UW seminar, 2026): hybrid solver-plus-network correction beats pure
surrogates by ~10x accuracy at equal network size; unrolled training
through solver Jacobians is the enabling machinery; statistics-matched
losses replace state matching beyond the chaotic horizon.

## The thesis worth testing

Unrolled differentiable training pays per CHAINED JACOBIAN, and its
failure mode is the exponential blow-up of products of step Jacobians
-- the talk's channel-flow closure needed dozens of no-gradient warm-up
steps to survive it. ST-FLIP at CFL 16 takes ~16x fewer steps per unit
of physical time than the CFL-1 solvers typically differentiated
through. **Headline claim to test: at equal physical horizon,
gradient chains through ST-FLIP are shorter and better-conditioned
than through CFL-1 FLIP, making inverse problems and hybrid training
materially cheaper.** If true, the property that makes our bakes fast
also makes us an unusually good differentiation substrate -- a genuine
research contribution, not a port.

Two structural obstacles are known upfront and are R0's subject:
CuPy has no autodiff (a differentiable core means a Warp or JAX
rewrite), and several solver stages are non-smooth.

## Phases, gates, kill points

### R0 -- differentiability inventory (cheap, CPU, ~days)

Catalogue every stage of `_step_core` by differentiability class:

- smooth (P2G weights, G2P, RK3 advection through a fixed field,
  pressure application, forces);
- fixed-point iterations (pressure CG/multigrid -> implicit-function-
  theorem adjoint at the converged solution, standard practice;
  capillary Helmholtz likewise);
- non-smooth but reparameterizable (temporal jitter: the xi draws are
  data, not parameters -- fix the noise per rollout and the jittered
  step is smooth in everything else);
- genuinely discrete (nsub counts, active-subset membership, sparse
  window bounds, particle add/remove at outflows, push-out iteration
  count): decide per item between freeze-at-forward-value
  (straight-through), smooth relaxation, or exclusion from the
  differentiated horizon.

Deliverable: a table in this doc. Gate: no stage is UNRESOLVABLE for
short-horizon (<= 2 physical seconds) unrolls. Kill: a load-bearing
stage (projection, P2G) proves non-adjointable in practice.

### R1 -- minimal differentiable core (Warp, small 3D)

Framework choice: **NVIDIA Warp** over JAX -- first-class Windows +
CUDA support (this project's environment), kernel-level autodiff,
Blender-adjacent ecosystem. Port ONLY: slab-integrated P2G with W_T,
Jacobi-CG projection (IFT adjoint), G2P, RK3 advection, gravity.
No reflection, no two-phase, no surface tension, no sparse window.

Gates: (a) forward parity with stflip on a 32^3 dam break at the
STATISTICS level (energy trace within 2 percent, same qualitative
surface) -- bitwise parity is a non-goal across frameworks; (b)
gradcheck (finite-difference agreement) per op and through a 4-step
unroll. Kill: per-step wall time > 5x stflip's at 64^3 (the adjoint
must not price itself out).

### R2 -- first inverse problem + the headline measurement

Task: initial-velocity control of a 2D/small-3D basin splash (the
classic adjoint fluid-control benchmark) via gradient descent through
the unrolled solver.

Measurements, pre-registered: (a) success of the control task at
CFL {1, 4, 16} with matched physical horizon; (b) gradient
signal-to-noise ratio (cosine similarity of gradients across noise
reparameterizations and across neighboring horizons) as a function of
CFL -- the headline claim predicts SNR(CFL 16) > SNR(CFL 1) at equal
physical time; (c) wall-clock per optimization step.

Gate: control task solved at CFL 16 in fewer optimizer steps x wall
time than CFL 1. Kill: gradient SNR at CFL 16 is WORSE (large steps
could in principle roughen the loss landscape -- if the data says so,
the thesis is dead and we publish the negative honestly).

### R3 -- hybrid corrector (the talk's central recipe)

Coarse ST-FLIP (32^3-48^3) + NN corrector (additive, post-projection)
trained with unrolling against 4x-reference rollouts; no-gradient
warm-up steps for distribution shift, exactly per the talk. Losses on
STATISTICS (energy spectra, surface-height PSD -- our existing metrics
library) not states.

Gate: corrected coarse rollouts match reference statistics within the
bands our SAMP/CALM studies used, at >= 4x wall-clock advantage over
the reference. Kill: corrector that only matches when trained per-scene
(no cross-scene generalization on a held-out scene family) -- a
per-scene corrector is useless for an addon.

### R4 -- product decision

Only reached if R2 AND R3 gate. Decide: ship a Warp-based "detail
boost" bake option vs keep as research. Explicit criteria: artist-visible
quality delta on the demo reel scenes, bake-time budget, model
distribution size, and the maintenance cost of a second solver core.

## Non-goals (fixed now)

- Pure surrogates (generality loss is fatal for an addon).
- PINNs (per the talk; per us).
- Foundation models / pretraining programs.
- Differentiating through rendering or reconstruction (control and
  correction losses live on particles and grids only).

## Relationship to GOAL-M1

[goal-directed-splash.md](goal-directed-splash.md) ships the
inverse-problem PRODUCT today with derivative-free search over the
existing force system; R2 would upgrade its inner loop from CEM to
gradients if and only if the headline claim survives measurement. The
two share objective definitions and scene builders by design.
