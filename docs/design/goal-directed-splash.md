# Goal-directed splash: derivative-free control over body forces (GOAL-M1)

Status: **design pre-registered; gates below were fixed before any study
ran.** Motivated by N. Thuerey's differentiable-solver program (see
[differentiable-stflip-research-plan.md](differentiable-stflip-research-plan.md)):
its inverse-problem recipe -- search cheap, refine expensive -- applied
WITHOUT solver gradients, so it ships against today's NumPy/CuPy solver.

## Problem

Artists direct fluids by goals ("the wave hits the character at frame
60", "the splash clears the wall but the porch stays dry"), not by
hand-tuning body-force vectors. The solver already has an
art-directable force system (`add_force`: DIRECTIONAL / VORTEX /
TURBULENCE / CONFINEMENT) -- a small, physical, continuous parameter
space. What is missing is the inverse map from a goal to force
parameters.

## Approach

1. **Time-windowed forces** (solver change, the only one): `add_force`
   gains `t_start=0.0, t_end=inf`; `_apply_forces` skips a force whose
   window excludes the current solver time. Defaults reproduce the old
   behavior bit-identically. Windows are what make "pulse" control
   possible (redirect the splash mid-flight, then let physics run).
2. **`stflip/control.py`** (new, bpy-free, dependency-free):
   - `ForceGene`: declares one optimizable force -- its type, and
     bounds for each free parameter (strength range, center box, unit
     direction, radius range, optional time window). Encodes to / decodes
     from a flat vector in [0, 1]^d.
   - `CrossEntropyOptimizer`: a ~60-line Cross-Entropy Method --
     Gaussian sampling, elite refit, std floor + annealing, fully
     deterministic under a seed. No new dependencies (CMA-ES via the
     `cma` package is a drop-in upgrade later if ever warranted).
   - Objectives: `mass_in_box` (fraction of particles inside an AABB at
     given frames), `keep_dry` (penalty for particles in a protected
     AABB), combined with weights. Objectives read particle positions
     only -- no reconstruction in the loop.
   - `optimize_forces(...)`: rollout runner -- builds the scene, runs N
     frames at the PROXY resolution, scores, iterates CEM; returns best
     parameters + full history for the artifact.
3. **Two-stage transfer**: optimize at proxy resolution (48^3 class,
   seconds per rollout on the 5090), optionally refine the best
   candidate set at an intermediate resolution, then apply the winning
   forces to the hero-resolution bake. The transfer is the scientific
   risk and gets its own gate (below), measured honestly.
4. **Demo tool** `tools/run_goal_splash.py`: dam break steered into an
   elevated catch basin the uncontrolled flow cannot reach, with a
   keep-dry zone; writes `validation/goal_splash_demo.json`.

Blender UI (goal empties + an Optimize operator) is explicitly OUT of
scope for GOAL-M1 -- the addon UI is being modified in a parallel
session; the feature lands headless-first.

## Pre-registered gates (fixed before any measurement)

- **G-CTRL-1 (mechanism)**: 32^3 dam break, one DIRECTIONAL force
  (strength + direction free): optimized mass-in-target-box at the
  gate frame >= 3x the no-force baseline, reproduced across 3 optimizer
  seeds.
- **G-CTRL-2 (unreachable goal)**: a target region the uncontrolled
  flow misses entirely (baseline objective < 0.005): optimized
  objective >= 0.05 (10x the miss threshold), with the keep-dry
  penalty active and satisfied.
- **G-CTRL-3 (transfer, ship gate)**: parameters optimized at 48^3
  retain >= 60 percent of the proxy objective IMPROVEMENT (over that
  resolution's own baseline) when re-simulated at 128^3. Below 60
  percent, the intermediate-resolution refine stage becomes mandatory
  and the gate re-runs; if it still fails, the feature ships marked
  experimental with the measured transfer table.
- **G-CTRL-4 (budget)**: the demo optimization completes in <= 30 min
  wall on the RTX 5090 (CUDA proxy rollouts).
- **G-CTRL-5 (hygiene)**: full suite + ruff; a solver with no
  registered forces, and a solver whose forces use default windows, are
  bit-identical to pre-change behavior (worktree harness); CEM unit
  tests converge on analytic objectives.

GPU gate studies run only after the hero-bake / whirlpool renders
release the GPU (honest timings and no contention).

## Verdict: SHIP AS EXPERIMENTAL (four iterations, 2026-07-17)

Artifacts: validation/goal_splash_demo.json (full battery, clean re-run
under the final optimizer), validation/goal_splash_iter3.json (32^3
demo probe), validation/goal_splash_iter4.json (two-stage demo).

- **G-CTRL-1 (mechanism): PASS, decisively.** Every optimizer seed in
  every iteration beat the no-force baseline by far more than 3x --
  baselines on this scene are ~0.000, optimized target mass 0.26-0.30.
- **G-CTRL-2 (unreachable goal): PASS.** The elevated catch basin is
  untouched by the uncontrolled flow (< 0.0005 mass); the optimizer
  reaches 0.26-0.31 with the keep-dry penalty satisfied. In every
  iteration the CEM DISCOVERED a pulsed force on its own (windows
  ~0.24-0.39 s to ~1.5-1.8 s), exercising the time-window feature
  without being told to.
- **G-CTRL-3 (transfer): PASS.** 48^3 -> 128^3 retention measured
  independently three times: 0.813, 0.729, 0.815. A raw 32^3 proxy
  transfers at only 0.599 -- exactly the borderline case this doc
  pre-committed to -- and the mandatory warm-started 48^3 refine stage
  (3 generations seeded at the coarse winner, init_std 0.08) restores
  retention to 0.815. Two-stage search is the shipped recommendation.
- **G-CTRL-4 (budget): FAIL at the pre-registered 30-minute bar.**
  Measured envelope on the RTX 5090: 48^3 demo 108 min; 32^3 demo
  57 min; two-stage (32^3 + refine) 64 min. The root cause is physics,
  not harness overhead: a forced rollout costs ~2x the baseline
  because the injected momentum raises vmax and therefore the
  substep count, and the CEM genuinely uses ~100+ rollouts (best
  scores kept improving through generation 9-10; patience-3 early
  stopping rarely triggered). The 30-minute bar was optimistic by
  roughly 2x for this scene class. Budget is a quality dial the
  artist controls: fewer generations give a rougher answer sooner.

Per the pre-registered fallback pattern, the feature ships HEADLESS
and EXPERIMENTAL: the mechanism, constraint handling, and two-stage
transfer are proven; the honest cost of a converged optimization on
5090-class hardware is about an hour, run as a background job. The
CuPy pool release per rollout and CEM warm-starting/early-stopping
landed as library improvements regardless.
