# ERR-M1 step-control diagnostics study: K1 COLLAPSE

Status: **kill-point study complete, verdict K1 collapse (documented
negative).** ERR-M2 (the multi-signal quality-guard controller) and ERR-M3
as designed in [the paper-limitations roadmap](paper-limitations-roadmap.md)
are CANCELLED. The surviving scope is a possible future one-line
vmax-predictor flag (below). The diagnostics, controller library, and study
tool stay -- independently valuable telemetry, and the controller law is
unit-tested and ready should production-scale evidence reopen the question.

## What was tested

`stflip/step_control.py` (review-corrected controller law: mutually
exclusive decay/release, non-resetting quiet counter, fresh-r=1 /
restore-r=floor asymmetry, dt-decrease signal masking, geometric-smoothed
capped vmax predictor, combined-floor dt candidate) plus gated solver
diagnostics (clamp-bind fraction post-hoc from dt_act, valid-normalised
under-sampled-face fractions, near-solid fast fraction, capillary-clamp
regime flag; bit-identical off, tested). `tools/run_step_control_study.py`
ran three failure episodes at CFL 16.

## Findings

1. **The Fig. 7 gap exists but stays benign at study scale.** Excluding the
   unbounded from-rest substep, actual/estimated CFL has p50 ~ 1.1 and p90
   ~ 2.0 on gravity scenes -- but every ratio above 1.5 occurred in the
   first two substeps (free-fall spin-up, which the reviewers correctly
   flagged as a no-alarm segment). Across 24 frames of a 24^3 dam break no
   genuine impact overshoot (> 1.5x at substep >= 2) occurred, so there was
   no failure event for any signal to lead.
2. **Tunneling is already guarded.** A CFL-16 jet against a 1.5-cell wall
   tunneled ZERO particles in every configuration tried -- the solver's
   local-CFL-1 sub-stepped advection does its job. The near-solid signal
   alarmed on 100 percent of approach substeps (correct hazard
   identification), but a guard with nothing to prevent earns no keep.
3. **False alarms:** the fraction signals were clean on the calm pool
   (0 percent); raw per-substep vmax growth alarmed 20 percent of calm
   substeps (max-statistic noise -- the geometric smoothing in
   `predicted_vmax` exists for exactly this).

Verdict per the roadmap kill criteria: **K1** -- nothing beyond the vmax
growth predictor adds a substep of lead time or precision, so the
initiative collapses to a vmax_pred-only candidate; the multi-signal
controller is not built.

## Scale caveat (recorded, not hidden)

The paper documents the estimated-vs-actual underestimate "during splashy
intervals" at production resolution. This study could not reproduce a
qualifying overshoot at CI scale (24^3), so the collapse verdict is "at
accessible scale". If a production-scale bake ever shows sustained
actual-CFL overshoots (the FrameStats diagnostics now record exactly this),
the surviving candidate is the cheapest possible fix: use
`step_control.predicted_vmax` in step_frame's dt candidate -- a one-line
class change directly addressing the documented gap -- validated by the
do-no-harm suite the ERR design specified (free-fall, gentle-flow, and
still-pool substep-count bounds). Nothing else from ERR-M2/M3 should be
revived without new evidence.

## Artifacts

- `validation/step_control_study.json` -- traces and evaluation.
- `tests/test_step_control.py` -- controller law (steady-state convergence
  both sides of A_RELEASE, non-resetting quiet counter, init asymmetry,
  masking), predictor bounds, solver capture bit-identity.
