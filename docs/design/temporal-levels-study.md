# TIME-M1 estimator study: GO (with scope caveats)

Status: **kill-point study complete, verdict GO.** TIME-M2 (solver
integration behind a default-off flag) is unlocked per
[the paper-limitations roadmap](paper-limitations-roadmap.md); TIME-M3
still arbitrates the dynamics question before anything is promoted.

## What was tested

Whether a W_T-weighted, ridge-regularised linear-in-tau fit -- algebraically
equivalent to retaining two time levels during P2G -- reconstructs the
slab-end value q(+1/2) better than today's one-sided W_T mean (whose
effective evaluation time is tau ~ +0.2266). Sample sets came from the
ACTUAL Eq. 10-11 jitter recursion (clamp included; an abrupt 1.5x
adaptive-dt arm included), weighted by the temporal kernel times separable
spatial-kernel weights at uniform in-cell offsets. Tool:
`tools/run_temporal_study.py`; artifact: `validation/temporal_study.json`
(288 cells, seed 0).

## Result

A single regulariser passes every roadmap gate -- four do:

| lam (units of 1/12) | s=2 gain (<= 0.95x) | s>=4 gain (<= 0.80x) | no-regression (<= 1.05x, all cells) |
| --- | --- | --- | --- |
| 0.01 - 0.3 | pass | pass | pass |
| 1.0 - 3.0 | pass | fail | pass |
| 10.0 | fail | fail | pass |

**Recommended lam = 0.1** (centre of the passing range, margin on both
sides). The no-regression bound held at every simulated cell including
pure-noise (s = 0) cells at full jitter and the abrupt-dt arm.

## Honesty notes

1. **Harder-than-required sampling.** Measured effective sample count at
   the gate was 10-15 (spatial weighting of 64 raw samples), below the
   30-60 the roadmap assumed. The gates passed under HARSHER noise than
   required, which strengthens the GO.
2. **What GO does not claim.** The one-sided mean carries a systematic
   phase-lag bias of 0.2734 * s * dt against a slab-end target, so at
   s >= 2 the fit wins largely by removing that bias. Whether slab-end
   evaluation actually improves simulation DYNAMICS (the paper's own
   thin-4D-grid and time-interpolated-G2P experiments found no visual
   gain) is deliberately NOT answered here -- that is TIME-M3's
   scene-level A/B, and the paper's negative result keeps expectations
   modest.
3. The curvature arm (quadratic-in-tau signal at magnitude s) was included
   in every gate; ridge at lam <= 0.3 tolerated the model mismatch.
