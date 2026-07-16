# SAMP-M5 sampling A/B study: KEEP EXPERIMENTAL

Status: **decision experiment complete.** Per the pre-registered adoption
rule in [the paper-limitations roadmap](paper-limitations-roadmap.md): the
PSD criterion failed while every safety gate passed, so
`temporal_sampling = "sobol_owen"` stays a documented, opt-in experimental
mode. SAMP-M6 (addon toggle / promotion) does NOT proceed; SAMP-M7
(spatial Sobol seeding) is deprioritized pending production-scale evidence.

## What was run

`tools/run_sampling_ab.py`: 128 cells (4 arms x 2 CFL targets x 2 samplers
x 8 seeds, Decision 6b) plus 16 dam-break runs, 24^3, on the roadmap's
restructured arm matrix. Artifact: `validation/sampling_ab_study.json`.

## Results

| gate | outcome |
| --- | --- |
| low-band PSD reduction >= 30 percent (gamma-active user arms) | **fail** -- drop-into-pool +1.6 percent +/- 7.4 (seed-dominated); translating slab exactly null (8x ratio 1.00) |
| spatial flatness no worse | pass (ratios 1.000-1.006) |
| spurious-KE floor no worse | pass (ratios 0.998-1.002) |
| no scrambling-specific spectral spikes | pass -- chaotic-splash peaks appear under BOTH samplers (pseudo max 3.04, sobol 4.79 on the same arm; identical 1.09/1.66 elsewhere) |
| dam-break KE non-regression | pass (max seed-mean deviation 1.4 percent) |

Bit-identical checkpoint resume and RNG-stream isolation were already
proven in SAMP-M3; nothing in this study weakens the safety story.

## Why the mechanism gate failed, honestly

1. **The mechanism is real but only isolable early.** On the controlled
   arm (gamma forced to 1) a single-seed early-time view shows an 86
   percent low-band reduction -- but that configuration SELF-EXCITES (full
   slab jitter continuously pumps kinetic energy into a calm pool, which
   is exactly why the paper attenuates gamma), so over 32 frames its PSD
   measures chaotic slosh: per-seed ratios span 0.05x-20x. The arm is an
   instrument for the mechanism, not a gateable target.
2. **The CFL axis never bound.** At 24^3 the calm scenes never reach
   target-binding speeds, so CFL 8 and CFL 16 produced identical
   trajectories (the same accessible-scale wall ERR-M1 hit). The paper's
   L3 regime -- large BINDING steps over calm water -- needs
   production-scale velocities somewhere in the domain.
3. **Chaos and advection swamp the signal on user arms.** The splash
   dominates the drop scene's height spectrum; particles advect through
   probe columns on the slab, decorrelating exactly the per-trajectory
   sequences the LDS improves.

## What would reopen promotion

A production-scale bake (CFL target actually binding over a calm surface,
resolution >= 128) showing a visible or measured low-band improvement.
The samplers, the metric, and this tool all exist; the experiment is one
command on a bigger machine budget. Until then: opt-in, experimental,
safe.

## Production-scale revival: RUN, and the gate failed -- KILL promotion

Measured 2026-07-16 (validation/samp_scale_128.json): the gated L3 arm
(drop into a quiescent pool) at 128^3, CFL 8, 48 frames, CUDA,
2 samplers x 8 seeds, the probe/PSD machinery above verbatim.
Sampler-neutral overrides for production scale only:
pressure_solver="multigrid", pcg_max_iter=1600 (the 24^3 Jacobi budget
stalls at 128^3).

The reopening condition was genuinely met this time: the CFL target
BOUND on 44 of 48 frames (mean 3.17 substeps/frame) -- the large
binding steps over calm water that the 24^3 study could never reach.
And the answer is now definitive:

| gate | outcome |
| --- | --- |
| low-band PSD reduction >= 30 percent | **fail** -- +10.6 percent +/- 16.6 (SE exceeds the mean; seed-dominated) |
| spatial flatness no worse | pass (ratio 1.02) |
| spurious-KE floor no worse | pass (ratio 0.97) |
| no sampler-specific spectral spikes | pass (3.03 pseudo vs 3.52 sobol, same regime) |

The mechanism the controlled arm isolated at early times simply does
not survive contact with a chaotic splash even at production scale and
binding CFL. The promotion question is CLOSED as a kill:
`temporal_sampling = "sobol_owen"` remains a safe, documented,
experimental opt-in; SAMP-M6 (default promotion / addon toggle) will
not proceed, and no further budget should be spent on it.
