# CALM-M4 deformation-aware jitter gate: MISS (documented negative)

Status: **acceptance study failed all three gates; the mode ships
default-off as an experimental research knob per the roadmap's
pre-registered miss protocol.** `Params.gamma_mode = "deformation"`
exists, is unit-tested, and is contraindicated for general use. The
default "speed" gate is bit-exact with earlier releases.

## What the mode does

Replaces the paper's speed-only jitter attenuation with
`a_p = max(phase-flux normal displacement, |n.D.n| strain rate,
grid-frame unsteadiness)` plus the surface-mode interiorness term. Two
review-driven design choices survived contact with measurement:

- The SYMMETRIC strain tensor is load-bearing: rigid rotation reads
  D = 0 (tested), where the full velocity-gradient norm would read
  sqrt(2) omega and pin the gate open on the stirred-pool surface.
- Normal displacement measured as PHASE FLUX |v . grad phi_s| dt rather
  than v . n_hat: the normalized direction is noise where gradients are
  small (measured jet-front gamma 0.59 from direction noise; 0.98 with
  the flux form on a resolved front).

## What the unit tests verified (mechanism level)

- River skin damps while the speed gate is pinned open (tangential flow
  at local CFL >= 1): deformation skin gamma < 0.7x speed's.
- Solid-body rotation does not collapse the gate (interior gamma > 0.95).
- Resolved advancing fronts (>= 6 cells) keep full jitter (0.979).
- KNOWN LIMITATION, pinned by test: features thinner than ~4 cells
  under-gate (2-cell slug front gamma 0.74) because the smoothed phase
  and all its derived inputs wash out -- the same thin-feature contract
  CALM-M2 documents.

## Why the acceptance study failed (8 seeds, 24^3,
validation/calm_m4_study.json)

| gate | target | measured |
| --- | --- | --- |
| river surface-roughness reduction vs speed gate | >= 30 percent | **-5 percent** |
| still-pool temporal-std no-regression | <= 1.10x | **3.90x** |
| dam-break KE trace | within 2 percent | **5.2 percent** |

1. **Damping jitter does not smooth a moving surface at accessible
   scale.**  The river's per-frame roughness is dominated by spatial
   Monte-Carlo sampling and edge churn, not temporal jitter -- the same
   finding as the SAMP-M5 study (per-frame scatter is invariant to the
   temporal deviate source).  The mechanism (gamma drops on the skin)
   works; the payoff channel does not exist at this scale.
2. **Restoring bulk jitter leaks into still-pool surfaces.**  The
   interiorness band on the smoothed phase reaches closer to a quiescent
   surface than CALM-M3's raw-phase band, and the extra near-surface
   jitter TRIPLES the still pool's temporal noise -- a real regression on
   exactly the scene class the CALM initiative exists to protect.
3. The changed gamma field shifts dam dynamics past the 2 percent KE
   envelope.

## Revival conditions

Production-scale evidence that temporal jitter visibly drives moving-
surface roughness (none at 24^3), PLUS a still-pool-safe interiorness
formulation (e.g. CALM-M3's raw-phase band composed with the deformation
activity only where local CFL >= 1). Until both exist, prefer
`gamma_mode = "surface"` (CALM-M3) for stratification hygiene and the
speed default for everything else.
